//! Hyperparameter (psi) score and curvature terms: the score-block
//! accumulation, the time-wiggle psi action, and the first-/second-order
//! psi terms together with their directional-derivative accumulators.

use super::*;

/// The order-≤3 row tower of whichever time-constant frame the family runs:
/// the two frames share the primary count but not the tower's static sparsity.
enum TimeConstantThirdTower {
    Gaussian(<StaticSlopeGeometry as SlopeRowGeometry<STATIC_SLOPE_PRIMARIES>>::Tower3),
    Anchored(<AnchoredStaticSlopeGeometry as SlopeRowGeometry<STATIC_SLOPE_PRIMARIES>>::Tower3),
}

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
        // One rank-1 update per follow-up channel — exactly one when the slope
        // is time-constant, three when it varies. Reading a single `primary[3]`
        // against `coefficient_design()` assembled the score of a DIFFERENT
        // model on a varying slope (#2765).
        for &(slope_primary, design) in self.slope_layout.primary_channels().as_slice() {
            design.axpy_row_into(row, primary[slope_primary], &mut score_g.view_mut())?;
        }
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

    /// Score pullback of the absorbed-influence block (#461). Unlike the identity
    /// flex blocks, the absorber's `p₁` coefficients enter through the single
    /// `o_infl` primary scalar with row design `Z̃_infl[row,:]`, so the block score
    /// is `primary[infl] · Z̃[row,:]` (the same projection `add_pullback` applies
    /// to the Hessian and `accumulate_dynamic_q_blockwise_row` to the gradient).
    pub(crate) fn accumulate_score_influence_block(
        &self,
        primary_layout: Option<&FlexPrimarySlices>,
        row: usize,
        primary: &Array1<f64>,
        score_i: &mut Array1<f64>,
    ) -> Result<(), String> {
        let Some(infl_idx) = primary_layout.and_then(|layout| layout.infl) else {
            return Ok(());
        };
        let z_tilde = self.influence_absorber.as_ref().ok_or_else(|| {
            "accumulate_score_influence_block: influence primary index present but no Z̃ design"
                .to_string()
        })?;
        let weight = primary[infl_idx];
        if weight != 0.0 {
            score_i.scaled_add(weight, &z_tilde.row(row));
        }
        Ok(())
    }

    /// Score pullback using actual Jacobians from q-geometry (timewiggle-correct).
    pub(crate) fn accumulate_score_with_q_geometry<Storage>(
        &self,
        row: usize,
        qg: &SurvivalMarginalSlopeDynamicRow,
        primary: &ndarray::ArrayBase<Storage, ndarray::Ix1>,
        score_t: &mut Array1<f64>,
        score_m: &mut Array1<f64>,
        score_g: &mut Array1<f64>,
    ) -> Result<(), String>
    where
        Storage: ndarray::Data<Elem = f64>,
    {
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
        // One rank-1 update per follow-up channel; see
        // `accumulate_score_blockwise` (#2765).
        for &(slope_primary, design) in self.slope_layout.primary_channels().as_slice() {
            design.axpy_row_into(row, primary[slope_primary], &mut score_g.view_mut())?;
        }
        Ok(())
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
        // A time wiggle moves every design ψ through the ζ composition, which differentiates the
        // time-wiggle map itself instead of lifting its Jacobian by hand (gam#2893).
        if self.timewiggle_zeta_available() {
            return self.timewiggle_design_psi_terms(
                block_states,
                derivative_blocks,
                psi_index,
                options,
            );
        }
        // Every other time-wiggle frame has no design-ψ calculus: per-score slopes, and the FLEX
        // program beside a follow-up-varying slope, which carries one slope primary.
        if self.flex_timewiggle_active() {
            return Err(SurvivalMarginalSlopeError::UnsupportedConfiguration {
                reason: "survival marginal-slope design ψ beside a time wiggle is served by the ζ \
                         composition only, which per-score slopes and a FLEX program beside a \
                         follow-up-varying slope do not have"
                    .to_string(),
            }
            .into());
        }
        let flex_active = self.effective_flex_active(block_states)?;
        let flex_primary = flex_active.then(|| flex_primary_slices(self));
        let slices = block_slices(self, block_states);
        let Some((block_idx, local_idx, p_psi, psi_label)) =
            self.psi_block_info(derivative_blocks, psi_index)?
        else {
            return Ok(None);
        };
        let deriv = &derivative_blocks[block_idx][local_idx];
        let beta_psi = match block_idx {
            1 => &block_states[1].beta,
            _ => &block_states[2].beta,
        };

        let p_t = slices.time.len();
        let p_m = slices.marginal.len();
        let p_g = slices.slope.len();
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
        ).map_err(|error| error.to_string())?;

        // Parallel accumulation: each worker gets its own block-local accumulators.
        type Acc = (
            f64,                     // objective_psi
            Array1<f64>,             // score_t
            Array1<f64>,             // score_m
            Array1<f64>,             // score_g
            Array1<f64>,             // score_h
            Array1<f64>,             // score_w
            Array1<f64>,             // score_i (absorbed influence)
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
                Array1::zeros(p_i),
                BlockHessianAccumulator::new(p_t, p_m, p_g, p_h, p_w, p_i),
            )
        };

        let row_iter = outer_row_indices(options, self.n).to_vec();
        let row_weights = outer_row_weights_by_index(options, self.n);
        // Process fixed row chunks in parallel and merge local cross-block
        // accumulators in row-chunk order for deterministic timewiggle assembly.
        let (objective_psi, score_t, score_m, score_g, score_h, score_w, score_i, acc) =
            chunked_row_reduction(
                row_iter.as_slice(),
                make_acc,
                |row, a| -> Result<(), String> {
                    let psi_row = psi_map
                        .row_vector(row)
                        .map_err(|e| format!("survival rowwise psi map: {e}"))?;

                    let channels =
                        psi_row_channels(self, flex_primary.as_ref(), row, block_idx, psi_row)?;
                    let dir = channels.direction(beta_psi.view());

                    let (mut f_pi, mut f_pipi) = if let Some(primary) = flex_primary.as_ref() {
                        let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
                        let (_, g, h) = self.compute_row_flex_primary_gradient_hessian_exact(
                            row,
                            block_states,
                            &q_geom,
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

                    for (loading, design_row) in channels.channels() {
                        let s1 = f_pi.dot(loading);
                        match block_idx {
                            1 => a.2.scaled_add(s1, design_row),
                            _ => a.3.scaled_add(s1, design_row),
                        }
                    }
                    let pb = f_pipi.dot(&dir);
                    self.accumulate_score_blockwise(row, &pb, &mut a.1, &mut a.2, &mut a.3)?;
                    self.accumulate_score_identity_blocks(
                        flex_primary.as_ref(),
                        &pb,
                        Some(&mut a.4),
                        Some(&mut a.5),
                    );
                    self.accumulate_score_influence_block(
                        flex_primary.as_ref(),
                        row,
                        &pb,
                        &mut a.6,
                    )?;

                    for (loading, design_row) in channels.channels() {
                        let right_primary = f_pipi.dot(loading);
                        a.7.add_rank1_psi_cross(self, row, block_idx, design_row, &right_primary)?;
                    }
                    a.7.add_pullback(self, row, &third)?;

                    Ok(())
                },
                |total, chunk| {
                    total.0 += chunk.0;
                    total.1 += &chunk.1;
                    total.2 += &chunk.2;
                    total.3 += &chunk.3;
                    total.4 += &chunk.4;
                    total.5 += &chunk.5;
                    total.6 += &chunk.6;
                    total.7.add(&chunk.7);
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
            .slice_mut(s![slices.slope.clone()])
            .assign(&score_g);
        if let Some(range) = slices.score_warp.as_ref() {
            score_psi.slice_mut(s![range.clone()]).assign(&score_h);
        }
        if let Some(range) = slices.link_dev.as_ref() {
            score_psi.slice_mut(s![range.clone()]).assign(&score_w);
        }
        if let Some(range) = slices.influence.as_ref() {
            score_psi.slice_mut(s![range.clone()]).assign(&score_i);
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
        // The batched tower is instantiated at `STATIC_SLOPE_PRIMARIES` and its
        // per-axis contraction writes an `N_PRIMARY × N_PRIMARY` block; a
        // follow-up-varying slope runs the six-primary frame, so this fast path
        // does not cover it and the per-axis route (which sizes every
        // primary-space vector from `core_primary_dimension`) does (#2765).
        if self.slope_is_follow_up_varying() {
            return Ok(None);
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
            ).map_err(|error| error.to_string())?;
            let loading = spatial_block_primary_loading(self, block_idx)?;
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
        let p_g = slices.slope.len();
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

                // The row program is direction-independent through order three.
                // Evaluate its exact statically sparse tower once, then contract
                // each ψ axis without repeating any transcendental algebra. Both
                // time-constant frames are four-primary, so the tower is dense
                // `4×4×4` either way; only its static sparsity differs, and the
                // contraction below reads it through the frame's own type.
                let third_tower = if self.anchored_law_active() {
                    TimeConstantThirdTower::Anchored(self.build_row_primary_third_tower::<
                        STATIC_SLOPE_PRIMARIES,
                        AnchoredStaticSlopeGeometry,
                    >(row, block_states)?)
                } else {
                    TimeConstantThirdTower::Gaussian(self.build_row_primary_third_tower::<
                        STATIC_SLOPE_PRIMARIES,
                        StaticSlopeGeometry,
                    >(row, block_states)?)
                };

                for axis_idx in 0..k {
                    let axis = &axes[axis_idx];
                    let psi_row = axis
                        .psi_map
                        .row_vector(row)
                        .map_err(|e| format!("survival rowwise psi map (batched): {e}"))?;
                    let dir =
                        primary_direction_from_psi_row(self, axis.block_idx, &psi_row, axis.beta_psi)?;
                    let third_stack = match &third_tower {
                        TimeConstantThirdTower::Gaussian(tower) => {
                            Self::contract_row_primary_third_tower::<
                                STATIC_SLOPE_PRIMARIES,
                                StaticSlopeGeometry,
                            >(tower, &dir)?
                        }
                        TimeConstantThirdTower::Anchored(tower) => {
                            Self::contract_row_primary_third_tower::<
                                STATIC_SLOPE_PRIMARIES,
                                AnchoredStaticSlopeGeometry,
                            >(tower, &dir)?
                        }
                    };
                    let mut third = Array2::from_shape_fn(
                        (N_PRIMARY, N_PRIMARY),
                        |(a, b)| third_stack[a][b],
                    );
                    if w != 1.0 {
                        third.mapv_inplace(|v| v * w);
                    }

                    let acc = &mut accs[axis_idx];

                    // objective_psi += w * (f_pi · dir)
                    acc.objective_psi += w * f_pi.dot(&dir);

                    // score_psi += w * (f_pi · loading) * psi_row, routed to
                    // the marginal or slope block depending on axis.
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
                .slice_mut(s![slices.slope.clone()])
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
        // A time wiggle takes the ζ composition; see `psi_terms_inner_with_options` (gam#2893).
        if self.timewiggle_zeta_available() {
            return self.timewiggle_design_psi_second_order_terms(
                block_states,
                derivative_blocks,
                psi_i,
                psi_j,
                options,
            );
        }
        // Every other time-wiggle frame has no design-ψ calculus: per-score slopes, and the FLEX
        // program beside a follow-up-varying slope, which carries one slope primary.
        if self.flex_timewiggle_active() {
            return Err(SurvivalMarginalSlopeError::UnsupportedConfiguration {
                reason: "survival marginal-slope design ψ beside a time wiggle is served by the ζ \
                         composition only, which per-score slopes and a FLEX program beside a \
                         follow-up-varying slope do not have"
                    .to_string(),
            }
            .into());
        }
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
        let beta_i = match block_idx_i {
            1 => &block_states[1].beta,
            _ => &block_states[2].beta,
        };
        let beta_j = match block_idx_j {
            1 => &block_states[1].beta,
            _ => &block_states[2].beta,
        };

        let p_t = slices.time.len();
        let p_m = slices.marginal.len();
        let p_g = slices.slope.len();
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
        ).map_err(|error| error.to_string())?;
        let psi_map_j = crate::custom_family::resolve_custom_family_x_psi_map(
            deriv_j,
            self.n,
            p_psi_j,
            0..self.n,
            label_j,
            &policy,
        ).map_err(|error| error.to_string())?;
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
            ).map_err(|error| error.to_string())?)
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
            score_i: Array1<f64>,
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
                score_i: Array1::zeros(p_i),
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
            score_i,
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

                let channels_i =
                    psi_row_channels(self, flex_primary.as_ref(), row, block_idx_i, psi_row_i)?;
                let channels_j =
                    psi_row_channels(self, flex_primary.as_ref(), row, block_idx_j, psi_row_j)?;
                let dir_i = channels_i.direction(beta_i.view());
                let dir_j = channels_j.direction(beta_j.view());

                // The cross-ψ rows, kept only when they are present AND not
                // identically zero. Both consumers below need the rows
                // themselves, so bind them here rather than re-deriving
                // presence from a bool and unwrapping the Option back open.
                let channels_ij = if same_block {
                    let r = psi_map_ij
                        .as_ref()
                        .expect("psi_map_ij built when same_block")
                        .row_vector(row)
                        .map_err(|e| format!("survival rowwise psi map: {e}"))?;
                    if r.iter().any(|v| v.abs() > 0.0) {
                        Some(psi_row_channels(self, flex_primary.as_ref(), row, block_idx_i, r)?)
                    } else {
                        None
                    }
                } else {
                    None
                };
                let dir_ij = match channels_ij.as_ref() {
                    Some(channels) => channels.direction(beta_i.view()),
                    None => Array1::<f64>::zeros(dir_i.len()),
                };

                let (mut f_pi, mut f_pipi) = if let Some(primary) = flex_primary.as_ref() {
                    let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
                    let (_, g, h) = self.compute_row_flex_primary_gradient_hessian_exact(
                        row,
                        block_states,
                        &q_geom,
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
                if let Some(channels) = channels_ij.as_ref() {
                    for (loading, design_row) in channels.channels() {
                        let s_ij = f_pi.dot(loading);
                        match block_idx_i {
                            1 => a.score_m.scaled_add(s_ij, design_row),
                            _ => a.score_g.scaled_add(s_ij, design_row),
                        }
                    }
                }
                let hessian_dir_j = f_pipi.dot(&dir_j);
                for (loading, design_row) in channels_i.channels() {
                    let s_i = loading.dot(&hessian_dir_j);
                    match block_idx_i {
                        1 => a.score_m.scaled_add(s_i, design_row),
                        _ => a.score_g.scaled_add(s_i, design_row),
                    }
                }
                let hessian_dir_i = f_pipi.dot(&dir_i);
                for (loading, design_row) in channels_j.channels() {
                    let s_j = loading.dot(&hessian_dir_i);
                    match block_idx_j {
                        1 => a.score_m.scaled_add(s_j, design_row),
                        _ => a.score_g.scaled_add(s_j, design_row),
                    }
                }
                let pb1 = f_pipi.dot(&dir_ij);
                self.accumulate_score_blockwise(
                    row,
                    &pb1,
                    &mut a.score_t,
                    &mut a.score_m,
                    &mut a.score_g,
                )?;
                self.accumulate_score_identity_blocks(
                    flex_primary.as_ref(),
                    &pb1,
                    Some(&mut a.score_h),
                    Some(&mut a.score_w),
                );
                self.accumulate_score_influence_block(
                    flex_primary.as_ref(),
                    row,
                    &pb1,
                    &mut a.score_i,
                )?;
                let pb2 = third_i.dot(&dir_j);
                self.accumulate_score_blockwise(
                    row,
                    &pb2,
                    &mut a.score_t,
                    &mut a.score_m,
                    &mut a.score_g,
                )?;
                self.accumulate_score_identity_blocks(
                    flex_primary.as_ref(),
                    &pb2,
                    Some(&mut a.score_h),
                    Some(&mut a.score_w),
                );
                self.accumulate_score_influence_block(
                    flex_primary.as_ref(),
                    row,
                    &pb2,
                    &mut a.score_i,
                )?;

                // Hessian
                if let Some(channels) = channels_ij.as_ref() {
                    for (loading, design_row) in channels.channels() {
                        let rp_ij = f_pipi.dot(loading);
                        a.hessian.add_rank1_psi_cross(
                            self,
                            row,
                            block_idx_i,
                            design_row,
                            &rp_ij,
                        )?;
                    }
                }
                for (loading_i, row_i) in channels_i.channels() {
                    for (loading_j, row_j) in channels_j.channels() {
                        a.hessian.add_psi_psi_outer(
                            block_idx_i,
                            row_i,
                            block_idx_j,
                            row_j,
                            loading_i.dot(&f_pipi.dot(loading_j)),
                        );
                    }
                }
                for (loading_i, row_i) in channels_i.channels() {
                    let rp_i = third_j.t().dot(loading_i);
                    a.hessian
                        .add_rank1_psi_cross(self, row, block_idx_i, row_i, &rp_i)?;
                }
                for (loading_j, row_j) in channels_j.channels() {
                    let rp_j = third_i.t().dot(loading_j);
                    a.hessian
                        .add_rank1_psi_cross(self, row, block_idx_j, row_j, &rp_j)?;
                }
                a.hessian.add_pullback(self, row, &fourth)?;
                let mut third_ij =
                    self.row_primary_third_contracted_general(row, block_states, &dir_ij)?;
                if w != 1.0 {
                    third_ij.mapv_inplace(|v| v * w);
                }
                a.hessian.add_pullback(self, row, &third_ij)?;

                Ok(())
            },
            |total, chunk| {
                total.objective_psi_psi += chunk.objective_psi_psi;
                total.score_t += &chunk.score_t;
                total.score_m += &chunk.score_m;
                total.score_g += &chunk.score_g;
                total.score_h += &chunk.score_h;
                total.score_w += &chunk.score_w;
                total.score_i += &chunk.score_i;
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
            .slice_mut(s![slices.slope.clone()])
            .assign(&score_g);
        if let Some(range) = slices.score_warp.as_ref() {
            score_psi_psi.slice_mut(s![range.clone()]).assign(&score_h);
        }
        if let Some(range) = slices.link_dev.as_ref() {
            score_psi_psi.slice_mut(s![range.clone()]).assign(&score_w);
        }
        if let Some(range) = slices.influence.as_ref() {
            score_psi_psi.slice_mut(s![range.clone()]).assign(&score_i);
        }

        Ok(Some(ExactNewtonJointPsiSecondOrderTerms {
            objective_psi_psi,
            score_psi_psi,
            hessian_psi_psi: Array2::zeros((0, 0)),
            hessian_psi_psi_operator: Some(Arc::new(hessian.into_operator(slices))),
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
        // Every other time-wiggle frame has no design-ψ calculus: per-score slopes, and the FLEX
        // program beside a follow-up-varying slope, which carries one slope primary.
        if self.flex_timewiggle_active() {
            return Err(SurvivalMarginalSlopeError::UnsupportedConfiguration {
                reason: "survival marginal-slope design ψ beside a time wiggle is served by the ζ \
                         composition only, which per-score slopes and a FLEX program beside a \
                         follow-up-varying slope do not have"
                    .to_string(),
            }
            .into());
        }
        let flex_active = self.effective_flex_active(block_states)?;
        let flex_primary = flex_active.then(|| flex_primary_slices(self));
        let slices = block_slices(self, block_states);
        let Some((block_idx, local_idx, p_psi, psi_label)) =
            self.psi_block_info(derivative_blocks, psi_index)?
        else {
            return Ok(None);
        };
        let deriv = &derivative_blocks[block_idx][local_idx];
        let beta_psi = match block_idx {
            1 => &block_states[1].beta,
            _ => &block_states[2].beta,
        };
        let d_beta_block = match block_idx {
            1 => d_beta_flat.slice(s![slices.marginal.clone()]),
            _ => d_beta_flat.slice(s![slices.slope.clone()]),
        };

        let p_t = slices.time.len();
        let p_m = slices.marginal.len();
        let p_g = slices.slope.len();
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
        ).map_err(|error| error.to_string())?;

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

                let channels =
                    psi_row_channels(self, flex_primary.as_ref(), row, block_idx, psi_row)?;
                let psi_dir = channels.direction(beta_psi.view());
                let psi_action = channels.direction(d_beta_block);
                let row_dir = self.row_primary_direction_from_flat_dynamic(
                    row,
                    block_states,
                    &slices,
                    d_beta_flat,
                )?;
                let w = row_weights[row];
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

                for (loading, design_row) in channels.channels() {
                    let right_primary = third_beta.t().dot(loading);
                    acc.add_rank1_psi_cross(self, row, block_idx, design_row, &right_primary)?;
                }
                acc.add_pullback(self, row, &fourth)?;
                let mut third_action =
                    self.row_primary_third_contracted_general(row, block_states, &psi_action)?;
                if w != 1.0 {
                    third_action.mapv_inplace(|v| v * w);
                }
                acc.add_pullback(self, row, &third_action)?;
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
        // A time wiggle takes the ζ composition; see `psi_terms_inner_with_options` (gam#2893).
        if self.timewiggle_zeta_available() {
            return self.timewiggle_design_psi_hessian_drift(
                block_states,
                derivative_blocks,
                psi_index,
                d_beta_flat,
                options,
            );
        }
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
    /// [`Self::psi_hessian_directional_derivative_with_options`] along EVERY
    /// joint coefficient axis of one design-derivative ψ axis, from ONE row
    /// sweep.
    ///
    /// The per-axis accumulator evaluates each row's third- and fourth-order
    /// kernels once per coefficient axis, so the ψ-hyper build's Firth branch,
    /// which needs the derivative along all `p` axes, swept the rows `3p` times
    /// per ψ axis (gam#979: 58 s per gradient at n=4800, p=89). Every per-axis
    /// input is linear in the direction: the row's primary direction is
    /// `L_row·e_a` for a fixed `P×p` loading (three time/marginal functionals
    /// and the slope channels' design rows), and the ψ action is
    /// `psi_row[a]` times a fixed placement. So each row's kernels are
    /// evaluated once per primary basis vector (`P` third-order calls and `P`
    /// fourth-order calls contracted with the ψ direction) and contracted per
    /// axis with the sparse loading column; only the design pullbacks remain
    /// per axis.
    ///
    /// A time wiggle takes the ζ sweep of `timewiggle_third` under the same row
    /// measure (gam#2893). A FLEX frame or a follow-up-varying slope without one
    /// carries its own primary layout and keeps the per-axis path: this returns
    /// `None` there, exactly as it
    /// does where the per-axis path has no ψ block, so a caller falls back to the
    /// per-axis sweep with identical semantics.
    pub(crate) fn psi_hessian_directional_derivatives_all_beta_axes_with_options(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Vec<Array2<f64>>>, String> {
        if self.timewiggle_zeta_available() {
            return self.timewiggle_design_psi_hessian_all_beta_axes(
                block_states,
                derivative_blocks,
                psi_index,
                &self.rigid_third_row_weights(options),
            );
        }
        if self.effective_flex_active(block_states)?
            || self.flex_timewiggle_active()
            || self.slope_is_follow_up_varying()
        {
            return Ok(None);
        }
        let slices = block_slices(self, block_states);
        let Some((block_idx, local_idx, p_psi, psi_label)) =
            self.psi_block_info(derivative_blocks, psi_index)?
        else {
            return Ok(None);
        };
        let deriv = &derivative_blocks[block_idx][local_idx];
        let loading = spatial_block_primary_loading(self, block_idx)?;
        let beta_psi = match block_idx {
            1 => &block_states[1].beta,
            _ => &block_states[2].beta,
        };
        let psi_block_range = match block_idx {
            1 => slices.marginal.clone(),
            _ => slices.slope.clone(),
        };
        let primary_dim = self.core_primary_dimension();
        // The ψ action of the per-axis path is `psi_row · d_beta_block` placed
        // in these primary slots (`PsiRowChannels::direction`).
        let mut placement = Array1::<f64>::zeros(primary_dim);
        if block_idx == 1 {
            placement[PRIMARY_Q0] = 1.0;
            placement[PRIMARY_Q1] = 1.0;
        } else {
            placement[PRIMARY_SLOPE] = 1.0;
        }
        let basis: Vec<Array1<f64>> = (0..primary_dim)
            .map(|k| {
                let mut e = Array1::<f64>::zeros(primary_dim);
                e[k] = 1.0;
                e
            })
            .collect();

        let p_t = slices.time.len();
        let p_m = slices.marginal.len();
        let p_g = slices.slope.len();
        let p_h = slices.score_warp.as_ref().map_or(0, |range| range.len());
        let p_w = slices.link_dev.as_ref().map_or(0, |range| range.len());
        let p_i = slices.influence.as_ref().map_or(0, |range| range.len());
        let p_total = slices.total;

        let policy = gam_runtime::resource::ResourcePolicy::default_library();
        let psi_map = crate::custom_family::resolve_custom_family_x_psi_map(
            deriv,
            self.n,
            p_psi,
            0..self.n,
            psi_label,
            &policy,
        )
        .map_err(|error| error.to_string())?;

        let row_iter = outer_row_indices(options, self.n).to_vec();
        let row_weights = outer_row_weights_by_index(options, self.n);
        let slope_channels = self.slope_layout.primary_channels();
        let accs = chunked_row_reduction(
            row_iter.as_slice(),
            || {
                (0..p_total)
                    .map(|_| BlockHessianAccumulator::new(p_t, p_m, p_g, p_h, p_w, p_i))
                    .collect::<Vec<_>>()
            },
            |row, accs| -> Result<(), String> {
                let psi_row = psi_map
                    .row_vector(row)
                    .map_err(|e| format!("survival rowwise psi map: {e}"))?;
                let psi_dir = primary_direction_from_psi_row(self, block_idx, &psi_row, beta_psi)?;
                let w = row_weights[row];
                // The row's kernels once per primary basis vector.
                let mut third: Vec<Array2<f64>> = Vec::with_capacity(primary_dim);
                let mut fourth: Vec<Array2<f64>> = Vec::with_capacity(primary_dim);
                for e in &basis {
                    let mut t = self.row_primary_third_contracted_general(row, block_states, e)?;
                    let mut f =
                        self.row_primary_fourth_contracted_general(row, block_states, e, &psi_dir)?;
                    if w != 1.0 {
                        t.mapv_inplace(|v| v * w);
                        f.mapv_inplace(|v| v * w);
                    }
                    third.push(t);
                    fourth.push(f);
                }
                // The third-order kernel contracted with the ψ placement: the
                // per-axis `third_action` is this times `psi_row[a]`.
                let mut action = Array2::<f64>::zeros((primary_dim, primary_dim));
                for (k, t) in third.iter().enumerate() {
                    if placement[k] != 0.0 {
                        action.scaled_add(placement[k], t);
                    }
                }
                // The row's loading `L_row`: each primary as a linear functional
                // of β (`row_primary_direction_from_flat_dynamic_with_q_geometry`).
                let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
                let slope_rows: Vec<(usize, Array1<f64>)> = slope_channels
                    .as_slice()
                    .iter()
                    .map(|&(primary, design)| -> Result<(usize, Array1<f64>), String> {
                        let chunk = design
                            .try_row_chunk(row..row + 1)
                            .map_err(|e| format!("survival slope channel design row: {e}"))?;
                        Ok((primary, chunk.row(0).to_owned()))
                    })
                    .collect::<Result<_, _>>()?;
                let mut row_dir = Array1::<f64>::zeros(primary_dim);
                let mut third_beta = Array2::<f64>::zeros((primary_dim, primary_dim));
                let mut fourth_a = Array2::<f64>::zeros((primary_dim, primary_dim));
                for axis in 0..p_total {
                    row_dir.fill(0.0);
                    if slices.time.contains(&axis) {
                        let j = axis - slices.time.start;
                        row_dir[PRIMARY_Q0] = q_geom.dq0_time[j];
                        row_dir[PRIMARY_Q1] = q_geom.dq1_time[j];
                        row_dir[PRIMARY_QD1] = q_geom.dqd1_time[j];
                    } else if slices.marginal.contains(&axis) {
                        let j = axis - slices.marginal.start;
                        row_dir[PRIMARY_Q0] = q_geom.dq0_marginal[j];
                        row_dir[PRIMARY_Q1] = q_geom.dq1_marginal[j];
                        row_dir[PRIMARY_QD1] = q_geom.dqd1_marginal[j];
                    } else if slices.slope.contains(&axis) {
                        let j = axis - slices.slope.start;
                        for (primary, design_row) in &slope_rows {
                            row_dir[*primary] = design_row[j];
                        }
                    }
                    let in_psi_block = psi_block_range.contains(&axis);
                    let direction_is_zero = row_dir.iter().all(|v| *v == 0.0);
                    if direction_is_zero && !in_psi_block {
                        continue;
                    }
                    third_beta.fill(0.0);
                    fourth_a.fill(0.0);
                    for k in 0..primary_dim {
                        if row_dir[k] != 0.0 {
                            third_beta.scaled_add(row_dir[k], &third[k]);
                            fourth_a.scaled_add(row_dir[k], &fourth[k]);
                        }
                    }
                    let right_primary = third_beta.t().dot(&loading);
                    accs[axis].add_rank1_psi_cross(self, row, block_idx, &psi_row, &right_primary)?;
                    accs[axis].add_pullback(self, row, &fourth_a)?;
                    if in_psi_block {
                        let coefficient = psi_row[axis - psi_block_range.start];
                        if coefficient != 0.0 {
                            let third_action = action.mapv(|v| v * coefficient);
                            accs[axis].add_pullback(self, row, &third_action)?;
                        }
                    }
                }
                Ok(())
            },
            |total, chunk| {
                for (t, c) in total.iter_mut().zip(chunk.iter()) {
                    t.add(c);
                }
            },
        )?;
        Ok(Some(accs.iter().map(|acc| acc.to_dense(&slices)).collect()))
    }

    pub(crate) fn psi_hessian_directional_derivative_operator_with_options(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Arc<dyn HyperOperator>>, String> {
        // A time wiggle takes the ζ composition; see `psi_terms_inner_with_options` (gam#2893).
        if self.timewiggle_zeta_available() {
            return Ok(self
                .timewiggle_design_psi_hessian_drift(
                    block_states,
                    derivative_blocks,
                    psi_index,
                    d_beta_flat,
                    options,
                )?
                .map(|matrix| {
                    Arc::new(gam_problem::DenseMatrixHyperOperator { matrix })
                        as Arc<dyn HyperOperator>
                }));
        }
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
