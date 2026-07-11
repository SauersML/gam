//! The exact-Newton joint evaluation methods on the family: GPU flex
//! dispatch, dense/gradient dynamic-q evaluation, time-wiggle and
//! flex-no-wiggle directional derivatives, and the blockwise exact-Newton
//! dispatchers (rigid / per-z / flexible / time-wiggle / mixed / dense /
//! sparse).

use super::*;

impl SurvivalMarginalSlopeFamily {
    /// Unified dense joint Hessian assembly for flex and timewiggle paths.
    /// Both paths use q-geometry Jacobians via accumulate_dynamic_q_joint_row.
    /// The rigid path (no flex, no timewiggle) uses the RowKernel fast path.
    /// Owned scratch buffers backing a [`SurvivalFlexGpuRowInputs`]
    /// descriptor for one fit-evaluation call.
    ///
    /// Held by-value across the GPU `try_*` entry so the borrowed slices
    /// in `as_inputs` live as long as the dispatcher invocation.
    pub(crate) fn build_survival_flex_gpu_row_batch(
        &self,
        block_states: &[ParameterBlockState],
        slices: &BlockSlices,
    ) -> Result<Option<SurvivalFlexGpuRowBatch>, String> {
        // Only the scalar-score path is currently representable in
        // `SurvivalFlexGpuRowInputs::score_dim == 1`.  Vector-score
        // and any per-row composition outside the canonical 3-block
        // (time, marginal, logslope) layout must take the CPU path.
        if self.score_dim() != 1 {
            return Ok(None);
        }
        // The absorbed Stage-1 influence channel (#461) adds a per-row index
        // offset `o_infl = Z̃_infl[row,:]·γ` to η₁ that the GPU flex kernel does
        // not yet carry (the on-device kernel emits a fixed 4-primary jet). Until
        // the survival flex GPU kernel grows the `o_infl` primary coordinate,
        // force CPU for absorber-active fits so the device path can never silently
        // drop the channel; the CPU path is the source of truth.
        if self.influence_absorber.is_some() {
            return Ok(None);
        }
        let n = self.n;
        let g_eta: &Array1<f64> = &block_states[2].eta;
        if g_eta.len() != n {
            return Ok(None);
        }
        let mut q0 = vec![0.0_f64; n];
        let mut q1 = vec![0.0_f64; n];
        let mut qd1 = vec![0.0_f64; n];
        let mut z = vec![0.0_f64; n];
        let mut g = vec![0.0_f64; n];
        // Per-row q-values reuse the canonical `row_dynamic_q_values`
        // helper so the GPU descriptor sees exactly the same q-geometry
        // the CPU per-row primary path consumes (timewiggle-aware when
        // active).
        for row in 0..n {
            let qv = self.row_dynamic_q_values(row, block_states)?;
            q0[row] = qv.q0;
            q1[row] = qv.q1;
            qd1[row] = qv.qd1;
            z[row] = self.z[[row, 0]];
            g[row] = g_eta[row];
        }
        let p = slices.total;
        // Materialize the joint-β vector in joint-block order.  Length
        // must equal `p = slices.total` to satisfy the GPU descriptor's
        // shape contract; mismatches force CPU fallback.
        let mut beta = vec![0.0_f64; p];
        // Returns `false` when the block β width does not match its joint
        // slice. That invariant (`block_states[i].beta.len() ==
        // design_i.ncols() == slice.len()`) holds for every well-formed inner
        // state; if it is ever violated the CPU per-row path hard-errors (e.g.
        // the `beta.len() != design_derivative_exit.ncols()` check). Silently
        // leaving the block as zeros would feed the GPU kernel a corrupted β
        // with no error and no fallback, so a mismatch forces CPU instead.
        let copy_block =
            |dst: &mut [f64], range: &std::ops::Range<usize>, src: &Array1<f64>| -> bool {
                if src.len() != range.len() {
                    return false;
                }
                if let Some(slice) = src.as_slice() {
                    dst[range.clone()].copy_from_slice(slice);
                } else {
                    // Non-contiguous Array1 — copy element-wise so the GPU
                    // descriptor still sees the canonical joint-block β.
                    for (offset, value) in src.iter().enumerate() {
                        dst[range.start + offset] = *value;
                    }
                }
                true
            };
        let mut block_widths_match = copy_block(&mut beta, &slices.time, &block_states[0].beta)
            && copy_block(&mut beta, &slices.marginal, &block_states[1].beta)
            && copy_block(&mut beta, &slices.logslope, &block_states[2].beta);
        if let Some(range) = slices.score_warp.as_ref() {
            block_widths_match =
                block_widths_match && copy_block(&mut beta, range, &block_states[3].beta);
        }
        if let Some(range) = slices.link_dev.as_ref() {
            let block_index = 3 + usize::from(self.score_warp.is_some());
            block_widths_match =
                block_widths_match && copy_block(&mut beta, range, &block_states[block_index].beta);
        }
        if !block_widths_match {
            return Ok(None);
        }
        Ok(Some(SurvivalFlexGpuRowBatch {
            n,
            p,
            q0,
            q1,
            qd1,
            z,
            g,
            beta,
            weights: self.weights.to_vec(),
            event: self.event.to_vec(),
        }))
    }

    /// Step-6 dispatcher for the joint-β gradient.  Builds the
    /// `SurvivalFlexGpuRowInputs` descriptor, runs the GPU policy
    /// `decide`, and routes through
    /// [`crate::survival::marginal_slope::gpu::try_survival_flex_gradient`].
    ///
    /// Returns:
    ///
    /// * `Ok(None)` when the GPU path is gated off (policy, shape,
    ///   backend-not-compiled, or runtime declined) — callers fall back
    ///   to the existing CPU per-row sweep.
    /// * `Ok(Some((nll, grad)))` when the GPU produced a usable answer.
    /// * `Err(_)` only when `gpu=required` was requested but the kernel is
    ///   not supported, mirroring the convention in `gpu::decide`.
    pub(crate) fn try_survival_flex_joint_dispatch_gradient(
        &self,
        block_states: &[ParameterBlockState],
        slices: &BlockSlices,
    ) -> Result<Option<(f64, Array1<f64>)>, String> {
        let decision =
            crate::survival::marginal_slope::gpu::row_primary_hessian_decision(self.n, N_PRIMARY);
        decision.clone().log();
        if !decision.use_gpu {
            decision.require_supported()?;
            return Ok(None);
        }
        let batch = match self.build_survival_flex_gpu_row_batch(block_states, slices)? {
            Some(b) => b,
            None => {
                decision.require_supported()?;
                return Ok(None);
            }
        };
        let inputs = batch.as_inputs(self);
        match crate::survival::marginal_slope::gpu::try_survival_flex_gradient(inputs, None, None)
            .map_err(|e| e.to_string())?
        {
            Some((nll, grad)) => {
                if grad.len() != slices.total {
                    return Err(format!(
                        "survival-flex GPU gradient returned len={} but joint p={}",
                        grad.len(),
                        slices.total
                    ));
                }
                Ok(Some((nll, grad)))
            }
            None => {
                decision.require_supported()?;
                Ok(None)
            }
        }
    }

    /// Step-6 dispatcher for the joint Hessian × vector product.  See
    /// [`Self::try_survival_flex_joint_dispatch_gradient`] for the
    /// `decide`/fallback semantics.
    pub(crate) fn try_survival_flex_joint_dispatch_hvp(
        &self,
        block_states: &[ParameterBlockState],
        slices: &BlockSlices,
        v: &Array1<f64>,
    ) -> Result<Option<Array1<f64>>, String> {
        let decision =
            crate::survival::marginal_slope::gpu::row_primary_hessian_decision(self.n, N_PRIMARY);
        decision.clone().log();
        if !decision.use_gpu {
            decision.require_supported()?;
            return Ok(None);
        }
        if v.len() != slices.total {
            return Ok(None);
        }
        let batch = match self.build_survival_flex_gpu_row_batch(block_states, slices)? {
            Some(b) => b,
            None => {
                decision.require_supported()?;
                return Ok(None);
            }
        };
        let inputs = batch.as_inputs(self);
        let v_slice = v
            .as_slice()
            .ok_or_else(|| "survival-flex GPU HVP requires contiguous v".to_string())?;
        match crate::survival::marginal_slope::gpu::try_survival_flex_hvp(inputs, v_slice, None)
            .map_err(|e| e.to_string())?
        {
            Some(hv) => {
                if hv.len() != slices.total {
                    return Err(format!(
                        "survival-flex GPU HVP returned len={} but joint p={}",
                        hv.len(),
                        slices.total
                    ));
                }
                Ok(Some(hv))
            }
            None => {
                decision.require_supported()?;
                Ok(None)
            }
        }
    }

    /// Step-6 dispatcher for the dense joint Hessian.  Returns
    /// `Ok(Some(H))` on a successful device assembly, `Ok(None)` for the
    /// CPU fallback, and `Err(_)` only for `gpu=required` shape mismatches.
    pub(crate) fn try_survival_flex_joint_dispatch_dense_hessian(
        &self,
        block_states: &[ParameterBlockState],
        slices: &BlockSlices,
    ) -> Result<Option<Array2<f64>>, String> {
        let decision =
            crate::survival::marginal_slope::gpu::row_primary_hessian_decision(self.n, N_PRIMARY);
        decision.clone().log();
        if !decision.use_gpu {
            decision.require_supported()?;
            return Ok(None);
        }
        let batch = match self.build_survival_flex_gpu_row_batch(block_states, slices)? {
            Some(b) => b,
            None => {
                decision.require_supported()?;
                return Ok(None);
            }
        };
        let inputs = batch.as_inputs(self);
        match crate::survival::marginal_slope::gpu::try_survival_flex_dense_hessian(
            inputs, None, None,
        )
        .map_err(|e| e.to_string())?
        {
            Some(h) => {
                let p = slices.total;
                if h.shape() != [p, p] {
                    return Err(format!(
                        "survival-flex GPU dense H returned shape {:?} but joint p={}",
                        h.shape(),
                        p
                    ));
                }
                Ok(Some(h))
            }
            None => {
                decision.require_supported()?;
                Ok(None)
            }
        }
    }

    pub(crate) fn evaluate_exact_newton_joint_dynamic_q_dense(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        let flex_active = self.effective_flex_active(block_states)?;
        if flex_active {
            self.validate_exact_monotonicity(block_states)?;
        }
        let slices = block_slices(self, block_states);
        // ── Step-6 dispatcher: try GPU dense H + gradient first ──────────
        //
        // The two `try_survival_flex_joint_dispatch_*` entries route the
        // joint-β work through
        // [`crate::survival::marginal_slope::gpu::try_survival_flex_dense_hessian`]
        // and [`crate::survival::marginal_slope::gpu::try_survival_flex_gradient`]
        // respectively, with the standard `gpu::decide` policy.
        //
        // State of the seam (#1133): the host-side Step-5 primary G/H
        // assembly (`try_device_step5_primary_assembly`) and the Step-6
        // joint-β pullback (`pullback_step6_joint_beta`) are both LANDED as
        // pure host algebra and CPU-verified — Step 5 is already the hot
        // per-row path in `compute_row_flex_primary_gradient_hessian_from_parts`
        // (the CPU sweep below routes every row through it). What remains is
        // ONLY the device substrate: an NVRTC/CUDA kernel that produces the
        // per-row jets + folds the Step-6 contraction on-device. Until that
        // kernel exists these batch entry points are called with `step6 =
        // None` and return `Ok(None)`, so the CPU per-row sweep below is the
        // production path. Threading assembled Step-5/Step-6 rows through the
        // batch entry points here would only duplicate the already-complete
        // per-block pullback in `accumulate_dynamic_q_joint_row` on the host,
        // so it is deliberately deferred to the device-kernel work.
        if flex_active && !self.flex_timewiggle_active() {
            if let Some(h) =
                self.try_survival_flex_joint_dispatch_dense_hessian(block_states, &slices)?
            {
                let (nll, grad) =
                    match self.try_survival_flex_joint_dispatch_gradient(block_states, &slices)? {
                        Some(pair) => pair,
                        None => {
                            return Err(
                                "survival-flex GPU dense H succeeded but gradient declined; \
                             prep dispatchers must compose consistently"
                                    .to_string(),
                            );
                        }
                    };
                return Ok((nll, grad, h));
            }
        }
        let primary = flex_primary_slices(self);
        let p_total = slices.total;
        let identity_blocks = if flex_active {
            flex_identity_block_pairs(&primary, &slices)
        } else {
            vec![]
        };

        type Acc = (f64, Array1<f64>, Array2<f64>);
        // Per-thread accumulator carries a `SurvivalMarginalSlopeDynamicRow`
        // workspace alongside the (nll, gradient, hessian) tuple so the nine
        // Array2/Array1 buffers inside it are reused across all rows assigned
        // to a single rayon worker. At large scale this eliminates the
        // ~80 GB-per-sweep allocator traffic the fresh-allocation path used.
        type AccWithWs = (Acc, SurvivalMarginalSlopeDynamicRow);
        let make_acc = || -> AccWithWs {
            (
                (
                    0.0,
                    Array1::zeros(p_total),
                    Array2::zeros((p_total, p_total)),
                ),
                SurvivalMarginalSlopeDynamicRow::empty_workspace(),
            )
        };

        let final_acc = gam_problem::outer_subsample::RowSet::All.par_try_reduce_fold(
            self.n,
            make_acc,
            |mut acc, row, _row_weight| -> Result<_, String> {
                let (state, q_geom) = &mut acc;
                self.row_dynamic_q_geometry_into(row, block_states, q_geom)?;
                let (row_nll, f_pi, f_pipi) = if flex_active {
                    self.compute_row_flex_primary_gradient_hessian_exact(
                        row,
                        block_states,
                        q_geom,
                        &primary,
                    )?
                } else {
                    self.compute_row_primary_gradient_hessian_uncached(row, block_states)?
                };
                state.0 -= row_nll;
                self.accumulate_dynamic_q_joint_row(
                    row,
                    &slices,
                    q_geom,
                    f_pi.view(),
                    f_pipi.view(),
                    &identity_blocks,
                    &mut state.1,
                    &mut state.2,
                )?;
                Ok(acc)
            },
            |mut left, right| -> Result<_, String> {
                left.0.0 += right.0.0;
                left.0.1 += &right.0.1;
                left.0.2 += &right.0.2;
                Ok(left)
            },
        )?;
        Ok(final_acc.0)
    }

    pub(crate) fn evaluate_exact_newton_joint_dense(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        if self.effective_flex_active(block_states)? || self.flex_timewiggle_active() {
            self.evaluate_exact_newton_joint_dynamic_q_dense(block_states)
        } else {
            let kern = SurvivalMarginalSlopeRowKernel::new(self.clone(), block_states.to_vec());
            let rows = crate::row_kernel::RowSet::All;
            let cache = build_row_kernel_cache(&kern, &rows)?;
            Ok((
                row_kernel_log_likelihood(&cache, &rows),
                -row_kernel_gradient(&kern, &cache, &rows),
                row_kernel_hessian_dense(&kern, &cache, &rows),
            ))
        }
    }

    pub(crate) fn evaluate_exact_newton_joint_gradient_dynamic_q(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<(f64, Array1<f64>), String> {
        let flex_active = self.effective_flex_active(block_states)?;
        let slices = block_slices(self, block_states);
        // ── Step-6 dispatcher: try GPU joint-β gradient first ────────────
        //
        // Routes through
        // [`crate::survival::marginal_slope::gpu::try_survival_flex_gradient`] via
        // the `gpu::decide` policy.  Returns `Ok(None)` until the device
        // CUDA kernel lands: the host-side Step-5 assembly + Step-6 joint-β
        // pullback are already LANDED + CPU-verified (Step 5 is the hot
        // per-row path), so only the on-device jet/contraction substrate is
        // outstanding (#1133). Until then the CPU per-row sweep below is the
        // production path and this dispatch is a no-op fast-fail.
        if flex_active && !self.flex_timewiggle_active() {
            if let Some(pair) =
                self.try_survival_flex_joint_dispatch_gradient(block_states, &slices)?
            {
                return Ok(pair);
            }
        }
        let primary = flex_primary_slices(self);
        let identity_blocks = if flex_active {
            flex_identity_block_pairs(&primary, &slices)
        } else {
            vec![]
        };
        type Acc = (f64, Array1<f64>);
        let make_acc = || -> Acc { (0.0, Array1::zeros(slices.total)) };

        gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            self.n,
            |range| -> Result<_, String> {
                let mut acc = make_acc();
                for row in range {
                    let q_geom = self.row_dynamic_q_gradient(row, block_states)?;
                    let (row_nll, f_pi) = if flex_active {
                        self.compute_row_flex_primary_gradient_exact(
                            row,
                            block_states,
                            &q_geom,
                            &primary,
                        )?
                    } else {
                        let (nll, grad, _) =
                            self.compute_row_primary_gradient_hessian_uncached(row, block_states)?;
                        (nll, grad)
                    };
                    acc.0 -= row_nll;
                    self.accumulate_dynamic_q_core_gradient_first_order(
                        row,
                        &slices,
                        &q_geom,
                        f_pi.slice(s![0..N_PRIMARY]),
                        &mut acc.1,
                    )?;
                    for (primary_range, joint_range) in &identity_blocks {
                        for local in 0..primary_range.len() {
                            acc.1[joint_range.start + local] -= f_pi[primary_range.start + local];
                        }
                    }
                }
                Ok(acc)
            },
            |mut left, right| -> Result<_, String> {
                left.0 += right.0;
                left.1 += &right.1;
                Ok(left)
            },
        )
        .map(|opt| opt.unwrap_or_else(make_acc))
    }

    /// Shared per-row pullback of the timewiggle q-map Jacobian/curvature
    /// derivatives into the joint Hessian accumulator.  Used by both the
    /// cached `_inner` path (with an empty `identity_blocks`) and the flex
    /// path (with non-empty `identity_blocks`); the only behavioural
    /// difference between those callers is whether the identity-block cross
    /// terms are added, which is driven entirely by `identity_blocks`.
    pub(crate) fn accumulate_timewiggle_directional_row(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        slices: &BlockSlices,
        q_geom: &SurvivalMarginalSlopeDynamicRow,
        f_pi: &Array1<f64>,
        h_pi: ArrayView2<'_, f64>,
        d_time: ndarray::ArrayView1<'_, f64>,
        d_marginal: ndarray::ArrayView1<'_, f64>,
        beta_time: &Array1<f64>,
        beta_time_w: ndarray::ArrayView1<'_, f64>,
        identity_blocks: &[(std::ops::Range<usize>, std::ops::Range<usize>)],
        acc: &mut Array2<f64>,
    ) -> Result<(), String> {
        let time_tail = self.time_wiggle_range();
        let p_base = time_tail.start;
        let p_time = slices.time.len();
        let p_marginal = slices.marginal.len();

        // ── Timewiggle Jacobian derivatives ────────────────
        let ec = self
            .design_entry
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("design_entry try_row_chunk: {e}"))?;
        let xc = self
            .design_exit
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("design_exit try_row_chunk: {e}"))?;
        let dc = self
            .design_derivative_exit
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("design_derivative_exit try_row_chunk: {e}"))?;
        let xe = ec.row(0).slice(s![..p_base]).to_owned();
        let xx = xc.row(0).slice(s![..p_base]).to_owned();
        let xd = dc.row(0).slice(s![..p_base]).to_owned();
        let mc = self
            .marginal_design
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("marginal_design try_row_chunk: {e}"))?;
        let mr = mc.row(0).to_owned();
        let dh0 = xe.dot(&d_time.slice(s![..p_base])) + mr.dot(&d_marginal);
        let dh1 = xx.dot(&d_time.slice(s![..p_base])) + mr.dot(&d_marginal);
        let ddr = xd.dot(&d_time.slice(s![..p_base]));
        let bm = block_states[1].eta[row];
        let h0 = xe.dot(&beta_time.slice(s![..p_base])) + self.offset_entry[row] + bm;
        let h1 = xx.dot(&beta_time.slice(s![..p_base])) + self.offset_exit[row] + bm;
        let dr = xd.dot(&beta_time.slice(s![..p_base])) + self.derivative_offset_exit[row];
        let eg = self
            .time_wiggle_geometry(Array1::from_vec(vec![h0]).view(), beta_time_w)?
            .ok_or_else(|| "timewiggle geometry missing at entry".to_string())?;
        let xg = self
            .time_wiggle_geometry(Array1::from_vec(vec![h1]).view(), beta_time_w)?
            .ok_or_else(|| "timewiggle geometry missing at exit".to_string())?;
        let (m2e, m3e) = (eg.d2q_dq02[0], eg.d3q_dq03[0]);
        let (m2x, m3x, m4x) = (xg.d2q_dq02[0], xg.d3q_dq03[0], xg.d4q_dq04[0]);

        // dJ_{q,time}[a] / dβ[d]
        let mut dj0t = vec![0.0f64; p_time];
        let mut dj1t = vec![0.0f64; p_time];
        let mut djdt = vec![0.0f64; p_time];
        for a in 0..p_base {
            dj0t[a] = m2e * dh0 * xe[a];
            dj1t[a] = m2x * dh1 * xx[a];
            djdt[a] = m3x * dh1 * dr * xx[a] + m2x * ddr * xx[a] + m2x * dh1 * xd[a];
        }
        for li in 0..time_tail.len() {
            let ci = time_tail.start + li;
            dj0t[ci] = eg.basis_d1[[0, li]] * dh0;
            dj1t[ci] = xg.basis_d1[[0, li]] * dh1;
            djdt[ci] = xg.basis_d2[[0, li]] * dh1 * dr + xg.basis_d1[[0, li]] * ddr;
        }
        let djt = [&dj0t[..], &dj1t[..], &djdt[..]];
        let mut dj0m = vec![0.0f64; p_marginal];
        let mut dj1m = vec![0.0f64; p_marginal];
        let mut djdm = vec![0.0f64; p_marginal];
        for a in 0..p_marginal {
            dj0m[a] = m2e * dh0 * mr[a];
            dj1m[a] = m2x * dh1 * mr[a];
            djdm[a] = m3x * dh1 * dr * mr[a] + m2x * ddr * mr[a];
        }
        let djm = [&dj0m[..], &dj1m[..], &djdm[..]];
        let jt: [&Array1<f64>; 3] = [&q_geom.dq0_time, &q_geom.dq1_time, &q_geom.dqd1_time];
        let jm: [&Array1<f64>; 3] = [
            &q_geom.dq0_marginal,
            &q_geom.dq1_marginal,
            &q_geom.dqd1_marginal,
        ];

        // Term 2: (dJ/d)^T H J + J^T H (dJ/d)
        for a in 0..p_time {
            for b in 0..p_time {
                let mut v = 0.0;
                for qu in 0..3 {
                    for qv in 0..3 {
                        v += h_pi[[qu, qv]] * (djt[qu][a] * jt[qv][b] + jt[qu][a] * djt[qv][b]);
                    }
                }
                acc[[slices.time.start + a, slices.time.start + b]] += v;
            }
        }
        for a in 0..p_marginal {
            for b in 0..p_marginal {
                let mut v = 0.0;
                for qu in 0..3 {
                    for qv in 0..3 {
                        v += h_pi[[qu, qv]] * (djm[qu][a] * jm[qv][b] + jm[qu][a] * djm[qv][b]);
                    }
                }
                acc[[slices.marginal.start + a, slices.marginal.start + b]] += v;
            }
        }
        for a in 0..p_time {
            for b in 0..p_marginal {
                let mut v = 0.0;
                for qu in 0..3 {
                    for qv in 0..3 {
                        v += h_pi[[qu, qv]] * (djt[qu][a] * jm[qv][b] + jt[qu][a] * djm[qv][b]);
                    }
                }
                acc[[slices.time.start + a, slices.marginal.start + b]] += v;
                acc[[slices.marginal.start + b, slices.time.start + a]] += v;
            }
        }
        let gc = self
            .logslope_design
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("logslope_design try_row_chunk: {e}"))?;
        let gr = gc.row(0);
        for a in 0..p_time {
            let mut w = 0.0;
            for qu in 0..3 {
                w += h_pi[[qu, 3]] * djt[qu][a];
            }
            for b in 0..slices.logslope.len() {
                let v = w * gr[b];
                acc[[slices.time.start + a, slices.logslope.start + b]] += v;
                acc[[slices.logslope.start + b, slices.time.start + a]] += v;
            }
        }
        for a in 0..p_marginal {
            let mut w = 0.0;
            for qu in 0..3 {
                w += h_pi[[qu, 3]] * djm[qu][a];
            }
            for b in 0..slices.logslope.len() {
                let v = w * gr[b];
                acc[[slices.marginal.start + a, slices.logslope.start + b]] += v;
                acc[[slices.logslope.start + b, slices.marginal.start + a]] += v;
            }
        }

        for (primary_range, joint_range) in identity_blocks {
            for local in 0..primary_range.len() {
                let primary_idx = primary_range.start + local;
                let joint_idx = joint_range.start + local;
                for a in 0..p_time {
                    let mut value = 0.0;
                    for qu in 0..3 {
                        value += h_pi[[qu, primary_idx]] * djt[qu][a];
                    }
                    acc[[slices.time.start + a, joint_idx]] += value;
                    acc[[joint_idx, slices.time.start + a]] += value;
                }
                for a in 0..p_marginal {
                    let mut value = 0.0;
                    for qu in 0..3 {
                        value += h_pi[[qu, primary_idx]] * djm[qu][a];
                    }
                    acc[[slices.marginal.start + a, joint_idx]] += value;
                    acc[[joint_idx, slices.marginal.start + a]] += value;
                }
            }
        }

        // Term 4: Σ_r f_r dK_r/d
        for a in 0..p_base {
            for b in 0..p_base {
                let dk0 = m3e * dh0 * xe[a] * xe[b];
                let dk1 = m3x * dh1 * xx[a] * xx[b];
                let dkd = m4x * dh1 * dr * xx[a] * xx[b]
                    + m3x * ddr * xx[a] * xx[b]
                    + m3x * dh1 * (xx[a] * xd[b] + xd[a] * xx[b]);
                acc[[slices.time.start + a, slices.time.start + b]] +=
                    f_pi[0] * dk0 + f_pi[1] * dk1 + f_pi[2] * dkd;
            }
        }
        for li in 0..time_tail.len() {
            let ci = time_tail.start + li;
            for a in 0..p_base {
                let dk0 = eg.basis_d2[[0, li]] * dh0 * xe[a];
                let dk1 = xg.basis_d2[[0, li]] * dh1 * xx[a];
                let dkd = xg.basis_d3[[0, li]] * dh1 * dr * xx[a]
                    + xg.basis_d2[[0, li]] * ddr * xx[a]
                    + xg.basis_d2[[0, li]] * dh1 * xd[a];
                let v = f_pi[0] * dk0 + f_pi[1] * dk1 + f_pi[2] * dkd;
                acc[[slices.time.start + a, slices.time.start + ci]] += v;
                acc[[slices.time.start + ci, slices.time.start + a]] += v;
            }
        }
        for a in 0..p_base {
            for b in 0..p_marginal {
                let dk0 = m3e * dh0 * xe[a] * mr[b];
                let dk1 = m3x * dh1 * xx[a] * mr[b];
                let dkd = m4x * dh1 * dr * xx[a] * mr[b]
                    + m3x * ddr * xx[a] * mr[b]
                    + m3x * dh1 * xd[a] * mr[b];
                let v = f_pi[0] * dk0 + f_pi[1] * dk1 + f_pi[2] * dkd;
                acc[[slices.time.start + a, slices.marginal.start + b]] += v;
                acc[[slices.marginal.start + b, slices.time.start + a]] += v;
            }
        }
        for li in 0..time_tail.len() {
            let ci = time_tail.start + li;
            for b in 0..p_marginal {
                let dk0 = eg.basis_d2[[0, li]] * dh0 * mr[b];
                let dk1 = xg.basis_d2[[0, li]] * dh1 * mr[b];
                let dkd =
                    xg.basis_d3[[0, li]] * dh1 * dr * mr[b] + xg.basis_d2[[0, li]] * ddr * mr[b];
                let v = f_pi[0] * dk0 + f_pi[1] * dk1 + f_pi[2] * dkd;
                acc[[slices.time.start + ci, slices.marginal.start + b]] += v;
                acc[[slices.marginal.start + b, slices.time.start + ci]] += v;
            }
        }
        for a in 0..p_marginal {
            for b in 0..p_marginal {
                let dk0 = m3e * dh0 * mr[a] * mr[b];
                let dk1 = m3x * dh1 * mr[a] * mr[b];
                let dkd = m4x * dh1 * dr * mr[a] * mr[b] + m3x * ddr * mr[a] * mr[b];
                acc[[slices.marginal.start + a, slices.marginal.start + b]] +=
                    f_pi[0] * dk0 + f_pi[1] * dk1 + f_pi[2] * dkd;
            }
        }
        Ok(())
    }

    /// Exact directional derivative of the joint Hessian for timewiggle-only
    /// models (no score-warp / link-deviation).  Computes the derivative by
    /// differentiating the J^T H J + f·K pullback through the timewiggle
    /// q-map geometry (equation 47 of the unified pullback framework).
    pub(crate) fn exact_newton_joint_hessian_directional_derivative_timewiggle_inner(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
        cache: Option<&EvalCache>,
    ) -> Result<Array2<f64>, String> {
        let slices = block_slices(self, block_states);
        let p_total = slices.total;
        let time_tail = self.time_wiggle_range();
        let d_time = d_beta_flat.slice(s![slices.time.clone()]);
        let d_marginal = d_beta_flat.slice(s![slices.marginal.clone()]);
        let beta_time = &block_states[0].beta;
        let beta_time_w = beta_time.slice(s![time_tail.clone()]);

        let result = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            self.n,
            |range| -> Result<Array2<f64>, String> {
                let mut acc = Array2::<f64>::zeros((p_total, p_total));
                for row in range {
                    let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
                    let primary_owned;
                    let (f_pi, h_pi) = if let Some(cached) = cache {
                        self.row_primary_gradient_hessian(row, cached)
                    } else {
                        primary_owned =
                            self.compute_row_primary_gradient_hessian_uncached(row, block_states)?;
                        (&primary_owned.1, &primary_owned.2)
                    };
                    // Inline primary direction from already-computed q_geom
                    // (avoids redundant row_dynamic_q_geometry call)
                    let d_logslope = d_beta_flat.slice(s![slices.logslope.clone()]);
                    let u_d = Array1::from_vec(vec![
                        q_geom.dq0_time.dot(&d_time) + q_geom.dq0_marginal.dot(&d_marginal),
                        q_geom.dq1_time.dot(&d_time) + q_geom.dq1_marginal.dot(&d_marginal),
                        q_geom.dqd1_time.dot(&d_time) + q_geom.dqd1_marginal.dot(&d_marginal),
                        self.logslope_design.dot_row_view(row, d_logslope),
                    ]);
                    let t_ud = self.row_primary_third_contracted(row, block_states, u_d.view())?;
                    let h_ud = h_pi.dot(&u_d);

                    // Term 1 + 3: reuse core accumulator with (H·u^d, T[u^d])
                    self.accumulate_dynamic_q_core_hessian(
                        row,
                        &slices,
                        &q_geom,
                        h_ud.view(),
                        t_ud.view(),
                        &mut acc,
                    )?;

                    self.accumulate_timewiggle_directional_row(
                        row,
                        block_states,
                        &slices,
                        &q_geom,
                        f_pi,
                        h_pi.view(),
                        d_time,
                        d_marginal,
                        beta_time,
                        beta_time_w,
                        &[],
                        &mut acc,
                    )?;
                }
                Ok(acc)
            },
            |mut a, b| -> Result<_, String> {
                a += &b;
                Ok(a)
            },
        )?
        .unwrap_or_else(|| Array2::<f64>::zeros((p_total, p_total)));
        Ok(result)
    }

    /// Exact directional derivative of the joint Hessian for simultaneous
    /// timewiggle + flexible score/link warps.
    ///
    /// This extends the timewiggle-only transport by keeping the full flexible
    /// primary Hessian/third contraction live while only differentiating the
    /// q-geometry Jacobian and K tensors for the dynamic q coordinates. The
    /// score/link primary coordinates remain identity-mapped, so their
    /// contribution enters through the shared pullback term plus cross-columns
    /// against the dJ correction.
    pub(crate) fn exact_newton_joint_hessian_directional_derivative_timewiggle_flex(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let slices = block_slices(self, block_states);
        let primary = flex_primary_slices(self);
        let identity_blocks = flex_identity_block_pairs(&primary, &slices);
        let p_total = slices.total;
        let time_tail = self.time_wiggle_range();
        let d_time = d_beta_flat.slice(s![slices.time.clone()]);
        let d_marginal = d_beta_flat.slice(s![slices.marginal.clone()]);
        let beta_time = &block_states[0].beta;
        let beta_time_w = beta_time.slice(s![time_tail.clone()]);

        let result = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            self.n,
            |range| -> Result<Array2<f64>, String> {
                let mut acc = Array2::<f64>::zeros((p_total, p_total));
                for row in range {
                    let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
                    let (_, f_pi, h_pi) = self.compute_row_flex_primary_gradient_hessian_exact(
                        row,
                        block_states,
                        &q_geom,
                        &primary,
                    )?;
                    let u_d = self.row_primary_direction_from_flat_dynamic_with_q_geometry(
                        row,
                        block_states,
                        &slices,
                        &q_geom,
                        d_beta_flat,
                    )?;
                    let t_ud =
                        self.row_flex_primary_third_contracted_exact(row, block_states, &u_d)?;
                    let h_ud = h_pi.dot(&u_d);

                    self.accumulate_dynamic_q_joint_row(
                        row,
                        &slices,
                        &q_geom,
                        h_ud.view(),
                        t_ud.view(),
                        &identity_blocks,
                        &mut Array1::zeros(p_total),
                        &mut acc,
                    )?;

                    self.accumulate_timewiggle_directional_row(
                        row,
                        block_states,
                        &slices,
                        &q_geom,
                        &f_pi,
                        h_pi.view(),
                        d_time,
                        d_marginal,
                        beta_time,
                        beta_time_w,
                        &identity_blocks,
                        &mut acc,
                    )?;
                }
                Ok(acc)
            },
            |mut a, b| -> Result<_, String> {
                a += &b;
                Ok(a)
            },
        )?
        .unwrap_or_else(|| Array2::<f64>::zeros((p_total, p_total)));
        Ok(result)
    }

    pub(crate) fn exact_newton_joint_hessian_directional_derivative_timewiggle_cached(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
        cache: &EvalCache,
    ) -> Result<Array2<f64>, String> {
        self.exact_newton_joint_hessian_directional_derivative_timewiggle_inner(
            block_states,
            d_beta_flat,
            Some(cache),
        )
    }

    pub(crate) fn exact_newton_joint_hessian_directional_derivative_timewiggle(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        self.exact_newton_joint_hessian_directional_derivative_timewiggle_inner(
            block_states,
            d_beta_flat,
            None,
        )
    }
    /// Fully exact second directional derivative D²H[d,e] for timewiggle-only.
    /// Differentiates DH[e] along d analytically using m₂–m₅ scalars.
    ///
    /// D²H[d,e] = J^T Ψ J  +  Σ γ_r K_r
    ///   + Σ bilinear(W_k, left_k, right_k)  for k in {T_e×dJ_d, T_d×dJ_e, H×d²J, H×dJ_d×dJ_e}
    ///   + dK cross-terms: (Hu_d)·dK_e + (Hu_e)·dK_d + f·d²K
    ///
    /// where Ψ = Q[u_d,u_e] + T[du_e/dd], γ = T_d·u_e + H·du_e/dd.
    pub(crate) fn exact_newton_joint_hessiansecond_directional_derivative_timewiggle(
        &self,
        block_states: &[ParameterBlockState],
        d_u: &Array1<f64>,
        d_v: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let slices = block_slices(self, block_states);
        let p_total = slices.total;
        let p_time = slices.time.len();
        let p_marginal = slices.marginal.len();
        let time_tail = self.time_wiggle_range();
        let p_base = time_tail.start;
        let du_t = d_u.slice(s![slices.time.clone()]);
        let du_m = d_u.slice(s![slices.marginal.clone()]);
        let du_g = d_u.slice(s![slices.logslope.clone()]);
        let dv_t = d_v.slice(s![slices.time.clone()]);
        let dv_m = d_v.slice(s![slices.marginal.clone()]);
        let dv_g = d_v.slice(s![slices.logslope.clone()]);
        let beta_time = &block_states[0].beta;
        let beta_tw = beta_time.slice(s![time_tail.clone()]);

        let result = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            self.n,
            |range| -> Result<Array2<f64>, String> {
                let mut acc = Array2::<f64>::zeros((p_total, p_total));
                for row in range {
                    let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
                    let (_, f_pi, h_pi) =
                        self.compute_row_primary_gradient_hessian_uncached(row, block_states)?;

                    // Primary directions
                    let ud = Array1::from_vec(vec![
                        q_geom.dq0_time.dot(&du_t) + q_geom.dq0_marginal.dot(&du_m),
                        q_geom.dq1_time.dot(&du_t) + q_geom.dq1_marginal.dot(&du_m),
                        q_geom.dqd1_time.dot(&du_t) + q_geom.dqd1_marginal.dot(&du_m),
                        self.logslope_design.dot_row_view(row, du_g),
                    ]);
                    let ue = Array1::from_vec(vec![
                        q_geom.dq0_time.dot(&dv_t) + q_geom.dq0_marginal.dot(&dv_m),
                        q_geom.dq1_time.dot(&dv_t) + q_geom.dq1_marginal.dot(&dv_m),
                        q_geom.dqd1_time.dot(&dv_t) + q_geom.dqd1_marginal.dot(&dv_m),
                        self.logslope_design.dot_row_view(row, dv_g),
                    ]);

                    let t_d = self.row_primary_third_contracted(row, block_states, ud.view())?;
                    let t_e = self.row_primary_third_contracted(row, block_states, ue.view())?;
                    let q_de = self.row_primary_fourth_contracted(
                        row,
                        block_states,
                        ud.view(),
                        ue.view(),
                    )?;
                    let h_ud = h_pi.dot(&ud);
                    let h_ue = h_pi.dot(&ue);

                    // ── Timewiggle geometry ─────────────────────────────
                    let ec = self
                        .design_entry
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("design_entry try_row_chunk: {e}"))?;
                    let xc = self
                        .design_exit
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("design_exit try_row_chunk: {e}"))?;
                    let dc = self
                        .design_derivative_exit
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("design_derivative_exit try_row_chunk: {e}"))?;
                    let xe = ec.row(0).slice(s![..p_base]).to_owned();
                    let xx = xc.row(0).slice(s![..p_base]).to_owned();
                    let xd = dc.row(0).slice(s![..p_base]).to_owned();
                    let mc = self
                        .marginal_design
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("marginal_design try_row_chunk: {e}"))?;
                    let mr = mc.row(0).to_owned();

                    let bm = block_states[1].eta[row];
                    let h0 = xe.dot(&beta_time.slice(s![..p_base])) + self.offset_entry[row] + bm;
                    let h1 = xx.dot(&beta_time.slice(s![..p_base])) + self.offset_exit[row] + bm;
                    let dr =
                        xd.dot(&beta_time.slice(s![..p_base])) + self.derivative_offset_exit[row];

                    let eg = self
                        .time_wiggle_geometry(Array1::from_vec(vec![h0]).view(), beta_tw)?
                        .ok_or_else(|| "timewiggle geometry missing".to_string())?;
                    let xg = self
                        .time_wiggle_geometry(Array1::from_vec(vec![h1]).view(), beta_tw)?
                        .ok_or_else(|| "timewiggle geometry missing".to_string())?;

                    let m2_en = eg.d2q_dq02[0];
                    let m3_en = eg.d3q_dq03[0];
                    let m4_en = eg.d4q_dq04[0];
                    let m2_ex = xg.d2q_dq02[0];
                    let m3_ex = xg.d3q_dq03[0];
                    let m4_ex = xg.d4q_dq04[0];
                    let m5_ex = xg.d5q_dq05[0];

                    // Direction scalars (h linear in β ⇒ d²h/ded = 0)
                    let dh0d = xe.dot(&du_t.slice(s![..p_base])) + mr.dot(&du_m);
                    let dh1d = xx.dot(&du_t.slice(s![..p_base])) + mr.dot(&du_m);
                    let ddrd = xd.dot(&du_t.slice(s![..p_base]));
                    let dh0e = xe.dot(&dv_t.slice(s![..p_base])) + mr.dot(&dv_m);
                    let dh1e = xx.dot(&dv_t.slice(s![..p_base])) + mr.dot(&dv_m);
                    let ddre = xd.dot(&dv_t.slice(s![..p_base]));

                    // du_e/dd = (dJ/dd)·e_v — primary direction of e perturbed by d
                    // dJ[q0,time_a]/dd = m2_en*dh0d*xe[a] for base, basis_d1*dh0d for wiggle
                    let due_d = {
                        let mut v = [0.0f64; 4];
                        for a in 0..p_base {
                            v[0] += m2_en * dh0d * xe[a] * dv_t[a];
                            v[1] += m2_ex * dh1d * xx[a] * dv_t[a];
                            v[2] += (m3_ex * dh1d * dr * xx[a]
                                + m2_ex * ddrd * xx[a]
                                + m2_ex * dh1d * xd[a])
                                * dv_t[a];
                        }
                        for li in 0..time_tail.len() {
                            let ci = time_tail.start + li;
                            v[0] += eg.basis_d1[[0, li]] * dh0d * dv_t[ci];
                            v[1] += xg.basis_d1[[0, li]] * dh1d * dv_t[ci];
                            v[2] += (xg.basis_d2[[0, li]] * dh1d * dr
                                + xg.basis_d1[[0, li]] * ddrd)
                                * dv_t[ci];
                        }
                        for a in 0..p_marginal {
                            v[0] += m2_en * dh0d * mr[a] * dv_m[a];
                            v[1] += m2_ex * dh1d * mr[a] * dv_m[a];
                            v[2] += (m3_ex * dh1d * dr * mr[a] + m2_ex * ddrd * mr[a]) * dv_m[a];
                        }
                        // v[3] = 0 (logslope J is constant)
                        Array1::from_vec(v.to_vec())
                    };

                    // Ψ = Q[ud,ue] + T[due_d]
                    let t_due =
                        self.row_primary_third_contracted(row, block_states, due_d.view())?;
                    let psi = &q_de + &t_due;

                    // γ = T_d·ue + H·due_d
                    let gamma = &t_d.dot(&ue) + &h_pi.dot(&due_d);

                    // ── Term A: J^T Ψ J + γ·K ─────────────────────────
                    self.accumulate_dynamic_q_core_hessian(
                        row,
                        &slices,
                        &q_geom,
                        gamma.view(),
                        psi.view(),
                        &mut acc,
                    )?;

                    let jt = [&q_geom.dq0_time, &q_geom.dq1_time, &q_geom.dqd1_time];
                    let jm = [
                        &q_geom.dq0_marginal,
                        &q_geom.dq1_marginal,
                        &q_geom.dqd1_marginal,
                    ];
                    let gc = self
                        .logslope_design
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("logslope_design try_row_chunk: {e}"))?;
                    let gr = gc.row(0);

                    // ── Helper: accumulate a symmetrized bilinear term ──
                    // Adds Σ W[qu,qv] * (left[qu,a]*right[qv,b] + right[qu,a]*left[qv,b])
                    // for all block pairs into acc.
                    macro_rules! accum_bilinear {
                        ($w:expr, $lt:expr, $lm:expr, $rt:expr, $rm:expr) => {
                            for a in 0..p_time {
                                for b in 0..p_time {
                                    let mut v = 0.0;
                                    for qu in 0..3 {
                                        for qv in 0..3 {
                                            v += $w[[qu, qv]]
                                                * ($lt[qu][a] * $rt[qv][b]
                                                    + $rt[qu][a] * $lt[qv][b]);
                                        }
                                    }
                                    acc[[slices.time.start + a, slices.time.start + b]] += v;
                                }
                            }
                            for a in 0..p_marginal {
                                for b in 0..p_marginal {
                                    let mut v = 0.0;
                                    for qu in 0..3 {
                                        for qv in 0..3 {
                                            v += $w[[qu, qv]]
                                                * ($lm[qu][a] * $rm[qv][b]
                                                    + $rm[qu][a] * $lm[qv][b]);
                                        }
                                    }
                                    acc[[slices.marginal.start + a, slices.marginal.start + b]] +=
                                        v;
                                }
                            }
                            for a in 0..p_time {
                                for b in 0..p_marginal {
                                    let mut v = 0.0;
                                    for qu in 0..3 {
                                        for qv in 0..3 {
                                            v += $w[[qu, qv]]
                                                * ($lt[qu][a] * $rm[qv][b]
                                                    + $rt[qu][a] * $lm[qv][b]);
                                        }
                                    }
                                    acc[[slices.time.start + a, slices.marginal.start + b]] += v;
                                    acc[[slices.marginal.start + b, slices.time.start + a]] += v;
                                }
                            }
                            for a in 0..p_time {
                                let mut w2 = 0.0;
                                for qu in 0..3 {
                                    w2 += $w[[qu, 3]] * $lt[qu][a];
                                }
                                for b in 0..slices.logslope.len() {
                                    let v = w2 * gr[b];
                                    acc[[slices.time.start + a, slices.logslope.start + b]] += v;
                                    acc[[slices.logslope.start + b, slices.time.start + a]] += v;
                                }
                            }
                            for a in 0..p_marginal {
                                let mut w2 = 0.0;
                                for qu in 0..3 {
                                    w2 += $w[[qu, 3]] * $lm[qu][a];
                                }
                                for b in 0..slices.logslope.len() {
                                    let v = w2 * gr[b];
                                    acc[[slices.marginal.start + a, slices.logslope.start + b]] +=
                                        v;
                                    acc[[slices.logslope.start + b, slices.marginal.start + a]] +=
                                        v;
                                }
                            }
                        };
                    }

                    // ── Build dJ arrays for both directions ────────────
                    // (same code as first directional, for d and e)
                    type DjArrays = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);
                    let build_dj = |dh0: f64, dh1: f64, ddr_val: f64| -> DjArrays {
                        let mut j0t = vec![0.0f64; p_time];
                        let mut j1t = vec![0.0f64; p_time];
                        let mut jdt = vec![0.0f64; p_time];
                        for a in 0..p_base {
                            j0t[a] = m2_en * dh0 * xe[a];
                            j1t[a] = m2_ex * dh1 * xx[a];
                            jdt[a] = m3_ex * dh1 * dr * xx[a]
                                + m2_ex * ddr_val * xx[a]
                                + m2_ex * dh1 * xd[a];
                        }
                        for li in 0..time_tail.len() {
                            let ci = time_tail.start + li;
                            j0t[ci] = eg.basis_d1[[0, li]] * dh0;
                            j1t[ci] = xg.basis_d1[[0, li]] * dh1;
                            jdt[ci] =
                                xg.basis_d2[[0, li]] * dh1 * dr + xg.basis_d1[[0, li]] * ddr_val;
                        }
                        let mut j0m = vec![0.0f64; p_marginal];
                        let mut j1m = vec![0.0f64; p_marginal];
                        let mut jdm = vec![0.0f64; p_marginal];
                        for a in 0..p_marginal {
                            j0m[a] = m2_en * dh0 * mr[a];
                            j1m[a] = m2_ex * dh1 * mr[a];
                            jdm[a] = m3_ex * dh1 * dr * mr[a] + m2_ex * ddr_val * mr[a];
                        }
                        (j0t, j1t, jdt, j0m, j1m, jdm)
                    };

                    let (djd0t, djd1t, djddt, djd0m, djd1m, djddm) = build_dj(dh0d, dh1d, ddrd);
                    let djd_t = [&djd0t[..], &djd1t[..], &djddt[..]];
                    let djd_m = [&djd0m[..], &djd1m[..], &djddm[..]];

                    let (dje0t, dje1t, djedt, dje0m, dje1m, djedm) = build_dj(dh0e, dh1e, ddre);
                    let dje_t = [&dje0t[..], &dje1t[..], &djedt[..]];
                    let dje_m = [&dje0m[..], &dje1m[..], &djedm[..]];

                    // ── d²J/ded (cross derivative, h linear ⇒ d²h=0) ──
                    let mut d2j0t = vec![0.0f64; p_time];
                    let mut d2j1t = vec![0.0f64; p_time];
                    let mut d2jdt = vec![0.0f64; p_time];
                    for a in 0..p_base {
                        d2j0t[a] = m3_en * dh0d * dh0e * xe[a];
                        d2j1t[a] = m3_ex * dh1d * dh1e * xx[a];
                        d2jdt[a] = m4_ex * dh1d * dh1e * dr * xx[a]
                            + m3_ex * (dh1d * ddre + dh1e * ddrd) * xx[a]
                            + m3_ex * dh1d * dh1e * xd[a];
                    }
                    for li in 0..time_tail.len() {
                        let ci = time_tail.start + li;
                        d2j0t[ci] = eg.basis_d2[[0, li]] * dh0d * dh0e;
                        d2j1t[ci] = xg.basis_d2[[0, li]] * dh1d * dh1e;
                        d2jdt[ci] = xg.basis_d3[[0, li]] * dh1d * dh1e * dr
                            + xg.basis_d2[[0, li]] * (dh1d * ddre + dh1e * ddrd);
                    }
                    let d2j_t = [&d2j0t[..], &d2j1t[..], &d2jdt[..]];
                    let mut d2j0m = vec![0.0f64; p_marginal];
                    let mut d2j1m = vec![0.0f64; p_marginal];
                    let mut d2jdm = vec![0.0f64; p_marginal];
                    for a in 0..p_marginal {
                        d2j0m[a] = m3_en * dh0d * dh0e * mr[a];
                        d2j1m[a] = m3_ex * dh1d * dh1e * mr[a];
                        d2jdm[a] = m4_ex * dh1d * dh1e * dr * mr[a]
                            + m3_ex * (dh1d * ddre + dh1e * ddrd) * mr[a];
                    }
                    let d2j_m = [&d2j0m[..], &d2j1m[..], &d2jdm[..]];

                    let jt_s: [&[f64]; 3] = [
                        jt[0].as_slice().unwrap(),
                        jt[1].as_slice().unwrap(),
                        jt[2].as_slice().unwrap(),
                    ];
                    let jm_s: [&[f64]; 3] = [
                        jm[0].as_slice().unwrap(),
                        jm[1].as_slice().unwrap(),
                        jm[2].as_slice().unwrap(),
                    ];

                    // ── Term B: bilinear cross-terms ────────────────────
                    // (dJ_d)^T T_e J + J^T T_e (dJ_d) — differentiated from Term 1
                    accum_bilinear!(t_e, djd_t, djd_m, jt_s, jm_s);
                    // (dJ_e)^T T_d J + J^T T_d (dJ_e) — symmetry partner
                    accum_bilinear!(t_d, dje_t, dje_m, jt_s, jm_s);
                    // (d²J)^T H J + J^T H (d²J) — from Term 2
                    accum_bilinear!(h_pi, d2j_t, d2j_m, jt_s, jm_s);
                    // (dJ_d)^T H (dJ_e) + (dJ_e)^T H (dJ_d) — from Term 2
                    accum_bilinear!(h_pi, djd_t, djd_m, dje_t, dje_m);

                    // ── Term C: dK cross-terms ──────────────────────────
                    // (H·ud)_r dK_r/de + (H·ue)_r dK_r/dd + f_r d²K_r/ded
                    //
                    // dK[q,a,b]/dd = d(K[q,a,b])/dd where K = m_{k}*product-of-design-rows
                    // d²K[q,a,b]/ded = m_{k+2}*dh_d*dh_e*(...) since d²h/ded=0
                    //
                    // For q0 base×base: K = m2_en*xe[a]*xe[b]
                    //   dK/dd = m3_en*dh0d*xe[a]*xe[b]
                    //   d²K/ded = m4_en*dh0d*dh0e*xe[a]*xe[b]
                    // For q1 base×base: K = m2_ex*xx[a]*xx[b]
                    //   dK/dd = m3_ex*dh1d*xx[a]*xx[b]
                    //   d²K/ded = m4_ex*dh1d*dh1e*xx[a]*xx[b]
                    // For qd1 base×base: K = m3_ex*dr*xx[a]*xx[b] + m2_ex*(xx[a]*xd[b]+xd[a]*xx[b])
                    //   dK/dd = m4_ex*dh1d*dr*xx[a]*xx[b] + m3_ex*ddrd*xx[a]*xx[b]
                    //         + m3_ex*dh1d*(xx[a]*xd[b]+xd[a]*xx[b])
                    //   d²K/ded = m5_ex*dh1d*dh1e*dr*xx[a]*xx[b]
                    //           + m4_ex*(dh1d*ddre+dh1e*ddrd)*xx[a]*xx[b]
                    //           + m4_ex*dh1d*dh1e*(xx[a]*xd[b]+xd[a]*xx[b])

                    // base×base time×time
                    for a in 0..p_base {
                        for b in 0..p_base {
                            let dke_0 = m3_en * dh0e * xe[a] * xe[b];
                            let dke_1 = m3_ex * dh1e * xx[a] * xx[b];
                            let dke_d = m4_ex * dh1e * dr * xx[a] * xx[b]
                                + m3_ex * ddre * xx[a] * xx[b]
                                + m3_ex * dh1e * (xx[a] * xd[b] + xd[a] * xx[b]);
                            let dkd_0 = m3_en * dh0d * xe[a] * xe[b];
                            let dkd_1 = m3_ex * dh1d * xx[a] * xx[b];
                            let dkd_d = m4_ex * dh1d * dr * xx[a] * xx[b]
                                + m3_ex * ddrd * xx[a] * xx[b]
                                + m3_ex * dh1d * (xx[a] * xd[b] + xd[a] * xx[b]);
                            let d2k_0 = m4_en * dh0d * dh0e * xe[a] * xe[b];
                            let d2k_1 = m4_ex * dh1d * dh1e * xx[a] * xx[b];
                            let d2k_d = m5_ex * dh1d * dh1e * dr * xx[a] * xx[b]
                                + m4_ex * (dh1d * ddre + dh1e * ddrd) * xx[a] * xx[b]
                                + m4_ex * dh1d * dh1e * (xx[a] * xd[b] + xd[a] * xx[b]);
                            acc[[slices.time.start + a, slices.time.start + b]] += h_ud[0] * dke_0
                                + h_ud[1] * dke_1
                                + h_ud[2] * dke_d
                                + h_ue[0] * dkd_0
                                + h_ue[1] * dkd_1
                                + h_ue[2] * dkd_d
                                + f_pi[0] * d2k_0
                                + f_pi[1] * d2k_1
                                + f_pi[2] * d2k_d;
                        }
                    }

                    // base×wiggle time×time
                    for li in 0..time_tail.len() {
                        let ci = time_tail.start + li;
                        for a in 0..p_base {
                            // K[q0] for base×wiggle: d2q0/dβ_base[a] dβ_wiggle[li]
                            //   = basis_d1[li]*xe[a] at entry  (m2 * x * basis is wrong; correct is basis_d1*x)
                            // Actually from q_geom: d2q0_time_time[[a, ci]] = basis_d1_entry[li]*xe[a]
                            // dK/dd = basis_d2[li]*dh0d*xe[a]
                            // d²K/ded = basis_d3[li]*dh0d*dh0e*xe[a]
                            let dke_0 = eg.basis_d2[[0, li]] * dh0e * xe[a];
                            let dke_1 = xg.basis_d2[[0, li]] * dh1e * xx[a];
                            let dke_d = xg.basis_d3[[0, li]] * dh1e * dr * xx[a]
                                + xg.basis_d2[[0, li]] * ddre * xx[a]
                                + xg.basis_d2[[0, li]] * dh1e * xd[a];
                            let dkd_0 = eg.basis_d2[[0, li]] * dh0d * xe[a];
                            let dkd_1 = xg.basis_d2[[0, li]] * dh1d * xx[a];
                            let dkd_d = xg.basis_d3[[0, li]] * dh1d * dr * xx[a]
                                + xg.basis_d2[[0, li]] * ddrd * xx[a]
                                + xg.basis_d2[[0, li]] * dh1d * xd[a];
                            let d2k_0 = eg.basis_d3[[0, li]] * dh0d * dh0e * xe[a];
                            let d2k_1 = xg.basis_d3[[0, li]] * dh1d * dh1e * xx[a];
                            let d2k_d = xg.basis_d4[[0, li]] * dh1d * dh1e * dr * xx[a]
                                + xg.basis_d3[[0, li]] * (dh1d * ddre + dh1e * ddrd) * xx[a]
                                + xg.basis_d3[[0, li]] * dh1d * dh1e * xd[a];
                            let v = h_ud[0] * dke_0
                                + h_ud[1] * dke_1
                                + h_ud[2] * dke_d
                                + h_ue[0] * dkd_0
                                + h_ue[1] * dkd_1
                                + h_ue[2] * dkd_d
                                + f_pi[0] * d2k_0
                                + f_pi[1] * d2k_1
                                + f_pi[2] * d2k_d;
                            acc[[slices.time.start + a, slices.time.start + ci]] += v;
                            acc[[slices.time.start + ci, slices.time.start + a]] += v;
                        }
                    }

                    // base×marginal time×marginal
                    for a in 0..p_base {
                        for b in 0..p_marginal {
                            let dke_0 = m3_en * dh0e * xe[a] * mr[b];
                            let dke_1 = m3_ex * dh1e * xx[a] * mr[b];
                            let dke_d = m4_ex * dh1e * dr * xx[a] * mr[b]
                                + m3_ex * ddre * xx[a] * mr[b]
                                + m3_ex * dh1e * xd[a] * mr[b];
                            let dkd_0 = m3_en * dh0d * xe[a] * mr[b];
                            let dkd_1 = m3_ex * dh1d * xx[a] * mr[b];
                            let dkd_d = m4_ex * dh1d * dr * xx[a] * mr[b]
                                + m3_ex * ddrd * xx[a] * mr[b]
                                + m3_ex * dh1d * xd[a] * mr[b];
                            let d2k_0 = m4_en * dh0d * dh0e * xe[a] * mr[b];
                            let d2k_1 = m4_ex * dh1d * dh1e * xx[a] * mr[b];
                            let d2k_d = m5_ex * dh1d * dh1e * dr * xx[a] * mr[b]
                                + m4_ex * (dh1d * ddre + dh1e * ddrd) * xx[a] * mr[b]
                                + m4_ex * dh1d * dh1e * xd[a] * mr[b];
                            let v = h_ud[0] * dke_0
                                + h_ud[1] * dke_1
                                + h_ud[2] * dke_d
                                + h_ue[0] * dkd_0
                                + h_ue[1] * dkd_1
                                + h_ue[2] * dkd_d
                                + f_pi[0] * d2k_0
                                + f_pi[1] * d2k_1
                                + f_pi[2] * d2k_d;
                            acc[[slices.time.start + a, slices.marginal.start + b]] += v;
                            acc[[slices.marginal.start + b, slices.time.start + a]] += v;
                        }
                    }

                    // wiggle×marginal
                    for li in 0..time_tail.len() {
                        let ci = time_tail.start + li;
                        for b in 0..p_marginal {
                            let dke_0 = eg.basis_d2[[0, li]] * dh0e * mr[b];
                            let dke_1 = xg.basis_d2[[0, li]] * dh1e * mr[b];
                            let dke_d = xg.basis_d3[[0, li]] * dh1e * dr * mr[b]
                                + xg.basis_d2[[0, li]] * ddre * mr[b];
                            let dkd_0 = eg.basis_d2[[0, li]] * dh0d * mr[b];
                            let dkd_1 = xg.basis_d2[[0, li]] * dh1d * mr[b];
                            let dkd_d = xg.basis_d3[[0, li]] * dh1d * dr * mr[b]
                                + xg.basis_d2[[0, li]] * ddrd * mr[b];
                            let d2k_0 = eg.basis_d3[[0, li]] * dh0d * dh0e * mr[b];
                            let d2k_1 = xg.basis_d3[[0, li]] * dh1d * dh1e * mr[b];
                            let d2k_d = xg.basis_d4[[0, li]] * dh1d * dh1e * dr * mr[b]
                                + xg.basis_d3[[0, li]] * (dh1d * ddre + dh1e * ddrd) * mr[b];
                            let v = h_ud[0] * dke_0
                                + h_ud[1] * dke_1
                                + h_ud[2] * dke_d
                                + h_ue[0] * dkd_0
                                + h_ue[1] * dkd_1
                                + h_ue[2] * dkd_d
                                + f_pi[0] * d2k_0
                                + f_pi[1] * d2k_1
                                + f_pi[2] * d2k_d;
                            acc[[slices.time.start + ci, slices.marginal.start + b]] += v;
                            acc[[slices.marginal.start + b, slices.time.start + ci]] += v;
                        }
                    }

                    // marginal×marginal
                    for a in 0..p_marginal {
                        for b in 0..p_marginal {
                            let dke_0 = m3_en * dh0e * mr[a] * mr[b];
                            let dke_1 = m3_ex * dh1e * mr[a] * mr[b];
                            let dke_d =
                                m4_ex * dh1e * dr * mr[a] * mr[b] + m3_ex * ddre * mr[a] * mr[b];
                            let dkd_0 = m3_en * dh0d * mr[a] * mr[b];
                            let dkd_1 = m3_ex * dh1d * mr[a] * mr[b];
                            let dkd_d =
                                m4_ex * dh1d * dr * mr[a] * mr[b] + m3_ex * ddrd * mr[a] * mr[b];
                            let d2k_0 = m4_en * dh0d * dh0e * mr[a] * mr[b];
                            let d2k_1 = m4_ex * dh1d * dh1e * mr[a] * mr[b];
                            let d2k_d = m5_ex * dh1d * dh1e * dr * mr[a] * mr[b]
                                + m4_ex * (dh1d * ddre + dh1e * ddrd) * mr[a] * mr[b];
                            acc[[slices.marginal.start + a, slices.marginal.start + b]] += h_ud[0]
                                * dke_0
                                + h_ud[1] * dke_1
                                + h_ud[2] * dke_d
                                + h_ue[0] * dkd_0
                                + h_ue[1] * dkd_1
                                + h_ue[2] * dkd_d
                                + f_pi[0] * d2k_0
                                + f_pi[1] * d2k_1
                                + f_pi[2] * d2k_d;
                        }
                    }
                }
                Ok(acc)
            },
            |mut a, b| -> Result<_, String> {
                a += &b;
                Ok(a)
            },
        )?
        .unwrap_or_else(|| Array2::<f64>::zeros((p_total, p_total)));
        Ok(result)
    }

    /// Exact first directional derivative for flex without timewiggle.
    /// J is constant (no wiggle), so DH[d] = J^T T[u^d] J + Σ (Hu^d)_r K_r.
    /// Scatter one row's directional-derivative primary quantities
    /// (`primary_gradient` = directional change of the primary gradient, i.e.
    /// `H_primary · u`; `primary_hessian` = directional change of the primary
    /// Hessian) through the q-geometry / identity-block pullback into the joint
    /// `acc`. This is the per-row inner body shared by both the
    /// first-directional (`t_ud`, `h_ud`) and second-directional (`q_de`,
    /// `gamma`) flex-no-wiggle paths, and by the batched all-axes Jeffreys
    /// override — so the three callers cannot drift.
    pub(crate) fn accumulate_directional_joint_hessian_row(
        &self,
        row: usize,
        slices: &BlockSlices,
        q_geom: &SurvivalMarginalSlopeDynamicRow,
        identity_blocks: &[(std::ops::Range<usize>, std::ops::Range<usize>)],
        primary_gradient: ndarray::ArrayView1<'_, f64>,
        primary_hessian: ndarray::ArrayView2<'_, f64>,
        acc: &mut Array2<f64>,
    ) -> Result<(), String> {
        // Core q-geometry pullback (Hessian only)
        self.accumulate_dynamic_q_core_hessian(
            row,
            slices,
            q_geom,
            primary_gradient,
            primary_hessian,
            acc,
        )?;
        // Identity block Hessian: cross + diagonal + cross-cross
        for (primary_range, joint_range) in identity_blocks {
            for local in 0..primary_range.len() {
                self.accumulate_identity_primary_cross_hessian(
                    row,
                    slices,
                    q_geom,
                    primary_hessian.slice(s![0..N_PRIMARY, primary_range.start + local]),
                    joint_range,
                    local,
                    acc,
                )?;
            }
            self.add_dense_submatrix(
                acc,
                joint_range,
                joint_range,
                primary_hessian.slice(s![primary_range.clone(), primary_range.clone()]),
            );
        }
        for li in 0..identity_blocks.len() {
            for ri in li + 1..identity_blocks.len() {
                let (lp, lj) = &identity_blocks[li];
                let (rp, rj) = &identity_blocks[ri];
                self.add_dense_symmetric_cross_submatrix(
                    acc,
                    lj,
                    rj,
                    primary_hessian.slice(s![lp.clone(), rp.clone()]),
                );
            }
        }
        Ok(())
    }

    pub(crate) fn exact_newton_joint_hessian_directional_derivative_flex_no_wiggle(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let slices = block_slices(self, block_states);
        let primary = flex_primary_slices(self);
        let p_total = slices.total;
        let identity_blocks = flex_identity_block_pairs(&primary, &slices);
        let result = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            self.n,
            |range| -> Result<Array2<f64>, String> {
                let mut acc = Array2::<f64>::zeros((p_total, p_total));
                for row in range {
                    let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
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
                    let t_ud =
                        self.row_flex_primary_third_contracted_exact(row, block_states, &u_d)?;
                    let h_ud = h_pi.dot(&u_d);
                    self.accumulate_directional_joint_hessian_row(
                        row,
                        &slices,
                        &q_geom,
                        &identity_blocks,
                        h_ud.view(),
                        t_ud.view(),
                        &mut acc,
                    )?;
                }
                Ok(acc)
            },
            |mut a, b| -> Result<_, String> {
                a += &b;
                Ok(a)
            },
        )?
        .unwrap_or_else(|| Array2::<f64>::zeros((p_total, p_total)));
        Ok(result)
    }

    /// Build-once all-axes variant of
    /// [`Self::exact_newton_joint_hessian_directional_derivative_flex_no_wiggle`].
    ///
    /// The Jeffreys all-axes sweep needs the directional joint-Hessian for each
    /// of the `p` coordinate axes. Calling the single-direction routine `p`
    /// times rebuilds, per row and per axis, the direction-independent flex
    /// geometry (`q`-geometry, primary Hessian, intercept solves, cached
    /// partitions, exact base timepoints) — a `p`-fold redundant cost that is
    /// the #979 flex marginal-slope hot path. This variant builds that geometry
    /// once per row (`FlexThirdRowBase` + the primary Hessian) and contracts it
    /// against each axis, so only the per-axis directional pieces are repeated.
    /// Each output matrix routes through the same per-row assemblers as the
    /// corresponding single-axis call (`row_flex_third_contract_from_base` +
    /// `accumulate_directional_joint_hessian_row`), so it equals that call up to
    /// the cross-row rayon reduction order.
    pub(crate) fn exact_newton_joint_hessian_directional_derivative_flex_no_wiggle_all_axes(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Vec<Array2<f64>>, String> {
        let slices = block_slices(self, block_states);
        let primary = flex_primary_slices(self);
        let p_total = slices.total;
        let identity_blocks = flex_identity_block_pairs(&primary, &slices);
        let zeros = || vec![Array2::<f64>::zeros((p_total, p_total)); p_total];
        let result = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            self.n,
            |range| -> Result<Vec<Array2<f64>>, String> {
                let mut acc = zeros();
                for row in range {
                    // Direction-independent per-row geometry, built once.
                    let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
                    let h_pi = self
                        .compute_row_flex_primary_gradient_hessian_exact(
                            row,
                            block_states,
                            &q_geom,
                            &primary,
                        )?
                        .2;
                    let base =
                        self.build_row_flex_third_base_with_states(row, block_states, &primary)?;
                    // Per-axis: only the primary-direction pullback, the third
                    // contraction against it, and the row accumulation are repeated.
                    for axis_idx in 0..p_total {
                        let mut axis = Array1::<f64>::zeros(p_total);
                        axis[axis_idx] = 1.0;
                        let u_d = self.row_primary_direction_from_flat_dynamic_with_q_geometry(
                            row,
                            block_states,
                            &slices,
                            &q_geom,
                            &axis,
                        )?;
                        let t_ud = self.row_flex_third_contract_from_base(&base, &u_d)?;
                        let h_ud = h_pi.dot(&u_d);
                        self.accumulate_directional_joint_hessian_row(
                            row,
                            &slices,
                            &q_geom,
                            &identity_blocks,
                            h_ud.view(),
                            t_ud.view(),
                            &mut acc[axis_idx],
                        )?;
                    }
                }
                Ok(acc)
            },
            |mut a, b| -> Result<_, String> {
                for (ai, bi) in a.iter_mut().zip(b.into_iter()) {
                    *ai += &bi;
                }
                Ok(a)
            },
        )?
        .unwrap_or_else(zeros);
        Ok(result)
    }

    /// Exact second directional derivative for flex without timewiggle.
    /// J constant ⇒ D²H[d,e] = J^T Q[u^d,u^e] J + Σ (T_d·u^e)_r K_r.
    pub(crate) fn exact_newton_joint_hessiansecond_directional_derivative_flex_no_wiggle(
        &self,
        block_states: &[ParameterBlockState],
        d_u: &Array1<f64>,
        d_v: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let slices = block_slices(self, block_states);
        let primary = flex_primary_slices(self);
        let p_total = slices.total;
        let identity_blocks = flex_identity_block_pairs(&primary, &slices);
        let result = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            self.n,
            |range| -> Result<Array2<f64>, String> {
                let mut acc = Array2::<f64>::zeros((p_total, p_total));
                for row in range {
                    let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
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
                    let q_de =
                        self.row_flex_primary_fourth_contracted_exact(row, block_states, &ud, &ue)?;
                    let t_d =
                        self.row_flex_primary_third_contracted_exact(row, block_states, &ud)?;
                    let gamma = t_d.dot(&ue);
                    // Hessian-only: accumulate q-core + identity block Hessian
                    self.accumulate_directional_joint_hessian_row(
                        row,
                        &slices,
                        &q_geom,
                        &identity_blocks,
                        gamma.view(),
                        q_de.view(),
                        &mut acc,
                    )?;
                }
                Ok(acc)
            },
            |mut a, b| -> Result<_, String> {
                a += &b;
                Ok(a)
            },
        )?
        .unwrap_or_else(|| Array2::<f64>::zeros((p_total, p_total)));
        Ok(result)
    }

    pub(crate) fn evaluate_blockwise_exact_newton(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<FamilyEvaluation, String> {
        if self.per_z_logslope_active() {
            return self.evaluate_blockwise_exact_newton_per_z(block_states);
        }
        if self.effective_flex_active(block_states)? {
            return self.evaluate_blockwise_exact_newton_flexible(block_states);
        }
        if self.flex_timewiggle_active() {
            return self.evaluate_blockwise_exact_newton_timewiggle(block_states);
        }

        // For all non-flex, non-timewiggle modes: use the dense joint path
        // when p is small enough.  This guarantees block Hessians are
        // principal blocks of the joint Hessian regardless of whether the
        // underlying designs happen to be sparse.
        let slices = block_slices(self, block_states);
        if slices.total < 512 {
            return self.evaluate_blockwise_exact_newton_dense(block_states);
        }

        // Large p (>= 512): joint dense Hessian is too expensive.
        // Fall back to blockwise sparse/mixed assembly for memory efficiency.
        let time_csrs = match (
            self.design_entry.as_sparse(),
            self.design_exit.as_sparse(),
            self.design_derivative_exit.as_sparse(),
        ) {
            (Some(e), Some(x), Some(d)) => Some((
                e.to_csr_arc().expect("entry CSR"),
                x.to_csr_arc().expect("exit CSR"),
                d.to_csr_arc().expect("deriv CSR"),
            )),
            _ => None,
        };
        let marginal_csr = self
            .marginal_design
            .as_sparse()
            .and_then(|s| s.to_csr_arc());
        let logslope_csr = self
            .logslope_design
            .as_sparse()
            .and_then(|s| s.to_csr_arc());

        let time_sparse = time_csrs.is_some();
        let marginal_sparse = marginal_csr.is_some();
        let logslope_sparse = logslope_csr.is_some();

        if time_sparse && marginal_sparse && logslope_sparse {
            self.evaluate_blockwise_exact_newton_sparse(
                block_states,
                &time_csrs.unwrap(),
                &marginal_csr.unwrap(),
                &logslope_csr.unwrap(),
            )
        } else if !time_sparse && !marginal_sparse && !logslope_sparse {
            self.evaluate_blockwise_exact_newton_dense(block_states)
        } else {
            self.evaluate_blockwise_exact_newton_mixed(
                block_states,
                time_csrs.as_ref(),
                marginal_csr.as_ref(),
                logslope_csr.as_ref(),
            )
        }
    }

    pub(crate) fn evaluate_blockwise_exact_newton_per_z(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<FamilyEvaluation, String> {
        if self.effective_flex_active(block_states)? || self.flex_timewiggle_active() {
            return Err(
                "survival marginal-slope per-z logslope surfaces currently require the rigid row kernel"
                    .to_string(),
            );
        }
        let slices = block_slices(self, block_states);
        let p_t = slices.time.len();
        let p_m = slices.marginal.len();
        let p_g = slices.logslope.len();
        let beta_time = &block_states[0].beta;
        let beta_logslope = &block_states[2].beta;
        let probit_scale = self.probit_frailty_scale();
        type PerZBlockAcc = (
            f64,
            Array1<f64>,
            Array1<f64>,
            Array1<f64>,
            Array2<f64>,
            Array2<f64>,
            Array2<f64>,
        );
        let make_per_z_acc = || -> PerZBlockAcc {
            (
                0.0,
                Array1::<f64>::zeros(p_t),
                Array1::<f64>::zeros(p_m),
                Array1::<f64>::zeros(p_g),
                Array2::<f64>::zeros((p_t, p_t)),
                Array2::<f64>::zeros((p_m, p_m)),
                Array2::<f64>::zeros((p_g, p_g)),
            )
        };
        let (ll, grad_t, grad_m, grad_g, hess_t, hess_m, hess_g): PerZBlockAcc =
            gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
                self.n,
                |range| -> Result<_, String> {
                    let mut acc = make_per_z_acc();
                    for row in range {
                    let q0 = self.design_entry.dot_row(row, beta_time)
                        + self.offset_entry[row]
                        + block_states[1].eta[row];
                    let q1 = self.design_exit.dot_row(row, beta_time)
                        + self.offset_exit[row]
                        + block_states[1].eta[row];
                    let qd1 = self.design_derivative_exit.dot_row(row, beta_time)
                        + self.derivative_offset_exit[row];
                    let slopes = self.logslope_surface_values_for_row(row, beta_logslope)?;
                    let z = self.z.row(row).to_vec();
                    let (nll, f_pi, f_pipi) = row_primary_closed_form_vector(
                        q0,
                        q1,
                        qd1,
                        &slopes,
                        &z,
                        &self.score_covariance,
                        self.weights[row],
                        self.event[row],
                        self.derivative_guard,
                        probit_scale,
                    )?;
                    acc.0 -= nll;
                    self.design_entry
                        .axpy_row_into(row, -f_pi[0], &mut acc.1.view_mut())?;
                    self.design_exit
                        .axpy_row_into(row, -f_pi[1], &mut acc.1.view_mut())?;
                    self.design_derivative_exit.axpy_row_into(
                        row,
                        -f_pi[2],
                        &mut acc.1.view_mut(),
                    )?;
                    self.marginal_design.axpy_row_into(
                        row,
                        -(f_pi[0] + f_pi[1]),
                        &mut acc.2.view_mut(),
                    )?;
                    let g_row = self.logslope_surface_row(row)?;
                    for (coord, range) in self.logslope_surface_ranges.iter().enumerate() {
                        let alpha = -f_pi[3 + coord];
                        for col in range.clone() {
                            acc.3[col] += alpha * g_row[col];
                        }
                    }
                    let time_designs = [
                        &self.design_entry,
                        &self.design_exit,
                        &self.design_derivative_exit,
                    ];
                    for a in 0..3 {
                        for b in 0..3 {
                            time_designs[a].row_outer_into(
                                row,
                                time_designs[b],
                                f_pipi[[a, b]],
                                &mut acc.4,
                            )?;
                        }
                    }
                    let alpha_mm =
                        f_pipi[[0, 0]] + f_pipi[[0, 1]] + f_pipi[[1, 0]] + f_pipi[[1, 1]];
                    self.marginal_design
                        .syr_row_into(row, alpha_mm, &mut acc.5)?;
                    for (a, range_a) in self.logslope_surface_ranges.iter().enumerate() {
                        for (b, range_b) in self.logslope_surface_ranges.iter().enumerate() {
                            let alpha = f_pipi[[3 + a, 3 + b]];
                            if alpha == 0.0 {
                                continue;
                            }
                            for ca in range_a.clone() {
                                let va = g_row[ca] * alpha;
                                for cb in range_b.clone() {
                                    acc.6[[ca, cb]] += va * g_row[cb];
                                }
                            }
                        }
                    }
                    Ok(acc)
                },
                |mut a, b| -> Result<_, String> {
                    a.0 += b.0;
                    a.1 += &b.1;
                    a.2 += &b.2;
                    a.3 += &b.3;
                    a.4 += &b.4;
                    a.5 += &b.5;
                    a.6 += &b.6;
                    Ok(a)
                },
            )?
            .unwrap_or_else(make_per_z_acc);
        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets: vec![
                BlockWorkingSet::ExactNewton {
                    gradient: grad_t,
                    hessian: SymmetricMatrix::Dense(hess_t),
                },
                BlockWorkingSet::ExactNewton {
                    gradient: grad_m,
                    hessian: SymmetricMatrix::Dense(hess_m),
                },
                BlockWorkingSet::ExactNewton {
                    gradient: grad_g,
                    hessian: SymmetricMatrix::Dense(hess_g),
                },
            ],
        })
    }

    pub(crate) fn evaluate_exact_newton_joint_dense_per_z(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        let slices = block_slices(self, block_states);
        let total = slices.total;
        let k = self.score_dim();
        let dim = 3 + k;
        let beta_time = &block_states[0].beta;
        let beta_logslope = &block_states[2].beta;
        let probit_scale = self.probit_frailty_scale();
        type PerZJointAcc = (f64, Array1<f64>, Array2<f64>);
        let (ll, grad, hess): PerZJointAcc = (0..self.n)
            .into_par_iter()
            .try_fold(
                || {
                    (
                        0.0,
                        Array1::<f64>::zeros(total),
                        Array2::<f64>::zeros((total, total)),
                    )
                },
                |mut acc, row| -> Result<_, String> {
                    let q0 = self.design_entry.dot_row(row, beta_time)
                        + self.offset_entry[row]
                        + block_states[1].eta[row];
                    let q1 = self.design_exit.dot_row(row, beta_time)
                        + self.offset_exit[row]
                        + block_states[1].eta[row];
                    let qd1 = self.design_derivative_exit.dot_row(row, beta_time)
                        + self.derivative_offset_exit[row];
                    let slopes = self.logslope_surface_values_for_row(row, beta_logslope)?;
                    let z = self.z.row(row).to_vec();
                    let (nll, f_pi, f_pipi) = row_primary_closed_form_vector(
                        q0,
                        q1,
                        qd1,
                        &slopes,
                        &z,
                        &self.score_covariance,
                        self.weights[row],
                        self.event[row],
                        self.derivative_guard,
                        probit_scale,
                    )?;
                    acc.0 -= nll;
                    let mut j = Array2::<f64>::zeros((dim, total));
                    let entry = self.design_entry.try_row_chunk(row..row + 1).map_err(|e| {
                        format!("evaluate_exact_newton_joint_dense_per_z entry row: {e}")
                    })?;
                    let exit = self.design_exit.try_row_chunk(row..row + 1).map_err(|e| {
                        format!("evaluate_exact_newton_joint_dense_per_z exit row: {e}")
                    })?;
                    let deriv = self
                        .design_derivative_exit
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| {
                            format!("evaluate_exact_newton_joint_dense_per_z derivative row: {e}")
                        })?;
                    let marginal =
                        self.marginal_design
                            .try_row_chunk(row..row + 1)
                            .map_err(|e| {
                                format!("evaluate_exact_newton_joint_dense_per_z marginal row: {e}")
                            })?;
                    j.slice_mut(s![0, slices.time.clone()])
                        .assign(&entry.row(0));
                    j.slice_mut(s![1, slices.time.clone()]).assign(&exit.row(0));
                    j.slice_mut(s![2, slices.time.clone()])
                        .assign(&deriv.row(0));
                    j.slice_mut(s![0, slices.marginal.clone()])
                        .assign(&marginal.row(0));
                    j.slice_mut(s![1, slices.marginal.clone()])
                        .assign(&marginal.row(0));
                    let g_row = self.logslope_surface_row(row)?;
                    for (coord, range) in self.logslope_surface_ranges.iter().enumerate() {
                        let global_range = (slices.logslope.start + range.start)
                            ..(slices.logslope.start + range.end);
                        j.slice_mut(s![3 + coord, global_range])
                            .assign(&g_row.slice(s![range.clone()]));
                    }
                    for a in 0..dim {
                        for col in 0..total {
                            acc.1[col] -= f_pi[a] * j[[a, col]];
                        }
                    }
                    for a in 0..dim {
                        for b in 0..dim {
                            let alpha = f_pipi[[a, b]];
                            if alpha == 0.0 {
                                continue;
                            }
                            for ca in 0..total {
                                let va = j[[a, ca]] * alpha;
                                if va == 0.0 {
                                    continue;
                                }
                                for cb in 0..total {
                                    acc.2[[ca, cb]] += va * j[[b, cb]];
                                }
                            }
                        }
                    }
                    Ok(acc)
                },
            )
            .try_reduce(
                || {
                    (
                        0.0,
                        Array1::<f64>::zeros(total),
                        Array2::<f64>::zeros((total, total)),
                    )
                },
                |mut a, b| -> Result<_, String> {
                    a.0 += b.0;
                    a.1 += &b.1;
                    a.2 += &b.2;
                    Ok(a)
                },
            )?;
        Ok((ll, grad, hess))
    }

    /// Blockwise exact-Newton for the flexible (score-warp / link-deviation)
    /// model.
    ///
    /// Accumulates exact per-block coefficient gradients and Hessians directly
    /// from the dynamic-q row jets. This preserves the exact block Newton
    /// update while avoiding dense full-joint assembly when the caller only
    /// needs block-local working sets.
    pub(crate) fn evaluate_blockwise_exact_newton_flexible(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<FamilyEvaluation, String> {
        self.validate_exact_monotonicity(block_states)?;
        let primary = flex_primary_slices(self);
        self.evaluate_blockwise_exact_newton_dynamic_q(block_states, &primary, |row, q_geom| {
            self.compute_row_flex_primary_gradient_hessian_exact(
                row,
                block_states,
                q_geom,
                &primary,
            )
        })
    }

    /// Blockwise exact-Newton for the time-wiggle model.
    ///
    /// Accumulates exact block-local Hessians directly from the 4D primary
    /// row calculus instead of materializing and slicing a dense joint Hessian.
    pub(crate) fn evaluate_blockwise_exact_newton_timewiggle(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<FamilyEvaluation, String> {
        let primary = flex_primary_slices(self);
        self.evaluate_blockwise_exact_newton_dynamic_q(block_states, &primary, |row, _| {
            self.compute_row_primary_gradient_hessian_uncached(row, block_states)
        })
    }

    pub(crate) fn evaluate_blockwise_exact_newton_mixed(
        &self,
        block_states: &[ParameterBlockState],
        time_csrs: Option<&(
            Arc<faer::sparse::SparseRowMat<usize, f64>>,
            Arc<faer::sparse::SparseRowMat<usize, f64>>,
            Arc<faer::sparse::SparseRowMat<usize, f64>>,
        )>,
        marginal_csr: Option<&Arc<faer::sparse::SparseRowMat<usize, f64>>>,
        logslope_csr: Option<&Arc<faer::sparse::SparseRowMat<usize, f64>>>,
    ) -> Result<FamilyEvaluation, String> {
        use gam_linalg::matrix::SparseHessianAccumulator;

        enum BlockwiseHessianAccumulator {
            Dense(Array2<f64>),
            Sparse(SparseHessianAccumulator),
        }

        impl BlockwiseHessianAccumulator {
            fn add_assign(&mut self, other: &Self) {
                match (self, other) {
                    (Self::Dense(lhs), Self::Dense(rhs)) => *lhs += rhs,
                    (Self::Sparse(lhs), Self::Sparse(rhs)) => lhs.add_values(&rhs.values),
                    // Per-block accumulators all share one storage decision
                    // (marginal_csr / logslope_csr) made at the top of
                    // `family_evaluate_blockwise`.
                    // SAFETY: mismatch ⇒ a newly added partial picked the
                    // wrong storage variant — invariant violation. Zeroing the
                    // accumulator would silently corrupt the joint Hessian, so
                    // fail loudly instead of fabricating a result.
                    _ => panic!("blockwise Hessian accumulator kind mismatch"),
                }
            }

            fn into_symmetric(self) -> SymmetricMatrix {
                match self {
                    Self::Dense(mat) => SymmetricMatrix::Dense(mat),
                    Self::Sparse(acc) => SymmetricMatrix::Sparse(acc.into_sparse_col_mat()),
                }
            }
        }

        let slices = block_slices(self, block_states);
        let p_t = slices.time.len();
        let p_m = slices.marginal.len();
        let p_g = slices.logslope.len();

        let time_pattern = time_csrs.map(|(entry, exit, deriv)| {
            SparseHessianAccumulator::from_multi_csr(
                &[entry.as_ref(), exit.as_ref(), deriv.as_ref()],
                p_t,
            )
        });
        let marginal_pattern =
            marginal_csr.map(|csr| SparseHessianAccumulator::from_single_csr(csr.as_ref(), p_m));
        let logslope_pattern =
            logslope_csr.map(|csr| SparseHessianAccumulator::from_single_csr(csr.as_ref(), p_g));

        let e_sparse = time_csrs.map(|(entry, _, _)| {
            let sym = entry.symbolic();
            (sym.row_ptr(), sym.col_idx(), entry.val())
        });
        let x_sparse = time_csrs.map(|(_, exit, _)| {
            let sym = exit.symbolic();
            (sym.row_ptr(), sym.col_idx(), exit.val())
        });
        let d_sparse = time_csrs.map(|(_, _, deriv)| {
            let sym = deriv.symbolic();
            (sym.row_ptr(), sym.col_idx(), deriv.val())
        });
        let m_sparse = marginal_csr.map(|csr| {
            let sym = csr.symbolic();
            (sym.row_ptr(), sym.col_idx(), csr.val())
        });
        let g_sparse = logslope_csr.map(|csr| {
            let sym = csr.symbolic();
            (sym.row_ptr(), sym.col_idx(), csr.val())
        });

        type MixedAcc = (
            f64,
            Array1<f64>,
            Array1<f64>,
            Array1<f64>,
            BlockwiseHessianAccumulator,
            BlockwiseHessianAccumulator,
            BlockwiseHessianAccumulator,
        );

        let make_acc = || -> MixedAcc {
            (
                0.0,
                Array1::zeros(p_t),
                Array1::zeros(p_m),
                Array1::zeros(p_g),
                time_pattern.as_ref().map_or_else(
                    || BlockwiseHessianAccumulator::Dense(Array2::zeros((p_t, p_t))),
                    |pattern| BlockwiseHessianAccumulator::Sparse(pattern.empty_clone()),
                ),
                marginal_pattern.as_ref().map_or_else(
                    || BlockwiseHessianAccumulator::Dense(Array2::zeros((p_m, p_m))),
                    |pattern| BlockwiseHessianAccumulator::Sparse(pattern.empty_clone()),
                ),
                logslope_pattern.as_ref().map_or_else(
                    || BlockwiseHessianAccumulator::Dense(Array2::zeros((p_g, p_g))),
                    |pattern| BlockwiseHessianAccumulator::Sparse(pattern.empty_clone()),
                ),
            )
        };

        let (
            ll,
            grad_time,
            grad_marginal,
            grad_logslope,
            hess_time,
            hess_marginal,
            hess_logslope,
        ): MixedAcc = (0..self.n)
            .into_par_iter()
                .try_fold(make_acc, |mut acc, row| -> Result<_, String> {
                    let (row_nll, f_pi, f_pipi) =
                        self.compute_row_primary_gradient_hessian_uncached(row, block_states)?;
                    acc.0 -= row_nll;

                    match &e_sparse {
                        Some((e_rp, e_ci, e_v)) => {
                            let gt = &mut acc.1;
                            for p in e_rp[row]..e_rp[row + 1] {
                                gt[e_ci[p]] -= f_pi[0] * e_v[p];
                            }
                            let (x_rp, x_ci, x_v) = x_sparse
                                .as_ref()
                                .expect("time sparse metadata should be present for exit design");
                            for p in x_rp[row]..x_rp[row + 1] {
                                gt[x_ci[p]] -= f_pi[1] * x_v[p];
                            }
                            let (d_rp, d_ci, d_v) = d_sparse.as_ref().expect(
                                "time sparse metadata should be present for derivative design",
                            );
                            for p in d_rp[row]..d_rp[row + 1] {
                                gt[d_ci[p]] -= f_pi[2] * d_v[p];
                            }
                        }
                        None => {
                            let mut time = acc.1.view_mut();
                            self.design_entry
                                .axpy_row_into(row, -f_pi[0], &mut time)
                                .expect("time entry axpy dim mismatch");
                            self.design_exit
                                .axpy_row_into(row, -f_pi[1], &mut time)
                                .expect("time exit axpy dim mismatch");
                            self.design_derivative_exit
                                .axpy_row_into(row, -f_pi[2], &mut time)
                                .expect("time deriv axpy dim mismatch");
                        }
                    }

                    match &m_sparse {
                        Some((m_rp, m_ci, m_v)) => {
                            let gm = &mut acc.2;
                            let alpha_m = -(f_pi[0] + f_pi[1]);
                            for p in m_rp[row]..m_rp[row + 1] {
                                gm[m_ci[p]] += alpha_m * m_v[p];
                            }
                        }
                        None => {
                            self.marginal_design
                                .axpy_row_into(row, -(f_pi[0] + f_pi[1]), &mut acc.2.view_mut())
                                .expect(
                                    "survival marginal block axpy should match block dimensions",
                                );
                        }
                    }

                    match &g_sparse {
                        Some((g_rp, g_ci, g_v)) => {
                            let gg = &mut acc.3;
                            for p in g_rp[row]..g_rp[row + 1] {
                                gg[g_ci[p]] -= f_pi[3] * g_v[p];
                            }
                        }
                        None => {
                            self.logslope_design
                                .axpy_row_into(row, -f_pi[3], &mut acc.3.view_mut())
                                .expect(
                                    "survival logslope block axpy should match block dimensions",
                                );
                        }
                    }

                    match &mut acc.4 {
                        BlockwiseHessianAccumulator::Dense(hess_time) => {
                            let designs = [
                                &self.design_entry,
                                &self.design_exit,
                                &self.design_derivative_exit,
                            ];
                            for a in 0..3 {
                                for b in 0..3 {
                                    designs[a]
                                        .row_outer_into(
                                            row,
                                            designs[b],
                                            f_pipi[[a, b]],
                                            &mut *hess_time,
                                        )
                                        .expect("time row_outer_into dim mismatch");
                                }
                            }
                        }
                        BlockwiseHessianAccumulator::Sparse(hess_time) => {
                            let (e_rp, e_ci, e_v) = e_sparse
                                .as_ref()
                                .expect("time sparse metadata should be present for entry design");
                            let (x_rp, x_ci, x_v) = x_sparse
                                .as_ref()
                                .expect("time sparse metadata should be present for exit design");
                            let (d_rp, d_ci, d_v) = d_sparse.as_ref().expect(
                                "time sparse metadata should be present for derivative design",
                            );
                            let row_slices: [(std::ops::Range<usize>, &[usize], &[f64]); 3] = [
                                (e_rp[row]..e_rp[row + 1], e_ci, e_v),
                                (x_rp[row]..x_rp[row + 1], x_ci, x_v),
                                (d_rp[row]..d_rp[row + 1], d_ci, d_v),
                            ];
                            for a in 0..3 {
                                for b in 0..3 {
                                    let alpha = f_pipi[[a, b]];
                                    if alpha == 0.0 {
                                        continue;
                                    }
                                    let (ref ra, cia, va) = row_slices[a];
                                    let (ref rb, cib, vb) = row_slices[b];
                                    for pi in ra.clone() {
                                        let ca = cia[pi];
                                        let xia = va[pi] * alpha;
                                        for pj in rb.clone() {
                                            let cb = cib[pj];
                                            if ca <= cb {
                                                hess_time.add_upper(ca, cb, xia * vb[pj]);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    let alpha_m = f_pipi[[0, 0]] + f_pipi[[0, 1]] + f_pipi[[1, 0]] + f_pipi[[1, 1]];
                    match &mut acc.5 {
                        BlockwiseHessianAccumulator::Dense(hess_marginal) => {
                            self.marginal_design
                                .syr_row_into(row, alpha_m, &mut *hess_marginal)
                                .expect(
                                    "survival marginal block syr should match block dimensions",
                                );
                        }
                        BlockwiseHessianAccumulator::Sparse(hess_marginal) => {
                            if alpha_m != 0.0 {
                                let (m_rp, m_ci, m_v) = m_sparse.as_ref().expect(
                                    "marginal sparse metadata should be present for sparse block",
                                );
                                for pi in m_rp[row]..m_rp[row + 1] {
                                    let ca = m_ci[pi];
                                    let xia = m_v[pi] * alpha_m;
                                    for pj in m_rp[row]..m_rp[row + 1] {
                                        let cb = m_ci[pj];
                                        if ca <= cb {
                                            hess_marginal.add_upper(ca, cb, xia * m_v[pj]);
                                        }
                                    }
                                }
                            }
                        }
                    }

                    let alpha_g = f_pipi[[3, 3]];
                    match &mut acc.6 {
                        BlockwiseHessianAccumulator::Dense(hess_logslope) => {
                            self.logslope_design
                                .syr_row_into(row, alpha_g, &mut *hess_logslope)
                                .expect(
                                    "survival logslope block syr should match block dimensions",
                                );
                        }
                        BlockwiseHessianAccumulator::Sparse(hess_logslope) => {
                            if alpha_g != 0.0 {
                                let (g_rp, g_ci, g_v) = g_sparse.as_ref().expect(
                                    "logslope sparse metadata should be present for sparse block",
                                );
                                for pi in g_rp[row]..g_rp[row + 1] {
                                    let ca = g_ci[pi];
                                    let xia = g_v[pi] * alpha_g;
                                    for pj in g_rp[row]..g_rp[row + 1] {
                                        let cb = g_ci[pj];
                                        if ca <= cb {
                                            hess_logslope.add_upper(ca, cb, xia * g_v[pj]);
                                        }
                                    }
                                }
                            }
                        }
                    }

                    Ok(acc)
                })
                .try_reduce(make_acc, |mut a, b| -> Result<_, String> {
                    a.0 += b.0;
                    a.1 += &b.1;
                    a.2 += &b.2;
                    a.3 += &b.3;
                    a.4.add_assign(&b.4);
                    a.5.add_assign(&b.5);
                    a.6.add_assign(&b.6);
                    Ok(a)
                })?;

        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets: {
                let slices = block_slices(self, block_states);
                let mut sets = vec![
                    BlockWorkingSet::ExactNewton {
                        gradient: grad_time,
                        hessian: hess_time.into_symmetric(),
                    },
                    BlockWorkingSet::ExactNewton {
                        gradient: grad_marginal,
                        hessian: hess_marginal.into_symmetric(),
                    },
                    BlockWorkingSet::ExactNewton {
                        gradient: grad_logslope,
                        hessian: hess_logslope.into_symmetric(),
                    },
                ];
                if let Some(range) = slices.score_warp.as_ref() {
                    sets.push(BlockWorkingSet::ExactNewton {
                        gradient: Array1::zeros(range.len()),
                        hessian: SymmetricMatrix::Dense(Array2::zeros((range.len(), range.len()))),
                    });
                }
                if let Some(range) = slices.link_dev.as_ref() {
                    sets.push(BlockWorkingSet::ExactNewton {
                        gradient: Array1::zeros(range.len()),
                        hessian: SymmetricMatrix::Dense(Array2::zeros((range.len(), range.len()))),
                    });
                }
                sets
            },
        })
    }

    // ── Dense path (original) ────────────────────────────────────────

    pub(crate) fn evaluate_blockwise_exact_newton_dense(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<FamilyEvaluation, String> {
        // Build RowKernel — the single source of truth for all exact-Newton
        // quantities.  The cache evaluates every row kernel once and stores
        // (nll_i, g_i[4], H_i[4×4]).
        let kern = SurvivalMarginalSlopeRowKernel::new(self.clone(), block_states.to_vec());
        let rows = crate::row_kernel::RowSet::All;
        let cache = build_row_kernel_cache(&kern, &rows)?;

        let ll = row_kernel_log_likelihood(&cache, &rows);

        // Joint gradient:  g = -Σ_i Jᵢᵀ gᵢ  (sign: gᵢ are NLL gradients,
        // we negate to get log-likelihood gradient).
        let nll_grad = row_kernel_gradient(&kern, &cache, &rows);
        let joint_gradient = -nll_grad;

        // Block-diagonal Hessians only — the inner solver consumes per-block
        // working sets, so we accumulate the principal time-time, m-m, and
        // g-g blocks directly instead of building the full joint Hessian and
        // slicing.  Cost falls from Θ(n·(p_t+p_m+p_g)²) to
        // Θ(n·(p_t²+p_m²+p_g²)).
        let slices = block_slices(self, block_states);
        let p_t = slices.time.len();
        let p_m = slices.marginal.len();
        let p_g = slices.logslope.len();
        let mut hess_time = Array2::<f64>::zeros((p_t, p_t));
        let mut hess_marginal = Array2::<f64>::zeros((p_m, p_m));
        let mut hess_logslope = Array2::<f64>::zeros((p_g, p_g));
        for row in 0..cache.n {
            let h = &cache.hessians[row];
            let mut h_arr = Array2::<f64>::zeros((4, 4));
            for a in 0..4 {
                for b in 0..4 {
                    h_arr[[a, b]] = h[a][b];
                }
            }
            self.add_pullback_block_diagonals(
                row,
                &h_arr,
                &mut hess_time,
                &mut hess_marginal,
                &mut hess_logslope,
            );
        }

        let mut blockworking_sets = vec![
            BlockWorkingSet::ExactNewton {
                gradient: joint_gradient.slice(s![slices.time.clone()]).to_owned(),
                hessian: SymmetricMatrix::Dense(hess_time),
            },
            BlockWorkingSet::ExactNewton {
                gradient: joint_gradient.slice(s![slices.marginal.clone()]).to_owned(),
                hessian: SymmetricMatrix::Dense(hess_marginal),
            },
            BlockWorkingSet::ExactNewton {
                gradient: joint_gradient.slice(s![slices.logslope.clone()]).to_owned(),
                hessian: SymmetricMatrix::Dense(hess_logslope),
            },
        ];
        if let Some(range) = slices.score_warp.as_ref() {
            // The 4-D row kernel does not span score_warp / link_dev primary
            // dimensions, so these blocks contribute zero gradient/Hessian
            // here — exactly what the joint-then-slice path produced.
            blockworking_sets.push(BlockWorkingSet::ExactNewton {
                gradient: joint_gradient.slice(s![range.clone()]).to_owned(),
                hessian: SymmetricMatrix::Dense(Array2::zeros((range.len(), range.len()))),
            });
        }
        if let Some(range) = slices.link_dev.as_ref() {
            blockworking_sets.push(BlockWorkingSet::ExactNewton {
                gradient: joint_gradient.slice(s![range.clone()]).to_owned(),
                hessian: SymmetricMatrix::Dense(Array2::zeros((range.len(), range.len()))),
            });
        }
        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets,
        })
    }

    // ── Sparse path ──────────────────────────────────────────────────

    pub(crate) fn evaluate_blockwise_exact_newton_sparse(
        &self,
        block_states: &[ParameterBlockState],
        time_csrs: &(
            Arc<faer::sparse::SparseRowMat<usize, f64>>,
            Arc<faer::sparse::SparseRowMat<usize, f64>>,
            Arc<faer::sparse::SparseRowMat<usize, f64>>,
        ),
        marginal_csr: &Arc<faer::sparse::SparseRowMat<usize, f64>>,
        logslope_csr: &Arc<faer::sparse::SparseRowMat<usize, f64>>,
    ) -> Result<FamilyEvaluation, String> {
        use gam_linalg::matrix::SparseHessianAccumulator;

        let slices = block_slices(self, block_states);
        let p_t = slices.time.len();
        let p_m = slices.marginal.len();
        let p_g = slices.logslope.len();

        let (ref csr_entry, ref csr_exit, ref csr_deriv) = *time_csrs;

        // Build symbolic sparsity patterns once.
        let pattern_time = SparseHessianAccumulator::from_multi_csr(
            &[csr_entry.as_ref(), csr_exit.as_ref(), csr_deriv.as_ref()],
            p_t,
        );
        let pattern_marginal =
            SparseHessianAccumulator::from_single_csr(marginal_csr.as_ref(), p_m);
        let pattern_logslope =
            SparseHessianAccumulator::from_single_csr(logslope_csr.as_ref(), p_g);

        // Pre-extract CSR symbolic parts for zero-overhead inner loop access.
        let e_sym = csr_entry.symbolic();
        let e_rp = e_sym.row_ptr();
        let e_ci = e_sym.col_idx();
        let e_v = csr_entry.val();

        let x_sym = csr_exit.symbolic();
        let x_rp = x_sym.row_ptr();
        let x_ci = x_sym.col_idx();
        let x_v = csr_exit.val();

        let d_sym = csr_deriv.symbolic();
        let d_rp = d_sym.row_ptr();
        let d_ci = d_sym.col_idx();
        let d_v = csr_deriv.val();

        let m_sym = marginal_csr.symbolic();
        let m_rp = m_sym.row_ptr();
        let m_ci = m_sym.col_idx();
        let m_v = marginal_csr.val();

        let g_sym = logslope_csr.symbolic();
        let g_rp = g_sym.row_ptr();
        let g_ci = g_sym.col_idx();
        let g_v = logslope_csr.val();

        // Accumulator type: gradients dense, Hessians sparse value buffers.
        type SAcc = (
            f64,
            Array1<f64>,
            Array1<f64>,
            Array1<f64>,
            SparseHessianAccumulator,
            SparseHessianAccumulator,
            SparseHessianAccumulator,
        );

        let make_acc = || -> SAcc {
            (
                0.0,
                Array1::zeros(p_t),
                Array1::zeros(p_m),
                Array1::zeros(p_g),
                pattern_time.empty_clone(),
                pattern_marginal.empty_clone(),
                pattern_logslope.empty_clone(),
            )
        };

        let (
            ll,
            grad_time,
            grad_marginal,
            grad_logslope,
            acc_time,
            acc_marginal,
            acc_logslope,
        ): SAcc = (0..self.n)
            .into_par_iter()
            .try_fold(make_acc, |mut acc, row| -> Result<_, String> {
                    let (row_nll, f_pi, f_pipi) =
                        self.compute_row_primary_gradient_hessian_uncached(row, block_states)?;
                    acc.0 -= row_nll;

                    // ── gradients (dense axpy via CSR scatter) ────────────
                    {
                        let gt = &mut acc.1;
                        for p in e_rp[row]..e_rp[row + 1] {
                            gt[e_ci[p]] -= f_pi[0] * e_v[p];
                        }
                        for p in x_rp[row]..x_rp[row + 1] {
                            gt[x_ci[p]] -= f_pi[1] * x_v[p];
                        }
                        for p in d_rp[row]..d_rp[row + 1] {
                            gt[d_ci[p]] -= f_pi[2] * d_v[p];
                        }
                    }
                    {
                        let gm = &mut acc.2;
                        let alpha_m = -(f_pi[0] + f_pi[1]);
                        for p in m_rp[row]..m_rp[row + 1] {
                            gm[m_ci[p]] += alpha_m * m_v[p];
                        }
                    }
                    {
                        let gg = &mut acc.3;
                        for p in g_rp[row]..g_rp[row + 1] {
                            gg[g_ci[p]] -= f_pi[3] * g_v[p];
                        }
                    }

                    // ── time Hessian: 3×3 cross-product scatter ──────────
                    // Only emit upper-triangle entries (ca <= cb) to avoid
                    // double-counting: SymmetricMatrix::Sparse mirrors the
                    // upper triangle into the lower.
                    let row_slices: [(std::ops::Range<usize>, &[usize], &[f64]); 3] = [
                        (e_rp[row]..e_rp[row + 1], e_ci, e_v),
                        (x_rp[row]..x_rp[row + 1], x_ci, x_v),
                        (d_rp[row]..d_rp[row + 1], d_ci, d_v),
                    ];
                    let ht = &mut acc.4;
                    for a in 0..3 {
                        for b in 0..3 {
                            let alpha = f_pipi[[a, b]];
                            if alpha == 0.0 {
                                continue;
                            }
                            let (ref ra, cia, va) = row_slices[a];
                            let (ref rb, cib, vb) = row_slices[b];
                            for pi in ra.clone() {
                                let ca = cia[pi];
                                let xia = va[pi] * alpha;
                                for pj in rb.clone() {
                                    let cb = cib[pj];
                                    if ca <= cb {
                                        ht.add_upper(ca, cb, xia * vb[pj]);
                                    }
                                }
                            }
                        }
                    }

                    // ── marginal Hessian: symmetric rank-1 scatter ───────
                    let alpha_m = f_pipi[[0, 0]] + f_pipi[[0, 1]] + f_pipi[[1, 0]] + f_pipi[[1, 1]];
                    if alpha_m != 0.0 {
                        let hm = &mut acc.5;
                        let m_start = m_rp[row];
                        let m_end = m_rp[row + 1];
                        for pi in m_start..m_end {
                            let ca = m_ci[pi];
                            let xia = m_v[pi] * alpha_m;
                            for pj in m_start..m_end {
                                let cb = m_ci[pj];
                                if ca <= cb {
                                    hm.add_upper(ca, cb, xia * m_v[pj]);
                                }
                            }
                        }
                    }

                    // ── logslope Hessian: symmetric rank-1 scatter ───────
                    let alpha_g = f_pipi[[3, 3]];
                    if alpha_g != 0.0 {
                        let hg = &mut acc.6;
                        let g_start = g_rp[row];
                        let g_end = g_rp[row + 1];
                        for pi in g_start..g_end {
                            let ca = g_ci[pi];
                            let xia = g_v[pi] * alpha_g;
                            for pj in g_start..g_end {
                                let cb = g_ci[pj];
                                if ca <= cb {
                                    hg.add_upper(ca, cb, xia * g_v[pj]);
                                }
                            }
                        }
                    }

                    Ok(acc)
                })
                .try_reduce(make_acc, |mut a, b| -> Result<_, String> {
                    a.0 += b.0;
                    a.1 += &b.1;
                    a.2 += &b.2;
                    a.3 += &b.3;
                    a.4.add_values(&b.4.values);
                    a.5.add_values(&b.5.values);
                    a.6.add_values(&b.6.values);
                    Ok(a)
                })?;

        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets: {
                let slices = block_slices(self, block_states);
                let mut sets = vec![
                    BlockWorkingSet::ExactNewton {
                        gradient: grad_time,
                        hessian: SymmetricMatrix::Sparse(acc_time.into_sparse_col_mat()),
                    },
                    BlockWorkingSet::ExactNewton {
                        gradient: grad_marginal,
                        hessian: SymmetricMatrix::Sparse(acc_marginal.into_sparse_col_mat()),
                    },
                    BlockWorkingSet::ExactNewton {
                        gradient: grad_logslope,
                        hessian: SymmetricMatrix::Sparse(acc_logslope.into_sparse_col_mat()),
                    },
                ];
                if let Some(range) = slices.score_warp.as_ref() {
                    sets.push(BlockWorkingSet::ExactNewton {
                        gradient: Array1::zeros(range.len()),
                        hessian: SymmetricMatrix::Dense(Array2::zeros((range.len(), range.len()))),
                    });
                }
                if let Some(range) = slices.link_dev.as_ref() {
                    sets.push(BlockWorkingSet::ExactNewton {
                        gradient: Array1::zeros(range.len()),
                        hessian: SymmetricMatrix::Dense(Array2::zeros((range.len(), range.len()))),
                    });
                }
                sets
            },
        })
    }
}

// ── CustomFamily impl ─────────────────────────────────────────────────

pub(crate) fn time_wiggle_basis_ncols(knots: &Array1<f64>, degree: usize) -> Result<usize, String> {
    if knots.is_empty() {
        return Err(SurvivalMarginalSlopeError::InvalidInput {
            reason: "survival-marginal-slope timewiggle requires at least one knot".to_string(),
        }
        .into());
    }
    let probe = 0.5 * (knots[0] + knots[knots.len() - 1]);
    let h0 = Array1::from_vec(vec![probe]);
    Ok(monotone_wiggle_basis_with_derivative_order(h0.view(), knots, degree, 0)?.ncols())
}

pub(crate) fn smgs_deleted_required_channel_reason(
    p_time: usize,
    p_marginal: usize,
    p_logslope: usize,
    w_time: usize,
    w_marginal: usize,
    w_logslope: usize,
) -> Option<&'static str> {
    if p_time > 0 && w_time == 0 {
        Some("time")
    } else if p_marginal > 0 && w_marginal == 0 {
        Some("marginal")
    } else if p_logslope > 0 && w_logslope == 0 {
        Some("logslope")
    } else {
        None
    }
}
