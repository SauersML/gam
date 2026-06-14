
// ── Workspace structs ─────────────────────────────────────────────────

struct SurvivalMarginalSlopePsiWorkspace {
    family: SurvivalMarginalSlopeFamily,
    block_states: Vec<ParameterBlockState>,
    specs: Vec<ParameterBlockSpec>,
    derivative_blocks: Vec<Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>>,
    cache: Option<EvalCache>,
    /// Outer-only ψ-calculus options. The `outer_score_subsample` field is
    /// the row mask threaded through `sigma_exact_joint_psi_terms_with_options`
    /// and the second-order / Hessian-drift counterparts to make the cached
    /// ψ calculus subsample-aware.
    options: BlockwiseFitOptions,
}


struct SurvivalMarginalSlopeExactNewtonJointHessianWorkspace {
    family: SurvivalMarginalSlopeFamily,
    block_states: Vec<ParameterBlockState>,
    joint_hessian_operator: Arc<dyn HyperOperator>,
    joint_hessian_diagonal: Array1<f64>,
    /// Cached joint log-likelihood and joint gradient from the same row pass
    /// that built the joint Hessian operator. Publishing these via
    /// `joint_log_likelihood_evaluation` / `joint_gradient_evaluation` lets
    /// the inner Newton driver in `custom_family.rs` skip its fallback
    /// separate-pass implementations and run on a single fused n-row sweep
    /// per workspace build.
    joint_log_likelihood: f64,
    joint_gradient: Array1<f64>,
    /// Cached per-row primary gradient + Hessian for timewiggle directional
    /// derivative reuse.  Built once during workspace construction so that
    /// repeated directional-derivative calls do not recompute them.
    eval_cache: Option<EvalCache>,
    /// Outer-only joint-Hessian directional-derivative options. The
    /// `outer_score_subsample` field is the row mask threaded through the
    /// `_with_options` directional-derivative helpers so the cached joint
    /// Hessian Hv-action paths can downscale to the stratified subsample at
    /// large scale. When `None`, the row iteration is identical to the
    /// legacy full-data path.
    options: BlockwiseFitOptions,
}


impl SurvivalMarginalSlopeExactNewtonJointHessianWorkspace {
    fn new(
        family: SurvivalMarginalSlopeFamily,
        block_states: Vec<ParameterBlockState>,
        options: BlockwiseFitOptions,
    ) -> Result<Self, String> {
        let (joint_hessian_operator, joint_hessian_diagonal, joint_log_likelihood, joint_gradient) =
            family.exact_newton_joint_hessian_operator(&block_states, &options)?;
        let eval_cache = if family.flex_timewiggle_active() && !family.flex_active() {
            Some(family.build_eval_cache(&block_states)?)
        } else {
            None
        };
        Ok(Self {
            family,
            block_states,
            joint_hessian_operator,
            joint_hessian_diagonal,
            joint_log_likelihood,
            joint_gradient,
            eval_cache,
            options,
        })
    }
}


impl ExactNewtonJointHessianWorkspace for SurvivalMarginalSlopeExactNewtonJointHessianWorkspace {
    fn joint_log_likelihood_evaluation(&self) -> Result<Option<f64>, String> {
        // Phase 2d fused-pass result: the same n-row sweep that produced
        // `joint_hessian_operator` also produced the joint log-likelihood,
        // so the inner Newton driver reads it from the workspace instead
        // of running a second full-data evaluation.
        Ok(Some(self.joint_log_likelihood))
    }

    fn joint_gradient_evaluation(
        &self,
    ) -> Result<Option<ExactNewtonJointGradientEvaluation>, String> {
        Ok(Some(ExactNewtonJointGradientEvaluation {
            log_likelihood: self.joint_log_likelihood,
            gradient: self.joint_gradient.clone(),
        }))
    }

    fn hessian_dense(&self) -> Result<Option<Array2<f64>>, String> {
        // The operator we already built in `Self::new` carries every block
        // (h_tt, h_mm, h_gg, h_tm, …) of the joint Hessian. Asking the family
        // to re-materialize a dense p×p Hessian via
        // `evaluate_exact_newton_joint_dynamic_q_dense` would re-walk all n
        // rows just to repeat the J^T H J + Σ f K pullback we just finished;
        // at large scale that is the same n-row sweep twice per inner
        // joint-Newton cycle. Reuse the operator's `to_dense()` instead — an
        // O(p²) block copy. Numerically identical to the dense path modulo
        // FMA summation order.
        Ok(Some(self.joint_hessian_operator.to_dense()))
    }

    fn hessian_matvec_available(&self) -> bool {
        true
    }

    fn hessian_matvec(&self, beta_flat: &Array1<f64>) -> Result<Option<Array1<f64>>, String> {
        Ok(Some(self.joint_hessian_operator.mul_vec(beta_flat)))
    }

    fn hessian_matvec_into(&self, v: &Array1<f64>, out: &mut Array1<f64>) -> Result<bool, String> {
        // Forward to HyperOperator's existing `mul_vec_into`, which writes the
        // matvec result directly into the caller-owned buffer with no
        // intermediate allocation. Used by inner-Newton PCG so each CG iter
        // avoids a fresh Array1<f64> on the survival large-scale hot path.
        if v.len() != self.joint_hessian_operator.dim()
            || out.len() != self.joint_hessian_operator.dim()
        {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "hessian_matvec_into: dim mismatch v={} out={} op={}",
                    v.len(),
                    out.len(),
                    self.joint_hessian_operator.dim()
                ),
            }
            .into());
        }
        // ── Step-6 dispatcher: try GPU joint-Hessian × v first ───────────
        //
        // Routes through
        // [`crate::families::survival_marginal_slope_gpu::try_survival_flex_hvp`] via the
        // `gpu::decide` policy.  Returns `Ok(None)` until the joint-β
        // device HVP assembly lands; on `Ok(Some(hv))` we write straight
        // into the caller-owned `out` buffer and skip the prebuilt
        // operator matvec.  The CPU `mul_vec_into` below remains the
        // production fallback path and is byte-for-byte identical to the
        // pre-Step-6 hot path.
        if self.family.effective_flex_active(&self.block_states)?
            && !self.family.flex_timewiggle_active()
        {
            let slices = block_slices(&self.family, &self.block_states);
            if let Some(hv) =
                self.family
                    .try_survival_flex_joint_dispatch_hvp(&self.block_states, &slices, v)?
            {
                if hv.len() == out.len() {
                    out.assign(&hv);
                    return Ok(true);
                }
            }
        }
        self.joint_hessian_operator
            .mul_vec_into(v.view(), out.view_mut());
        Ok(true)
    }

    fn hessian_diagonal(&self) -> Result<Option<Array1<f64>>, String> {
        Ok(Some(self.joint_hessian_diagonal.clone()))
    }

    fn directional_derivative_operator(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn HyperOperator>>, String> {
        if self.family.effective_flex_active(&self.block_states)?
            && !self.family.flex_timewiggle_active()
        {
            return self
                .family
                .exact_newton_joint_hessian_directional_derivative_operator_flex_no_wiggle_with_options(
                    &self.block_states,
                    d_beta_flat,
                    &self.options,
                )
                .map(Some);
        }
        if let Some(cache) = self.eval_cache.as_ref() {
            return self
                .family
                .exact_newton_joint_hessian_directional_derivative_timewiggle_cached(
                    &self.block_states,
                    d_beta_flat,
                    cache,
                )
                .map(|matrix| {
                    Some(Arc::new(
                        crate::solver::estimate::reml::unified::DenseMatrixHyperOperator { matrix },
                    ) as Arc<dyn HyperOperator>)
                });
        }
        self.family
            .exact_newton_joint_hessian_directional_derivative(&self.block_states, d_beta_flat)
            .map(|result| {
                result.map(|matrix| {
                    Arc::new(
                        crate::solver::estimate::reml::unified::DenseMatrixHyperOperator { matrix },
                    ) as Arc<dyn HyperOperator>
                })
            })
    }

    fn directional_derivative(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if let Some(cache) = self.eval_cache.as_ref() {
            return self
                .family
                .exact_newton_joint_hessian_directional_derivative_timewiggle_cached(
                    &self.block_states,
                    d_beta_flat,
                    cache,
                )
                .map(Some);
        }
        self.family
            .exact_newton_joint_hessian_directional_derivative(&self.block_states, d_beta_flat)
    }

    fn second_directional_derivative_operator(
        &self,
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn HyperOperator>>, String> {
        if self.family.effective_flex_active(&self.block_states)?
            && !self.family.flex_timewiggle_active()
        {
            return self
                .family
                .exact_newton_joint_hessiansecond_directional_derivative_operator_flex_no_wiggle_with_options(
                    &self.block_states,
                    d_beta_u_flat,
                    d_beta_v_flat,
                    &self.options,
                )
                .map(Some);
        }
        self.family
            .exact_newton_joint_hessiansecond_directional_derivative(
                &self.block_states,
                d_beta_u_flat,
                d_beta_v_flat,
            )
            .map(|result| {
                result.map(|matrix| {
                    Arc::new(
                        crate::solver::estimate::reml::unified::DenseMatrixHyperOperator { matrix },
                    ) as Arc<dyn HyperOperator>
                })
            })
    }

    fn second_directional_derivative(
        &self,
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family
            .exact_newton_joint_hessiansecond_directional_derivative(
                &self.block_states,
                d_beta_u_flat,
                d_beta_v_flat,
            )
    }
}


impl SurvivalMarginalSlopePsiWorkspace {
    fn new(
        family: SurvivalMarginalSlopeFamily,
        block_states: Vec<ParameterBlockState>,
        specs: Vec<ParameterBlockSpec>,
        derivative_blocks: Vec<Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>>,
        options: BlockwiseFitOptions,
    ) -> Result<Self, String> {
        let cache = if family.flex_active() {
            None
        } else {
            Some(family.build_eval_cache(&block_states)?)
        };
        Ok(Self {
            family,
            block_states,
            specs,
            derivative_blocks,
            cache,
            options,
        })
    }
}


impl crate::families::marginal_slope_shared::MarginalSlopePsiFamily
    for SurvivalMarginalSlopePsiWorkspace
{
    fn is_sigma_aux(&self, psi_index: usize) -> bool {
        self.family
            .is_sigma_aux_index(&self.derivative_blocks, psi_index)
    }

    fn sigma_first_order_terms(&self) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        self.family.sigma_exact_joint_psi_terms_with_options(
            &self.block_states,
            &self.specs,
            &self.options,
        )
    }

    fn psi_first_order_terms(
        &self,
        psi_index: usize,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        self.family.psi_terms_inner_with_options(
            &self.block_states,
            &self.derivative_blocks,
            psi_index,
            self.cache.as_ref(),
            &self.options,
        )
    }

    fn psi_first_order_terms_all(&self) -> Result<Option<Vec<ExactNewtonJointPsiTerms>>, String> {
        let total: usize = self.derivative_blocks.iter().map(Vec::len).sum();
        if total == 0 {
            return Ok(Some(Vec::new()));
        }
        let psi_indices: Vec<usize> = (0..total).collect();
        self.family.psi_terms_inner_batched_with_options(
            &self.block_states,
            &self.derivative_blocks,
            &psi_indices,
            self.cache.as_ref(),
            &self.options,
        )
    }

    fn both_sigma_aux_second_order(&self, psi_i: usize, psi_j: usize) -> bool {
        psi_i == psi_j
    }

    fn sigma_second_order_terms(
        &self,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        self.family
            .sigma_exact_joint_psisecond_order_terms_with_options(&self.block_states, &self.options)
    }

    fn mixed_sigma_aux_second_order(
        &self,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        Ok(None)
    }

    fn psi_second_order_terms(
        &self,
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        self.family.psi_second_order_terms_inner_with_options(
            &self.block_states,
            &self.derivative_blocks,
            psi_i,
            psi_j,
            self.cache.as_ref(),
            &self.options,
        )
    }

    fn sigma_hessian_directional_derivative(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family
            .sigma_exact_joint_psihessian_directional_derivative_with_options(
                &self.block_states,
                d_beta_flat,
                &self.options,
            )
    }

    fn psi_hessian_directional_derivative(
        &self,
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn HyperOperator>>, String> {
        self.family
            .psi_hessian_directional_derivative_operator_with_options(
                &self.block_states,
                &self.derivative_blocks,
                psi_index,
                d_beta_flat,
                &self.options,
            )
    }
}


// ── RowKernel<4> implementation ───────────────────────────────────────

struct SurvivalMarginalSlopeRowKernel {
    family: SurvivalMarginalSlopeFamily,
    block_states: Vec<ParameterBlockState>,
    slices: BlockSlices,
}


impl SurvivalMarginalSlopeRowKernel {
    fn new(family: SurvivalMarginalSlopeFamily, block_states: Vec<ParameterBlockState>) -> Self {
        let slices = block_slices(&family, &block_states);
        Self {
            family,
            block_states,
            slices,
        }
    }
}


impl RowKernel<4> for SurvivalMarginalSlopeRowKernel {
    fn n_rows(&self) -> usize {
        self.family.n
    }
    fn n_coefficients(&self) -> usize {
        self.slices.total
    }

    fn row_kernel(&self, row: usize) -> Result<(f64, [f64; 4], [[f64; 4]; 4]), String> {
        let beta_time = &self.block_states[0].beta;
        let q0 = self.family.design_entry.dot_row(row, beta_time)
            + self.family.offset_entry[row]
            + self.block_states[1].eta[row];
        let q1 = self.family.design_exit.dot_row(row, beta_time)
            + self.family.offset_exit[row]
            + self.block_states[1].eta[row];
        let qd1 = self.family.design_derivative_exit.dot_row(row, beta_time)
            + self.family.derivative_offset_exit[row];
        self.family.row_primary_closed_form_rigid(
            row,
            q0,
            q1,
            qd1,
            &self.block_states,
            self.family.probit_frailty_scale(),
        )
    }

    fn jacobian_action(&self, row: usize, d_beta: &[f64]) -> [f64; 4] {
        let d_beta = ndarray::ArrayView1::from(d_beta);
        let d_time = d_beta.slice(s![self.slices.time.clone()]);
        let d_marginal = d_beta.slice(s![self.slices.marginal.clone()]);
        let d_logslope = d_beta.slice(s![self.slices.logslope.clone()]);
        [
            self.family.design_entry.dot_row_view(row, d_time)
                + self.family.marginal_design.dot_row_view(row, d_marginal),
            self.family.design_exit.dot_row_view(row, d_time)
                + self.family.marginal_design.dot_row_view(row, d_marginal),
            self.family.design_derivative_exit.dot_row_view(row, d_time),
            self.family.logslope_design.dot_row_view(row, d_logslope),
        ]
    }

    fn jacobian_action_matrix(&self, factor: ArrayView2<'_, f64>) -> Option<Array2<f64>> {
        if factor.nrows() != self.slices.total {
            return None;
        }
        let n_rows = self.family.n;
        // Whole-projection build: each axis uses the batched design matvec
        // (`fast_ab` on dense, one operator `dot` per column on operator-backed
        // designs).
        Some(self.assemble_jf(factor, n_rows, |design, factor_block| {
            survival_axis_jf_via_design(design, factor_block, n_rows)
        }))
    }

    fn jacobian_action_matrix_rows(
        &self,
        factor: ArrayView2<'_, f64>,
        start: usize,
        end: usize,
    ) -> Array2<f64> {
        if factor.nrows() != self.slices.total {
            // Shape contract broken (the tiled trace always passes the
            // coefficient-width factor, so this is defensive only): fall back
            // to the exact generic per-row build over the range.
            return crate::families::row_kernel::row_kernel_jacobian_action_matrix_generic_rows(
                self, factor, start, end,
            );
        }
        // Block-tiled build for one row-tile: dense designs slice to a
        // contiguous row block and GEMM (`fast_ab`), operator/sparse designs
        // fall to a row-local dot over the range. Bounds peak memory to the
        // tile while keeping BLAS-3 on the materialized designs.
        let b = end.saturating_sub(start);
        self.assemble_jf(factor, b, |design, factor_block| {
            survival_axis_jf_via_design_rows(design, factor_block, start, end)
        })
    }

    fn jacobian_transpose_action(&self, row: usize, v: &[f64; 4], out: &mut [f64]) {
        {
            let mut time = ndarray::ArrayViewMut1::from(&mut out[self.slices.time.clone()]);
            self.family
                .design_entry
                .axpy_row_into(row, v[0], &mut time)
                .expect("time entry axpy dim mismatch");
            self.family
                .design_exit
                .axpy_row_into(row, v[1], &mut time)
                .expect("time exit axpy dim mismatch");
            self.family
                .design_derivative_exit
                .axpy_row_into(row, v[2], &mut time)
                .expect("time deriv axpy dim mismatch");
        }
        {
            let mut marginal = ndarray::ArrayViewMut1::from(&mut out[self.slices.marginal.clone()]);
            self.family
                .marginal_design
                .axpy_row_into(row, v[0] + v[1], &mut marginal)
                .expect("marginal axpy dim mismatch");
        }
        {
            let mut logslope = ndarray::ArrayViewMut1::from(&mut out[self.slices.logslope.clone()]);
            self.family
                .logslope_design
                .axpy_row_into(row, v[3], &mut logslope)
                .expect("logslope axpy dim mismatch");
        }
    }

    fn add_pullback_hessian(&self, row: usize, h: &[[f64; 4]; 4], target: &mut Array2<f64>) {
        let mut h_arr = Array2::<f64>::zeros((4, 4));
        for a in 0..4 {
            for b in 0..4 {
                h_arr[[a, b]] = h[a][b];
            }
        }
        self.family
            .add_pullback_primary_hessian(target, row, &self.slices, &h_arr);
    }

    fn add_diagonal_quadratic(&self, row: usize, h: &[[f64; 4]; 4], diag: &mut [f64]) {
        let designs: [(usize, &DesignMatrix); 3] = [
            (0, &self.family.design_entry),
            (1, &self.family.design_exit),
            (2, &self.family.design_derivative_exit),
        ];
        for &(pi, des) in &designs {
            {
                let mut td = ndarray::ArrayViewMut1::from(&mut diag[self.slices.time.clone()]);
                des.squared_axpy_row_into(row, h[pi][pi], &mut td)
                    .expect("time squared_axpy dim mismatch");
            }
            for &(pj, des_j) in &designs {
                if pj <= pi {
                    continue;
                }
                let mut td = ndarray::ArrayViewMut1::from(&mut diag[self.slices.time.clone()]);
                des.crossdiag_axpy_row_into(row, des_j, 2.0 * h[pi][pj], &mut td)
                    .expect("time crossdiag dim mismatch");
            }
        }
        {
            let alpha = h[0][0] + 2.0 * h[0][1] + h[1][1];
            let mut md = ndarray::ArrayViewMut1::from(&mut diag[self.slices.marginal.clone()]);
            self.family
                .marginal_design
                .squared_axpy_row_into(row, alpha, &mut md)
                .expect("marginal squared_axpy dim mismatch");
        }
        {
            let mut gd = ndarray::ArrayViewMut1::from(&mut diag[self.slices.logslope.clone()]);
            self.family
                .logslope_design
                .squared_axpy_row_into(row, h[3][3], &mut gd)
                .expect("logslope squared_axpy dim mismatch");
        }
    }

    fn row_third_contracted(&self, row: usize, dir: &[f64; 4]) -> Result<[[f64; 4]; 4], String> {
        // Batched path: one k=5 MultiDirJet [e_0..e_3, dir] covers all 6
        // off-diagonal (a,b) entries via mask reads; 4 small k=3 jets handle
        // diagonal ∂²_{e_a} ∂_{dir} entries. Replaces the legacy 10 separate
        // calls into row_neglog_directional_refs.
        let dir_view = ndarray::aview1(&dir[..]);
        self.family
            .row_primary_third_contracted_batched(row, &self.block_states, dir_view)
    }

    fn row_fourth_contracted(
        &self,
        row: usize,
        dir_u: &[f64; 4],
        dir_v: &[f64; 4],
    ) -> Result<[[f64; 4]; 4], String> {
        // Batched path: one k=6 MultiDirJet [e_0..e_3, dir_u, dir_v] covers
        // the 6 off-diagonal (a,b) entries; 4 k=4 jets handle the diagonal
        // ∂²_{e_a} ∂_{dir_u} ∂_{dir_v} entries.
        let u_view = ndarray::aview1(&dir_u[..]);
        let v_view = ndarray::aview1(&dir_v[..]);
        self.family
            .row_primary_fourth_contracted_batched(row, &self.block_states, u_view, v_view)
    }
}


impl SurvivalMarginalSlopeRowKernel {
    /// Assemble the `(n_out × 4·rank)` joint Jacobian-action projection `Jᵢ · F`
    /// from the four primary axes — `[entry+marginal | exit+marginal |
    /// derivative | logslope]` — given a per-axis builder `axis(design,
    /// factor_block)` that produces that design's `n_out × rank` contribution.
    /// The whole-projection path passes the batched builder; the block-tiled
    /// path passes the row-range builder. Either way at most one axis transient
    /// is alive at a time: the marginal block feeds both the entry and exit
    /// axes, so it is built once and dropped, and every other axis is a
    /// statement-scoped temporary — keeping the assembly peak at
    /// `output + one n_out×rank block` rather than five blocks at once.
    fn assemble_jf<F>(&self, factor: ArrayView2<'_, f64>, n_out: usize, axis: F) -> Array2<f64>
    where
        F: Fn(&DesignMatrix, ArrayView2<'_, f64>) -> Array2<f64>,
    {
        let rank = factor.ncols();
        if rank == 0 {
            return Array2::<f64>::zeros((n_out, 0));
        }
        let f_time = factor.slice(s![self.slices.time.clone(), ..]);
        let f_marginal = factor.slice(s![self.slices.marginal.clone(), ..]);
        let f_logslope = factor.slice(s![self.slices.logslope.clone(), ..]);

        let mut jf = Array2::<f64>::zeros((n_out, 4 * rank));
        {
            let jf_marginal = axis(&self.family.marginal_design, f_marginal);
            {
                let mut axis0 = jf.slice_mut(s![.., 0..rank]);
                axis0.assign(&axis(&self.family.design_entry, f_time));
                axis0 += &jf_marginal;
            }
            {
                let mut axis1 = jf.slice_mut(s![.., rank..2 * rank]);
                axis1.assign(&axis(&self.family.design_exit, f_time));
                axis1 += &jf_marginal;
            }
        }
        jf.slice_mut(s![.., 2 * rank..3 * rank])
            .assign(&axis(&self.family.design_derivative_exit, f_time));
        jf.slice_mut(s![.., 3 * rank..4 * rank])
            .assign(&axis(&self.family.logslope_design, f_logslope));
        jf
    }
}


fn survival_axis_jf_via_design(
    design: &DesignMatrix,
    factor_block: ArrayView2<'_, f64>,
    n_rows: usize,
) -> Array2<f64> {
    let rank = factor_block.ncols();
    if rank == 0 {
        return Array2::<f64>::zeros((n_rows, 0));
    }
    let factor = factor_block.as_standard_layout().into_owned();
    match design.as_dense_ref() {
        Some(dense) => fast_ab(dense, &factor),
        None => {
            let mut out = Array2::<f64>::zeros((n_rows, rank));
            for c in 0..rank {
                let result = design.dot(&factor.column(c).to_owned());
                out.column_mut(c).assign(&result);
            }
            out
        }
    }
}


/// Row-range analogue of [`survival_axis_jf_via_design`]: one design's
/// `(end-start) × rank` Jacobian-action block over rows `[start, end)`. Dense
/// designs slice to a contiguous row block and GEMM via `fast_ab` (BLAS-3,
/// zero-copy slice); sparse/operator designs fall to a row-local
/// `dot_row_view` over the range (each entry touches only its own row, so
/// tiling never re-walks the full operator). Used by the block-tiled trace to
/// hold one row-tile at a time instead of the whole `n × rank` projection.
fn survival_axis_jf_via_design_rows(
    design: &DesignMatrix,
    factor_block: ArrayView2<'_, f64>,
    start: usize,
    end: usize,
) -> Array2<f64> {
    let b = end.saturating_sub(start);
    let rank = factor_block.ncols();
    if rank == 0 {
        return Array2::<f64>::zeros((b, 0));
    }
    let factor = factor_block.as_standard_layout().into_owned();
    match design.as_dense_ref() {
        Some(dense) => {
            let block = dense.slice(s![start..end, ..]);
            fast_ab(&block, &factor)
        }
        None => {
            let mut out = Array2::<f64>::zeros((b, rank));
            for (i, row) in (start..end).enumerate() {
                for c in 0..rank {
                    out[[i, c]] = design.dot_row_view(row, factor.column(c));
                }
            }
            out
        }
    }
}


impl SurvivalMarginalSlopeFamily {
    /// Unified dense joint Hessian assembly for flex and timewiggle paths.
    /// Both paths use q-geometry Jacobians via accumulate_dynamic_q_joint_row.
    /// The rigid path (no flex, no timewiggle) uses the RowKernel fast path.
    /// Owned scratch buffers backing a [`SurvivalFlexGpuRowInputs`]
    /// descriptor for one fit-evaluation call.
    ///
    /// Held by-value across the GPU `try_*` entry so the borrowed slices
    /// in `as_inputs` live as long as the dispatcher invocation.
    fn build_survival_flex_gpu_row_batch(
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
    /// [`crate::families::survival_marginal_slope_gpu::try_survival_flex_gradient`].
    ///
    /// Returns:
    ///
    /// * `Ok(None)` when the GPU path is gated off (policy, shape,
    ///   backend-not-compiled, or runtime declined) — callers fall back
    ///   to the existing CPU per-row sweep.
    /// * `Ok(Some((nll, grad)))` when the GPU produced a usable answer.
    /// * `Err(_)` only when `gpu=force` was requested but the kernel is
    ///   not supported, mirroring the convention in `gpu::decide`.
    fn try_survival_flex_joint_dispatch_gradient(
        &self,
        block_states: &[ParameterBlockState],
        slices: &BlockSlices,
    ) -> Result<Option<(f64, Array1<f64>)>, String> {
        let decision = crate::families::survival_marginal_slope_gpu::row_primary_hessian_decision(
            self.n, N_PRIMARY,
        );
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
        match crate::families::survival_marginal_slope_gpu::try_survival_flex_gradient(inputs, None)
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
    fn try_survival_flex_joint_dispatch_hvp(
        &self,
        block_states: &[ParameterBlockState],
        slices: &BlockSlices,
        v: &Array1<f64>,
    ) -> Result<Option<Array1<f64>>, String> {
        let decision = crate::families::survival_marginal_slope_gpu::row_primary_hessian_decision(
            self.n, N_PRIMARY,
        );
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
        match crate::families::survival_marginal_slope_gpu::try_survival_flex_hvp(inputs, v_slice)
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
    /// CPU fallback, and `Err(_)` only for `gpu=force` shape mismatches.
    fn try_survival_flex_joint_dispatch_dense_hessian(
        &self,
        block_states: &[ParameterBlockState],
        slices: &BlockSlices,
    ) -> Result<Option<Array2<f64>>, String> {
        let decision = crate::families::survival_marginal_slope_gpu::row_primary_hessian_decision(
            self.n, N_PRIMARY,
        );
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
        match crate::families::survival_marginal_slope_gpu::try_survival_flex_dense_hessian(
            inputs, None,
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

    fn evaluate_exact_newton_joint_dynamic_q_dense(
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
        // [`crate::families::survival_marginal_slope_gpu::try_survival_flex_dense_hessian`]
        // and [`crate::families::survival_marginal_slope_gpu::try_survival_flex_gradient`]
        // respectively, with the standard `gpu::decide` policy.  Both
        // currently return `Ok(None)` until the NVRTC joint-β assembly
        // lands (Steps 4 + 5 sibling commits), so the CPU per-row sweep
        // below remains the production path; the wiring exists so the
        // device dispatch becomes hot the moment the GPU side composes
        // the prep cells (`try_device_partition_cells` +
        // `try_device_cell_primary_fixed_partials`) into a joint pullback.
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

        let final_acc = (0..self.n)
            .into_par_iter()
            .try_fold(make_acc, |mut acc, row| -> Result<_, String> {
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
            })
            .try_reduce(make_acc, |mut left, right| -> Result<_, String> {
                left.0.0 += right.0.0;
                left.0.1 += &right.0.1;
                left.0.2 += &right.0.2;
                Ok(left)
            })?;
        Ok(final_acc.0)
    }

    fn evaluate_exact_newton_joint_dense(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        if self.effective_flex_active(block_states)? || self.flex_timewiggle_active() {
            self.evaluate_exact_newton_joint_dynamic_q_dense(block_states)
        } else {
            let kern = SurvivalMarginalSlopeRowKernel::new(self.clone(), block_states.to_vec());
            let rows = crate::families::row_kernel::RowSet::All;
            let cache = build_row_kernel_cache(&kern, &rows)?;
            Ok((
                row_kernel_log_likelihood(&cache, &rows),
                -row_kernel_gradient(&kern, &cache, &rows),
                row_kernel_hessian_dense(&kern, &cache, &rows),
            ))
        }
    }

    fn evaluate_exact_newton_joint_gradient_dynamic_q(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<(f64, Array1<f64>), String> {
        let flex_active = self.effective_flex_active(block_states)?;
        let slices = block_slices(self, block_states);
        // ── Step-6 dispatcher: try GPU joint-β gradient first ────────────
        //
        // Routes through
        // [`crate::families::survival_marginal_slope_gpu::try_survival_flex_gradient`] via
        // the `gpu::decide` policy.  Returns `Ok(None)` until the joint-β
        // device assembly lands (Steps 4 + 5 sibling commits), at which
        // point this site becomes the hot dispatch with no further
        // family-side rework.
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

        (0..self.n)
            .into_par_iter()
            .try_fold(make_acc, |mut acc, row| -> Result<_, String> {
                let q_geom = self.row_dynamic_q_gradient(row, block_states)?;
                let (row_nll, f_pi) = if flex_active {
                    self.compute_row_flex_primary_gradient_exact(
                        row,
                        block_states,
                        &q_geom,
                        &primary,
                    )?
                } else {
                    let (nll, grad_arr, _) = self.row_primary_closed_form_rigid(
                        row,
                        q_geom.q0,
                        q_geom.q1,
                        q_geom.qd1,
                        block_states,
                        self.probit_frailty_scale(),
                    )?;
                    (nll, Array1::from_vec(grad_arr.to_vec()))
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
                Ok(acc)
            })
            .try_reduce(make_acc, |mut left, right| -> Result<_, String> {
                left.0 += right.0;
                left.1 += &right.1;
                Ok(left)
            })
    }

    /// Shared per-row pullback of the timewiggle q-map Jacobian/curvature
    /// derivatives into the joint Hessian accumulator.  Used by both the
    /// cached `_inner` path (with an empty `identity_blocks`) and the flex
    /// path (with non-empty `identity_blocks`); the only behavioural
    /// difference between those callers is whether the identity-block cross
    /// terms are added, which is driven entirely by `identity_blocks`.
    #[allow(clippy::too_many_arguments)]
    fn accumulate_timewiggle_directional_row(
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
    fn exact_newton_joint_hessian_directional_derivative_timewiggle_inner(
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

        let result = (0..self.n)
            .into_par_iter()
            .try_fold(
                || Array2::<f64>::zeros((p_total, p_total)),
                |mut acc, row| -> Result<Array2<f64>, String> {
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
                    Ok(acc)
                },
            )
            .try_reduce(
                || Array2::<f64>::zeros((p_total, p_total)),
                |mut a, b| -> Result<_, String> {
                    a += &b;
                    Ok(a)
                },
            )?;
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
    fn exact_newton_joint_hessian_directional_derivative_timewiggle_flex(
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

        let result = (0..self.n)
            .into_par_iter()
            .try_fold(
                || Array2::<f64>::zeros((p_total, p_total)),
                |mut acc, row| -> Result<Array2<f64>, String> {
                    let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
                    let (_, f_pi, h_pi) = self.compute_row_flex_primary_gradient_hessian_exact(
                        row,
                        block_states,
                        &q_geom,
                        &primary,
                    )?;
                    let u_d = self.row_primary_direction_from_flat_dynamic(
                        row,
                        block_states,
                        &slices,
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

                    Ok(acc)
                },
            )
            .try_reduce(
                || Array2::<f64>::zeros((p_total, p_total)),
                |mut a, b| -> Result<_, String> {
                    a += &b;
                    Ok(a)
                },
            )?;
        Ok(result)
    }

    fn exact_newton_joint_hessian_directional_derivative_timewiggle_cached(
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

    fn exact_newton_joint_hessian_directional_derivative_timewiggle(
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
    fn exact_newton_joint_hessiansecond_directional_derivative_timewiggle(
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

        let result = (0..self.n)
            .into_par_iter()
            .try_fold(
                || Array2::<f64>::zeros((p_total, p_total)),
                |mut acc, row| -> Result<Array2<f64>, String> {
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

                    Ok(acc)
                },
            )
            .try_reduce(
                || Array2::<f64>::zeros((p_total, p_total)),
                |mut a, b| -> Result<_, String> {
                    a += &b;
                    Ok(a)
                },
            )?;
        Ok(result)
    }

    /// Exact first directional derivative for flex without timewiggle.
    /// J is constant (no wiggle), so DH[d] = J^T T[u^d] J + Σ (Hu^d)_r K_r.
    fn exact_newton_joint_hessian_directional_derivative_flex_no_wiggle(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let slices = block_slices(self, block_states);
        let primary = flex_primary_slices(self);
        let p_total = slices.total;
        let identity_blocks = flex_identity_block_pairs(&primary, &slices);
        let result = (0..self.n)
            .into_par_iter()
            .try_fold(
                || Array2::<f64>::zeros((p_total, p_total)),
                |mut acc, row| -> Result<Array2<f64>, String> {
                    let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
                    let h_pi = self
                        .compute_row_flex_primary_gradient_hessian_exact(
                            row,
                            block_states,
                            &q_geom,
                            &primary,
                        )?
                        .2;
                    let u_d = self.row_primary_direction_from_flat_dynamic(
                        row,
                        block_states,
                        &slices,
                        d_beta_flat,
                    )?;
                    let t_ud =
                        self.row_flex_primary_third_contracted_exact(row, block_states, &u_d)?;
                    let h_ud = h_pi.dot(&u_d);
                    // Core q-geometry pullback (Hessian only)
                    self.accumulate_dynamic_q_core_hessian(
                        row,
                        &slices,
                        &q_geom,
                        h_ud.view(),
                        t_ud.view(),
                        &mut acc,
                    )?;
                    // Identity block Hessian: cross + diagonal + cross-cross
                    for (primary_range, joint_range) in &identity_blocks {
                        for local in 0..primary_range.len() {
                            self.accumulate_identity_primary_cross_hessian(
                                row,
                                &slices,
                                &q_geom,
                                t_ud.slice(s![0..N_PRIMARY, primary_range.start + local]),
                                joint_range,
                                local,
                                &mut acc,
                            )?;
                        }
                        self.add_dense_submatrix(
                            &mut acc,
                            joint_range,
                            joint_range,
                            t_ud.slice(s![primary_range.clone(), primary_range.clone()]),
                        );
                    }
                    for li in 0..identity_blocks.len() {
                        for ri in li + 1..identity_blocks.len() {
                            let (lp, lj) = &identity_blocks[li];
                            let (rp, rj) = &identity_blocks[ri];
                            self.add_dense_symmetric_cross_submatrix(
                                &mut acc,
                                lj,
                                rj,
                                t_ud.slice(s![lp.clone(), rp.clone()]),
                            );
                        }
                    }
                    Ok(acc)
                },
            )
            .try_reduce(
                || Array2::<f64>::zeros((p_total, p_total)),
                |mut a, b| -> Result<_, String> {
                    a += &b;
                    Ok(a)
                },
            )?;
        Ok(result)
    }

    /// Exact second directional derivative for flex without timewiggle.
    /// J constant ⇒ D²H[d,e] = J^T Q[u^d,u^e] J + Σ (T_d·u^e)_r K_r.
    fn exact_newton_joint_hessiansecond_directional_derivative_flex_no_wiggle(
        &self,
        block_states: &[ParameterBlockState],
        d_u: &Array1<f64>,
        d_v: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let slices = block_slices(self, block_states);
        let primary = flex_primary_slices(self);
        let p_total = slices.total;
        let identity_blocks = flex_identity_block_pairs(&primary, &slices);
        let result = (0..self.n)
            .into_par_iter()
            .try_fold(
                || Array2::<f64>::zeros((p_total, p_total)),
                |mut acc, row| -> Result<Array2<f64>, String> {
                    let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
                    let ud = self.row_primary_direction_from_flat_dynamic(
                        row,
                        block_states,
                        &slices,
                        d_u,
                    )?;
                    let ue = self.row_primary_direction_from_flat_dynamic(
                        row,
                        block_states,
                        &slices,
                        d_v,
                    )?;
                    let q_de =
                        self.row_flex_primary_fourth_contracted_exact(row, block_states, &ud, &ue)?;
                    let t_d =
                        self.row_flex_primary_third_contracted_exact(row, block_states, &ud)?;
                    let gamma = t_d.dot(&ue);
                    // Hessian-only: accumulate q-core + identity block Hessian
                    self.accumulate_dynamic_q_core_hessian(
                        row,
                        &slices,
                        &q_geom,
                        gamma.view(),
                        q_de.view(),
                        &mut acc,
                    )?;
                    for (primary_range, joint_range) in &identity_blocks {
                        for local in 0..primary_range.len() {
                            self.accumulate_identity_primary_cross_hessian(
                                row,
                                &slices,
                                &q_geom,
                                q_de.slice(s![0..N_PRIMARY, primary_range.start + local]),
                                joint_range,
                                local,
                                &mut acc,
                            )?;
                        }
                        self.add_dense_submatrix(
                            &mut acc,
                            joint_range,
                            joint_range,
                            q_de.slice(s![primary_range.clone(), primary_range.clone()]),
                        );
                    }
                    for li in 0..identity_blocks.len() {
                        for ri in li + 1..identity_blocks.len() {
                            let (lp, lj) = &identity_blocks[li];
                            let (rp, rj) = &identity_blocks[ri];
                            self.add_dense_symmetric_cross_submatrix(
                                &mut acc,
                                lj,
                                rj,
                                q_de.slice(s![lp.clone(), rp.clone()]),
                            );
                        }
                    }
                    Ok(acc)
                },
            )
            .try_reduce(
                || Array2::<f64>::zeros((p_total, p_total)),
                |mut a, b| -> Result<_, String> {
                    a += &b;
                    Ok(a)
                },
            )?;
        Ok(result)
    }

    fn evaluate_blockwise_exact_newton(
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

    fn evaluate_blockwise_exact_newton_per_z(
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
        let (ll, grad_t, grad_m, grad_g, hess_t, hess_m, hess_g): PerZBlockAcc = (0..self.n)
            .into_par_iter()
            .try_fold(
                || {
                    (
                        0.0,
                        Array1::<f64>::zeros(p_t),
                        Array1::<f64>::zeros(p_m),
                        Array1::<f64>::zeros(p_g),
                        Array2::<f64>::zeros((p_t, p_t)),
                        Array2::<f64>::zeros((p_m, p_m)),
                        Array2::<f64>::zeros((p_g, p_g)),
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
            )
            .try_reduce(
                || {
                    (
                        0.0,
                        Array1::<f64>::zeros(p_t),
                        Array1::<f64>::zeros(p_m),
                        Array1::<f64>::zeros(p_g),
                        Array2::<f64>::zeros((p_t, p_t)),
                        Array2::<f64>::zeros((p_m, p_m)),
                        Array2::<f64>::zeros((p_g, p_g)),
                    )
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
            )?;
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

    fn evaluate_exact_newton_joint_dense_per_z(
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
    fn evaluate_blockwise_exact_newton_flexible(
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
    fn evaluate_blockwise_exact_newton_timewiggle(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<FamilyEvaluation, String> {
        let primary = flex_primary_slices(self);
        self.evaluate_blockwise_exact_newton_dynamic_q(block_states, &primary, |row, _| {
            self.compute_row_primary_gradient_hessian_uncached(row, block_states)
        })
    }

    fn evaluate_blockwise_exact_newton_mixed(
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
        use crate::matrix::SparseHessianAccumulator;

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
                    // wrong storage variant — invariant violation.
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

    fn evaluate_blockwise_exact_newton_dense(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<FamilyEvaluation, String> {
        // Build RowKernel — the single source of truth for all exact-Newton
        // quantities.  The cache evaluates every row kernel once and stores
        // (nll_i, g_i[4], H_i[4×4]).
        let kern = SurvivalMarginalSlopeRowKernel::new(self.clone(), block_states.to_vec());
        let rows = crate::families::row_kernel::RowSet::All;
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

    fn evaluate_blockwise_exact_newton_sparse(
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
        use crate::matrix::SparseHessianAccumulator;

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

fn time_wiggle_basis_ncols(knots: &Array1<f64>, degree: usize) -> Result<usize, String> {
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


fn smgs_deleted_required_channel_reason(
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


impl CustomFamily for SurvivalMarginalSlopeFamily {
    /// #808: engage the inner self-vanishing Levenberg–Marquardt μ on a
    /// full-rank-but-ill-conditioned penalized Hessian. Clustered-PC marginal +
    /// log-slope share a matern PC basis → `H_pen` is full rank (`nullity == 0`)
    /// yet cond ≈ 5.8e6; the nullity-only μ gate would leave the trust-region
    /// Newton oscillating on the near-singular mode. μ is self-vanishing
    /// (∝ ‖∇L − Sβ‖∞ → 0 at the fixed point), so the converged β is the exact
    /// unconditioned solution — log-slope conditioned, NOT reduced. Survival-local.
    fn levenberg_on_ill_conditioning(&self) -> bool {
        true
    }

    fn persistent_warm_start_fingerprint(
        &self,
        specs: &[ParameterBlockSpec],
        options: &BlockwiseFitOptions,
    ) -> Option<String> {
        if !parameter_block_specs_match_rows(specs, self.n)
            || !options.inner_tol.is_finite()
            || options.inner_tol <= 0.0
        {
            return None;
        }
        let mut hasher = crate::cache::Fingerprinter::new();
        hasher.write_str("survival-marginal-slope-family");
        hasher.write_usize(self.n);
        hasher.write_usize(self.event.len());
        for &value in self.event.iter() {
            hasher.write_f64(value);
        }
        hasher.write_usize(self.weights.len());
        for &value in self.weights.iter() {
            hasher.write_f64(value);
        }
        hasher.write_usize(self.z.nrows());
        hasher.write_usize(self.z.ncols());
        for &value in self.z.iter() {
            hasher.write_f64(value);
        }
        match self.gaussian_frailty_sd {
            Some(value) => {
                hasher.write_bool(true);
                hasher.write_f64(value);
            }
            None => hasher.write_bool(false),
        }
        hasher.write_f64(self.derivative_guard);
        hasher.write_usize(self.offset_entry.len());
        for &value in self.offset_entry.iter() {
            hasher.write_f64(value);
        }
        hasher.write_usize(self.offset_exit.len());
        for &value in self.offset_exit.iter() {
            hasher.write_f64(value);
        }
        hasher.write_usize(self.derivative_offset_exit.len());
        for &value in self.derivative_offset_exit.iter() {
            hasher.write_f64(value);
        }
        Some(hasher.finish_hex())
    }

    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    fn coefficient_hessian_cost(&self, specs: &[ParameterBlockSpec]) -> u64 {
        // Operator-aware: the rigid K=4 RowKernel + RowKernelHessianWorkspace
        // adapter (see `exact_newton_joint_hessian_workspace`) applies joint
        // Hv at O(n · (p_time + p_marginal + p_logslope + p_flex)) per call.
        // Report the operator work model so diagnostics and first-order-only
        // policies reflect the representation that actually executes.
        crate::families::coefficient_cost::joint_coupled_operator_aware_hessian_cost(
            self.n as u64,
            specs,
        )
    }

    fn outer_derivative_policy(
        &self,
        specs: &[ParameterBlockSpec],
        psi_dim: usize,
        options: &BlockwiseFitOptions,
    ) -> crate::custom_family::OuterDerivativePolicy {
        use crate::custom_family::OuterDerivativePolicy;

        let capability = self.exact_outer_derivative_order(specs, options);
        let rho_dim = specs
            .iter()
            .map(|spec| spec.penalties.len() as u128)
            .sum::<u128>();
        let k = rho_dim.saturating_add(psi_dim as u128).max(1);

        let predicted_hessian_work = if !self.flex_active() && !self.flex_timewiggle_active() {
            // Rigid survival marginal-slope evaluates outer rho/psi
            // coordinate corrections and projected logdet traces through
            // row-kernel/HVP paths.  The projected subspace trace reductions
            // are batched in `reml::unified`, so the shared X·U_S work is
            // paid once per derivative group rather than once per coordinate.
            // Model the work that actually executes: one row-kernel pass per
            // outer coordinate and coefficient axis, plus the fixed four
            // primary survival channels.
            let p_total = specs
                .iter()
                .map(|spec| spec.design.ncols() as u128)
                .sum::<u128>();
            (self.n as u128)
                .saturating_mul(k)
                .saturating_mul(p_total.saturating_add(N_PRIMARY as u128))
        } else {
            // Flex/time-wiggle survival paths have higher-order dynamic-q
            // row geometry. Keep the generic dense policy there until those
            // paths have their own measured row-work model.
            let (gradient_work, hessian_work) =
                crate::custom_family::default_outer_derivative_policy_costs(
                    specs,
                    psi_dim,
                    self.coefficient_gradient_cost(specs),
                    self.coefficient_hessian_cost(specs),
                );
            return OuterDerivativePolicy {
                capability,
                predicted_gradient_work: gradient_work,
                predicted_hessian_work: hessian_work,
                subsample_capable: true,
            };
        };

        OuterDerivativePolicy {
            capability,
            predicted_gradient_work: predicted_hessian_work / 2,
            predicted_hessian_work,
            subsample_capable: true,
        }
    }

    fn exact_newton_joint_psi_workspace_for_first_order_terms(&self) -> bool {
        true
    }

    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        self.evaluate_blockwise_exact_newton(block_states)
    }

    fn log_likelihood_only(&self, block_states: &[ParameterBlockState]) -> Result<f64, String> {
        let options = BlockwiseFitOptions {
            auto_outer_subsample: false,
            ..BlockwiseFitOptions::default()
        };
        SurvivalMarginalSlopeFamily::log_likelihood_only_with_options(self, block_states, &options)
    }

    fn log_likelihood_only_with_options(
        &self,
        block_states: &[ParameterBlockState],
        options: &BlockwiseFitOptions,
    ) -> Result<f64, String> {
        let owned;
        let options: &BlockwiseFitOptions = match self.install_auto_outer_subsample_options(options)
        {
            Some(cloned) => {
                owned = cloned;
                &owned
            }
            None => options,
        };
        SurvivalMarginalSlopeFamily::log_likelihood_only_with_options(self, block_states, options)
    }

    fn has_explicit_joint_hessian(&self) -> bool {
        true
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        let total = block_states
            .iter()
            .map(|state| state.beta.len())
            .sum::<usize>();
        if total >= 512 {
            return Ok(None);
        }
        if self.per_z_logslope_active() {
            return Ok(Some(
                self.evaluate_exact_newton_joint_dense_per_z(block_states)?
                    .2,
            ));
        }
        Ok(Some(
            self.evaluate_exact_newton_joint_dense(block_states)?.2,
        ))
    }

    fn exact_newton_joint_gradient_evaluation(
        &self,
        block_states: &[ParameterBlockState],
        block_specs: &[ParameterBlockSpec],
    ) -> Result<Option<ExactNewtonJointGradientEvaluation>, String> {
        assert!(block_specs.len() <= isize::MAX as usize);
        if self.per_z_logslope_active() {
            let (log_likelihood, gradient, _) =
                self.evaluate_exact_newton_joint_dense_per_z(block_states)?;
            return Ok(Some(ExactNewtonJointGradientEvaluation {
                log_likelihood,
                gradient,
            }));
        }
        if self.effective_flex_active(block_states)? || self.flex_timewiggle_active() {
            let (log_likelihood, gradient) =
                self.evaluate_exact_newton_joint_gradient_dynamic_q(block_states)?;
            return Ok(Some(ExactNewtonJointGradientEvaluation {
                log_likelihood,
                gradient,
            }));
        }
        let kern = SurvivalMarginalSlopeRowKernel::new(self.clone(), block_states.to_vec());
        let rows = crate::families::row_kernel::RowSet::All;
        let cache = build_row_kernel_cache(&kern, &rows)?;
        Ok(Some(ExactNewtonJointGradientEvaluation {
            log_likelihood: row_kernel_log_likelihood(&cache, &rows),
            gradient: -row_kernel_gradient(&kern, &cache, &rows),
        }))
    }

    fn requires_joint_outer_hyper_path(&self) -> bool {
        true
    }

    fn exact_newton_joint_hessian_workspace(
        &self,
        block_states: &[ParameterBlockState],
        block_specs: &[ParameterBlockSpec],
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        assert!(block_specs.len() <= isize::MAX as usize);
        if self.per_z_logslope_active() {
            return Ok(None);
        }
        if !self.effective_flex_active(block_states)? && !self.flex_timewiggle_active() {
            let kern = SurvivalMarginalSlopeRowKernel::new(self.clone(), block_states.to_vec());
            return Ok(Some(Arc::new(RowKernelHessianWorkspace::new(kern)?)));
        }
        Ok(Some(Arc::new(
            SurvivalMarginalSlopeExactNewtonJointHessianWorkspace::new(
                self.clone(),
                block_states.to_vec(),
                BlockwiseFitOptions::default(),
            )?,
        )))
    }

    fn exact_newton_joint_hessian_workspace_with_options(
        &self,
        block_states: &[ParameterBlockState],
        block_specs: &[ParameterBlockSpec],
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        assert!(block_specs.len() <= isize::MAX as usize);
        if self.per_z_logslope_active() {
            return Ok(None);
        }
        if !self.effective_flex_active(block_states)? && !self.flex_timewiggle_active() {
            // Rigid path: RowKernel<4> operator wired through the supplied
            // `RowSet`. The cache and every assembly function honour the
            // mask uniformly through the Horvitz–Thompson weights on each
            // `WeightedOuterRow`.
            let kern = SurvivalMarginalSlopeRowKernel::new(self.clone(), block_states.to_vec());
            let rows = crate::families::row_kernel::RowSet::from_options(options, self.n);
            return Ok(Some(Arc::new(RowKernelHessianWorkspace::with_rows(
                kern, rows,
            )?)));
        }
        // Flex / timewiggle path. This workspace is constructed by the INNER
        // joint Newton solver. Inner-coefficient evaluation must use the same
        // row measure across (objective, gradient, Hessian, HVP); otherwise
        // the inner Newton step is wrong. The operator builder
        // `exact_newton_joint_hessian_operator` and the directional-derivative
        // HVP helpers both read row iteration + HT weights from `options` via
        // `outer_row_indices` / `outer_row_weights_by_index`, so any
        // `outer_score_subsample` mask propagates consistently across all
        // three computations.
        //
        // We must NOT auto-install an outer-score subsample mask at this
        // call site. The historical bug was: the install fired here under
        // an outer-derivative scope (because the inner Newton runs nested
        // inside an outer derivative eval), the workspace cached a mask the
        // HVPs honoured, while an older version of the operator builder
        // unconditionally iterated `0..self.n` and therefore disagreed with
        // the HVPs. The failing large-scale log emitted
        //   `phase=1 eval=N/12 ... K=19661`
        // immediately before a ~430 s full-data Hessian build for exactly
        // this reason. The principled fix is to make outer subsampling an
        // explicit upstream decision: outer code that wants HT subsampling
        // on the workspace's HVPs sets `options.outer_score_subsample` itself,
        // never via a side-effect installed inside an inner-coefficient
        // workspace constructor.
        Ok(Some(Arc::new(
            SurvivalMarginalSlopeExactNewtonJointHessianWorkspace::new(
                self.clone(),
                block_states.to_vec(),
                options.clone(),
            )?,
        )))
    }

    fn inner_coefficient_hessian_hvp_available(&self, specs: &[ParameterBlockSpec]) -> bool {
        // The workspace impl above unconditionally returns `Some(workspace)`
        // — the rigid path produces a `RowKernelHessianWorkspace` and the
        // flex path produces a
        // `SurvivalMarginalSlopeExactNewtonJointHessianWorkspace`. Both
        // route the joint Hessian through Hv operators rather than dense
        // assembly.
        !self.per_z_logslope_active() && parameter_block_specs_match_rows(specs, self.n)
    }

    fn outer_hyper_hessian_hvp_available(&self, specs: &[ParameterBlockSpec]) -> bool {
        // The exact outer Hessian over θ=(ρ,ψ[,log σ]) can be applied without
        // pairwise θθ materialization: coefficient-Hessian drift terms use the
        // joint-Hessian workspace's directional-derivative operators, and ψ
        // drift terms use `SurvivalMarginalSlopePsiWorkspace` to return
        // `DriftDerivResult::Operator`. Advertising this capability lets the
        // outer planner keep ARC/Newton curvature at large n or large ψ_dim
        // while routing the representation through matrix-free HVPs.
        !self.per_z_logslope_active() && parameter_block_specs_match_rows(specs, self.n)
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if self.per_z_logslope_active() {
            return Ok(None);
        }
        if self.effective_flex_active(block_states)? {
            return if self.flex_timewiggle_active() {
                self.exact_newton_joint_hessian_directional_derivative_timewiggle_flex(
                    block_states,
                    d_beta_flat,
                )
            } else {
                self.exact_newton_joint_hessian_directional_derivative_flex_no_wiggle(
                    block_states,
                    d_beta_flat,
                )
            }
            .map(Some);
        }
        if self.flex_timewiggle_active() {
            return self
                .exact_newton_joint_hessian_directional_derivative_timewiggle(
                    block_states,
                    d_beta_flat,
                )
                .map(Some);
        }
        let kern = SurvivalMarginalSlopeRowKernel::new(self.clone(), block_states.to_vec());
        let sl = d_beta_flat.as_slice().ok_or("non-contiguous d_beta")?;
        crate::families::row_kernel::row_kernel_directional_derivative(
            &kern,
            &crate::families::row_kernel::RowSet::All,
            sl,
        )
        .map(Some)
    }

    fn exact_newton_joint_hessiansecond_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if self.per_z_logslope_active() {
            return Ok(None);
        }
        if self.effective_flex_active(block_states)? {
            if self.flex_timewiggle_active() {
                return self
                    .exact_newton_joint_hessiansecond_directional_derivative_timewiggle(
                        block_states,
                        d_beta_u_flat,
                        d_beta_v_flat,
                    )
                    .map(Some);
            }
            return self
                .exact_newton_joint_hessiansecond_directional_derivative_flex_no_wiggle(
                    block_states,
                    d_beta_u_flat,
                    d_beta_v_flat,
                )
                .map(Some);
        }
        if self.flex_timewiggle_active() {
            return self
                .exact_newton_joint_hessiansecond_directional_derivative_timewiggle(
                    block_states,
                    d_beta_u_flat,
                    d_beta_v_flat,
                )
                .map(Some);
        }
        let kern = SurvivalMarginalSlopeRowKernel::new(self.clone(), block_states.to_vec());
        let su = d_beta_u_flat.as_slice().ok_or("non-contiguous d_beta_u")?;
        let sv = d_beta_v_flat.as_slice().ok_or("non-contiguous d_beta_v")?;
        crate::families::row_kernel::row_kernel_second_directional_derivative(
            &kern,
            &crate::families::row_kernel::RowSet::All,
            su,
            sv,
        )
        .map(Some)
    }

    fn exact_newton_joint_psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        if self.is_sigma_aux_index(derivative_blocks, psi_index) {
            return self.sigma_exact_joint_psi_terms(block_states, specs);
        }
        self.psi_terms(block_states, derivative_blocks, psi_index)
    }

    fn exact_newton_joint_psisecond_order_terms(
        &self,
        block_states: &[ParameterBlockState],
        block_specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        assert!(block_specs.len() <= isize::MAX as usize);
        if self.is_sigma_aux_index(derivative_blocks, psi_i)
            || self.is_sigma_aux_index(derivative_blocks, psi_j)
        {
            if psi_i == psi_j {
                return self.sigma_exact_joint_psisecond_order_terms(block_states);
            }
            return Ok(None);
        }
        self.psi_second_order_terms(block_states, derivative_blocks, psi_i, psi_j)
    }

    fn exact_newton_joint_psihessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        assert!(block_specs.len() <= isize::MAX as usize);
        if self.is_sigma_aux_index(derivative_blocks, psi_index) {
            return self
                .sigma_exact_joint_psihessian_directional_derivative(block_states, d_beta_flat);
        }
        self.psi_hessian_directional_derivative(
            block_states,
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
        if self.per_z_logslope_active() {
            return Ok(None);
        }
        Ok(Some(Arc::new(
            crate::families::marginal_slope_shared::MarginalSlopeExactNewtonPsiWorkspace::new(
                SurvivalMarginalSlopePsiWorkspace::new(
                    self.clone(),
                    block_states.to_vec(),
                    specs.to_vec(),
                    derivative_blocks.to_vec(),
                    BlockwiseFitOptions::default(),
                )?,
            ),
        )))
    }

    fn exact_newton_joint_psi_workspace_with_options(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Arc<dyn ExactNewtonJointPsiWorkspace>>, String> {
        let owned;
        let options: &BlockwiseFitOptions = match self.install_auto_outer_subsample_options(options)
        {
            Some(cloned) => {
                owned = cloned;
                &owned
            }
            None => options,
        };
        Ok(Some(Arc::new(
            crate::families::marginal_slope_shared::MarginalSlopeExactNewtonPsiWorkspace::new(
                SurvivalMarginalSlopePsiWorkspace::new(
                    self.clone(),
                    block_states.to_vec(),
                    specs.to_vec(),
                    derivative_blocks.to_vec(),
                    options.clone(),
                )?,
            ),
        )))
    }

    fn block_linear_constraints(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        block_spec: &ParameterBlockSpec,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
        assert!(!block_spec.name.is_empty());
        if block_idx == 0 {
            return self.effective_time_linear_constraints();
        }
        if self.score_warp.is_some() && block_idx == 3 {
            return self
                .score_warp
                .as_ref()
                .map(|runtime| self.score_warp_linear_constraints(runtime))
                .transpose();
        }
        let link_block_idx = if self.score_warp.is_some() { 4 } else { 3 };
        if self.link_dev.is_some() && block_idx == link_block_idx {
            return Ok(self
                .link_dev
                .as_ref()
                .map(DeviationRuntime::structural_monotonicity_constraints));
        }
        Ok(None)
    }

    fn max_feasible_step_size(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        delta: &Array1<f64>,
    ) -> Result<Option<f64>, String> {
        if block_idx == 0 {
            return self.max_feasible_time_step(&block_states[0].beta, delta);
        }
        Ok(None)
    }

    fn post_update_block_beta(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        block_spec: &ParameterBlockSpec,
        beta: Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        assert!(!block_spec.name.is_empty());
        if block_idx >= block_states.len() {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "post-update block index {} out of range for {} blocks",
                    block_idx,
                    block_states.len()
                ),
            }
            .into());
        }
        if block_idx == 0 {
            let current = &block_states[0].beta;
            self.validate_time_qd1_feasible(current, "current")?;
            self.validate_time_qd1_feasible(&beta, "proposed")?;
            return Ok(beta);
        }
        if self.score_warp.is_some()
            && block_idx == 3
            && let Some(runtime) = &self.score_warp
        {
            let current = &block_states[3].beta;
            if current.len() != beta.len() {
                return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                        reason: format!(
                            "survival score-warp post-update beta length mismatch: current={}, proposed={}",
                            current.len(),
                            beta.len()
                        ),
                    }
                    .into());
            }
            let expected = runtime.basis_dim() * self.score_dim();
            if beta.len() != expected {
                return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                        reason: format!(
                            "survival score-warp post-update beta length mismatch: proposed={}, expected {expected} for K={} and basis dim {}",
                            beta.len(),
                            self.score_dim(),
                            runtime.basis_dim()
                        ),
                    }
                    .into());
            }
            for coord in 0..self.score_dim() {
                let range = score_warp_component_range(runtime, coord);
                let proposed_local = beta.slice(s![range.clone()]).to_owned();
                runtime
                    .monotonicity_feasible(&proposed_local, &format!("score_warp_dev[z{coord}]"))?;
            }
            return Ok(beta);
        }
        let link_block_idx = if self.score_warp.is_some() { 4 } else { 3 };
        if self.link_dev.is_some()
            && block_idx == link_block_idx
            && let Some(runtime) = &self.link_dev
        {
            let current = block_states
                .get(link_block_idx)
                .map(|state| &state.beta)
                .ok_or_else(|| "missing survival link-deviation block state".to_string())?;
            if current.len() != beta.len() {
                return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                        reason: format!(
                            "survival link-deviation post-update beta length mismatch: current={}, proposed={}",
                            current.len(),
                            beta.len()
                        ),
                    }
                    .into());
            }
            runtime.monotonicity_feasible(current, "link_dev current")?;
            runtime.monotonicity_feasible(&beta, "link_dev proposed")?;
            return Ok(beta);
        }
        Ok(beta)
    }
}


// ── β-dependent Jacobian callbacks for the three primary blocks ──────────
//
// Three n_outputs=3 stacked Jacobians (one per primary output: η0, η1, ad1).
// Row ordering: all n rows for η0, then all n rows for η1, then all n rows ad1.
//
// Family scalars struct: passed through FamilyLinearizationState when the
// family has evaluated β and needs the β-dependent per-row values.
// At initialization (β=0, family_scalars=None) callbacks fall back to the
// β=0 linearization: c=1, g=0, q0=0, q1=0, qd1=0.

/// Per-row scalars for survival marginal-slope Jacobian evaluation at a given β.
///
/// Fields:
/// - `q0_i`: entry-time probit argument (per-row, length n)
/// - `q1_i`: exit-time probit argument (per-row, length n)
/// - `qd1_i`: derivative probit argument (per-row, length n)
/// - `g_i`: per-row log-slope value `g = logslope_design · β_logslope`
/// - `c_i`: `sqrt(1 + (s·g_i)²)` (per-row, length n)
/// - `s`: probit scale (scalar, = `probit_frailty_scale()`)
/// - `z_i`: per-row covariate score (length n)
/// - `observed_g_i`: per-row observed baseline g (length n, pre-β contribution)
pub struct SurvivalMarginalSlopeFamilyScalars {
    pub q0_i: Vec<f64>,
    pub q1_i: Vec<f64>,
    pub qd1_i: Vec<f64>,
    pub g_i: Vec<f64>,
    pub c_i: Vec<f64>,
    pub s: f64,
    pub z_i: Vec<f64>,
    pub observed_g_i: Vec<f64>,
}


impl SurvivalMarginalSlopeFamilyScalars {
    /// Construct with c_i computed from g_i and s.
    pub fn new(
        q0_i: Vec<f64>,
        q1_i: Vec<f64>,
        qd1_i: Vec<f64>,
        g_i: Vec<f64>,
        s: f64,
        z_i: Vec<f64>,
        observed_g_i: Vec<f64>,
    ) -> Self {
        let c_i: Vec<f64> = g_i
            .iter()
            .map(|&g| (1.0 + (s * g).powi(2)).sqrt())
            .collect();
        Self {
            q0_i,
            q1_i,
            qd1_i,
            g_i,
            c_i,
            s,
            z_i,
            observed_g_i,
        }
    }
}


/// n_outputs=3 stacked Jacobian for the logslope block.
///
/// The logslope block contributes `g_i = logslope_design[i] · β` to each row.
/// The three stacked output rows for row i are:
///
/// ```text
/// ∂η0[i]/∂β = (q0[i] · s²·g[i]/c[i] + s·z[i]) · G[i,:]
/// ∂η1[i]/∂β = (q1[i] · s²·g[i]/c[i] + s·z[i]) · G[i,:]
/// ∂ad1[i]/∂β = qd1[i] · s²·g[i]/c[i] · G[i,:]
/// ```
///
/// At g=0 (β=0 init): c=1, s²·g/c=0, so:
/// ```text
/// ∂η0[i]/∂β = s·z[i] · G[i,:]
/// ∂η1[i]/∂β = s·z[i] · G[i,:]
/// ∂ad1[i]/∂β = 0
/// ```
pub struct LogslopeBlockJacobian {
    /// The logslope basis design (n × p_logslope). Held behind an `Arc` so a
    /// materialized design is shared with its owner rather than deep-copied —
    /// at biobank scale each retained `n × p` copy in these construction-time
    /// callbacks was hundreds of MiB held for the whole fit (#979 OOM).
    design: Arc<Array2<f64>>,
    /// Per-row covariate score z_i (length n).
    z: Vec<f64>,
    /// Probit scale s.
    s: f64,
}


impl LogslopeBlockJacobian {
    pub fn new(design: impl Into<Arc<Array2<f64>>>, z: Vec<f64>, s: f64) -> Self {
        Self {
            design: design.into(),
            z,
            s,
        }
    }
}


impl crate::custom_family::BlockEffectiveJacobian for LogslopeBlockJacobian {
    fn effective_jacobian_rows(
        &self,
        state: &crate::custom_family::FamilyLinearizationState<'_>,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, String> {
        let n = self.design.nrows();
        let p = self.design.ncols();
        let rows = rows.start.min(n)..rows.end.min(n);
        let chunk = rows.end - rows.start;
        // Read s_f from the linearization state so that outer-loop σ updates are
        // reflected without requiring the spec to be rebuilt.  Every construction
        // site sets probit_frailty_scale = 1.0 when it does not know the family's
        // σ; `self.s` carries the construction-time value as a fallback.  Use the
        // state value when positive and finite; fall back to self.s otherwise.
        // For the no-frailty case both are 1.0 so the choice is immaterial.
        let s = if state.probit_frailty_scale > 0.0 && state.probit_frailty_scale.is_finite() {
            state.probit_frailty_scale
        } else {
            self.s
        };

        // Compute per-row g_i = logslope_design[i,:] · β directly from state.beta.
        // This block owns the logslope design so g is always self-computable without
        // family_scalars.  Truncate to min(p, beta.len()) to handle the pre-fit
        // initialisation call where beta may be shorter or empty.
        let beta = state.beta;
        let p_use = p.min(beta.len());
        let mut g_rows = vec![0.0_f64; chunk];
        for i in rows.clone() {
            let local_i = i - rows.start;
            for j in 0..p_use {
                g_rows[local_i] += self.design[[i, j]] * beta[j];
            }
        }

        // Hard contract: when any g_i is nonzero the per-row primary scalars
        // (q0, q1, qd1) from the time/marginal blocks are required for the correct
        // hyperbolic formula (q·s²g/c + s·z).  Those scalars live in family_scalars.
        // A caller operating at non-init β must populate them.
        let scalars: Option<&SurvivalMarginalSlopeFamilyScalars> = state
            .family_scalars
            .as_ref()
            .and_then(|a| a.downcast_ref::<SurvivalMarginalSlopeFamilyScalars>());

        let any_nonzero_g = g_rows.iter().any(|&gi| gi != 0.0);
        if any_nonzero_g && scalars.is_none() {
            return Err("survival marginal-slope logslope block requires \
                 SurvivalMarginalSlopeFamilyScalars when beta != 0 \
                 (g_i != 0 for at least one row); got family_scalars: None. \
                 The caller must compute per-row (q0, q1, qd1) at the current \
                 beta and pass them via FamilyLinearizationState::family_scalars."
                .to_string());
        }

        let mut jac = Array2::<f64>::zeros((3 * chunk, p));

        for i in rows.clone() {
            let local_i = i - rows.start;
            // g_i computed from beta above; c_i from family_scalars when present,
            // otherwise computed from g_i.  q0/q1/qd1 from family_scalars -
            // guaranteed present by the contract check whenever g_i != 0.
            let g = g_rows[local_i];
            let (q0, q1, qd1, c) = match scalars {
                Some(sc) => (sc.q0_i[i], sc.q1_i[i], sc.qd1_i[i], sc.c_i[i]),
                None => {
                    // g == 0.0 here (enforced by contract above), so c = 1.
                    // The q terms vanish: q * s^2 * 0 / 1 = 0.
                    (0.0_f64, 0.0_f64, 0.0_f64, 1.0_f64)
                }
            };
            let z_i = self.z[i];
            let sg_over_c = if g == 0.0 { 0.0 } else { s * s * g / c };
            let coeff_eta0 = q0 * sg_over_c + s * z_i;
            let coeff_eta1 = q1 * sg_over_c + s * z_i;
            let coeff_ad1 = qd1 * sg_over_c;

            for j in 0..p {
                let g_ij = self.design[[i, j]];
                jac[[local_i, j]] = coeff_eta0 * g_ij;
                jac[[chunk + local_i, j]] = coeff_eta1 * g_ij;
                jac[[2 * chunk + local_i, j]] = coeff_ad1 * g_ij;
            }
        }
        Ok(jac)
    }

    fn n_outputs(&self) -> usize {
        3
    }
}


/// n_outputs=3 stacked Jacobian for the marginal block.
///
/// The marginal block contributes identically to q0 and q1 (both entry and
/// exit probit arguments) but not to ad1 (the derivative). The stacked Jacobian is:
///
/// ```text
/// ∂η0[i]/∂β = c[i] · M[i,:]
/// ∂η1[i]/∂β = c[i] · M[i,:]
/// ∂ad1[i]/∂β = 0
/// ```
///
/// At g=0 (β=0 init): c=1, so each row is just M[i,:].
pub struct MarginalBlockJacobian {
    /// The marginal basis design (n × p_marginal), `Arc`-shared with its
    /// owner (see [`LogslopeBlockJacobian::design`]).
    design: Arc<Array2<f64>>,
}


impl MarginalBlockJacobian {
    pub fn new(design: impl Into<Arc<Array2<f64>>>) -> Self {
        Self {
            design: design.into(),
        }
    }
}


impl crate::custom_family::BlockEffectiveJacobian for MarginalBlockJacobian {
    fn effective_jacobian_rows(
        &self,
        state: &crate::custom_family::FamilyLinearizationState<'_>,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, String> {
        let n = self.design.nrows();
        let p = self.design.ncols();
        let rows = rows.start.min(n)..rows.end.min(n);
        let chunk = rows.end - rows.start;

        // c_i = sqrt(1 + (s * g_i)^2) depends on the logslope block's g at the
        // current beta.  This block does not own the logslope design so it cannot
        // compute c from beta alone.  Hard contract: when state.beta is non-empty
        // (post-init), family_scalars must carry SurvivalMarginalSlopeFamilyScalars
        // so the correct c_i is used.  At init (beta empty or all-zero), c_i = 1
        // exactly and family_scalars may be omitted.
        let scalars: Option<&SurvivalMarginalSlopeFamilyScalars> = state
            .family_scalars
            .as_ref()
            .and_then(|a| a.downcast_ref::<SurvivalMarginalSlopeFamilyScalars>());

        let beta_nonzero = state.beta.iter().any(|&b| b != 0.0);
        if beta_nonzero && scalars.is_none() {
            return Err("survival marginal-slope marginal block requires \
                 SurvivalMarginalSlopeFamilyScalars when beta != 0 (c_i != 1 in general); \
                 got family_scalars: None. The caller must populate per-row c_i via \
                 FamilyLinearizationState::family_scalars."
                .to_string());
        }

        let mut jac = Array2::<f64>::zeros((3 * chunk, p));

        for i in rows.clone() {
            let local_i = i - rows.start;
            let c = match scalars {
                Some(sc) => sc.c_i[i],
                // beta is all-zero here (enforced above), so g = 0 and c = 1.
                None => 1.0_f64,
            };
            for j in 0..p {
                let m_ij = c * self.design[[i, j]];
                jac[[local_i, j]] = m_ij;
                jac[[chunk + local_i, j]] = m_ij;
                // jac[[2*n + i, j]] = 0 -- ad1 row stays zero
            }
        }
        Ok(jac)
    }

    fn n_outputs(&self) -> usize {
        3
    }
}


/// n_outputs=3 stacked Jacobian for the time block.
///
/// The time block contributes separately to η0 (entry), η1 (exit), and ad1
/// (derivative) via three distinct design matrices. The stacked Jacobian is:
///
/// ```text
/// ∂η0[i]/∂β = c[i] · T_entry[i,:]
/// ∂η1[i]/∂β = c[i] · T_exit[i,:]
/// ∂ad1[i]/∂β = c[i] · T_deriv[i,:]
/// ```
///
/// At g=0 (β=0 init): c=1.
pub struct TimeBlockJacobian {
    // `Arc`-shared with their owners (see [`LogslopeBlockJacobian::design`]).
    design_entry: Arc<Array2<f64>>,
    design_exit: Arc<Array2<f64>>,
    design_deriv: Arc<Array2<f64>>,
}


impl TimeBlockJacobian {
    pub fn new(
        design_entry: impl Into<Arc<Array2<f64>>>,
        design_exit: impl Into<Arc<Array2<f64>>>,
        design_deriv: impl Into<Arc<Array2<f64>>>,
    ) -> Self {
        Self {
            design_entry: design_entry.into(),
            design_exit: design_exit.into(),
            design_deriv: design_deriv.into(),
        }
    }
}


impl crate::custom_family::BlockEffectiveJacobian for TimeBlockJacobian {
    fn effective_jacobian_rows(
        &self,
        state: &crate::custom_family::FamilyLinearizationState<'_>,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, String> {
        let n = self.design_entry.nrows();
        let p = self.design_entry.ncols();
        let rows = rows.start.min(n)..rows.end.min(n);
        let chunk = rows.end - rows.start;

        if self.design_exit.nrows() != n || self.design_deriv.nrows() != n {
            return Err(format!(
                "TimeBlockJacobian: design row count mismatch \
                 entry={n} exit={} deriv={}",
                self.design_exit.nrows(),
                self.design_deriv.nrows(),
            ));
        }
        if self.design_exit.ncols() != p || self.design_deriv.ncols() != p {
            return Err(format!(
                "TimeBlockJacobian: design col count mismatch \
                 entry={p} exit={} deriv={}",
                self.design_exit.ncols(),
                self.design_deriv.ncols(),
            ));
        }

        // c_i = sqrt(1 + (s * g_i)^2) depends on the logslope block's g.  This block
        // does not own the logslope design.  Hard contract: when beta is non-empty/nonzero,
        // family_scalars must carry SurvivalMarginalSlopeFamilyScalars with the correct c_i.
        // At init (beta empty or all-zero), c_i = 1 exactly.
        let scalars: Option<&SurvivalMarginalSlopeFamilyScalars> = state
            .family_scalars
            .as_ref()
            .and_then(|a| a.downcast_ref::<SurvivalMarginalSlopeFamilyScalars>());

        let beta_nonzero = state.beta.iter().any(|&b| b != 0.0);
        if beta_nonzero && scalars.is_none() {
            return Err("survival marginal-slope time block requires \
                 SurvivalMarginalSlopeFamilyScalars when beta != 0 (c_i != 1 in general); \
                 got family_scalars: None. The caller must populate per-row c_i via \
                 FamilyLinearizationState::family_scalars."
                .to_string());
        }

        let mut jac = Array2::<f64>::zeros((3 * chunk, p));

        for i in rows.clone() {
            let local_i = i - rows.start;
            let c = match scalars {
                Some(sc) => sc.c_i[i],
                // beta is all-zero here (enforced above), so g = 0 and c = 1.
                None => 1.0_f64,
            };
            for j in 0..p {
                jac[[local_i, j]] = c * self.design_entry[[i, j]];
                jac[[chunk + local_i, j]] = c * self.design_exit[[i, j]];
                jac[[2 * chunk + local_i, j]] = c * self.design_deriv[[i, j]];
            }
        }
        Ok(jac)
    }

    fn n_outputs(&self) -> usize {
        3
    }
}


// ── Full flex-chain Jacobian support ─────────────────────────────────
//
// When score_warp (β_h) and/or link_dev (β_w) are non-zero the rigid
// hyperbolic formula η = q·c + s·z·g is no longer exact.  The calibrated
// intercept `a` is an implicit function of all primary coordinates and the
// correct linearization is the IFT chain:
//
//   ∂η/∂u_k = chi · a_u[k] + rho[k]
//
// computed by `compute_survival_timepoint_exact`.  `chain_primary_to_joint_row`
// contracts these per-primary-coordinate derivatives with the block-level
// primary-to-coefficient maps to produce the joint-β Jacobian row.

/// Per-row IFT-corrected derivatives from the full flex chain.
///
/// Populated once per fit at the current β by a higher-level driver and
/// stored in `FamilyLinearizationState::family_scalars`.  When present, the
/// block Jacobian callbacks (`LogslopeBlockJacobian`, etc.) switch from the
/// rigid hyperbolic formula to the exact IFT chain.
///
/// Field layout:
/// - `eta_u_entry[i]`:  `∂η0/∂u_k` for primary slot `k=0..p_primary`, row `i`
/// - `eta_u_exit[i]`:   `∂η1/∂u_k`, row `i`
/// - `chi_exit[i]`:      observed χ₁ at exit (the density factor, > 0)
/// - `chi_u_exit[i]`:   `∂χ₁/∂u_k`, row `i`
/// - `d_exit[i]`:        normalisation D₁ at exit (> 0)
/// - `d_u_exit[i]`:     `∂D₁/∂u_k`, row `i`
/// - `q1_i[i]`:          exit-time probit argument q₁
/// - `qd1_i[i]`:         exit-time derivative qd₁
/// - `dq0_time[i]`:     `∂q0/∂β_time` row vector (length p_time)
/// - `dq1_time[i]`:     `∂q1/∂β_time` row vector (length p_time)
/// - `dqd1_time[i]`:    `∂qd1/∂β_time` row vector (length p_time)
/// - `dq0_marginal[i]`: `∂q0/∂β_marginal` row vector (length p_marginal)
/// - `dq1_marginal[i]`: `∂q1/∂β_marginal` row vector (length p_marginal)
/// - `dqd1_marginal[i]`:`∂qd1/∂β_marginal` row vector (length p_marginal)
/// - `logslope_design`:  the logslope design matrix (n × p_logslope)
/// - `score_warp_design`:score_warp design matrix (n × p_h), or empty
/// - `link_dev_design`:  link_dev design matrix (n × p_w), or empty
/// - `p_primary`:        total number of primary coordinates
/// - `p_q0`, `p_q1`, `p_qd1`, `p_g`: primary slot indices (scalar)
/// - `p_h_start`, `p_h_len`: h-slot range in primary
/// - `p_w_start`, `p_w_len`: w-slot range in primary
pub struct SurvivalFlexFamilyScalars {
    /// Shape n × p_primary; row i = ∂η0/∂u
    pub eta_u_entry: Array2<f64>,
    /// Shape n × p_primary; row i = ∂η1/∂u
    pub eta_u_exit: Array2<f64>,
    /// Length n; χ₁ at exit
    pub chi_exit: Vec<f64>,
    /// Shape n × p_primary; row i = ∂χ₁/∂u
    pub chi_u_exit: Array2<f64>,
    /// Length n; D₁ at exit
    pub d_exit: Vec<f64>,
    /// Shape n × p_primary; row i = ∂D₁/∂u
    pub d_u_exit: Array2<f64>,
    /// Length n; exit-time probit argument q₁
    pub q1_i: Vec<f64>,
    /// Length n; exit-time derivative qd₁
    pub qd1_i: Vec<f64>,
    /// ∂q0/∂β_time per row; shape n × p_time
    pub dq0_time: Array2<f64>,
    /// ∂q1/∂β_time per row; shape n × p_time
    pub dq1_time: Array2<f64>,
    /// ∂qd1/∂β_time per row; shape n × p_time
    pub dqd1_time: Array2<f64>,
    /// ∂q0/∂β_marginal per row; shape n × p_marginal
    pub dq0_marginal: Array2<f64>,
    /// ∂q1/∂β_marginal per row; shape n × p_marginal
    pub dq1_marginal: Array2<f64>,
    /// ∂qd1/∂β_marginal per row; shape n × p_marginal
    pub dqd1_marginal: Array2<f64>,
    /// Logslope design (n × p_logslope)
    pub logslope_design: Array2<f64>,
    /// Score-warp design (n × p_h); zero-column if inactive
    pub score_warp_design: Array2<f64>,
    /// Link-dev design (n × p_w); zero-column if inactive
    pub link_dev_design: Array2<f64>,
    /// Total primary coordinate count
    pub p_primary: usize,
    /// Primary slot index for q0 (= 0)
    pub idx_q0: usize,
    /// Primary slot index for q1 (= 1)
    pub idx_q1: usize,
    /// Primary slot index for qd1 (= 2)
    pub idx_qd1: usize,
    /// Primary slot index for g (= 3)
    pub idx_g: usize,
    /// Start of h-slots in primary coordinate vector
    pub h_start: usize,
    /// Number of h-slots (0 if score_warp inactive)
    pub h_len: usize,
    /// Start of w-slots in primary coordinate vector
    pub w_start: usize,
    /// Number of w-slots (0 if link_dev inactive)
    pub w_len: usize,
}


/// Contract a primary-coordinate derivative vector `coeff_u` (length
/// `p_primary`) with the block-level primary-to-coefficient maps, writing the
/// joint-β Jacobian row into `out` (length = `slices.total`).
///
/// # Channel routing
///
/// | Block slice       | Primary slots touched            | Map                  |
/// |-------------------|----------------------------------|----------------------|
/// | `slices.time`     | idx_q0, idx_q1, idx_qd1          | `q.dq*_time`         |
/// | `slices.marginal` | idx_q0, idx_q1, idx_qd1          | `q.dq*_marginal`     |
/// | `slices.logslope` | idx_g                            | `logslope_row`       |
/// | `slices.score_warp`| h-slots                         | identity             |
/// | `slices.link_dev` | w-slots                          | identity             |
///
/// At rigid β=0 with `beta_h = beta_w = 0` this reduces to:
/// - time: `chi * dq*_time + rho_q* * dq*_time` which, for the rigid formula
///   (rho_q = 0 for off-diagonal primaries), equals `c * dq*_time`
/// - logslope: `(q*c1 + s*z) * logslope_row`
///
/// # Arguments
/// - `coeff_u`: `ArrayView1<f64>` of length `p_primary`; the ∂·/∂u vector
/// - `q`: per-row primary chain maps (dq*_time, dq*_marginal)
/// - `logslope_row`: the logslope design row for this observation
/// - `flex`: the flex scalars (carries slot indices and p_primary)
/// - `slices`: the block-β offset layout
/// - `out`: output buffer of length `slices.total`; zeroed on entry by contract
pub fn chain_primary_to_joint_row(
    coeff_u: ndarray::ArrayView1<'_, f64>,
    q: &SurvivalFlexPerRowChain,
    logslope_row: ndarray::ArrayView1<'_, f64>,
    flex: &SurvivalFlexFamilyScalars,
    slices: &SurvivalBlockSlicesPublic,
    out: &mut [f64],
) {
    assert_eq!(
        coeff_u.len(),
        flex.p_primary,
        "chain_primary_to_joint_row: coeff_u len {} != p_primary {}",
        coeff_u.len(),
        flex.p_primary
    );
    assert_eq!(
        out.len(),
        slices.total,
        "chain_primary_to_joint_row: out len {} != slices.total {}",
        out.len(),
        slices.total
    );

    let cu_q0 = coeff_u[flex.idx_q0];
    let cu_q1 = coeff_u[flex.idx_q1];
    let cu_qd1 = coeff_u[flex.idx_qd1];
    let cu_g = coeff_u[flex.idx_g];

    // Time block: Σ_k (cu_q0 * dq0_time[k] + cu_q1 * dq1_time[k] + cu_qd1 * dqd1_time[k])
    for (k, dst) in out[slices.time.clone()].iter_mut().enumerate() {
        *dst = cu_q0 * q.dq0_time[k] + cu_q1 * q.dq1_time[k] + cu_qd1 * q.dqd1_time[k];
    }

    // Marginal block: same with marginal maps
    for (k, dst) in out[slices.marginal.clone()].iter_mut().enumerate() {
        *dst = cu_q0 * q.dq0_marginal[k] + cu_q1 * q.dq1_marginal[k] + cu_qd1 * q.dqd1_marginal[k];
    }

    // Logslope block: cu_g * logslope_row[k]
    for (k, dst) in out[slices.logslope.clone()].iter_mut().enumerate() {
        *dst = cu_g * logslope_row[k];
    }

    // Score-warp block: identity map through h-slots
    if let Some(sw_range) = slices.score_warp.as_ref() {
        for (local_k, dst) in out[sw_range.clone()].iter_mut().enumerate() {
            let primary_idx = flex.h_start + local_k;
            if primary_idx < flex.p_primary {
                *dst = coeff_u[primary_idx];
            }
        }
    }

    // Link-dev block: identity map through w-slots
    if let Some(ld_range) = slices.link_dev.as_ref() {
        for (local_k, dst) in out[ld_range.clone()].iter_mut().enumerate() {
            let primary_idx = flex.w_start + local_k;
            if primary_idx < flex.p_primary {
                *dst = coeff_u[primary_idx];
            }
        }
    }
}


/// Per-row primary chain maps extracted from `SurvivalMarginalSlopeDynamicRow`.
/// Passed by value (slice references) to `chain_primary_to_joint_row`.
pub struct SurvivalFlexPerRowChain<'a> {
    pub dq0_time: &'a [f64],
    pub dq1_time: &'a [f64],
    pub dqd1_time: &'a [f64],
    pub dq0_marginal: &'a [f64],
    pub dq1_marginal: &'a [f64],
    pub dqd1_marginal: &'a [f64],
}


/// Public mirror of the private `BlockSlices` for use in `chain_primary_to_joint_row`.
pub struct SurvivalBlockSlicesPublic {
    pub time: std::ops::Range<usize>,
    pub marginal: std::ops::Range<usize>,
    pub logslope: std::ops::Range<usize>,
    pub score_warp: Option<std::ops::Range<usize>>,
    pub link_dev: Option<std::ops::Range<usize>>,
    pub total: usize,
}


// ── 6-output flex-chain Jacobian for the logslope block ─────────────────────
//
// n_outputs = 6: η0, η1, log_chi1, log_D1, q1_log_phi_q1, log_qd1.
// When flex state is absent this is a hard error (must use 3-output rigid path
// LogslopeBlockJacobian instead). When flex state is present (SurvivalFlexFamilyScalars
// in family_scalars), uses the full IFT chain.

/// 6-output stacked Jacobian for the **logslope** block using the IFT flex chain.
///
/// Output ordering (for row `i`, outputs stacked as `[0..n], [n..2n], ..., [5n..6n]`):
///
/// - Row 0·n + i: `∂η0[i]/∂β_logslope` (entry probit index)
/// - Row 1·n + i: `∂η1[i]/∂β_logslope` (exit probit index)
/// - Row 2·n + i: `∂log_chi1[i]/∂β_logslope` = `chi_u_exit[i] / chi_exit[i]` chained
/// - Row 3·n + i: `∂log_D1[i]/∂β_logslope`   = `d_u_exit[i] / d_exit[i]` chained
/// - Row 4·n + i: `∂(q1·log_phi_q1)[i]/∂β_logslope` = `q1_i[i] * ∂q1/∂β_logslope`
///               (zero for logslope block since q1 does not depend on β_logslope)
/// - Row 5·n + i: `∂log_qd1[i]/∂β_logslope` = (1/qd1_i[i]) * ∂qd1/∂β_logslope
///               (zero for logslope block since qd1 does not depend on β_logslope)
///
/// When `family_scalars` is `None` (rigid path, no flex), returns `Err` — use
/// `LogslopeBlockJacobian` (3-output) for the rigid hyperbolic path instead.
pub struct LogslopeFlexBlockJacobian {
    /// Logslope design (n × p_logslope); shared with the block spec.
    design: Array2<f64>,
}


impl LogslopeFlexBlockJacobian {
    pub fn new(design: Array2<f64>) -> Self {
        Self { design }
    }
}


impl crate::custom_family::BlockEffectiveJacobian for LogslopeFlexBlockJacobian {
    fn effective_jacobian_rows(
        &self,
        state: &crate::custom_family::FamilyLinearizationState<'_>,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, String> {
        let flex: &SurvivalFlexFamilyScalars = state
            .family_scalars
            .as_ref()
            .and_then(|a| a.downcast_ref::<SurvivalFlexFamilyScalars>())
            .ok_or_else(|| {
                "LogslopeFlexBlockJacobian requires SurvivalFlexFamilyScalars in \
                 family_scalars (flex path); for the rigid path use LogslopeBlockJacobian"
                    .to_string()
            })?;

        let n = self.design.nrows();
        let p = self.design.ncols();
        let rows = rows.start.min(n)..rows.end.min(n);
        let chunk = rows.end - rows.start;
        if flex.eta_u_entry.nrows() != n || flex.eta_u_exit.nrows() != n {
            return Err(format!(
                "LogslopeFlexBlockJacobian: flex scalars have {} rows but design has {n}",
                flex.eta_u_entry.nrows()
            ));
        }

        // For the logslope block: ∂u_g/∂β_j = design[i,j]; all other primary
        // coords are zero (time, marginal, h, w do not depend on β_logslope).
        // So: ∂output/∂β_j = output_u[idx_g] * design[i,j].
        const N_OUT: usize = 6;
        let mut jac = Array2::<f64>::zeros((N_OUT * chunk, p));

        for i in rows.clone() {
            let local_i = i - rows.start;
            let g_idx = flex.idx_g;
            let cu_eta0_g = flex.eta_u_entry[[i, g_idx]];
            let cu_eta1_g = flex.eta_u_exit[[i, g_idx]];
            let chi1 = flex.chi_exit[i];
            let cu_logchi1_g = if chi1 > 0.0 && chi1.is_finite() {
                flex.chi_u_exit[[i, g_idx]] / chi1
            } else {
                0.0
            };
            let d1 = flex.d_exit[i];
            let cu_logd1_g = if d1 > 0.0 && d1.is_finite() {
                flex.d_u_exit[[i, g_idx]] / d1
            } else {
                0.0
            };
            // q1 and qd1 do not depend on β_logslope → rows 4 and 5 are zero.

            for j in 0..p {
                let x_ij = self.design[[i, j]];
                jac[[local_i, j]] = cu_eta0_g * x_ij;
                jac[[chunk + local_i, j]] = cu_eta1_g * x_ij;
                jac[[2 * chunk + local_i, j]] = cu_logchi1_g * x_ij;
                jac[[3 * chunk + local_i, j]] = cu_logd1_g * x_ij;
                // rows 4 and 5 remain zero
            }
        }
        Ok(jac)
    }

    fn n_outputs(&self) -> usize {
        6
    }
}


/// 6-output stacked Jacobian for the **marginal** block using the IFT flex chain.
///
/// For the marginal block:
/// - ∂u_{q0}/∂β_m = dq0_marginal[i,k]  (row of the chain map)
/// - ∂u_{q1}/∂β_m = dq1_marginal[i,k]
/// - ∂u_{qd1}/∂β_m = dqd1_marginal[i,k]
/// - all other primary slots: zero
///
/// Outputs: same 6 as `LogslopeFlexBlockJacobian`.
pub struct MarginalFlexBlockJacobian;


impl crate::custom_family::BlockEffectiveJacobian for MarginalFlexBlockJacobian {
    fn effective_jacobian_rows(
        &self,
        state: &crate::custom_family::FamilyLinearizationState<'_>,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, String> {
        let flex: &SurvivalFlexFamilyScalars = state
            .family_scalars
            .as_ref()
            .and_then(|a| a.downcast_ref::<SurvivalFlexFamilyScalars>())
            .ok_or_else(|| {
                "MarginalFlexBlockJacobian requires SurvivalFlexFamilyScalars in family_scalars"
                    .to_string()
            })?;

        let n = flex.dq0_marginal.nrows();
        let p = flex.dq0_marginal.ncols();
        let rows = rows.start.min(n)..rows.end.min(n);
        let chunk = rows.end - rows.start;
        if p == 0 {
            return Ok(Array2::<f64>::zeros((6 * chunk, 0)));
        }

        const N_OUT: usize = 6;
        let mut jac = Array2::<f64>::zeros((N_OUT * chunk, p));

        let q0_idx = flex.idx_q0;
        let q1_idx = flex.idx_q1;
        let qd1_idx = flex.idx_qd1;

        for i in rows.clone() {
            let local_i = i - rows.start;
            let chi1 = flex.chi_exit[i];
            let d1 = flex.d_exit[i];
            let q1 = flex.q1_i[i];
            let qd1 = flex.qd1_i[i];

            for k in 0..p {
                let dq0 = flex.dq0_marginal[[i, k]];
                let dq1 = flex.dq1_marginal[[i, k]];
                let dqd1 = flex.dqd1_marginal[[i, k]];

                // ∂η0/∂β_m[k] = Σ_u eta_u_entry[i,u] * ∂u/∂β_m[k]
                //             = eta_u_entry[q0] * dq0 + eta_u_entry[q1]*dq1 + eta_u_entry[qd1]*dqd1
                // (only q-primary slots are non-zero for the marginal block)
                let deta0 = flex.eta_u_entry[[i, q0_idx]] * dq0
                    + flex.eta_u_entry[[i, q1_idx]] * dq1
                    + flex.eta_u_entry[[i, qd1_idx]] * dqd1;
                let deta1 = flex.eta_u_exit[[i, q0_idx]] * dq0
                    + flex.eta_u_exit[[i, q1_idx]] * dq1
                    + flex.eta_u_exit[[i, qd1_idx]] * dqd1;
                let dlogchi1 = if chi1 > 0.0 && chi1.is_finite() {
                    (flex.chi_u_exit[[i, q0_idx]] * dq0
                        + flex.chi_u_exit[[i, q1_idx]] * dq1
                        + flex.chi_u_exit[[i, qd1_idx]] * dqd1)
                        / chi1
                } else {
                    0.0
                };
                let dlogd1 = if d1 > 0.0 && d1.is_finite() {
                    (flex.d_u_exit[[i, q0_idx]] * dq0
                        + flex.d_u_exit[[i, q1_idx]] * dq1
                        + flex.d_u_exit[[i, qd1_idx]] * dqd1)
                        / d1
                } else {
                    0.0
                };
                // ∂(log_phi(q1))/∂β_m[k] = -q1 * dq1  (d/dq1 of -0.5*q1^2 = -q1)
                let dq1_logphi = -q1 * dq1;
                // ∂log_qd1/∂β_m[k] = dqd1 / qd1
                let dlogqd1 = if qd1 > 0.0 { dqd1 / qd1 } else { 0.0 };

                jac[[local_i, k]] = deta0;
                jac[[chunk + local_i, k]] = deta1;
                jac[[2 * chunk + local_i, k]] = dlogchi1;
                jac[[3 * chunk + local_i, k]] = dlogd1;
                jac[[4 * chunk + local_i, k]] = dq1_logphi;
                jac[[5 * chunk + local_i, k]] = dlogqd1;
            }
        }
        Ok(jac)
    }

    fn n_outputs(&self) -> usize {
        6
    }
}


/// 6-output stacked Jacobian for the **time** block using the IFT flex chain.
///
/// For the time block:
/// - ∂u_{q0}/∂β_t = dq0_time[i,k]
/// - ∂u_{q1}/∂β_t = dq1_time[i,k]
/// - ∂u_{qd1}/∂β_t = dqd1_time[i,k]
///
/// Outputs: same 6 as above.
pub struct TimeFlexBlockJacobian;


impl crate::custom_family::BlockEffectiveJacobian for TimeFlexBlockJacobian {
    fn effective_jacobian_rows(
        &self,
        state: &crate::custom_family::FamilyLinearizationState<'_>,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, String> {
        let flex: &SurvivalFlexFamilyScalars = state
            .family_scalars
            .as_ref()
            .and_then(|a| a.downcast_ref::<SurvivalFlexFamilyScalars>())
            .ok_or_else(|| {
                "TimeFlexBlockJacobian requires SurvivalFlexFamilyScalars in family_scalars"
                    .to_string()
            })?;

        let n = flex.dq0_time.nrows();
        let p = flex.dq0_time.ncols();
        let rows = rows.start.min(n)..rows.end.min(n);
        let chunk = rows.end - rows.start;
        if p == 0 {
            return Ok(Array2::<f64>::zeros((6 * chunk, 0)));
        }

        const N_OUT: usize = 6;
        let mut jac = Array2::<f64>::zeros((N_OUT * chunk, p));

        let q0_idx = flex.idx_q0;
        let q1_idx = flex.idx_q1;
        let qd1_idx = flex.idx_qd1;

        for i in rows.clone() {
            let local_i = i - rows.start;
            let chi1 = flex.chi_exit[i];
            let d1 = flex.d_exit[i];
            let q1 = flex.q1_i[i];
            let qd1 = flex.qd1_i[i];

            for k in 0..p {
                let dq0 = flex.dq0_time[[i, k]];
                let dq1 = flex.dq1_time[[i, k]];
                let dqd1 = flex.dqd1_time[[i, k]];

                let deta0 = flex.eta_u_entry[[i, q0_idx]] * dq0
                    + flex.eta_u_entry[[i, q1_idx]] * dq1
                    + flex.eta_u_entry[[i, qd1_idx]] * dqd1;
                let deta1 = flex.eta_u_exit[[i, q0_idx]] * dq0
                    + flex.eta_u_exit[[i, q1_idx]] * dq1
                    + flex.eta_u_exit[[i, qd1_idx]] * dqd1;
                let dlogchi1 = if chi1 > 0.0 && chi1.is_finite() {
                    (flex.chi_u_exit[[i, q0_idx]] * dq0
                        + flex.chi_u_exit[[i, q1_idx]] * dq1
                        + flex.chi_u_exit[[i, qd1_idx]] * dqd1)
                        / chi1
                } else {
                    0.0
                };
                let dlogd1 = if d1 > 0.0 && d1.is_finite() {
                    (flex.d_u_exit[[i, q0_idx]] * dq0
                        + flex.d_u_exit[[i, q1_idx]] * dq1
                        + flex.d_u_exit[[i, qd1_idx]] * dqd1)
                        / d1
                } else {
                    0.0
                };
                // ∂(log_phi(q1))/∂β_t[k] = -q1 * dq1
                let dq1_logphi = -q1 * dq1;
                let dlogqd1 = if qd1 > 0.0 { dqd1 / qd1 } else { 0.0 };

                jac[[local_i, k]] = deta0;
                jac[[chunk + local_i, k]] = deta1;
                jac[[2 * chunk + local_i, k]] = dlogchi1;
                jac[[3 * chunk + local_i, k]] = dlogd1;
                jac[[4 * chunk + local_i, k]] = dq1_logphi;
                jac[[5 * chunk + local_i, k]] = dlogqd1;
            }
        }
        Ok(jac)
    }

    fn n_outputs(&self) -> usize {
        6
    }
}


/// 6-output stacked Jacobian for the **score_warp** block using the IFT flex chain.
///
/// For score_warp: ∂u_{h_j}/∂β_h[k] = δ_{j,k} (identity map in h-slots).
/// All other primary slots are zero.
///
/// Outputs: same 6 as above.
pub struct ScoreWarpFlexBlockJacobian;


impl crate::custom_family::BlockEffectiveJacobian for ScoreWarpFlexBlockJacobian {
    fn effective_jacobian_rows(
        &self,
        state: &crate::custom_family::FamilyLinearizationState<'_>,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, String> {
        let flex: &SurvivalFlexFamilyScalars = state
            .family_scalars
            .as_ref()
            .and_then(|a| a.downcast_ref::<SurvivalFlexFamilyScalars>())
            .ok_or_else(|| {
                "ScoreWarpFlexBlockJacobian requires SurvivalFlexFamilyScalars in family_scalars"
                    .to_string()
            })?;

        let n = flex.score_warp_design.nrows();
        let p = flex.score_warp_design.ncols();
        let rows = rows.start.min(n)..rows.end.min(n);
        let chunk = rows.end - rows.start;
        if p == 0 || flex.h_len == 0 {
            return Ok(Array2::<f64>::zeros((6 * chunk, p)));
        }
        if flex.h_len != p {
            return Err(format!(
                "ScoreWarpFlexBlockJacobian: h_len={} but score_warp_design has {} cols",
                flex.h_len, p
            ));
        }

        const N_OUT: usize = 6;
        let mut jac = Array2::<f64>::zeros((N_OUT * chunk, p));

        for i in rows.clone() {
            let local_i = i - rows.start;
            let chi1 = flex.chi_exit[i];
            let d1 = flex.d_exit[i];

            for k in 0..p {
                // h-slot k in primary coordinate space
                let h_idx = flex.h_start + k;
                if h_idx >= flex.p_primary {
                    continue;
                }
                // ∂η0/∂β_h[k] = eta_u_entry[i, h_idx]
                let deta0 = flex.eta_u_entry[[i, h_idx]];
                let deta1 = flex.eta_u_exit[[i, h_idx]];
                let dlogchi1 = if chi1 > 0.0 && chi1.is_finite() {
                    flex.chi_u_exit[[i, h_idx]] / chi1
                } else {
                    0.0
                };
                let dlogd1 = if d1 > 0.0 && d1.is_finite() {
                    flex.d_u_exit[[i, h_idx]] / d1
                } else {
                    0.0
                };
                // rows 4 and 5 (q1_logphi, log_qd1) are zero: q1, qd1 do not depend on β_h

                jac[[local_i, k]] = deta0;
                jac[[chunk + local_i, k]] = deta1;
                jac[[2 * chunk + local_i, k]] = dlogchi1;
                jac[[3 * chunk + local_i, k]] = dlogd1;
            }
        }
        Ok(jac)
    }

    fn n_outputs(&self) -> usize {
        6
    }
}


/// 6-output stacked Jacobian for the **link_dev** block using the IFT flex chain.
///
/// For link_dev: ∂u_{w_j}/∂β_w[k] = δ_{j,k} (identity map in w-slots).
///
/// Outputs: same 6 as above.
pub struct LinkDevFlexBlockJacobian;


impl crate::custom_family::BlockEffectiveJacobian for LinkDevFlexBlockJacobian {
    fn effective_jacobian_rows(
        &self,
        state: &crate::custom_family::FamilyLinearizationState<'_>,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, String> {
        let flex: &SurvivalFlexFamilyScalars = state
            .family_scalars
            .as_ref()
            .and_then(|a| a.downcast_ref::<SurvivalFlexFamilyScalars>())
            .ok_or_else(|| {
                "LinkDevFlexBlockJacobian requires SurvivalFlexFamilyScalars in family_scalars"
                    .to_string()
            })?;

        let n = flex.link_dev_design.nrows();
        let p = flex.link_dev_design.ncols();
        let rows = rows.start.min(n)..rows.end.min(n);
        let chunk = rows.end - rows.start;
        if p == 0 || flex.w_len == 0 {
            return Ok(Array2::<f64>::zeros((6 * chunk, p)));
        }
        if flex.w_len != p {
            return Err(format!(
                "LinkDevFlexBlockJacobian: w_len={} but link_dev_design has {} cols",
                flex.w_len, p
            ));
        }

        const N_OUT: usize = 6;
        let mut jac = Array2::<f64>::zeros((N_OUT * chunk, p));

        for i in rows.clone() {
            let local_i = i - rows.start;
            let chi1 = flex.chi_exit[i];
            let d1 = flex.d_exit[i];

            for k in 0..p {
                let w_idx = flex.w_start + k;
                if w_idx >= flex.p_primary {
                    continue;
                }
                let deta0 = flex.eta_u_entry[[i, w_idx]];
                let deta1 = flex.eta_u_exit[[i, w_idx]];
                let dlogchi1 = if chi1 > 0.0 && chi1.is_finite() {
                    flex.chi_u_exit[[i, w_idx]] / chi1
                } else {
                    0.0
                };
                let dlogd1 = if d1 > 0.0 && d1.is_finite() {
                    flex.d_u_exit[[i, w_idx]] / d1
                } else {
                    0.0
                };

                jac[[local_i, k]] = deta0;
                jac[[chunk + local_i, k]] = deta1;
                jac[[2 * chunk + local_i, k]] = dlogchi1;
                jac[[3 * chunk + local_i, k]] = dlogd1;
            }
        }
        Ok(jac)
    }

    fn n_outputs(&self) -> usize {
        6
    }
}


// ── Timewiggle-active Jacobians ───────────────────────────────────────
//
// When timewiggle is active, (q0, q1, qd1) are nonlinear functions of
// (β_time, β_marginal) through the composition:
//
//   h0 = X_entry_base[i] · β_t_base + offset_entry[i] + M[i] · β_m
//   q0 = h0 + B(h0) · β_tw           (B = monotone wiggle basis)
//
// and analogously for q1 and qd1.  The chain rule gives:
//
//   ∂q0/∂β_t[j < p_base] = (1 + B'(h0)·β_tw) · X_entry[i,j]
//                         = dq_dq0(h0) · X_entry[i,j]
//   ∂q0/∂β_t[p_base + k] = B_k(h0)
//   ∂q0/∂β_m[j]          = dq_dq0(h0) · M[i,j]
//
// Since η_r = c · q_r + … and ∂η_r/∂β_block = c · ∂q_r/∂β_block,
// the stacked Jacobian for each block is:
//
//   J[i,       j] = c_i · ∂q0/∂β_block[j]
//   J[n + i,   j] = c_i · ∂q1/∂β_block[j]
//   J[2*n + i, j] = c_i · ∂qd1/∂β_block[j]
//
// where c_i = sqrt(1 + (s · g_i)²) and g_i = G[i] · β_g.
//
// At β = 0: dq_dq0 = 1, d²q/dh² = 0, c_i = 1, so both timewiggle
// callbacks reduce to the rigid-path `TimeBlockJacobian` /
// `MarginalBlockJacobian` values.
//
// Joint β layout (same for both callbacks):
//   [β_t (p_time) | β_m (p_m) | β_g (p_g) | …]
//
// p_time = p_base + p_tw where p_tw = time_wiggle_ncols.

/// n_outputs = 3 stacked Jacobian for the **time** block when timewiggle
/// is active.  Computes `c_i` from the embedded logslope design and
/// joint β, so no `family_scalars` are required.
pub struct SmsTimewiggleTimeJacobian {
    design_entry: Arc<Array2<f64>>,
    design_exit: Arc<Array2<f64>>,
    design_deriv: Arc<Array2<f64>>,
    design_marginal: Arc<Array2<f64>>,
    design_logslope: Arc<Array2<f64>>,
    offset_entry: Arc<Array1<f64>>,
    offset_exit: Arc<Array1<f64>>,
    offset_deriv: Arc<Array1<f64>>,
    /// Fixed marginal-predictor offset. The full marginal predictor entering
    /// the entry/exit channels is `design_marginal·β_m + marginal_offset`
    /// (see `row_dynamic_q_values`); this is the β-independent part.
    marginal_offset: Arc<Array1<f64>>,
    time_wiggle_knots: Array1<f64>,
    time_wiggle_degree: usize,
    /// Full time block width (= design_entry.ncols()).
    p_time: usize,
    /// Wiggle tail width.
    p_tw: usize,
    /// Marginal block width (for joint β parsing).
    p_m: usize,
    /// Logslope block width (for joint β parsing).
    p_g: usize,
    /// Probit frailty scale s.
    probit_scale: f64,
}


impl SmsTimewiggleTimeJacobian {
    /// Construct.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        design_entry: Arc<Array2<f64>>,
        design_exit: Arc<Array2<f64>>,
        design_deriv: Arc<Array2<f64>>,
        design_marginal: Arc<Array2<f64>>,
        design_logslope: Arc<Array2<f64>>,
        offset_entry: Arc<Array1<f64>>,
        offset_exit: Arc<Array1<f64>>,
        offset_deriv: Arc<Array1<f64>>,
        marginal_offset: Arc<Array1<f64>>,
        time_wiggle_knots: Array1<f64>,
        time_wiggle_degree: usize,
        p_tw: usize,
        p_m: usize,
        p_g: usize,
        probit_scale: f64,
    ) -> Self {
        let p_time = design_entry.ncols();
        Self {
            design_entry,
            design_exit,
            design_deriv,
            design_marginal,
            design_logslope,
            offset_entry,
            offset_exit,
            offset_deriv,
            marginal_offset,
            time_wiggle_knots,
            time_wiggle_degree,
            p_time,
            p_tw,
            p_m,
            p_g,
            probit_scale,
        }
    }
}


impl crate::custom_family::BlockEffectiveJacobian for SmsTimewiggleTimeJacobian {
    fn effective_jacobian_rows(
        &self,
        state: &crate::custom_family::FamilyLinearizationState<'_>,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, String> {
        let n = self.design_entry.nrows();
        let p = self.p_time;
        let rows = rows.start.min(n)..rows.end.min(n);
        let chunk = rows.end - rows.start;
        let p_base = p.saturating_sub(self.p_tw);

        let beta = state.beta;
        // β_t = joint β[0 .. p_time]
        let beta_t = if beta.len() >= p { &beta[..p] } else { beta };
        let beta_t_base = &beta_t[..p_base.min(beta_t.len())];
        // β_tw must always be a length-`p_tw` vector. The timewiggle block
        // exists whenever `self.p_tw > 0`, independent of how many coefficients
        // the caller supplied: the identifiability canonicaliser calls this at
        // the β=0 linearisation point with `beta = &[]` (see
        // `BlockJacobianAsRowOp::from_callback`), so inferring "no wiggle block"
        // from an empty slice — the old behaviour — wrongly drove `beta_tw`
        // empty, made `sms_tw_first_order_geom` return `None`, and zeroed the
        // wiggle tail columns. That made the time block look structurally
        // aliased ("block 0 fully aliased") even though ∂q/∂β_tw[j] = B_j(h) ≠ 0
        // at β=0. Zero-pad to `self.p_tw` so the basis is always evaluated.
        let zero_tw: Vec<f64>;
        let beta_tw: &[f64] = if beta_t.len() >= p_base + self.p_tw {
            &beta_t[p_base..p_base + self.p_tw]
        } else {
            zero_tw = vec![0.0; self.p_tw];
            &zero_tw
        };
        // β_m = joint β[p_time .. p_time + p_m]
        let beta_m = {
            let s = p;
            let e = (s + self.p_m).min(beta.len());
            if e > s { &beta[s..e] } else { &[][..] }
        };
        // β_g = joint β[p_time + p_m .. p_time + p_m + p_g]
        let beta_g = {
            let s = p + self.p_m;
            let e = (s + self.p_g).min(beta.len());
            if e > s { &beta[s..e] } else { &[][..] }
        };

        let sc = self.probit_scale;
        let knots = &self.time_wiggle_knots;
        let degree = self.time_wiggle_degree;

        let mut jac = Array2::<f64>::zeros((3 * chunk, p));

        for i in rows.clone() {
            let local_i = i - rows.start;
            // c_i computed directly from logslope design and joint β_g.
            let g_i: f64 = beta_g
                .iter()
                .enumerate()
                .filter(|&(j, _)| j < self.design_logslope.ncols())
                .map(|(j, &b)| self.design_logslope[[i, j]] * b)
                .sum();
            let c_i = (1.0_f64 + (sc * g_i).powi(2)).sqrt();

            // Base marginal η contribution.
            let eta_m: f64 = beta_m
                .iter()
                .enumerate()
                .filter(|&(j, _)| j < self.design_marginal.ncols())
                .map(|(j, &b)| self.design_marginal[[i, j]] * b)
                .sum();

            // The marginal predictor (coefficient part `eta_m` plus the fixed
            // `marginal_offset`) enters BOTH entry and exit channels but NOT
            // the derivative channel — see `row_dynamic_q_values`.
            let h0: f64 = self.offset_entry[i]
                + eta_m
                + self.marginal_offset[i]
                + (0..p_base.min(beta_t_base.len()).min(self.design_entry.ncols()))
                    .map(|j| self.design_entry[[i, j]] * beta_t_base[j])
                    .sum::<f64>();
            let h1: f64 = self.offset_exit[i]
                + eta_m
                + self.marginal_offset[i]
                + (0..p_base.min(beta_t_base.len()).min(self.design_exit.ncols()))
                    .map(|j| self.design_exit[[i, j]] * beta_t_base[j])
                    .sum::<f64>();
            let d_raw: f64 = self.offset_deriv[i]
                + (0..p_base.min(beta_t_base.len()).min(self.design_deriv.ncols()))
                    .map(|j| self.design_deriv[[i, j]] * beta_t_base[j])
                    .sum::<f64>();

            let beta_tw_view = ndarray::ArrayView1::from(beta_tw);
            let eg = sms_tw_first_order_geom(
                ndarray::ArrayView1::from(&[h0][..]),
                beta_tw_view,
                knots,
                degree,
            )?;
            let xg = sms_tw_first_order_geom(
                ndarray::ArrayView1::from(&[h1][..]),
                beta_tw_view,
                knots,
                degree,
            )?;

            let (entry_dq, exit_dq, exit_d2q, entry_basis, exit_basis, exit_basis_d1) =
                match (eg, xg) {
                    (Some(eg), Some(xg)) => (
                        eg.dq_dq0[0],
                        xg.dq_dq0[0],
                        xg.d2q_dq02[0],
                        Some(eg.basis),
                        Some(xg.basis),
                        Some(xg.basis_d1),
                    ),
                    _ => (1.0_f64, 1.0_f64, 0.0_f64, None, None, None),
                };

            // Base columns j < p_base.
            for j in 0..p_base.min(self.design_entry.ncols()) {
                let xe = self.design_entry[[i, j]];
                let xx = self.design_exit[[i, j]];
                let xd = self.design_deriv[[i, j]];
                jac[[local_i, j]] = c_i * entry_dq * xe;
                jac[[chunk + local_i, j]] = c_i * exit_dq * xx;
                jac[[2 * chunk + local_i, j]] = c_i * (exit_d2q * d_raw * xx + exit_dq * xd);
            }

            // Wiggle tail columns.
            for local_idx in 0..self.p_tw {
                let col = p_base + local_idx;
                let b0 = entry_basis.as_ref().map_or(0.0, |b| b[[0, local_idx]]);
                let b1 = exit_basis.as_ref().map_or(0.0, |b| b[[0, local_idx]]);
                let bd1 = exit_basis_d1.as_ref().map_or(0.0, |b| b[[0, local_idx]]);
                jac[[local_i, col]] = c_i * b0;
                jac[[chunk + local_i, col]] = c_i * b1;
                jac[[2 * chunk + local_i, col]] = c_i * bd1 * d_raw;
            }
        }
        Ok(jac)
    }

    fn n_outputs(&self) -> usize {
        3
    }
}


/// n_outputs = 3 stacked Jacobian for the **marginal** block when timewiggle
/// is active.
pub struct SmsTimewiggleMarginalJacobian {
    design_entry: Arc<Array2<f64>>,
    design_exit: Arc<Array2<f64>>,
    design_deriv: Arc<Array2<f64>>,
    design_marginal: Arc<Array2<f64>>,
    design_logslope: Arc<Array2<f64>>,
    offset_entry: Arc<Array1<f64>>,
    offset_exit: Arc<Array1<f64>>,
    offset_deriv: Arc<Array1<f64>>,
    /// Fixed marginal-predictor offset (β-independent part of the marginal
    /// predictor entering the entry/exit channels; see `row_dynamic_q_values`).
    marginal_offset: Arc<Array1<f64>>,
    time_wiggle_knots: Array1<f64>,
    time_wiggle_degree: usize,
    p_time: usize,
    p_tw: usize,
    p_g: usize,
    probit_scale: f64,
}


impl SmsTimewiggleMarginalJacobian {
    /// Construct.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        design_entry: Arc<Array2<f64>>,
        design_exit: Arc<Array2<f64>>,
        design_deriv: Arc<Array2<f64>>,
        design_marginal: Arc<Array2<f64>>,
        design_logslope: Arc<Array2<f64>>,
        offset_entry: Arc<Array1<f64>>,
        offset_exit: Arc<Array1<f64>>,
        offset_deriv: Arc<Array1<f64>>,
        marginal_offset: Arc<Array1<f64>>,
        time_wiggle_knots: Array1<f64>,
        time_wiggle_degree: usize,
        p_time: usize,
        p_tw: usize,
        p_g: usize,
        probit_scale: f64,
    ) -> Self {
        Self {
            design_entry,
            design_exit,
            design_deriv,
            design_marginal,
            design_logslope,
            offset_entry,
            offset_exit,
            offset_deriv,
            marginal_offset,
            time_wiggle_knots,
            time_wiggle_degree,
            p_time,
            p_tw,
            p_g,
            probit_scale,
        }
    }
}


impl crate::custom_family::BlockEffectiveJacobian for SmsTimewiggleMarginalJacobian {
    fn effective_jacobian_rows(
        &self,
        state: &crate::custom_family::FamilyLinearizationState<'_>,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, String> {
        let n = self.design_marginal.nrows();
        let p_m = self.design_marginal.ncols();
        let rows = rows.start.min(n)..rows.end.min(n);
        let chunk = rows.end - rows.start;
        let p_t = self.p_time;
        let p_base = p_t.saturating_sub(self.p_tw);

        let beta = state.beta;
        let beta_t = if beta.len() >= p_t {
            &beta[..p_t]
        } else {
            beta
        };
        let beta_t_base = &beta_t[..p_base.min(beta_t.len())];
        let beta_tw = if beta_t.len() > p_base {
            &beta_t[p_base..]
        } else {
            &[][..]
        };
        let beta_m = {
            let s = p_t;
            let e = (s + p_m).min(beta.len());
            if e > s { &beta[s..e] } else { &[][..] }
        };
        let beta_g = {
            let s = p_t + p_m;
            let e = (s + self.p_g).min(beta.len());
            if e > s { &beta[s..e] } else { &[][..] }
        };

        let sc = self.probit_scale;
        let knots = &self.time_wiggle_knots;
        let degree = self.time_wiggle_degree;

        let mut jac = Array2::<f64>::zeros((3 * chunk, p_m));

        for i in rows.clone() {
            let local_i = i - rows.start;
            let g_i: f64 = beta_g
                .iter()
                .enumerate()
                .filter(|&(j, _)| j < self.design_logslope.ncols())
                .map(|(j, &b)| self.design_logslope[[i, j]] * b)
                .sum();
            let c_i = (1.0_f64 + (sc * g_i).powi(2)).sqrt();

            let eta_m: f64 = beta_m
                .iter()
                .enumerate()
                .filter(|&(j, _)| j < p_m)
                .map(|(j, &b)| self.design_marginal[[i, j]] * b)
                .sum();

            // Marginal predictor (eta_m + fixed marginal_offset) enters entry
            // and exit channels alike (see `row_dynamic_q_values`).
            let h0: f64 = self.offset_entry[i]
                + eta_m
                + self.marginal_offset[i]
                + (0..p_base.min(beta_t_base.len()).min(self.design_entry.ncols()))
                    .map(|j| self.design_entry[[i, j]] * beta_t_base[j])
                    .sum::<f64>();
            let h1: f64 = self.offset_exit[i]
                + eta_m
                + self.marginal_offset[i]
                + (0..p_base.min(beta_t_base.len()).min(self.design_exit.ncols()))
                    .map(|j| self.design_exit[[i, j]] * beta_t_base[j])
                    .sum::<f64>();
            let d_raw: f64 = self.offset_deriv[i]
                + (0..p_base.min(beta_t_base.len()).min(self.design_deriv.ncols()))
                    .map(|j| self.design_deriv[[i, j]] * beta_t_base[j])
                    .sum::<f64>();

            let beta_tw_view = ndarray::ArrayView1::from(beta_tw);
            let eg = sms_tw_first_order_geom(
                ndarray::ArrayView1::from(&[h0][..]),
                beta_tw_view,
                knots,
                degree,
            )?;
            let xg = sms_tw_first_order_geom(
                ndarray::ArrayView1::from(&[h1][..]),
                beta_tw_view,
                knots,
                degree,
            )?;

            let (entry_dq, exit_dq, exit_d2q) = match (eg, xg) {
                (Some(eg), Some(xg)) => (eg.dq_dq0[0], xg.dq_dq0[0], xg.d2q_dq02[0]),
                _ => (1.0_f64, 1.0_f64, 0.0_f64),
            };

            for j in 0..p_m {
                let m_ij = self.design_marginal[[i, j]];
                jac[[local_i, j]] = c_i * entry_dq * m_ij;
                jac[[chunk + local_i, j]] = c_i * exit_dq * m_ij;
                jac[[2 * chunk + local_i, j]] = c_i * exit_d2q * d_raw * m_ij;
            }
        }
        Ok(jac)
    }

    fn n_outputs(&self) -> usize {
        3
    }
}


/// Compute timewiggle first-order geometry at a single evaluation point `h0`.
///
/// Returns `Ok(None)` when `beta_tw` is empty (no active wiggle columns).
/// This is a free-function mirror of
/// `SurvivalMarginalSlopeFamily::time_wiggle_first_order_geometry` for use in
/// `BlockEffectiveJacobian` impls that do not hold a family reference.
fn sms_tw_first_order_geom(
    h0: ndarray::ArrayView1<'_, f64>,
    beta_tw: ndarray::ArrayView1<'_, f64>,
    knots: &Array1<f64>,
    degree: usize,
) -> Result<Option<SurvivalTimeWiggleFirstOrderGeometry>, String> {
    if beta_tw.is_empty() {
        return Ok(None);
    }
    let basis = monotone_wiggle_basis_with_derivative_order(h0, knots, degree, 0)?;
    let basis_d1 = monotone_wiggle_basis_with_derivative_order(h0, knots, degree, 1)?;
    let basis_d2 = monotone_wiggle_basis_with_derivative_order(h0, knots, degree, 2)?;
    if basis.ncols() != beta_tw.len()
        || basis_d1.ncols() != beta_tw.len()
        || basis_d2.ncols() != beta_tw.len()
    {
        return Err(format!(
            "sms_tw_first_order_geom: basis/beta_tw width mismatch \
             B/B'/B''={}/{}/{} beta_tw={}",
            basis.ncols(),
            basis_d1.ncols(),
            basis_d2.ncols(),
            beta_tw.len(),
        ));
    }
    let dq_dq0 = fast_av(&basis_d1, &beta_tw) + 1.0;
    let d2q_dq02 = fast_av(&basis_d2, &beta_tw);
    Ok(Some(SurvivalTimeWiggleFirstOrderGeometry {
        basis,
        basis_d1,
        basis_d2,
        dq_dq0,
        d2q_dq02,
    }))
}


/// Overwrite the timewiggle tail columns of the dense time-channel
/// primary-Jacobian slots with their analytic value at the β=0 pilot primary
/// state, in place.
///
/// When `timewiggle(...)` is active the workflow disables the base time basis
/// and appends the wiggle coefficient slots as **zero placeholder** columns
/// (the design width is a coefficient-layout convention, not the Jacobian).
/// Feeding those zeros into the identifiability compiler makes the whole time
/// block look structurally zero ("block 0 fully aliased") — a bad-Jacobian
/// artifact, not a real alias. The true primary-channel derivative of the time
/// channels w.r.t. wiggle coefficient `j`, at the linearisation point β=0, is
/// the monotone-wiggle basis value at the pilot coordinate:
///
/// ```text
///   ∂q0  / ∂β_tw[j] = B_j(h0)          h0    = q0_pilot  = offset_entry
///   ∂q1  / ∂β_tw[j] = B_j(h1)          h1    = q1_pilot  = offset_exit + marginal_offset
///   ∂qd1 / ∂β_tw[j] = B'_j(h1)·d_raw   d_raw = qd1_pilot = derivative_offset_exit
/// ```
///
/// Base columns (`j < p_base`) reduce to the raw design at β=0 (dq/dh = 1,
/// d²q/dh² = 0) and are left untouched. The logslope-curvature factor `c_i` is
/// **not** applied here; it enters the Fisher Gram through
/// [`SurvivalRowHessian`], matching the raw-design convention for base columns.
fn overwrite_timewiggle_time_slots_at_pilot(
    dq0: &mut Array2<f64>,
    dq1: &mut Array2<f64>,
    dqd1: &mut Array2<f64>,
    timewiggle: &TimeWiggleBlockInput,
    h0: &Array1<f64>,
    h1: &Array1<f64>,
    d_raw: &Array1<f64>,
) -> Result<(), String> {
    let p_tw = timewiggle.ncols;
    if p_tw == 0 {
        return Ok(());
    }
    let p_time = dq0.ncols();
    let p_base = p_time.saturating_sub(p_tw);
    let n = dq0.nrows();
    let knots = &timewiggle.knots;
    let degree = timewiggle.degree;
    let b0 = monotone_wiggle_basis_with_derivative_order(h0.view(), knots, degree, 0)?;
    let b1 = monotone_wiggle_basis_with_derivative_order(h1.view(), knots, degree, 0)?;
    let b1d = monotone_wiggle_basis_with_derivative_order(h1.view(), knots, degree, 1)?;
    if b0.ncols() != p_tw || b1.ncols() != p_tw || b1d.ncols() != p_tw {
        return Err(format!(
            "overwrite_timewiggle_time_slots_at_pilot: basis width B/B/B'={}/{}/{} != p_tw={p_tw}",
            b0.ncols(),
            b1.ncols(),
            b1d.ncols(),
        ));
    }
    for i in 0..n {
        for j in 0..p_tw {
            let col = p_base + j;
            dq0[[i, col]] = b0[[i, j]];
            dq1[[i, col]] = b1[[i, j]];
            dqd1[[i, col]] = b1d[[i, j]] * d_raw[i];
        }
    }
    Ok(())
}


// ── Building block specs ──────────────────────────────────────────────

fn build_time_blockspec(
    time_block: &TimeBlockInput,
    design_exit: &DesignMatrix,
    rho: Array1<f64>,
    beta_hint: Option<Array1<f64>>,
) -> ParameterBlockSpec {
    // Share the three dense design matrices with the multi-output Jacobian
    // via `Arc` — `try_to_dense_arc` is zero-copy for materialized designs,
    // so the callback retains no duplicate `n × p` storage. Falls back to no
    // callback if densification fails.
    let jac_cb: Option<Arc<dyn crate::custom_family::BlockEffectiveJacobian>> = (|| {
        let d_entry = time_block
            .design_entry
            .try_to_dense_arc("build_time_blockspec::entry")
            .ok()?;
        let d_exit = design_exit
            .try_to_dense_arc("build_time_blockspec::exit")
            .ok()?;
        let d_deriv = time_block
            .design_derivative_exit
            .try_to_dense_arc("build_time_blockspec::deriv")
            .ok()?;
        if d_entry.dim() != d_exit.dim() || d_entry.dim() != d_deriv.dim() {
            return None;
        }
        Some(Arc::new(TimeBlockJacobian::new(d_entry, d_exit, d_deriv))
            as Arc<dyn crate::custom_family::BlockEffectiveJacobian>)
    })();

    ParameterBlockSpec {
        name: "time_surface".to_string(),
        design: design_exit.clone(),
        offset: Array1::zeros(design_exit.nrows()),
        penalties: time_block
            .penalties
            .iter()
            .cloned()
            .map(PenaltyMatrix::Dense)
            .collect(),
        nullspace_dims: time_block.nullspace_dims.clone(),
        initial_log_lambdas: rho,
        initial_beta: beta_hint,
        gauge_priority: 200,
        jacobian_callback: jac_cb,
        stacked_design: None,
        stacked_offset: None,
    }
}


fn build_logslope_blockspec(
    design: &TermCollectionDesign,
    baseline: f64,
    offset: &Array1<f64>,
    rho: Array1<f64>,
    beta_hint: Option<Array1<f64>>,
    z_scaling: Arc<[f64]>,
    probit_scale: f64,
) -> ParameterBlockSpec {
    let z_vec = z_scaling.to_vec();
    let jac_cb: Option<Arc<dyn crate::custom_family::BlockEffectiveJacobian>> = design
        .design
        .try_to_dense_arc("build_logslope_blockspec")
        .ok()
        .map(|d| {
            Arc::new(LogslopeBlockJacobian::new(d, z_vec, probit_scale))
                as Arc<dyn crate::custom_family::BlockEffectiveJacobian>
        });

    ParameterBlockSpec {
        name: "logslope_surface".to_string(),
        design: design.design.clone(),
        offset: offset + baseline,
        penalties: design.penalties_as_penalty_matrix(),
        nullspace_dims: design.nullspace_dims.clone(),
        initial_log_lambdas: rho,
        initial_beta: beta_hint,
        gauge_priority: 120,
        jacobian_callback: jac_cb,
        stacked_design: None,
        stacked_offset: None,
    }
}


fn build_marginal_blockspec(
    design: &TermCollectionDesign,
    offset: &Array1<f64>,
    rho: Array1<f64>,
    beta_hint: Option<Array1<f64>>,
) -> ParameterBlockSpec {
    let jac_cb: Option<Arc<dyn crate::custom_family::BlockEffectiveJacobian>> = design
        .design
        .try_to_dense_arc("build_marginal_blockspec")
        .ok()
        .map(|d| {
            Arc::new(MarginalBlockJacobian::new(d))
                as Arc<dyn crate::custom_family::BlockEffectiveJacobian>
        });

    ParameterBlockSpec {
        name: "marginal_surface".to_string(),
        design: design.design.clone(),
        offset: offset.clone(),
        penalties: design.penalties_as_penalty_matrix(),
        nullspace_dims: design.nullspace_dims.clone(),
        initial_log_lambdas: rho,
        initial_beta: beta_hint,
        gauge_priority: 150,
        jacobian_callback: jac_cb,
        stacked_design: None,
        stacked_offset: None,
    }
}


fn inner_fit(
    family: &SurvivalMarginalSlopeFamily,
    blocks: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
) -> Result<UnifiedFitResult, String> {
    fit_custom_family(family, blocks, options).map_err(|e| e.to_string())
}


/// Marginal-slope guard policy: the guard is required to be strictly positive
/// (`q'(t) ≥ guard > 0`), because the row-wise representation here is the *only*
/// place the monotonicity barrier lives — a zero guard would silently collapse
/// it. Coefficient-free row feasibility uses the family's epsilon-scaled slack
/// (`survival_derivative_guard_tolerance`).
const MARGINAL_SLOPE_GUARD_POLICY: GuardConstraintPolicy = GuardConstraintPolicy {
    guard_policy: GuardPolicy::Positive,
    feasibility: FeasibilityTolerance::EpsilonScaled,
};


fn time_derivative_guard_constraints(
    design_derivative_exit: &DesignMatrix,
    derivative_offset_exit: &Array1<f64>,
    derivative_guard: f64,
) -> Result<Option<LinearInequalityConstraints>, String> {
    build_time_derivative_guard_constraints(
        design_derivative_exit,
        derivative_offset_exit,
        derivative_guard,
        MARGINAL_SLOPE_GUARD_POLICY,
    )
    .map_err(map_guard_constraint_failure)
}


/// Render a shared guard-constraint failure into the marginal-slope error
/// vocabulary, preserving the family's historical wording.
fn map_guard_constraint_failure(failure: GuardConstraintFailure) -> String {
    match failure {
        GuardConstraintFailure::RowOffsetMismatch { rows, offsets } => {
            SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival marginal-slope derivative guard constraints require matching rows/offsets: rows={rows}, offsets={offsets}"
                ),
            }
            .into()
        }
        GuardConstraintFailure::GuardOutOfRange { guard, range } => {
            SurvivalMarginalSlopeError::InvalidInput {
                reason: format!(
                    "survival marginal-slope derivative guard must be finite and {range}, got {guard}"
                ),
            }
            .into()
        }
        GuardConstraintFailure::NonFiniteOffset { row, offset } => {
            SurvivalMarginalSlopeError::MonotonicityViolation {
                reason: format!(
                    "survival marginal-slope derivative guard constraints require finite derivative offsets; found offset[{row}]={offset}"
                ),
            }
            .into()
        }
        GuardConstraintFailure::NonFiniteDesign { row, col } => {
            SurvivalMarginalSlopeError::MonotonicityViolation {
                reason: format!(
                    "survival marginal-slope derivative guard constraints require finite derivative design entries; found row {row}, column {col}"
                ),
            }
            .into()
        }
        GuardConstraintFailure::InfeasibleRow {
            row,
            offset,
            guard,
            no_time_coefficients,
        } => {
            let reason = if no_time_coefficients {
                format!(
                    "survival marginal-slope derivative guard is infeasible at row {row}: offset={offset:.3e} < guard={guard:.3e} with no time coefficients"
                )
            } else {
                format!(
                    "survival marginal-slope derivative guard is infeasible at row {row}: zero derivative design row with offset={offset:.3e} < guard={guard:.3e}"
                )
            };
            SurvivalMarginalSlopeError::MonotonicityViolation { reason }.into()
        }
    }
}


fn append_timewiggle_tail_nonnegative_constraints(
    base: Option<LinearInequalityConstraints>,
    p_total: usize,
    time_wiggle_ncols: usize,
) -> Result<Option<LinearInequalityConstraints>, String> {
    let p_wiggle = time_wiggle_ncols.min(p_total);
    if p_wiggle == 0 {
        return Ok(base);
    }
    if let Some(base_constraints) = base.as_ref() {
        if base_constraints.a.ncols() != p_total {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival marginal-slope time constraint width mismatch: constraints={}, time block={p_total}",
                    base_constraints.a.ncols()
                ),
            }
            .into());
        }
        if base_constraints.a.nrows() != base_constraints.b.len() {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival marginal-slope time constraint row mismatch: A rows={}, b len={}",
                    base_constraints.a.nrows(),
                    base_constraints.b.len()
                ),
            }
            .into());
        }
    }

    let base_rows = base.as_ref().map_or(0, |constraints| constraints.a.nrows());
    let rows = base_rows + p_wiggle;
    let mut a = Array2::<f64>::zeros((rows, p_total));
    let mut b = Array1::<f64>::zeros(rows);

    if let Some(base_constraints) = base {
        a.slice_mut(s![..base_rows, ..]).assign(&base_constraints.a);
        b.slice_mut(s![..base_rows]).assign(&base_constraints.b);
    }

    let tail_start = p_total - p_wiggle;
    for (row_offset, col) in (tail_start..p_total).enumerate() {
        a[[base_rows + row_offset, col]] = 1.0;
    }
    Ok(Some(LinearInequalityConstraints { a, b }))
}


fn mean_abs(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for v in values {
        sum += v.abs();
        count += 1;
    }
    if count == 0 { 0.0 } else { sum / count as f64 }
}


fn block_log_lambda_seeds<'a, I>(design: &DesignMatrix, penalty_locals: I) -> Vec<f64>
where
    I: IntoIterator<Item = &'a Array2<f64>>,
{
    let unit_weights = Array1::<f64>::ones(design.nrows());
    let likelihood_scale = match design.diag_gram(&unit_weights) {
        Ok(d) => mean_abs(d.iter().copied()).max(1.0e-8),
        Err(_) => 1.0,
    };
    penalty_locals
        .into_iter()
        .map(|s| {
            let penalty_scale = mean_abs(s.diag().iter().copied()).max(1.0e-8);
            (likelihood_scale / penalty_scale).ln().clamp(-12.0, 12.0)
        })
        .collect()
}


fn joint_setup(
    data: ArrayView2<'_, f64>,
    time_penalties: usize,
    marginalspec: &TermCollectionSpec,
    marginal_penalties: usize,
    logslopespec: &TermCollectionSpec,
    logslope_penalties: usize,
    core_rho0_seed: &[f64],
    extra_rho0: &[f64],
    pinned_rho_slots: &[(usize, f64)],
    initial_sigma: Option<f64>,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> ExactJointHyperSetup {
    let marginal_terms = spatial_length_scale_term_indices(marginalspec);
    let logslope_terms = spatial_length_scale_term_indices(logslopespec);
    let core_len = time_penalties + marginal_penalties + logslope_penalties;
    let rho_dim = core_len + extra_rho0.len();
    let mut rho0vec = Array1::<f64>::zeros(rho_dim);
    assert_eq!(
        core_rho0_seed.len(),
        core_len,
        "core_rho0_seed length must equal time+marginal+logslope penalty count"
    );
    for (idx, value) in core_rho0_seed.iter().copied().enumerate().take(core_len) {
        rho0vec[idx] = value;
    }
    if !extra_rho0.is_empty() {
        let start = core_len;
        for (idx, value) in extra_rho0.iter().copied().enumerate() {
            rho0vec[start + idx] = value;
        }
    }
    let mut rho_lower = Array1::<f64>::from_elem(rho_dim, -12.0);
    let mut rho_upper = Array1::<f64>::from_elem(rho_dim, 12.0);
    // Pin fixed-ridge penalty slots (e.g. the #461 influence absorber) to a
    // degenerate box so the outer REML optimizer can never move their log-λ:
    // the absorber ridge is a fixed training-time leakage absorber, not a
    // smooth/learned surface. Seed rho0 at the pinned value too so the start
    // point is feasible.
    for &(slot, value) in pinned_rho_slots {
        assert!(
            slot < rho_dim,
            "pinned rho slot {slot} out of range (rho_dim={rho_dim})"
        );
        rho0vec[slot] = value;
        rho_lower[slot] = value;
        rho_upper[slot] = value;
    }
    // Time block has no spatial length scales (pure B-spline on time)
    let empty_kappa = SpatialLogKappaCoords::new_with_dims(Array1::zeros(0), vec![]);
    let marginal_kappa = SpatialLogKappaCoords::from_length_scales_aniso(
        marginalspec,
        &marginal_terms,
        kappa_options,
    )
    .reseed_from_data(data, marginalspec, &marginal_terms, kappa_options);
    let logslope_kappa = SpatialLogKappaCoords::from_length_scales_aniso(
        logslopespec,
        &logslope_terms,
        kappa_options,
    )
    .reseed_from_data(data, logslopespec, &logslope_terms, kappa_options);
    let mut values = empty_kappa.as_array().to_vec();
    values.extend(marginal_kappa.as_array().iter());
    values.extend(logslope_kappa.as_array().iter());
    let marginal_dims = marginal_kappa.dims_per_term().to_vec();
    let logslope_dims = logslope_kappa.dims_per_term().to_vec();
    let mut dims = empty_kappa.dims_per_term().to_vec();
    dims.extend(marginal_dims.iter().copied());
    dims.extend(logslope_dims.iter().copied());
    let log_kappa0 =
        SpatialLogKappaCoords::new_with_dims(Array1::from_vec(values.clone()), dims.clone());
    // Bounds: concatenate [empty | marginal data-aware | logslope data-aware]
    let marginal_lower = SpatialLogKappaCoords::lower_bounds_aniso_from_data(
        data,
        marginalspec,
        &marginal_terms,
        &marginal_dims,
        kappa_options,
    );
    let logslope_lower = SpatialLogKappaCoords::lower_bounds_aniso_from_data(
        data,
        logslopespec,
        &logslope_terms,
        &logslope_dims,
        kappa_options,
    );
    let mut lower_vals = Vec::with_capacity(dims.iter().sum());
    lower_vals.extend(marginal_lower.as_array().iter());
    lower_vals.extend(logslope_lower.as_array().iter());
    let log_kappa_lower =
        SpatialLogKappaCoords::new_with_dims(Array1::from_vec(lower_vals), dims.clone());
    let marginal_upper = SpatialLogKappaCoords::upper_bounds_aniso_from_data(
        data,
        marginalspec,
        &marginal_terms,
        &marginal_dims,
        kappa_options,
    );
    let logslope_upper = SpatialLogKappaCoords::upper_bounds_aniso_from_data(
        data,
        logslopespec,
        &logslope_terms,
        &logslope_dims,
        kappa_options,
    );
    let mut upper_vals = Vec::with_capacity(dims.iter().sum());
    upper_vals.extend(marginal_upper.as_array().iter());
    upper_vals.extend(logslope_upper.as_array().iter());
    let log_kappa_upper = SpatialLogKappaCoords::new_with_dims(Array1::from_vec(upper_vals), dims);
    // Project seed onto bounds; spec.length_scale is a hint, not a constraint.
    let log_kappa0 = log_kappa0.clamp_to_bounds(&log_kappa_lower, &log_kappa_upper);
    let setup = ExactJointHyperSetup::new(
        rho0vec,
        rho_lower,
        rho_upper,
        log_kappa0,
        log_kappa_lower,
        log_kappa_upper,
    );
    if let Some(sigma) = initial_sigma.filter(|sigma| *sigma > 0.0) {
        setup.with_auxiliary(
            Array1::from_vec(vec![sigma.ln()]),
            Array1::from_vec(vec![-12.0]),
            Array1::from_vec(vec![6.0]),
        )
    } else {
        setup
    }
}


fn validate_spec(spec: &SurvivalMarginalSlopeTermSpec) -> Result<(), String> {
    let n = spec.age_entry.len();
    log::info!(
        "[survival-marginal-slope] fit start n={} marginal_terms={} logslope_terms={}",
        n,
        spec.marginalspec.linear_terms.len()
            + spec.marginalspec.random_effect_terms.len()
            + spec.marginalspec.smooth_terms.len(),
        spec.logslopespec.linear_terms.len()
            + spec.logslopespec.random_effect_terms.len()
            + spec.logslopespec.smooth_terms.len(),
    );
    if spec.age_exit.len() != n
        || spec.event_target.len() != n
        || spec.weights.len() != n
        || spec.z.nrows() != n
        || spec.z.ncols() == 0
        || spec.marginal_offset.len() != n
        || spec.logslope_offset.len() != n
    {
        return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
            reason: format!(
                "survival-marginal-slope row mismatch: entry={}, exit={}, event={}, weights={}, z={}x{}, marginal_offset={}, logslope_offset={}",
                n,
                spec.age_exit.len(),
                spec.event_target.len(),
                spec.weights.len(),
                spec.z.nrows(),
                spec.z.ncols(),
                spec.marginal_offset.len(),
                spec.logslope_offset.len()
            ),
        }
        .into());
    }
    if spec.weights.iter().any(|&w| !w.is_finite() || w < 0.0) {
        return Err(SurvivalMarginalSlopeError::InvalidInput {
            reason: "survival-marginal-slope requires finite non-negative weights".to_string(),
        }
        .into());
    }
    if let Some(jac) = spec.score_influence_jacobian.as_ref() {
        // #461 absorbed influence Jacobian `J = ∂z/∂θ₁` (n × p₁): must align with
        // the fit rows and be finite. A zero-column J carries no leakage
        // directions; the build site treats it as no absorber, but a row
        // mismatch or non-finite entry is a hard error (the residualization Gram
        // and the per-row Z̃ projection both assume `n` aligned finite rows).
        if jac.nrows() != n {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival-marginal-slope score_influence_jacobian has {} rows, expected {n}",
                    jac.nrows()
                ),
            }
            .into());
        }
        if jac.iter().any(|&v| !v.is_finite()) {
            return Err(SurvivalMarginalSlopeError::InvalidInput {
                reason: "survival-marginal-slope score_influence_jacobian must be finite"
                    .to_string(),
            }
            .into());
        }
    }
    if spec.z.iter().any(|&zi| !zi.is_finite()) {
        return Err(SurvivalMarginalSlopeError::InvalidInput {
            reason: "survival-marginal-slope requires finite z values".to_string(),
        }
        .into());
    }
    if spec.marginal_offset.iter().any(|&value| !value.is_finite()) {
        return Err(SurvivalMarginalSlopeError::InvalidInput {
            reason: "survival-marginal-slope requires finite marginal offsets".to_string(),
        }
        .into());
    }
    if spec.logslope_offset.iter().any(|&value| !value.is_finite()) {
        return Err(SurvivalMarginalSlopeError::InvalidInput {
            reason: "survival-marginal-slope requires finite logslope offsets".to_string(),
        }
        .into());
    }
    spec.frailty.validate_for_marginal_slope()?;
    match &spec.frailty {
        FrailtySpec::None => {}
        FrailtySpec::GaussianShift { sigma_fixed } => {
            let Some(sigma) = sigma_fixed else {
                return Err(SurvivalMarginalSlopeError::UnsupportedConfiguration {
                    reason:
                        "survival-marginal-slope requires GaussianShift sigma_fixed or FrailtySpec::None; learnable GaussianShift sigma is not implemented for the exact marginal-slope outer solver"
                            .to_string(),
                }
                .into());
            };
            if !sigma.is_finite() || *sigma < 0.0 {
                return Err(SurvivalMarginalSlopeError::InvalidInput {
                    reason: format!(
                        "survival-marginal-slope requires GaussianShift sigma >= 0, got {sigma}"
                    ),
                }
                .into());
            }
        }
        FrailtySpec::HazardMultiplier { .. } => {
            return Err(SurvivalMarginalSlopeError::InvalidInput {
                reason: "survival-marginal-slope does not support FrailtySpec::HazardMultiplier"
                    .to_string(),
            }
            .into());
        }
    }
    if spec.event_target.iter().any(|&d| d != 0.0 && d != 1.0) {
        return Err(SurvivalMarginalSlopeError::InvalidInput {
            reason: "survival-marginal-slope requires binary event indicators (0.0 or 1.0)"
                .to_string(),
        }
        .into());
    }
    // Fast-fail on a degenerate all-censored design: the marginal-slope partial
    // likelihood has no events to anchor the hazard scale, so the outer/inner
    // solve cannot make progress and otherwise spins without termination (#789B).
    if !spec.event_target.is_empty() && spec.event_target.iter().all(|&d| d == 0.0) {
        return Err(SurvivalMarginalSlopeError::InvalidInput {
            reason: "survival-marginal-slope requires at least one event (event==1); the supplied design is entirely censored (all event==0), which has no finite marginal-slope fit"
                .to_string(),
        }
        .into());
    }
    if !spec.derivative_guard.is_finite() || spec.derivative_guard <= 0.0 {
        return Err(SurvivalMarginalSlopeError::InvalidInput {
            reason: format!(
                "survival-marginal-slope requires derivative_guard > 0, got {}",
                spec.derivative_guard
            ),
        }
        .into());
    }
    for i in 0..n {
        if spec.age_exit[i] < spec.age_entry[i] {
            return Err(SurvivalMarginalSlopeError::InvalidInput {
                reason: format!(
                    "survival-marginal-slope row {i}: exit time ({}) < entry time ({})",
                    spec.age_exit[i], spec.age_entry[i]
                ),
            }
            .into());
        }
    }
    let n_entry = spec.time_block.design_entry.nrows();
    let n_exit = spec.time_block.design_exit.nrows();
    let n_deriv = spec.time_block.design_derivative_exit.nrows();
    if n_entry != n || n_exit != n || n_deriv != n {
        return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
            reason: format!(
                "survival-marginal-slope time block design row mismatch: \
                 data={n}, design_entry={n_entry}, design_exit={n_exit}, design_derivative_exit={n_deriv}"
            ),
        }
        .into());
    }
    let p_entry = spec.time_block.design_entry.ncols();
    let p_exit = spec.time_block.design_exit.ncols();
    let p_deriv = spec.time_block.design_derivative_exit.ncols();
    if p_exit != p_entry || p_deriv != p_entry {
        return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
            reason: format!(
                "survival-marginal-slope time block design column mismatch: entry={p_entry}, exit={p_exit}, deriv={p_deriv}"
            ),
        }
        .into());
    }
    if !spec.time_block.time_monotonicity.requires_row_constraints()
        && !matches!(
            spec.time_block.time_monotonicity,
            crate::families::survival_location_scale::TimeBlockMonotonicity::StructuralISpline
        )
    {
        return Err(SurvivalMarginalSlopeError::UnsupportedConfiguration {
            reason: format!(
                "survival-marginal-slope requires a row-constraint or structural-I-spline time block; got {:?}",
                spec.time_block.time_monotonicity
            ),
        }
        .into());
    }
    if let Some(beta0) = &spec.time_block.initial_beta {
        match spec.time_block.time_monotonicity {
            crate::families::survival_location_scale::TimeBlockMonotonicity::StructuralISpline => {
                // Under the I-spline base, the only feasibility constraint on
                // `initial_beta` is the γ ≥ 0 coordinate cone — the row-wise
                // `D γ + o ≥ guard` constraints are vacuous (I-spline derivatives
                // ≥ 0, `add_survival_time_derivative_guard_offset` makes `o ≥ guard`
                // already). Running the row-wise generator here would duplicate
                // information and risk a misleading row-wise rejection.
                if spec.time_block.design_derivative_exit.ncols() != beta0.len() {
                    return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                        reason: format!(
                            "survival-marginal-slope time_block initial_beta length mismatch under StructuralISpline: got {}, expected {}",
                            beta0.len(),
                            spec.time_block.design_derivative_exit.ncols()
                        ),
                    }
                    .into());
                }
                for (j, &g) in beta0.iter().enumerate() {
                    if !g.is_finite() {
                        return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
                            reason: format!(
                                "survival-marginal-slope time_block initial_beta is non-finite at coordinate {j} under StructuralISpline: got {g}"
                            ),
                        }
                        .into());
                    }
                    if g < -1e-12 {
                        return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
                            reason: format!(
                                "survival-marginal-slope time_block initial_beta violates γ ≥ 0 at coordinate {j} under StructuralISpline: got {g:.3e}"
                            ),
                        }
                        .into());
                    }
                }
            }
            _ => {
                let derivative_constraints = time_derivative_guard_constraints(
                    &spec.time_block.design_derivative_exit,
                    &spec.time_block.derivative_offset_exit,
                    spec.derivative_guard,
                )?;
                if let Some(constraints) = derivative_constraints.as_ref() {
                    if beta0.len() != constraints.a.ncols() {
                        return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                            reason: format!(
                                "survival-marginal-slope time_block initial_beta length mismatch: got {}, expected {}",
                                beta0.len(),
                                constraints.a.ncols()
                            ),
                        }
                        .into());
                    }
                    for row in 0..constraints.a.nrows() {
                        let slack = constraints.a.row(row).dot(beta0) - constraints.b[row];
                        if slack < -1e-10 {
                            return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
                                reason: format!(
                                    "survival-marginal-slope time_block initial_beta violates derivative guard constraint at row {row}: slack={slack:.3e}"
                                ),
                            }
                            .into());
                        }
                    }
                }
            }
        }
    }
    if let Some(timewiggle) = spec.timewiggle_block.as_ref() {
        if timewiggle.degree != 3 {
            return Err(SurvivalMarginalSlopeError::UnsupportedConfiguration {
                reason: format!(
                    "survival-marginal-slope timewiggle requires cubic degree=3, got {}",
                    timewiggle.degree
                ),
            }
            .into());
        }
        let derived_ncols = time_wiggle_basis_ncols(&timewiggle.knots, timewiggle.degree)?;
        if derived_ncols == 0 {
            return Err(SurvivalMarginalSlopeError::InvalidInput {
                reason:
                    "survival-marginal-slope timewiggle requires at least one wiggle coefficient"
                        .to_string(),
            }
            .into());
        }
        if timewiggle.ncols != derived_ncols {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival-marginal-slope timewiggle metadata width mismatch: metadata={}, basis={derived_ncols}",
                    timewiggle.ncols
                ),
            }
            .into());
        }
        if spec.time_block.design_exit.ncols() < derived_ncols {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival-marginal-slope timewiggle requests {} tail columns but time block only has {} columns",
                    derived_ncols,
                    spec.time_block.design_exit.ncols()
                ),
            }
            .into());
        }
    }
    Ok(())
}


fn install_time_nullspace_shrinkage_penalty(
    time_block: &mut TimeBlockInput,
) -> Result<bool, String> {
    let p = time_block.design_exit.ncols();
    if p == 0 || time_block.penalties.is_empty() {
        return Ok(false);
    }
    if time_block.nullspace_dims.len() != time_block.penalties.len() {
        return Err(format!(
            "survival-marginal-slope time_block nullspace_dims length {} does not match penalties {}",
            time_block.nullspace_dims.len(),
            time_block.penalties.len(),
        ));
    }

    let mut aggregate = Array2::<f64>::zeros((p, p));
    for (idx, penalty) in time_block.penalties.iter().enumerate() {
        if penalty.nrows() != p || penalty.ncols() != p {
            return Err(format!(
                "survival-marginal-slope time_block penalty {idx} must be {p}x{p}, got {}x{}",
                penalty.nrows(),
                penalty.ncols(),
            ));
        }
        let scale = penalty
            .iter()
            .try_fold(0.0_f64, |acc, &value| {
                value.is_finite().then_some(acc.max(value.abs()))
            })
            .ok_or_else(|| {
                format!(
                    "survival-marginal-slope time_block penalty {idx} contains non-finite values"
                )
            })?;
        if scale > 0.0 {
            ndarray::Zip::from(&mut aggregate)
                .and(penalty)
                .for_each(|agg, &value| *agg += value / scale);
        }
    }

    let Some(shrinkage) = crate::terms::basis::build_nullspace_shrinkage_penalty(&aggregate)
        .map_err(|err| format!("survival-marginal-slope time_block nullspace shrinkage: {err}"))?
    else {
        return Ok(false);
    };
    if shrinkage.sym_penalty.nrows() != p || shrinkage.sym_penalty.ncols() != p {
        return Err(format!(
            "survival-marginal-slope time_block nullspace shrinkage penalty must be {p}x{p}, got {}x{}",
            shrinkage.sym_penalty.nrows(),
            shrinkage.sym_penalty.ncols(),
        ));
    }
    time_block.penalties.push(shrinkage.sym_penalty);
    time_block.nullspace_dims.push(0);
    log::info!(
        "[survival-marginal-slope] added time_block nullspace shrinkage penalty (p={p}, penalties={})",
        time_block.penalties.len(),
    );
    Ok(true)
}


fn concatenate_term_specs(specs: &[TermCollectionSpec]) -> TermCollectionSpec {
    let mut out = TermCollectionSpec {
        linear_terms: Vec::new(),
        random_effect_terms: Vec::new(),
        smooth_terms: Vec::new(),
    };
    for spec in specs {
        out.linear_terms.extend(spec.linear_terms.clone());
        out.random_effect_terms
            .extend(spec.random_effect_terms.clone());
        out.smooth_terms.extend(spec.smooth_terms.clone());
    }
    out
}


fn shift_penalty(mut penalty: BlockwisePenalty, offset: usize) -> BlockwisePenalty {
    penalty.col_range = (penalty.col_range.start + offset)..(penalty.col_range.end + offset);
    penalty
}


fn combine_logslope_surface_designs(
    mut designs: Vec<TermCollectionDesign>,
    specs: &[TermCollectionSpec],
) -> Result<
    (
        TermCollectionDesign,
        TermCollectionSpec,
        Vec<std::ops::Range<usize>>,
    ),
    String,
> {
    if designs.is_empty() {
        return Err(
            "survival marginal-slope requires at least one logslope surface design".to_string(),
        );
    }
    if designs.len() == 1 {
        let design = designs.remove(0);
        let range = 0..design.design.ncols();
        let spec = specs
            .first()
            .cloned()
            .ok_or_else(|| "missing logslope surface spec".to_string())?;
        return Ok((design, spec, vec![range]));
    }
    if designs.iter().any(|design| {
        design.linear_constraints.is_some() || design.coefficient_lower_bounds.is_some()
    }) {
        return Err(
            "per-z logslope surface concatenation does not support coefficient bounds or linear constraints"
                .to_string(),
        );
    }

    let mut ranges = Vec::with_capacity(designs.len());
    let mut offset = 0usize;
    let mut blocks = Vec::with_capacity(designs.len());
    let mut penalties = Vec::new();
    let mut nullspace_dims = Vec::new();
    let mut penaltyinfo = Vec::new();
    let mut dropped_penaltyinfo = Vec::new();
    let mut linear_ranges = Vec::new();
    let mut random_effect_ranges = Vec::new();
    let mut random_effect_levels = Vec::new();
    let mut combined = designs[0].clone();
    combined.smooth.term_designs.clear();
    combined.smooth.penalties.clear();
    combined.smooth.nullspace_dims.clear();
    combined.smooth.penaltyinfo.clear();
    combined.smooth.dropped_penaltyinfo.clear();
    combined.smooth.terms.clear();
    combined.smooth.coefficient_lower_bounds = None;
    combined.smooth.linear_constraints = None;

    for (surface_idx, design) in designs.into_iter().enumerate() {
        let width = design.design.ncols();
        ranges.push(offset..offset + width);
        blocks.push(design.design.clone());
        for (local_penalty_idx, penalty) in design.penalties.iter().cloned().enumerate() {
            let global_index = penalties.len();
            penalties.push(shift_penalty(penalty, offset));
            if let Some(info) = design.penaltyinfo.get(local_penalty_idx) {
                let mut info = info.clone();
                info.global_index = global_index;
                if let Some(termname) = info.termname.as_mut() {
                    *termname = format!("logslope[z{surface_idx}]::{termname}");
                }
                penaltyinfo.push(info);
            }
        }
        nullspace_dims.extend(design.nullspace_dims.iter().copied());
        dropped_penaltyinfo.extend(design.dropped_penaltyinfo.iter().cloned());
        linear_ranges.extend(design.linear_ranges.iter().cloned().map(|(name, range)| {
            (
                format!("logslope[z{surface_idx}]::{name}"),
                (range.start + offset)..(range.end + offset),
            )
        }));
        random_effect_ranges.extend(design.random_effect_ranges.iter().cloned().map(
            |(name, range)| {
                (
                    format!("logslope[z{surface_idx}]::{name}"),
                    (range.start + offset)..(range.end + offset),
                )
            },
        ));
        random_effect_levels.extend(design.random_effect_levels.iter().cloned());
        offset += width;
    }
    combined.design = DesignMatrix::hstack(blocks)
        .map_err(|e| format!("survival marginal-slope logslope hstack: {e}"))?;
    combined.penalties = penalties;
    combined.nullspace_dims = nullspace_dims;
    combined.penaltyinfo = penaltyinfo;
    combined.dropped_penaltyinfo = dropped_penaltyinfo;
    combined.coefficient_lower_bounds = None;
    combined.linear_constraints = None;
    combined.intercept_range = 0..0;
    combined.linear_ranges = linear_ranges;
    combined.random_effect_ranges = random_effect_ranges;
    combined.random_effect_levels = random_effect_levels;
    Ok((combined, concatenate_term_specs(specs), ranges))
}


/// Compute a baseline slope from the actual survival marginal-slope likelihood,
/// using the baseline offsets alone as a time-only pilot q(t).
///
/// This is a safeguarded 1D Newton solve on the true row objective. It does not
/// use a coarse fixed grid scan.
fn pooled_survival_baseline(
    event: &Array1<f64>,
    weights: &Array1<f64>,
    z: &Array1<f64>,
    q0: &Array1<f64>,
    q1: &Array1<f64>,
    qd1: &Array1<f64>,
    probit_scale: f64,
) -> f64 {
    let n = event.len();
    if n == 0 {
        return 0.0;
    }
    let objective_grad_hess = |slope: f64| -> Option<(f64, f64, f64)> {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let triples: Option<Vec<(f64, f64, f64)>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let (row_obj, row_grad, row_hess) = row_primary_closed_form(
                    q0[i],
                    q1[i],
                    qd1[i],
                    slope,
                    z[i],
                    weights[i],
                    event[i],
                    0.0,
                    probit_scale,
                )
                .ok()?;
                Some((row_obj, row_grad[3], row_hess[3][3]))
            })
            .collect();
        let triples = triples?;
        Some(
            triples
                .into_iter()
                .fold((0.0_f64, 0.0_f64, 0.0_f64), |(o, g, h), (oi, gi, hi)| {
                    (o + oi, g + gi, h + hi)
                }),
        )
    };

    let Some(state0) = objective_grad_hess(0.0) else {
        return 0.0;
    };
    if !state0.0.is_finite() {
        return 0.0;
    }
    if state0.1.abs() < 1e-8 {
        return 0.0;
    }

    let mut best_slope = 0.0;
    let mut best = state0;

    let mut bracket_lo = if state0.1 <= 0.0 {
        Some((0.0, state0))
    } else {
        None
    };
    let mut bracket_hi = if state0.1 >= 0.0 {
        Some((0.0, state0))
    } else {
        None
    };
    let mut step = 0.5f64;
    for _ in 0..48 {
        for &candidate in &[-step, step] {
            if let Some(state) = objective_grad_hess(candidate) {
                if state.0 < best.0 {
                    best_slope = candidate;
                    best = state;
                }
                if state.1 <= 0.0 {
                    bracket_lo = Some((candidate, state));
                }
                if state.1 >= 0.0 {
                    bracket_hi = Some((candidate, state));
                }
                if let (Some((lo, lo_state)), Some((hi, hi_state))) = (bracket_lo, bracket_hi)
                    && lo < hi
                    && lo_state.1 <= 0.0
                    && hi_state.1 >= 0.0
                {
                    let mut slope = best_slope.clamp(lo, hi);
                    let mut state = if (slope - lo).abs() < f64::EPSILON {
                        lo_state
                    } else if (slope - hi).abs() < f64::EPSILON {
                        hi_state
                    } else {
                        match objective_grad_hess(slope) {
                            Some(s) => s,
                            None => {
                                slope = 0.5 * (lo + hi);
                                objective_grad_hess(slope).unwrap_or(best)
                            }
                        }
                    };

                    let mut bracket_lo = (lo, lo_state);
                    let mut bracket_hi = (hi, hi_state);
                    for _ in 0..60 {
                        if state.1.abs() < 1e-8 || (bracket_hi.0 - bracket_lo.0).abs() < 1e-8 {
                            break;
                        }
                        let mut candidate = 0.5 * (bracket_lo.0 + bracket_hi.0);
                        if state.2.is_finite() && state.2 > 0.0 {
                            let newton = slope - state.1 / state.2;
                            if newton > bracket_lo.0 && newton < bracket_hi.0 {
                                candidate = newton;
                            }
                        }
                        let Some(candidate_state) = objective_grad_hess(candidate) else {
                            candidate = 0.5 * (bracket_lo.0 + bracket_hi.0);
                            let Some(mid_state) = objective_grad_hess(candidate) else {
                                break;
                            };
                            if mid_state.0 < best.0 {
                                best_slope = candidate;
                                best = mid_state;
                            }
                            if mid_state.1 <= 0.0 {
                                bracket_lo = (candidate, mid_state);
                            } else {
                                bracket_hi = (candidate, mid_state);
                            }
                            slope = candidate;
                            state = mid_state;
                            continue;
                        };
                        if candidate_state.0 < best.0 {
                            best_slope = candidate;
                            best = candidate_state;
                        }
                        if candidate_state.1 <= 0.0 {
                            bracket_lo = (candidate, candidate_state);
                        } else {
                            bracket_hi = (candidate, candidate_state);
                        }
                        slope = candidate;
                        state = candidate_state;
                    }
                    return if best.0.is_finite() { best_slope } else { 0.0 };
                }
            }
        }
        step *= 2.0;
    }
    if best.0.is_finite() { best_slope } else { 0.0 }
}


// ── Public fitting function ───────────────────────────────────────────

/// Whether the optional score-warp / link-deviation flex blocks participate in
/// a family build. The rigid warm-start pilot must construct its family and
/// blocks with `OffForRigidPilot` so the cold-start coefficient solve cannot
/// silently activate the survival flex exact-Joint-Newton path.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum FlexActivation {
    OffForRigidPilot,
    On,
}


/// Which coordinate system the `marginal`/`logslope` designs handed to
/// `build_blocks` live in — and therefore whether the side-bound V+M-exact
/// pulled-back penalties (`*_penalties_vm`) apply.
///
/// The V+M-exact cutover compiles the marginal/logslope designs into a reduced
/// (column-dropped, reparameterised) coordinate system and pulls each block's
/// penalty back through that block's own `V_b` to a matching compiled-width
/// `Vᵀ S V`. Those pulled-back penalties are valid **only** against the
/// compiled designs they were derived from. Two distinct callers reach
/// `build_blocks` with structurally different designs:
///
/// * [`BlockDesignCoords::PostCutover`] — the construction-site calls (rigid
///   warm-start pilot, initial derivative probe) pass the post-cutover designs
///   the `*_penalties_vm` were pulled back through. The compiled penalties are
///   authoritative here and must replace the raw-width penalties that
///   `build_*_blockspec` derives from `TermCollectionDesign.penalties` (which
///   the cutover intentionally leaves at raw width for predict-time consumers).
/// * [`BlockDesignCoords::RematerializedRaw`] — the outer spatial-length-scale
///   optimizer re-materialises *raw*-width marginal/logslope designs from the
///   boot `TermCollectionSpec`s on every κ probe and routes them here. The raw
///   design-derived penalties are authoritative; the compiled `*_penalties_vm`
///   are at the wrong width/parametrisation and must NOT be installed (doing so
///   is the #788 `block i penalty 0 must be NxN, got KxK` shape mismatch — and,
///   in the no-column-drop-but-`V≠I` case where widths happen to coincide, a
///   silent installation of a `Vᵀ S V` penalty onto a raw design).
///
/// Distinguishing by an explicit, named coordinate tag — rather than guessing
/// from a `penalty.shape() == design.shape()` width coincidence — makes the
/// provenance an auditable property of each call site and lets the compiled
/// path assert width agreement loudly instead of relying on it holding.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum BlockDesignCoords {
    PostCutover,
    RematerializedRaw,
}


/// One block in the joint training-row design layout consumed by
/// [`joint_training_design_preflight`]. Holds a dense `(n x p_block)`
/// slice and the block tag used in the failure diagnostic.
pub(crate) struct JointPreflightSegment {
    pub block: JointPreflightBlock,
    pub columns: DesignMatrix,
}


/// W-metric joint training-row design preflight.
///
/// Stacks the per-block training-row designs horizontally into a single
/// `n x p_joint` matrix `J`, pre-scales by `sqrt(W)`, and runs a thin-SVD
/// via [`crate::faer_ndarray::FaerSvd`]. Always returns `Ok(())`; if any
/// singular value falls at or below the numerical-rank tolerance
/// `sigma_max * max(n, p_joint) * 16 * f64::EPSILON`, emits an info-level
/// log line localising each alias to its dominant `(block, local_col, weight)`
/// triple. The canonical-gauge pipeline in
/// `custom_family.rs::canonicalize_for_identifiability` attributes the alias
/// drops via `gauge_priority` and reduces the specs before any inner Newton
/// fires, so this preflight is diagnostic-only.
pub(crate) fn joint_training_design_preflight(
    segments: &[JointPreflightSegment],
    weights: &Array1<f64>,
) -> Result<(), SurvivalMarginalSlopeError> {
    use crate::faer_ndarray::{FaerEigh, fast_xt_diag_y};

    if segments.is_empty() {
        return Ok(());
    }
    let n = weights.len();
    let mut p_joint = 0usize;
    let mut block_ranges: Vec<(JointPreflightBlock, usize, usize)> =
        Vec::with_capacity(segments.len());
    for seg in segments {
        if seg.columns.nrows() != n {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "joint preflight: block {} has {} rows, weights have {}",
                    seg.block,
                    seg.columns.nrows(),
                    n,
                ),
            });
        }
        let start = p_joint;
        let end = p_joint + seg.columns.ncols();
        block_ranges.push((seg.block, start, end));
        p_joint = end;
    }
    if p_joint == 0 {
        return Ok(());
    }

    for (i, &w) in weights.iter().enumerate() {
        if !w.is_finite() || w < 0.0 {
            return Err(SurvivalMarginalSlopeError::InvalidInput {
                reason: format!("joint preflight: weights[{i}] = {w} is not finite/non-negative",),
            });
        }
    }

    // W-metric joint Gram `G = Jᵀ diag(w) J`, accumulated over fixed-height
    // row chunks of the operator-backed segments. The previous implementation
    // stacked a dense `(n, p_joint)` sqrt(W)-scaled copy of the joint design
    // and ran a thin-SVD over it — at biobank scale that transient (plus the
    // SVD's own workspace) was a multi-GiB contributor to the #979 survival
    // construction-phase OOM, and the budget guard that protected against it
    // silently skipped the diagnostic at exactly the scale where it matters.
    // The Gram route needs `O(chunk × p_joint + p_joint²)` memory at any n,
    // so the preflight always runs. Singular values come back as √eigenvalue;
    // squaring the spectrum coarsens the detectable near-alias floor from
    // ~`dim·ε·σ_max` to ~`sqrt(dim·ε)·σ_max`, which is immaterial here:
    // structural aliases (σ exactly 0) are detected identically, and this
    // preflight is observability-only — the canonical-gauge RRQR audit
    // downstream remains the fail-closed authority on borderline cases.
    const PREFLIGHT_GRAM_ROW_CHUNK: usize = 4096;
    let mut gram = Array2::<f64>::zeros((p_joint, p_joint));
    let mut chunk_start = 0usize;
    while chunk_start < n {
        let chunk_end = (chunk_start + PREFLIGHT_GRAM_ROW_CHUNK).min(n);
        let chunks: Vec<Array2<f64>> = segments
            .iter()
            .map(|seg| {
                seg.columns
                    .try_row_chunk(chunk_start..chunk_end)
                    .map_err(|e| SurvivalMarginalSlopeError::NumericalFailure {
                        reason: format!(
                            "joint preflight: block {} rows {chunk_start}..{chunk_end}: {e}",
                            seg.block
                        ),
                    })
            })
            .collect::<Result<_, _>>()?;
        let w_chunk = weights.slice(s![chunk_start..chunk_end]);
        for (s, (_, s0, s1)) in block_ranges.iter().enumerate() {
            for (t, (_, t0, t1)) in block_ranges.iter().enumerate().skip(s) {
                let g_st = fast_xt_diag_y(&chunks[s], &w_chunk, &chunks[t]);
                let mut dst = gram.slice_mut(s![*s0..*s1, *t0..*t1]);
                dst += &g_st;
                if t > s {
                    let mut dst_t = gram.slice_mut(s![*t0..*t1, *s0..*s1]);
                    dst_t += &g_st.t();
                }
            }
        }
        chunk_start = chunk_end;
    }

    let (eigvals, eigvecs) =
        gram.eigh(faer::Side::Lower)
            .map_err(|e| SurvivalMarginalSlopeError::NumericalFailure {
                reason: format!("joint preflight: W-metric Gram eigh failed: {e:?}"),
            })?;
    // σ_i = sqrt(max(λ_i, 0)); roundoff can push numerically-zero Gram
    // eigenvalues slightly negative.
    let sigma: Vec<f64> = eigvals.iter().map(|&l| l.max(0.0).sqrt()).collect();
    let sigma_max = sigma.iter().copied().fold(0.0_f64, f64::max);
    let rank_dim = n.max(p_joint) as f64;
    // Gram-spectrum near-alias floor (see the block comment above): aliases
    // are directions whose singular value is at or below
    // `σ_max · sqrt(dim · 16ε)`.
    let rank_tol = sigma_max * (rank_dim * 16.0 * f64::EPSILON).sqrt();

    let alias_idx: Vec<usize> = sigma
        .iter()
        .enumerate()
        .filter_map(|(idx, &s)| (s <= rank_tol).then_some(idx))
        .collect();
    let rank = sigma.len() - alias_idx.len();

    if alias_idx.is_empty() {
        let sigma_min = sigma.iter().copied().fold(f64::INFINITY, f64::min);
        let condition = if sigma_min > 0.0 {
            sigma_max / sigma_min
        } else {
            f64::INFINITY
        };
        log::info!(
            "[survival-marginal-slope/preflight] joint design full-rank: n={n} p_joint={p_joint} \
             sigma_min={sigma_min:.3e} sigma_max={sigma_max:.3e} kappa={condition:.3e} tol={rank_tol:.3e}",
        );
        return Ok(());
    }

    let structural_alias = p_joint.saturating_sub(n.min(p_joint));

    let mut columns: Vec<(JointPreflightBlock, usize, f64)> = Vec::new();
    for &idx in alias_idx.iter() {
        // Eigenvector column `idx` of the Gram is the right singular vector
        // of the collapsing direction.
        let v_col = eigvecs.column(idx);
        let mut best_j = 0usize;
        let mut best_w = 0.0_f64;
        for j in 0..p_joint {
            let w = v_col[j].abs();
            if w > best_w {
                best_w = w;
                best_j = j;
            }
        }
        let (block, local_col) = block_ranges
            .iter()
            .find_map(|(b, start, end)| {
                (best_j >= *start && best_j < *end).then_some((*b, best_j - *start))
            })
            .ok_or_else(|| SurvivalMarginalSlopeError::NumericalFailure {
                reason: format!(
                    "joint preflight: alias column index {best_j} outside block ranges (p_joint={p_joint})",
                ),
            })?;
        columns.push((block, local_col, best_w));
    }

    let block_summary = block_ranges
        .iter()
        .map(|(b, start, end)| format!("{b}=[{start}..{end})"))
        .collect::<Vec<_>>()
        .join(", ");
    let dominant_summary = columns
        .iter()
        .map(|(b, c, w)| format!("{b}[{c}] (|v|={w:.3e})"))
        .collect::<Vec<_>>()
        .join("; ");
    // Informational: rank deficiency at this point is handled gracefully by
    // the canonical-gauge pipeline downstream (`canonicalize_for_identifiability`
    // in `custom_family.rs`). That pipeline runs a joint RRQR audit with
    // `gauge_priority` attribution and converts attributed alias drops into
    // per-block selection matrices `T_i`, then solves on reduced specs and
    // lifts coefficients back via `β_raw = T_i θ`. Aborting here would defeat
    // the canonical reduction.
    log::info!(
        "[survival-marginal-slope/preflight] joint design W-metric rank-deficient: \
         rank={rank}/{p_joint} (sigma_max={sigma_max:.3e}, rank_tol={rank_tol:.3e}, n={n}, \
         structural_alias={structural_alias}, alias_directions={alias_count}). \
         Block layout: {block_summary}. Dominant block x column per alias: {dominant_summary}. \
         Canonical-gauge pipeline (gauge_priority: time=200, marginal=150, logslope=120, \
         score_warp=80, link_dev=60) will attribute the alias to lower-priority blocks and \
         proceed with reduced specs.",
        alias_count = alias_idx.len(),
    );
    Ok(())
}
