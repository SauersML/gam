//! The `CustomFamily` trait implementation: the dispatch boundary between
//! the outer hyperparameter loop and the inner coefficient solve (cost,
//! policy, evaluate, joint-Hessian workspace construction, psi terms,
//! constraints, and post-update feasibility).

use super::*;

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
        let mut hasher = crate::warm_start::Fingerprinter::new();
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
            // are batched in `reml::reml_outer_engine`, so the shared X·U_S work is
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
