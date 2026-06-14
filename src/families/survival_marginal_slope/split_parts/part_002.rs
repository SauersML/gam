
// ── Row-level NLL computation ─────────────────────────────────────────

impl SurvivalMarginalSlopeFamily {
    #[inline]
    fn probit_frailty_scale(&self) -> f64 {
        probit_frailty_scale(self.gaussian_frailty_sd)
    }

    fn z_subsample_key(&self) -> Array1<f64> {
        self.z.column(0).to_owned()
    }

    #[inline]
    fn score_dim(&self) -> usize {
        assert_eq!(self.score_covariance.dim(), self.z.ncols());
        self.z.ncols()
    }

    fn score_warp_basis_dim(&self) -> usize {
        self.score_warp
            .as_ref()
            .map_or(0, DeviationRuntime::basis_dim)
    }

    fn score_warp_coord_basis_index(&self, local_idx: usize) -> Result<(usize, usize), String> {
        let basis_dim = self.score_warp_basis_dim();
        if basis_dim == 0 {
            return Err(
                "survival score-warp coordinate lookup without score-warp runtime".to_string(),
            );
        }
        let coord = local_idx / basis_dim;
        if coord >= self.score_dim() {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival score-warp local index {local_idx} exceeds K={} per-z blocks with basis dim {basis_dim}",
                    self.score_dim()
                ),
            }
            .into());
        }
        Ok((coord, local_idx % basis_dim))
    }

    fn score_warp_beta_for_coord(
        &self,
        beta_h: &Array1<f64>,
        coord: usize,
    ) -> Result<Array1<f64>, String> {
        let runtime = self
            .score_warp
            .as_ref()
            .ok_or_else(|| "missing survival score-warp runtime".to_string())?;
        score_warp_component_beta(runtime, beta_h, coord)
    }

    #[inline]
    fn zero_score_warp_span() -> exact_kernel::LocalSpanCubic {
        exact_kernel::LocalSpanCubic {
            left: 0.0,
            right: 1.0,
            c0: 0.0,
            c1: 0.0,
            c2: 0.0,
            c3: 0.0,
        }
    }

    fn add_local_span_cubic(
        left: f64,
        right: f64,
        target: &mut exact_kernel::LocalSpanCubic,
        span: exact_kernel::LocalSpanCubic,
    ) {
        target.left = left;
        target.right = right;
        target.c0 += span.c0;
        target.c1 += span.c1;
        target.c2 += span.c2;
        target.c3 += span.c3;
    }

    fn score_warp_local_cubic_at(
        &self,
        beta_h: Option<&Array1<f64>>,
        value: f64,
    ) -> Result<exact_kernel::LocalSpanCubic, String> {
        let (Some(runtime), Some(beta_h)) = (self.score_warp.as_ref(), beta_h) else {
            return Ok(Self::zero_score_warp_span());
        };
        if self.score_dim() == 1 {
            return runtime.local_cubic_at(beta_h, value);
        }
        let mut sum = Self::zero_score_warp_span();
        for coord in 0..self.score_dim() {
            let local_beta = score_warp_component_beta(runtime, beta_h, coord)?;
            let span = runtime.local_cubic_at(&local_beta, value)?;
            if coord == 0 {
                sum = exact_kernel::LocalSpanCubic {
                    left: span.left,
                    right: span.right,
                    c0: 0.0,
                    c1: 0.0,
                    c2: 0.0,
                    c3: 0.0,
                };
            }
            Self::add_local_span_cubic(span.left, span.right, &mut sum, span);
        }
        Ok(sum)
    }

    fn score_warp_observed_value(
        &self,
        row: usize,
        beta_h: Option<&Array1<f64>>,
    ) -> Result<f64, String> {
        let (Some(runtime), Some(beta_h)) = (self.score_warp.as_ref(), beta_h) else {
            return Ok(0.0);
        };
        let mut value = 0.0;
        for coord in 0..self.score_dim() {
            let local_beta = score_warp_component_beta(runtime, beta_h, coord)?;
            let z_coord = self.z[[row, coord]];
            value += runtime
                .local_cubic_at(&local_beta, z_coord)?
                .evaluate(z_coord);
        }
        Ok(value)
    }

    fn observed_score_projection(&self, row: usize) -> f64 {
        if self.score_dim() == 1 {
            self.z[[row, 0]]
        } else {
            self.z.row(row).sum()
        }
    }

    fn integration_score_basis_coefficients(
        &self,
        local_idx: usize,
        z_basis: f64,
        multiplier: f64,
    ) -> Result<[f64; 4], String> {
        let runtime = self
            .score_warp
            .as_ref()
            .ok_or_else(|| "missing survival score-warp runtime".to_string())?;
        let (_, basis_idx) = self.score_warp_coord_basis_index(local_idx)?;
        let basis_span = runtime.basis_cubic_at(basis_idx, z_basis)?;
        Ok(exact_kernel::score_basis_cell_coefficients(
            basis_span, multiplier,
        ))
    }

    fn observed_score_basis_coefficients(
        &self,
        row: usize,
        local_idx: usize,
        z_obs: f64,
        multiplier: f64,
    ) -> Result<[f64; 4], String> {
        let runtime = self
            .score_warp
            .as_ref()
            .ok_or_else(|| "missing survival score-warp runtime".to_string())?;
        if self.score_dim() == 1 {
            let basis_span = runtime.basis_cubic_at(local_idx, z_obs)?;
            return Ok(exact_kernel::score_basis_cell_coefficients(
                basis_span, multiplier,
            ));
        }
        let (coord, basis_idx) = self.score_warp_coord_basis_index(local_idx)?;
        let z_coord = self.z[[row, coord]];
        let basis_span = runtime.basis_cubic_at(basis_idx, z_coord)?;
        Ok([multiplier * basis_span.evaluate(z_coord), 0.0, 0.0, 0.0])
    }

    fn logslope_vector_for_row(
        &self,
        row: usize,
        logslope_eta: &Array1<f64>,
    ) -> Result<Vec<f64>, String> {
        let k = self.score_dim();
        if self.logslope_surface_ranges.len() == k && k > 1 && logslope_eta.len() == self.n {
            return Err(
                    "survival marginal-slope internal logslope vector requested scalar eta for a per-z surface layout"
                        .to_string(),
                );
        }
        if logslope_eta.len() == self.n {
            return Ok(vec![logslope_eta[row]; k]);
        }
        if logslope_eta.len() == self.n * k {
            let start = row * k;
            return Ok(logslope_eta.slice(s![start..start + k]).to_vec());
        }
        Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
            reason: format!(
                "survival marginal-slope logslope eta length {} is incompatible with n={} and score dim K={k}",
                logslope_eta.len(),
                self.n
            ),
        }
        .into())
    }

    fn per_z_logslope_active(&self) -> bool {
        self.score_dim() > 1 && self.logslope_surface_ranges.len() == self.score_dim()
    }

    fn logslope_surface_values_for_row(
        &self,
        row: usize,
        beta_logslope: &Array1<f64>,
    ) -> Result<Vec<f64>, String> {
        let k = self.score_dim();
        if !self.per_z_logslope_active() {
            return self.logslope_vector_for_row(row, beta_logslope);
        }
        let g_row = self
            .logslope_design
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("logslope_surface_values_for_row logslope row chunk: {e}"))?;
        let row_view = g_row.row(0);
        let mut out = Vec::with_capacity(k);
        for range in &self.logslope_surface_ranges {
            out.push(
                row_view
                    .slice(s![range.clone()])
                    .dot(&beta_logslope.slice(s![range.clone()])),
            );
        }
        Ok(out)
    }

    fn logslope_surface_row(&self, row: usize) -> Result<Array1<f64>, String> {
        let chunk = self
            .logslope_design
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("logslope_surface_row logslope row chunk: {e}"))?;
        Ok(chunk.row(0).to_owned())
    }

    fn shared_logslope_covariance_scale(&self) -> Result<f64, String> {
        let k = self.score_dim();
        if k == 1 {
            return Ok(1.0);
        }
        let ones = vec![1.0; k];
        self.score_covariance.quadratic_form(&ones).map_err(|err| {
            format!("survival marginal-slope shared log-slope covariance scale: {err}")
        })
    }

    fn exact_shared_score_summary(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        context: &str,
    ) -> Result<(f64, f64), String> {
        let k = self.score_dim();
        if k == 1 {
            return Ok((self.z[[row, 0]], 1.0));
        }
        let logslope_eta_len = block_states[2].eta.len();
        if logslope_eta_len != self.n {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "{context}: survival marginal-slope exact shared-slope calculus for K={k} requires one log-slope eta per row (n={}); got eta len {logslope_eta_len}. Per-z log-slope derivatives require a {}-primary row kernel.",
                    self.n,
                    3 + k
                ),
            }
            .into());
        }
        Ok((
            self.z.row(row).sum(),
            self.shared_logslope_covariance_scale()?,
        ))
    }

    fn row_primary_closed_form_rigid(
        &self,
        row: usize,
        q0: f64,
        q1: f64,
        qd1: f64,
        block_states: &[ParameterBlockState],
        probit_scale: f64,
    ) -> Result<(f64, [f64; N_PRIMARY], [[f64; N_PRIMARY]; N_PRIMARY]), String> {
        let logslope_eta = &block_states[2].eta;
        let k = self.score_dim();
        if k == 1 {
            return row_primary_closed_form(
                q0,
                q1,
                qd1,
                logslope_eta[row],
                self.z[[row, 0]],
                self.weights[row],
                self.event[row],
                self.derivative_guard,
                probit_scale,
            )
            .map_err(|err| with_row_context(err, row));
        }
        if logslope_eta.len() != self.n {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival marginal-slope exact rigid row calculus for K={k} requires one shared log-slope surface (eta len n={}); got eta len {}. Per-z log-slope derivatives require a {}-primary row kernel.",
                    self.n,
                    logslope_eta.len(),
                    3 + k
                ),
            }
            .into());
        }
        let (z_sum, covariance_ones) =
            self.exact_shared_score_summary(row, block_states, "row_primary_closed_form_rigid")?;
        row_primary_closed_form_shared_score(
            q0,
            q1,
            qd1,
            logslope_eta[row],
            z_sum,
            covariance_ones,
            self.weights[row],
            self.event[row],
            self.derivative_guard,
            probit_scale,
        )
        .map_err(|err| with_row_context(err, row))
    }

    fn ensure_scalar_flex_exact_score_geometry(&self, context: &str) -> Result<(), String> {
        if self.score_dim() == 1 {
            return Ok(());
        }
        Err(SurvivalMarginalSlopeError::UnsupportedConfiguration {
            reason: format!(
                "{context}: survival marginal-slope exact flexible row calculus is scalar-z only; K={} must use the rigid shared-slope vector kernel or a widened per-z primary kernel",
                self.score_dim()
            ),
        }
        .into())
    }

    fn row_neglog_rigid_vector_value(
        &self,
        row: usize,
        q_geom: SurvivalMarginalSlopeDynamicRowValues,
        block_states: &[ParameterBlockState],
        probit_scale: f64,
    ) -> Result<f64, String> {
        let slopes = if self.per_z_logslope_active() {
            self.logslope_surface_values_for_row(row, &block_states[2].beta)?
        } else {
            self.logslope_vector_for_row(row, &block_states[2].eta)?
        };
        let z = self.z.row(row).to_vec();
        survival_marginal_slope_vector_neglog(
            q_geom.q0,
            q_geom.q1,
            q_geom.qd1,
            &slopes,
            &z,
            &self.score_covariance,
            self.weights[row],
            self.event[row],
            self.derivative_guard,
            probit_scale,
        )
    }

    /// Two-phase auto-subsample entry: when `options.auto_outer_subsample` is
    /// enabled, an outer-derivative-scoped `OuterEvalContext` is present, and
    /// Phase 1 still has budget, this returns a cloned `BlockwiseFitOptions`
    /// carrying a freshly built stratified Horvitz-Thompson mask. Otherwise
    /// returns `None` and the caller uses the original options unchanged.
    ///
    /// Keying is on the outer ρ published by the smoothing optimizer through
    /// `options.outer_eval_context` — never on the inner β. During inner
    /// trust-region / joint-Newton trial steps β changes between calls at the
    /// same outer ρ, so β-keying would re-fire phase prints and rebuild the
    /// row mask inside one outer eval, which makes the trust-region ratio
    /// compare objectives evaluated on different row measures (invalid).
    /// Inner-scope contexts (set by `coefficient_line_search_options`) make
    /// this entry return `None` immediately.
    ///
    /// CONTRACT: MUST NOT be called from inner-coefficient paths (line-search,
    /// trust-region globalization). The InnerCoefficient scope guard below is
    /// the enforcement mechanism; this comment makes the contract obvious.
    /// Cf. `src/solver/row_measure.rs` and the TR row-measure invariant in
    /// `inner_blockwise_fit`.
    fn install_auto_outer_subsample_options(
        &self,
        options: &BlockwiseFitOptions,
    ) -> Option<BlockwiseFitOptions> {
        let ctx = options.outer_eval_context.as_ref()?;
        if !matches!(ctx.scope, crate::custom_family::EvalScope::OuterDerivative) {
            return None;
        }
        let event_secondary: Vec<u8> = self
            .event
            .iter()
            .map(|v| if *v > 0.5 { 1u8 } else { 0u8 })
            .collect();
        let z_key = self.z_subsample_key();
        crate::families::marginal_slope_shared::maybe_install_auto_outer_subsample(
            options,
            z_key.as_slice().expect("z key must be contiguous"),
            Some(&event_secondary),
            ctx.rho.as_slice().expect("outer rho must be contiguous"),
            &self.auto_subsample_phase_counter,
            &self.auto_subsample_last_rho,
            SURVIVAL_MGS_AUTO_SUBSAMPLE_PHASE1_BUDGET,
            "survival-mgs",
            // Per-K work-unit cost for the survival marginal-slope outer
            // gradient kernel. Calibrated from the large-scale repro
            // (n=195_780, K=19_661, predicted_gradient_work ≈ 4.33×10⁹):
            //   per_K-unit cost ≈ 4.33e9 / 19_661 ≈ 220_000 units.
            // With `AUTO_OUTER_WORK_BUDGET = 5×10⁸`, this caps
            //   K_work ≈ 5e8 / 250_000 ≈ 2_000,
            // bounding outer gradient work below ~5×10⁸ units even
            // when the noise-only rule would request K ≈ 0.1n. Without
            // this cap the rigid pilot and outer line search spend
            // ~57 minutes per evaluation on large-scale joint designs
            // before the identifiability gate even gets a chance to
            // veto rank-deficient configurations.
            250_000,
        )
    }

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
            let total: Result<f64, String> = row_iter
                .into_par_iter()
                .try_fold(
                    || 0.0,
                    |mut ll, weighted| -> Result<_, String> {
                        ll -= weighted.weight
                            * self.row_neglog_flex_value(weighted.index, block_states)?;
                        Ok(ll)
                    },
                )
                .try_reduce(
                    || 0.0,
                    |left, right| -> Result<_, String> { Ok(left + right) },
                );
            return total;
        }
        // True fast path: K=1 keeps the original scalar closed form; K>1
        // uses the covariance-aware vector likelihood. Exact rigid
        // gradient/Hessian paths below use the same c(a)=sqrt(1+r'Sigma r)
        // algebra through `row_primary_closed_form_rigid`.
        let guard = self.derivative_guard;
        let probit_scale = self.probit_frailty_scale();
        let score_dim = self.score_dim();
        let total: Result<f64, String> = row_iter
            .into_par_iter()
            .try_fold(
                || 0.0,
                |mut ll, weighted| -> Result<_, String> {
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
                        return Ok(ll);
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
                    Ok(ll)
                },
            )
            .try_reduce(
                || 0.0,
                |left, right| -> Result<_, String> { Ok(left + right) },
            );
        total
    }

    fn is_sigma_aux_index(
        &self,
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> bool {
        shared_is_sigma_aux_index(self.gaussian_frailty_sd, derivative_blocks, psi_index)
    }

    fn sigma_scale_jet(
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

    fn row_neglog_directional_with_scale_jet(
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

    fn row_sigma_primary_terms(
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

    fn sigma_exact_joint_psi_terms(
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

    fn sigma_exact_joint_psisecond_order_terms(
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

    fn sigma_exact_joint_psihessian_directional_derivative(
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

    fn flex_timewiggle_active(&self) -> bool {
        self.time_wiggle_ncols > 0
    }

    fn time_wiggle_range(&self) -> std::ops::Range<usize> {
        let p_total = self.design_exit.ncols();
        let p_w = self.time_wiggle_ncols.min(p_total);
        (p_total - p_w)..p_total
    }

    fn time_wiggle_first_order_geometry(
        &self,
        h0: ndarray::ArrayView1<'_, f64>,
        beta_w: ndarray::ArrayView1<'_, f64>,
    ) -> Result<Option<SurvivalTimeWiggleFirstOrderGeometry>, String> {
        let (Some(knots), Some(degree)) =
            (self.time_wiggle_knots.as_ref(), self.time_wiggle_degree)
        else {
            return Ok(None);
        };
        let basis = monotone_wiggle_basis_with_derivative_order(h0, knots, degree, 0)?;
        let basis_d1 = monotone_wiggle_basis_with_derivative_order(h0, knots, degree, 1)?;
        let basis_d2 = monotone_wiggle_basis_with_derivative_order(h0, knots, degree, 2)?;
        if basis.ncols() != beta_w.len()
            || basis_d1.ncols() != beta_w.len()
            || basis_d2.ncols() != beta_w.len()
        {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival marginal-slope timewiggle basis/beta mismatch: B..B''={},{},{} betaw={}",
                    basis.ncols(),
                    basis_d1.ncols(),
                    basis_d2.ncols(),
                    beta_w.len()
                ),
            }
            .into());
        }
        let dq_dq0 = fast_av(&basis_d1, &beta_w) + 1.0;
        let d2q_dq02 = fast_av(&basis_d2, &beta_w);
        Ok(Some(SurvivalTimeWiggleFirstOrderGeometry {
            basis,
            basis_d1,
            basis_d2,
            dq_dq0,
            d2q_dq02,
        }))
    }

    fn time_wiggle_geometry(
        &self,
        h0: ndarray::ArrayView1<'_, f64>,
        beta_w: ndarray::ArrayView1<'_, f64>,
    ) -> Result<Option<SurvivalTimeWiggleGeometry>, String> {
        let first = self.time_wiggle_first_order_geometry(h0, beta_w)?;
        let Some(first) = first else {
            return Ok(None);
        };
        let (Some(knots), Some(degree)) =
            (self.time_wiggle_knots.as_ref(), self.time_wiggle_degree)
        else {
            return Ok(None);
        };
        let basis_d3 = monotone_wiggle_basis_with_derivative_order(h0, knots, degree, 3)?;
        let basis_d4 = monotone_wiggle_basis_with_derivative_order(h0, knots, degree, 4)?;
        let basis_d5 = monotone_wiggle_basis_with_derivative_order(h0, knots, degree, 5)?;
        if basis_d3.ncols() != beta_w.len()
            || basis_d4.ncols() != beta_w.len()
            || basis_d5.ncols() != beta_w.len()
        {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival marginal-slope timewiggle high-order basis/beta mismatch: B'''..B'''''={},{},{} betaw={}",
                    basis_d3.ncols(),
                    basis_d4.ncols(),
                    basis_d5.ncols(),
                    beta_w.len()
                ),
            }
            .into());
        }
        let d3q_dq03 = fast_av(&basis_d3, &beta_w);
        let d4q_dq04 = fast_av(&basis_d4, &beta_w);
        let d5q_dq05 = fast_av(&basis_d5, &beta_w);
        Ok(Some(SurvivalTimeWiggleGeometry {
            basis: first.basis,
            basis_d1: first.basis_d1,
            basis_d2: first.basis_d2,
            basis_d3,
            basis_d4,
            dq_dq0: first.dq_dq0,
            d2q_dq02: first.d2q_dq02,
            d3q_dq03,
            d4q_dq04,
            d5q_dq05,
        }))
    }

    fn row_dynamic_q_values(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
    ) -> Result<SurvivalMarginalSlopeDynamicRowValues, String> {
        let beta_time = &block_states[0].beta;
        if !self.flex_timewiggle_active() {
            return Ok(SurvivalMarginalSlopeDynamicRowValues {
                q0: self.design_entry.dot_row(row, beta_time)
                    + self.offset_entry[row]
                    + block_states[1].eta[row],
                q1: self.design_exit.dot_row(row, beta_time)
                    + self.offset_exit[row]
                    + block_states[1].eta[row],
                qd1: self.design_derivative_exit.dot_row(row, beta_time)
                    + self.derivative_offset_exit[row],
            });
        }

        let time_tail = self.time_wiggle_range();
        let p_base = time_tail.start;
        let beta_time_base = beta_time.slice(s![..p_base]);
        let beta_time_w = beta_time.slice(s![time_tail.clone()]);
        let entry_chunk = self
            .design_entry
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("row_dynamic_q_values design_entry: {e}"))?;
        let exit_chunk = self
            .design_exit
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("row_dynamic_q_values design_exit: {e}"))?;
        let deriv_chunk = self
            .design_derivative_exit
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("row_dynamic_q_values design_derivative_exit: {e}"))?;
        let x_entry_base = entry_chunk.row(0).slice(s![..p_base]).to_owned();
        let x_exit_base = exit_chunk.row(0).slice(s![..p_base]).to_owned();
        let x_deriv_base = deriv_chunk.row(0).slice(s![..p_base]).to_owned();

        let base_marginal = block_states[1].eta[row];
        let h0 = x_entry_base.dot(&beta_time_base) + self.offset_entry[row] + base_marginal;
        let h1 = x_exit_base.dot(&beta_time_base) + self.offset_exit[row] + base_marginal;
        let d_raw = x_deriv_base.dot(&beta_time_base) + self.derivative_offset_exit[row];

        let entry_geom = self
            .time_wiggle_first_order_geometry(Array1::from_vec(vec![h0]).view(), beta_time_w)?
            .ok_or_else(|| {
                "survival marginal-slope timewiggle metadata is present but value geometry could not be built at entry"
                    .to_string()
            })?;
        let exit_geom = self
            .time_wiggle_first_order_geometry(Array1::from_vec(vec![h1]).view(), beta_time_w)?
            .ok_or_else(|| {
                "survival marginal-slope timewiggle metadata is present but value geometry could not be built at exit"
                    .to_string()
            })?;

        Ok(SurvivalMarginalSlopeDynamicRowValues {
            q0: h0 + entry_geom.basis.row(0).dot(&beta_time_w),
            q1: h1 + exit_geom.basis.row(0).dot(&beta_time_w),
            qd1: exit_geom.dq_dq0[0] * d_raw,
        })
    }

    fn row_dynamic_q_gradient(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
    ) -> Result<SurvivalMarginalSlopeDynamicRowGradient, String> {
        let beta_time = &block_states[0].beta;
        let beta_marginal = &block_states[1].beta;
        let p_time = beta_time.len();
        let p_marginal = beta_marginal.len();

        let mut out = SurvivalMarginalSlopeDynamicRowGradient {
            q0: 0.0,
            q1: 0.0,
            qd1: 0.0,
            dq0_time: Array1::zeros(p_time),
            dq1_time: Array1::zeros(p_time),
            dqd1_time: Array1::zeros(p_time),
            dq0_marginal: Array1::zeros(p_marginal),
            dq1_marginal: Array1::zeros(p_marginal),
            dqd1_marginal: Array1::zeros(p_marginal),
        };

        if !self.flex_timewiggle_active() {
            out.q0 = self.design_entry.dot_row(row, beta_time)
                + self.offset_entry[row]
                + block_states[1].eta[row];
            out.q1 = self.design_exit.dot_row(row, beta_time)
                + self.offset_exit[row]
                + block_states[1].eta[row];
            out.qd1 = self.design_derivative_exit.dot_row(row, beta_time)
                + self.derivative_offset_exit[row];
            let time_entry_chunk = self
                .design_entry
                .try_row_chunk(row..row + 1)
                .map_err(|e| format!("row_dynamic_q_gradient design_entry: {e}"))?;
            let time_exit_chunk = self
                .design_exit
                .try_row_chunk(row..row + 1)
                .map_err(|e| format!("row_dynamic_q_gradient design_exit: {e}"))?;
            let time_deriv_chunk = self
                .design_derivative_exit
                .try_row_chunk(row..row + 1)
                .map_err(|e| format!("row_dynamic_q_gradient design_derivative_exit: {e}"))?;
            out.dq0_time.assign(&time_entry_chunk.row(0));
            out.dq1_time.assign(&time_exit_chunk.row(0));
            out.dqd1_time.assign(&time_deriv_chunk.row(0));
            if p_marginal > 0 {
                let marginal_chunk = self
                    .marginal_design
                    .try_row_chunk(row..row + 1)
                    .map_err(|e| format!("row_dynamic_q_gradient marginal_design: {e}"))?;
                let marginal_row = marginal_chunk.row(0);
                out.dq0_marginal.assign(&marginal_row);
                out.dq1_marginal.assign(&marginal_row);
            }
            return Ok(out);
        }

        let time_tail = self.time_wiggle_range();
        let p_base = time_tail.start;
        let beta_time_base = beta_time.slice(s![..p_base]);
        let beta_time_w = beta_time.slice(s![time_tail.clone()]);
        let entry_chunk = self
            .design_entry
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("row_dynamic_q_gradient design_entry: {e}"))?;
        let exit_chunk = self
            .design_exit
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("row_dynamic_q_gradient design_exit: {e}"))?;
        let deriv_chunk = self
            .design_derivative_exit
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("row_dynamic_q_gradient design_derivative_exit: {e}"))?;
        let x_entry_base = entry_chunk.row(0).slice(s![..p_base]).to_owned();
        let x_exit_base = exit_chunk.row(0).slice(s![..p_base]).to_owned();
        let x_deriv_base = deriv_chunk.row(0).slice(s![..p_base]).to_owned();
        let marginal_row = if p_marginal > 0 {
            let marginal_chunk = self
                .marginal_design
                .try_row_chunk(row..row + 1)
                .map_err(|e| format!("row_dynamic_q_gradient marginal_design: {e}"))?;
            Some(marginal_chunk.row(0).to_owned())
        } else {
            None
        };

        let base_marginal = block_states[1].eta[row];
        let h0 = x_entry_base.dot(&beta_time_base) + self.offset_entry[row] + base_marginal;
        let h1 = x_exit_base.dot(&beta_time_base) + self.offset_exit[row] + base_marginal;
        let d_raw = x_deriv_base.dot(&beta_time_base) + self.derivative_offset_exit[row];

        let entry_geom = self
            .time_wiggle_first_order_geometry(Array1::from_vec(vec![h0]).view(), beta_time_w)?
            .ok_or_else(|| {
                "survival marginal-slope timewiggle metadata is present but gradient geometry could not be built at entry"
                    .to_string()
            })?;
        let exit_geom = self
            .time_wiggle_first_order_geometry(Array1::from_vec(vec![h1]).view(), beta_time_w)?
            .ok_or_else(|| {
                "survival marginal-slope timewiggle metadata is present but gradient geometry could not be built at exit"
                    .to_string()
            })?;

        out.q0 = h0 + entry_geom.basis.row(0).dot(&beta_time_w);
        out.q1 = h1 + exit_geom.basis.row(0).dot(&beta_time_w);
        out.qd1 = exit_geom.dq_dq0[0] * d_raw;

        for j in 0..p_base {
            out.dq0_time[j] = entry_geom.dq_dq0[0] * x_entry_base[j];
            out.dq1_time[j] = exit_geom.dq_dq0[0] * x_exit_base[j];
            out.dqd1_time[j] = exit_geom.d2q_dq02[0] * d_raw * x_exit_base[j]
                + exit_geom.dq_dq0[0] * x_deriv_base[j];
        }
        for local_idx in 0..time_tail.len() {
            let coeff_idx = time_tail.start + local_idx;
            out.dq0_time[coeff_idx] = entry_geom.basis[[0, local_idx]];
            out.dq1_time[coeff_idx] = exit_geom.basis[[0, local_idx]];
            out.dqd1_time[coeff_idx] = exit_geom.basis_d1[[0, local_idx]] * d_raw;
        }

        if let Some(marginal_row) = marginal_row.as_ref() {
            for j in 0..p_marginal {
                out.dq0_marginal[j] = entry_geom.dq_dq0[0] * marginal_row[j];
                out.dq1_marginal[j] = exit_geom.dq_dq0[0] * marginal_row[j];
                out.dqd1_marginal[j] = exit_geom.d2q_dq02[0] * d_raw * marginal_row[j];
            }
        }

        Ok(out)
    }

    fn row_dynamic_q_geometry(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
    ) -> Result<SurvivalMarginalSlopeDynamicRow, String> {
        let mut out = SurvivalMarginalSlopeDynamicRow::empty_workspace();
        self.row_dynamic_q_geometry_into(row, block_states, &mut out)?;
        Ok(out)
    }

    /// Pooled-buffer variant of [`row_dynamic_q_geometry`]. Resizes and
    /// zero-fills `out` in place, then writes the same row geometry into the
    /// caller-owned workspace. Used by hot rayon `try_fold` accumulators that
    /// thread one `SurvivalMarginalSlopeDynamicRow` per worker thread to
    /// eliminate the ~250 KB-per-call allocator traffic the fresh-allocation
    /// path would incur at large-scale `n`.
    fn row_dynamic_q_geometry_into(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        out: &mut SurvivalMarginalSlopeDynamicRow,
    ) -> Result<(), String> {
        let beta_time = &block_states[0].beta;
        let beta_marginal = &block_states[1].beta;
        let p_time = beta_time.len();
        let p_marginal = beta_marginal.len();

        out.reset(p_time, p_marginal);

        if !self.flex_timewiggle_active() {
            out.q0 = self.design_entry.dot_row(row, beta_time)
                + self.offset_entry[row]
                + block_states[1].eta[row];
            out.q1 = self.design_exit.dot_row(row, beta_time)
                + self.offset_exit[row]
                + block_states[1].eta[row];
            out.qd1 = self.design_derivative_exit.dot_row(row, beta_time)
                + self.derivative_offset_exit[row];
            let time_entry_chunk = self
                .design_entry
                .try_row_chunk(row..row + 1)
                .map_err(|e| format!("row_dynamic_q_geometry design_entry: {e}"))?;
            let time_exit_chunk = self
                .design_exit
                .try_row_chunk(row..row + 1)
                .map_err(|e| format!("row_dynamic_q_geometry design_exit: {e}"))?;
            let time_deriv_chunk = self
                .design_derivative_exit
                .try_row_chunk(row..row + 1)
                .map_err(|e| format!("row_dynamic_q_geometry design_derivative_exit: {e}"))?;
            // Perf (#large-scale): assign the design rows directly from the chunk
            // views into the reused `out` buffers. The previous `.to_owned()`
            // round-trip allocated three fresh `Array1<f64>` per row only to
            // immediately copy them in — removing it drops O(n·p_time) heap
            // traffic with bit-identical values.
            out.dq0_time.assign(&time_entry_chunk.row(0));
            out.dq1_time.assign(&time_exit_chunk.row(0));
            out.dqd1_time.assign(&time_deriv_chunk.row(0));
            if p_marginal > 0 {
                let marginal_chunk = self
                    .marginal_design
                    .try_row_chunk(row..row + 1)
                    .map_err(|e| format!("row_dynamic_q_geometry marginal_design: {e}"))?;
                let marginal_row = marginal_chunk.row(0);
                out.dq0_marginal.assign(&marginal_row);
                out.dq1_marginal.assign(&marginal_row);
            }
            return Ok(());
        }

        let time_tail = self.time_wiggle_range();
        let p_base = time_tail.start;
        let beta_time_base = beta_time.slice(s![..p_base]);
        let beta_time_w = beta_time.slice(s![time_tail.clone()]);
        let entry_chunk = self
            .design_entry
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("row_dynamic_q_geometry design_entry: {e}"))?;
        let exit_chunk = self
            .design_exit
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("row_dynamic_q_geometry design_exit: {e}"))?;
        let deriv_chunk = self
            .design_derivative_exit
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("row_dynamic_q_geometry design_derivative_exit: {e}"))?;
        // Perf (#large-scale): hold the base/marginal design rows as borrowed views
        // into the chunk storage rather than `.to_owned()` copies. Every use
        // below is either a `.dot(...)` or scalar indexing, both of which work
        // directly on `ArrayView1`, so this is bit-identical while removing the
        // per-row Array1 allocations.
        let entry_row_view = entry_chunk.row(0);
        let exit_row_view = exit_chunk.row(0);
        let deriv_row_view = deriv_chunk.row(0);
        let x_entry_base = entry_row_view.slice(s![..p_base]);
        let x_exit_base = exit_row_view.slice(s![..p_base]);
        let x_deriv_base = deriv_row_view.slice(s![..p_base]);
        let marginal_chunk = if p_marginal > 0 {
            Some(
                self.marginal_design
                    .try_row_chunk(row..row + 1)
                    .map_err(|e| format!("row_dynamic_q_geometry marginal_design: {e}"))?,
            )
        } else {
            None
        };
        let marginal_row = marginal_chunk.as_ref().map(|chunk| chunk.row(0));

        let base_marginal = block_states[1].eta[row];
        let h0 = x_entry_base.dot(&beta_time_base) + self.offset_entry[row] + base_marginal;
        let h1 = x_exit_base.dot(&beta_time_base) + self.offset_exit[row] + base_marginal;
        let d_raw = x_deriv_base.dot(&beta_time_base) + self.derivative_offset_exit[row];

        let entry_geom = self
            .time_wiggle_geometry(Array1::from_vec(vec![h0]).view(), beta_time_w)?
            .ok_or_else(|| {
                "survival marginal-slope timewiggle metadata is present but geometry could not be built at entry"
                    .to_string()
            })?;
        let exit_geom = self
            .time_wiggle_geometry(Array1::from_vec(vec![h1]).view(), beta_time_w)?
            .ok_or_else(|| {
                "survival marginal-slope timewiggle metadata is present but geometry could not be built at exit"
                    .to_string()
            })?;

        out.q0 = h0 + entry_geom.basis.row(0).dot(&beta_time_w);
        out.q1 = h1 + exit_geom.basis.row(0).dot(&beta_time_w);
        out.qd1 = exit_geom.dq_dq0[0] * d_raw;

        for j in 0..p_base {
            out.dq0_time[j] = entry_geom.dq_dq0[0] * x_entry_base[j];
            out.dq1_time[j] = exit_geom.dq_dq0[0] * x_exit_base[j];
            out.dqd1_time[j] = exit_geom.d2q_dq02[0] * d_raw * x_exit_base[j]
                + exit_geom.dq_dq0[0] * x_deriv_base[j];
            for k in 0..p_base {
                out.d2q0_time_time[[j, k]] =
                    entry_geom.d2q_dq02[0] * x_entry_base[j] * x_entry_base[k];
                out.d2q1_time_time[[j, k]] =
                    exit_geom.d2q_dq02[0] * x_exit_base[j] * x_exit_base[k];
                out.d2qd1_time_time[[j, k]] =
                    exit_geom.d3q_dq03[0] * d_raw * x_exit_base[j] * x_exit_base[k]
                        + exit_geom.d2q_dq02[0]
                            * (x_exit_base[j] * x_deriv_base[k] + x_deriv_base[j] * x_exit_base[k]);
            }
        }
        for local_idx in 0..time_tail.len() {
            let coeff_idx = time_tail.start + local_idx;
            out.dq0_time[coeff_idx] = entry_geom.basis[[0, local_idx]];
            out.dq1_time[coeff_idx] = exit_geom.basis[[0, local_idx]];
            out.dqd1_time[coeff_idx] = exit_geom.basis_d1[[0, local_idx]] * d_raw;
            for j in 0..p_base {
                let q0_tw = entry_geom.basis_d1[[0, local_idx]] * x_entry_base[j];
                let q1_tw = exit_geom.basis_d1[[0, local_idx]] * x_exit_base[j];
                out.d2q0_time_time[[j, coeff_idx]] = q0_tw;
                out.d2q0_time_time[[coeff_idx, j]] = q0_tw;
                out.d2q1_time_time[[j, coeff_idx]] = q1_tw;
                out.d2q1_time_time[[coeff_idx, j]] = q1_tw;
                let qd1_cross = exit_geom.basis_d2[[0, local_idx]] * d_raw * x_exit_base[j]
                    + exit_geom.basis_d1[[0, local_idx]] * x_deriv_base[j];
                out.d2qd1_time_time[[j, coeff_idx]] = qd1_cross;
                out.d2qd1_time_time[[coeff_idx, j]] = qd1_cross;
            }
        }

        if let Some(marginal_row) = marginal_row.as_ref() {
            for j in 0..p_marginal {
                out.dq0_marginal[j] = entry_geom.dq_dq0[0] * marginal_row[j];
                out.dq1_marginal[j] = exit_geom.dq_dq0[0] * marginal_row[j];
                out.dqd1_marginal[j] = exit_geom.d2q_dq02[0] * d_raw * marginal_row[j];
                for k in 0..p_marginal {
                    out.d2q0_marginal_marginal[[j, k]] =
                        entry_geom.d2q_dq02[0] * marginal_row[j] * marginal_row[k];
                    out.d2q1_marginal_marginal[[j, k]] =
                        exit_geom.d2q_dq02[0] * marginal_row[j] * marginal_row[k];
                    out.d2qd1_marginal_marginal[[j, k]] =
                        exit_geom.d3q_dq03[0] * d_raw * marginal_row[j] * marginal_row[k];
                }
                for k in 0..p_base {
                    out.d2q0_time_marginal[[k, j]] =
                        entry_geom.d2q_dq02[0] * x_entry_base[k] * marginal_row[j];
                    out.d2q1_time_marginal[[k, j]] =
                        exit_geom.d2q_dq02[0] * x_exit_base[k] * marginal_row[j];
                    out.d2qd1_time_marginal[[k, j]] =
                        exit_geom.d3q_dq03[0] * d_raw * x_exit_base[k] * marginal_row[j]
                            + exit_geom.d2q_dq02[0] * x_deriv_base[k] * marginal_row[j];
                }
                for local_idx in 0..time_tail.len() {
                    let coeff_idx = time_tail.start + local_idx;
                    out.d2q0_time_marginal[[coeff_idx, j]] =
                        entry_geom.basis_d1[[0, local_idx]] * marginal_row[j];
                    out.d2q1_time_marginal[[coeff_idx, j]] =
                        exit_geom.basis_d1[[0, local_idx]] * marginal_row[j];
                    out.d2qd1_time_marginal[[coeff_idx, j]] =
                        exit_geom.basis_d2[[0, local_idx]] * d_raw * marginal_row[j];
                }
            }
        }

        Ok(())
    }

    fn timewiggle_marginal_psi_row_lift(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        primary_layout: Option<&FlexPrimarySlices>,
        psi_row: &Array1<f64>,
        beta_marginal: &Array1<f64>,
    ) -> Result<TimewiggleMarginalPsiRowLift, String> {
        let beta_time = &block_states[0].beta;
        let p_time = beta_time.len();
        let time_tail = self.time_wiggle_range();
        let p_base = time_tail.start;
        let beta_time_w = beta_time.slice(s![time_tail.clone()]);
        let entry_chunk = self
            .design_entry
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("timewiggle_marginal_psi_row_lift design_entry: {e}"))?;
        let exit_chunk = self
            .design_exit
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("timewiggle_marginal_psi_row_lift design_exit: {e}"))?;
        let deriv_chunk = self
            .design_derivative_exit
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("timewiggle_marginal_psi_row_lift design_derivative_exit: {e}"))?;
        let x_entry_base = entry_chunk.row(0).slice(s![..p_base]).to_owned();
        let x_exit_base = exit_chunk.row(0).slice(s![..p_base]).to_owned();
        let x_deriv_base = deriv_chunk.row(0).slice(s![..p_base]).to_owned();
        let marginal_chunk = self
            .marginal_design
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("timewiggle_marginal_psi_row_lift marginal_design: {e}"))?;
        let marginal_row = marginal_chunk.row(0).to_owned();

        let base_marginal = block_states[1].eta[row];
        let h0 = x_entry_base.dot(&beta_time.slice(s![..p_base]))
            + self.offset_entry[row]
            + base_marginal;
        let h1 =
            x_exit_base.dot(&beta_time.slice(s![..p_base])) + self.offset_exit[row] + base_marginal;
        let d_raw =
            x_deriv_base.dot(&beta_time.slice(s![..p_base])) + self.derivative_offset_exit[row];
        let entry_geom = self
            .time_wiggle_geometry(Array1::from_vec(vec![h0]).view(), beta_time_w)?
            .ok_or_else(|| "missing entry timewiggle geometry for marginal psi lift".to_string())?;
        let exit_geom = self
            .time_wiggle_geometry(Array1::from_vec(vec![h1]).view(), beta_time_w)?
            .ok_or_else(|| "missing exit timewiggle geometry for marginal psi lift".to_string())?;

        let mu = psi_row.dot(beta_marginal);
        let mut dir =
            Array1::<f64>::zeros(primary_layout.map_or(N_PRIMARY, |primary| primary.total));
        let q0_idx = primary_layout.map_or(0, |primary| primary.q0);
        let q1_idx = primary_layout.map_or(1, |primary| primary.q1);
        let qd1_idx = primary_layout.map_or(2, |primary| primary.qd1);
        dir[q0_idx] = entry_geom.dq_dq0[0] * mu;
        dir[q1_idx] = exit_geom.dq_dq0[0] * mu;
        dir[qd1_idx] = exit_geom.d2q_dq02[0] * d_raw * mu;

        let mut u_q0_time = Array1::<f64>::zeros(p_time);
        let mut u_q1_time = Array1::<f64>::zeros(p_time);
        let mut u_qd1_time = Array1::<f64>::zeros(p_time);
        for j in 0..p_base {
            u_q0_time[j] = entry_geom.d2q_dq02[0] * mu * x_entry_base[j];
            u_q1_time[j] = exit_geom.d2q_dq02[0] * mu * x_exit_base[j];
            u_qd1_time[j] = mu
                * (exit_geom.d3q_dq03[0] * d_raw * x_exit_base[j]
                    + exit_geom.d2q_dq02[0] * x_deriv_base[j]);
        }
        for local_idx in 0..time_tail.len() {
            let coeff_idx = time_tail.start + local_idx;
            u_q0_time[coeff_idx] = entry_geom.basis_d1[[0, local_idx]] * mu;
            u_q1_time[coeff_idx] = exit_geom.basis_d1[[0, local_idx]] * mu;
            u_qd1_time[coeff_idx] = exit_geom.basis_d2[[0, local_idx]] * d_raw * mu;
        }

        let u_q0_marginal =
            psi_row * entry_geom.dq_dq0[0] + &marginal_row * (entry_geom.d2q_dq02[0] * mu);
        let u_q1_marginal =
            psi_row * exit_geom.dq_dq0[0] + &marginal_row * (exit_geom.d2q_dq02[0] * mu);
        let u_qd1_marginal = psi_row * (exit_geom.d2q_dq02[0] * d_raw)
            + &marginal_row * (exit_geom.d3q_dq03[0] * d_raw * mu);

        Ok(TimewiggleMarginalPsiRowLift {
            dir,
            u_q0_time,
            u_q1_time,
            u_qd1_time,
            u_q0_marginal,
            u_q1_marginal,
            u_qd1_marginal,
            x_entry_base,
            x_exit_base,
            x_deriv_base,
            marginal_row,
            entry_basis_d1: entry_geom.basis_d1.row(0).to_owned(),
            entry_basis_d2: entry_geom.basis_d2.row(0).to_owned(),
            exit_basis_d1: exit_geom.basis_d1.row(0).to_owned(),
            exit_basis_d2: exit_geom.basis_d2.row(0).to_owned(),
            exit_basis_d3: exit_geom.basis_d3.row(0).to_owned(),
            entry_m2: entry_geom.d2q_dq02[0],
            entry_m3: entry_geom.d3q_dq03[0],
            exit_m2: exit_geom.d2q_dq02[0],
            exit_m3: exit_geom.d3q_dq03[0],
            exit_m4: exit_geom.d4q_dq04[0],
            d_raw,
            mu,
            psi_row: psi_row.clone(),
        })
    }

    fn time_derivative_lower_bound(&self) -> f64 {
        assert!(
            self.derivative_guard.is_finite() && self.derivative_guard > 0.0,
            "survival marginal-slope derivative guard must be finite and positive: derivative_guard={}",
            self.derivative_guard
        );
        self.derivative_guard
    }

    fn flex_active(&self) -> bool {
        // The absorbed influence block (#461) rides the dynamic-Q primary-jet
        // path (it adds the `o_infl` primary coordinate), so it counts as "flex"
        // for dispatch purposes even when no score_warp / link_dev is present —
        // the rigid closed-form row kernel has no `o_infl` channel.
        self.score_warp.is_some() || self.link_dev.is_some() || self.influence_absorber.is_some()
    }

    fn effective_flex_active(&self, block_states: &[ParameterBlockState]) -> Result<bool, String> {
        if self.score_warp.is_some() && self.flex_score_beta(block_states)?.is_none() {
            return Err(SurvivalMarginalSlopeError::InvalidInput {
                reason: "missing survival score-warp block state".to_string(),
            }
            .into());
        }
        if self.link_dev.is_some() && self.flex_link_beta(block_states)?.is_none() {
            return Err(SurvivalMarginalSlopeError::InvalidInput {
                reason: "missing survival link-deviation block state".to_string(),
            }
            .into());
        }
        if self.influence_absorber.is_some() && self.flex_influence_beta(block_states)?.is_none() {
            return Err(SurvivalMarginalSlopeError::InvalidInput {
                reason: "missing survival influence-absorber block state".to_string(),
            }
            .into());
        }
        Ok(self.flex_active())
    }

    fn flex_score_beta<'a>(
        &self,
        block_states: &'a [ParameterBlockState],
    ) -> Result<Option<&'a Array1<f64>>, String> {
        if self.score_warp.is_none() {
            return Ok(None);
        }
        block_states
            .get(3)
            .map(|state| Some(&state.beta))
            .ok_or_else(|| "missing survival score-warp block state".to_string())
    }

    fn flex_link_beta<'a>(
        &self,
        block_states: &'a [ParameterBlockState],
    ) -> Result<Option<&'a Array1<f64>>, String> {
        if self.link_dev.is_none() {
            return Ok(None);
        }
        let idx = if self.score_warp.is_some() { 4 } else { 3 };
        block_states
            .get(idx)
            .map(|state| Some(&state.beta))
            .ok_or_else(|| "missing survival link-deviation block state".to_string())
    }

    /// Coefficient `γ` of the absorbed Stage-1 influence block (#461). The
    /// absorber is the trailing block, so its index is `3 + score_warp? +
    /// link_dev?`. `None` when no influence Jacobian was installed.
    fn flex_influence_beta<'a>(
        &self,
        block_states: &'a [ParameterBlockState],
    ) -> Result<Option<&'a Array1<f64>>, String> {
        if self.influence_absorber.is_none() {
            return Ok(None);
        }
        let idx = 3 + usize::from(self.score_warp.is_some()) + usize::from(self.link_dev.is_some());
        block_states
            .get(idx)
            .map(|state| Some(&state.beta))
            .ok_or_else(|| "missing survival influence-absorber block state".to_string())
    }

    /// Per-row absorbed-influence index offset `o_infl[row] = Z̃_infl[row,:]·γ`.
    /// Returns `0.0` when no absorber is installed (the additive shift vanishes),
    /// so callers can fold it unconditionally into the de-nested observed `η₁`.
    fn influence_index_offset(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
    ) -> Result<f64, String> {
        let (Some(z_tilde), Some(gamma)) = (
            self.influence_absorber.as_ref(),
            self.flex_influence_beta(block_states)?,
        ) else {
            return Ok(0.0);
        };
        if gamma.len() != z_tilde.ncols() {
            return Err(format!(
                "survival influence-absorber β length {} != Z̃_infl columns {}",
                gamma.len(),
                z_tilde.ncols()
            ));
        }
        Ok(z_tilde.row(row).dot(gamma))
    }

    fn denested_partition_cells(
        &self,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<Vec<exact_kernel::DenestedPartitionCell>, String> {
        if self.score_dim() == 1 {
            return shared_denested_partition_cells(
                a,
                b,
                self.score_warp.as_ref(),
                beta_h,
                self.link_dev.as_ref(),
                beta_w,
                self.probit_frailty_scale(),
            );
        }
        let score_breaks = self
            .score_warp
            .as_ref()
            .map(|runtime| runtime.breakpoints().to_vec())
            .unwrap_or_default();
        let link_breaks = self
            .link_dev
            .as_ref()
            .map(|runtime| runtime.breakpoints().to_vec())
            .unwrap_or_default();
        let mut cells = exact_kernel::build_denested_partition_cells_with_tails(
            a,
            b,
            &score_breaks,
            &link_breaks,
            |z| self.score_warp_local_cubic_at(beta_h, z),
            |u| {
                if let (Some(runtime), Some(beta_w)) = (self.link_dev.as_ref(), beta_w) {
                    runtime.local_cubic_at(beta_w, u)
                } else {
                    Ok(Self::zero_score_warp_span())
                }
            },
        )?;
        let scale = self.probit_frailty_scale();
        if scale != 1.0 {
            for partition_cell in &mut cells {
                partition_cell.cell.c0 *= scale;
                partition_cell.cell.c1 *= scale;
                partition_cell.cell.c2 *= scale;
                partition_cell.cell.c3 *= scale;
            }
        }
        Ok(cells)
    }

    fn evaluate_denested_survival_calibration(
        &self,
        a: f64,
        q: f64,
        slope: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<(f64, f64, f64), String> {
        let cells = self.denested_partition_cells(a, slope, beta_h, beta_w)?;
        let scale = self.probit_frailty_scale();
        let mut f = -crate::probability::normal_cdf(-q);
        let mut f_a = 0.0;
        let mut f_aa = 0.0;
        for partition_cell in cells {
            let pos_cell = partition_cell.cell;
            let neg_cell = exact_kernel::DenestedCubicCell {
                left: pos_cell.left,
                right: pos_cell.right,
                c0: -pos_cell.c0,
                c1: -pos_cell.c1,
                c2: -pos_cell.c2,
                c3: -pos_cell.c3,
            };
            let state = exact_kernel::evaluate_cell_moments(neg_cell, 9)?;
            f += state.value;
            let (dc_da_pos, _) = exact_kernel::denested_cell_coefficient_partials(
                partition_cell.score_span,
                partition_cell.link_span,
                a,
                slope,
            );
            let (dc_daa_pos, _, _) = exact_kernel::denested_cell_second_partials(
                partition_cell.score_span,
                partition_cell.link_span,
                a,
                slope,
            );
            let dc_da = scale_coeff4(dc_da_pos, -scale);
            let dc_daa = scale_coeff4(dc_daa_pos, -scale);
            f_a += exact_kernel::cell_first_derivative_from_moments(&dc_da, &state.moments)?;
            f_aa += exact_kernel::cell_second_derivative_from_moments(
                neg_cell,
                &dc_da,
                &dc_da,
                &dc_daa,
                &state.moments,
            )?;
        }
        Ok((f, f_a, f_aa))
    }
    fn solve_row_survival_intercept_with_slot(
        &self,
        q: f64,
        slope: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        slot: Option<(usize, SurvivalInterceptSlotKind)>,
    ) -> Result<(f64, f64), String> {
        let eval = |a: f64| -> Result<(f64, f64, f64), String> {
            self.evaluate_denested_survival_calibration(a, q, slope, beta_h, beta_w)
        };
        let probit_scale = self.probit_frailty_scale();
        let a_closed_form = q * rigid_observed_scale(slope, probit_scale) / probit_scale;

        // Prefer the previous PIRLS iter's converged intercept as the initial
        // guess; β changes only a little between consecutive PIRLS iterations,
        // so the previous answer is typically within a few root-solver steps
        // of the new one. If the slot is None (no cache wired) or the stored
        // bits decode to a non-finite value (uninitialised NaN sentinel /
        // stale), fall back to the closed-form rigid seed — preserving the
        // exact pre-warm-start behaviour.
        // Tag the cache entry with a 64-bit hash of (beta_h, beta_w) so that
        // rejected trust-region trials and subsampled probes cannot poison
        // the global per-row root: each trial keys under its own β, so a
        // write at β_A is invisible to a subsequent read at β_B. Consecutive
        // evaluations at the same β share the tag and reuse the warm start.
        let beta_tag = hash_intercept_warm_start_key(beta_h, beta_w);
        let cached_a = slot.and_then(|(row, kind)| {
            self.intercept_warm_starts
                .as_ref()
                .and_then(|cache| cache.load(row, kind, beta_tag))
        });
        let a_init = cached_a.unwrap_or(a_closed_form);
        let mut solve_result = super::monotone_root::solve_monotone_root_detailed(
            eval,
            a_init,
            "survival intercept",
            1e-12,
            64,
            64,
        );
        // If the warm-started solve failed, retry once from the closed-form
        // seed. Cached `a` from a prior PIRLS iter can be far enough from the
        // current root (e.g., after a large β step) that the bracketing search
        // exhausts; the closed-form seed always sits in the correct basin.
        if cached_a.is_some() && solve_result.is_err() {
            solve_result = super::monotone_root::solve_monotone_root_detailed(
                eval,
                a_closed_form,
                "survival intercept",
                1e-12,
                64,
                64,
            );
        }
        // This routine also emits its own format!()-based String errors below
        // (non-finite derivative, residual rejection), so the enclosing return
        // type stays Result<_, String>; convert the typed solver error here.
        let solution = solve_result.map_err(|e| e.to_string())?;
        let a = solution.root;
        // The solver already evaluated `eval` at `solution.root` during the
        // refine loop and returned the resulting `residual` (best_f) and
        // `abs_deriv` (best_abs_deriv). Reusing them here saves one full
        // calibration evaluation per row × 2 (entry + exit) per joint-Newton
        // sweep — at large-scale n=320k this is 640k spared evaluations per pass.
        let residual = solution.residual;
        let abs_deriv = solution.abs_deriv;
        if !abs_deriv.is_finite() || abs_deriv == 0.0 {
            return Err(SurvivalMarginalSlopeError::NumericalFailure {
                reason: format!(
                    "survival marginal-slope intercept solve failed: \
                     zero or non-finite derivative at a={a:.6}"
                ),
            }
            .into());
        }

        let target_survival = crate::probability::normal_cdf(-q);
        let achieved_survival = target_survival + residual;
        let tail_mass = target_survival.min(1.0 - target_survival).max(0.0);
        let probability_tol = SURVIVAL_INTERCEPT_ABS_RESIDUAL_TOL
            .max(SURVIVAL_INTERCEPT_REL_TAIL_RESIDUAL_TOL * tail_mass);
        let mut log_tail_residual = None;
        // Always accept if probability-space residual is within tolerance:
        // a perfectly-converged probability solve (residual=0) is the best
        // achievable answer, and rejecting it because the deep-tail log
        // computation has its own floating-point noise (~6e-8 at |q|>=7)
        // would discard a correct intercept. When tail_mass is small we
        // *additionally* accept tight log-space agreement, so well-resolved
        // tails that drift slightly outside the absolute probability_tol
        // (which can be ulp-bounded) still validate.
        let residual_ok = if tail_mass < SURVIVAL_INTERCEPT_LOG_TAIL_THRESHOLD {
            let probability_pass = residual.abs() <= probability_tol;
            let (achieved_tail, target_log_tail) = if target_survival <= 0.5 {
                let (target_log_survival, _) = signed_probit_logcdf_and_mills_ratio(-q);
                (achieved_survival, target_log_survival)
            } else {
                let (target_log_failure, _) = signed_probit_logcdf_and_mills_ratio(q);
                (1.0 - achieved_survival, target_log_failure)
            };
            let log_pass = if target_log_tail.is_finite()
                && achieved_tail.is_finite()
                && achieved_tail > 0.0
            {
                let log_residual = achieved_tail.ln() - target_log_tail;
                log_tail_residual = Some(log_residual);
                log_residual.abs() <= SURVIVAL_INTERCEPT_REL_TAIL_RESIDUAL_TOL
            } else {
                false
            };
            probability_pass || log_pass
        } else {
            residual.abs() <= probability_tol
        };

        if !residual_ok {
            let log_tail_detail = log_tail_residual
                .map(|value| format!(", log_tail_residual={value:.3e}"))
                .unwrap_or_default();
            return Err(SurvivalMarginalSlopeError::IntegrationFailed {
                reason: format!(
                    "survival marginal-slope intercept solve failed: \
                     residual={residual:.3e} at a={a:.6}, target survival={target_survival:.6e}, \
                     achieved survival={achieved_survival:.6e}, probability_tol={probability_tol:.3e}\
                     {log_tail_detail}"
                ),
            }
            .into());
        }

        // Cache the converged intercept for the next PIRLS iter, if a slot
        // was provided. When `slot` is None this is a no-op, preserving the
        // exact pre-warm-start behaviour. The stamp is the β-tagged key
        // computed above: only future reads at the same β observe this
        // write, so rejected or subsampled trials cannot leak their roots
        // into accepted full-data evaluations.
        if let Some((row, kind)) = slot
            && let Some(cache) = self.intercept_warm_starts.as_ref()
        {
            cache.store(row, kind, a, beta_tag);
        }

        Ok((a, abs_deriv))
    }

    fn max_feasible_time_step(
        &self,
        beta: &Array1<f64>,
        delta: &Array1<f64>,
    ) -> Result<Option<f64>, String> {
        let Some(constraints) = self.effective_time_linear_constraints()? else {
            return Ok(None);
        };
        crate::families::marginal_slope_shared::feasible_step_fraction(
            &constraints,
            beta,
            delta,
            |beta_len, delta_len, expected| {
                SurvivalMarginalSlopeError::IncompatibleDimensions {
                    reason: format!(
                        "survival marginal-slope time-step dimension mismatch: beta={beta_len}, delta={delta_len}, expected {expected}"
                    ),
                }
                .into()
            },
            |row, slack| {
                SurvivalMarginalSlopeError::MonotonicityViolation {
                    reason: format!(
                        "survival marginal-slope current time block violates derivative guard at row {row}: slack={slack:.3e}"
                    ),
                }
                .into()
            },
        )
        .map(Some)
    }

    fn effective_time_linear_constraints(
        &self,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        if let Some(constraints) = self.time_linear_constraints.as_ref() {
            return Ok(Some(constraints.clone()));
        }
        append_timewiggle_tail_nonnegative_constraints(
            time_derivative_guard_constraints(
                &self.design_derivative_exit,
                self.derivative_offset_exit.as_ref(),
                self.derivative_guard,
            )?,
            self.design_exit.ncols(),
            self.time_wiggle_ncols,
        )
    }

    fn score_warp_linear_constraints(
        &self,
        runtime: &DeviationRuntime,
    ) -> Result<LinearInequalityConstraints, String> {
        let scalar = runtime.structural_monotonicity_constraints();
        let basis_dim = runtime.basis_dim();
        if scalar.a.ncols() != basis_dim {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival score-warp scalar constraint width mismatch: constraints={}, basis={basis_dim}",
                    scalar.a.ncols()
                ),
            }
            .into());
        }
        let score_dim = self.score_dim();
        let rows_per_coord = scalar.a.nrows();
        let total_rows = rows_per_coord * score_dim;
        let total_cols = basis_dim * score_dim;
        let mut a = Array2::<f64>::zeros((total_rows, total_cols));
        let mut b = Array1::<f64>::zeros(total_rows);
        for coord in 0..score_dim {
            let row_start = coord * rows_per_coord;
            let col_range = score_warp_component_range(runtime, coord);
            a.slice_mut(s![row_start..row_start + rows_per_coord, col_range])
                .assign(&scalar.a);
            b.slice_mut(s![row_start..row_start + rows_per_coord])
                .assign(&scalar.b);
        }
        Ok(LinearInequalityConstraints::from_paired(a, b))
    }

    fn validate_time_qd1_feasible(&self, beta: &Array1<f64>, label: &str) -> Result<(), String> {
        if beta.is_empty() {
            return Ok(());
        }
        if beta.len() != self.design_derivative_exit.ncols() {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival marginal-slope time-block {label} length mismatch: beta={}, derivative columns={}",
                    beta.len(),
                    self.design_derivative_exit.ncols()
                ),
            }
            .into());
        }
        let n_rows = self.derivative_offset_exit.len();
        if n_rows == 0 {
            return Ok(());
        }
        let qd_design = self.design_derivative_exit.matrixvectormultiply(beta);
        if qd_design.len() != n_rows {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival marginal-slope time-block {label} row count mismatch: design rows={} vs offset rows={n_rows}",
                    qd_design.len()
                ),
            }
            .into());
        }
        let guard = self.derivative_guard;
        // The monotonicity guard `qd1 = design·beta + offset >= guard` is enforced
        // through `time_linear_constraints`, which the inequality-constrained
        // active-set Newton solver satisfies only to its primal-feasibility
        // tolerance measured in the *scaled* constraint-row coordinate system.
        // `time_derivative_guard_constraints` normalizes each row by
        // `scale = max(||design_row||, |guard - offset|, 1)`, so a scaled slack
        // of `ACTIVE_SET_PRIMAL_FEASIBILITY_TOL` corresponds to a raw `qd1`
        // shortfall of up to `ACTIVE_SET_PRIMAL_FEASIBILITY_TOL * scale_row`.
        // Validating the raw `qd1` against a band of only `256·eps` therefore
        // demands ~9 orders of magnitude more precision than the solver that
        // produced `beta` can deliver, and spuriously rejects iterates that sit
        // exactly on the feasible boundary. The feasibility check here must use
        // the same scaling the constraint builder applied so that "the solver
        // calls this feasible" and "this validator calls this feasible" coincide.
        let derivative_dense = self.design_derivative_exit.to_dense_cow();
        let mut worst_scaled_violation = 0.0_f64;
        let mut worst_row = 0usize;
        let mut worst_qd1 = f64::INFINITY;
        let mut worst_scale = 1.0_f64;
        for row in 0..n_rows {
            let offset = self.derivative_offset_exit[row];
            let qd1 = qd_design[row] + offset;
            if !qd1.is_finite() || !offset.is_finite() {
                return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
                    reason: format!(
                        "survival marginal-slope time-block {label} produced non-finite baseline \
                         derivative at row {row}: qd1={qd1:.3e}, offset={offset:.3e}"
                    ),
                }
                .into());
            }
            // Per-row normalization identical to the constraint builder.
            let mut row_norm_sq = 0.0_f64;
            for col in 0..derivative_dense.ncols() {
                let v = derivative_dense[[row, col]];
                row_norm_sq += v * v;
            }
            let row_norm = row_norm_sq.sqrt();
            let rhs = guard - offset;
            let scale = row_norm.max(rhs.abs()).max(1.0);
            // Scaled violation = max(0, (guard - qd1) / scale); zero rows of the
            // design contribute no constraint (the bound is then carried by the
            // offset alone and checked at constraint-build time), so they cannot
            // be repaired by `beta` and are excluded from the scaled metric.
            let shortfall = guard - qd1;
            if shortfall > 0.0 && row_norm_sq > 1e-24 {
                let scaled = shortfall / scale;
                if scaled > worst_scaled_violation {
                    worst_scaled_violation = scaled;
                    worst_row = row;
                    worst_qd1 = qd1;
                    worst_scale = scale;
                }
            }
        }
        // A safety factor of 4 absorbs accumulation of the solver's per-row
        // tolerance and the small projection drift in the unconstrained
        // re-evaluation of `qd1` here versus the scaled constraint residual;
        // it stays far below any value that would admit a genuine monotonicity
        // violation (which is O(1e-3..1e0) when the fit truly diverges).
        let feasibility_band = 4.0 * crate::pirls::ACTIVE_SET_PRIMAL_FEASIBILITY_TOL;
        if worst_scaled_violation > feasibility_band {
            return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
                reason: format!(
                    "survival marginal-slope time-block {label} beta violates monotonicity at row {worst_row}: \
                     qd1={worst_qd1:.3e} < guard={guard:.3e} (scaled violation {worst_scaled_violation:.3e} \
                     exceeds solver feasibility band {feasibility_band:.3e}; row scale {worst_scale:.3e}); \
                     the derivative guard must be represented in time_linear_constraints, not repaired by \
                     post-update projection"
                ),
            }
            .into());
        }
        Ok(())
    }

    fn validate_exact_monotonicity(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<(), String> {
        if let Some(runtime) = &self.score_warp {
            let beta_h = self
                .flex_score_beta(block_states)?
                .ok_or_else(|| "missing survival score-warp coefficients".to_string())?;
            let expected = runtime.basis_dim() * self.score_dim();
            if beta_h.len() != expected {
                return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                    reason: format!(
                        "survival score-warp beta length mismatch: got {}, expected {expected} for K={} and basis dim {}",
                        beta_h.len(),
                        self.score_dim(),
                        runtime.basis_dim()
                    ),
                }
                .into());
            }
            for coord in 0..self.score_dim() {
                let local_beta = self.score_warp_beta_for_coord(beta_h, coord)?;
                runtime.monotonicity_feasible(
                    &local_beta,
                    &format!("survival marginal-slope score-warp[z{coord}]"),
                )?;
            }
        }
        if let Some(runtime) = &self.link_dev {
            let beta_w = self
                .flex_link_beta(block_states)?
                .ok_or_else(|| "missing survival link-deviation coefficients".to_string())?;
            runtime.monotonicity_feasible(beta_w, "survival marginal-slope link deviation")?;
        }
        Ok(())
    }

    fn observed_denested_eta_chi(
        &self,
        row: usize,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<(f64, f64), String> {
        let z_obs = self.observed_score_projection(row);
        let obs = self.observed_denested_cell_partials(row, a, b, beta_h, beta_w)?;
        let eta = eval_coeff4_at(&obs.coeff, z_obs);
        let chi = eval_coeff4_at(&obs.dc_da, z_obs);
        Ok((eta, chi))
    }

    fn observed_denested_cell_partials(
        &self,
        row: usize,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<ObservedDenestedCellPartials, String> {
        let z_obs = self.observed_score_projection(row);
        if self.score_dim() == 1 {
            return shared_observed_denested_cell_partials(
                z_obs,
                a,
                b,
                self.score_warp.as_ref(),
                beta_h,
                self.link_dev.as_ref(),
                beta_w,
                self.probit_frailty_scale(),
            );
        }

        // Observed vector-z contribution is the row-wise direct-sum value
        //     h_i = sum_k W_k(z_ik) beta_k.
        // In the denested additive transport, h_i enters as b*h_i at the
        // observed row.  Holding z_i fixed makes this a constant coefficient
        // in the observed polynomial while preserving the standard link
        // partials in a and b.
        let h_obs = self.score_warp_observed_value(row, beta_h)?;
        let u_obs = a + b * z_obs;
        let link_span = if let (Some(runtime), Some(beta_w)) = (self.link_dev.as_ref(), beta_w) {
            runtime.local_cubic_at(beta_w, u_obs)?
        } else {
            Self::zero_score_warp_span()
        };
        let (d0, d1, d2, d3) = exact_kernel::transformed_link_cubic(link_span, a, b);
        let coeff_raw = [a + b * h_obs + d0, b + d1, d2, d3];
        let shift = a - link_span.left;
        let alpha1 = link_span.c1;
        let alpha2 = link_span.c2;
        let alpha3 = link_span.c3;
        let dc_da_raw = [
            1.0 + alpha1 + 2.0 * alpha2 * shift + 3.0 * alpha3 * shift * shift,
            b * (2.0 * alpha2 + 6.0 * alpha3 * shift),
            3.0 * alpha3 * b * b,
            0.0,
        ];
        let dc_db_raw = [
            h_obs,
            1.0 + alpha1 + 2.0 * alpha2 * shift + 3.0 * alpha3 * shift * shift,
            2.0 * b * (alpha2 + 3.0 * alpha3 * shift),
            3.0 * alpha3 * b * b,
        ];
        let (dc_daa_raw, dc_dab_raw, dc_dbb_raw) =
            exact_kernel::link_basis_cell_second_partials(link_span, a, b);
        let (dc_daaa_raw, dc_daab_raw, dc_dabb_raw, dc_dbbb_raw) =
            exact_kernel::link_basis_cell_third_partials(link_span);
        let scale = self.probit_frailty_scale();
        Ok(ObservedDenestedCellPartials {
            coeff: scale_coeff4(coeff_raw, scale),
            dc_da: scale_coeff4(dc_da_raw, scale),
            dc_db: scale_coeff4(dc_db_raw, scale),
            dc_daa: scale_coeff4(dc_daa_raw, scale),
            dc_dab: scale_coeff4(dc_dab_raw, scale),
            dc_dbb: scale_coeff4(dc_dbb_raw, scale),
            dc_daaa: scale_coeff4(dc_daaa_raw, scale),
            dc_daab: scale_coeff4(dc_daab_raw, scale),
            dc_dabb: scale_coeff4(dc_dabb_raw, scale),
            dc_dbbb: scale_coeff4(dc_dbbb_raw, scale),
        })
    }

    fn denested_cell_primary_fixed_partials(
        &self,
        primary: &FlexPrimarySlices,
        a: f64,
        b: f64,
        score_span: exact_kernel::LocalSpanCubic,
        link_span: exact_kernel::LocalSpanCubic,
        z_basis: f64,
        u_basis: f64,
    ) -> Result<DenestedCellPrimaryFixedPartials, String> {
        let scale = self.probit_frailty_scale();
        let r = primary.total;
        let mut coeff_u = vec![[0.0; 4]; r];
        let mut coeff_au = vec![[0.0; 4]; r];
        let mut coeff_bu = vec![[0.0; 4]; r];
        let mut coeff_aau = vec![[0.0; 4]; r];
        let mut coeff_abu = vec![[0.0; 4]; r];
        let mut coeff_bbu = vec![[0.0; 4]; r];
        let mut coeff_aaau = vec![[0.0; 4]; r];
        let mut coeff_aabu = vec![[0.0; 4]; r];
        let mut coeff_abbu = vec![[0.0; 4]; r];
        let mut coeff_bbbu = vec![[0.0; 4]; r];

        let (dc_da_raw, dc_db_raw) =
            exact_kernel::denested_cell_coefficient_partials(score_span, link_span, a, b);
        let (dc_daa_raw, dc_dab_raw, dc_dbb_raw) =
            exact_kernel::denested_cell_second_partials(score_span, link_span, a, b);
        let (dc_daaa_raw, dc_daab_raw, dc_dabb_raw, dc_dbbb_raw) =
            exact_kernel::denested_cell_third_partials(link_span);
        let dc_da = scale_coeff4(dc_da_raw, scale);
        let dc_db = scale_coeff4(dc_db_raw, scale);
        let dc_daa = scale_coeff4(dc_daa_raw, scale);
        let dc_dab = scale_coeff4(dc_dab_raw, scale);
        let dc_dbb = scale_coeff4(dc_dbb_raw, scale);
        let dc_daaa = scale_coeff4(dc_daaa_raw, scale);
        let dc_daab = scale_coeff4(dc_daab_raw, scale);
        let dc_dabb = scale_coeff4(dc_dabb_raw, scale);
        let dc_dbbb = scale_coeff4(dc_dbbb_raw, scale);

        coeff_u[primary.g] = dc_db;
        coeff_au[primary.g] = dc_dab;
        coeff_bu[primary.g] = dc_dbb;
        coeff_aau[primary.g] = dc_daab;
        coeff_abu[primary.g] = dc_dabb;
        coeff_bbu[primary.g] = dc_dbbb;
        coeff_aaau[primary.g] = [0.0; 4];
        coeff_aabu[primary.g] = [0.0; 4];
        coeff_abbu[primary.g] = [0.0; 4];
        coeff_bbbu[primary.g] = [0.0; 4];

        if let Some(h_range) = primary.h.as_ref().filter(|_| self.score_warp.is_some()) {
            for local_idx in 0..h_range.len() {
                let idx = h_range.start + local_idx;
                coeff_u[idx] = scale_coeff4(
                    self.integration_score_basis_coefficients(local_idx, z_basis, b)?,
                    scale,
                );
                coeff_bu[idx] = scale_coeff4(
                    self.integration_score_basis_coefficients(local_idx, z_basis, 1.0)?,
                    scale,
                );
            }
        }

        if let (Some(w_range), Some(runtime)) = (primary.w.as_ref(), self.link_dev.as_ref()) {
            for local_idx in 0..w_range.len() {
                let basis_span = runtime.basis_cubic_at(local_idx, u_basis)?;
                let idx = w_range.start + local_idx;
                coeff_u[idx] = scale_coeff4(
                    exact_kernel::link_basis_cell_coefficients(basis_span, a, b),
                    scale,
                );
                let (dc_aw_raw, dc_bw_raw) =
                    exact_kernel::link_basis_cell_coefficient_partials(basis_span, a, b);
                let link_second = exact_kernel::link_basis_cell_second_partials(basis_span, a, b);
                let (dc_aaw_raw, dc_abw_raw, dc_bbw_raw) =
                    (link_second.0, link_second.1, link_second.2);
                let (dc_aaaw_raw, dc_aabw_raw, dc_abbw_raw, dc_bbbw_raw) =
                    exact_kernel::link_basis_cell_third_partials(basis_span);
                coeff_au[idx] = scale_coeff4(dc_aw_raw, scale);
                coeff_bu[idx] = scale_coeff4(dc_bw_raw, scale);
                coeff_aau[idx] = scale_coeff4(dc_aaw_raw, scale);
                coeff_abu[idx] = scale_coeff4(dc_abw_raw, scale);
                coeff_bbu[idx] = scale_coeff4(dc_bbw_raw, scale);
                coeff_aaau[idx] = scale_coeff4(dc_aaaw_raw, scale);
                coeff_aabu[idx] = scale_coeff4(dc_aabw_raw, scale);
                coeff_abbu[idx] = scale_coeff4(dc_abbw_raw, scale);
                coeff_bbbu[idx] = scale_coeff4(dc_bbbw_raw, scale);
            }
        }

        Ok(DenestedCellPrimaryFixedPartials {
            dc_da,
            dc_daa,
            dc_daaa,
            coeff_u,
            coeff_au,
            coeff_bu,
            coeff_aau,
            coeff_abu,
            coeff_bbu,
            coeff_aaau,
            coeff_aabu,
            coeff_abbu,
            coeff_bbbu,
        })
    }

    fn observed_fixed_eta_second_partial(
        &self,
        primary: &FlexPrimarySlices,
        obs: &ObservedDenestedCellPartials,
        row: usize,
        u: usize,
        v: usize,
        z_obs: f64,
        u_obs: f64,
        a: f64,
        b: f64,
    ) -> Result<f64, String> {
        let scale = self.probit_frailty_scale();
        if u == primary.g && v == primary.g {
            return Ok(eval_coeff4_at(&obs.dc_dbb, z_obs));
        }
        if u == primary.g {
            if let Some(h_range) = primary.h.as_ref()
                && v >= h_range.start
                && v < h_range.end
            {
                let local_idx = v - h_range.start;
                return Ok(eval_coeff4_at(
                    &scale_coeff4(
                        self.observed_score_basis_coefficients(row, local_idx, z_obs, 1.0)?,
                        scale,
                    ),
                    z_obs,
                ));
            }
            if let Some(w_range) = primary.w.as_ref()
                && v >= w_range.start
                && v < w_range.end
            {
                let local_idx = v - w_range.start;
                let runtime = self
                    .link_dev
                    .as_ref()
                    .ok_or_else(|| "missing survival link runtime".to_string())?;
                let basis_span = runtime.basis_cubic_at(local_idx, u_obs)?;
                let (_, dc_bw) =
                    exact_kernel::link_basis_cell_coefficient_partials(basis_span, a, b);
                return Ok(eval_coeff4_at(&scale_coeff4(dc_bw, scale), z_obs));
            }
        }
        if v == primary.g {
            return self
                .observed_fixed_eta_second_partial(primary, obs, row, v, u, z_obs, u_obs, a, b);
        }
        Ok(0.0)
    }

    fn observed_fixed_chi_second_partial(
        &self,
        primary: &FlexPrimarySlices,
        obs: &ObservedDenestedCellPartials,
        u: usize,
        v: usize,
        z_obs: f64,
        u_obs: f64,
        a: f64,
        b: f64,
    ) -> Result<f64, String> {
        let scale = self.probit_frailty_scale();
        if u == primary.g && v == primary.g {
            return Ok(eval_coeff4_at(&obs.dc_dabb, z_obs));
        }
        if u == primary.g
            && let Some(w_range) = primary.w.as_ref()
            && v >= w_range.start
            && v < w_range.end
        {
            let local_idx = v - w_range.start;
            let runtime = self
                .link_dev
                .as_ref()
                .ok_or_else(|| "missing survival link runtime".to_string())?;
            let basis_span = runtime.basis_cubic_at(local_idx, u_obs)?;
            let (_, dc_abw, _) = exact_kernel::link_basis_cell_second_partials(basis_span, a, b);
            return Ok(eval_coeff4_at(&scale_coeff4(dc_abw, scale), z_obs));
        }
        if v == primary.g {
            return self.observed_fixed_chi_second_partial(primary, obs, v, u, z_obs, u_obs, a, b);
        }
        Ok(0.0)
    }

    fn evaluate_survival_denom_d(
        &self,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<f64, String> {
        // Density normalization is |F'(a)| for the same calibration equation
        // solved by `solve_row_survival_intercept`. Reusing that exact
        // derivative convention avoids sign drift between the solver path and
        // the direct-check path.
        let (_, f_a, _) = self.evaluate_denested_survival_calibration(a, 0.0, b, beta_h, beta_w)?;
        let d = f_a.abs();
        if !d.is_finite() || d <= 0.0 {
            return Err(SurvivalMarginalSlopeError::NumericalFailure {
                reason: format!(
                    "survival marginal-slope produced non-positive calibration derivative |F'(a)|={d:.3e}"
                ),
            }
            .into());
        }
        Ok(d)
    }

    fn row_neglog_flex_value(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
    ) -> Result<f64, String> {
        let q_geom = self.row_dynamic_q_values(row, block_states)?;
        let g = block_states[2].eta[row];
        let beta_h = self.flex_score_beta(block_states)?;
        let beta_w = self.flex_link_beta(block_states)?;
        // Absorbed Stage-1 influence offset (#461): a per-row additive shift of
        // the de-nested observed index η₁ (un-`c(g)`-scaled), `0.0` when no
        // absorber is installed.
        let o_infl = self.influence_index_offset(row, block_states)?;
        self.row_neglog_flex_value_from_parts(
            row, q_geom.q0, q_geom.q1, q_geom.qd1, g, beta_h, beta_w, o_infl,
        )
    }

    fn row_neglog_flex_value_from_parts(
        &self,
        row: usize,
        q0: f64,
        q1: f64,
        qd1: f64,
        g: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        o_infl: f64,
    ) -> Result<f64, String> {
        if survival_derivative_guard_violated(qd1, self.derivative_guard) {
            return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
                reason: format!(
                    "survival marginal-slope monotonicity violated at row {row}: qd1={qd1:.3e} < guard={:.3e}",
                    self.derivative_guard
                ),
            }
            .into());
        }
        let (a0, _) = self.solve_row_survival_intercept_with_slot(
            q0,
            g,
            beta_h,
            beta_w,
            Some((row, SurvivalInterceptSlotKind::Entry)),
        )?;
        let (a1, d1) = self.solve_row_survival_intercept_with_slot(
            q1,
            g,
            beta_h,
            beta_w,
            Some((row, SurvivalInterceptSlotKind::Exit)),
        )?;
        if !d1.is_finite() || d1 <= 0.0 {
            return Err(SurvivalMarginalSlopeError::NumericalFailure {
                reason: format!(
                    "survival marginal-slope row {row} produced non-positive density normalization D1={d1:.3e} (calibration derivative {:.3e})",
                    d1
                ),
            }
            .into());
        }
        // The absorbed-influence offset shifts the observed index η₁ additively
        // at both the entry (eta0) and exit (eta1) calibration roots — `o_infl`
        // is independent of the calibration intercept `a`, so the de-nesting
        // derivative `chi1 = ∂η₁/∂a` is unchanged.
        let (eta0_raw, _) = self.observed_denested_eta_chi(row, a0, g, beta_h, beta_w)?;
        let (eta1_raw, chi1) = self.observed_denested_eta_chi(row, a1, g, beta_h, beta_w)?;
        let eta0 = eta0_raw + o_infl;
        let eta1 = eta1_raw + o_infl;
        if !chi1.is_finite() || chi1 <= 0.0 {
            return Err(SurvivalMarginalSlopeError::NumericalFailure {
                reason: format!(
                    "survival marginal-slope row {row} produced non-positive observed chi1={chi1:.3e}"
                ),
            }
            .into());
        }
        let wi = self.weights[row];
        let di = self.event[row];
        let (log_surv0, _) = signed_probit_logcdf_and_mills_ratio(-eta0);
        let (log_surv1, _) = signed_probit_logcdf_and_mills_ratio(-eta1);
        let log_phi_eta1 = -0.5 * (eta1 * eta1 + std::f64::consts::TAU.ln());
        let log_phi_q1 = -0.5 * (q1 * q1 + std::f64::consts::TAU.ln());
        Ok(wi
            * (log_surv0
                - (1.0 - di) * log_surv1
                - di * log_phi_eta1
                - di * chi1.ln()
                - di * log_phi_q1
                + di * d1.ln()
                - di * qd1.ln()))
    }

    /// Build a cached partition: cells + moment states + fixed partials,
    /// computed once per (a, b, β_h, β_w) and reused across the three
    /// integration passes (F, D, D_uv).
    ///
    /// The cell-table assembly and the per-cell primary-fixed-partials
    /// assembly route through the GPU-shaped `try_device_*` seams in
    /// [`crate::families::survival_marginal_slope_gpu_prep`].  Until the matching NVRTC kernels
    /// land, both seams return `Ok(None)` and the call site falls back to
    /// the existing CPU implementation, so behavior is preserved.
    fn build_cached_partition_with_moment_order(
        &self,
        primary: &FlexPrimarySlices,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        moment_order: usize,
    ) -> Result<CachedPartitionCells, String> {
        // ── 1. partition cells via the device seam, CPU fallback on decline ──
        let raw_cells = {
            let row_input =
                crate::families::survival_marginal_slope_gpu_prep::PartitionCellsRowInputs {
                    a,
                    b,
                    beta_h: beta_h.and_then(|b| b.as_slice()),
                    beta_w: beta_w.and_then(|b| b.as_slice()),
                };
            let dev =
                crate::families::survival_marginal_slope_gpu_prep::try_device_partition_cells(
                    std::slice::from_ref(&row_input),
                )
                .map_err(|e| e.to_string())?;
            match dev {
                Some(mut by_row) if by_row.len() == 1 => by_row.remove(0),
                _ => self.denested_partition_cells(a, b, beta_h, beta_w)?,
            }
        };

        // ── 2. per-cell prelude (neg_cell, z_mid, u_mid, moment state) ──
        let n = raw_cells.len();
        let mut neg_cells = Vec::with_capacity(n);
        let mut z_mids = Vec::with_capacity(n);
        let mut u_mids = Vec::with_capacity(n);
        let mut states = Vec::with_capacity(n);
        let mut fp_inputs = Vec::<
            crate::families::survival_marginal_slope_gpu_prep::CellPrimaryFixedPartialsCellInputs,
        >::with_capacity(n);
        for partition_cell in &raw_cells {
            let cell = partition_cell.cell;
            let neg_cell = exact_kernel::DenestedCubicCell {
                left: cell.left,
                right: cell.right,
                c0: -cell.c0,
                c1: -cell.c1,
                c2: -cell.c2,
                c3: -cell.c3,
            };
            let z_mid = exact_kernel::interval_probe_point(cell.left, cell.right)?;
            let u_mid = a + b * z_mid;
            let state = exact_kernel::evaluate_cell_moments(cell, moment_order)?;
            neg_cells.push(neg_cell);
            z_mids.push(z_mid);
            u_mids.push(u_mid);
            states.push(state);
            fp_inputs.push(
                crate::families::survival_marginal_slope_gpu_prep::CellPrimaryFixedPartialsCellInputs {
                    score_span: partition_cell.score_span,
                    link_span: partition_cell.link_span,
                    z_basis: z_mid,
                    u_basis: u_mid,
                },
            );
        }

        // ── 3. per-cell fixed partials via the device seam, CPU fallback ──
        let layout = crate::families::survival_marginal_slope_gpu_prep::FlexPrimaryLayout {
            r: u32::try_from(primary.total).map_err(|_| {
                format!(
                    "build_cached_partition_with_moment_order: primary.total={} exceeds u32",
                    primary.total
                )
            })?,
            g_slot: u32::try_from(primary.g).map_err(|_| {
                format!(
                    "build_cached_partition_with_moment_order: primary.g={} exceeds u32",
                    primary.g
                )
            })?,
        };
        let row_fp_input =
            crate::families::survival_marginal_slope_gpu_prep::CellPrimaryFixedPartialsRowInputs {
                a,
                b,
                cells: &fp_inputs,
                layout,
            };
        let dev_fixed = crate::families::survival_marginal_slope_gpu_prep::try_device_cell_primary_fixed_partials(
            std::slice::from_ref(&row_fp_input),
        )
        .map_err(|e| e.to_string())?;
        // When the device path returns flat-packed partials, reconstruct
        // the per-cell `DenestedCellPrimaryFixedPartials` from the device
        // buffer via the `from_flat_slice` shim — byte-identical to what
        // the CPU per-cell helper would produce for the supported
        // (trivial-span) shape.  Any decline drops through to the CPU
        // per-cell loop below.
        if let Some(out) = dev_fixed.as_ref()
            && out.partials.len() == 1
            && out.partials[0].len() == n
        {
            let mut cells = Vec::with_capacity(n);
            for (idx, partition_cell) in raw_cells.into_iter().enumerate() {
                let flat = &out.partials[0][idx];
                let fixed = DenestedCellPrimaryFixedPartials::from_flat_slice(
                    flat.as_slice(),
                    primary.total,
                )?;
                cells.push(CachedCellEntry {
                    partition_cell,
                    neg_cell: neg_cells[idx],
                    state: states[idx].clone(),
                    fixed,
                });
            }
            return Ok(CachedPartitionCells { cells });
        }
        let mut cells = Vec::with_capacity(n);
        for (idx, partition_cell) in raw_cells.into_iter().enumerate() {
            let fixed = self.denested_cell_primary_fixed_partials(
                primary,
                a,
                b,
                partition_cell.score_span,
                partition_cell.link_span,
                z_mids[idx],
                u_mids[idx],
            )?;
            cells.push(CachedCellEntry {
                partition_cell,
                neg_cell: neg_cells[idx],
                state: states[idx].clone(),
                fixed,
            });
        }
        Ok(CachedPartitionCells { cells })
    }

    fn build_cached_partition(
        &self,
        primary: &FlexPrimarySlices,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<CachedPartitionCells, String> {
        self.build_cached_partition_with_moment_order(primary, a, b, beta_h, beta_w, 24)
    }

    fn compute_survival_timepoint_first_order_exact(
        &self,
        row: usize,
        primary: &FlexPrimarySlices,
        q: f64,
        q_index: usize,
        a: f64,
        b: f64,
        d_calibration: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        o_infl: f64,
    ) -> Result<SurvivalFlexTimepointFirstOrderExact, String> {
        let p = primary.total;
        let cached =
            self.build_cached_partition_with_moment_order(primary, a, b, beta_h, beta_w, 9)?;

        struct FirstOrderCellAccum {
            f_u: Vec<f64>,
        }

        let cell_accums = cached
            .cells
            .iter()
            .map(|entry| -> Result<FirstOrderCellAccum, String> {
                let state = &entry.state;
                let fixed = &entry.fixed;
                let mut f_u = vec![0.0; p];
                for u in 0..p {
                    let neg_coeff_u = fixed.coeff_u[u].map(|value| -value);
                    f_u[u] = exact_kernel::cell_first_derivative_from_moments(
                        &neg_coeff_u,
                        &state.moments,
                    )?;
                }
                Ok(FirstOrderCellAccum { f_u })
            })
            .collect::<Result<Vec<_>, String>>()?;

        let mut f_u = Array1::<f64>::zeros(p);
        for acc in cell_accums {
            for u in 0..p {
                f_u[u] += acc.f_u[u];
            }
        }
        f_u[q_index] += crate::probability::normal_pdf(q);

        let d_check = self.evaluate_survival_denom_d(a, b, beta_h, beta_w)?;
        if !d_check.is_finite() || d_check <= 0.0 {
            return Err(SurvivalMarginalSlopeError::NumericalFailure {
                reason: format!(
                    "survival marginal-slope row {row} produced non-positive density normalization D={d_check:.3e}"
                ),
            }
            .into());
        }
        let d_rel_err = (d_check - d_calibration).abs() / d_check.max(d_calibration.abs()).max(1.0);
        if !d_calibration.is_finite() || d_calibration <= 0.0 || d_rel_err > 1e-8 {
            return Err(SurvivalMarginalSlopeError::NumericalFailure {
                reason: format!(
                    "survival marginal-slope row {row} produced inconsistent calibration derivative: solve={d_calibration:.12e}, direct={d_check:.12e}"
                ),
            }
            .into());
        }

        let mut a_u = Array1::<f64>::zeros(p);
        for u in 0..p {
            a_u[u] = f_u[u] / d_check;
        }

        let d_u_cell_accums = cached
            .cells
            .iter()
            .map(|entry| -> Result<Vec<f64>, String> {
                let cell = entry.partition_cell.cell;
                let state = &entry.state;
                let fixed = &entry.fixed;
                let eta_poly = vec![cell.c0, cell.c1, cell.c2, cell.c3];
                let chi_poly = fixed.dc_da.to_vec();
                let eta_aa_poly = fixed.dc_daa.to_vec();
                let mut d_u = vec![0.0; p];
                for u in 0..p {
                    let eta_u_poly = poly_add(&poly_scale(&chi_poly, a_u[u]), &fixed.coeff_u[u]);
                    let chi_u_poly =
                        poly_add(&poly_scale(&eta_aa_poly, a_u[u]), &fixed.coeff_au[u]);
                    let integrand = poly_sub(
                        &chi_u_poly,
                        &poly_mul(&poly_mul(&chi_poly, &eta_poly), &eta_u_poly),
                    );
                    d_u[u] = exact_kernel::cell_polynomial_integral_from_moments(
                        &integrand,
                        &state.moments,
                        "survival D_t first derivative",
                    )?;
                }
                Ok(d_u)
            })
            .collect::<Result<Vec<_>, String>>()?;
        let mut d_u = Array1::<f64>::zeros(p);
        for cell_d_u in d_u_cell_accums {
            for u in 0..p {
                d_u[u] += cell_d_u[u];
            }
        }

        let z_obs = self.observed_score_projection(row);
        let u_obs = a + b * z_obs;
        let obs = self.observed_denested_cell_partials(row, a, b, beta_h, beta_w)?;
        // Absorbed-influence offset: observed-η shift only (see
        // `compute_survival_timepoint_exact`). `eta += o_infl`, and the trailing
        // `infl` channel carries the direct partial `∂η₁/∂o_infl = 1` via
        // `rho[infl]`; calibration-side `a_u`/`chi`/`chi_u`/`d_u` are untouched.
        let eta = eval_coeff4_at(&obs.coeff, z_obs) + o_infl;
        let chi = eval_coeff4_at(&obs.dc_da, z_obs);
        let eta_aa = eval_coeff4_at(&obs.dc_daa, z_obs);

        let mut rho = Array1::<f64>::zeros(p);
        let mut tau = Array1::<f64>::zeros(p);
        let scale = self.probit_frailty_scale();
        rho[primary.g] = eval_coeff4_at(&obs.dc_db, z_obs);
        tau[primary.g] = eval_coeff4_at(&obs.dc_dab, z_obs);
        if let Some(infl) = primary.infl {
            rho[infl] = 1.0;
        }

        if let Some(h_range) = primary.h.as_ref().filter(|_| self.score_warp.is_some()) {
            for local_idx in 0..h_range.len() {
                let idx = h_range.start + local_idx;
                rho[idx] = eval_coeff4_at(
                    &scale_coeff4(
                        self.observed_score_basis_coefficients(row, local_idx, z_obs, b)?,
                        scale,
                    ),
                    z_obs,
                );
            }
        }
        if let (Some(w_range), Some(runtime)) = (primary.w.as_ref(), self.link_dev.as_ref()) {
            for local_idx in 0..w_range.len() {
                let basis_span = runtime.basis_cubic_at(local_idx, u_obs)?;
                let idx = w_range.start + local_idx;
                rho[idx] = eval_coeff4_at(
                    &scale_coeff4(
                        exact_kernel::link_basis_cell_coefficients(basis_span, a, b),
                        scale,
                    ),
                    z_obs,
                );
                let (dc_aw, _) =
                    exact_kernel::link_basis_cell_coefficient_partials(basis_span, a, b);
                tau[idx] = eval_coeff4_at(&scale_coeff4(dc_aw, scale), z_obs);
            }
        }

        let mut eta_u = Array1::<f64>::zeros(p);
        let mut chi_u = Array1::<f64>::zeros(p);
        for u in 0..p {
            eta_u[u] = chi * a_u[u] + rho[u];
            chi_u[u] = eta_aa * a_u[u] + tau[u];
        }

        Ok(SurvivalFlexTimepointFirstOrderExact {
            eta,
            chi,
            d: d_check,
            eta_u,
            chi_u,
            d_u,
        })
    }

    fn compute_survival_timepoint_exact(
        &self,
        row: usize,
        primary: &FlexPrimarySlices,
        q: f64,
        q_index: usize,
        a: f64,
        b: f64,
        d_calibration: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        o_infl: f64,
        need_d_uv: bool,
    ) -> Result<SurvivalFlexTimepointExact, String> {
        let p = primary.total;
        let cached = self.build_cached_partition(primary, a, b, beta_h, beta_w)?;

        struct ExactTimepointCellAccum {
            f_aa: f64,
            f_u: Vec<f64>,
            f_au: Vec<f64>,
            f_uv: Vec<f64>,
        }

        let cell_accums = cached
            .cells
            .iter()
            .map(|entry| -> Result<ExactTimepointCellAccum, String> {
                let neg_cell = entry.neg_cell;
                let state = &entry.state;
                let fixed = &entry.fixed;
                let neg_dc_da = fixed.dc_da.map(|value| -value);
                let neg_dc_daa = fixed.dc_daa.map(|value| -value);
                let mut f_u = vec![0.0; p];
                let mut f_au = vec![0.0; p];
                let mut f_uv = vec![0.0; p * p];
                let f_aa = exact_kernel::cell_second_derivative_from_moments(
                    neg_cell,
                    &neg_dc_da,
                    &neg_dc_da,
                    &neg_dc_daa,
                    &state.moments,
                )?;
                for u in 0..p {
                    let neg_coeff_u = fixed.coeff_u[u].map(|value| -value);
                    let neg_coeff_au = fixed.coeff_au[u].map(|value| -value);
                    f_u[u] = exact_kernel::cell_first_derivative_from_moments(
                        &neg_coeff_u,
                        &state.moments,
                    )?;
                    f_au[u] = exact_kernel::cell_second_derivative_from_moments(
                        neg_cell,
                        &neg_dc_da,
                        &neg_coeff_u,
                        &neg_coeff_au,
                        &state.moments,
                    )?;
                }
                for u in 0..p {
                    let neg_coeff_u = fixed.coeff_u[u].map(|value| -value);
                    for v in u..p {
                        let second_coeff = if u == primary.g {
                            fixed.coeff_bu[v]
                        } else if v == primary.g {
                            fixed.coeff_bu[u]
                        } else {
                            [0.0; 4]
                        };
                        let neg_coeff_v = fixed.coeff_u[v].map(|value| -value);
                        let neg_second_coeff = second_coeff.map(|value| -value);
                        let value = exact_kernel::cell_second_derivative_from_moments(
                            neg_cell,
                            &neg_coeff_u,
                            &neg_coeff_v,
                            &neg_second_coeff,
                            &state.moments,
                        )?;
                        f_uv[u * p + v] = value;
                        f_uv[v * p + u] = value;
                    }
                }
                Ok(ExactTimepointCellAccum {
                    f_aa,
                    f_u,
                    f_au,
                    f_uv,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;

        let mut f_u = Array1::<f64>::zeros(p);
        let mut f_au = Array1::<f64>::zeros(p);
        let mut f_uv = Array2::<f64>::zeros((p, p));
        let mut f_aa = 0.0;
        for acc in cell_accums {
            f_aa += acc.f_aa;
            for u in 0..p {
                f_u[u] += acc.f_u[u];
                f_au[u] += acc.f_au[u];
                for v in 0..p {
                    f_uv[[u, v]] += acc.f_uv[u * p + v];
                }
            }
        }

        let phi_q = crate::probability::normal_pdf(q);
        f_u[q_index] += phi_q;
        f_uv[[q_index, q_index]] += -q * phi_q;

        let d_check = self.evaluate_survival_denom_d(a, b, beta_h, beta_w)?;
        if !d_check.is_finite() || d_check <= 0.0 {
            return Err(SurvivalMarginalSlopeError::NumericalFailure {
                reason: format!(
                    "survival marginal-slope row {row} produced non-positive density normalization D={d_check:.3e}"
                ),
            }
            .into());
        }
        let d_rel_err = (d_check - d_calibration).abs() / d_check.max(d_calibration.abs()).max(1.0);
        if !d_calibration.is_finite() || d_calibration <= 0.0 || d_rel_err > 1e-8 {
            return Err(SurvivalMarginalSlopeError::NumericalFailure {
                reason: format!(
                    "survival marginal-slope row {row} produced inconsistent calibration derivative: solve={d_calibration:.12e}, direct={d_check:.12e}"
                ),
            }
            .into());
        }

        let mut a_u = Array1::<f64>::zeros(p);
        for u in 0..p {
            a_u[u] = f_u[u] / d_check;
        }

        let d_u_cell_accums = cached
            .cells
            .iter()
            .map(|entry| -> Result<Vec<f64>, String> {
                let cell = entry.partition_cell.cell;
                let state = &entry.state;
                let fixed = &entry.fixed;
                let eta_poly = vec![cell.c0, cell.c1, cell.c2, cell.c3];
                let chi_poly = fixed.dc_da.to_vec();
                let mut d_u = vec![0.0; p];
                for u in 0..p {
                    let eta_u_poly = poly_add(&poly_scale(&chi_poly, a_u[u]), &fixed.coeff_u[u]);
                    let chi_u_poly =
                        poly_add(&poly_scale(&fixed.dc_daa, a_u[u]), &fixed.coeff_au[u]);
                    let integrand = poly_sub(
                        &chi_u_poly,
                        &poly_mul(&poly_mul(&chi_poly, &eta_poly), &eta_u_poly),
                    );
                    d_u[u] = exact_kernel::cell_polynomial_integral_from_moments(
                        &integrand,
                        &state.moments,
                        "survival D_t first derivative",
                    )?;
                }
                Ok(d_u)
            })
            .collect::<Result<Vec<_>, String>>()?;
        let mut d_u = Array1::<f64>::zeros(p);
        for cell_d_u in d_u_cell_accums {
            for u in 0..p {
                d_u[u] += cell_d_u[u];
            }
        }

        let mut a_uv = Array2::<f64>::zeros((p, p));
        for u in 0..p {
            for v in u..p {
                let value =
                    (f_uv[[u, v]] - d_u[u] * a_u[v] - d_u[v] * a_u[u] - f_aa * a_u[u] * a_u[v])
                        / d_check;
                a_uv[[u, v]] = value;
                a_uv[[v, u]] = value;
            }
        }

        let z_obs = self.observed_score_projection(row);
        let u_obs = a + b * z_obs;
        let obs = self.observed_denested_cell_partials(row, a, b, beta_h, beta_w)?;
        // The absorbed-influence offset shifts the OBSERVED index η₁ additively
        // (#461). It is independent of the calibration intercept `a`, so it
        // touches only `eta` itself and the trailing `infl` primary partial
        // `∂η₁/∂o_infl = 1` (set via `rho[infl]` below); every calibration-side
        // quantity (`a_u`, `chi`, `chi_u`, `d_u`, all second partials) is
        // untouched because `o_infl` never enters the de-nested cells.
        let eta = eval_coeff4_at(&obs.coeff, z_obs) + o_infl;
        let chi = eval_coeff4_at(&obs.dc_da, z_obs);
        let eta_aa = eval_coeff4_at(&obs.dc_daa, z_obs);
        let eta_aaa = eval_coeff4_at(&obs.dc_daaa, z_obs);

        let mut rho = Array1::<f64>::zeros(p);
        let mut tau = Array1::<f64>::zeros(p);
        let mut tau_a = Array1::<f64>::zeros(p);
        let scale = self.probit_frailty_scale();
        rho[primary.g] = eval_coeff4_at(&obs.dc_db, z_obs);
        // Direct observed partial of the absorber channel: `∂η₁/∂o_infl = 1`
        // (and `a_u[infl] = 0`), so `eta_u[infl] = chi·0 + rho[infl] = 1`. All
        // other `infl` entries (tau, tau_a, second partials) stay zero.
        if let Some(infl) = primary.infl {
            rho[infl] = 1.0;
        }
        tau[primary.g] = eval_coeff4_at(&obs.dc_dab, z_obs);
        tau_a[primary.g] = eval_coeff4_at(&obs.dc_daab, z_obs);

        if let Some(h_range) = primary.h.as_ref().filter(|_| self.score_warp.is_some()) {
            for local_idx in 0..h_range.len() {
                let idx = h_range.start + local_idx;
                rho[idx] = eval_coeff4_at(
                    &scale_coeff4(
                        self.observed_score_basis_coefficients(row, local_idx, z_obs, b)?,
                        scale,
                    ),
                    z_obs,
                );
            }
        }
        if let (Some(w_range), Some(runtime)) = (primary.w.as_ref(), self.link_dev.as_ref()) {
            for local_idx in 0..w_range.len() {
                let basis_span = runtime.basis_cubic_at(local_idx, u_obs)?;
                let idx = w_range.start + local_idx;
                rho[idx] = eval_coeff4_at(
                    &scale_coeff4(
                        exact_kernel::link_basis_cell_coefficients(basis_span, a, b),
                        scale,
                    ),
                    z_obs,
                );
                let (dc_aw, _) =
                    exact_kernel::link_basis_cell_coefficient_partials(basis_span, a, b);
                let (dc_aaw, _, _) =
                    exact_kernel::link_basis_cell_second_partials(basis_span, a, b);
                tau[idx] = eval_coeff4_at(&scale_coeff4(dc_aw, scale), z_obs);
                tau_a[idx] = eval_coeff4_at(&scale_coeff4(dc_aaw, scale), z_obs);
            }
        }

        let mut eta_u = Array1::<f64>::zeros(p);
        let mut chi_u = Array1::<f64>::zeros(p);
        for u in 0..p {
            eta_u[u] = chi * a_u[u] + rho[u];
            chi_u[u] = eta_aa * a_u[u] + tau[u];
        }

        let mut eta_uv = Array2::<f64>::zeros((p, p));
        let mut chi_uv = Array2::<f64>::zeros((p, p));
        for u in 0..p {
            for v in u..p {
                let r_uv = self.observed_fixed_eta_second_partial(
                    primary, &obs, row, u, v, z_obs, u_obs, a, b,
                )?;
                let chi_uv_fixed = self
                    .observed_fixed_chi_second_partial(primary, &obs, u, v, z_obs, u_obs, a, b)?;

                let eta_val = chi * a_uv[[u, v]]
                    + eta_aa * a_u[u] * a_u[v]
                    + tau[u] * a_u[v]
                    + tau[v] * a_u[u]
                    + r_uv;
                eta_uv[[u, v]] = eta_val;
                eta_uv[[v, u]] = eta_val;

                let chi_val = eta_aa * a_uv[[u, v]]
                    + eta_aaa * a_u[u] * a_u[v]
                    + tau_a[u] * a_u[v]
                    + tau_a[v] * a_u[u]
                    + chi_uv_fixed;
                chi_uv[[u, v]] = chi_val;
                chi_uv[[v, u]] = chi_val;
            }
        }

        let mut d_uv = Array2::<f64>::zeros((p, p));
        if need_d_uv {
            let d_uv_cell_accums = cached
                .cells
                .iter()
                .map(|entry| -> Result<Vec<f64>, String> {
                    let cell = entry.partition_cell.cell;
                    let state = &entry.state;
                    let fixed = &entry.fixed;
                    let eta_poly = vec![cell.c0, cell.c1, cell.c2, cell.c3];
                    let chi_poly = fixed.dc_da.to_vec();
                    let eta_aa_poly = fixed.dc_daa.to_vec();
                    let eta_aaa_poly = fixed.dc_daaa.to_vec();
                    let mut eta_u_poly = vec![PolyVec::new(); p];
                    let mut chi_u_poly = vec![PolyVec::new(); p];
                    let mut d_uv = vec![0.0; p * p];
                    for u in 0..p {
                        eta_u_poly[u] = poly_add(&poly_scale(&chi_poly, a_u[u]), &fixed.coeff_u[u]);
                        chi_u_poly[u] =
                            poly_add(&poly_scale(&eta_aa_poly, a_u[u]), &fixed.coeff_au[u]);
                    }
                    for u in 0..p {
                        for v in u..p {
                            let r_uv_fixed = if u == primary.g {
                                fixed.coeff_bu[v].to_vec()
                            } else if v == primary.g {
                                fixed.coeff_bu[u].to_vec()
                            } else {
                                vec![0.0; 4]
                            };
                            let chi_uv_fixed = if u == primary.g {
                                fixed.coeff_abu[v].to_vec()
                            } else if v == primary.g {
                                fixed.coeff_abu[u].to_vec()
                            } else {
                                vec![0.0; 4]
                            };
                            let eta_uv_poly = poly_add(
                                &poly_add(
                                    &poly_add(
                                        &poly_scale(&chi_poly, a_uv[[u, v]]),
                                        &poly_scale(&eta_aa_poly, a_u[u] * a_u[v]),
                                    ),
                                    &poly_scale(&fixed.coeff_au[u], a_u[v]),
                                ),
                                &poly_add(&poly_scale(&fixed.coeff_au[v], a_u[u]), &r_uv_fixed),
                            );
                            let chi_uv_poly = poly_add(
                                &poly_add(
                                    &poly_add(
                                        &poly_scale(&eta_aa_poly, a_uv[[u, v]]),
                                        &poly_scale(&eta_aaa_poly, a_u[u] * a_u[v]),
                                    ),
                                    &poly_scale(&fixed.coeff_aau[u], a_u[v]),
                                ),
                                &poly_add(&poly_scale(&fixed.coeff_aau[v], a_u[u]), &chi_uv_fixed),
                            );
                            let term2 = poly_scale(
                                &poly_mul(&poly_mul(&chi_u_poly[v], &eta_poly), &eta_u_poly[u]),
                                -1.0,
                            );
                            let term3 = poly_scale(
                                &poly_mul(&poly_mul(&chi_u_poly[u], &eta_poly), &eta_u_poly[v]),
                                -1.0,
                            );
                            let term4 = poly_scale(
                                &poly_mul(
                                    &chi_poly,
                                    &poly_add(
                                        &poly_mul(&eta_u_poly[u], &eta_u_poly[v]),
                                        &poly_mul(&eta_poly, &eta_uv_poly),
                                    ),
                                ),
                                -1.0,
                            );
                            let term5 = poly_mul(
                                &chi_poly,
                                &poly_mul(
                                    &poly_mul(&eta_poly, &eta_poly),
                                    &poly_mul(&eta_u_poly[u], &eta_u_poly[v]),
                                ),
                            );
                            let integrand = poly_add(
                                &poly_add(&poly_add(&chi_uv_poly, &term2), &term3),
                                &poly_add(&term4, &term5),
                            );
                            let value = exact_kernel::cell_polynomial_integral_from_moments(
                                &integrand,
                                &state.moments,
                                "survival D_t second derivative",
                            )?;
                            d_uv[u * p + v] = value;
                            d_uv[v * p + u] = value;
                        }
                    }
                    Ok(d_uv)
                })
                .collect::<Result<Vec<_>, String>>()?;
            for cell_d_uv in d_uv_cell_accums {
                for u in 0..p {
                    for v in 0..p {
                        d_uv[[u, v]] += cell_d_uv[u * p + v];
                    }
                }
            }
        }

        Ok(SurvivalFlexTimepointExact {
            eta,
            chi,
            d: d_check,
            eta_u,
            eta_uv,
            chi_u,
            chi_uv,
            d_u,
            d_uv,
        })
    }

    fn compute_row_flex_primary_gradient_hessian_exact(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        q_geom: &SurvivalMarginalSlopeDynamicRow,
        primary: &FlexPrimarySlices,
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        self.ensure_scalar_flex_exact_score_geometry(
            "compute_row_flex_primary_gradient_hessian_exact",
        )?;
        let g = block_states[2].eta[row];
        let beta_h = self.flex_score_beta(block_states)?;
        let beta_w = self.flex_link_beta(block_states)?;
        let o_infl = self.influence_index_offset(row, block_states)?;
        self.compute_row_flex_primary_gradient_hessian_from_parts(
            row, q_geom.q0, q_geom.q1, q_geom.qd1, g, beta_h, beta_w, o_infl, primary,
        )
    }

    fn compute_row_flex_primary_gradient_exact(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        q_geom: &SurvivalMarginalSlopeDynamicRowGradient,
        primary: &FlexPrimarySlices,
    ) -> Result<(f64, Array1<f64>), String> {
        self.ensure_scalar_flex_exact_score_geometry("compute_row_flex_primary_gradient_exact")?;
        let g = block_states[2].eta[row];
        let beta_h = self.flex_score_beta(block_states)?;
        let beta_w = self.flex_link_beta(block_states)?;
        let o_infl = self.influence_index_offset(row, block_states)?;
        self.compute_row_flex_primary_gradient_from_parts(
            row, q_geom.q0, q_geom.q1, q_geom.qd1, g, beta_h, beta_w, o_infl, primary,
        )
    }

    fn compute_row_flex_primary_gradient_from_parts(
        &self,
        row: usize,
        q0: f64,
        q1: f64,
        qd1: f64,
        g: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        o_infl: f64,
        primary: &FlexPrimarySlices,
    ) -> Result<(f64, Array1<f64>), String> {
        if survival_derivative_guard_violated(qd1, self.derivative_guard) {
            return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
                reason: format!(
                    "survival marginal-slope monotonicity violated at row {row}: qd1={qd1:.3e} < guard={:.3e}",
                    self.derivative_guard
                ),
            }
            .into());
        }

        let (a0, d0) = self.solve_row_survival_intercept_with_slot(
            q0,
            g,
            beta_h,
            beta_w,
            Some((row, SurvivalInterceptSlotKind::Entry)),
        )?;
        let (a1, d1) = self.solve_row_survival_intercept_with_slot(
            q1,
            g,
            beta_h,
            beta_w,
            Some((row, SurvivalInterceptSlotKind::Exit)),
        )?;
        let entry = self.compute_survival_timepoint_first_order_exact(
            row, primary, q0, primary.q0, a0, g, d0, beta_h, beta_w, o_infl,
        )?;
        let exit = self.compute_survival_timepoint_first_order_exact(
            row, primary, q1, primary.q1, a1, g, d1, beta_h, beta_w, o_infl,
        )?;

        if !exit.chi.is_finite() || exit.chi <= 0.0 {
            return Err(SurvivalMarginalSlopeError::NumericalFailure {
                reason: format!(
                    "survival marginal-slope row {row} produced non-positive observed chi1={:.3e}",
                    exit.chi
                ),
            }
            .into());
        }

        let wi = self.weights[row];
        let di = self.event[row];
        let (log_surv0, _) = signed_probit_logcdf_and_mills_ratio(-entry.eta);
        let (log_surv1, _) = signed_probit_logcdf_and_mills_ratio(-exit.eta);
        let (entry_k1, _, _, _) = signed_probit_neglog_derivatives_up_to_fourth(-entry.eta, -wi)?;
        let (exit_k1, _, _, _) =
            signed_probit_neglog_derivatives_up_to_fourth(-exit.eta, wi * (1.0 - di))?;
        let log_phi_eta1 = -0.5 * (exit.eta * exit.eta + std::f64::consts::TAU.ln());
        let log_phi_q1 = -0.5 * (q1 * q1 + std::f64::consts::TAU.ln());
        let row_nll = wi
            * (log_surv0
                - (1.0 - di) * log_surv1
                - di * log_phi_eta1
                - di * exit.chi.ln()
                - di * log_phi_q1
                + di * exit.d.ln()
                - di * qd1.ln());

        let p = primary.total;
        let mut grad = Array1::<f64>::zeros(p);
        let entry_u1 = -entry_k1;
        let exit_surv_u1 = -exit_k1;

        for u in 0..p {
            grad[u] += entry_u1 * entry.eta_u[u];
            grad[u] += exit_surv_u1 * exit.eta_u[u];
            grad[u] += wi * di * exit.eta * exit.eta_u[u];
            grad[u] -= wi * di * exit.chi_u[u] / exit.chi;
            if u == primary.q1 {
                grad[u] += wi * di * q1;
            }
            grad[u] += wi * di * exit.d_u[u] / exit.d;
            if u == primary.qd1 {
                grad[u] -= wi * di / qd1;
            }
        }

        Ok((row_nll, grad))
    }

    fn compute_row_flex_primary_gradient_hessian_from_parts(
        &self,
        row: usize,
        q0: f64,
        q1: f64,
        qd1: f64,
        g: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        o_infl: f64,
        primary: &FlexPrimarySlices,
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        if survival_derivative_guard_violated(qd1, self.derivative_guard) {
            return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
                reason: format!(
                    "survival marginal-slope monotonicity violated at row {row}: qd1={qd1:.3e} < guard={:.3e}",
                    self.derivative_guard
                ),
            }
            .into());
        }

        let (a0, d0) = self.solve_row_survival_intercept_with_slot(
            q0,
            g,
            beta_h,
            beta_w,
            Some((row, SurvivalInterceptSlotKind::Entry)),
        )?;
        let (a1, d1) = self.solve_row_survival_intercept_with_slot(
            q1,
            g,
            beta_h,
            beta_w,
            Some((row, SurvivalInterceptSlotKind::Exit)),
        )?;
        let entry = self.compute_survival_timepoint_exact(
            row, primary, q0, primary.q0, a0, g, d0, beta_h, beta_w, o_infl, false,
        )?;
        let exit = self.compute_survival_timepoint_exact(
            row, primary, q1, primary.q1, a1, g, d1, beta_h, beta_w, o_infl, true,
        )?;

        if !exit.chi.is_finite() || exit.chi <= 0.0 {
            return Err(SurvivalMarginalSlopeError::NumericalFailure {
                reason: format!(
                    "survival marginal-slope row {row} produced non-positive observed chi1={:.3e}",
                    exit.chi
                ),
            }
            .into());
        }

        let wi = self.weights[row];
        let di = self.event[row];
        let (log_surv0, _) = signed_probit_logcdf_and_mills_ratio(-entry.eta);
        let (log_surv1, _) = signed_probit_logcdf_and_mills_ratio(-exit.eta);
        let (entry_k1, entry_k2, _, _) =
            signed_probit_neglog_derivatives_up_to_fourth(-entry.eta, -wi)?;
        let (exit_k1, exit_k2, _, _) =
            signed_probit_neglog_derivatives_up_to_fourth(-exit.eta, wi * (1.0 - di))?;

        // ── Step-6 dispatcher: try GPU Step-5 assembly first ──────────────
        //
        // The per-row primary G/H assembly is pure scalar/vector algebra over
        // the jets; the prep dispatchers (`try_device_partition_cells` +
        // `try_device_cell_primary_fixed_partials`) already produced these
        // jets via `compute_survival_timepoint_exact` →
        // `build_cached_partition_with_moment_order`, so the only remaining
        // family-side hop is the Step-5 G/H pullback in
        // [`crate::families::survival_marginal_slope_gpu::try_device_step5_primary_assembly`].
        //
        // The GPU entry takes flat `&[f64]` views; both `Array1` and `Array2`
        // returned by the timepoint-exact pass live in standard contiguous
        // layout (built via `Array1::zeros` / `Array2::zeros` above), so the
        // `as_slice()` extractions below are infallible in practice; we
        // route through the CPU fallback when any of them happens not to be
        // contiguous, preserving the legacy code path as the source of
        // truth.
        let p = primary.total;
        let entry_eta_u = entry.eta_u.as_slice();
        let entry_eta_uv = entry.eta_uv.as_slice();
        let entry_chi_u = entry.chi_u.as_slice();
        let entry_chi_uv = entry.chi_uv.as_slice();
        let entry_d_u = entry.d_u.as_slice();
        let entry_d_uv = entry.d_uv.as_slice();
        let exit_eta_u = exit.eta_u.as_slice();
        let exit_eta_uv = exit.eta_uv.as_slice();
        let exit_chi_u = exit.chi_u.as_slice();
        let exit_chi_uv = exit.chi_uv.as_slice();
        let exit_d_u = exit.d_u.as_slice();
        let exit_d_uv = exit.d_uv.as_slice();
        let all_contiguous = entry_eta_u.is_some()
            && entry_eta_uv.is_some()
            && entry_chi_u.is_some()
            && entry_chi_uv.is_some()
            && entry_d_u.is_some()
            && entry_d_uv.is_some()
            && exit_eta_u.is_some()
            && exit_eta_uv.is_some()
            && exit_chi_u.is_some()
            && exit_chi_uv.is_some()
            && exit_d_u.is_some()
            && exit_d_uv.is_some();
        if all_contiguous
            && exit.chi.is_finite()
            && exit.chi > 0.0
            && exit.d.is_finite()
            && exit.d > 0.0
        {
            let row_inputs = [
                crate::families::survival_marginal_slope_gpu::SurvivalFlexStep5RowInputs {
                    entry: crate::families::survival_marginal_slope_gpu::SurvivalFlexTimepointJet {
                        eta: entry.eta,
                        chi: entry.chi,
                        d: entry.d,
                        eta_u: entry_eta_u.unwrap(),
                        eta_uv: entry_eta_uv.unwrap(),
                        chi_u: entry_chi_u.unwrap(),
                        chi_uv: entry_chi_uv.unwrap(),
                        d_u: entry_d_u.unwrap(),
                        d_uv: entry_d_uv.unwrap(),
                    },
                    exit: crate::families::survival_marginal_slope_gpu::SurvivalFlexTimepointJet {
                        eta: exit.eta,
                        chi: exit.chi,
                        d: exit.d,
                        eta_u: exit_eta_u.unwrap(),
                        eta_uv: exit_eta_uv.unwrap(),
                        chi_u: exit_chi_u.unwrap(),
                        chi_uv: exit_chi_uv.unwrap(),
                        d_u: exit_d_u.unwrap(),
                        d_uv: exit_d_uv.unwrap(),
                    },
                    wi,
                    di,
                    q1,
                    qd1,
                    q1_index: primary.q1,
                    qd1_index: primary.qd1,
                    entry_k1,
                    entry_k2,
                    exit_k1,
                    exit_k2,
                    log_surv0,
                    log_surv1,
                },
            ];
            // `try_device_step5_primary_assembly` is the device-shape Step-5
            // pullback (`survival_marginal_slope_gpu`). Its current body
            // is CPU-resident scalar algebra producing the same `(row_nll,
            // grad, hess)` the inline CPU loop below builds; when the NVRTC
            // kernel lands, this call site becomes the device-dispatch seam
            // without further family-side rework.  We fall back to the
            // inline CPU loop on `Err` so any device-side validation failure
            // surfaces as a row-level CPU re-evaluation rather than a hard
            // panic.
            if let Ok(mut out) =
                crate::families::survival_marginal_slope_gpu::try_device_step5_primary_assembly(
                    &row_inputs,
                )
                && out.len() == 1
            {
                let row = out.remove(0);
                if row.grad.len() == p && row.hess.len() == p * p {
                    let grad = Array1::from_vec(row.grad);
                    let hess =
                        Array2::from_shape_vec((p, p), row.hess).map_err(|e| e.to_string())?;
                    return Ok((row.row_nll, grad, hess));
                }
            }
        }

        let log_phi_eta1 = -0.5 * (exit.eta * exit.eta + std::f64::consts::TAU.ln());
        let log_phi_q1 = -0.5 * (q1 * q1 + std::f64::consts::TAU.ln());
        let row_nll = wi
            * (log_surv0
                - (1.0 - di) * log_surv1
                - di * log_phi_eta1
                - di * exit.chi.ln()
                - di * log_phi_q1
                + di * exit.d.ln()
                - di * qd1.ln());

        let mut grad = Array1::<f64>::zeros(p);
        let mut hess = Array2::<f64>::zeros((p, p));
        let entry_u1 = -entry_k1;
        let entry_u2 = entry_k2;
        let exit_surv_u1 = -exit_k1;
        let exit_surv_u2 = exit_k2;

        for u in 0..p {
            grad[u] += entry_u1 * entry.eta_u[u];
            grad[u] += exit_surv_u1 * exit.eta_u[u];
            grad[u] += wi * di * exit.eta * exit.eta_u[u];
            grad[u] -= wi * di * exit.chi_u[u] / exit.chi;
            if u == primary.q1 {
                grad[u] += wi * di * q1;
            }
            grad[u] += wi * di * exit.d_u[u] / exit.d;
            if u == primary.qd1 {
                grad[u] -= wi * di / qd1;
            }
        }

        for u in 0..p {
            for v in u..p {
                let mut value = 0.0;
                value +=
                    entry_u2 * entry.eta_u[u] * entry.eta_u[v] + entry_u1 * entry.eta_uv[[u, v]];
                value += exit_surv_u2 * exit.eta_u[u] * exit.eta_u[v]
                    + exit_surv_u1 * exit.eta_uv[[u, v]];
                value += wi * di * (exit.eta_u[u] * exit.eta_u[v] + exit.eta * exit.eta_uv[[u, v]]);
                value -= wi
                    * di
                    * (exit.chi_uv[[u, v]] / exit.chi
                        - (exit.chi_u[u] * exit.chi_u[v]) / (exit.chi * exit.chi));
                if u == primary.q1 && v == primary.q1 {
                    value += wi * di;
                }
                value += wi
                    * di
                    * (exit.d_uv[[u, v]] / exit.d
                        - (exit.d_u[u] * exit.d_u[v]) / (exit.d * exit.d));
                if u == primary.qd1 && v == primary.qd1 {
                    value += wi * di / (qd1 * qd1);
                }
                hess[[u, v]] = value;
                hess[[v, u]] = value;
            }
        }

        Ok((row_nll, grad, hess))
    }

    /// Batched variant of the per-row NLL multi-directional jet, returning the full
    /// `MultiDirJet` for the per-row NLL composed against an arbitrary
    /// ordered list of `dirs` (one direction per jet bit).
    ///
    /// Callers reading higher-order contracted tensors (third/fourth)
    /// previously looped the 10 upper-triangular (a, b) pairs and called
    /// `row_neglog_directional_refs` ten times, each rebuilding the same
    /// q-geometry, the same primary jets, and rerunning the same composed
    /// unary derivatives. By packing the basis directions e_0..e_3 alongside
    /// the user-supplied `dir` (and optional `dir_v`) into one larger jet, the
    /// per-row q-geometry + jet-construction work is shared across all 10
    /// upper-triangular cells. Off-diagonal entries are then a single mask
    /// read; diagonal entries fall back to a small per-axis jet that still
    /// reuses no work between cells.
    ///
    /// `dirs.len()` must be in `1..=8` (MAX_DIRS in `jet_partitions`).
    fn row_neglog_directional_jet_batched(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dirs: &[ArrayView1<'_, f64>],
    ) -> Result<MultiDirJet, String> {
        crate::families::jet_partitions::ROW_NEGLOG_CALLS
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let k = dirs.len();
        if k == 0 || k > 8 {
            return Err(SurvivalMarginalSlopeError::InvalidInput {
                reason: format!(
                    "survival marginal-slope row directional jet expects 1..=8 directions, got {k}"
                ),
            }
            .into());
        }
        let wi = self.weights[row];
        let di = self.event[row];
        let (z_sum, covariance_ones) = self.exact_shared_score_summary(
            row,
            block_states,
            "row_neglog_directional_jet_batched",
        )?;

        // Primary scalar jets: q0, q1, qd1, g. Fixed-capacity stack buffers
        // (k ≤ 8) avoid heap allocation in the per-row hot loop.
        let mut q0_first_buf = [0.0f64; 8];
        let mut q1_first_buf = [0.0f64; 8];
        let mut qd1_first_buf = [0.0f64; 8];
        let mut g_first_buf = [0.0f64; 8];
        for (i, dir) in dirs.iter().enumerate() {
            q0_first_buf[i] = dir[0];
            q1_first_buf[i] = dir[1];
            qd1_first_buf[i] = dir[2];
            g_first_buf[i] = dir[3];
        }
        let q0_first: &[f64] = &q0_first_buf[..k];
        let q1_first: &[f64] = &q1_first_buf[..k];
        let qd1_first: &[f64] = &qd1_first_buf[..k];
        let g_first: &[f64] = &g_first_buf[..k];

        // Reuse the realized q-geometry (timewiggle / frailty manifold).
        let q_geom = self.row_dynamic_q_values(row, block_states)?;
        let q0_val = q_geom.q0;
        let q1_val = q_geom.q1;
        let qd1_val = q_geom.qd1;
        let g_val = block_states[2].eta[row];

        let q0_jet = MultiDirJet::linear(k, q0_val, q0_first);
        let q1_jet = MultiDirJet::linear(k, q1_val, q1_first);
        let qd1_jet = MultiDirJet::linear(k, qd1_val, qd1_first);
        let g_jet = MultiDirJet::linear(k, g_val, g_first);

        // observed_g = frailty_scale * g
        let observed_g_jet = g_jet.scale(self.probit_frailty_scale());
        // c = sqrt(1 + observed_g^2 * 1' Sigma 1) for a shared K-vector slope.
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
        let qd1_val_chk = qd1_jet.coeff(0);
        let ad1_val = ad1_jet.coeff(0);
        if survival_derivative_guard_violated(qd1_val_chk, qd1_lower) {
            return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
                reason: format!(
                    "survival marginal-slope monotonicity violated at row {row}: raw time derivative={qd1_val_chk:.3e} must be at least derivative_guard={qd1_lower:.3e}; transformed time derivative={ad1_val:.3e}"
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

        let total = exit_term
            .add(&entry_term)
            .add(&event_density_term)
            .add(&time_deriv_term);

        Ok(total)
    }

    /// Build the row's third-order contracted tensor `T[a][b] = ∂_{e_a} ∂_{e_b} ∂_{dir}` of NLL_i
    /// against the four primary axes (a, b ∈ {0..N_PRIMARY}). Six off-diagonal
    /// entries are read from a single batched k=5 jet over [e_0..e_3, dir];
    /// four diagonal entries (a == b) need order-3 in a single basis axis and
    /// fall back to a k=3 jet [e_a, e_a, dir]. Result is symmetric.
    fn row_primary_third_contracted_batched(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dir: ArrayView1<'_, f64>,
    ) -> Result<[[f64; 4]; 4], String> {
        let e0 = unit_primary_direction_ref(0).view();
        let e1 = unit_primary_direction_ref(1).view();
        let e2 = unit_primary_direction_ref(2).view();
        let e3 = unit_primary_direction_ref(3).view();
        // One k=5 jet covers all 6 off-diagonal (a < b) entries via masks
        // `(1<<a) | (1<<b) | (1<<4)`.
        let batched =
            self.row_neglog_directional_jet_batched(row, block_states, &[e0, e1, e2, e3, dir])?;
        let dir_bit = 1usize << 4;
        let mut r = [[0.0_f64; 4]; 4];
        for a in 0..N_PRIMARY {
            for b in (a + 1)..N_PRIMARY {
                let mask = (1usize << a) | (1usize << b) | dir_bit;
                let value = batched.coeff(mask);
                r[a][b] = value;
                r[b][a] = value;
            }
        }
        // Diagonal entries: ∂²_{e_a} ∂_{dir} require two copies of e_a in
        // distinct bit positions. A k=3 jet [e_a, e_a, dir] read at full
        // mask (7) supplies that.
        for a in 0..N_PRIMARY {
            let ea = unit_primary_direction_ref(a).view();
            let diag_jet =
                self.row_neglog_directional_jet_batched(row, block_states, &[ea, ea, dir])?;
            r[a][a] = diag_jet.coeff(diag_jet.full_mask());
        }
        Ok(r)
    }

    /// Build the row's fourth-order contracted tensor
    /// `T[a][b] = ∂_{e_a} ∂_{e_b} ∂_{dir_u} ∂_{dir_v}` of NLL_i, mirrored.
    fn row_primary_fourth_contracted_batched(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dir_u: ArrayView1<'_, f64>,
        dir_v: ArrayView1<'_, f64>,
    ) -> Result<[[f64; 4]; 4], String> {
        let e0 = unit_primary_direction_ref(0).view();
        let e1 = unit_primary_direction_ref(1).view();
        let e2 = unit_primary_direction_ref(2).view();
        let e3 = unit_primary_direction_ref(3).view();
        // One k=6 jet covers all 6 off-diagonal (a < b) entries via masks
        // `(1<<a) | (1<<b) | (1<<4) | (1<<5)`.
        let batched = self.row_neglog_directional_jet_batched(
            row,
            block_states,
            &[e0, e1, e2, e3, dir_u, dir_v],
        )?;
        let u_bit = 1usize << 4;
        let v_bit = 1usize << 5;
        let mut r = [[0.0_f64; 4]; 4];
        for a in 0..N_PRIMARY {
            for b in (a + 1)..N_PRIMARY {
                let mask = (1usize << a) | (1usize << b) | u_bit | v_bit;
                let value = batched.coeff(mask);
                r[a][b] = value;
                r[b][a] = value;
            }
        }
        for a in 0..N_PRIMARY {
            let ea = unit_primary_direction_ref(a).view();
            let diag_jet = self.row_neglog_directional_jet_batched(
                row,
                block_states,
                &[ea, ea, dir_u, dir_v],
            )?;
            r[a][a] = diag_jet.coeff(diag_jet.full_mask());
        }
        Ok(r)
    }

    /// Compute per-row primary gradient and Hessian using the closed-form
    /// scalar kernel.  The hot inner computation uses stack arrays only;
    /// conversion to Array1/Array2 happens once at the boundary for API
    /// compatibility with outer-derivative paths.
    fn compute_row_primary_gradient_hessian_uncached(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        let q_geom = self.row_dynamic_q_values(row, block_states)?;
        let (nll, grad_arr, hess_arr) = self.row_primary_closed_form_rigid(
            row,
            q_geom.q0,
            q_geom.q1,
            q_geom.qd1,
            block_states,
            self.probit_frailty_scale(),
        )?;
        // Convert stack arrays to ndarray types at the boundary.
        let grad = Array1::from_vec(grad_arr.to_vec());
        let mut hess = Array2::zeros((N_PRIMARY, N_PRIMARY));
        for i in 0..N_PRIMARY {
            for j in 0..N_PRIMARY {
                hess[[i, j]] = hess_arr[i][j];
            }
        }
        Ok((nll, grad, hess))
    }

    fn build_eval_cache(&self, block_states: &[ParameterBlockState]) -> Result<EvalCache, String> {
        let row_bases = (0..self.n)
            .into_par_iter()
            .map(|row| {
                let (_, gradient, hessian) =
                    self.compute_row_primary_gradient_hessian_uncached(row, block_states)?;
                Ok(RowPrimaryBase { gradient, hessian })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(EvalCache { row_bases })
    }

    fn row_primary_gradient_hessian<'a>(
        &self,
        row: usize,
        cache: &'a EvalCache,
    ) -> (&'a Array1<f64>, &'a Array2<f64>) {
        let base = &cache.row_bases[row];
        (&base.gradient, &base.hessian)
    }

    fn offset_channel_geometry(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<(OffsetChannelResiduals, OffsetChannelCurvatures), String> {
        let flex_active = self.effective_flex_active(block_states)?;
        let primary = flex_active.then(|| flex_primary_slices(self));
        let rows = (0..self.n)
            .into_par_iter()
            .map(
                |row| -> Result<(usize, f64, f64, f64, [[f64; 3]; 3]), String> {
                    if self.weights[row] <= 0.0 {
                        return Ok((row, 0.0, 0.0, 0.0, [[0.0; 3]; 3]));
                    }
                    let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
                    let (gradient, hessian) = if let Some(primary) = primary.as_ref() {
                        let (_, gradient, hessian) = self
                            .compute_row_flex_primary_gradient_hessian_exact(
                                row,
                                block_states,
                                &q_geom,
                                primary,
                            )?;
                        (gradient, hessian)
                    } else {
                        let (_, gradient, hessian) =
                            self.compute_row_primary_gradient_hessian_uncached(row, block_states)?;
                        (gradient, hessian)
                    };
                    let channel = [0usize, 1usize, 2usize];
                    let mut curvature = [[0.0; 3]; 3];
                    for a in 0..3 {
                        for b in 0..3 {
                            curvature[a][b] = hessian[[channel[a], channel[b]]];
                        }
                    }
                    Ok((row, gradient[1], gradient[0], gradient[2], curvature))
                },
            )
            .collect::<Result<Vec<_>, String>>()?;
        let mut exit = Array1::<f64>::zeros(self.n);
        let mut entry = Array1::<f64>::zeros(self.n);
        let mut derivative = Array1::<f64>::zeros(self.n);
        let mut curvatures = vec![[[0.0; 3]; 3]; self.n];
        for (row, r_exit, r_entry, r_derivative, curvature) in rows {
            exit[row] = r_exit;
            entry[row] = r_entry;
            derivative[row] = r_derivative;
            curvatures[row] = curvature;
        }
        Ok((
            OffsetChannelResiduals {
                exit,
                entry,
                derivative,
            },
            OffsetChannelCurvatures { rows: curvatures },
        ))
    }

    fn accumulate_dynamic_q_core_gradient(
        &self,
        row: usize,
        slices: &BlockSlices,
        q_geom: &SurvivalMarginalSlopeDynamicRow,
        primary_gradient: ndarray::ArrayView1<'_, f64>,
        joint_gradient: &mut Array1<f64>,
    ) -> Result<(), String> {
        let dq_time = [&q_geom.dq0_time, &q_geom.dq1_time, &q_geom.dqd1_time];
        let dq_marginal = [
            &q_geom.dq0_marginal,
            &q_geom.dq1_marginal,
            &q_geom.dqd1_marginal,
        ];

        for (q_idx, dq) in dq_time.iter().enumerate() {
            for coeff_idx in 0..dq.len() {
                joint_gradient[slices.time.start + coeff_idx] -=
                    primary_gradient[q_idx] * dq[coeff_idx];
            }
        }
        for (q_idx, dq) in dq_marginal.iter().enumerate() {
            for coeff_idx in 0..dq.len() {
                joint_gradient[slices.marginal.start + coeff_idx] -=
                    primary_gradient[q_idx] * dq[coeff_idx];
            }
        }
        self.logslope_design.axpy_row_into(
            row,
            -primary_gradient[3],
            &mut joint_gradient.slice_mut(s![slices.logslope.clone()]),
        )?;
        Ok(())
    }

    fn accumulate_dynamic_q_core_gradient_first_order(
        &self,
        row: usize,
        slices: &BlockSlices,
        q_geom: &SurvivalMarginalSlopeDynamicRowGradient,
        primary_gradient: ndarray::ArrayView1<'_, f64>,
        joint_gradient: &mut Array1<f64>,
    ) -> Result<(), String> {
        let dq_time = [&q_geom.dq0_time, &q_geom.dq1_time, &q_geom.dqd1_time];
        let dq_marginal = [
            &q_geom.dq0_marginal,
            &q_geom.dq1_marginal,
            &q_geom.dqd1_marginal,
        ];

        for (q_idx, dq) in dq_time.iter().enumerate() {
            for coeff_idx in 0..dq.len() {
                joint_gradient[slices.time.start + coeff_idx] -=
                    primary_gradient[q_idx] * dq[coeff_idx];
            }
        }
        for (q_idx, dq) in dq_marginal.iter().enumerate() {
            for coeff_idx in 0..dq.len() {
                joint_gradient[slices.marginal.start + coeff_idx] -=
                    primary_gradient[q_idx] * dq[coeff_idx];
            }
        }
        self.logslope_design.axpy_row_into(
            row,
            -primary_gradient[3],
            &mut joint_gradient.slice_mut(s![slices.logslope.clone()]),
        )?;
        Ok(())
    }

    fn accumulate_dynamic_q_core_hessian(
        &self,
        row: usize,
        slices: &BlockSlices,
        q_geom: &SurvivalMarginalSlopeDynamicRow,
        primary_gradient: ndarray::ArrayView1<'_, f64>,
        primary_hessian: ArrayView2<'_, f64>,
        joint_hessian: &mut Array2<f64>,
    ) -> Result<(), String> {
        // Perf (#large-scale): scatter each core block-Hessian contribution
        // directly into `joint_hessian` as it is computed, instead of building
        // six fresh `Array2` per row in `dynamic_q_core_hessian_blocks` and
        // then copying them in. This removes the per-row heap traffic (6×p²
        // allocations + zero-fills + the logslope-row `to_owned`) and the
        // extra read/write pass over the temporaries. Each cell receives the
        // identical `value` it did before (the temporaries were written with
        // `=` then added here), so the accumulated result is bit-identical.
        let p_t = slices.time.len();
        let p_m = slices.marginal.len();
        let p_g = slices.logslope.len();

        let dq_time = [&q_geom.dq0_time, &q_geom.dq1_time, &q_geom.dqd1_time];
        let dq_marginal = [
            &q_geom.dq0_marginal,
            &q_geom.dq1_marginal,
            &q_geom.dqd1_marginal,
        ];
        let d2q_time_time = [
            &q_geom.d2q0_time_time,
            &q_geom.d2q1_time_time,
            &q_geom.d2qd1_time_time,
        ];
        let d2q_marginal_marginal = [
            &q_geom.d2q0_marginal_marginal,
            &q_geom.d2q1_marginal_marginal,
            &q_geom.d2qd1_marginal_marginal,
        ];
        let d2q_time_marginal = [
            &q_geom.d2q0_time_marginal,
            &q_geom.d2q1_time_marginal,
            &q_geom.d2qd1_time_marginal,
        ];
        let logslope_chunk = self
            .logslope_design
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("accumulate_dynamic_q_core_hessian logslope: {e}"))?;
        let logslope_row = logslope_chunk.row(0);

        let t0 = slices.time.start;
        let m0 = slices.marginal.start;
        let g0 = slices.logslope.start;

        // time × time
        for a in 0..p_t {
            for b in 0..p_t {
                let mut value = 0.0;
                for q_u in 0..3 {
                    for q_v in 0..3 {
                        value += primary_hessian[[q_u, q_v]] * dq_time[q_u][a] * dq_time[q_v][b];
                    }
                    value += primary_gradient[q_u] * d2q_time_time[q_u][[a, b]];
                }
                joint_hessian[[t0 + a, t0 + b]] += value;
            }
        }
        // marginal × marginal
        for a in 0..p_m {
            for b in 0..p_m {
                let mut value = 0.0;
                for q_u in 0..3 {
                    for q_v in 0..3 {
                        value +=
                            primary_hessian[[q_u, q_v]] * dq_marginal[q_u][a] * dq_marginal[q_v][b];
                    }
                    value += primary_gradient[q_u] * d2q_marginal_marginal[q_u][[a, b]];
                }
                joint_hessian[[m0 + a, m0 + b]] += value;
            }
        }
        // logslope × logslope (rank-1: h_gg · xxᵀ); zero cells skipped exactly
        // as the prior `Array2::zeros`-backed block left them zero.
        let h_gg_scale = primary_hessian[[3, 3]];
        if h_gg_scale != 0.0 {
            for a in 0..p_g {
                let xa = logslope_row[a];
                if xa == 0.0 {
                    continue;
                }
                let row_scale = h_gg_scale * xa;
                for b in 0..p_g {
                    joint_hessian[[g0 + a, g0 + b]] += row_scale * logslope_row[b];
                }
            }
        }
        // time × marginal (symmetric scatter)
        for a in 0..p_t {
            for b in 0..p_m {
                let mut value = 0.0;
                for q_u in 0..3 {
                    for q_v in 0..3 {
                        value +=
                            primary_hessian[[q_u, q_v]] * dq_time[q_u][a] * dq_marginal[q_v][b];
                    }
                    value += primary_gradient[q_u] * d2q_time_marginal[q_u][[a, b]];
                }
                joint_hessian[[t0 + a, m0 + b]] += value;
                joint_hessian[[m0 + b, t0 + a]] += value;
            }
        }
        // time × logslope (symmetric scatter)
        for a in 0..p_t {
            let mut weight = 0.0;
            for q_u in 0..3 {
                weight += primary_hessian[[q_u, 3]] * dq_time[q_u][a];
            }
            if weight != 0.0 {
                for b in 0..p_g {
                    let value = weight * logslope_row[b];
                    joint_hessian[[t0 + a, g0 + b]] += value;
                    joint_hessian[[g0 + b, t0 + a]] += value;
                }
            }
        }
        // marginal × logslope (symmetric scatter)
        for a in 0..p_m {
            let mut weight = 0.0;
            for q_u in 0..3 {
                weight += primary_hessian[[q_u, 3]] * dq_marginal[q_u][a];
            }
            if weight != 0.0 {
                for b in 0..p_g {
                    let value = weight * logslope_row[b];
                    joint_hessian[[m0 + a, g0 + b]] += value;
                    joint_hessian[[g0 + b, m0 + a]] += value;
                }
            }
        }
        Ok(())
    }

    fn accumulate_dynamic_q_blockwise_gradient(
        &self,
        row: usize,
        q_geom: &SurvivalMarginalSlopeDynamicRow,
        primary_gradient: ndarray::ArrayView1<'_, f64>,
        grad_time: &mut Array1<f64>,
        grad_marginal: &mut Array1<f64>,
        grad_logslope: &mut Array1<f64>,
    ) -> Result<(), String> {
        let dq_time = [&q_geom.dq0_time, &q_geom.dq1_time, &q_geom.dqd1_time];
        let dq_marginal = [
            &q_geom.dq0_marginal,
            &q_geom.dq1_marginal,
            &q_geom.dqd1_marginal,
        ];
        for (q_idx, dq) in dq_time.iter().enumerate() {
            for coeff_idx in 0..dq.len() {
                grad_time[coeff_idx] -= primary_gradient[q_idx] * dq[coeff_idx];
            }
        }
        for (q_idx, dq) in dq_marginal.iter().enumerate() {
            for coeff_idx in 0..dq.len() {
                grad_marginal[coeff_idx] -= primary_gradient[q_idx] * dq[coeff_idx];
            }
        }
        self.logslope_design.axpy_row_into(
            row,
            -primary_gradient[3],
            &mut grad_logslope.view_mut(),
        )?;
        Ok(())
    }

    fn accumulate_dynamic_q_core_block_hessians(
        &self,
        row: usize,
        q_geom: &SurvivalMarginalSlopeDynamicRow,
        primary_gradient: ndarray::ArrayView1<'_, f64>,
        primary_hessian: ArrayView2<'_, f64>,
        hess_time: &mut Array2<f64>,
        hess_marginal: &mut Array2<f64>,
        hess_logslope: &mut Array2<f64>,
    ) -> Result<(), String> {
        // Perf (#large-scale): accumulate the three diagonal block-Hessian
        // contributions directly into the caller's per-thread workspace
        // buffers with `+=`, rather than allocating three fresh
        // `Array2::zeros` per row (`dynamic_q_core_diagonal_hessian_blocks`)
        // and then folding them in with `*hess += &local`. This removes
        // O(n·p²) heap allocation + zero-fill + a redundant add pass. The
        // arithmetic per cell is identical: the old path wrote `value` into a
        // zeroed local then added it here, so accumulating `value` directly is
        // bit-identical. The logslope block skips zero cells exactly as before
        // (adding the implicit zeros was a no-op).
        let p_t = hess_time.nrows();
        let p_m = hess_marginal.nrows();
        let p_g = hess_logslope.nrows();

        let dq_time = [&q_geom.dq0_time, &q_geom.dq1_time, &q_geom.dqd1_time];
        let dq_marginal = [
            &q_geom.dq0_marginal,
            &q_geom.dq1_marginal,
            &q_geom.dqd1_marginal,
        ];
        let d2q_time_time = [
            &q_geom.d2q0_time_time,
            &q_geom.d2q1_time_time,
            &q_geom.d2qd1_time_time,
        ];
        let d2q_marginal_marginal = [
            &q_geom.d2q0_marginal_marginal,
            &q_geom.d2q1_marginal_marginal,
            &q_geom.d2qd1_marginal_marginal,
        ];
        let logslope_chunk = self
            .logslope_design
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("accumulate_dynamic_q_core_block_hessians logslope: {e}"))?;
        let logslope_row = logslope_chunk.row(0);

        for a in 0..p_t {
            for b in 0..p_t {
                let mut value = 0.0;
                for q_u in 0..3 {
                    for q_v in 0..3 {
                        value += primary_hessian[[q_u, q_v]] * dq_time[q_u][a] * dq_time[q_v][b];
                    }
                    value += primary_gradient[q_u] * d2q_time_time[q_u][[a, b]];
                }
                hess_time[[a, b]] += value;
            }
        }
        for a in 0..p_m {
            for b in 0..p_m {
                let mut value = 0.0;
                for q_u in 0..3 {
                    for q_v in 0..3 {
                        value +=
                            primary_hessian[[q_u, q_v]] * dq_marginal[q_u][a] * dq_marginal[q_v][b];
                    }
                    value += primary_gradient[q_u] * d2q_marginal_marginal[q_u][[a, b]];
                }
                hess_marginal[[a, b]] += value;
            }
        }
        let h_gg_scale = primary_hessian[[3, 3]];
        if h_gg_scale != 0.0 {
            for a in 0..p_g {
                let xa = logslope_row[a];
                if xa == 0.0 {
                    continue;
                }
                let row_scale = h_gg_scale * xa;
                for b in 0..p_g {
                    hess_logslope[[a, b]] += row_scale * logslope_row[b];
                }
            }
        }
        Ok(())
    }

    fn accumulate_dynamic_q_blockwise_row(
        &self,
        row: usize,
        q_geom: &SurvivalMarginalSlopeDynamicRow,
        primary: &FlexPrimarySlices,
        primary_gradient: ndarray::ArrayView1<'_, f64>,
        primary_hessian: ArrayView2<'_, f64>,
        acc: &mut DynamicQBlockwiseAccumulator,
    ) -> Result<(), String> {
        self.accumulate_dynamic_q_blockwise_gradient(
            row,
            q_geom,
            primary_gradient.slice(s![0..N_PRIMARY]),
            &mut acc.grad_time,
            &mut acc.grad_marginal,
            &mut acc.grad_logslope,
        )?;
        self.accumulate_dynamic_q_core_block_hessians(
            row,
            q_geom,
            primary_gradient.slice(s![0..N_PRIMARY]),
            primary_hessian.slice(s![0..N_PRIMARY, 0..N_PRIMARY]),
            &mut acc.hess_time,
            &mut acc.hess_marginal,
            &mut acc.hess_logslope,
        )?;
        if let (Some(primary_range), Some(gradient), Some(hessian)) = (
            primary.h.as_ref(),
            acc.grad_score_warp.as_mut(),
            acc.hess_score_warp.as_mut(),
        ) {
            *gradient -= &primary_gradient.slice(s![primary_range.clone()]);
            *hessian += &primary_hessian
                .slice(s![primary_range.clone(), primary_range.clone()])
                .to_owned();
        }
        if let (Some(primary_range), Some(gradient), Some(hessian)) = (
            primary.w.as_ref(),
            acc.grad_link_dev.as_mut(),
            acc.hess_link_dev.as_mut(),
        ) {
            *gradient -= &primary_gradient.slice(s![primary_range.clone()]);
            *hessian += &primary_hessian
                .slice(s![primary_range.clone(), primary_range.clone()])
                .to_owned();
        }
        // Absorbed-influence diagonal block (#461). Unlike the identity flex
        // blocks above (whose basis IS the primary coordinate, so the block grad
        // is a slice copy), the absorber's `p₁` coefficients project from the
        // single `o_infl` primary scalar through `Z̃_infl[row,:]`:
        //   grad_i = -primary_gradient[infl] · Z̃[row,i]
        //   hess_ij += primary_hessian[[infl,infl]] · Z̃[row,i] · Z̃[row,j]
        if let (Some(infl_idx), Some(gradient), Some(hessian)) = (
            primary.infl,
            acc.grad_influence.as_mut(),
            acc.hess_influence.as_mut(),
        ) {
            let z_tilde = self.influence_absorber.as_ref().ok_or_else(|| {
                "accumulate_dynamic_q_blockwise_row: influence primary index present but no Z̃ design"
                    .to_string()
            })?;
            let z_row = z_tilde.row(row);
            let g_infl = primary_gradient[infl_idx];
            let h_infl = primary_hessian[[infl_idx, infl_idx]];
            for i in 0..z_row.len() {
                gradient[i] -= g_infl * z_row[i];
                if h_infl != 0.0 {
                    let hz = h_infl * z_row[i];
                    for j in 0..z_row.len() {
                        hessian[[i, j]] += hz * z_row[j];
                    }
                }
            }
        }
        Ok(())
    }

    fn accumulate_identity_primary_cross_hessian(
        &self,
        row: usize,
        slices: &BlockSlices,
        q_geom: &SurvivalMarginalSlopeDynamicRow,
        core_hessian_column: ndarray::ArrayView1<'_, f64>,
        joint_block: &std::ops::Range<usize>,
        joint_local: usize,
        joint_hessian: &mut Array2<f64>,
    ) -> Result<(), String> {
        let joint_idx = joint_block.start + joint_local;
        let dq_time = [&q_geom.dq0_time, &q_geom.dq1_time, &q_geom.dqd1_time];
        let dq_marginal = [
            &q_geom.dq0_marginal,
            &q_geom.dq1_marginal,
            &q_geom.dqd1_marginal,
        ];

        for coeff_idx in 0..slices.time.len() {
            let mut value = 0.0;
            for q_idx in 0..3 {
                value += core_hessian_column[q_idx] * dq_time[q_idx][coeff_idx];
            }
            joint_hessian[[slices.time.start + coeff_idx, joint_idx]] += value;
            joint_hessian[[joint_idx, slices.time.start + coeff_idx]] += value;
        }
        for coeff_idx in 0..slices.marginal.len() {
            let mut value = 0.0;
            for q_idx in 0..3 {
                value += core_hessian_column[q_idx] * dq_marginal[q_idx][coeff_idx];
            }
            joint_hessian[[slices.marginal.start + coeff_idx, joint_idx]] += value;
            joint_hessian[[joint_idx, slices.marginal.start + coeff_idx]] += value;
        }
        let logslope_chunk = self
            .logslope_design
            .try_row_chunk(row..row + 1)
            .map_err(|e| {
                format!("accumulate_identity_primary_cross_hessian logslope try_row_chunk: {e}")
            })?;
        let logslope_row = logslope_chunk.row(0);
        let logslope_weight = core_hessian_column[3];
        if logslope_weight != 0.0 {
            for coeff_idx in 0..slices.logslope.len() {
                let value = logslope_weight * logslope_row[coeff_idx];
                joint_hessian[[slices.logslope.start + coeff_idx, joint_idx]] += value;
                joint_hessian[[joint_idx, slices.logslope.start + coeff_idx]] += value;
            }
        }
        Ok(())
    }

    /// Perf (#large-scale): pre-scaled variant of
    /// [`Self::accumulate_identity_primary_cross_hessian`]. The influence
    /// absorber needs the `o_infl` core-Hessian column scaled by the per-row
    /// per-coefficient factor `Z̃[row, i]`. The previous call site materialised
    /// `&core_col.to_owned() * z_i` — two fresh `Array1` allocations per (row,
    /// influence-coefficient) pair — purely to pass a scaled view in. Here the
    /// scale is folded `core_hessian_column[q] * scale` *before* multiplying by
    /// the Jacobian, exactly matching the original operand grouping
    /// `(core_col[q] * z_i) * dq`, so every accumulated cell is bit-identical
    /// while the per-row heap traffic in the influence-active path is removed.
    fn accumulate_identity_primary_cross_hessian_scaled(
        &self,
        row: usize,
        slices: &BlockSlices,
        q_geom: &SurvivalMarginalSlopeDynamicRow,
        core_hessian_column: ndarray::ArrayView1<'_, f64>,
        scale: f64,
        joint_block: &std::ops::Range<usize>,
        joint_local: usize,
        joint_hessian: &mut Array2<f64>,
    ) -> Result<(), String> {
        let joint_idx = joint_block.start + joint_local;
        let dq_time = [&q_geom.dq0_time, &q_geom.dq1_time, &q_geom.dqd1_time];
        let dq_marginal = [
            &q_geom.dq0_marginal,
            &q_geom.dq1_marginal,
            &q_geom.dqd1_marginal,
        ];
        // Fold the scale into the three primary-q weights up front so the
        // per-coefficient inner product reads `(core[q] * scale) * dq[q]`,
        // matching the pre-scaled-column arithmetic exactly.
        let scaled_core = [
            core_hessian_column[0] * scale,
            core_hessian_column[1] * scale,
            core_hessian_column[2] * scale,
        ];

        for coeff_idx in 0..slices.time.len() {
            let mut value = 0.0;
            for q_idx in 0..3 {
                value += scaled_core[q_idx] * dq_time[q_idx][coeff_idx];
            }
            joint_hessian[[slices.time.start + coeff_idx, joint_idx]] += value;
            joint_hessian[[joint_idx, slices.time.start + coeff_idx]] += value;
        }
        for coeff_idx in 0..slices.marginal.len() {
            let mut value = 0.0;
            for q_idx in 0..3 {
                value += scaled_core[q_idx] * dq_marginal[q_idx][coeff_idx];
            }
            joint_hessian[[slices.marginal.start + coeff_idx, joint_idx]] += value;
            joint_hessian[[joint_idx, slices.marginal.start + coeff_idx]] += value;
        }
        let logslope_weight = core_hessian_column[3] * scale;
        if logslope_weight != 0.0 {
            let logslope_chunk = self
                .logslope_design
                .try_row_chunk(row..row + 1)
                .map_err(|e| {
                    format!(
                        "accumulate_identity_primary_cross_hessian_scaled logslope try_row_chunk: {e}"
                    )
                })?;
            let logslope_row = logslope_chunk.row(0);
            for coeff_idx in 0..slices.logslope.len() {
                let value = logslope_weight * logslope_row[coeff_idx];
                joint_hessian[[slices.logslope.start + coeff_idx, joint_idx]] += value;
                joint_hessian[[joint_idx, slices.logslope.start + coeff_idx]] += value;
            }
        }
        Ok(())
    }

    fn add_dense_submatrix(
        &self,
        joint_hessian: &mut Array2<f64>,
        target_rows: &std::ops::Range<usize>,
        target_cols: &std::ops::Range<usize>,
        source: ArrayView2<'_, f64>,
    ) {
        for row_local in 0..target_rows.len() {
            for col_local in 0..target_cols.len() {
                joint_hessian[[target_rows.start + row_local, target_cols.start + col_local]] +=
                    source[[row_local, col_local]];
            }
        }
    }

    fn add_dense_symmetric_cross_submatrix(
        &self,
        joint_hessian: &mut Array2<f64>,
        left_range: &std::ops::Range<usize>,
        right_range: &std::ops::Range<usize>,
        source: ArrayView2<'_, f64>,
    ) {
        for left_local in 0..left_range.len() {
            for right_local in 0..right_range.len() {
                let value = source[[left_local, right_local]];
                joint_hessian[[
                    left_range.start + left_local,
                    right_range.start + right_local,
                ]] += value;
                joint_hessian[[
                    right_range.start + right_local,
                    left_range.start + left_local,
                ]] += value;
            }
        }
    }

    fn accumulate_dynamic_q_joint_row(
        &self,
        row: usize,
        slices: &BlockSlices,
        q_geom: &SurvivalMarginalSlopeDynamicRow,
        primary_gradient: ndarray::ArrayView1<'_, f64>,
        primary_hessian: ArrayView2<'_, f64>,
        identity_blocks: &[(std::ops::Range<usize>, std::ops::Range<usize>)],
        joint_gradient: &mut Array1<f64>,
        joint_hessian: &mut Array2<f64>,
    ) -> Result<(), String> {
        self.accumulate_dynamic_q_core_gradient(
            row,
            slices,
            q_geom,
            primary_gradient.slice(s![0..N_PRIMARY]),
            joint_gradient,
        )?;
        self.accumulate_dynamic_q_core_hessian(
            row,
            slices,
            q_geom,
            primary_gradient.slice(s![0..N_PRIMARY]),
            primary_hessian.slice(s![0..N_PRIMARY, 0..N_PRIMARY]),
            joint_hessian,
        )?;

        for (primary_range, joint_range) in identity_blocks {
            for local in 0..primary_range.len() {
                joint_gradient[joint_range.start + local] -=
                    primary_gradient[primary_range.start + local];
                self.accumulate_identity_primary_cross_hessian(
                    row,
                    slices,
                    q_geom,
                    primary_hessian.slice(s![0..N_PRIMARY, primary_range.start + local]),
                    joint_range,
                    local,
                    joint_hessian,
                )?;
            }
            self.add_dense_submatrix(
                joint_hessian,
                joint_range,
                joint_range,
                primary_hessian.slice(s![primary_range.clone(), primary_range.clone()]),
            );
        }

        for left_idx in 0..identity_blocks.len() {
            for right_idx in left_idx + 1..identity_blocks.len() {
                let (left_primary, left_joint) = &identity_blocks[left_idx];
                let (right_primary, right_joint) = &identity_blocks[right_idx];
                self.add_dense_symmetric_cross_submatrix(
                    joint_hessian,
                    left_joint,
                    right_joint,
                    primary_hessian.slice(s![left_primary.clone(), right_primary.clone()]),
                );
            }
        }

        // Absorbed Stage-1 influence block (#461). The absorber is a SINGLE
        // primary scalar `o_infl` at index `primary.infl` whose `p₁` joint
        // coefficients `γ` map to it through the residualized design row
        // `Z̃_infl[row,:]` (NOT an identity block — unlike score_warp/link_dev
        // whose bases are themselves primary coordinates). It therefore projects
        // like a non-identity single-scalar channel: each γ-coefficient `i` acts
        // as a copy of the `o_infl` primary direction scaled by `Z̃[row,i]`, so
        // its gradient and all cross-Hessians (vs core time/marginal/logslope and
        // vs the identity flex blocks) and its own diagonal are the `o_infl`
        // primary entries weighted by `Z̃[row,·]`.
        if let (Some(infl_primary), Some(infl_joint)) =
            (flex_primary_slices(self).infl, slices.influence.as_ref())
        {
            let z_tilde = self.influence_absorber.as_ref().ok_or_else(|| {
                "accumulate_dynamic_q_joint_row: influence primary index present but no Z̃ design"
                    .to_string()
            })?;
            let z_row = z_tilde.row(row);
            let core_col = primary_hessian.slice(s![0..N_PRIMARY, infl_primary]);
            // Per-coefficient gradient + cross with core blocks (time/marginal/
            // logslope), reusing the identity-channel core-cross helper with the
            // `o_infl` core Hessian column scaled by `Z̃[row, i]`.
            for i in 0..z_row.len() {
                let z_i = z_row[i];
                joint_gradient[infl_joint.start + i] -= primary_gradient[infl_primary] * z_i;
                if z_i != 0.0 {
                    // Perf (#large-scale): pass the unscaled `o_infl` core-Hessian
                    // column plus `z_i` to the pre-scaled cross-Hessian helper,
                    // which folds the scale in `(core[q] * z_i) * dq` form —
                    // bit-identical to the previous `&core_col.to_owned() * z_i`
                    // path but without the two per-(row,coeff) Array1 allocations.
                    self.accumulate_identity_primary_cross_hessian_scaled(
                        row,
                        slices,
                        q_geom,
                        core_col,
                        z_i,
                        infl_joint,
                        i,
                        joint_hessian,
                    )?;
                }
            }
            // Influence × influence diagonal: primary_hessian[[infl,infl]]·Z̃Z̃ᵀ.
            let ii_weight = primary_hessian[[infl_primary, infl_primary]];
            if ii_weight != 0.0 {
                for i in 0..z_row.len() {
                    for j in 0..z_row.len() {
                        joint_hessian[[infl_joint.start + i, infl_joint.start + j]] +=
                            ii_weight * z_row[i] * z_row[j];
                    }
                }
            }
            // Influence × identity-flex (score_warp/link_dev) cross-blocks: each
            // flex coefficient `f` (a primary coordinate) crossed with each
            // absorber coefficient `i` is `primary_hessian[[flex, infl]]·Z̃[row,i]`.
            for (flex_primary, flex_joint) in identity_blocks {
                for f in 0..flex_primary.len() {
                    let weight = primary_hessian[[flex_primary.start + f, infl_primary]];
                    if weight == 0.0 {
                        continue;
                    }
                    let fj = flex_joint.start + f;
                    for i in 0..z_row.len() {
                        let value = weight * z_row[i];
                        joint_hessian[[fj, infl_joint.start + i]] += value;
                        joint_hessian[[infl_joint.start + i, fj]] += value;
                    }
                }
            }
        }

        Ok(())
    }

    fn evaluate_blockwise_exact_newton_dynamic_q<RowTerms>(
        &self,
        block_states: &[ParameterBlockState],
        primary: &FlexPrimarySlices,
        row_terms: RowTerms,
    ) -> Result<FamilyEvaluation, String>
    where
        RowTerms: Fn(
                usize,
                &SurvivalMarginalSlopeDynamicRow,
            ) -> Result<(f64, Array1<f64>, Array2<f64>), String>
            + Sync,
    {
        let slices = block_slices(self, block_states);
        let make_acc = || DynamicQBlockwiseAccumulator::new(&slices);
        // See `evaluate_exact_newton_joint_dynamic_q_dense` for rationale.
        let make_acc_ws = || {
            (
                make_acc(),
                SurvivalMarginalSlopeDynamicRow::empty_workspace(),
            )
        };

        let acc = (0..self.n)
            .into_par_iter()
            .try_fold(make_acc_ws, |mut acc, row| -> Result<_, String> {
                let (state, q_geom) = &mut acc;
                self.row_dynamic_q_geometry_into(row, block_states, q_geom)?;
                let (row_nll, primary_gradient, primary_hessian) = row_terms(row, q_geom)?;
                state.log_likelihood -= row_nll;
                self.accumulate_dynamic_q_blockwise_row(
                    row,
                    q_geom,
                    primary,
                    primary_gradient.view(),
                    primary_hessian.view(),
                    state,
                )?;
                Ok(acc)
            })
            .try_reduce(make_acc_ws, |mut left, right| -> Result<_, String> {
                left.0.add_assign(&right.0);
                Ok(left)
            })?
            .0;

        Ok(acc.into_family_evaluation())
    }

    fn row_primary_third_contracted(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dir: ArrayView1<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        // Batched path delegating to the shared k=5 jet helper. The stack
        // tensor is then copied once into Array2 at the API boundary.
        let r = self.row_primary_third_contracted_batched(row, block_states, dir)?;
        let mut out = Array2::<f64>::zeros((N_PRIMARY, N_PRIMARY));
        for a in 0..N_PRIMARY {
            for b in 0..N_PRIMARY {
                out[[a, b]] = r[a][b];
            }
        }
        Ok(out)
    }

    /// Second-order cross-coefficient for parameter pair (u, v) from the
    /// b-family.  In survival, the only b-coupling is through g (the slope
    /// parameter), so this is nonzero only when u==g or v==g.
    fn cell_pair_second_coeff(
        &self,
        primary: &FlexPrimarySlices,
        coeff_bu: &[[f64; 4]],
        u: usize,
        v: usize,
    ) -> [f64; 4] {
        if u == primary.g {
            coeff_bu[v]
        } else if v == primary.g {
            coeff_bu[u]
        } else {
            [0.0; 4]
        }
    }

    /// Third-order a-cross-coefficient for parameter pair (u, v) from the
    /// ab-family. Nonzero only when u==g or v==g.
    fn cell_pair_third_coeff_a(
        &self,
        primary: &FlexPrimarySlices,
        coeff_abu: &[[f64; 4]],
        u: usize,
        v: usize,
    ) -> [f64; 4] {
        if u == primary.g {
            coeff_abu[v]
        } else if v == primary.g {
            coeff_abu[u]
        } else {
            [0.0; 4]
        }
    }

    /// Directional derivative of the fixed eta second partial r_uv w.r.t.
    /// a contraction direction.  Only (g,g), (g,h), (g,w) entries are nonzero.
    fn observed_fixed_eta_second_partial_dir(
        &self,
        primary: &FlexPrimarySlices,
        obs: &ObservedDenestedCellPartials,
        u: usize,
        v: usize,
        z_obs: f64,
        u_obs: f64,
        a: f64,
        b: f64,
        a_dir: f64,
        dir: &Array1<f64>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<f64, String> {
        let scale = self.probit_frailty_scale();
        if u == primary.g && v == primary.g {
            let dc_dbbb_val = {
                let zero_span = exact_kernel::LocalSpanCubic {
                    left: 0.0,
                    right: 1.0,
                    c0: 0.0,
                    c1: 0.0,
                    c2: 0.0,
                    c3: 0.0,
                };
                let link_span_obs = if let (Some(rt), Some(bw)) = (self.link_dev.as_ref(), beta_w) {
                    rt.local_cubic_at(bw, u_obs)?
                } else {
                    zero_span
                };
                let (_, _, _, dc_dbbb_raw) =
                    exact_kernel::denested_cell_third_partials(link_span_obs);
                eval_coeff4_at(&scale_coeff4(dc_dbbb_raw, scale), z_obs)
            };
            return Ok(eval_coeff4_at(&obs.dc_dabb, z_obs) * a_dir + dc_dbbb_val * dir[primary.g]);
        }
        if u == primary.g || v == primary.g {
            let other = if u == primary.g { v } else { u };
            if let Some(h_range) = primary.h.as_ref()
                && other >= h_range.start
                && other < h_range.end
            {
                return Ok(0.0);
            }
            if let Some(w_range) = primary.w.as_ref()
                && other >= w_range.start
                && other < w_range.end
            {
                let local_idx = other - w_range.start;
                let runtime = self
                    .link_dev
                    .as_ref()
                    .ok_or_else(|| "missing link runtime".to_string())?;
                let basis_span = runtime.basis_cubic_at(local_idx, u_obs)?;
                let (_, dc_abw, dc_bbw) =
                    exact_kernel::link_basis_cell_second_partials(basis_span, a, b);
                return Ok(eval_coeff4_at(&scale_coeff4(dc_abw, scale), z_obs) * a_dir
                    + eval_coeff4_at(&scale_coeff4(dc_bbw, scale), z_obs) * dir[primary.g]);
            }
        }
        Ok(0.0)
    }

    /// Directional derivative of the fixed chi second partial.
    fn observed_fixed_chi_second_partial_dir(
        &self,
        primary: &FlexPrimarySlices,
        u: usize,
        v: usize,
        z_obs: f64,
        u_obs: f64,
        a_dir: f64,
        dir: &Array1<f64>,
    ) -> Result<f64, String> {
        let scale = self.probit_frailty_scale();
        if u == primary.g && v == primary.g {
            return Ok(0.0);
        }
        if u == primary.g || v == primary.g {
            let other = if u == primary.g { v } else { u };
            if let Some(w_range) = primary.w.as_ref()
                && other >= w_range.start
                && other < w_range.end
            {
                let local_idx = other - w_range.start;
                let runtime = self
                    .link_dev
                    .as_ref()
                    .ok_or_else(|| "missing link runtime".to_string())?;
                let basis_span = runtime.basis_cubic_at(local_idx, u_obs)?;
                let (_, dc_aabw, dc_abbw, _) =
                    exact_kernel::link_basis_cell_third_partials(basis_span);
                return Ok(eval_coeff4_at(&scale_coeff4(dc_aabw, scale), z_obs) * a_dir
                    + eval_coeff4_at(&scale_coeff4(dc_abbw, scale), z_obs) * dir[primary.g]);
            }
        }
        Ok(0.0)
    }

    /// Compute directional extensions of a timepoint's exact quantities.
    /// Given the base `SurvivalFlexTimepointExact`, returns the directional
    /// derivatives eta_uv_dir, chi_uv_dir, d_u_dir, d_uv_dir contracted
    /// with `dir`.
    fn compute_survival_timepoint_directional_exact(
        &self,
        row: usize,
        primary: &FlexPrimarySlices,
        q: f64,
        q_index: usize,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        dir: &Array1<f64>,
        need_d_uv_dir: bool,
    ) -> Result<SurvivalFlexTimepointDirectionalExact, String> {
        let p = primary.total;
        let cached = self.build_cached_partition(primary, a, b, beta_h, beta_w)?;

        struct DirectionalTimepointCellAccum {
            f_a: f64,
            f_aa: f64,
            f_u: Vec<f64>,
            f_au: Vec<f64>,
            f_uv: Vec<f64>,
            f_a_dir: f64,
            f_aa_dir: f64,
            f_au_dir: Vec<f64>,
            f_uv_dir: Vec<f64>,
        }

        let cell_accums = cached
            .cells
            .iter()
            .map(
                |cell_entry| -> Result<DirectionalTimepointCellAccum, String> {
                    let neg_cell = cell_entry.neg_cell;
                    let state = &cell_entry.state;
                    let fixed = &cell_entry.fixed;
                    let neg_dc_da: [f64; 4] = fixed.dc_da.map(|v| -v);
                    let neg_dc_daa: [f64; 4] = fixed.dc_daa.map(|v| -v);

                    let f_a = exact_kernel::cell_first_derivative_from_moments(
                        &neg_dc_da,
                        &state.moments,
                    )?;
                    let f_aa = exact_kernel::cell_second_derivative_from_moments(
                        neg_cell,
                        &neg_dc_da,
                        &neg_dc_da,
                        &neg_dc_daa,
                        &state.moments,
                    )?;

                    let mut neg_coeff_dir = [0.0; 4];
                    let mut neg_coeff_a_dir = [0.0; 4];
                    let mut neg_coeff_aa_dir = [0.0; 4];
                    for c in 0..p {
                        if dir[c] == 0.0 {
                            continue;
                        }
                        for k in 0..4 {
                            neg_coeff_dir[k] -= fixed.coeff_u[c][k] * dir[c];
                            neg_coeff_a_dir[k] -= fixed.coeff_au[c][k] * dir[c];
                            neg_coeff_aa_dir[k] -= fixed.coeff_aau[c][k] * dir[c];
                        }
                    }

                    let f_a_dir = exact_kernel::cell_second_derivative_from_moments(
                        neg_cell,
                        &neg_dc_da,
                        &neg_coeff_dir,
                        &neg_coeff_a_dir,
                        &state.moments,
                    )?;
                    let f_aa_dir = exact_kernel::cell_third_derivative_from_moments(
                        neg_cell,
                        &neg_dc_da,
                        &neg_dc_da,
                        &neg_coeff_dir,
                        &neg_dc_daa,
                        &neg_coeff_a_dir,
                        &neg_coeff_a_dir,
                        &neg_coeff_aa_dir,
                        &state.moments,
                    )?;

                    let mut f_u = vec![0.0; p];
                    let mut f_au = vec![0.0; p];
                    let mut f_uv = vec![0.0; p * p];
                    let mut f_au_dir = vec![0.0; p];
                    let mut f_uv_dir = vec![0.0; p * p];
                    for u in 0..p {
                        let neg_coeff_u = fixed.coeff_u[u].map(|v| -v);
                        let neg_coeff_au = fixed.coeff_au[u].map(|v| -v);

                        f_u[u] = exact_kernel::cell_first_derivative_from_moments(
                            &neg_coeff_u,
                            &state.moments,
                        )?;
                        f_au[u] = exact_kernel::cell_second_derivative_from_moments(
                            neg_cell,
                            &neg_dc_da,
                            &neg_coeff_u,
                            &neg_coeff_au,
                            &state.moments,
                        )?;

                        let mut neg_coeff_u_dir = [0.0; 4];
                        let mut neg_coeff_au_dir = [0.0; 4];
                        for c in 0..p {
                            if dir[c] == 0.0 {
                                continue;
                            }
                            let sc = self.cell_pair_second_coeff(primary, &fixed.coeff_bu, u, c);
                            let sca = self.cell_pair_third_coeff_a(primary, &fixed.coeff_abu, u, c);
                            for k in 0..4 {
                                neg_coeff_u_dir[k] -= sc[k] * dir[c];
                                neg_coeff_au_dir[k] -= sca[k] * dir[c];
                            }
                        }

                        f_au_dir[u] = exact_kernel::cell_third_derivative_from_moments(
                            neg_cell,
                            &neg_dc_da,
                            &neg_coeff_u,
                            &neg_coeff_dir,
                            &neg_coeff_au,
                            &neg_coeff_a_dir,
                            &neg_coeff_u_dir,
                            &neg_coeff_au_dir,
                            &state.moments,
                        )?;
                    }

                    for u in 0..p {
                        for v in u..p {
                            let neg_coeff_u = fixed.coeff_u[u].map(|val| -val);
                            let neg_coeff_v = fixed.coeff_u[v].map(|val| -val);
                            let sc_uv = self.cell_pair_second_coeff(primary, &fixed.coeff_bu, u, v);
                            let neg_sc_uv = sc_uv.map(|val| -val);

                            let base_val = exact_kernel::cell_second_derivative_from_moments(
                                neg_cell,
                                &neg_coeff_u,
                                &neg_coeff_v,
                                &neg_sc_uv,
                                &state.moments,
                            )?;
                            f_uv[u * p + v] = base_val;
                            f_uv[v * p + u] = base_val;

                            let mut neg_coeff_u_dir = [0.0; 4];
                            let mut neg_coeff_v_dir = [0.0; 4];
                            for c in 0..p {
                                if dir[c] == 0.0 {
                                    continue;
                                }
                                let sc_uc =
                                    self.cell_pair_second_coeff(primary, &fixed.coeff_bu, u, c);
                                let sc_vc =
                                    self.cell_pair_second_coeff(primary, &fixed.coeff_bu, v, c);
                                for k in 0..4 {
                                    neg_coeff_u_dir[k] -= sc_uc[k] * dir[c];
                                    neg_coeff_v_dir[k] -= sc_vc[k] * dir[c];
                                }
                            }

                            let dir_val = exact_kernel::cell_third_derivative_from_moments(
                                neg_cell,
                                &neg_coeff_u,
                                &neg_coeff_v,
                                &neg_coeff_dir,
                                &neg_sc_uv,
                                &neg_coeff_u_dir,
                                &neg_coeff_v_dir,
                                &[0.0; 4], // third cross vanishes for cubic cells
                                &state.moments,
                            )?;
                            f_uv_dir[u * p + v] = dir_val;
                            f_uv_dir[v * p + u] = dir_val;
                        }
                    }

                    Ok(DirectionalTimepointCellAccum {
                        f_a,
                        f_aa,
                        f_u,
                        f_au,
                        f_uv,
                        f_a_dir,
                        f_aa_dir,
                        f_au_dir,
                        f_uv_dir,
                    })
                },
            )
            .collect::<Result<Vec<_>, String>>()?;

        let mut f_a = 0.0;
        let mut f_aa = 0.0;
        let mut f_u = Array1::<f64>::zeros(p);
        let mut f_au = Array1::<f64>::zeros(p);
        let mut f_uv = Array2::<f64>::zeros((p, p));
        let mut f_a_dir = 0.0;
        let mut f_aa_dir = 0.0;
        let mut f_au_dir = Array1::<f64>::zeros(p);
        let mut f_uv_dir = Array2::<f64>::zeros((p, p));
        for acc in cell_accums {
            f_a += acc.f_a;
            f_aa += acc.f_aa;
            f_a_dir += acc.f_a_dir;
            f_aa_dir += acc.f_aa_dir;
            for u in 0..p {
                f_u[u] += acc.f_u[u];
                f_au[u] += acc.f_au[u];
                f_au_dir[u] += acc.f_au_dir[u];
                for v in 0..p {
                    f_uv[[u, v]] += acc.f_uv[u * p + v];
                    f_uv_dir[[u, v]] += acc.f_uv_dir[u * p + v];
                }
            }
        }

        let phi_q = crate::probability::normal_pdf(q);
        f_u[q_index] += phi_q;
        f_uv[[q_index, q_index]] += -q * phi_q;
        f_uv_dir[[q_index, q_index]] += dir[q_index] * (1.0 - q * q) * phi_q;

        let inv_f_a = 1.0 / f_a;
        let mut a_u = Array1::<f64>::zeros(p);
        for u in 0..p {
            a_u[u] = -f_u[u] * inv_f_a;
        }
        let mut a_uv = Array2::<f64>::zeros((p, p));
        for u in 0..p {
            for v in u..p {
                let val =
                    -(f_uv[[u, v]] + f_au[u] * a_u[v] + f_au[v] * a_u[u] + f_aa * a_u[u] * a_u[v])
                        * inv_f_a;
                a_uv[[u, v]] = val;
                a_uv[[v, u]] = val;
            }
        }
        let a_dir = a_u.dot(dir);
        let a_u_dir = a_uv.dot(dir);
        let mut a_uv_dir = Array2::<f64>::zeros((p, p));
        for u in 0..p {
            for v in u..p {
                let n_dir = f_uv_dir[[u, v]]
                    + f_au_dir[u] * a_u[v]
                    + f_au[u] * a_u_dir[v]
                    + f_au_dir[v] * a_u[u]
                    + f_au[v] * a_u_dir[u]
                    + f_aa_dir * a_u[u] * a_u[v]
                    + f_aa * (a_u_dir[u] * a_u[v] + a_u[u] * a_u_dir[v]);
                let val = -(n_dir + f_a_dir * a_uv[[u, v]]) * inv_f_a;
                a_uv_dir[[u, v]] = val;
                a_uv_dir[[v, u]] = val;
            }
        }

        // Observed-point quantities and their dir-extensions
        let z_obs = self.observed_score_projection(row);
        let u_obs = a + b * z_obs;
        let obs = self.observed_denested_cell_partials(row, a, b, beta_h, beta_w)?;
        let chi_val = eval_coeff4_at(&obs.dc_da, z_obs);
        let eta_aa = eval_coeff4_at(&obs.dc_daa, z_obs);
        let eta_aaa = eval_coeff4_at(&obs.dc_daaa, z_obs);
        let scale = self.probit_frailty_scale();

        let mut g_u_fixed = vec![[0.0; 4]; p];
        let mut tau = Array1::<f64>::zeros(p);
        let mut g_au_fixed = vec![[0.0; 4]; p];
        let mut tau_a = Array1::<f64>::zeros(p);
        let mut g_bu_fixed = vec![[0.0; 4]; p];
        let mut g_aau_fixed = vec![[0.0; 4]; p];
        let mut g_abu_fixed = vec![[0.0; 4]; p];
        let mut g_bbu_fixed = vec![[0.0; 4]; p];
        let mut g_aaau_fixed = vec![[0.0; 4]; p];
        let mut g_aabu_fixed = vec![[0.0; 4]; p];
        let mut g_abbu_fixed = vec![[0.0; 4]; p];
        let mut g_bbbu_fixed = vec![[0.0; 4]; p];

        g_u_fixed[primary.g] = obs.dc_db;
        g_au_fixed[primary.g] = obs.dc_dab;
        g_bu_fixed[primary.g] = obs.dc_dbb;
        g_aau_fixed[primary.g] = obs.dc_daab;
        g_abu_fixed[primary.g] = obs.dc_dabb;
        g_bbu_fixed[primary.g] = obs.dc_dbbb;
        g_aaau_fixed[primary.g] = [0.0; 4];
        g_aabu_fixed[primary.g] = [0.0; 4];
        g_abbu_fixed[primary.g] = [0.0; 4];
        g_bbbu_fixed[primary.g] = [0.0; 4];

        tau[primary.g] = eval_coeff4_at(&obs.dc_dab, z_obs);
        tau_a[primary.g] = eval_coeff4_at(&obs.dc_daab, z_obs);
        if let (Some(w_range), Some(runtime)) = (primary.w.as_ref(), self.link_dev.as_ref()) {
            for local_idx in 0..w_range.len() {
                let basis_span = runtime.basis_cubic_at(local_idx, u_obs)?;
                let idx = w_range.start + local_idx;
                let (dc_aw, _) =
                    exact_kernel::link_basis_cell_coefficient_partials(basis_span, a, b);
                let (dc_aaw, dc_abw, dc_bbw) =
                    exact_kernel::link_basis_cell_second_partials(basis_span, a, b);
                let (dc_aaaw, dc_aabw, dc_abbw, dc_bbbw) =
                    exact_kernel::link_basis_cell_third_partials(basis_span);
                g_u_fixed[idx] = scale_coeff4(
                    exact_kernel::link_basis_cell_coefficients(basis_span, a, b),
                    scale,
                );
                g_au_fixed[idx] = scale_coeff4(dc_aw, scale);
                g_bu_fixed[idx] = scale_coeff4(
                    exact_kernel::link_basis_cell_coefficient_partials(basis_span, a, b).1,
                    scale,
                );
                g_aau_fixed[idx] = scale_coeff4(dc_aaw, scale);
                g_abu_fixed[idx] = scale_coeff4(dc_abw, scale);
                g_bbu_fixed[idx] = scale_coeff4(dc_bbw, scale);
                g_aaau_fixed[idx] = scale_coeff4(dc_aaaw, scale);
                g_aabu_fixed[idx] = scale_coeff4(dc_aabw, scale);
                g_abbu_fixed[idx] = scale_coeff4(dc_abbw, scale);
                g_bbbu_fixed[idx] = scale_coeff4(dc_bbbw, scale);
                tau[idx] = eval_coeff4_at(&scale_coeff4(dc_aw, scale), z_obs);
                tau_a[idx] = eval_coeff4_at(&scale_coeff4(dc_aaw, scale), z_obs);
            }
        }

        if let Some(h_range) = primary.h.as_ref().filter(|_| self.score_warp.is_some()) {
            for local_idx in 0..h_range.len() {
                let idx = h_range.start + local_idx;
                g_u_fixed[idx] = scale_coeff4(
                    self.observed_score_basis_coefficients(row, local_idx, z_obs, b)?,
                    scale,
                );
                g_bu_fixed[idx] = scale_coeff4(
                    self.observed_score_basis_coefficients(row, local_idx, z_obs, 1.0)?,
                    scale,
                );
            }
        }

        let g_jet = SparsePrimaryCoeffJetView::new(
            primary.g,
            primary.h.as_ref(),
            primary.w.as_ref(),
            &[],
            &g_au_fixed,
            &g_bu_fixed,
            &g_aau_fixed,
            &g_abu_fixed,
            &g_bbu_fixed,
            &g_aaau_fixed,
            &g_aabu_fixed,
            &g_abbu_fixed,
            &g_bbbu_fixed,
        );

        let chi_dir = eta_aa * a_dir + tau.dot(dir);
        let eta_aa_dir = eta_aaa * a_dir
            + eval_coeff4_at(
                &g_jet.directional_family(g_jet.aa_first, dir, COEFF_SUPPORT_GW),
                z_obs,
            );
        let eta_aaa_dir = eval_coeff4_at(
            &g_jet.directional_family(g_jet.aaa_first, dir, COEFF_SUPPORT_GW),
            z_obs,
        );

        let mut tau_dir = Array1::<f64>::zeros(p);
        let mut tau_a_dir = Array1::<f64>::zeros(p);
        for u in 0..p {
            let fixed_tau_dir =
                g_jet.param_directional_from_b_family(g_jet.ab_first, u, dir, COEFF_SUPPORT_GW);
            tau_dir[u] = eval_coeff4_at(&g_jet.aa_first[u], z_obs) * a_dir
                + eval_coeff4_at(&fixed_tau_dir, z_obs);

            let fixed_tau_a_dir =
                g_jet.param_directional_from_b_family(g_jet.aab_first, u, dir, COEFF_SUPPORT_GW);
            tau_a_dir[u] = eval_coeff4_at(&g_jet.aaa_first[u], z_obs) * a_dir
                + eval_coeff4_at(&fixed_tau_a_dir, z_obs);
        }

        let mut eta_uv_dir = Array2::<f64>::zeros((p, p));
        let mut chi_uv_dir = Array2::<f64>::zeros((p, p));
        for u in 0..p {
            for v in u..p {
                let r_uv_dir = self.observed_fixed_eta_second_partial_dir(
                    primary, &obs, u, v, z_obs, u_obs, a, b, a_dir, dir, beta_w,
                )?;
                let chi_uv_fixed_dir = self.observed_fixed_chi_second_partial_dir(
                    primary, u, v, z_obs, u_obs, a_dir, dir,
                )?;

                let eta_val = chi_dir * a_uv[[u, v]]
                    + chi_val * a_uv_dir[[u, v]]
                    + eta_aa_dir * a_u[u] * a_u[v]
                    + eta_aa * (a_u_dir[u] * a_u[v] + a_u[u] * a_u_dir[v])
                    + tau_dir[u] * a_u[v]
                    + tau[u] * a_u_dir[v]
                    + tau_dir[v] * a_u[u]
                    + tau[v] * a_u_dir[u]
                    + r_uv_dir;
                eta_uv_dir[[u, v]] = eta_val;
                eta_uv_dir[[v, u]] = eta_val;

                let chi_v = eta_aa_dir * a_uv[[u, v]]
                    + eta_aa * a_uv_dir[[u, v]]
                    + eta_aaa_dir * a_u[u] * a_u[v]
                    + eta_aaa * (a_u_dir[u] * a_u[v] + a_u[u] * a_u_dir[v])
                    + tau_a_dir[u] * a_u[v]
                    + tau_a[u] * a_u_dir[v]
                    + tau_a_dir[v] * a_u[u]
                    + tau_a[v] * a_u_dir[u]
                    + chi_uv_fixed_dir;
                chi_uv_dir[[u, v]] = chi_v;
                chi_uv_dir[[v, u]] = chi_v;
            }
        }

        // D_u_dir: directional derivative of the density normalization first derivative.
        let d_u_dir_cell_accums = cached
            .cells
            .iter()
            .map(|cell_entry| -> Result<Array1<f64>, String> {
                let mut d_u_dir = Array1::<f64>::zeros(p);
                let cell = cell_entry.partition_cell.cell;
                let state_ref = &cell_entry.state;
                let fixed = &cell_entry.fixed;
                let eta_poly = vec![cell.c0, cell.c1, cell.c2, cell.c3];
                let chi_poly = fixed.dc_da.to_vec();
                let eta_aa_poly = fixed.dc_daa.to_vec();

                let mut eta_u_poly = vec![PolyVec::new(); p];
                let mut chi_u_poly = vec![PolyVec::new(); p];
                for u in 0..p {
                    eta_u_poly[u] =
                        poly_add(&poly_scale(&chi_poly, a_u[u]), fixed.coeff_u[u].as_ref());
                    chi_u_poly[u] = poly_add(
                        &poly_scale(&eta_aa_poly, a_u[u]),
                        fixed.coeff_au[u].as_ref(),
                    );
                }

                let mut coeff_dir_poly = vec![0.0; 4];
                let mut coeff_a_dir_poly = vec![0.0; 4];
                for c in 0..p {
                    if dir[c] == 0.0 {
                        continue;
                    }
                    for k in 0..4 {
                        coeff_dir_poly[k] += fixed.coeff_u[c][k] * dir[c];
                        coeff_a_dir_poly[k] += fixed.coeff_au[c][k] * dir[c];
                    }
                }
                let eta_dir_poly = poly_add(&poly_scale(&chi_poly, a_dir), &coeff_dir_poly);

                for u in 0..p {
                    let mut eta_u_dir_fixed = vec![0.0; 4];
                    let mut chi_u_dir_fixed = vec![0.0; 4];
                    for c in 0..p {
                        if dir[c] == 0.0 {
                            continue;
                        }
                        let sc = self.cell_pair_second_coeff(primary, &fixed.coeff_bu, u, c);
                        let sca = self.cell_pair_third_coeff_a(primary, &fixed.coeff_abu, u, c);
                        for k in 0..4 {
                            eta_u_dir_fixed[k] += sc[k] * dir[c];
                            chi_u_dir_fixed[k] += sca[k] * dir[c];
                        }
                    }
                    let eta_u_dir_poly = poly_add(
                        &poly_add(
                            &poly_scale(&chi_poly, a_u_dir[u]),
                            &poly_scale(&eta_aa_poly, a_u[u] * a_dir),
                        ),
                        &eta_u_dir_fixed,
                    );
                    let eta_aaa_poly = fixed.dc_daaa.to_vec();
                    let chi_u_dir_poly = poly_add(
                        &poly_add(
                            &poly_scale(&eta_aa_poly, a_u_dir[u]),
                            &poly_scale(&eta_aaa_poly, a_u[u] * a_dir),
                        ),
                        &chi_u_dir_fixed,
                    );

                    // D_u integrand: chi_u - chi * eta * eta_u
                    let integrand_base = poly_sub(
                        &chi_u_poly[u],
                        &poly_mul(&poly_mul(&chi_poly, &eta_poly), &eta_u_poly[u]),
                    );
                    // Polynomial derivative of integrand w.r.t. dir
                    let integrand_dir = poly_sub(
                        &poly_sub(
                            &poly_sub(
                                &chi_u_dir_poly,
                                &poly_mul(&poly_mul(&coeff_a_dir_poly, &eta_poly), &eta_u_poly[u]),
                            ),
                            &poly_mul(&poly_mul(&chi_poly, &eta_dir_poly), &eta_u_poly[u]),
                        ),
                        &poly_mul(&poly_mul(&chi_poly, &eta_poly), &eta_u_dir_poly),
                    );
                    // Moment-weighting correction: -eta*eta_dir * integrand_base
                    let full_integrand = poly_sub(
                        &integrand_dir,
                        &poly_mul(&poly_mul(&eta_poly, &eta_dir_poly), &integrand_base),
                    );

                    d_u_dir[u] += exact_kernel::cell_polynomial_integral_from_moments(
                        &full_integrand,
                        &state_ref.moments,
                        "survival D_t first derivative directional",
                    )?;
                }
                Ok(d_u_dir)
            })
            .collect::<Result<Vec<_>, String>>()?;
        let mut d_u_dir = Array1::<f64>::zeros(p);
        for cell_d_u_dir in d_u_dir_cell_accums {
            for u in 0..p {
                d_u_dir[u] += cell_d_u_dir[u];
            }
        }

        // D_uv_dir
        let mut d_uv_dir = Array2::<f64>::zeros((p, p));
        if need_d_uv_dir {
            let d_uv_dir_cell_accums = cached
                .cells
                .iter()
                .map(|cell_entry| -> Result<Array2<f64>, String> {
                    let mut d_uv_dir = Array2::<f64>::zeros((p, p));
                    let cell = cell_entry.partition_cell.cell;
                    let state_ref = &cell_entry.state;
                    let fixed = &cell_entry.fixed;
                    let eta_poly = vec![cell.c0, cell.c1, cell.c2, cell.c3];
                    let chi_poly = fixed.dc_da.to_vec();
                    let eta_aa_poly = fixed.dc_daa.to_vec();
                    let eta_aaa_poly = fixed.dc_daaa.to_vec();

                    let mut eta_u_poly = vec![PolyVec::new(); p];
                    let mut chi_u_poly = vec![PolyVec::new(); p];
                    for u in 0..p {
                        eta_u_poly[u] =
                            poly_add(&poly_scale(&chi_poly, a_u[u]), fixed.coeff_u[u].as_ref());
                        chi_u_poly[u] = poly_add(
                            &poly_scale(&eta_aa_poly, a_u[u]),
                            fixed.coeff_au[u].as_ref(),
                        );
                    }
                    let mut coeff_dir_poly = vec![0.0; 4];
                    let mut coeff_a_dir_poly = vec![0.0; 4];
                    for c in 0..p {
                        if dir[c] == 0.0 {
                            continue;
                        }
                        for k in 0..4 {
                            coeff_dir_poly[k] += fixed.coeff_u[c][k] * dir[c];
                            coeff_a_dir_poly[k] += fixed.coeff_au[c][k] * dir[c];
                        }
                    }
                    let eta_dir_poly = poly_add(&poly_scale(&chi_poly, a_dir), &coeff_dir_poly);
                    let chi_dir_poly =
                        poly_add(&poly_scale(&eta_aa_poly, a_dir), &coeff_a_dir_poly);

                    for u in 0..p {
                        for v in u..p {
                            let r_uv_fixed = if u == primary.g {
                                fixed.coeff_bu[v].to_vec()
                            } else if v == primary.g {
                                fixed.coeff_bu[u].to_vec()
                            } else {
                                vec![0.0; 4]
                            };

                            let eta_uv_poly = poly_add(
                                &poly_add(
                                    &poly_add(
                                        &poly_scale(&chi_poly, a_uv[[u, v]]),
                                        &poly_scale(&eta_aa_poly, a_u[u] * a_u[v]),
                                    ),
                                    &poly_scale(fixed.coeff_au[u].as_ref(), a_u[v]),
                                ),
                                &poly_add(
                                    &poly_scale(fixed.coeff_au[v].as_ref(), a_u[u]),
                                    &r_uv_fixed,
                                ),
                            );

                            // D_uv integrand: 5 terms
                            let t1 = poly_add(
                                &poly_add(
                                    &poly_scale(&eta_aa_poly, a_uv[[u, v]]),
                                    &poly_scale(&eta_aaa_poly, a_u[u] * a_u[v]),
                                ),
                                &poly_add(
                                    &poly_scale(fixed.coeff_aau[u].as_ref(), a_u[v]),
                                    &poly_add(
                                        &poly_scale(fixed.coeff_aau[v].as_ref(), a_u[u]),
                                        &if u == primary.g {
                                            fixed.coeff_abu[v].to_vec()
                                        } else if v == primary.g {
                                            fixed.coeff_abu[u].to_vec()
                                        } else {
                                            vec![0.0; 4]
                                        },
                                    ),
                                ),
                            );
                            let t2 = poly_scale(
                                &poly_mul(&poly_mul(&chi_u_poly[v], &eta_poly), &eta_u_poly[u]),
                                -1.0,
                            );
                            let t3 = poly_scale(
                                &poly_mul(&poly_mul(&chi_u_poly[u], &eta_poly), &eta_u_poly[v]),
                                -1.0,
                            );
                            let t4 = poly_scale(
                                &poly_mul(
                                    &chi_poly,
                                    &poly_add(
                                        &poly_mul(&eta_u_poly[u], &eta_u_poly[v]),
                                        &poly_mul(&eta_poly, &eta_uv_poly),
                                    ),
                                ),
                                -1.0,
                            );
                            let t5 = poly_mul(
                                &chi_poly,
                                &poly_mul(
                                    &poly_mul(&eta_poly, &eta_poly),
                                    &poly_mul(&eta_u_poly[u], &eta_u_poly[v]),
                                ),
                            );
                            let i_base =
                                poly_add(&poly_add(&poly_add(&t1, &t2), &t3), &poly_add(&t4, &t5));

                            // Polynomial dir-derivatives of per-u quantities
                            let mut eu_dir_fixed_u = vec![0.0; 4];
                            let mut eu_dir_fixed_v = vec![0.0; 4];
                            let mut cu_dir_fixed_u = vec![0.0; 4];
                            let mut cu_dir_fixed_v = vec![0.0; 4];
                            for c in 0..p {
                                if dir[c] == 0.0 {
                                    continue;
                                }
                                let sc_u =
                                    self.cell_pair_second_coeff(primary, &fixed.coeff_bu, u, c);
                                let sc_v =
                                    self.cell_pair_second_coeff(primary, &fixed.coeff_bu, v, c);
                                let sca_u =
                                    self.cell_pair_third_coeff_a(primary, &fixed.coeff_abu, u, c);
                                let sca_v =
                                    self.cell_pair_third_coeff_a(primary, &fixed.coeff_abu, v, c);
                                for k in 0..4 {
                                    eu_dir_fixed_u[k] += sc_u[k] * dir[c];
                                    eu_dir_fixed_v[k] += sc_v[k] * dir[c];
                                    cu_dir_fixed_u[k] += sca_u[k] * dir[c];
                                    cu_dir_fixed_v[k] += sca_v[k] * dir[c];
                                }
                            }
                            let eta_u_dir_poly_u = poly_add(
                                &poly_add(
                                    &poly_scale(&chi_poly, a_u_dir[u]),
                                    &poly_scale(&eta_aa_poly, a_u[u] * a_dir),
                                ),
                                &eu_dir_fixed_u,
                            );
                            let eta_u_dir_poly_v = poly_add(
                                &poly_add(
                                    &poly_scale(&chi_poly, a_u_dir[v]),
                                    &poly_scale(&eta_aa_poly, a_u[v] * a_dir),
                                ),
                                &eu_dir_fixed_v,
                            );
                            let chi_u_dir_poly_u = poly_add(
                                &poly_add(
                                    &poly_scale(&eta_aa_poly, a_u_dir[u]),
                                    &poly_scale(&eta_aaa_poly, a_u[u] * a_dir),
                                ),
                                &cu_dir_fixed_u,
                            );
                            let chi_u_dir_poly_v = poly_add(
                                &poly_add(
                                    &poly_scale(&eta_aa_poly, a_u_dir[v]),
                                    &poly_scale(&eta_aaa_poly, a_u[v] * a_dir),
                                ),
                                &cu_dir_fixed_v,
                            );
                            let eta_uv_dir_poly = poly_add(
                                &poly_add(
                                    &poly_scale(&chi_poly, a_uv_dir[[u, v]]),
                                    &poly_scale(
                                        &eta_aa_poly,
                                        a_u_dir[u] * a_u[v]
                                            + a_u[u] * a_u_dir[v]
                                            + a_uv[[u, v]] * a_dir,
                                    ),
                                ),
                                &poly_add(
                                    &poly_add(
                                        &poly_scale(fixed.coeff_au[u].as_ref(), a_u_dir[v]),
                                        &poly_scale(fixed.coeff_au[v].as_ref(), a_u_dir[u]),
                                    ),
                                    &{
                                        let mut fp = vec![0.0; 4];
                                        for c in 0..p {
                                            if dir[c] == 0.0 {
                                                continue;
                                            }
                                            let sca_u = self.cell_pair_third_coeff_a(
                                                primary,
                                                &fixed.coeff_abu,
                                                u,
                                                c,
                                            );
                                            let sca_v = self.cell_pair_third_coeff_a(
                                                primary,
                                                &fixed.coeff_abu,
                                                v,
                                                c,
                                            );
                                            for k in 0..4 {
                                                fp[k] += sca_u[k] * dir[c] * a_u[v]
                                                    + sca_v[k] * dir[c] * a_u[u];
                                            }
                                        }
                                        fp
                                    },
                                ),
                            );

                            // Differentiate each of the 5 integrand terms
                            let t1_dir = poly_add(
                                &poly_add(
                                    &poly_scale(&eta_aa_poly, a_uv_dir[[u, v]]),
                                    &poly_scale(
                                        &eta_aaa_poly,
                                        a_u_dir[u] * a_u[v]
                                            + a_u[u] * a_u_dir[v]
                                            + a_uv[[u, v]] * a_dir,
                                    ),
                                ),
                                &poly_add(
                                    &poly_scale(fixed.coeff_aau[u].as_ref(), a_u_dir[v]),
                                    &poly_scale(fixed.coeff_aau[v].as_ref(), a_u_dir[u]),
                                ),
                            );
                            let t2_dir = poly_scale(
                                &poly_add(
                                    &poly_add(
                                        &poly_mul(
                                            &poly_mul(&chi_u_dir_poly_v, &eta_poly),
                                            &eta_u_poly[u],
                                        ),
                                        &poly_mul(
                                            &poly_mul(&chi_u_poly[v], &eta_dir_poly),
                                            &eta_u_poly[u],
                                        ),
                                    ),
                                    &poly_mul(
                                        &poly_mul(&chi_u_poly[v], &eta_poly),
                                        &eta_u_dir_poly_u,
                                    ),
                                ),
                                -1.0,
                            );
                            let t3_dir = poly_scale(
                                &poly_add(
                                    &poly_add(
                                        &poly_mul(
                                            &poly_mul(&chi_u_dir_poly_u, &eta_poly),
                                            &eta_u_poly[v],
                                        ),
                                        &poly_mul(
                                            &poly_mul(&chi_u_poly[u], &eta_dir_poly),
                                            &eta_u_poly[v],
                                        ),
                                    ),
                                    &poly_mul(
                                        &poly_mul(&chi_u_poly[u], &eta_poly),
                                        &eta_u_dir_poly_v,
                                    ),
                                ),
                                -1.0,
                            );
                            let t4_dir = poly_scale(
                                &poly_add(
                                    &poly_mul(
                                        &chi_dir_poly,
                                        &poly_add(
                                            &poly_mul(&eta_u_poly[u], &eta_u_poly[v]),
                                            &poly_mul(&eta_poly, &eta_uv_poly),
                                        ),
                                    ),
                                    &poly_mul(
                                        &chi_poly,
                                        &poly_add(
                                            &poly_add(
                                                &poly_mul(&eta_u_dir_poly_u, &eta_u_poly[v]),
                                                &poly_mul(&eta_u_poly[u], &eta_u_dir_poly_v),
                                            ),
                                            &poly_add(
                                                &poly_mul(&eta_dir_poly, &eta_uv_poly),
                                                &poly_mul(&eta_poly, &eta_uv_dir_poly),
                                            ),
                                        ),
                                    ),
                                ),
                                -1.0,
                            );
                            let t5_dir = poly_add(
                                &poly_mul(
                                    &chi_dir_poly,
                                    &poly_mul(
                                        &poly_mul(&eta_poly, &eta_poly),
                                        &poly_mul(&eta_u_poly[u], &eta_u_poly[v]),
                                    ),
                                ),
                                &poly_mul(
                                    &chi_poly,
                                    &poly_add(
                                        &poly_mul(
                                            &poly_scale(&poly_mul(&eta_dir_poly, &eta_poly), 2.0),
                                            &poly_mul(&eta_u_poly[u], &eta_u_poly[v]),
                                        ),
                                        &poly_mul(
                                            &poly_mul(&eta_poly, &eta_poly),
                                            &poly_add(
                                                &poly_mul(&eta_u_dir_poly_u, &eta_u_poly[v]),
                                                &poly_mul(&eta_u_poly[u], &eta_u_dir_poly_v),
                                            ),
                                        ),
                                    ),
                                ),
                            );

                            let i_base_dir = poly_add(
                                &poly_add(&poly_add(&t1_dir, &t2_dir), &t3_dir),
                                &poly_add(&t4_dir, &t5_dir),
                            );
                            let full_integrand = poly_sub(
                                &i_base_dir,
                                &poly_mul(&poly_mul(&eta_poly, &eta_dir_poly), &i_base),
                            );

                            let value = exact_kernel::cell_polynomial_integral_from_moments(
                                &full_integrand,
                                &state_ref.moments,
                                "survival D_t second derivative directional",
                            )?;
                            d_uv_dir[[u, v]] += value;
                            d_uv_dir[[v, u]] = d_uv_dir[[u, v]];
                        }
                    }
                    Ok(d_uv_dir)
                })
                .collect::<Result<Vec<_>, String>>()?;
            for cell_d_uv_dir in d_uv_dir_cell_accums {
                for u in 0..p {
                    for v in 0..p {
                        d_uv_dir[[u, v]] += cell_d_uv_dir[[u, v]];
                    }
                }
            }
        }

        Ok(SurvivalFlexTimepointDirectionalExact {
            eta_uv_dir,
            chi_uv_dir,
            d_u_dir,
            d_uv_dir,
        })
    }

    /// Exact mixed bidirectional extension D_{d1} D_{d2} of the timepoint
    /// quantities. This carries the calibration solve, observed eta/chi
    /// transport, and density-normalization transport analytically.
    fn compute_survival_timepoint_bidirectional_exact(
        &self,
        row: usize,
        primary: &FlexPrimarySlices,
        q: f64,
        q_index: usize,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        dir1: &Array1<f64>,
        dir2: &Array1<f64>,
    ) -> Result<SurvivalFlexTimepointBiDirectionalExact, String> {
        self.compute_survival_timepoint_bidirectional_exact_full(
            row, primary, q, q_index, a, b, beta_h, beta_w, dir1, dir2,
        )
    }

    fn compute_survival_timepoint_bidirectional_exact_full(
        &self,
        row: usize,
        primary: &FlexPrimarySlices,
        q: f64,
        q_index: usize,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        dir1: &Array1<f64>,
        dir2: &Array1<f64>,
    ) -> Result<SurvivalFlexTimepointBiDirectionalExact, String> {
        let p = primary.total;
        let zero4 = [0.0; 4];
        let cached = self.build_cached_partition(primary, a, b, beta_h, beta_w)?;

        struct BiDirectionalTimepointCellAccum {
            f_a: f64,
            f_aa: f64,
            f_u: Array1<f64>,
            f_au: Array1<f64>,
            f_uv: Array2<f64>,
            f_a_d1: f64,
            f_aa_d1: f64,
            f_au_d1: Array1<f64>,
            f_uv_d1: Array2<f64>,
            f_a_d2: f64,
            f_aa_d2: f64,
            f_au_d2: Array1<f64>,
            f_uv_d2: Array2<f64>,
            f_a_d12: f64,
            f_aa_d12: f64,
            f_au_d12: Array1<f64>,
            f_uv_d12: Array2<f64>,
        }

        let cell_accums = cached
            .cells
            .iter()
            .map(|ce| -> Result<BiDirectionalTimepointCellAccum, String> {
                let mut f_a = 0.0f64;
                let mut f_aa = 0.0f64;
                let mut f_u = Array1::<f64>::zeros(p);
                let mut f_au = Array1::<f64>::zeros(p);
                let mut f_uv = Array2::<f64>::zeros((p, p));
                let mut f_a_d1 = 0.0f64;
                let mut f_aa_d1 = 0.0f64;
                let mut f_au_d1 = Array1::<f64>::zeros(p);
                let mut f_uv_d1 = Array2::<f64>::zeros((p, p));
                let mut f_a_d2 = 0.0f64;
                let mut f_aa_d2 = 0.0f64;
                let mut f_au_d2 = Array1::<f64>::zeros(p);
                let mut f_uv_d2 = Array2::<f64>::zeros((p, p));
                let mut f_a_d12 = 0.0f64;
                let mut f_aa_d12 = 0.0f64;
                let mut f_au_d12 = Array1::<f64>::zeros(p);
                let mut f_uv_d12 = Array2::<f64>::zeros((p, p));
                let nc = ce.neg_cell;
                let st = &ce.state;
                let fx = &ce.fixed;
                let da = fx.dc_da.map(|v| -v);
                let daa = fx.dc_daa.map(|v| -v);

                f_a += exact_kernel::cell_first_derivative_from_moments(&da, &st.moments)?;
                f_aa += exact_kernel::cell_second_derivative_from_moments(
                    nc,
                    &da,
                    &da,
                    &daa,
                    &st.moments,
                )?;

                let mut cd1 = [0.0; 4];
                let mut ca1 = [0.0; 4];
                let mut caa1 = [0.0; 4];
                let mut cd2 = [0.0; 4];
                let mut ca2 = [0.0; 4];
                let mut caa2 = [0.0; 4];
                let mut cd12 = [0.0; 4];
                let mut ca12 = [0.0; 4];
                for c in 0..p {
                    for k in 0..4 {
                        if dir1[c] != 0.0 {
                            cd1[k] -= fx.coeff_u[c][k] * dir1[c];
                            ca1[k] -= fx.coeff_au[c][k] * dir1[c];
                            caa1[k] -= fx.coeff_aau[c][k] * dir1[c];
                        }
                        if dir2[c] != 0.0 {
                            cd2[k] -= fx.coeff_u[c][k] * dir2[c];
                            ca2[k] -= fx.coeff_au[c][k] * dir2[c];
                            caa2[k] -= fx.coeff_aau[c][k] * dir2[c];
                        }
                    }
                }
                for c1 in 0..p {
                    if dir1[c1] == 0.0 {
                        continue;
                    }
                    for c2 in 0..p {
                        if dir2[c2] == 0.0 {
                            continue;
                        }
                        let sc = self.cell_pair_second_coeff(primary, &fx.coeff_bu, c1, c2);
                        let sca = self.cell_pair_third_coeff_a(primary, &fx.coeff_abu, c1, c2);
                        for k in 0..4 {
                            cd12[k] -= sc[k] * dir1[c1] * dir2[c2];
                            ca12[k] -= sca[k] * dir1[c1] * dir2[c2];
                        }
                    }
                }

                f_a_d1 += exact_kernel::cell_second_derivative_from_moments(
                    nc,
                    &da,
                    &cd1,
                    &ca1,
                    &st.moments,
                )?;
                f_a_d2 += exact_kernel::cell_second_derivative_from_moments(
                    nc,
                    &da,
                    &cd2,
                    &ca2,
                    &st.moments,
                )?;
                f_a_d12 += exact_kernel::cell_third_derivative_from_moments(
                    nc,
                    &da,
                    &cd1,
                    &cd2,
                    &ca1,
                    &ca2,
                    &cd12,
                    &ca12,
                    &st.moments,
                )?;
                f_aa_d1 += exact_kernel::cell_third_derivative_from_moments(
                    nc,
                    &da,
                    &da,
                    &cd1,
                    &daa,
                    &ca1,
                    &ca1,
                    &caa1,
                    &st.moments,
                )?;
                f_aa_d2 += exact_kernel::cell_third_derivative_from_moments(
                    nc,
                    &da,
                    &da,
                    &cd2,
                    &daa,
                    &ca2,
                    &ca2,
                    &caa2,
                    &st.moments,
                )?;
                f_aa_d12 += exact_kernel::cell_fourth_derivative_from_moments(
                    nc,
                    &da,
                    &da,
                    &cd1,
                    &cd2,
                    &daa,
                    &ca1,
                    &ca2,
                    &ca1,
                    &ca2,
                    &cd12,
                    &caa1,
                    &caa2,
                    &ca12,
                    &ca12,
                    &[0.0; 4],
                    &st.moments,
                )?;

                for u in 0..p {
                    let cu = fx.coeff_u[u].map(|v| -v);
                    let cau = fx.coeff_au[u].map(|v| -v);
                    f_u[u] += exact_kernel::cell_first_derivative_from_moments(&cu, &st.moments)?;
                    f_au[u] += exact_kernel::cell_second_derivative_from_moments(
                        nc,
                        &da,
                        &cu,
                        &cau,
                        &st.moments,
                    )?;
                    let mut cu1 = [0.0; 4];
                    let mut cau1 = [0.0; 4];
                    let mut cu2 = [0.0; 4];
                    let mut cau2 = [0.0; 4];
                    for c in 0..p {
                        let sc = self.cell_pair_second_coeff(primary, &fx.coeff_bu, u, c);
                        let sca = self.cell_pair_third_coeff_a(primary, &fx.coeff_abu, u, c);
                        for k in 0..4 {
                            if dir1[c] != 0.0 {
                                cu1[k] -= sc[k] * dir1[c];
                                cau1[k] -= sca[k] * dir1[c];
                            }
                            if dir2[c] != 0.0 {
                                cu2[k] -= sc[k] * dir2[c];
                                cau2[k] -= sca[k] * dir2[c];
                            }
                        }
                    }
                    f_au_d1[u] += exact_kernel::cell_third_derivative_from_moments(
                        nc,
                        &da,
                        &cu,
                        &cd1,
                        &cau,
                        &ca1,
                        &cu1,
                        &cau1,
                        &st.moments,
                    )?;
                    f_au_d2[u] += exact_kernel::cell_third_derivative_from_moments(
                        nc,
                        &da,
                        &cu,
                        &cd2,
                        &cau,
                        &ca2,
                        &cu2,
                        &cau2,
                        &st.moments,
                    )?;
                    f_au_d12[u] += exact_kernel::cell_fourth_derivative_from_moments(
                        nc,
                        &da,
                        &cu,
                        &cd1,
                        &cd2,
                        &cau,
                        &ca1,
                        &ca2,
                        &cu1,
                        &cu2,
                        &cd12,
                        &cau1,
                        &cau2,
                        &ca12,
                        &[0.0; 4],
                        &[0.0; 4],
                        &st.moments,
                    )?;
                }
                for u in 0..p {
                    for v in u..p {
                        let cu = fx.coeff_u[u].map(|x| -x);
                        let cv = fx.coeff_u[v].map(|x| -x);
                        let sc = self
                            .cell_pair_second_coeff(primary, &fx.coeff_bu, u, v)
                            .map(|x| -x);
                        let bv = exact_kernel::cell_second_derivative_from_moments(
                            nc,
                            &cu,
                            &cv,
                            &sc,
                            &st.moments,
                        )?;
                        f_uv[[u, v]] += bv;
                        if u != v {
                            f_uv[[v, u]] += bv;
                        }
                        let mut cu1 = [0.0; 4];
                        let mut cv1 = [0.0; 4];
                        let mut cu2 = [0.0; 4];
                        let mut cv2 = [0.0; 4];
                        for c in 0..p {
                            let suc = self.cell_pair_second_coeff(primary, &fx.coeff_bu, u, c);
                            let svc = self.cell_pair_second_coeff(primary, &fx.coeff_bu, v, c);
                            for k in 0..4 {
                                if dir1[c] != 0.0 {
                                    cu1[k] -= suc[k] * dir1[c];
                                    cv1[k] -= svc[k] * dir1[c];
                                }
                                if dir2[c] != 0.0 {
                                    cu2[k] -= suc[k] * dir2[c];
                                    cv2[k] -= svc[k] * dir2[c];
                                }
                            }
                        }
                        let d1v = exact_kernel::cell_third_derivative_from_moments(
                            nc,
                            &cu,
                            &cv,
                            &cd1,
                            &sc,
                            &cu1,
                            &cv1,
                            &[0.0; 4],
                            &st.moments,
                        )?;
                        f_uv_d1[[u, v]] += d1v;
                        if u != v {
                            f_uv_d1[[v, u]] += d1v;
                        }
                        let d2v = exact_kernel::cell_third_derivative_from_moments(
                            nc,
                            &cu,
                            &cv,
                            &cd2,
                            &sc,
                            &cu2,
                            &cv2,
                            &[0.0; 4],
                            &st.moments,
                        )?;
                        f_uv_d2[[u, v]] += d2v;
                        if u != v {
                            f_uv_d2[[v, u]] += d2v;
                        }
                        let d12v = exact_kernel::cell_fourth_derivative_from_moments(
                            nc,
                            &cu,
                            &cv,
                            &cd1,
                            &cd2,
                            &sc,
                            &cu1,
                            &cu2,
                            &cv1,
                            &cv2,
                            &cd12,
                            &[0.0; 4],
                            &[0.0; 4],
                            &[0.0; 4],
                            &[0.0; 4],
                            &[0.0; 4],
                            &st.moments,
                        )?;
                        f_uv_d12[[u, v]] += d12v;
                        if u != v {
                            f_uv_d12[[v, u]] += d12v;
                        }
                    }
                }

                Ok(BiDirectionalTimepointCellAccum {
                    f_a,
                    f_aa,
                    f_u,
                    f_au,
                    f_uv,
                    f_a_d1,
                    f_aa_d1,
                    f_au_d1,
                    f_uv_d1,
                    f_a_d2,
                    f_aa_d2,
                    f_au_d2,
                    f_uv_d2,
                    f_a_d12,
                    f_aa_d12,
                    f_au_d12,
                    f_uv_d12,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;

        let mut f_a = 0.0f64;
        let mut f_aa = 0.0f64;
        let mut f_u = Array1::<f64>::zeros(p);
        let mut f_au = Array1::<f64>::zeros(p);
        let mut f_uv = Array2::<f64>::zeros((p, p));
        let mut f_a_d1 = 0.0f64;
        let mut f_aa_d1 = 0.0f64;
        let mut f_au_d1 = Array1::<f64>::zeros(p);
        let mut f_uv_d1 = Array2::<f64>::zeros((p, p));
        let mut f_a_d2 = 0.0f64;
        let mut f_aa_d2 = 0.0f64;
        let mut f_au_d2 = Array1::<f64>::zeros(p);
        let mut f_uv_d2 = Array2::<f64>::zeros((p, p));
        let mut f_a_d12 = 0.0f64;
        let mut f_aa_d12 = 0.0f64;
        let mut f_au_d12 = Array1::<f64>::zeros(p);
        let mut f_uv_d12 = Array2::<f64>::zeros((p, p));

        for acc in cell_accums {
            f_a += acc.f_a;
            f_aa += acc.f_aa;
            f_a_d1 += acc.f_a_d1;
            f_aa_d1 += acc.f_aa_d1;
            f_a_d2 += acc.f_a_d2;
            f_aa_d2 += acc.f_aa_d2;
            f_a_d12 += acc.f_a_d12;
            f_aa_d12 += acc.f_aa_d12;
            for u in 0..p {
                f_u[u] += acc.f_u[u];
                f_au[u] += acc.f_au[u];
                f_au_d1[u] += acc.f_au_d1[u];
                f_au_d2[u] += acc.f_au_d2[u];
                f_au_d12[u] += acc.f_au_d12[u];
                for v in 0..p {
                    f_uv[[u, v]] += acc.f_uv[[u, v]];
                    f_uv_d1[[u, v]] += acc.f_uv_d1[[u, v]];
                    f_uv_d2[[u, v]] += acc.f_uv_d2[[u, v]];
                    f_uv_d12[[u, v]] += acc.f_uv_d12[[u, v]];
                }
            }
        }

        let phi_q = crate::probability::normal_pdf(q);
        f_u[q_index] += phi_q;
        f_uv[[q_index, q_index]] += -q * phi_q;
        f_uv_d1[[q_index, q_index]] += dir1[q_index] * (1.0 - q * q) * phi_q;
        f_uv_d2[[q_index, q_index]] += dir2[q_index] * (1.0 - q * q) * phi_q;
        f_uv_d12[[q_index, q_index]] += dir1[q_index] * dir2[q_index] * q * (q * q - 3.0) * phi_q;

        let inv = 1.0 / f_a;
        let mut au = Array1::<f64>::zeros(p);
        for u in 0..p {
            au[u] = -f_u[u] * inv;
        }
        let mut auv = Array2::<f64>::zeros((p, p));
        for u in 0..p {
            for v in u..p {
                let val =
                    -(f_uv[[u, v]] + f_au[u] * au[v] + f_au[v] * au[u] + f_aa * au[u] * au[v])
                        * inv;
                auv[[u, v]] = val;
                auv[[v, u]] = val;
            }
        }
        let ad1 = au.dot(dir1);
        let ad2 = au.dot(dir2);
        let aud1 = auv.dot(dir1);
        let aud2 = auv.dot(dir2);

        let mut auvd1 = Array2::<f64>::zeros((p, p));
        let mut auvd2 = Array2::<f64>::zeros((p, p));
        for u in 0..p {
            for v in u..p {
                let n1 = f_uv_d1[[u, v]]
                    + f_au_d1[u] * au[v]
                    + f_au[u] * aud1[v]
                    + f_au_d1[v] * au[u]
                    + f_au[v] * aud1[u]
                    + f_aa_d1 * au[u] * au[v]
                    + f_aa * (aud1[u] * au[v] + au[u] * aud1[v]);
                let v1 = -(n1 + f_a_d1 * auv[[u, v]]) * inv;
                auvd1[[u, v]] = v1;
                auvd1[[v, u]] = v1;

                let n2 = f_uv_d2[[u, v]]
                    + f_au_d2[u] * au[v]
                    + f_au[u] * aud2[v]
                    + f_au_d2[v] * au[u]
                    + f_au[v] * aud2[u]
                    + f_aa_d2 * au[u] * au[v]
                    + f_aa * (aud2[u] * au[v] + au[u] * aud2[v]);
                let v2 = -(n2 + f_a_d2 * auv[[u, v]]) * inv;
                auvd2[[u, v]] = v2;
                auvd2[[v, u]] = v2;
            }
        }

        let ad12 = aud2.dot(dir1);
        let aud12 = auvd2.dot(dir1);
        let mut auvd12 = Array2::<f64>::zeros((p, p));
        for u in 0..p {
            for v in u..p {
                let n = f_uv_d12[[u, v]]
                    + f_au_d12[u] * au[v]
                    + f_au_d1[u] * aud2[v]
                    + f_au_d2[u] * aud1[v]
                    + f_au[u] * aud12[v]
                    + f_au_d12[v] * au[u]
                    + f_au_d1[v] * aud2[u]
                    + f_au_d2[v] * aud1[u]
                    + f_au[v] * aud12[u]
                    + f_aa_d12 * au[u] * au[v]
                    + f_aa_d1 * (aud2[u] * au[v] + au[u] * aud2[v])
                    + f_aa_d2 * (aud1[u] * au[v] + au[u] * aud1[v])
                    + f_aa
                        * (aud12[u] * au[v]
                            + aud1[u] * aud2[v]
                            + aud2[u] * aud1[v]
                            + au[u] * aud12[v]);
                let val =
                    -(n + f_a_d12 * auv[[u, v]] + f_a_d1 * auvd2[[u, v]] + f_a_d2 * auvd1[[u, v]])
                        * inv;
                auvd12[[u, v]] = val;
                auvd12[[v, u]] = val;
            }
        }

        let obs = self.observed_denested_cell_partials(row, a, b, beta_h, beta_w)?;
        let z_obs = self.observed_score_projection(row);
        let u_obs = a + b * z_obs;
        let scale = self.probit_frailty_scale();

        let mut g_u_fixed = vec![[0.0; 4]; p];
        let mut g_au_fixed = vec![[0.0; 4]; p];
        let mut g_bu_fixed = vec![[0.0; 4]; p];
        let mut g_aau_fixed = vec![[0.0; 4]; p];
        let mut g_abu_fixed = vec![[0.0; 4]; p];
        let mut g_bbu_fixed = vec![[0.0; 4]; p];
        let mut g_aaau_fixed = vec![[0.0; 4]; p];
        let mut g_aabu_fixed = vec![[0.0; 4]; p];
        let mut g_abbu_fixed = vec![[0.0; 4]; p];
        let mut g_bbbu_fixed = vec![[0.0; 4]; p];

        g_u_fixed[primary.g] = obs.dc_db;
        g_au_fixed[primary.g] = obs.dc_dab;
        g_bu_fixed[primary.g] = obs.dc_dbb;
        g_aau_fixed[primary.g] = obs.dc_daab;
        g_abu_fixed[primary.g] = obs.dc_dabb;
        g_bbu_fixed[primary.g] = obs.dc_dbbb;
        g_aaau_fixed[primary.g] = [0.0; 4];
        g_aabu_fixed[primary.g] = [0.0; 4];
        g_abbu_fixed[primary.g] = [0.0; 4];
        g_bbbu_fixed[primary.g] = [0.0; 4];

        if let Some(h_range) = primary.h.as_ref().filter(|_| self.score_warp.is_some()) {
            for local_idx in 0..h_range.len() {
                let idx = h_range.start + local_idx;
                g_u_fixed[idx] = scale_coeff4(
                    self.observed_score_basis_coefficients(row, local_idx, z_obs, b)?,
                    scale,
                );
                g_bu_fixed[idx] = scale_coeff4(
                    self.observed_score_basis_coefficients(row, local_idx, z_obs, 1.0)?,
                    scale,
                );
            }
        }
        if let (Some(w_range), Some(runtime)) = (primary.w.as_ref(), self.link_dev.as_ref()) {
            for local_idx in 0..w_range.len() {
                let basis_span = runtime.basis_cubic_at(local_idx, u_obs)?;
                let idx = w_range.start + local_idx;
                let (dc_aw, dc_bw) =
                    exact_kernel::link_basis_cell_coefficient_partials(basis_span, a, b);
                let (dc_aaw, dc_abw, dc_bbw) =
                    exact_kernel::link_basis_cell_second_partials(basis_span, a, b);
                let (dc_aaaw, dc_aabw, dc_abbw, dc_bbbw) =
                    exact_kernel::link_basis_cell_third_partials(basis_span);
                g_u_fixed[idx] = scale_coeff4(
                    exact_kernel::link_basis_cell_coefficients(basis_span, a, b),
                    scale,
                );
                g_au_fixed[idx] = scale_coeff4(dc_aw, scale);
                g_bu_fixed[idx] = scale_coeff4(dc_bw, scale);
                g_aau_fixed[idx] = scale_coeff4(dc_aaw, scale);
                g_abu_fixed[idx] = scale_coeff4(dc_abw, scale);
                g_bbu_fixed[idx] = scale_coeff4(dc_bbw, scale);
                g_aaau_fixed[idx] = scale_coeff4(dc_aaaw, scale);
                g_aabu_fixed[idx] = scale_coeff4(dc_aabw, scale);
                g_abbu_fixed[idx] = scale_coeff4(dc_abbw, scale);
                g_bbbu_fixed[idx] = scale_coeff4(dc_bbbw, scale);
            }
        }

        let g_jet = SparsePrimaryCoeffJetView::new(
            primary.g,
            primary.h.as_ref(),
            primary.w.as_ref(),
            &[],
            &g_au_fixed,
            &g_bu_fixed,
            &g_aau_fixed,
            &g_abu_fixed,
            &g_bbu_fixed,
            &g_aaau_fixed,
            &g_aabu_fixed,
            &g_abbu_fixed,
            &g_bbbu_fixed,
        );

        let chi = eval_coeff4_at(&obs.dc_da, z_obs);
        let eta_aa = eval_coeff4_at(&obs.dc_daa, z_obs);
        let eta_aaa = eval_coeff4_at(&obs.dc_daaa, z_obs);
        let chi_jet = scalar_composite_bilinear(
            chi,
            eta_aa,
            eta_aaa,
            eval_coeff4_at(
                &g_jet.directional_family(g_jet.a_first, dir1, COEFF_SUPPORT_GW),
                z_obs,
            ),
            eval_coeff4_at(
                &g_jet.directional_family(g_jet.a_first, dir2, COEFF_SUPPORT_GW),
                z_obs,
            ),
            eval_coeff4_at(
                &g_jet.mixed_directional_from_b_family(
                    g_jet.ab_first,
                    dir1,
                    dir2,
                    COEFF_SUPPORT_GW,
                ),
                z_obs,
            ),
            eval_coeff4_at(
                &g_jet.directional_family(g_jet.aa_first, dir1, COEFF_SUPPORT_GW),
                z_obs,
            ),
            eval_coeff4_at(
                &g_jet.directional_family(g_jet.aa_first, dir2, COEFF_SUPPORT_GW),
                z_obs,
            ),
            ad1,
            ad2,
            ad12,
        );
        let eta_aa_jet = scalar_composite_bilinear(
            eta_aa,
            eta_aaa,
            0.0,
            eval_coeff4_at(
                &g_jet.directional_family(g_jet.aa_first, dir1, COEFF_SUPPORT_GW),
                z_obs,
            ),
            eval_coeff4_at(
                &g_jet.directional_family(g_jet.aa_first, dir2, COEFF_SUPPORT_GW),
                z_obs,
            ),
            eval_coeff4_at(
                &g_jet.mixed_directional_from_b_family(
                    g_jet.aab_first,
                    dir1,
                    dir2,
                    COEFF_SUPPORT_GW,
                ),
                z_obs,
            ),
            eval_coeff4_at(
                &g_jet.directional_family(g_jet.aaa_first, dir1, COEFF_SUPPORT_GW),
                z_obs,
            ),
            eval_coeff4_at(
                &g_jet.directional_family(g_jet.aaa_first, dir2, COEFF_SUPPORT_GW),
                z_obs,
            ),
            ad1,
            ad2,
            ad12,
        );
        let eta_aaa_jet = MultiDirJet::bilinear(
            eta_aaa,
            eval_coeff4_at(
                &g_jet.directional_family(g_jet.aaa_first, dir1, COEFF_SUPPORT_GW),
                z_obs,
            ),
            eval_coeff4_at(
                &g_jet.directional_family(g_jet.aaa_first, dir2, COEFF_SUPPORT_GW),
                z_obs,
            ),
            eval_coeff4_at(
                &g_jet.mixed_directional_from_b_family(
                    g_jet.aab_first,
                    dir1,
                    dir2,
                    COEFF_SUPPORT_GW,
                ),
                z_obs,
            ),
        );

        let mut a_u_jets = Vec::with_capacity(p);
        let mut tau_jets = Vec::with_capacity(p);
        let mut tau_a_jets = Vec::with_capacity(p);
        for u in 0..p {
            a_u_jets.push(MultiDirJet::bilinear(au[u], aud1[u], aud2[u], aud12[u]));
            tau_jets.push(scalar_composite_bilinear(
                eval_coeff4_at(&g_au_fixed[u], z_obs),
                eval_coeff4_at(&g_aau_fixed[u], z_obs),
                eval_coeff4_at(&g_aaau_fixed[u], z_obs),
                eval_coeff4_at(
                    &g_jet.param_directional_from_b_family(
                        g_jet.ab_first,
                        u,
                        dir1,
                        COEFF_SUPPORT_GW,
                    ),
                    z_obs,
                ),
                eval_coeff4_at(
                    &g_jet.param_directional_from_b_family(
                        g_jet.ab_first,
                        u,
                        dir2,
                        COEFF_SUPPORT_GW,
                    ),
                    z_obs,
                ),
                eval_coeff4_at(
                    &g_jet.param_mixed_from_bb_family(
                        g_jet.abb_first,
                        u,
                        dir1,
                        dir2,
                        COEFF_SUPPORT_GW,
                    ),
                    z_obs,
                ),
                eval_coeff4_at(
                    &g_jet.param_directional_from_b_family(
                        g_jet.aab_first,
                        u,
                        dir1,
                        COEFF_SUPPORT_GW,
                    ),
                    z_obs,
                ),
                eval_coeff4_at(
                    &g_jet.param_directional_from_b_family(
                        g_jet.aab_first,
                        u,
                        dir2,
                        COEFF_SUPPORT_GW,
                    ),
                    z_obs,
                ),
                ad1,
                ad2,
                ad12,
            ));
            tau_a_jets.push(scalar_composite_bilinear(
                eval_coeff4_at(&g_aau_fixed[u], z_obs),
                eval_coeff4_at(&g_aaau_fixed[u], z_obs),
                0.0,
                eval_coeff4_at(
                    &g_jet.param_directional_from_b_family(
                        g_jet.aab_first,
                        u,
                        dir1,
                        COEFF_SUPPORT_GW,
                    ),
                    z_obs,
                ),
                eval_coeff4_at(
                    &g_jet.param_directional_from_b_family(
                        g_jet.aab_first,
                        u,
                        dir2,
                        COEFF_SUPPORT_GW,
                    ),
                    z_obs,
                ),
                eval_coeff4_at(
                    &g_jet.param_mixed_from_bb_family(
                        g_jet.abb_first,
                        u,
                        dir1,
                        dir2,
                        COEFF_SUPPORT_GW,
                    ),
                    z_obs,
                ),
                0.0,
                0.0,
                ad1,
                ad2,
                ad12,
            ));
        }

        let mut eta_uv_uv = Array2::<f64>::zeros((p, p));
        let mut chi_uv_uv = Array2::<f64>::zeros((p, p));
        let mut d_uv_uv = Array2::<f64>::zeros((p, p));

        for u in 0..p {
            for v in u..p {
                let a_uv_jet = MultiDirJet::bilinear(
                    auv[[u, v]],
                    auvd1[[u, v]],
                    auvd2[[u, v]],
                    auvd12[[u, v]],
                );
                let a_u_prod = a_u_jets[u].mul(&a_u_jets[v]);
                let r_uv_jet = scalar_composite_bilinear(
                    eval_coeff4_at(
                        &g_jet.pair_from_b_family(g_jet.b_first, u, v, COEFF_SUPPORT_GHW),
                        z_obs,
                    ),
                    eval_coeff4_at(
                        &g_jet.pair_from_b_family(g_jet.ab_first, u, v, COEFF_SUPPORT_GW),
                        z_obs,
                    ),
                    eval_coeff4_at(
                        &g_jet.pair_from_b_family(g_jet.aab_first, u, v, COEFF_SUPPORT_GW),
                        z_obs,
                    ),
                    eval_coeff4_at(
                        &g_jet.pair_directional_from_bb_family(
                            g_jet.bb_first,
                            u,
                            v,
                            dir1,
                            COEFF_SUPPORT_GHW,
                        ),
                        z_obs,
                    ),
                    eval_coeff4_at(
                        &g_jet.pair_directional_from_bb_family(
                            g_jet.bb_first,
                            u,
                            v,
                            dir2,
                            COEFF_SUPPORT_GHW,
                        ),
                        z_obs,
                    ),
                    eval_coeff4_at(
                        &g_jet.pair_mixed_from_bbb_family(
                            g_jet.bbb_first,
                            u,
                            v,
                            dir1,
                            dir2,
                            COEFF_SUPPORT_GHW,
                        ),
                        z_obs,
                    ),
                    eval_coeff4_at(
                        &g_jet.pair_directional_from_bb_family(
                            g_jet.abb_first,
                            u,
                            v,
                            dir1,
                            COEFF_SUPPORT_GW,
                        ),
                        z_obs,
                    ),
                    eval_coeff4_at(
                        &g_jet.pair_directional_from_bb_family(
                            g_jet.abb_first,
                            u,
                            v,
                            dir2,
                            COEFF_SUPPORT_GW,
                        ),
                        z_obs,
                    ),
                    ad1,
                    ad2,
                    ad12,
                );
                let chi_uv_fixed_jet = scalar_composite_bilinear(
                    eval_coeff4_at(
                        &g_jet.pair_from_b_family(g_jet.ab_first, u, v, COEFF_SUPPORT_GW),
                        z_obs,
                    ),
                    eval_coeff4_at(
                        &g_jet.pair_from_b_family(g_jet.aab_first, u, v, COEFF_SUPPORT_GW),
                        z_obs,
                    ),
                    0.0,
                    eval_coeff4_at(
                        &g_jet.pair_directional_from_bb_family(
                            g_jet.abb_first,
                            u,
                            v,
                            dir1,
                            COEFF_SUPPORT_GW,
                        ),
                        z_obs,
                    ),
                    eval_coeff4_at(
                        &g_jet.pair_directional_from_bb_family(
                            g_jet.abb_first,
                            u,
                            v,
                            dir2,
                            COEFF_SUPPORT_GW,
                        ),
                        z_obs,
                    ),
                    eval_coeff4_at(
                        &g_jet.pair_mixed_from_bbb_family(
                            g_jet.bbb_first,
                            u,
                            v,
                            dir1,
                            dir2,
                            COEFF_SUPPORT_GW,
                        ),
                        z_obs,
                    ),
                    0.0,
                    0.0,
                    ad1,
                    ad2,
                    ad12,
                );

                let eta_uv_jet = chi_jet
                    .mul(&a_uv_jet)
                    .add(&eta_aa_jet.mul(&a_u_prod))
                    .add(&tau_jets[u].mul(&a_u_jets[v]))
                    .add(&tau_jets[v].mul(&a_u_jets[u]))
                    .add(&r_uv_jet);
                let chi_uv_jet = eta_aa_jet
                    .mul(&a_uv_jet)
                    .add(&eta_aaa_jet.mul(&a_u_prod))
                    .add(&tau_a_jets[u].mul(&a_u_jets[v]))
                    .add(&tau_a_jets[v].mul(&a_u_jets[u]))
                    .add(&chi_uv_fixed_jet);

                eta_uv_uv[[u, v]] = eta_uv_jet.coeff(3);
                eta_uv_uv[[v, u]] = eta_uv_uv[[u, v]];
                chi_uv_uv[[u, v]] = chi_uv_jet.coeff(3);
                chi_uv_uv[[v, u]] = chi_uv_uv[[u, v]];
            }
        }

        let primary_view = SparsePrimaryCoeffJetView::new(
            primary.g,
            primary.h.as_ref(),
            primary.w.as_ref(),
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
        );
        let d_uv_uv_cell_accums = cached
            .cells
            .iter()
            .map(|ce| -> Result<Array2<f64>, String> {
                let mut d_uv_uv = Array2::<f64>::zeros((p, p));
                let cell = ce.partition_cell.cell;
                let st = &ce.state;
                let fx = &ce.fixed;
                let eta_base = [cell.c0, cell.c1, cell.c2, cell.c3];

                let coeff_dir1 =
                    primary_view.directional_family(&fx.coeff_u, dir1, COEFF_SUPPORT_GHW);
                let coeff_dir2 =
                    primary_view.directional_family(&fx.coeff_u, dir2, COEFF_SUPPORT_GHW);
                let coeff_dir12 = primary_view.mixed_directional_from_b_family(
                    &fx.coeff_bu,
                    dir1,
                    dir2,
                    COEFF_SUPPORT_GHW,
                );
                let coeff_a_dir1 =
                    primary_view.directional_family(&fx.coeff_au, dir1, COEFF_SUPPORT_GHW);
                let coeff_a_dir2 =
                    primary_view.directional_family(&fx.coeff_au, dir2, COEFF_SUPPORT_GHW);
                let coeff_a_dir12 = primary_view.mixed_directional_from_b_family(
                    &fx.coeff_abu,
                    dir1,
                    dir2,
                    COEFF_SUPPORT_GHW,
                );
                let coeff_aa_dir1 =
                    primary_view.directional_family(&fx.coeff_aau, dir1, COEFF_SUPPORT_GHW);
                let coeff_aa_dir2 =
                    primary_view.directional_family(&fx.coeff_aau, dir2, COEFF_SUPPORT_GHW);
                let coeff_aa_dir12 = primary_view.mixed_directional_from_b_family(
                    &fx.coeff_aabu,
                    dir1,
                    dir2,
                    COEFF_SUPPORT_GHW,
                );
                let coeff_aaa_dir1 =
                    primary_view.directional_family(&fx.coeff_aaau, dir1, COEFF_SUPPORT_GHW);
                let coeff_aaa_dir2 =
                    primary_view.directional_family(&fx.coeff_aaau, dir2, COEFF_SUPPORT_GHW);
                let coeff_aaa_dir12 = primary_view.mixed_directional_from_b_family(
                    &fx.coeff_aabu,
                    dir1,
                    dir2,
                    COEFF_SUPPORT_GHW,
                );

                let eta_poly_jet = coeff4_composite_bilinear(
                    &eta_base,
                    &fx.dc_da,
                    &fx.dc_daa,
                    &coeff_dir1,
                    &coeff_dir2,
                    &coeff_dir12,
                    &coeff_a_dir1,
                    &coeff_a_dir2,
                    ad1,
                    ad2,
                    ad12,
                );
                let chi_poly_jet = coeff4_composite_bilinear(
                    &fx.dc_da,
                    &fx.dc_daa,
                    &fx.dc_daaa,
                    &coeff_a_dir1,
                    &coeff_a_dir2,
                    &coeff_a_dir12,
                    &coeff_aa_dir1,
                    &coeff_aa_dir2,
                    ad1,
                    ad2,
                    ad12,
                );
                let eta_aa_poly_jet = coeff4_composite_bilinear(
                    &fx.dc_daa,
                    &fx.dc_daaa,
                    &zero4,
                    &coeff_aa_dir1,
                    &coeff_aa_dir2,
                    &coeff_aa_dir12,
                    &coeff_aaa_dir1,
                    &coeff_aaa_dir2,
                    ad1,
                    ad2,
                    ad12,
                );
                let eta_aaa_poly_jet = coeff4_fixed_bilinear(
                    &fx.dc_daaa,
                    &coeff_aaa_dir1,
                    &coeff_aaa_dir2,
                    &coeff_aaa_dir12,
                );

                let mut eta_u_poly_jets = Vec::with_capacity(p);
                let mut chi_u_poly_jets = Vec::with_capacity(p);
                let mut coeff_au_fixed_jets = Vec::with_capacity(p);
                let mut coeff_aau_fixed_jets = Vec::with_capacity(p);
                for u in 0..p {
                    let coeff_u_dir1 = primary_view.param_directional_from_b_family(
                        &fx.coeff_bu,
                        u,
                        dir1,
                        COEFF_SUPPORT_GHW,
                    );
                    let coeff_u_dir2 = primary_view.param_directional_from_b_family(
                        &fx.coeff_bu,
                        u,
                        dir2,
                        COEFF_SUPPORT_GHW,
                    );
                    let coeff_u_dir12 = primary_view.param_mixed_from_bb_family(
                        &fx.coeff_bbu,
                        u,
                        dir1,
                        dir2,
                        COEFF_SUPPORT_GHW,
                    );
                    let coeff_au_dir1 = primary_view.param_directional_from_b_family(
                        &fx.coeff_abu,
                        u,
                        dir1,
                        COEFF_SUPPORT_GHW,
                    );
                    let coeff_au_dir2 = primary_view.param_directional_from_b_family(
                        &fx.coeff_abu,
                        u,
                        dir2,
                        COEFF_SUPPORT_GHW,
                    );
                    let coeff_au_dir12 = primary_view.param_mixed_from_bb_family(
                        &fx.coeff_abbu,
                        u,
                        dir1,
                        dir2,
                        COEFF_SUPPORT_GHW,
                    );
                    let coeff_aau_dir1 = primary_view.param_directional_from_b_family(
                        &fx.coeff_aabu,
                        u,
                        dir1,
                        COEFF_SUPPORT_GHW,
                    );
                    let coeff_aau_dir2 = primary_view.param_directional_from_b_family(
                        &fx.coeff_aabu,
                        u,
                        dir2,
                        COEFF_SUPPORT_GHW,
                    );
                    let coeff_aau_dir12 = primary_view.param_mixed_from_bb_family(
                        &fx.coeff_abbu,
                        u,
                        dir1,
                        dir2,
                        COEFF_SUPPORT_GHW,
                    );

                    let coeff_u_fixed_jet = coeff4_fixed_bilinear(
                        &fx.coeff_u[u],
                        &coeff_u_dir1,
                        &coeff_u_dir2,
                        &coeff_u_dir12,
                    );
                    let coeff_au_fixed_jet = coeff4_fixed_bilinear(
                        &fx.coeff_au[u],
                        &coeff_au_dir1,
                        &coeff_au_dir2,
                        &coeff_au_dir12,
                    );
                    let coeff_aau_fixed_jet = coeff4_fixed_bilinear(
                        &fx.coeff_aau[u],
                        &coeff_aau_dir1,
                        &coeff_aau_dir2,
                        &coeff_aau_dir12,
                    );

                    eta_u_poly_jets.push(poly_add_jets(
                        &poly_scale_jets(&chi_poly_jet, &a_u_jets[u]),
                        &coeff_u_fixed_jet,
                    ));
                    chi_u_poly_jets.push(poly_add_jets(
                        &poly_scale_jets(&eta_aa_poly_jet, &a_u_jets[u]),
                        &coeff_au_fixed_jet,
                    ));
                    coeff_au_fixed_jets.push(coeff_au_fixed_jet);
                    coeff_aau_fixed_jets.push(coeff_aau_fixed_jet);
                }

                for u in 0..p {
                    for v in u..p {
                        let a_uv_jet = MultiDirJet::bilinear(
                            auv[[u, v]],
                            auvd1[[u, v]],
                            auvd2[[u, v]],
                            auvd12[[u, v]],
                        );
                        let a_u_prod = a_u_jets[u].mul(&a_u_jets[v]);
                        let r_uv_fixed_jet = coeff4_fixed_bilinear(
                            &self.cell_pair_second_coeff(primary, &fx.coeff_bu, u, v),
                            &primary_view.pair_directional_from_bb_family(
                                &fx.coeff_bbu,
                                u,
                                v,
                                dir1,
                                COEFF_SUPPORT_GHW,
                            ),
                            &primary_view.pair_directional_from_bb_family(
                                &fx.coeff_bbu,
                                u,
                                v,
                                dir2,
                                COEFF_SUPPORT_GHW,
                            ),
                            &primary_view.pair_mixed_from_bbb_family(
                                &fx.coeff_bbbu,
                                u,
                                v,
                                dir1,
                                dir2,
                                COEFF_SUPPORT_GHW,
                            ),
                        );
                        let chi_uv_fixed_jet = coeff4_fixed_bilinear(
                            &self.cell_pair_third_coeff_a(primary, &fx.coeff_abu, u, v),
                            &primary_view.pair_directional_from_bb_family(
                                &fx.coeff_abbu,
                                u,
                                v,
                                dir1,
                                COEFF_SUPPORT_GHW,
                            ),
                            &primary_view.pair_directional_from_bb_family(
                                &fx.coeff_abbu,
                                u,
                                v,
                                dir2,
                                COEFF_SUPPORT_GHW,
                            ),
                            &primary_view.pair_mixed_from_bbb_family(
                                &fx.coeff_bbbu,
                                u,
                                v,
                                dir1,
                                dir2,
                                COEFF_SUPPORT_GHW,
                            ),
                        );

                        let eta_uv_poly_jet = poly_add_jets(
                            &poly_add_jets(
                                &poly_scale_jets(&chi_poly_jet, &a_uv_jet),
                                &poly_scale_jets(&eta_aa_poly_jet, &a_u_prod),
                            ),
                            &poly_add_jets(
                                &poly_scale_jets(&coeff_au_fixed_jets[u], &a_u_jets[v]),
                                &poly_add_jets(
                                    &poly_scale_jets(&coeff_au_fixed_jets[v], &a_u_jets[u]),
                                    &r_uv_fixed_jet,
                                ),
                            ),
                        );
                        let chi_uv_poly_jet = poly_add_jets(
                            &poly_add_jets(
                                &poly_scale_jets(&eta_aa_poly_jet, &a_uv_jet),
                                &poly_scale_jets(&eta_aaa_poly_jet, &a_u_prod),
                            ),
                            &poly_add_jets(
                                &poly_scale_jets(&coeff_aau_fixed_jets[u], &a_u_jets[v]),
                                &poly_add_jets(
                                    &poly_scale_jets(&coeff_aau_fixed_jets[v], &a_u_jets[u]),
                                    &chi_uv_fixed_jet,
                                ),
                            ),
                        );

                        let t1 = chi_uv_poly_jet.clone();
                        let t2 = poly_scale_jets(
                            &poly_mul_jets(
                                &poly_mul_jets(&chi_u_poly_jets[v], &eta_poly_jet),
                                &eta_u_poly_jets[u],
                            ),
                            &MultiDirJet::constant(2, -1.0),
                        );
                        let t3 = poly_scale_jets(
                            &poly_mul_jets(
                                &poly_mul_jets(&chi_u_poly_jets[u], &eta_poly_jet),
                                &eta_u_poly_jets[v],
                            ),
                            &MultiDirJet::constant(2, -1.0),
                        );
                        let t4 = poly_scale_jets(
                            &poly_mul_jets(
                                &chi_poly_jet,
                                &poly_add_jets(
                                    &poly_mul_jets(&eta_u_poly_jets[u], &eta_u_poly_jets[v]),
                                    &poly_mul_jets(&eta_poly_jet, &eta_uv_poly_jet),
                                ),
                            ),
                            &MultiDirJet::constant(2, -1.0),
                        );
                        let t5 = poly_mul_jets(
                            &chi_poly_jet,
                            &poly_mul_jets(
                                &poly_mul_jets(&eta_poly_jet, &eta_poly_jet),
                                &poly_mul_jets(&eta_u_poly_jets[u], &eta_u_poly_jets[v]),
                            ),
                        );
                        let i_base_jet = poly_add_jets(
                            &poly_add_jets(&poly_add_jets(&t1, &t2), &t3),
                            &poly_add_jets(&t4, &t5),
                        );

                        let i_base = poly_coeff_mask(&i_base_jet, 0);
                        let i_base_d2 = poly_coeff_mask(&i_base_jet, 2);
                        let i_base_d12 = poly_coeff_mask(&i_base_jet, 3);
                        let eta_poly = poly_coeff_mask(&eta_poly_jet, 0);
                        let eta_d1_poly = poly_coeff_mask(&eta_poly_jet, 1);
                        let eta_d2_poly = poly_coeff_mask(&eta_poly_jet, 2);
                        let eta_d12_poly = poly_coeff_mask(&eta_poly_jet, 3);

                        let correction = poly_add(
                            &poly_mul(
                                &poly_add(
                                    &poly_mul(&eta_d2_poly, &eta_d1_poly),
                                    &poly_mul(&eta_poly, &eta_d12_poly),
                                ),
                                &i_base,
                            ),
                            &poly_mul(&poly_mul(&eta_poly, &eta_d1_poly), &i_base_d2),
                        );
                        let full_integrand = poly_sub(&i_base_d12, &correction);
                        let value = exact_kernel::cell_polynomial_integral_from_moments(
                            &full_integrand,
                            &st.moments,
                            "survival D_t second derivative bidirectional",
                        )?;
                        d_uv_uv[[u, v]] += value;
                        d_uv_uv[[v, u]] = d_uv_uv[[u, v]];
                    }
                }
                Ok(d_uv_uv)
            })
            .collect::<Result<Vec<_>, String>>()?;
        for cell_d_uv_uv in d_uv_uv_cell_accums {
            for u in 0..p {
                for v in 0..p {
                    d_uv_uv[[u, v]] += cell_d_uv_uv[[u, v]];
                }
            }
        }

        Ok(SurvivalFlexTimepointBiDirectionalExact {
            eta_uv_uv,
            chi_uv_uv,
            d_uv_uv,
        })
    }

    /// Exact third-order directional contraction for the flexible survival
    /// path.  Returns D_dir H[u,v] where H is the primary-space NLL Hessian.
    pub(crate) fn row_flex_primary_third_contracted_exact(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dir: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        self.ensure_scalar_flex_exact_score_geometry("row_flex_primary_third_contracted_exact")?;
        let primary = flex_primary_slices(self);
        let p = primary.total;
        if dir.len() != p {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival third contracted: dir length {} != primary dimension {p}",
                    dir.len()
                ),
            }
            .into());
        }
        if dir.iter().all(|v| v.abs() == 0.0) {
            return Ok(Array2::<f64>::zeros((p, p)));
        }

        let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
        let q0 = q_geom.q0;
        let q1 = q_geom.q1;
        let qd1 = q_geom.qd1;
        let g = block_states[2].eta[row];
        let beta_h = self.flex_score_beta(block_states)?;
        let beta_w = self.flex_link_beta(block_states)?;
        let o_infl = self.influence_index_offset(row, block_states)?;

        if survival_derivative_guard_violated(qd1, self.derivative_guard) {
            return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
                reason: format!(
                    "survival third contracted monotonicity violated at row {row}: qd1={qd1:.3e}"
                ),
            }
            .into());
        }

        let (a0, d0) = self.solve_row_survival_intercept_with_slot(
            q0,
            g,
            beta_h,
            beta_w,
            Some((row, SurvivalInterceptSlotKind::Entry)),
        )?;
        let (a1, d1) = self.solve_row_survival_intercept_with_slot(
            q1,
            g,
            beta_h,
            beta_w,
            Some((row, SurvivalInterceptSlotKind::Exit)),
        )?;

        let entry = self.compute_survival_timepoint_exact(
            row, &primary, q0, primary.q0, a0, g, d0, beta_h, beta_w, o_infl, false,
        )?;
        let exit = self.compute_survival_timepoint_exact(
            row, &primary, q1, primary.q1, a1, g, d1, beta_h, beta_w, o_infl, true,
        )?;

        if !exit.chi.is_finite() || exit.chi <= 0.0 {
            return Err(SurvivalMarginalSlopeError::NumericalFailure {
                reason: format!(
                    "survival third contracted row {row}: non-positive chi1={:.3e}",
                    exit.chi,
                ),
            }
            .into());
        }

        let entry_ext = self.compute_survival_timepoint_directional_exact(
            row, &primary, q0, primary.q0, a0, g, beta_h, beta_w, dir, false,
        )?;
        let exit_ext = self.compute_survival_timepoint_directional_exact(
            row, &primary, q1, primary.q1, a1, g, beta_h, beta_w, dir, true,
        )?;

        // Delegate the per-(u, v) assembly to the Block 10 GPU-substrate
        // pure assembler in `crate::families::survival_marginal_slope_gpu`.  This is the
        // single source of truth for the third-contraction inner loop —
        // shared with the GPU dispatch path so CPU/GPU cannot drift.
        let entry_b = block10_pack_base(&entry);
        let exit_b = block10_pack_base(&exit);
        let entry_d = block10_pack_dir(&entry_ext);
        let exit_d = block10_pack_dir(&exit_ext);
        let dir_vec: Vec<f64> = dir.to_vec();
        let inputs = crate::families::survival_marginal_slope_gpu::SurvivalFlexBlock10ThirdInputs {
            p,
            qd1_index: primary.qd1,
            qd1,
            w: self.weights[row],
            d: self.event[row],
            dir: &dir_vec,
            entry_base: &entry_b,
            exit_base: &exit_b,
            entry_ext: &entry_d,
            exit_ext: &exit_d,
        };
        let flat =
            crate::families::survival_marginal_slope_gpu::cpu_oracle_third_contraction(&inputs)?;
        Ok(Array2::<f64>::from_shape_vec((p, p), flat).map_err(|e| e.to_string())?)
    }

    /// Fourth-order directional contraction for the flexible survival path.
    ///
    /// The mixed second-directional timepoint transport is carried exactly
    /// through the implicit intercept solve, the observed-point eta/chi jets,
    /// and the cellwise density-normalization integrand.
    pub(crate) fn row_flex_primary_fourth_contracted_exact(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dir_u: &Array1<f64>,
        dir_v: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        self.ensure_scalar_flex_exact_score_geometry("row_flex_primary_fourth_contracted_exact")?;
        let primary = flex_primary_slices(self);
        let p = primary.total;
        if dir_u.len() != p || dir_v.len() != p {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival fourth contracted: dir lengths ({},{}) != {p}",
                    dir_u.len(),
                    dir_v.len(),
                ),
            }
            .into());
        }
        if dir_u.iter().all(|v| v.abs() == 0.0) || dir_v.iter().all(|v| v.abs() == 0.0) {
            return Ok(Array2::<f64>::zeros((p, p)));
        }

        let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
        let q0 = q_geom.q0;
        let q1 = q_geom.q1;
        let qd1 = q_geom.qd1;
        let g = block_states[2].eta[row];
        let beta_h = self.flex_score_beta(block_states)?;
        let beta_w = self.flex_link_beta(block_states)?;
        let o_infl = self.influence_index_offset(row, block_states)?;

        if survival_derivative_guard_violated(qd1, self.derivative_guard) {
            return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
                reason: format!(
                    "survival fourth contracted monotonicity violated at row {row}: qd1={qd1:.3e}"
                ),
            }
            .into());
        }

        let (a0, d0) = self.solve_row_survival_intercept_with_slot(
            q0,
            g,
            beta_h,
            beta_w,
            Some((row, SurvivalInterceptSlotKind::Entry)),
        )?;
        let (a1, d1) = self.solve_row_survival_intercept_with_slot(
            q1,
            g,
            beta_h,
            beta_w,
            Some((row, SurvivalInterceptSlotKind::Exit)),
        )?;

        let entry_base = self.compute_survival_timepoint_exact(
            row, &primary, q0, primary.q0, a0, g, d0, beta_h, beta_w, o_infl, false,
        )?;
        let exit_base = self.compute_survival_timepoint_exact(
            row, &primary, q1, primary.q1, a1, g, d1, beta_h, beta_w, o_infl, true,
        )?;

        if !exit_base.chi.is_finite() || exit_base.chi <= 0.0 {
            return Err(SurvivalMarginalSlopeError::NumericalFailure {
                reason: format!(
                    "survival fourth contracted row {row}: non-positive chi1={:.3e}",
                    exit_base.chi,
                ),
            }
            .into());
        }

        let entry_ext_u = self.compute_survival_timepoint_directional_exact(
            row, &primary, q0, primary.q0, a0, g, beta_h, beta_w, dir_u, false,
        )?;
        let entry_ext_v = self.compute_survival_timepoint_directional_exact(
            row, &primary, q0, primary.q0, a0, g, beta_h, beta_w, dir_v, false,
        )?;
        let exit_ext_u = self.compute_survival_timepoint_directional_exact(
            row, &primary, q1, primary.q1, a1, g, beta_h, beta_w, dir_u, true,
        )?;
        let exit_ext_v = self.compute_survival_timepoint_directional_exact(
            row, &primary, q1, primary.q1, a1, g, beta_h, beta_w, dir_v, true,
        )?;

        // Bidirectional extensions D_{d1} D_{d2} (η_uv, χ_uv, D_uv) via exact
        // IFT second-order recursion through the cell kernel.
        let entry_bi = self.compute_survival_timepoint_bidirectional_exact(
            row, &primary, q0, primary.q0, a0, g, beta_h, beta_w, dir_u, dir_v,
        )?;
        let exit_bi = self.compute_survival_timepoint_bidirectional_exact(
            row, &primary, q1, primary.q1, a1, g, beta_h, beta_w, dir_u, dir_v,
        )?;

        // Delegate the per-(u, v) assembly + averaged-ordered
        // symmetrization to the Block 10 GPU-substrate pure assembler.
        // Single source of truth shared with the GPU dispatch path.
        let entry_b = block10_pack_base(&entry_base);
        let exit_b = block10_pack_base(&exit_base);
        let entry_d1 = block10_pack_dir(&entry_ext_u);
        let entry_d2 = block10_pack_dir(&entry_ext_v);
        let exit_d1 = block10_pack_dir(&exit_ext_u);
        let exit_d2 = block10_pack_dir(&exit_ext_v);
        let entry_bi_p = block10_pack_bi(&entry_bi);
        let exit_bi_p = block10_pack_bi(&exit_bi);
        let dir_u_vec: Vec<f64> = dir_u.to_vec();
        let dir_v_vec: Vec<f64> = dir_v.to_vec();
        let inputs =
            crate::families::survival_marginal_slope_gpu::SurvivalFlexBlock10FourthInputs {
                p,
                qd1_index: primary.qd1,
                qd1,
                w: self.weights[row],
                d: self.event[row],
                dir_u: &dir_u_vec,
                dir_v: &dir_v_vec,
                entry_base: &entry_b,
                exit_base: &exit_b,
                entry_ext_u: &entry_d1,
                entry_ext_v: &entry_d2,
                exit_ext_u: &exit_d1,
                exit_ext_v: &exit_d2,
                entry_bi: &entry_bi_p,
                exit_bi: &exit_bi_p,
            };
        let flat =
            crate::families::survival_marginal_slope_gpu::cpu_oracle_fourth_contraction(&inputs)
                .map_err(|e| format!("block10 fourth contraction: {e}"))?;
        let mut out = Array2::<f64>::zeros((p, p));
        for u in 0..p {
            for v in 0..p {
                out[[u, v]] = flat[u * p + v];
            }
        }
        Ok(out)
    }

    fn row_primary_third_contracted_general(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dir: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        if self.effective_flex_active(block_states)? {
            self.row_flex_primary_third_contracted_exact(row, block_states, dir)
        } else {
            self.row_primary_third_contracted(row, block_states, dir.view())
        }
    }

    fn row_primary_fourth_contracted(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dir_u: ArrayView1<'_, f64>,
        dir_v: ArrayView1<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        // Batched path delegating to the shared k=6 jet helper.
        let r = self.row_primary_fourth_contracted_batched(row, block_states, dir_u, dir_v)?;
        let mut out = Array2::<f64>::zeros((N_PRIMARY, N_PRIMARY));
        for a in 0..N_PRIMARY {
            for b in 0..N_PRIMARY {
                out[[a, b]] = r[a][b];
            }
        }
        Ok(out)
    }

    fn row_primary_fourth_contracted_general(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dir_u: &Array1<f64>,
        dir_v: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        if self.effective_flex_active(block_states)? {
            self.row_flex_primary_fourth_contracted_exact(row, block_states, dir_u, dir_v)
        } else {
            self.row_primary_fourth_contracted(row, block_states, dir_u.view(), dir_v.view())
        }
    }

    // ── Pullback through design matrices ──────────────────────────────

    /// Accumulate the pullback of a primary-space Hessian into coefficient-space.
    ///
    /// Writes directly into `target` subslices via sparse-aware row primitives —
    /// no dense row buffers or temporary blocks are allocated.
    fn add_pullback_primary_hessian(
        &self,
        target: &mut Array2<f64>,
        row: usize,
        slices: &BlockSlices,
        primary_hessian: &Array2<f64>,
    ) {
        let h = primary_hessian;
        let time_designs = [
            &self.design_entry,
            &self.design_exit,
            &self.design_derivative_exit,
        ];

        // time-time block: Σ_{a,b} H[a,b] * time_a_row ⊗ time_b_row
        for a in 0..3 {
            for b in 0..3 {
                let alpha = h[[a, b]];
                if alpha == 0.0 {
                    continue;
                }
                time_designs[a]
                    .row_outer_into_view(
                        row,
                        time_designs[b],
                        alpha,
                        target.slice_mut(s![slices.time.clone(), slices.time.clone()]),
                    )
                    .expect("time block row_outer_into dimension mismatch");
            }
        }

        // marginal-marginal block: (H[0,0]+H[0,1]+H[1,0]+H[1,1]) * m_row ⊗ m_row
        self.marginal_design
            .syr_row_into_view(
                row,
                h[[0, 0]] + h[[0, 1]] + h[[1, 0]] + h[[1, 1]],
                target.slice_mut(s![slices.marginal.clone(), slices.marginal.clone()]),
            )
            .expect("marginal syr_row_into dimension mismatch");

        // logslope-logslope block: H[3,3] * g_row ⊗ g_row
        self.logslope_design
            .syr_row_into_view(
                row,
                h[[3, 3]],
                target.slice_mut(s![slices.logslope.clone(), slices.logslope.clone()]),
            )
            .expect("logslope syr_row_into dimension mismatch");

        // marginal-logslope block: (H[0,3]+H[1,3]) * m_row ⊗ g_row  (+ transpose)
        {
            let alpha_mg = h[[0, 3]] + h[[1, 3]];
            if alpha_mg != 0.0 {
                self.marginal_design
                    .row_outer_into_view(
                        row,
                        &self.logslope_design,
                        alpha_mg,
                        target.slice_mut(s![slices.marginal.clone(), slices.logslope.clone()]),
                    )
                    .expect("marginal-logslope row_outer_into dimension mismatch");
                self.logslope_design
                    .row_outer_into_view(
                        row,
                        &self.marginal_design,
                        alpha_mg,
                        target.slice_mut(s![slices.logslope.clone(), slices.marginal.clone()]),
                    )
                    .expect("logslope-marginal row_outer_into dimension mismatch");
            }
        }

        // time-logslope block: H[a,3] * time_a_row ⊗ g_row  (+ transpose)
        for a in 0..3 {
            let alpha = h[[a, 3]];
            if alpha == 0.0 {
                continue;
            }
            time_designs[a]
                .row_outer_into_view(
                    row,
                    &self.logslope_design,
                    alpha,
                    target.slice_mut(s![slices.time.clone(), slices.logslope.clone()]),
                )
                .expect("time-logslope row_outer_into dimension mismatch");
            self.logslope_design
                .row_outer_into_view(
                    row,
                    time_designs[a],
                    alpha,
                    target.slice_mut(s![slices.logslope.clone(), slices.time.clone()]),
                )
                .expect("logslope-time row_outer_into dimension mismatch");
        }

        // time-marginal block: (H[a,0]+H[a,1]) * time_a_row ⊗ m_row  (+ transpose)
        for a in 0..3 {
            let alpha = h[[a, 0]] + h[[a, 1]];
            if alpha == 0.0 {
                continue;
            }
            time_designs[a]
                .row_outer_into_view(
                    row,
                    &self.marginal_design,
                    alpha,
                    target.slice_mut(s![slices.time.clone(), slices.marginal.clone()]),
                )
                .expect("time-marginal row_outer_into dimension mismatch");
            self.marginal_design
                .row_outer_into_view(
                    row,
                    time_designs[a],
                    alpha,
                    target.slice_mut(s![slices.marginal.clone(), slices.time.clone()]),
                )
                .expect("marginal-time row_outer_into dimension mismatch");
        }
    }

    /// Block-diagonal-only pullback: writes only the principal time-time,
    /// marginal-marginal, and logslope-logslope rowwise contributions into
    /// per-block targets. Used by `evaluate()` to populate per-block working
    /// sets without ever materializing the cross blocks.
    fn add_pullback_block_diagonals(
        &self,
        row: usize,
        primary_hessian: &Array2<f64>,
        time_target: &mut Array2<f64>,
        marginal_target: &mut Array2<f64>,
        logslope_target: &mut Array2<f64>,
    ) {
        let h = primary_hessian;
        let time_designs = [
            &self.design_entry,
            &self.design_exit,
            &self.design_derivative_exit,
        ];
        for a in 0..3 {
            for b in 0..3 {
                let alpha = h[[a, b]];
                if alpha == 0.0 {
                    continue;
                }
                time_designs[a]
                    .row_outer_into_view(row, time_designs[b], alpha, time_target.view_mut())
                    .expect("time block row_outer_into dimension mismatch");
            }
        }
        let alpha_mm = h[[0, 0]] + h[[0, 1]] + h[[1, 0]] + h[[1, 1]];
        self.marginal_design
            .syr_row_into_view(row, alpha_mm, marginal_target.view_mut())
            .expect("marginal syr_row_into dimension mismatch");
        self.logslope_design
            .syr_row_into_view(row, h[[3, 3]], logslope_target.view_mut())
            .expect("logslope syr_row_into dimension mismatch");
    }

    fn row_primary_direction_from_flat_dynamic(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        slices: &BlockSlices,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        let flex_primary = self
            .effective_flex_active(block_states)?
            .then(|| flex_primary_slices(self));
        let mut out = Array1::<f64>::zeros(flex_primary.as_ref().map_or(N_PRIMARY, |p| p.total));
        let d_time = d_beta_flat.slice(s![slices.time.clone()]);
        let d_marginal = d_beta_flat.slice(s![slices.marginal.clone()]);
        let q_geom = self.row_dynamic_q_geometry(row, block_states)?;

        let q0_dir = q_geom.dq0_time.dot(&d_time) + q_geom.dq0_marginal.dot(&d_marginal);
        let q1_dir = q_geom.dq1_time.dot(&d_time) + q_geom.dq1_marginal.dot(&d_marginal);
        let qd1_dir = q_geom.dqd1_time.dot(&d_time) + q_geom.dqd1_marginal.dot(&d_marginal);
        let g_dir = self
            .logslope_design
            .dot_row_view(row, d_beta_flat.slice(s![slices.logslope.clone()]));

        if let Some(primary) = flex_primary.as_ref() {
            out[primary.q0] = q0_dir;
            out[primary.q1] = q1_dir;
            out[primary.qd1] = qd1_dir;
            out[primary.g] = g_dir;
            for (primary_range, block_range) in flex_identity_block_pairs(primary, slices) {
                out.slice_mut(s![primary_range])
                    .assign(&d_beta_flat.slice(s![block_range]));
            }
        } else {
            out[0] = q0_dir;
            out[1] = q1_dir;
            out[2] = qd1_dir;
            out[3] = g_dir;
        }
        Ok(out)
    }

    // ── Psi (spatial length-scale) derivatives ────────────────────────

    // ── Psi terms (first and second order) ────────────────────────────
    //
    // All three psi methods (first-order, second-order, directional derivative)
    // use block-local accumulation via BlockHessianAccumulator. Per-row work is
    // O(max(p_block²)) instead of O(p²), eliminating the dense p×p bottleneck
    // that breaks multi-axis Duchon / per-axis length scaling.

    /// Resolve psi block info: (block_idx, local_idx, p_block, label).
    fn psi_block_info(
        &self,
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<(usize, usize, usize, &'static str)>, String> {
        let Some((block_idx, local_idx)) = psi_derivative_location(derivative_blocks, psi_index)
        else {
            return Ok(None);
        };
        match block_idx {
            1 => Ok(Some((
                block_idx,
                local_idx,
                self.marginal_design.ncols(),
                "SurvivalMarginalSlope marginal",
            ))),
            2 => Ok(Some((
                block_idx,
                local_idx,
                self.logslope_design.ncols(),
                "SurvivalMarginalSlope logslope",
            ))),
            _ => Err(SurvivalMarginalSlopeError::UnsupportedConfiguration {
                reason: format!(
                    "survival marginal-slope psi: only baseline/slope spatial blocks are supported, got block {block_idx}"
                ),
            }
            .into()),
        }
    }

    /// Accumulate block-local score from a primary-space vector (replaces
    /// pullback_primary_vector + score += for score accumulation).
    fn accumulate_score_blockwise(
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
        self.logslope_design
            .axpy_row_into(row, primary[3], &mut score_g.view_mut())?;
        Ok(())
    }

    fn accumulate_score_identity_blocks(
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
    fn accumulate_score_with_q_geometry(
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
        self.logslope_design
            .axpy_row_into(row, primary[3], &mut score_g.view_mut())?;
        Ok(())
    }

    /// U_{iB}^α contribution to score (eq 46, term 1) for timewiggle + marginal ψ.
    fn accumulate_score_timewiggle_psi_u(
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
    fn timewiggle_psi_action(
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

    fn psi_terms_inner(
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
        let policy = crate::resource::ResourcePolicy::default_library();
        let psi_map = crate::families::custom_family::resolve_custom_family_x_psi_map(
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
            psi_map: crate::families::custom_family::PsiDesignMap,
            loading: Array1<f64>,
            beta_psi: &'a Array1<f64>,
        }
        let policy = crate::resource::ResourcePolicy::default_library();
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
            let psi_map = crate::families::custom_family::resolve_custom_family_x_psi_map(
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

    fn psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        self.psi_terms_inner(block_states, derivative_blocks, psi_index, None)
    }

    fn psi_second_order_terms_inner(
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
        let policy = crate::resource::ResourcePolicy::default_library();
        let psi_map_i = crate::families::custom_family::resolve_custom_family_x_psi_map(
            deriv_i,
            self.n,
            p_psi_i,
            0..self.n,
            label_i,
            &policy,
        )?;
        let psi_map_j = crate::families::custom_family::resolve_custom_family_x_psi_map(
            deriv_j,
            self.n,
            p_psi_j,
            0..self.n,
            label_j,
            &policy,
        )?;
        let psi_map_ij = if same_block {
            Some(
                crate::families::custom_family::resolve_custom_family_x_psi_psi_map(
                    deriv_i,
                    deriv_j,
                    local_idx_j,
                    self.n,
                    p_psi_i,
                    0..self.n,
                    label_i,
                    &policy,
                )?,
            )
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

    fn psi_second_order_terms(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        self.psi_second_order_terms_inner(block_states, derivative_blocks, psi_i, psi_j, None)
    }

    fn psi_hessian_directional_derivative(
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
    fn psi_hessian_directional_derivative_accumulator(
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
        let policy = crate::resource::ResourcePolicy::default_library();
        let psi_map = crate::families::custom_family::resolve_custom_family_x_psi_map(
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

    fn exact_newton_joint_hessian_operator(
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
        let final_acc = row_iter
            .into_par_iter()
            .try_fold(make_fused, |mut acc, row| -> Result<FusedAcc, String> {
                let (state, nll_acc, grad_acc, q_geom) = &mut acc;
                self.row_dynamic_q_geometry_into(row, block_states, q_geom)?;
                let (row_nll, mut g, mut h) = if let Some(ref primary) = primary {
                    self.compute_row_flex_primary_gradient_hessian_exact(
                        row,
                        block_states,
                        q_geom,
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
                *nll_acc -= row_nll * w;
                // Joint gradient: q-geometry pullback for time/marginal/logslope
                // primary outputs, plus identity contribution for flex blocks
                // (score_warp_dev, link_dev). Matches the gradient half of
                // `accumulate_dynamic_q_joint_row`.
                self.accumulate_dynamic_q_core_gradient(
                    row,
                    &slices,
                    q_geom,
                    g.slice(s![0..N_PRIMARY]),
                    grad_acc,
                )?;
                for (primary_range, joint_range) in identity_blocks.iter() {
                    for local in 0..primary_range.len() {
                        grad_acc[joint_range.start + local] -= g[primary_range.start + local];
                    }
                }
                // Block-Hessian pullback (unchanged).
                state.add_pullback_with_q_geometry(self, row, q_geom, &g, &h)?;
                Ok(acc)
            })
            .try_reduce(make_fused, |mut a, b| -> Result<FusedAcc, String> {
                a.0.add(&b.0);
                a.1 += b.1;
                a.2 += &b.2;
                Ok(a)
            })?;
        // The fourth field (DynRow workspace) is per-thread scratch; the
        // leftover from the last reducer thread carries no aggregate meaning
        // and falls out of scope here. The second field accumulates the
        // *signed* log-likelihood (`state.0 -= row_nll`), mirroring the
        // sign convention in `evaluate_exact_newton_joint_dynamic_q_dense`.
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
        let acc = row_iter
            .into_par_iter()
            .try_fold(make_acc_ws, |mut acc, row| -> Result<_, String> {
                let (state, q_geom) = &mut acc;
                self.row_dynamic_q_geometry_into(row, block_states, q_geom)?;
                let h_pi = self
                    .compute_row_flex_primary_gradient_hessian_exact(
                        row,
                        block_states,
                        q_geom,
                        &primary,
                    )?
                    .2;
                let u_d = self.row_primary_direction_from_flat_dynamic(
                    row,
                    block_states,
                    &slices,
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
                state.add_pullback_with_q_geometry(self, row, q_geom, &h_ud, &t_ud)?;
                Ok(acc)
            })
            .try_reduce(make_acc_ws, |mut a, b| -> Result<_, String> {
                a.0.add(&b.0);
                Ok(a)
            })?
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
        let acc = row_iter
            .into_par_iter()
            .try_fold(make_acc_ws, |mut acc, row| -> Result<_, String> {
                let (state, q_geom) = &mut acc;
                self.row_dynamic_q_geometry_into(row, block_states, q_geom)?;
                let ud =
                    self.row_primary_direction_from_flat_dynamic(row, block_states, &slices, d_u)?;
                let ue =
                    self.row_primary_direction_from_flat_dynamic(row, block_states, &slices, d_v)?;
                let mut q_de =
                    self.row_flex_primary_fourth_contracted_exact(row, block_states, &ud, &ue)?;
                let t_d = self.row_flex_primary_third_contracted_exact(row, block_states, &ud)?;
                let mut gamma = t_d.dot(&ue);
                let w = row_weights[row];
                if w != 1.0 {
                    gamma.mapv_inplace(|v| v * w);
                    q_de.mapv_inplace(|v| v * w);
                }
                state.add_pullback_with_q_geometry(self, row, q_geom, &gamma, &q_de)?;
                Ok(acc)
            })
            .try_reduce(make_acc_ws, |mut a, b| -> Result<_, String> {
                a.0.add(&b.0);
                Ok(a)
            })?
            .0;
        Ok(Arc::new(acc.into_operator(slices)) as Arc<dyn HyperOperator>)
    }
}
