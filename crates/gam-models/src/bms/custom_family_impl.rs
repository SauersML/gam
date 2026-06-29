use super::cell_moment_assembly::{
    assemble_bms_block_local_s_psi, fill_link_basis_cell_coeff_jet, fill_score_basis_cell_coeff_jet,
};
use super::exact_eval_cache::*;
use super::family::*;
use super::gradient_paths::*;
use super::hessian_paths::*;
use super::row_kernel::*;
use super::*;

impl CustomFamily for BernoulliMarginalSlopeFamily {
    // Bernoulli marginal-slope fits have a genuine separation regime
    // (near-perfectly-classified rows), so opt into the self-limiting
    // Jeffreys/Firth curvature that bounds the coefficient there. The trait
    // default flipped to OFF in gam#1395 (the flat-prior exact-Newton objective
    // carries no Jeffreys term); families with a real separation regime opt in.
    fn joint_jeffreys_term_required(&self) -> bool {
        true
    }

    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    // Historically the link-deviation basis used a β-dependent moment anchor
    // (empirical at the rigid-pilot η₀); as PIRLS drifted η away from η₀,
    // `Σᵢ wᵢ b(η_currentᵢ)` re-acquired a constant component aliased with
    // the location intercept, collapsing σ_min(joint H+S) to ridge_floor and
    // leaking spurious terms into d log|H|/dρ. The basis is now constructed
    // with a β-INDEPENDENT smoothness-penalty null-space drop (see
    // `DeviationRuntime::try_new` and
    // `smoothness_nullspace_orthogonal_complement` in `deviation_runtime.rs`):
    // the basis structurally cannot represent polynomials of degree
    // `< max_penalty_derivative_order`, so the location block's intercept and
    // any unpenalized location-linear absorb constants/linears in η at every
    // PIRLS iteration. With `S` full-rank on the transformed basis,
    // `σ_min(H+S) ≥ smallest positive eigenvalue of S` — orders of magnitude
    // above ridge_floor — so the original "near-null direction" no longer
    // appears. We retain `HardPseudo` for the exact-newton outer hessian
    // path because numerical rounding can still leave eigenvalues that are
    // very small (though no longer at the floor); excluding σ ≤ ε from both
    // log|H| and its gradient consistently is harmless when there is no
    // genuine null space and remains correct if a residual one ever
    // re-appears (defense in depth, mirroring
    // BinomialLocationScaleWiggleFamily).
    fn pseudo_logdet_mode(&self) -> crate::custom_family::PseudoLogdetMode {
        crate::custom_family::PseudoLogdetMode::HardPseudo
    }

    /// #979/#1040 (binary twin of the survival-MS hang): engage the inner
    /// self-vanishing Levenberg–Marquardt μ on a FULL-RANK-but-ILL-CONDITIONED
    /// penalized Hessian, not only on a rank-deficient one.
    ///
    /// When a flex deviation block is active (`score_warp` / `link_dev`) the
    /// joint inner solve takes the CONSTRAINED active-set QP path
    /// (`block_linear_constraints` emits the structural monotonicity rows). The
    /// marginal and log-slope surfaces share one matern PC basis, so even with
    /// the β-independent null-space drop that lifts `σ_min(H+S)` off `ridge_floor`
    /// (see `pseudo_logdet_mode` above), `H_pen` stays full-rank yet severely
    /// ill-conditioned (cond ≫ 1e4) for the probit-over-2–3-PCs fits. With the
    /// constrained Levenberg floor gated on `nullity > 0` ALONE, the near-null
    /// mode is undamped: the active-set minimiser is unique only to round-off
    /// along it, the proposal slides an O(1) step every cycle, `step_inf` never
    /// exhausts, the KKT / constrained-fixed-point certificate never fires, and
    /// the inner joint-Newton grinds its full cycle budget on EVERY outer ρ-eval
    /// — the ~35s-per-seed pre-warm slowdown (#979). The self-vanishing μ
    /// (∝ projected KKT residual → 0 at the fixed point) gives the near-null
    /// mode a tiny positive curvature so the minimiser is unique and `step_inf`
    /// exhausts, WITHOUT moving the converged β (the link-deviation / log-slope
    /// target is preserved exactly). Applied only when genuinely ill-conditioned
    /// (`cond > LEVENBERG_ILL_CONDITIONING_THRESHOLD`); the rigid
    /// (no-flex / unconstrained) fit keeps the EXACT undamped Newton/KKT solve
    /// and its quadratic convergence.
    fn levenberg_on_ill_conditioning(&self) -> bool {
        true
    }

    fn coefficient_hessian_cost(&self, specs: &[ParameterBlockSpec]) -> u64 {
        // Operator-aware: rigid Bernoulli marginal-slope wires the K=2
        // RowKernel through a matrix-free workspace that applies joint Hv at
        // O(n · (p_marginal + p_logslope + p_flex)) per call. Only fall back
        // to the dense `n · (Σ p_b)²` build when `use_joint_matrix_free_path`
        // declines the operator path.
        crate::location_scale_engine::location_scale_coefficient_hessian_cost(
            self.y.len() as u64,
            specs,
        )
    }

    fn exact_outer_derivative_order(
        &self,
        specs: &[ParameterBlockSpec],
        _: &BlockwiseFitOptions,
    ) -> crate::custom_family::ExactOuterDerivativeOrder {
        use crate::custom_family::ExactOuterDerivativeOrder;

        let flex_active = self.score_warp.is_some() || self.link_dev.is_some();
        let coefficient_work = self
            .coefficient_hessian_cost(specs)
            .max(self.coefficient_gradient_cost(specs));
        let dense_available = self.outer_hyper_hessian_dense_available(specs);
        let hvp_available = self.outer_hyper_hessian_hvp_available(specs);
        // FLEX (`score_warp` / `link_dev`) advertises the EXACT matrix-free
        // profiled outer θ-HVP just like the rigid path: the operator is
        // assembled from the family's directional-derivative kernels and
        // reuses the flex row stream once per Hv (near-gradient cost), so the
        // outer Newton/ARC loop converges in a couple of iterations rather
        // than the several full-inner-resolve BFGS line searches a first-order
        // demotion would pay. The REML/LAML optimum is unchanged; only the
        // outer optimizer geometry improves. When the HVP operator is not
        // available we still fall through to the gradient-only route below.
        if !dense_available && !hvp_available {
            if log_exact_work(self.y.len()) {
                log::info!(
                    "[BMS outer-derivative-policy] n={} p={} flex={} order=First reason=no-outer-hessian dense_available={} outer_hvp_available={} coefficient_work={}",
                    self.y.len(),
                    specs.iter().map(|spec| spec.design.ncols()).sum::<usize>(),
                    flex_active,
                    dense_available,
                    hvp_available,
                    coefficient_work,
                );
            }
            return ExactOuterDerivativeOrder::First;
        }

        let order = crate::custom_family::exact_outer_order_with_outer_hvp(
            specs,
            coefficient_work,
            hvp_available,
        );
        if log_exact_work(self.y.len()) {
            let p_total = specs.iter().map(|spec| spec.design.ncols()).sum::<usize>();
            let matrix_free_inner_requested =
                crate::custom_family::use_joint_matrix_free_path(p_total, self.y.len());
            let inner_route = if matrix_free_inner_requested
                && self.inner_coefficient_hessian_hvp_available(specs)
            {
                "workspace-hvp"
            } else if p_total < 512 {
                "workspace-dense"
            } else if self.inner_coefficient_hessian_hvp_available(specs) {
                "workspace-hvp"
            } else {
                "direct-dense"
            };
            log::info!(
                "[BMS outer-derivative-policy] n={} p={} flex={} order={:?} declared_hessian=analytic inner_route={} matrix_free_inner_requested={} dense_available={} outer_hvp_available={} coefficient_work={}",
                self.y.len(),
                p_total,
                flex_active,
                order,
                inner_route,
                matrix_free_inner_requested,
                dense_available,
                hvp_available,
                coefficient_work,
            );
        }
        order
    }

    fn outer_derivative_policy(
        &self,
        specs: &[ParameterBlockSpec],
        psi_dim: usize,
        options: &BlockwiseFitOptions,
    ) -> crate::custom_family::OuterDerivativePolicy {
        let capability = self.exact_outer_derivative_order(specs, options);
        let grad_cost = self.coefficient_gradient_cost(specs);
        let hess_cost = self.coefficient_hessian_cost(specs);
        let (predicted_gradient_work, predicted_hessian_work) =
            crate::custom_family::default_outer_derivative_policy_costs(
                specs, psi_dim, grad_cost, hess_cost,
            );
        crate::custom_family::OuterDerivativePolicy {
            capability,
            predicted_gradient_work,
            predicted_hessian_work,
            subsample_capable: true,
        }
    }

    fn outer_seed_config(&self, n_params: usize) -> gam_solve::seeding::SeedConfig {
        let mut config = gam_solve::seeding::SeedConfig::default();
        if n_params == 0 {
            return config;
        }
        // #979: BMS startup seed screening runs real inner solves. With the
        // default multi-seed pool, large marginal-slope fits can spend minutes
        // rejecting equivalent seeds before the first outer step. Keep the
        // principled GLM candidate grid alive (the symmetric over-/under-smooth
        // stability anchors at rho={2,4} that startup validation relies on),
        // but budget exactly one screened start so only a single inner solve
        // is paid at startup rather than the full screening cascade.
        config.max_seeds = 6;
        config.seed_budget = 1;
        // Two cycles is below the observed KKT reachability floor for
        // marginal-slope startup seeds: it rejects every candidate, then pays
        // an immediate second screening pass at cap=8. Start at the first
        // viable cap and let the existing cascade escalate only when needed.
        config.screen_max_inner_iterations = 8;
        config
    }

    fn exact_newton_joint_psi_workspace_for_first_order_terms(&self) -> bool {
        true
    }

    fn batched_outer_gradient_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        rho: &Array1<f64>,
        options: &BlockwiseFitOptions,
        hessian_workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
    ) -> Result<Option<BatchedOuterGradientTerms>, String> {
        let psi_dim: usize = derivative_blocks.iter().map(Vec::len).sum();
        if block_states.len() != specs.len() {
            return Ok(None);
        }
        if derivative_blocks.len() != specs.len() {
            return Ok(None);
        }
        for psi_index in 0..psi_dim {
            if self.is_sigma_aux_index(derivative_blocks, psi_index) {
                return Ok(None);
            }
        }
        // Two-phase auto-subsample schedule. Phase 1: stratified
        // Horvitz–Thompson mask (≈ 1 % gradient noise) for the first
        // BMS_AUTO_SUBSAMPLE_PHASE1_BUDGET outer evaluations, where
        // BFGS makes the bulk of its progress on the noisy gradient.
        // Phase 2: full data for every subsequent evaluation, so the
        // optimizer can drive ‖∇‖ below the user's tight `outer_tol`
        // without bumping into the stochastic noise floor. The phase
        // counter is per-family-instance, so each fresh fit (which
        // constructs a new family) starts at Phase 1.
        //
        // The mask is auto-derived only when `options.auto_outer_subsample`
        // is enabled and the caller has not already supplied a mask of their
        // own. All gating logic lives
        // in `maybe_install_auto_outer_subsample` so the survival
        // families can reuse the same schedule.
        let stratum_secondary: Vec<u8> = self
            .y
            .iter()
            .map(|v| if *v > 0.5 { 1u8 } else { 0u8 })
            .collect();
        let owned_options;
        let options: &BlockwiseFitOptions =
            match crate::marginal_slope_shared::maybe_install_auto_outer_subsample(
                options,
                self.z.as_slice().expect("z must be contiguous"),
                Some(&stratum_secondary),
                rho.as_slice().expect("outer rho must be contiguous"),
                &self.auto_subsample_phase_counter,
                &self.auto_subsample_last_rho,
                BMS_AUTO_SUBSAMPLE_PHASE1_BUDGET,
                "BMS",
                // Per-K work-unit cost for the BMS outer gradient kernel.
                // BMS uses a Polya–Gamma augmentation with a per-row
                // pseudo-Bernoulli factorisation that is materially cheaper
                // than the survival marginal-slope risk-set scan. Empirical
                // cost is ~50_000 units per K-unit at large scale, which
                // with `AUTO_OUTER_WORK_BUDGET = 5×10⁸` caps
                //   K_work ≈ 5e8 / 50_000 = 10_000,
                // matching the existing default `min_k = 10_000` and so
                // never binding tighter than the noise rule in current
                // production configurations — the cap exists to guard
                // against pathological per-row cost regressions, not to
                // change today's nominal K.
                50_000,
                30_000,
                10_000,
                1_000,
            ) {
                Some(cloned) => {
                    owned_options = cloned;
                    &owned_options
                }
                None => options,
            };
        let ranges = Self::block_ranges_from_specs(specs);
        let total = ranges.last().map(|(_, end)| *end).unwrap_or(0);
        let theta_dim = rho.len() + psi_dim;
        if total == 0 {
            return Ok(Some(BatchedOuterGradientTerms {
                objective_theta: Array1::zeros(theta_dim),
                trace_h_inv_hdot: Array1::zeros(theta_dim),
                trace_s_pinv_sdot: Array1::zeros(theta_dim),
            }));
        }
        if rho.len() != specs.iter().map(|spec| spec.penalties.len()).sum::<usize>() {
            return Ok(None);
        }
        if total >= 512 {
            return Ok(None);
        }

        let batched_started = std::time::Instant::now();
        let beta = Self::flatten_block_state_betas_for_specs(block_states, specs)?;
        if log_exact_work(self.y.len()) {
            log::info!(
                "[BMS batched outer-gradient] start n={} p={} rho={} subsample_rows={} workspace={}",
                self.y.len(),
                total,
                rho.len(),
                options
                    .outer_score_subsample
                    .as_ref()
                    .map_or(self.y.len(), |subsample| subsample.len()),
                hessian_workspace.is_some()
            );
        }
        let hessian_started = std::time::Instant::now();
        let mut h = if let Some(workspace) = hessian_workspace.as_ref() {
            // Outer batched-gradient assembly genuinely needs the dense
            // joint Hessian (it pulls back `H_β` against an explicit factor),
            // so bypass the workspace's matrix-free inner-route gate via
            // `hessian_dense_forced` — that always materialises through the
            // fused row pass instead of falling through to column-basis HVP.
            workspace.hessian_dense_forced()?.ok_or_else(|| {
                "bernoulli marginal-slope batched gradient requires dense exact joint Hessian below p=512"
                    .to_string()
            })?
        } else {
            self.exact_newton_joint_hessian(block_states)?.ok_or_else(|| {
                "bernoulli marginal-slope batched gradient could not build dense exact joint Hessian"
                    .to_string()
            })?
        };
        if log_exact_work(self.y.len()) {
            log::info!(
                "[BMS batched outer-gradient] dense-hessian ready n={} p={} elapsed={:.3}s",
                self.y.len(),
                total,
                hessian_started.elapsed().as_secs_f64()
            );
        }
        if h.nrows() != total || h.ncols() != total {
            return Err(format!(
                "bernoulli marginal-slope batched gradient Hessian shape {}x{} != {total}x{total}",
                h.nrows(),
                h.ncols()
            ));
        }

        let penalty_started = std::time::Instant::now();
        let ridge = options.ridge_floor.max(1e-15);
        let trace_diagonal_ridge = if options.ridge_policy.include_quadratic_penalty
            || options.ridge_policy.include_penalty_logdet
        {
            ridge
        } else {
            0.0
        };
        let mut objective_theta = Array1::<f64>::zeros(theta_dim);
        let mut trace_s_pinv_sdot = Array1::<f64>::zeros(theta_dim);
        let mut penalty_cursor = 0usize;
        let mut per_block_rho: Vec<Array1<f64>> = Vec::with_capacity(specs.len());
        let mut penalties_dense: Vec<Vec<Array2<f64>>> = Vec::with_capacity(specs.len());
        for (block_idx, spec) in specs.iter().enumerate() {
            let count = spec.penalties.len();
            let block_rho = rho
                .slice(s![penalty_cursor..penalty_cursor + count])
                .to_owned();
            let lambdas = block_rho.mapv(f64::exp);
            per_block_rho.push(block_rho);
            let (start, end) = ranges[block_idx];
            let beta_block = beta.slice(s![start..end]);
            let mut s_lambda = Array2::<f64>::zeros((end - start, end - start));
            let mut block_penalties = Vec::with_capacity(count);
            for (local_idx, penalty) in spec.penalties.iter().enumerate() {
                let dense = penalty.to_dense();
                let lambda = lambdas[local_idx];
                let s_beta = dense.dot(&beta_block);
                objective_theta[penalty_cursor + local_idx] =
                    0.5 * lambda * beta_block.dot(&s_beta);
                s_lambda.scaled_add(lambda, &dense);
                block_penalties.push(dense);
            }
            h.slice_mut(s![start..end, start..end])
                .scaled_add(1.0, &s_lambda);
            penalties_dense.push(block_penalties);
            penalty_cursor += count;
        }
        if trace_diagonal_ridge != 0.0 {
            for diag in 0..total {
                h[[diag, diag]] += trace_diagonal_ridge;
            }
        }

        let penalty_logdet_ridge = if options.ridge_policy.include_penalty_logdet {
            ridge
        } else {
            0.0
        };
        let mut penalty_logdet_blocks = Vec::with_capacity(specs.len());
        penalty_cursor = 0;
        for (block_idx, block_rho) in per_block_rho.iter().enumerate() {
            let lambdas = block_rho.mapv(f64::exp).to_vec();
            let pld = gam_solve::estimate::reml::penalty_logdet::PenaltyPseudologdet::from_components(
                &penalties_dense[block_idx],
                &lambdas,
                penalty_logdet_ridge,
            )
            .map_err(|e| {
                format!("bernoulli marginal-slope penalty logdet failed for block {block_idx}: {e}")
            })?;
            let first = pld.rho_derivatives(&penalties_dense[block_idx], &lambdas).0;
            for (local_idx, value) in first.iter().enumerate() {
                trace_s_pinv_sdot[penalty_cursor + local_idx] = *value;
            }
            penalty_cursor += block_rho.len();
            penalty_logdet_blocks.push(pld);
        }
        if log_exact_work(self.y.len()) {
            log::info!(
                "[BMS batched outer-gradient] penalty assembly/logdet done n={} p={} rho={} elapsed={:.3}s",
                self.y.len(),
                total,
                rho.len(),
                penalty_started.elapsed().as_secs_f64()
            );
        }

        let spectral_started = std::time::Instant::now();
        let spectral =
            DenseSpectralOperator::from_symmetric_with_mode(&h, self.pseudo_logdet_mode())?;
        let factor = spectral.logdet_gradient_factor();
        if log_exact_work(self.y.len()) {
            log::info!(
                "[BMS batched outer-gradient] spectral factor done n={} p={} rank={} elapsed={:.3}s",
                self.y.len(),
                total,
                factor.ncols(),
                spectral_started.elapsed().as_secs_f64()
            );
        }
        let mut trace_h_inv_hdot = Array1::<f64>::zeros(theta_dim);
        let mut directions = Array2::<f64>::zeros((total, theta_dim));
        let direction_started = std::time::Instant::now();
        penalty_cursor = 0;
        for (block_idx, spec) in specs.iter().enumerate() {
            let (start, end) = ranges[block_idx];
            let beta_block = beta.slice(s![start..end]);
            for (local_idx, _penalty) in spec.penalties.iter().enumerate() {
                let idx = penalty_cursor + local_idx;
                let lambda = rho[idx].exp();
                let dense = &penalties_dense[block_idx][local_idx];
                trace_h_inv_hdot[idx] +=
                    spectral.trace_logdet_block_local(dense, lambda, start, end);
                let curvature_rhs = dense.dot(&beta_block).mapv(|value| lambda * value);
                let mut rhs = Array1::<f64>::zeros(total);
                rhs.slice_mut(s![start..end]).assign(&curvature_rhs);
                let v = spectral.solve(&rhs);
                directions.column_mut(idx).assign(&(-&v));
            }
            penalty_cursor += spec.penalties.len();
        }

        let psi_cache = if psi_dim == 0 {
            None
        } else {
            Some(self.build_exact_eval_cache_with_options(block_states, Some(options))?)
        };
        if let Some(cache) = psi_cache.as_ref() {
            let mut axes: Vec<PsiAxisSpec> = Vec::with_capacity(psi_dim);
            let mut psi_locations: Vec<(usize, usize)> = Vec::with_capacity(psi_dim);
            for psi_index in 0..psi_dim {
                let Some((block_idx, local_idx)) =
                    psi_derivative_location(derivative_blocks, psi_index)
                else {
                    return Ok(None);
                };
                axes.push(self.resolve_psi_axis_spec(derivative_blocks, block_idx, local_idx)?);
                psi_locations.push((block_idx, local_idx));
            }
            let psi_terms = self.run_psi_row_pass_for_axes(block_states, cache, options, &axes)?;
            if psi_terms.len() != psi_dim {
                return Err(format!(
                    "bernoulli marginal-slope batched gradient psi terms length {} != psi_dim {psi_dim}",
                    psi_terms.len()
                ));
            }
            // EXPLICIT Firth VALUE ψ-derivative (gam#1607). This batched override is
            // the ValueAndGradient outer-gradient path; the contracted/hypercoord
            // gradient (`build_psi_hyper_coords`) folds `−∂_ψΦ` into each ψ coord's
            // objective, so this path MUST carry the identical term or the batched
            // outer gradient diverges from the hypercoord gradient (and from the
            // centered FD of the now-Firth-corrected value the operator-HVP Hessian
            // is validated against). Built once from the same `(Z_J, H_info)` pieces
            // and gated identically: `None` for non-Jeffreys families and `0.0` when
            // the conditioning gate skips the term, so clean fits are byte-unchanged.
            let jeffreys_hphi_ctx = crate::custom_family::build_jeffreys_hphi_ctx(
                self,
                block_states,
                specs,
                derivative_blocks,
                total,
            )?;
            for (psi_index, ((block_idx, local_idx), terms)) in psi_locations
                .into_iter()
                .zip(psi_terms.into_iter())
                .enumerate()
            {
                let idx = rho.len() + psi_index;
                if terms.score_psi.len() != total {
                    return Err(format!(
                        "bernoulli marginal-slope batched gradient psi score length {} != p {total}",
                        terms.score_psi.len()
                    ));
                }
                if terms.hessian_psi.nrows() > 0
                    && (terms.hessian_psi.nrows() != total || terms.hessian_psi.ncols() != total)
                {
                    return Err(format!(
                        "bernoulli marginal-slope batched gradient psi Hessian shape {}x{} != {total}x{total}",
                        terms.hessian_psi.nrows(),
                        terms.hessian_psi.ncols()
                    ));
                }
                let (start, end) = ranges[block_idx];
                let p_block = end - start;
                let deriv = &derivative_blocks[block_idx][local_idx];
                let s_psi_local =
                    assemble_bms_block_local_s_psi(deriv, &per_block_rho[block_idx], p_block);
                let beta_block = beta.slice(s![start..end]);
                let s_psi_beta_local = s_psi_local.dot(&beta_block);
                objective_theta[idx] =
                    terms.objective_psi + 0.5 * beta_block.dot(&s_psi_beta_local);
                if let Some((z_j, h_joint)) = jeffreys_hphi_ctx.as_ref() {
                    // `∂_ψ H_info|_β`: the explicit ψ-derivative of the Jeffreys
                    // information (the dense ψ-Hessian, or the materialized
                    // operator when the row pass streams it), matching exactly the
                    // perturbation `build_psi_hyper_coords` contracts.
                    let pert_info: Option<Array2<f64>> =
                        if let Some(op) = terms.hessian_psi_operator.as_ref() {
                            Some(op.mul_mat(&Array2::<f64>::eye(total)))
                        } else if terms.hessian_psi.nrows() == total
                            && terms.hessian_psi.ncols() == total
                        {
                            Some(terms.hessian_psi.clone())
                        } else {
                            None
                        };
                    if let Some(pert_info) = pert_info.as_ref() {
                        let phi_psi = gam_solve::estimate::reml::jeffreys_subspace::joint_jeffreys_phi_explicit_param_derivative(
                            h_joint.view(),
                            z_j.view(),
                            pert_info,
                        )?;
                        objective_theta[idx] -= phi_psi;
                    }
                }
                let mut rhs = terms.score_psi.clone();
                {
                    let mut rhs_block = rhs.slice_mut(s![start..end]);
                    rhs_block += &s_psi_beta_local;
                }
                let v = spectral.solve(&rhs);
                directions.column_mut(idx).assign(&(-&v));
                if terms.hessian_psi.nrows() > 0 {
                    trace_h_inv_hdot[idx] += spectral.trace_logdet_gradient(&terms.hessian_psi);
                }
                if let Some(operator) = terms.hessian_psi_operator.as_ref() {
                    trace_h_inv_hdot[idx] += spectral.trace_logdet_operator(operator.as_ref());
                }
                trace_h_inv_hdot[idx] +=
                    spectral.trace_logdet_block_local(&s_psi_local, 1.0, start, end);
                // NOTE (gam#1607): the hypercoord path's Firth ψ-CURVATURE drift
                // `∂_ψ H_Φ` is added ONLY on `build_psi_hyper_coords`' dense drift
                // branch; for this BMS row-pass the ψ-Hessian arrives in OPERATOR form
                // and `build_psi_hyper_coords` takes its operator branch too, which
                // carries NO `∂_ψ H_Φ` term — so this batched override must likewise
                // omit it. The residual `trace_h_inv_hdot` gap between the batched and
                // hypercoord paths (~2e-3 here) is a SEPARATE, non-Firth discrepancy
                // between the two ψ-Hessian term sources (`run_psi_row_pass_for_axes`
                // vs `workspace.first_order_terms`), tracked under #1607 — NOT a missing
                // Firth curvature (adding one drives the gap larger, the wrong way).
                trace_s_pinv_sdot[idx] =
                    penalty_logdet_blocks[block_idx].tau_gradient_component(&s_psi_local);
            }
        }
        if log_exact_work(self.y.len()) {
            log::info!(
                "[BMS batched outer-gradient] direction solves done n={} p={} theta={} rho={} psi={} elapsed={:.3}s",
                self.y.len(),
                total,
                theta_dim,
                rho.len(),
                psi_dim,
                direction_started.elapsed().as_secs_f64()
            );
        }

        let started = std::time::Instant::now();
        // The workspace's projected trace path is full-data only and
        // does not consult `options.outer_score_subsample`. When a
        // subsample is active (auto-derived above or explicitly
        // supplied) we route through the family-side fallback which
        // honors the mask; otherwise the workspace fast path is fine.
        let workspace_traces = if options.outer_score_subsample.is_some() {
            None
        } else if let Some(workspace) = hessian_workspace.as_ref() {
            workspace.projected_directional_derivative_traces(factor, &directions)?
        } else {
            None
        };
        let correction_traces = if let Some(traces) = workspace_traces {
            traces
        } else {
            let owned_cache = if let Some(subsample) = options.outer_score_subsample.as_ref() {
                self.build_exact_eval_cache_for_selected_context_rows(
                    block_states,
                    options,
                    subsample.mask.as_slice(),
                )?
            } else if let Some(cache) = psi_cache {
                cache
            } else {
                self.build_exact_eval_cache_with_options(block_states, Some(options))?
            };
            if options.outer_score_subsample.is_some() {
                let weighted_rows = outer_weighted_rows(options, self.y.len());
                self.batched_rho_correction_logdet_traces_for_rows(
                    block_states,
                    &owned_cache,
                    factor,
                    &directions,
                    &weighted_rows,
                )?
            } else {
                self.batched_rho_correction_logdet_traces_full_rows(
                    block_states,
                    &owned_cache,
                    factor,
                    &directions,
                )?
            }
        };
        trace_h_inv_hdot += &correction_traces;
        if log_exact_work(self.y.len()) {
            log::info!(
                "[BMS batched outer-gradient] done n={} p={} theta={} rho={} psi={} trace_elapsed={:.3}s total_elapsed={:.3}s",
                self.y.len(),
                total,
                theta_dim,
                rho.len(),
                psi_dim,
                started.elapsed().as_secs_f64(),
                batched_started.elapsed().as_secs_f64()
            );
        }

        Ok(Some(BatchedOuterGradientTerms {
            objective_theta,
            trace_h_inv_hdot,
            trace_s_pinv_sdot,
        }))
    }

    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        self.validate_exact_monotonicity(block_states)?;
        self.evaluate_blockwise_exact_newton(block_states)
    }

    fn log_likelihood_only(&self, block_states: &[ParameterBlockState]) -> Result<f64, String> {
        Self::log_likelihood_only_with_options(self, block_states, &BlockwiseFitOptions::default())
    }

    /// Options-aware override: outer hyper-derivative callers may pass a row
    /// subsample via `outer_score_subsample`. Coefficient inner line searches
    /// inherit that caller-supplied row measure unchanged (preserving trust-region
    /// ratio consistency) and disable auto-install so no fresh mask can fire
    /// mid-iteration; the early-exit monotone lower-bound proof holds for any
    /// row set with non-negative weights, so the partial-NLL threshold is sound
    /// regardless of whether the inherited measure is full-data or subsampled.
    fn log_likelihood_only_with_options(
        &self,
        block_states: &[ParameterBlockState],
        options: &BlockwiseFitOptions,
    ) -> Result<f64, String> {
        Self::log_likelihood_only_with_options(self, block_states, options)
    }

    fn supports_log_likelihood_early_exit(&self) -> bool {
        true
    }

    fn has_explicit_joint_hessian(&self) -> bool {
        true
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        let slices = block_slices(self);
        if slices.total >= 512 {
            return Ok(None);
        }
        if !self.effective_flex_active(block_states)? {
            let kern = BernoulliRigidRowKernel::new(self.clone(), block_states.to_vec());
            let cache = build_row_kernel_cache(&kern, &crate::row_kernel::RowSet::All)?;
            return Ok(Some(row_kernel_hessian_dense(
                &kern,
                &cache,
                &crate::row_kernel::RowSet::All,
            )));
        }

        // Build the dense joint Hessian by accumulating per-row primary
        // Hessians once per row and pulling each one back through the design
        // matrices via `BernoulliBlockHessianAccumulator::add_pullback`. The
        // earlier implementation materialized the dense matrix column-by-column
        // by calling `exact_newton_joint_hessian_matvec_from_cache` once per
        // unit basis vector, and each matvec re-ran
        // `compute_row_analytic_flex_into` (cell partition + cell-moment
        // evaluation + basis evaluation) for every row. That made the dense
        // build cost `slices.total * n * per_row_cell_work`. The flex
        // per-row work is direction-independent, so the column loop was
        // recomputing the same primary-space Hessian `slices.total` times.
        // For the bern_scorewarp / bern_sw_frailty integration cases the
        // outer factor of `slices.total ≈ 35` blew the optimizer past the
        // 240s timeout even on a 300-row dataset.
        //
        // The single-pass form below evaluates the per-row primary Hessian
        // exactly once per row and lets the existing accumulator distribute
        // every (a, b) entry into the marginal/logslope/h/w block targets.
        let cache = self.build_exact_eval_cache_with_order(block_states)?;
        self.exact_newton_joint_hessian_dense_from_cache(block_states, &cache)
            .map(Some)
    }

    fn exact_newton_joint_gradient_evaluation(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
    ) -> Result<Option<ExactNewtonJointGradientEvaluation>, String> {
        self.validate_exact_monotonicity(block_states)?;
        if !self.effective_flex_active(block_states)? {
            let kern = BernoulliRigidRowKernel::new(self.clone(), block_states.to_vec());
            let cache = build_row_kernel_cache(&kern, &crate::row_kernel::RowSet::All)?;
            return Ok(Some(ExactNewtonJointGradientEvaluation {
                log_likelihood: row_kernel_log_likelihood(
                    &cache,
                    &crate::row_kernel::RowSet::All,
                ),
                gradient: Self::exact_newton_score_from_objective_gradient(row_kernel_gradient(
                    &kern,
                    &cache,
                    &crate::row_kernel::RowSet::All,
                )),
            }));
        }

        let cache = self.build_exact_eval_cache_with_order(block_states)?;
        self.exact_newton_joint_gradient_evaluation_from_cache(block_states, &cache)
            .map(Some)
    }

    fn requires_joint_outer_hyper_path(&self) -> bool {
        true
    }

    fn exact_newton_joint_hessian_workspace(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        if !self.effective_flex_active(block_states)? {
            // Rigid path: use generic RowKernel<2> operator
            let kern = BernoulliRigidRowKernel::new(self.clone(), block_states.to_vec());
            Ok(Some(Arc::new(RowKernelHessianWorkspace::new(kern)?)))
        } else {
            // Flex path: keep existing workspace for variable-K primary space
            Ok(Some(Arc::new(
                BernoulliMarginalSlopeExactNewtonJointHessianWorkspace::new(
                    self.clone(),
                    block_states.to_vec(),
                    BlockwiseFitOptions::default(),
                )?,
            )))
        }
    }

    fn exact_newton_joint_hessian_workspace_with_options(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        if !self.effective_flex_active(block_states)? {
            // Rigid path: RowKernel<2> operator wired through the supplied
            // `RowSet`. With no outer subsample this is `RowSet::All`
            // (full-data, bit-identical to the pre-threading behaviour);
            // with an outer subsample installed, the cache and every
            // assembly function honour it uniformly via the Horvitz–Thompson
            // weights carried on each `WeightedOuterRow`.
            let kern = BernoulliRigidRowKernel::new(self.clone(), block_states.to_vec());
            let rows = crate::row_kernel::row_set_from_options(options, self.y.len());
            Ok(Some(Arc::new(RowKernelHessianWorkspace::with_rows(
                kern, rows,
            )?)))
        } else {
            // Flex path: store the options on the workspace so the
            // directional-derivative methods route through the
            // `_with_options` helpers and pick up the outer-row subsample.
            Ok(Some(Arc::new(
                BernoulliMarginalSlopeExactNewtonJointHessianWorkspace::new(
                    self.clone(),
                    block_states.to_vec(),
                    options.clone(),
                )?,
            )))
        }
    }

    fn inner_coefficient_hessian_hvp_available(&self, specs: &[ParameterBlockSpec]) -> bool {
        // The workspace impl above unconditionally returns `Some(workspace)`
        // — the rigid path produces a `RowKernelHessianWorkspace` and the
        // flex path produces a
        // `BernoulliMarginalSlopeExactNewtonJointHessianWorkspace`. Both
        // route the joint Hessian through Hv operators rather than dense
        // assembly. This advertises β-space representation support only; it
        // does not imply a profiled outer θ-HVP.
        parameter_block_specs_match_rows(specs, self.y.len())
    }

    fn outer_hyper_hessian_hvp_available(&self, specs: &[ParameterBlockSpec]) -> bool {
        // Binary twin of `SurvivalMarginalSlopeFamily`: the exact profiled
        // outer Hessian over θ=(ρ,ψ[,log σ]) is applied matrix-free, without
        // pairwise θθ materialization. The coefficient-Hessian drift terms
        // (D_β H[u_k]) come from `exact_newton_joint_hessian_directional_derivative`
        // and `...second_directional_derivative`; the ψ-drift terms come from
        // `exact_newton_joint_psi_terms`; the joint mode response H u_k = −g_k
        // is solved through the inner Hv workspace. The generic exact
        // joint-hyper assembler in `custom_family::outer_objective` consumes
        // these directional kernels to produce an
        // `OuterHessianOperator`, so advertising this capability lifts the
        // outer-derivative order to `Second` and routes the planner to ARC /
        // exact outer Newton instead of BFGS — converging the outer ρ-loop in
        // ≤ a couple of iterations rather than several BFGS line searches, each
        // of which costs a full inner re-solve. The operator is the EXACT
        // analytic θ-HVP (no finite differences, no quasi-Newton surrogate),
        // so the REML/LAML optimum is bit-identical to the first-order path.
        parameter_block_specs_match_rows(specs, self.y.len())
    }

    fn inner_joint_workspace_gradient_available(&self, specs: &[ParameterBlockSpec]) -> bool {
        parameter_block_specs_match_rows(specs, self.y.len())
    }

    fn inner_joint_workspace_log_likelihood_available(&self, specs: &[ParameterBlockSpec]) -> bool {
        parameter_block_specs_match_rows(specs, self.y.len())
    }

    /// Force the matrix-free inner-Newton/PCG path for BMS flex at large-scale
    /// scale, on top of the generic `use_joint_matrix_free_path` heuristic.
    ///
    /// At n≈195k with `linkwiggle()` / `logslope_formula linkwiggle()`, dense
    /// joint-H assembly streams every row through the expensive flex row
    /// kernel and pays a BLAS-3 design-matrix gram per chunk on top — ~63s
    /// per inner cycle. Each HVP reuses the row stream at near-gradient cost
    /// (~3s), and PCG with the joint penalty preconditioner typically
    /// converges in a handful of iters. The generic gate only fires for
    /// `p >= 128`, but BMS-flex per-row work is heavy enough that the matrix-
    /// free path wins well below that — drop the `p` floor for this family.
    fn prefers_matrix_free_inner_joint(
        &self,
        _: &[ParameterBlockSpec],
        states: &[ParameterBlockState],
    ) -> bool {
        if self.y.len() < 16_384 {
            return false;
        }
        self.effective_flex_active(states).unwrap_or(false)
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if !self.effective_flex_active(block_states)? {
            let kern = BernoulliRigidRowKernel::new(self.clone(), block_states.to_vec());
            let sl = d_beta_flat.as_slice().ok_or("non-contiguous d_beta")?;
            crate::row_kernel::row_kernel_directional_derivative(
                &kern,
                &crate::row_kernel::RowSet::All,
                sl,
            )
            .map(Some)
        } else {
            let cache = self.build_exact_eval_cache(block_states)?;
            self.exact_newton_joint_hessian_directional_derivative_from_cache(
                block_states,
                d_beta_flat,
                &cache,
            )
        }
    }

    fn exact_newton_joint_hessiansecond_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if !self.effective_flex_active(block_states)? {
            let kern = BernoulliRigidRowKernel::new(self.clone(), block_states.to_vec());
            let su = d_beta_u_flat.as_slice().ok_or("non-contiguous d_beta_u")?;
            let sv = d_beta_v_flat.as_slice().ok_or("non-contiguous d_beta_v")?;
            crate::row_kernel::row_kernel_second_directional_derivative(
                &kern,
                &crate::row_kernel::RowSet::All,
                su,
                sv,
            )
            .map(Some)
        } else {
            let cache = self.build_exact_eval_cache(block_states)?;
            self.exact_newton_joint_hessiansecond_directional_derivative_from_cache(
                block_states,
                d_beta_u_flat,
                d_beta_v_flat,
                &cache,
            )
        }
    }

    fn joint_jeffreys_information_directional_derivative_all_axes_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Vec<Array2<f64>>>, String> {
        // Same trust dispatch as the per-axis
        // `exact_newton_joint_hessian_directional_derivative_with_specs`: an
        // untrusted, structurally-decoupled spec returns `None` so the caller
        // keeps the released semantics (the inner Jeffreys term then degenerates
        // to its value-only contribution), exactly as the per-axis path.
        if !self.outer_default_trustworthy_for_joint_hessian(specs)
            && !self.joint_hessian_is_structurally_coupled(block_states)?
        {
            return Ok(None);
        }
        // Flex-active fits use the dense exact-cache path; only the rigid
        // marginal-slope kernel has the BLAS-3 design-row-Gram batched override.
        // Fall back to the generic per-axis assembly (bit-for-bit identical to
        // the trait default) when flex is active.
        if self.effective_flex_active(block_states)? {
            let p = specs.iter().map(|spec| spec.design.ncols()).sum::<usize>();
            let mut axes = Vec::with_capacity(p);
            for a in 0..p {
                let mut axis = Array1::<f64>::zeros(p);
                axis[a] = 1.0;
                match self.joint_jeffreys_information_directional_derivative_with_specs(
                    block_states,
                    specs,
                    &axis,
                )? {
                    Some(m) => axes.push(m),
                    None => return Ok(None),
                }
            }
            return Ok(Some(axes));
        }
        let kern = BernoulliRigidRowKernel::new(self.clone(), block_states.to_vec());
        // The dispatcher takes the BLAS-3 override when available and otherwise
        // runs the generic per-axis sweep — both bit-faithful to the per-axis
        // `exact_newton_joint_hessian_directional_derivative` the default calls.
        crate::row_kernel::row_kernel_directional_derivative_all_axes(
            &kern,
            &crate::row_kernel::RowSet::All,
        )
        .map(Some)
    }

    fn joint_jeffreys_information_second_directional_all_axes_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_u_flat: &Array1<f64>,
    ) -> Result<Option<Vec<Array2<f64>>>, String> {
        // Same trust dispatch as the per-axis
        // `exact_newton_joint_hessian_second_directional_derivative_with_specs`:
        // an untrusted, structurally-decoupled spec returns `None` so the caller
        // keeps the released zero-drift semantics, exactly as the per-axis path.
        if !self.outer_default_trustworthy_for_joint_hessian(specs)
            && !self.joint_hessian_is_structurally_coupled(block_states)?
        {
            return Ok(None);
        }
        // Flex-active fits use the dense exact-cache path; only the rigid
        // marginal-slope kernel has the BLAS-3 design-row-Gram batched override.
        // Fall back to the generic per-axis assembly (bit-for-bit identical to the
        // trait default) when flex is active.
        if self.effective_flex_active(block_states)? {
            let p = specs.iter().map(|spec| spec.design.ncols()).sum::<usize>();
            let mut axes = Vec::with_capacity(p);
            for a in 0..p {
                let mut axis = Array1::<f64>::zeros(p);
                axis[a] = 1.0;
                match self.joint_jeffreys_information_second_directional_derivative_with_specs(
                    block_states,
                    specs,
                    d_beta_u_flat,
                    &axis,
                )? {
                    Some(m) => axes.push(m),
                    None => return Ok(None),
                }
            }
            return Ok(Some(axes));
        }
        let kern = BernoulliRigidRowKernel::new(self.clone(), block_states.to_vec());
        let su = d_beta_u_flat
            .as_slice()
            .ok_or("non-contiguous d_beta_u for batched all-axes second directional")?;
        // The dispatcher takes the BLAS-3 override when available and otherwise
        // runs the generic per-axis sweep — both bit-faithful to the per-axis
        // `exact_newton_joint_hessiansecond_directional_derivative` the default
        // would call.
        crate::row_kernel::row_kernel_second_directional_derivative_all_axes(
            &kern,
            &crate::row_kernel::RowSet::All,
            su,
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
        let cache = self.build_exact_eval_cache(block_states)?;
        self.exact_newton_joint_psi_terms_from_cache(
            block_states,
            derivative_blocks,
            psi_index,
            &cache,
        )
    }

    fn exact_newton_joint_psisecond_order_terms(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        if self.is_sigma_aux_index(derivative_blocks, psi_i)
            || self.is_sigma_aux_index(derivative_blocks, psi_j)
        {
            if self.is_sigma_aux_index(derivative_blocks, psi_i)
                && self.is_sigma_aux_index(derivative_blocks, psi_j)
            {
                return self.sigma_exact_joint_psisecond_order_terms(block_states);
            }
            return Err("bernoulli marginal-slope mixed log-sigma/spatial psi second derivatives require cross auxiliary terms; only pure log-sigma second derivatives are supported"
                        .to_string());
        }
        let cache = self.build_exact_eval_cache(block_states)?;
        self.exact_newton_joint_psisecond_order_terms_from_cache(
            block_states,
            derivative_blocks,
            psi_i,
            psi_j,
            &cache,
        )
    }

    fn exact_newton_joint_psihessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if self.is_sigma_aux_index(derivative_blocks, psi_index) {
            return self
                .sigma_exact_joint_psihessian_directional_derivative(block_states, d_beta_flat);
        }
        let cache = self.build_exact_eval_cache(block_states)?;
        self.exact_newton_joint_psihessian_directional_derivative_from_cache(
            block_states,
            derivative_blocks,
            psi_index,
            d_beta_flat,
            &cache,
        )
    }

    fn exact_newton_joint_psi_workspace(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
    ) -> Result<Option<Arc<dyn ExactNewtonJointPsiWorkspace>>, String> {
        Ok(Some(Arc::new(
            crate::marginal_slope_shared::MarginalSlopeExactNewtonPsiWorkspace::new(
                BernoulliMarginalSlopeExactNewtonJointPsiWorkspace::new(
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
        Ok(Some(Arc::new(
            crate::marginal_slope_shared::MarginalSlopeExactNewtonPsiWorkspace::new(
                BernoulliMarginalSlopeExactNewtonJointPsiWorkspace::new(
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
        spec: &ParameterBlockSpec,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        if block_states.len() == usize::MAX
            || block_idx == usize::MAX
            || spec.design.ncols() == usize::MAX
        {
            return Err("unreachable bernoulli marginal-slope constraint state".to_string());
        }
        if self.score_block_index().is_some_and(|idx| block_idx == idx) {
            return Ok(self
                .score_warp
                .as_ref()
                .map(DeviationRuntime::structural_monotonicity_constraints));
        }
        if self.link_block_index().is_some_and(|idx| block_idx == idx) {
            return Ok(self
                .link_dev
                .as_ref()
                .map(DeviationRuntime::structural_monotonicity_constraints));
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
        self.validate_exact_block_state_shapes(block_states)?;
        if block_idx >= block_states.len() {
            return Err(format!(
                "post-update block index {} out of range for {} blocks",
                block_idx,
                block_states.len()
            ));
        }
        if self.score_block_index().is_some_and(|idx| block_idx == idx)
            && let (Some(runtime), Some(score)) =
                (&self.score_warp, self.score_block_state(block_states)?)
        {
            let current = &score.beta;
            if current.len() != beta.len() {
                return Err(format!(
                    "score-warp post-update beta length mismatch: current={}, proposed={}",
                    current.len(),
                    beta.len()
                ));
            }
            validate_monotone_structural_feasible(runtime, current, "score_warp_dev current")?;
            validate_monotone_structural_feasible(runtime, &beta, "score_warp_dev proposed")?;
            return Ok(beta);
        }
        if self.link_block_index().is_some_and(|idx| block_idx == idx)
            && let (Some(runtime), Some(link)) =
                (&self.link_dev, self.link_block_state(block_states)?)
        {
            let current = &link.beta;
            if current.len() != beta.len() {
                return Err(format!(
                    "link-deviation post-update beta length mismatch: current={}, proposed={}",
                    current.len(),
                    beta.len()
                ));
            }
            validate_monotone_structural_feasible(runtime, current, "link_dev current")?;
            validate_monotone_structural_feasible(runtime, &beta, "link_dev proposed")?;
            return Ok(beta);
        }
        Ok(beta)
    }
}

impl BernoulliMarginalSlopeExactNewtonJointHessianWorkspace {
    pub(super) fn new(
        family: BernoulliMarginalSlopeFamily,
        block_states: Vec<ParameterBlockState>,
        options: BlockwiseFitOptions,
    ) -> Result<Self, String> {
        let started = std::time::Instant::now();
        let process_monitor_guard = gam_runtime::process_monitor::track_scope(format!(
            "BMS Hessian-workspace build n={} p={} subsample_rows={}",
            family.y.len(),
            block_slices(&family).total,
            options
                .outer_score_subsample
                .as_ref()
                .map_or(family.y.len(), |subsample| subsample.len())
        ));
        if log_exact_work(family.y.len()) {
            log::info!(
                "[BMS Hessian-workspace] build start n={} p={} subsample_rows={}",
                family.y.len(),
                block_slices(&family).total,
                options
                    .outer_score_subsample
                    .as_ref()
                    .map_or(family.y.len(), |subsample| subsample.len())
            );
        }
        // Build (or reuse, at a bit-identical β) the exact-cache with per-row
        // primary Hessians materialized at construction time. The matrix-free
        // CG / inner-Newton loops contract these against many trial directions
        // at the same β, so caching the `r×r` blocks once amortizes the
        // cell-moment + flex-jet rebuild over every Hv product. The same-β
        // reuse store additionally elides the whole O(n·cells) build when the
        // outer loop revisits an already-evaluated β̂ (Value→ValueAndGradient at
        // one ρ, or a line-search ρ that maps back to a seen β̂).
        let cache = family.build_or_reuse_shared_exact_cache(&block_states, &options, true)?;
        if log_exact_work(family.y.len()) {
            log::info!(
                "[BMS Hessian-workspace] build done n={} p={} primary_hessian_cache={} elapsed={:.3}s",
                family.y.len(),
                cache.slices.total,
                cache.row_primary_hessians.is_some(),
                started.elapsed().as_secs_f64()
            );
        }
        let workspace = Self::from_arc_cache(family, block_states, cache, options);
        drop(process_monitor_guard);
        workspace
    }

    pub(super) fn from_arc_cache(
        family: BernoulliMarginalSlopeFamily,
        block_states: Vec<ParameterBlockState>,
        cache: Arc<BernoulliMarginalSlopeExactEvalCache>,
        options: BlockwiseFitOptions,
    ) -> Result<Self, String> {
        Ok(Self {
            family,
            block_states,
            cache,
            matvec_calls: AtomicUsize::new(0),
            fused_gradient_dense: OnceLock::new(),
            options,
        })
    }

    pub(super) fn fused_gradient_dense(
        &self,
    ) -> Result<Arc<ExactNewtonJointFusedDenseEvaluation>, String> {
        self.fused_gradient_dense
            .get_or_init(|| {
                self.family
                    .exact_newton_joint_fused_gradient_dense_from_cache(
                        &self.block_states,
                        &self.cache,
                    )
                    .map(Arc::new)
            })
            .clone()
    }

    /// Matrix-free inner-Newton/CG route for BMS flex large-n.
    ///
    /// Auto-selected when the workspace's per-row primary Hessian cache could
    /// not be materialized (`n*r*r*8 > row_primary_cache_budget`). In that
    /// regime the dense joint-H build streams all `n` rows and pays the full
    /// flex row-kernel cost per chunk plus a BLAS-3 design-matrix gram on top;
    /// at large-scale shape (n≈195k, p≈44) that pushes one dense build past 60s
    /// while each HVP reuses the same row stream at ~gradient-pass cost (~3s).
    /// PCG with the joint penalty preconditioner typically converges in a
    /// handful of HVPs, so routing the inner solve through the operator path
    /// beats per-cycle dense reassembly.
    pub(super) fn matrix_free_inner_route(&self) -> bool {
        if self.cache.row_primary_hessians.is_tiled() {
            return true;
        }
        if self.cache.row_primary_hessians.is_some() {
            return false;
        }
        match self.family.effective_flex_active(&self.block_states) {
            Ok(true) => {}
            _ => return false,
        }
        // Tiny problems should still take the dense path: a single dense build
        // is cheaper than CG bookkeeping when row counts are small.
        self.family.y.len() >= 16_384
    }
}

impl ExactNewtonJointHessianWorkspace for BernoulliMarginalSlopeExactNewtonJointHessianWorkspace {
    fn hessian_dense(&self) -> Result<Option<Array2<f64>>, String> {
        if self.cache.slices.total >= 512 {
            return Ok(None);
        }
        if self.matrix_free_inner_route() {
            // Route the inner-Newton solve through `hessian_matvec` /
            // `hessian_diagonal`. Callers that strictly need a dense matrix
            // (outer-Hessian assembly, logdet factor) reach the fused dense
            // build through `hessian_dense_forced`.
            if log_exact_work(self.family.y.len()) {
                log::info!(
                    "[BMS inner] route=matrix-free-CG n={} p={} primary_hessian_cache=false reason=flex+large-n",
                    self.family.y.len(),
                    self.cache.slices.total
                );
            }
            return Ok(None);
        }
        // Device dense-block shortcut: when the row-primary Hessian is pinned on
        // the GPU and `p_total` fits the shared-memory cap, dispatch the
        // device-resident dense build instead of the CPU fused gradient pass.
        #[cfg(target_os = "linux")]
        {
            if let Some(device_state) = self.cache.row_primary_hessians.device() {
                let p_total = self.cache.slices.total;
                if p_total <= crate::bms::gpu::row::DENSE_BLOCK_MAX_P {
                    match crate::bms::gpu::row::launch_bms_flex_row_dense_block(
                        device_state,
                    ) {
                        Ok(flat) => {
                            let h_arr =
                                Array2::from_shape_vec((p_total, p_total), flat).map_err(|e| {
                                    format!(
                                        "BMS hessian_dense: dense_block reshape \
                                             {p_total}x{p_total} failed: {e}"
                                    )
                                })?;
                            return Ok(Some(h_arr));
                        }
                        Err(err) => {
                            log::info!(
                                "[BMS hessian_dense] gpu_dense_block_failed: {err}; \
                                 falling back to CPU fused-gradient dense build"
                            );
                        }
                    }
                }
            }
        }
        if log_exact_work(self.family.y.len()) {
            log::info!(
                "[BMS inner] route=dense n={} p={} primary_hessian_cache={}",
                self.family.y.len(),
                self.cache.slices.total,
                self.cache.row_primary_hessians.is_some()
            );
        }
        self.fused_gradient_dense()
            .map(|fused| Some(fused.hessian.clone()))
    }

    fn hessian_source_preference(&self) -> crate::custom_family::JointHessianSourcePreference {
        if self.matrix_free_inner_route() {
            crate::custom_family::JointHessianSourcePreference::Operator
        } else {
            crate::custom_family::JointHessianSourcePreference::Dense
        }
    }

    fn hessian_dense_forced(&self) -> Result<Option<Array2<f64>>, String> {
        // Callers that genuinely require a dense joint Hessian (e.g. outer
        // batched-gradient assembly that pulls back the dense `H_β`, or the LAML
        // logdet factorization in `MatrixFreeSpdOperator::materialize_dense_
        // operator`) bypass the matrix-free route gate above. The fused row pass
        // is the structural direct-dense path here: it streams every row exactly
        // ONCE and uses BLAS-3 for the `XᵀWX` pullback — O(n·p²) total. The only
        // alternative for a consumer that needs the full dense matrix is
        // canonical-basis reconstruction `H·I`, which re-walks all `n` rows once
        // per column = `total` full-n passes = O(total·n·p). One pass beats
        // `total` passes at every scale, so there is NO size at which the matvec
        // reconstruction wins; the previous `total >= 512` early-return forced
        // exactly that losing path for the large-p outer logdet (the biobank BMS
        // rigid/flex fit at p in the hundreds). The matrix-free INNER solve never
        // calls `hessian_dense_forced` (it pulls HVPs through `hessian_matvec`),
        // so serving the structural one-pass build here does not regress the
        // inner solve it was designed to keep matrix-free.
        // shared-memory cap, dispatch the device-resident dense build
        // instead of round-tripping through the CPU fused gradient pass.
        // Numerics match the CPU pullback to within reduction-order f.p.
        // noise (see `bms_flex_row_dense_block_kernel_matches_cpu_pullback`).
        #[cfg(target_os = "linux")]
        {
            if let Some(device_state) = self.cache.row_primary_hessians.device() {
                let p_total = self.cache.slices.total;
                if p_total <= crate::bms::gpu::row::DENSE_BLOCK_MAX_P {
                    match crate::bms::gpu::row::launch_bms_flex_row_dense_block(
                        device_state,
                    ) {
                        Ok(flat) => {
                            let h_arr =
                                Array2::from_shape_vec((p_total, p_total), flat).map_err(|e| {
                                    format!(
                                        "BMS hessian_dense_forced: dense_block reshape \
                                             {p_total}x{p_total} failed: {e}"
                                    )
                                })?;
                            return Ok(Some(h_arr));
                        }
                        Err(err) => {
                            log::info!(
                                "[BMS hessian_dense_forced] gpu_dense_block_failed: {err}; \
                                 falling back to CPU fused-gradient dense build"
                            );
                        }
                    }
                }
            }
        }
        self.fused_gradient_dense()
            .map(|fused| Some(fused.hessian.clone()))
    }

    fn joint_log_likelihood_evaluation(&self) -> Result<Option<f64>, String> {
        self.family
            .log_likelihood_from_exact_cache(&self.block_states, &self.cache)
            .map(Some)
    }

    fn joint_gradient_evaluation(
        &self,
    ) -> Result<Option<ExactNewtonJointGradientEvaluation>, String> {
        if self.cache.slices.total < 512 && !self.matrix_free_inner_route() {
            // The only current consumer of workspace-side joint gradients is
            // the exact joint-Newton path. For bounded dense systems it will
            // request the dense Hessian in the same cycle, so build the fused
            // row pass once and let `hessian_dense` reuse it. When the
            // matrix-free inner route is active, the inner solver will pull
            // the Hessian through HVPs instead, so the gradient pass must
            // stay gradient-only and not force a dense build.
            return self.fused_gradient_dense().map(|fused| {
                Some(ExactNewtonJointGradientEvaluation {
                    log_likelihood: fused.gradient.log_likelihood,
                    gradient: fused.gradient.gradient.clone(),
                })
            });
        }
        self.family
            .exact_newton_joint_gradient_evaluation_from_cache(&self.block_states, &self.cache)
            .map(Some)
    }

    fn hessian_matvec_available(&self) -> bool {
        true
    }

    fn hessian_matvec(&self, beta_flat: &Array1<f64>) -> Result<Option<Array1<f64>>, String> {
        // Hv-action against the full coefficient Hessian is consumed by inner
        // PCG / inner-Newton paths (not outer-only score), so the row mask
        // does not apply here — keep full-data semantics.
        let call = self.matvec_calls.fetch_add(1, Ordering::Relaxed) + 1;
        let started = std::time::Instant::now();
        let process_monitor_guard = gam_runtime::process_monitor::track_scope(format!(
            "BMS Hessian-Hv call={call} n={} p={}",
            self.family.y.len(),
            self.cache.slices.total
        ));
        let result = self
            .family
            .exact_newton_joint_hessian_matvec_from_cache(
                beta_flat,
                &self.block_states,
                &self.cache,
            )
            .map(Some);
        if log_exact_work(self.family.y.len()) && (call <= 3 || call.is_power_of_two()) {
            log::info!(
                "[BMS Hessian-Hv] call={} n={} p={} primary_hessian_cache={} elapsed={:.3}s",
                call,
                self.family.y.len(),
                self.cache.slices.total,
                self.cache.row_primary_hessians.is_some(),
                started.elapsed().as_secs_f64()
            );
        }
        drop(process_monitor_guard);
        result
    }

    fn hessian_matvec_into(&self, v: &Array1<f64>, out: &mut Array1<f64>) -> Result<bool, String> {
        let call = self.matvec_calls.fetch_add(1, Ordering::Relaxed) + 1;
        let started = std::time::Instant::now();
        let process_monitor_guard = gam_runtime::process_monitor::track_scope(format!(
            "BMS Hessian-Hv (into) call={call} n={} p={}",
            self.family.y.len(),
            self.cache.slices.total
        ));
        self.family
            .exact_newton_joint_hessian_matvec_from_cache_into(
                v,
                &self.block_states,
                &self.cache,
                out,
            )?;
        if log_exact_work(self.family.y.len()) && (call <= 3 || call.is_power_of_two()) {
            log::info!(
                "[BMS Hessian-Hv] call={} n={} p={} primary_hessian_cache={} elapsed={:.3}s (into)",
                call,
                self.family.y.len(),
                self.cache.slices.total,
                self.cache.row_primary_hessians.is_some(),
                started.elapsed().as_secs_f64()
            );
        }
        drop(process_monitor_guard);
        Ok(true)
    }

    fn hessian_apply_mat(
        &self,
        v_cols: &Array2<f64>,
        out: &mut Array2<f64>,
    ) -> Result<bool, String> {
        let total = self.cache.slices.total;
        if v_cols.nrows() != total || out.nrows() != total {
            return Err(format!(
                "BMS hessian_apply_mat: row mismatch v_cols={}x{} out={}x{} expected rows={total}",
                v_cols.nrows(),
                v_cols.ncols(),
                out.nrows(),
                out.ncols()
            ));
        }
        if v_cols.ncols() != out.ncols() {
            return Err(format!(
                "BMS hessian_apply_mat: column mismatch v_cols has {} columns, out has {}",
                v_cols.ncols(),
                out.ncols()
            ));
        }
        let call = self.matvec_calls.fetch_add(1, Ordering::Relaxed) + 1;
        let started = std::time::Instant::now();
        let process_monitor_guard = gam_runtime::process_monitor::track_scope(format!(
            "BMS Hessian-Hv (mat) call={call} n={} p={} n_rhs={}",
            self.family.y.len(),
            total,
            v_cols.ncols()
        ));
        self.family
            .exact_newton_joint_hessian_matvec_mat_from_cache_into(
                v_cols,
                &self.block_states,
                &self.cache,
                out,
            )?;
        if log_exact_work(self.family.y.len()) && (call <= 3 || call.is_power_of_two()) {
            log::info!(
                "[BMS Hessian-Hv] call={} n={} p={} n_rhs={} primary_hessian_cache={} elapsed={:.3}s (mat)",
                call,
                self.family.y.len(),
                total,
                v_cols.ncols(),
                self.cache.row_primary_hessians.is_some(),
                started.elapsed().as_secs_f64()
            );
        }
        drop(process_monitor_guard);
        Ok(true)
    }

    fn hessian_diagonal(&self) -> Result<Option<Array1<f64>>, String> {
        // Diagonal is consumed by inner preconditioners; full-data semantics
        // preserved (see `hessian_matvec`).
        self.family
            .exact_newton_joint_hessian_diagonal_from_cache(&self.block_states, &self.cache)
            .map(Some)
    }

    fn projected_directional_derivative_traces(
        &self,
        factor: &Array2<f64>,
        directions: &Array2<f64>,
    ) -> Result<Option<Array1<f64>>, String> {
        let traces = if self.options.outer_score_subsample.is_some() {
            let weighted_rows = outer_weighted_rows(&self.options, self.family.y.len());
            self.family.batched_rho_correction_logdet_traces_for_rows(
                &self.block_states,
                &self.cache,
                factor,
                directions,
                &weighted_rows,
            )?
        } else {
            self.family.batched_rho_correction_logdet_traces_full_rows(
                &self.block_states,
                &self.cache,
                factor,
                directions,
            )?
        };
        Ok(Some(traces))
    }

    fn directional_derivative_operator(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn HyperOperator>>, String> {
        self.family
            .exact_newton_joint_hessian_directional_derivative_operator_from_cache_with_options(
                &self.block_states,
                d_beta_flat,
                &self.cache,
                &self.options,
            )
    }

    fn directional_derivative_operators(
        &self,
        d_beta_flats: &[Array1<f64>],
    ) -> Result<Vec<Option<Arc<dyn HyperOperator>>>, String> {
        self.family
            .exact_newton_joint_hessian_directional_derivative_operators_from_cache_with_options(
                &self.block_states,
                d_beta_flats,
                &self.cache,
                &self.options,
            )
    }

    fn directional_derivative(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family
            .exact_newton_joint_hessian_directional_derivative_from_cache_with_options(
                &self.block_states,
                d_beta_flat,
                &self.cache,
                &self.options,
            )
    }

    fn second_directional_derivative_operator(
        &self,
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn HyperOperator>>, String> {
        self.family
            .exact_newton_joint_hessiansecond_directional_derivative_operator_from_cache_with_options(
                &self.block_states,
                d_beta_u_flat,
                d_beta_v_flat,
                &self.cache,
                &self.options,
            )
    }

    fn second_directional_derivative_operators(
        &self,
        d_beta_pairs: &[(Array1<f64>, Array1<f64>)],
    ) -> Result<Vec<Option<Arc<dyn HyperOperator>>>, String> {
        self.family
            .exact_newton_joint_hessiansecond_directional_derivative_operators_from_cache_with_options(
                &self.block_states,
                d_beta_pairs,
                &self.cache,
                &self.options,
            )
    }

    fn second_directional_derivative(
        &self,
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family
            .exact_newton_joint_hessiansecond_directional_derivative_from_cache_with_options(
                &self.block_states,
                d_beta_u_flat,
                d_beta_v_flat,
                &self.cache,
                &self.options,
            )
    }
}

impl BernoulliMarginalSlopeFamily {
    pub(super) fn block_ranges_from_specs(specs: &[ParameterBlockSpec]) -> Vec<(usize, usize)> {
        let mut cursor = 0usize;
        specs
            .iter()
            .map(|spec| {
                let start = cursor;
                cursor += spec.design.ncols();
                (start, cursor)
            })
            .collect()
    }

    pub(super) fn flatten_block_state_betas_for_specs(
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Array1<f64>, String> {
        if block_states.len() != specs.len() {
            return Err(format!(
                "bernoulli marginal-slope batched gradient state/spec mismatch: states={}, specs={}",
                block_states.len(),
                specs.len()
            ));
        }
        let total = specs.iter().map(|spec| spec.design.ncols()).sum::<usize>();
        let mut beta = Array1::<f64>::zeros(total);
        let mut cursor = 0usize;
        for (idx, (state, spec)) in block_states.iter().zip(specs.iter()).enumerate() {
            let width = spec.design.ncols();
            if state.beta.len() != width {
                return Err(format!(
                    "bernoulli marginal-slope batched gradient block {idx} beta length {} != spec width {}",
                    state.beta.len(),
                    width
                ));
            }
            beta.slice_mut(s![cursor..cursor + width])
                .assign(&state.beta);
            cursor += width;
        }
        Ok(beta)
    }

    pub(super) fn row_factor_primary_projection(
        &self,
        row: usize,
        slices: &BlockSlices,
        primary: &PrimarySlices,
        factor: &Array2<f64>,
        out: &mut [f64],
    ) -> Result<(), String> {
        let rank = factor.ncols();
        if out.len() != primary.total * rank {
            return Err(format!(
                "primary projection scratch length {} != {}",
                out.len(),
                primary.total * rank
            ));
        }
        out.fill(0.0);
        for col in 0..rank {
            out[primary.q * rank + col] = self
                .marginal_design
                .dot_row_view(row, factor.slice(s![slices.marginal.clone(), col]));
            out[primary.logslope * rank + col] = self
                .logslope_design
                .dot_row_view(row, factor.slice(s![slices.logslope.clone(), col]));
        }
        if let (Some(block_range), Some(primary_range)) = (slices.h.as_ref(), primary.h.as_ref()) {
            for (local, block_idx) in block_range.clone().enumerate() {
                let primary_idx = primary_range.start + local;
                for col in 0..rank {
                    out[primary_idx * rank + col] = factor[[block_idx, col]];
                }
            }
        }
        if let (Some(block_range), Some(primary_range)) = (slices.w.as_ref(), primary.w.as_ref()) {
            for (local, block_idx) in block_range.clone().enumerate() {
                let primary_idx = primary_range.start + local;
                for col in 0..rank {
                    out[primary_idx * rank + col] = factor[[block_idx, col]];
                }
            }
        }
        Ok(())
    }

    pub(super) fn row_primary_gram_from_projection(
        primary_total: usize,
        rank: usize,
        projection: &[f64],
    ) -> Vec<f64> {
        let mut gram = vec![0.0; primary_total * primary_total];
        for a in 0..primary_total {
            for b in 0..=a {
                let mut sum = 0.0;
                let a_base = a * rank;
                let b_base = b * rank;
                for col in 0..rank {
                    sum += projection[a_base + col] * projection[b_base + col];
                }
                gram[a * primary_total + b] = sum;
                gram[b * primary_total + a] = sum;
            }
        }
        gram
    }

    pub(super) fn primary_tail_block_pairs(
        slices: &BlockSlices,
        primary: &PrimarySlices,
    ) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        if let (Some(block_range), Some(primary_range)) = (slices.h.as_ref(), primary.h.as_ref()) {
            out.extend(
                block_range
                    .clone()
                    .enumerate()
                    .map(|(offset, block_idx)| (primary_range.start + offset, block_idx)),
            );
        }
        if let (Some(block_range), Some(primary_range)) = (slices.w.as_ref(), primary.w.as_ref()) {
            out.extend(
                block_range
                    .clone()
                    .enumerate()
                    .map(|(offset, block_idx)| (primary_range.start + offset, block_idx)),
            );
        }
        out
    }

    pub(super) fn primary_tail_tail_gram(
        primary_total: usize,
        rank: usize,
        factor: &Array2<f64>,
        tail_pairs: &[(usize, usize)],
    ) -> Vec<f64> {
        let mut gram = vec![0.0; primary_total * primary_total];
        for (a_pos, &(primary_a, block_a)) in tail_pairs.iter().enumerate() {
            for &(primary_b, block_b) in tail_pairs.iter().take(a_pos + 1) {
                let mut sum = 0.0;
                for col in 0..rank {
                    sum += factor[[block_a, col]] * factor[[block_b, col]];
                }
                gram[primary_a * primary_total + primary_b] = sum;
                gram[primary_b * primary_total + primary_a] = sum;
            }
        }
        gram
    }

    pub(super) fn row_primary_trace_contract(third: &Array2<f64>, gram: &[f64]) -> f64 {
        let r = third.nrows();
        assert_eq!(third.ncols(), r);
        assert_eq!(gram.len(), r * r);
        let mut total = 0.0;
        for a in 0..r {
            for b in 0..r {
                total += third[[a, b]] * gram[a * r + b];
            }
        }
        total
    }

    pub(super) fn row_primary_third_contracted_many_with_moments(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
        row_dirs: &[Array1<f64>],
    ) -> Result<Vec<Array2<f64>>, String> {
        let primary = &cache.primary;
        let r = primary.total;
        if row_dirs.is_empty() {
            return Ok(Vec::new());
        }
        if let Some((idx, dir)) = row_dirs.iter().enumerate().find(|(_, dir)| dir.len() != r) {
            return Err(format!(
                "bernoulli marginal-slope row third direction {idx} length {} != {r}",
                dir.len()
            ));
        }
        if row_dirs.len() == 1 {
            return Ok(vec![self.row_primary_third_contracted_recompute(
                row,
                block_states,
                cache,
                row_ctx,
                &row_dirs[0],
            )?]);
        }
        if row_dirs.iter().all(|dir| {
            Self::primary_direction_is_zero(dir, primary)
                || Self::single_primary_axis(dir, primary).is_some()
        }) {
            let zero = || Array2::<f64>::zeros((r, r));
            if row_dirs
                .iter()
                .all(|dir| Self::primary_direction_is_zero(dir, primary))
            {
                return Ok(row_dirs.iter().map(|_| zero()).collect());
            }
            if let Some(tensors) = self.flex_axis_third_tensors_for_row(block_states, cache, row)? {
                return Ok(row_dirs
                    .iter()
                    .map(|dir| {
                        if Self::primary_direction_is_zero(dir, primary) {
                            zero()
                        } else {
                            let (axis, scalar) = Self::single_primary_axis(dir, primary)
                                .expect("all directions checked as zero or single-axis");
                            let mut out = tensors.third[axis].clone();
                            out.mapv_inplace(|value| value * scalar);
                            out
                        }
                    })
                    .collect());
            }
        }
        if !self.effective_flex_active(block_states)? {
            let t = self.rigid_third_full_cached(block_states, cache, row)?;
            return row_dirs
                .iter()
                .map(|dir| {
                    let m = contract_third_full(t, dir[0], dir[1]);
                    let mut out = Array2::<f64>::zeros((2, 2));
                    out[[0, 0]] = m[0][0];
                    out[[0, 1]] = m[0][1];
                    out[[1, 0]] = m[1][0];
                    out[[1, 1]] = m[1][1];
                    Ok(out)
                })
                .collect();
        }
        if !row_ctx.intercept.is_finite() || !row_ctx.m_a.is_finite() || row_ctx.m_a <= 0.0 {
            return Err(
                "non-finite flexible row context in batched third-order contraction".to_string(),
            );
        }

        let point = self.primary_point_from_block_states(row, block_states, primary)?;
        let (q, b, beta_h_owned, beta_w_owned) = self.primary_point_components(&point, primary);
        let beta_h = beta_h_owned.as_ref();
        let beta_w = beta_w_owned.as_ref();
        if let Some(grid) = self.latent_measure.empirical_grid_for_training_row(row)? {
            return row_dirs
                .iter()
                .map(|dir| {
                    self.empirical_flex_row_third_contracted_recompute(
                        row, primary, q, b, beta_h, beta_w, row_ctx, dir, &grid,
                    )
                })
                .collect();
        }

        let a = row_ctx.intercept;
        let marginal = self.marginal_link_map(q)?;
        let h_range = primary.h.as_ref();
        let w_range = primary.w.as_ref();
        let score_runtime = self.score_warp.as_ref();
        let link_runtime = self.link_dev.as_ref();
        let scale = self.probit_frailty_scale();
        let zero_family = vec![[0.0; 4]; r];
        let n_dirs = row_dirs.len();

        let mut f_a = 0.0;
        let mut f_aa = 0.0;
        let mut f_u = Array1::<f64>::zeros(r);
        let mut f_au = Array1::<f64>::zeros(r);
        let mut f_uv = Array2::<f64>::zeros((r, r));
        let mut f_a_dir = vec![0.0; n_dirs];
        let mut f_aa_dir = vec![0.0; n_dirs];
        let mut f_au_dir = vec![0.0; n_dirs * r];
        let mut f_uv_dir = vec![0.0; n_dirs * r * r];

        let owned_cells;
        let cells: &[CachedDenestedCellMoments] = if let Some(cached) =
            self.row_cell_moments_for_third_degree15(cache, row)?
        {
            cached
        } else {
            let partitions = self.denested_partition_cells(a, b, beta_h, beta_w)?;
            owned_cells = partitions
                .into_iter()
                .map(|partition_cell| {
                    exact_kernel::evaluate_cell_derivative_moments_uncached(partition_cell.cell, 15)
                        .map(|state| CachedDenestedCellMoments {
                            partition_cell,
                            state,
                        })
                })
                .collect::<Result<Vec<_>, String>>()?;
            &owned_cells
        };

        Self::accumulate_primary_third_cell_moments(
            cells,
            a,
            b,
            scale,
            r,
            h_range,
            w_range,
            score_runtime,
            link_runtime,
            &zero_family,
            row_dirs,
            "score-warp batched third direction",
            "link-wiggle batched third direction",
            &mut f_a,
            &mut f_aa,
            &mut f_u,
            &mut f_au,
            &mut f_uv,
            &mut f_a_dir,
            &mut f_aa_dir,
            &mut f_au_dir,
            &mut f_uv_dir,
        )?;

        f_u[0] = -marginal.mu1;
        f_uv[[0, 0]] = -marginal.mu2;

        let inv_f_a = 1.0 / f_a;
        let mut a_u = Array1::<f64>::zeros(r);
        for u in 0..r {
            a_u[u] = -f_u[u] * inv_f_a;
        }
        let mut a_uv = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let val =
                    -(f_uv[[u, v]] + f_au[u] * a_u[v] + f_au[v] * a_u[u] + f_aa * a_u[u] * a_u[v])
                        * inv_f_a;
                a_uv[[u, v]] = val;
                a_uv[[v, u]] = val;
            }
        }

        let z_obs = self.z[row];
        let u_obs = a + b * z_obs;
        let obs = self.observed_denested_cell_partials(row, a, b, beta_h, beta_w)?;
        let eta_val = eval_coeff4_at(&obs.coeff, z_obs);

        let mut g_u_fixed = vec![[0.0; 4]; r];
        let mut g_au_fixed = vec![[0.0; 4]; r];
        let mut g_bu_fixed = vec![[0.0; 4]; r];
        let mut g_aau_fixed = vec![[0.0; 4]; r];
        let mut g_abu_fixed = vec![[0.0; 4]; r];
        let mut g_bbu_fixed = vec![[0.0; 4]; r];

        g_u_fixed[1] = obs.dc_db;
        g_au_fixed[1] = obs.dc_dab;
        g_bu_fixed[1] = obs.dc_dbb;
        g_aau_fixed[1] = obs.dc_daab;
        g_abu_fixed[1] = obs.dc_dabb;
        g_bbu_fixed[1] = obs.dc_dbbb;

        if let (Some(h_range), Some(runtime)) = (h_range, score_runtime) {
            Self::for_each_deviation_basis_cubic_at(
                runtime,
                h_range,
                z_obs,
                "score-warp batched third observed",
                |_, idx, basis_span| {
                    fill_score_basis_cell_coeff_jet(
                        idx,
                        basis_span,
                        b,
                        scale,
                        &mut g_u_fixed,
                        &mut g_bu_fixed,
                    );
                    Ok(())
                },
            )?;
        }

        if let (Some(w_range), Some(runtime)) = (w_range, link_runtime) {
            Self::for_each_deviation_basis_cubic_at(
                runtime,
                w_range,
                u_obs,
                "link-wiggle batched third observed",
                |_, idx, basis_span| {
                    fill_link_basis_cell_coeff_jet(
                        idx,
                        basis_span,
                        a,
                        b,
                        scale,
                        &mut g_u_fixed,
                        &mut g_au_fixed,
                        &mut g_bu_fixed,
                        &mut g_aau_fixed,
                        &mut g_abu_fixed,
                        &mut g_bbu_fixed,
                    );
                    Ok(())
                },
            )?;
        }

        let g_jet = SparsePrimaryCoeffJetView::new(
            1,
            h_range,
            w_range,
            &g_u_fixed,
            &g_au_fixed,
            &g_bu_fixed,
            &g_aau_fixed,
            &g_abu_fixed,
            &g_bbu_fixed,
            &zero_family,
            &zero_family,
            &zero_family,
            &zero_family,
        );

        let g_a = eval_coeff4_at(&obs.dc_da, z_obs);
        let g_aa = eval_coeff4_at(&obs.dc_daa, z_obs);
        let g_aaa = eval_coeff4_at(&obs.dc_daaa, z_obs);
        let mut g_u = Array1::<f64>::zeros(r);
        let mut g_au = Array1::<f64>::zeros(r);
        let mut g_aau = Array1::<f64>::zeros(r);
        let mut g_uv = Array2::<f64>::zeros((r, r));
        let mut g_auv = Array2::<f64>::zeros((r, r));
        for u in 1..r {
            g_u[u] = eval_coeff4_at(&g_jet.first[u], z_obs);
            g_au[u] = eval_coeff4_at(&g_jet.a_first[u], z_obs);
            g_aau[u] = eval_coeff4_at(&g_jet.aa_first[u], z_obs);
        }
        for u in 1..r {
            for v in u..r {
                let second_coeff = g_jet.pair_from_b_family(g_jet.b_first, u, v, COEFF_SUPPORT_BHW);
                let val = eval_coeff4_at(&second_coeff, z_obs);
                g_uv[[u, v]] = val;
                g_uv[[v, u]] = val;

                let third_coeff = g_jet.pair_from_b_family(g_jet.ab_first, u, v, COEFF_SUPPORT_BW);
                let third_val = eval_coeff4_at(&third_coeff, z_obs);
                g_auv[[u, v]] = third_val;
                g_auv[[v, u]] = third_val;
            }
        }

        let eta_u = g_a * &a_u + &g_u;
        let mut eta_uv = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let val = g_a * a_uv[[u, v]]
                    + g_aa * a_u[u] * a_u[v]
                    + g_au[u] * a_u[v]
                    + g_au[v] * a_u[u]
                    + g_uv[[u, v]];
                eta_uv[[u, v]] = val;
                eta_uv[[v, u]] = val;
            }
        }

        let y_i = self.y[row];
        let w_i = self.weights[row];
        let s_y = 2.0 * y_i - 1.0;
        let m = s_y * eta_val;
        let (k1, k2, k3, _) = signed_probit_neglog_derivatives_up_to_fourth(m, w_i)?;
        let u1 = s_y * k1;
        let u3 = s_y * k3;
        let mut out = (0..n_dirs)
            .map(|_| Array2::<f64>::zeros((r, r)))
            .collect::<Vec<_>>();

        for (dir_idx, dir) in row_dirs.iter().enumerate() {
            let dir_base = dir_idx * r * r;
            f_uv_dir[dir_base] = -dir[0] * marginal.mu3;

            let a_dir = a_u.dot(dir);
            let a_u_dir = a_uv.dot(dir);
            let g_dir_fixed = g_jet.directional_family(g_jet.first, dir, COEFF_SUPPORT_BHW);
            let g_a_dir_fixed = g_jet.directional_family(g_jet.a_first, dir, COEFF_SUPPORT_BW);
            let g_aa_dir_fixed = g_jet.directional_family(g_jet.aa_first, dir, COEFF_SUPPORT_BW);
            let g_dir = eval_coeff4_at(&g_dir_fixed, z_obs);
            let g_a_dir = eval_coeff4_at(&g_a_dir_fixed, z_obs);
            let g_aa_dir = eval_coeff4_at(&g_aa_dir_fixed, z_obs);
            let eta_dir = g_a * a_dir + g_dir;
            let eta_u_dir = eta_uv.dot(dir);
            let dg_a_dir = g_aa * a_dir + g_a_dir;
            let dg_aa_dir = g_aaa * a_dir + g_aa_dir;
            let mut dg_au_dir = Array1::<f64>::zeros(r);
            for u in 0..r {
                let coeff =
                    g_jet.param_directional_from_b_family(g_jet.ab_first, u, dir, COEFF_SUPPORT_BW);
                dg_au_dir[u] = g_aau[u] * a_dir + eval_coeff4_at(&coeff, z_obs);
            }

            for u in 0..r {
                for v in u..r {
                    let fuvd = f_uv_dir[dir_base + u * r + v];
                    let n_dir = fuvd
                        + f_au_dir[dir_idx * r + u] * a_u[v]
                        + f_au[u] * a_u_dir[v]
                        + f_au_dir[dir_idx * r + v] * a_u[u]
                        + f_au[v] * a_u_dir[u]
                        + f_aa_dir[dir_idx] * a_u[u] * a_u[v]
                        + f_aa * (a_u_dir[u] * a_u[v] + a_u[u] * a_u_dir[v]);
                    let a_uv_dir = -(n_dir + f_a_dir[dir_idx] * a_uv[[u, v]]) * inv_f_a;
                    let third_coeff = g_jet.pair_directional_from_bb_family(
                        g_jet.bb_first,
                        u,
                        v,
                        dir,
                        COEFF_SUPPORT_BW,
                    );
                    let dg_uv_dir = g_auv[[u, v]] * a_dir + eval_coeff4_at(&third_coeff, z_obs);
                    let eta_uv_dir = dg_a_dir * a_uv[[u, v]]
                        + g_a * a_uv_dir
                        + dg_aa_dir * a_u[u] * a_u[v]
                        + g_aa * (a_u_dir[u] * a_u[v] + a_u[u] * a_u_dir[v])
                        + dg_au_dir[u] * a_u[v]
                        + g_au[u] * a_u_dir[v]
                        + dg_au_dir[v] * a_u[u]
                        + g_au[v] * a_u_dir[u]
                        + dg_uv_dir;
                    let val = u3 * eta_u[u] * eta_u[v] * eta_dir
                        + k2 * (eta_uv[[u, v]] * eta_dir
                            + eta_u[u] * eta_u_dir[v]
                            + eta_u[v] * eta_u_dir[u])
                        + u1 * eta_uv_dir;
                    out[dir_idx][[u, v]] = val;
                    out[dir_idx][[v, u]] = val;
                }
            }
        }

        Ok(out)
    }

    pub(super) fn batched_rho_correction_logdet_traces_for_rows(
        &self,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
        factor: &Array2<f64>,
        directions: &Array2<f64>,
        weighted_rows: &[WeightedOuterRow],
    ) -> Result<Array1<f64>, String> {
        let slices = &cache.slices;
        let primary = &cache.primary;
        let rank = factor.ncols();
        let n_dirs = directions.ncols();
        if factor.nrows() != slices.total || directions.nrows() != slices.total {
            return Err(format!(
                "bernoulli marginal-slope batched trace shape mismatch: factor={}x{}, directions={}x{}, p={}",
                factor.nrows(),
                factor.ncols(),
                directions.nrows(),
                directions.ncols(),
                slices.total
            ));
        }
        let started = std::time::Instant::now();
        if log_exact_work(self.y.len()) {
            log::info!(
                "[BMS rho-correction-trace] sampled start n={} rows={} p={} rank={} dirs={}",
                self.y.len(),
                weighted_rows.len(),
                slices.total,
                rank,
                n_dirs
            );
        }
        let traces = weighted_rows
            .par_iter()
            .try_fold(
                || vec![0.0; n_dirs],
                |mut acc, wr| -> Result<_, String> {
                    let row = wr.index;
                    let row_ctx = Self::row_ctx(cache, row);
                    let mut projection = vec![0.0; primary.total * rank];
                    self.row_factor_primary_projection(
                        row,
                        slices,
                        primary,
                        factor,
                        &mut projection,
                    )?;
                    let gram =
                        Self::row_primary_gram_from_projection(primary.total, rank, &projection);
                    if n_dirs == 1 {
                        let d_beta = directions.column(0).to_owned();
                        let row_dir =
                            self.row_primary_direction_from_flat(row, slices, primary, &d_beta)?;
                        let row_traces = self.row_primary_third_trace_many_with_moments(
                            row,
                            block_states,
                            cache,
                            row_ctx,
                            &[row_dir],
                            &gram,
                        )?;
                        acc[0] += wr.weight * row_traces[0];
                        return Ok(acc);
                    }
                    let trace_gradient = self.row_primary_third_trace_gradient_with_moments(
                        row,
                        block_states,
                        cache,
                        row_ctx,
                        &gram,
                    )?;
                    for dir_idx in 0..n_dirs {
                        let mut trace = trace_gradient[primary.q]
                            * self.marginal_design.dot_row_view(
                                row,
                                directions.slice(s![slices.marginal.clone(), dir_idx]),
                            );
                        trace += trace_gradient[primary.logslope]
                            * self.logslope_design.dot_row_view(
                                row,
                                directions.slice(s![slices.logslope.clone(), dir_idx]),
                            );
                        if let (Some(block_range), Some(primary_range)) =
                            (slices.h.as_ref(), primary.h.as_ref())
                        {
                            for (offset, block_idx) in block_range.clone().enumerate() {
                                trace += trace_gradient[primary_range.start + offset]
                                    * directions[[block_idx, dir_idx]];
                            }
                        }
                        if let (Some(block_range), Some(primary_range)) =
                            (slices.w.as_ref(), primary.w.as_ref())
                        {
                            for (offset, block_idx) in block_range.clone().enumerate() {
                                trace += trace_gradient[primary_range.start + offset]
                                    * directions[[block_idx, dir_idx]];
                            }
                        }
                        acc[dir_idx] += wr.weight * trace;
                    }
                    Ok(acc)
                },
            )
            .try_reduce(
                || vec![0.0; n_dirs],
                |mut left, right| -> Result<_, String> {
                    for (l, r) in left.iter_mut().zip(right.iter()) {
                        *l += *r;
                    }
                    Ok(left)
                },
            )?;
        if log_exact_work(self.y.len()) {
            log::info!(
                "[BMS rho-correction-trace] sampled done n={} rows={} p={} rank={} dirs={} elapsed={:.3}s",
                self.y.len(),
                weighted_rows.len(),
                slices.total,
                rank,
                n_dirs,
                started.elapsed().as_secs_f64()
            );
        }
        Ok(Array1::from_vec(traces))
    }

    pub(super) fn batched_rho_correction_logdet_traces_full_rows(
        &self,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
        factor: &Array2<f64>,
        directions: &Array2<f64>,
    ) -> Result<Array1<f64>, String> {
        let slices = &cache.slices;
        let primary = &cache.primary;
        let n = self.y.len();
        let rank = factor.ncols();
        let n_dirs = directions.ncols();
        if n == 0 || rank == 0 || n_dirs == 0 {
            return Ok(Array1::zeros(n_dirs));
        }
        let rows_per_chunk = {
            const TARGET_BYTES: usize = 8 * 1024 * 1024;
            let panels = 4usize;
            let width = rank + n_dirs;
            (TARGET_BYTES / (panels * width.max(1) * 8)).max(512).min(n)
        };
        let factor_m = factor.slice(s![slices.marginal.clone(), ..]);
        let factor_g = factor.slice(s![slices.logslope.clone(), ..]);
        let dir_m = directions.slice(s![slices.marginal.clone(), ..]);
        let dir_g = directions.slice(s![slices.logslope.clone(), ..]);
        let tail_pairs = Self::primary_tail_block_pairs(slices, primary);
        let tail_tail_gram = Self::primary_tail_tail_gram(primary.total, rank, factor, &tail_pairs);
        let n_chunks = n.div_ceil(rows_per_chunk);
        let started = std::time::Instant::now();
        let completed_chunks = AtomicUsize::new(0);
        let progress_step = (n_chunks / 10).max(1);
        if log_exact_work(n) {
            log::info!(
                "[BMS rho-correction-trace] full start n={} chunks={} rows_per_chunk={} p={} rank={} dirs={}",
                n,
                n_chunks,
                rows_per_chunk,
                slices.total,
                rank,
                n_dirs
            );
        }
        let traces = (0..n_chunks)
            .into_par_iter()
            .map(|chunk_idx| -> Result<Vec<f64>, String> {
                // This chunk runs on a Rayon worker and issues `fast_ab` GEMMs
                // below; `with_nested_parallel` pins their faer parallelism to
                // `Par::Seq` so they do not re-fan the global Rayon pool against
                // this already-parallel row fan-out (the rayon×BLAS
                // oversubscription that intermittently stalled the joint-Newton
                // `hessian_qp` cycle). Bit-identical: faer partitions the matmul
                // output, never the contracted axis.
                gam_problem::with_nested_parallel(|| {
                let start = chunk_idx * rows_per_chunk;
                let end = (start + rows_per_chunk).min(n);
                let rows = start..end;
                // Zero-copy fast path: when BOTH designs are materialised dense
                // the chunk rows are read straight from the stored matrix as
                // borrowed `ArrayView2` slices. `try_row_chunk` would `.to_owned()`
                // a fresh `(rows × p)` `Array2` for every chunk on every
                // ρ-homotopy pre-warm pass — the dominant `OwnedRepr<f64>`
                // alloc/`drop_in_place` churn of the cold marginal-slope fit.
                // `fast_ab` is generic over `Data<Elem = f64>`, so the view feeds
                // the identical BLAS-3 kernel with identical arithmetic.
                let (proj_m, proj_g, dir_proj_m, dir_proj_g) = match (
                    self.marginal_design.as_dense_ref(),
                    self.logslope_design.as_dense_ref(),
                ) {
                    (Some(x_full), Some(g_full)) => {
                        let x_chunk = x_full.slice(s![rows.clone(), ..]);
                        let g_chunk = g_full.slice(s![rows.clone(), ..]);
                        (
                            gam_linalg::faer_ndarray::fast_ab(&x_chunk, &factor_m),
                            gam_linalg::faer_ndarray::fast_ab(&g_chunk, &factor_g),
                            gam_linalg::faer_ndarray::fast_ab(&x_chunk, &dir_m),
                            gam_linalg::faer_ndarray::fast_ab(&g_chunk, &dir_g),
                        )
                    }
                    _ => {
                        let x_chunk = self
                            .marginal_design
                            .try_row_chunk(rows.clone())
                            .map_err(|err| format!("marginal trace row chunk failed: {err}"))?;
                        let g_chunk = self
                            .logslope_design
                            .try_row_chunk(rows.clone())
                            .map_err(|err| format!("logslope trace row chunk failed: {err}"))?;
                        (
                            gam_linalg::faer_ndarray::fast_ab(&x_chunk, &factor_m),
                            gam_linalg::faer_ndarray::fast_ab(&g_chunk, &factor_g),
                            gam_linalg::faer_ndarray::fast_ab(&x_chunk, &dir_m),
                            gam_linalg::faer_ndarray::fast_ab(&g_chunk, &dir_g),
                        )
                    }
                };
                let mut acc = vec![0.0; n_dirs];
                let mut gram = vec![0.0; primary.total * primary.total];
                let mut row_dir = Array1::<f64>::zeros(primary.total);
                for local in 0..(end - start) {
                    let row = start + local;
                    gram.copy_from_slice(&tail_tail_gram);
                    let mut qq = 0.0;
                    let mut qg = 0.0;
                    let mut gg = 0.0;
                    for col in 0..rank {
                        let qv = proj_m[[local, col]];
                        let gv = proj_g[[local, col]];
                        qq += qv * qv;
                        qg += qv * gv;
                        gg += gv * gv;
                    }
                    gram[primary.q * primary.total + primary.q] = qq;
                    gram[primary.q * primary.total + primary.logslope] = qg;
                    gram[primary.logslope * primary.total + primary.q] = qg;
                    gram[primary.logslope * primary.total + primary.logslope] = gg;
                    for &(primary_idx, block_idx) in &tail_pairs {
                        let mut q_tail = 0.0;
                        let mut g_tail = 0.0;
                        for col in 0..rank {
                            let tail = factor[[block_idx, col]];
                            q_tail += proj_m[[local, col]] * tail;
                            g_tail += proj_g[[local, col]] * tail;
                        }
                        gram[primary.q * primary.total + primary_idx] = q_tail;
                        gram[primary_idx * primary.total + primary.q] = q_tail;
                        gram[primary.logslope * primary.total + primary_idx] = g_tail;
                        gram[primary_idx * primary.total + primary.logslope] = g_tail;
                    }
                    let row_ctx = Self::row_ctx(cache, row);
                    if n_dirs == 1 {
                        row_dir.fill(0.0);
                        row_dir[primary.q] = dir_proj_m[[local, 0]];
                        row_dir[primary.logslope] = dir_proj_g[[local, 0]];
                        if let (Some(block_range), Some(primary_range)) =
                            (slices.h.as_ref(), primary.h.as_ref())
                        {
                            for (offset, block_idx) in block_range.clone().enumerate() {
                                row_dir[primary_range.start + offset] = directions[[block_idx, 0]];
                            }
                        }
                        if let (Some(block_range), Some(primary_range)) =
                            (slices.w.as_ref(), primary.w.as_ref())
                        {
                            for (offset, block_idx) in block_range.clone().enumerate() {
                                row_dir[primary_range.start + offset] = directions[[block_idx, 0]];
                            }
                        }
                        let row_traces = self.row_primary_third_trace_many_with_moments(
                            row,
                            block_states,
                            cache,
                            row_ctx,
                            std::slice::from_ref(&row_dir),
                            &gram,
                        )?;
                        acc[0] += row_traces[0];
                        continue;
                    }
                    let trace_gradient = self.row_primary_third_trace_gradient_with_moments(
                        row,
                        block_states,
                        cache,
                        row_ctx,
                        &gram,
                    )?;
                    for dir_idx in 0..n_dirs {
                        let mut trace = trace_gradient[primary.q] * dir_proj_m[[local, dir_idx]]
                            + trace_gradient[primary.logslope] * dir_proj_g[[local, dir_idx]];
                        if let (Some(block_range), Some(primary_range)) =
                            (slices.h.as_ref(), primary.h.as_ref())
                        {
                            for (offset, block_idx) in block_range.clone().enumerate() {
                                trace += trace_gradient[primary_range.start + offset]
                                    * directions[[block_idx, dir_idx]];
                            }
                        }
                        if let (Some(block_range), Some(primary_range)) =
                            (slices.w.as_ref(), primary.w.as_ref())
                        {
                            for (offset, block_idx) in block_range.clone().enumerate() {
                                trace += trace_gradient[primary_range.start + offset]
                                    * directions[[block_idx, dir_idx]];
                            }
                        }
                        acc[dir_idx] += trace;
                    }
                }
                if log_exact_work(n) {
                    let done = completed_chunks.fetch_add(1, Ordering::Relaxed) + 1;
                    if done == n_chunks || done % progress_step == 0 {
                        log::info!(
                            "[BMS rho-correction-trace] full progress chunks={}/{} rows={}/{} elapsed={:.3}s",
                            done,
                            n_chunks,
                            (done * rows_per_chunk).min(n),
                            n,
                            started.elapsed().as_secs_f64()
                        );
                    }
                }
                Ok(acc)
                })
            })
            .try_reduce(
                || vec![0.0; n_dirs],
                |mut left, right| -> Result<_, String> {
                    for (l, r) in left.iter_mut().zip(right.iter()) {
                        *l += *r;
                    }
                    Ok(left)
                },
            )?;
        if log_exact_work(n) {
            log::info!(
                "[BMS rho-correction-trace] full done n={} chunks={} p={} rank={} dirs={} elapsed={:.3}s",
                n,
                n_chunks,
                slices.total,
                rank,
                n_dirs,
                started.elapsed().as_secs_f64()
            );
        }
        Ok(Array1::from_vec(traces))
    }
}

impl BernoulliMarginalSlopeExactNewtonJointPsiWorkspace {
    pub(super) fn new(
        family: BernoulliMarginalSlopeFamily,
        block_states: Vec<ParameterBlockState>,
        specs: Vec<ParameterBlockSpec>,
        derivative_blocks: Vec<Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>>,
        options: BlockwiseFitOptions,
    ) -> Result<Self, String> {
        // Build (or reuse, at a bit-identical β) the exact-cache. This workspace
        // does not materialize per-row primary Hessians, so it keys a separate
        // store slot from the Hessian-workspace build.
        let cache = family.build_or_reuse_shared_exact_cache(&block_states, &options, false)?;
        // Prime the per-row uncontracted third-derivative tensor at workspace
        // construction (rigid path only). The build runs at top-level rayon
        // here, so the parallel row pass uses all cores. If we instead let
        // it run lazily inside `build_psi_hyper_coords` axis calls, those
        // calls are themselves at top level — so leaving lazy would also be
        // parallel — but priming here lifts the first-axis cost out of the
        // workspace's `first_order_terms` measurement. The warm-up writes the
        // `RayonSafeOnce` interior fields of the (possibly shared) cache; that
        // is idempotent and yields the same values across a shared `Arc`.
        if !family.effective_flex_active(&block_states)? {
            let warmed_third = family.rigid_third_full_cached(&block_states, cache.as_ref(), 0)?;
            ensure_finite_third_full_cache_row(
                warmed_third,
                "BernoulliMarginalSlopeExactNewtonJointPsiWorkspace third-cache warm-up",
            )?;
            // Outer-Hessian path consumes per-row fourth-tensor over every
            // (ψ-axis-i, ψ-axis-j) pair — prime here too so the 528-pair
            // sweep reads a populated cache instead of triggering the
            // 8-direction empirical jet on its first per-pair call.
            let warmed_fourth =
                family.rigid_fourth_full_cached(&block_states, cache.as_ref(), 0)?;
            ensure_finite_fourth_full_cache_row(
                warmed_fourth,
                "BernoulliMarginalSlopeExactNewtonJointPsiWorkspace fourth-cache warm-up",
            )?;
        }
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

impl crate::marginal_slope_shared::MarginalSlopePsiFamily
    for BernoulliMarginalSlopeExactNewtonJointPsiWorkspace
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
        self.family
            .exact_newton_joint_psi_terms_from_cache_with_options(
                &self.block_states,
                &self.derivative_blocks,
                psi_index,
                &self.cache,
                &self.options,
            )
    }

    fn psi_first_order_terms_all(&self) -> Result<Option<Vec<ExactNewtonJointPsiTerms>>, String> {
        let total: usize = self.derivative_blocks.iter().map(Vec::len).sum();
        if total == 0 {
            return Ok(Some(Vec::new()));
        }
        for psi_index in 0..total {
            if self
                .family
                .is_sigma_aux_index(&self.derivative_blocks, psi_index)
            {
                return Ok(None);
            }
        }
        let mut axes: Vec<PsiAxisSpec> = Vec::with_capacity(total);
        for psi_index in 0..total {
            let Some((block_idx, local_idx)) =
                psi_derivative_location(&self.derivative_blocks, psi_index)
            else {
                return Ok(None);
            };
            axes.push(self.family.resolve_psi_axis_spec(
                &self.derivative_blocks,
                block_idx,
                local_idx,
            )?);
        }
        let results = self.family.run_psi_row_pass_for_axes(
            &self.block_states,
            &self.cache,
            &self.options,
            &axes,
        )?;
        Ok(Some(results))
    }

    fn both_sigma_aux_second_order(&self, psi_i: usize, psi_j: usize) -> bool {
        self.family
            .is_sigma_aux_index(&self.derivative_blocks, psi_i)
            && self
                .family
                .is_sigma_aux_index(&self.derivative_blocks, psi_j)
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
        Err("bernoulli marginal-slope mixed log-sigma/spatial psi second derivatives require cross auxiliary terms; only pure log-sigma second derivatives are supported"
            .to_string())
    }

    fn psi_second_order_terms(
        &self,
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        self.family
            .exact_newton_joint_psisecond_order_terms_from_cache_with_options(
                &self.block_states,
                &self.derivative_blocks,
                psi_i,
                psi_j,
                &self.cache,
                &self.options,
            )
    }

    fn psi_second_order_terms_contracted(
        &self,
        alpha_psi: &[f64],
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiSecondOrderContracted>, String>
    {
        self.family
            .exact_newton_joint_psisecond_order_terms_contracted_from_cache_with_options(
                &self.block_states,
                &self.derivative_blocks,
                alpha_psi,
                &self.cache,
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
            .exact_newton_joint_psihessian_directional_derivative_operator_from_cache_with_options(
                &self.block_states,
                &self.derivative_blocks,
                psi_index,
                d_beta_flat,
                &self.cache,
                &self.options,
            )
    }
}
