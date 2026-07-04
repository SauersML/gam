use super::*;
use std::sync::RwLock;

impl<'a> RemlState<'a> {
    pub(crate) const POLISH_NORM_RATIO: f64 = 0.25;

    pub(crate) fn hypergradient_owner_key(&self) -> usize {
        self as *const _ as usize
    }

    pub(crate) fn ift_quality_step_cap(&self, default_cap: f64) -> f64 {
        let states = ift_quality_states().lock().unwrap();
        states
            .get(&self.hypergradient_owner_key())
            .and_then(|state| state.next_step_cap)
            .filter(|cap| cap.is_finite() && *cap > 0.0)
            .unwrap_or(default_cap)
    }

    pub(crate) fn take_ift_quality_flat_override(&self) -> bool {
        let mut states = ift_quality_states().lock().unwrap();
        let Some(state) = states.get_mut(&self.hypergradient_owner_key()) else {
            return false;
        };
        let fallback = state.fallback_next_flat;
        state.fallback_next_flat = false;
        fallback
    }

    pub(crate) fn clear_ift_quality_runtime_state(&self) {
        let mut states = ift_quality_states().lock().unwrap();
        states.remove(&self.hypergradient_owner_key());
    }

    pub(crate) fn record_ift_prediction_quality(
        &self,
        quality: f64,
        current_cap: f64,
    ) -> Option<f64> {
        if !quality.is_finite() || quality < 0.0 || !current_cap.is_finite() || current_cap <= 0.0 {
            return None;
        }
        let mut states = ift_quality_states().lock().unwrap();
        let state = states.entry(self.hypergradient_owner_key()).or_default();
        state.quality_history.push(quality);
        while state.quality_history.len() > IFT_QUALITY_HISTORY_CAP {
            state.quality_history.remove(0);
        }
        let rolling_quality =
            state.quality_history.iter().sum::<f64>() / state.quality_history.len() as f64;
        let next_step_cap = if rolling_quality < IFT_QUALITY_GROW_BAND {
            current_cap * IFT_STEP_CAP_GROW_FACTOR
        } else if rolling_quality < IFT_QUALITY_SHRINK_BAND {
            current_cap
        } else {
            current_cap * IFT_STEP_CAP_SHRINK_FACTOR
        };
        state.next_step_cap = Some(next_step_cap);
        state.fallback_next_flat = rolling_quality >= IFT_QUALITY_FLAT_FALLBACK_BAND;
        Some(next_step_cap)
    }

    pub(crate) fn reset_hypergradient_budget_controller(&self) {
        let mut budgets = hypergradient_budgets().lock().unwrap();
        budgets.remove(&self.hypergradient_owner_key());
    }

    pub(crate) fn hypergradient_trace_state(
        &self,
    ) -> Arc<Mutex<super::reml_outer_engine::StochasticTraceState>> {
        let mut budgets = hypergradient_budgets().lock().unwrap();
        let state = budgets
            .entry(self.hypergradient_owner_key())
            .or_insert_with(HyperGradientRuntimeState::new);
        Arc::clone(&state.trace_state)
    }

    pub(crate) fn reset_hypergradient_trace_telemetry(
        trace_state: &Arc<Mutex<super::reml_outer_engine::StochasticTraceState>>,
    ) {
        let mut trace = match trace_state.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        trace.last_linear_residual_norm = None;
        trace.last_probe_sigma_sq = None;
        trace.last_probe_count = 0;
    }

    pub(crate) fn hypergradient_adaptive_kkt_override(
        &self,
        pirls_config: &pirls::PirlsConfig,
    ) -> Option<pirls::AdaptiveKktTolerance> {
        let budgets = hypergradient_budgets().lock().unwrap();
        let tau = budgets
            .get(&self.hypergradient_owner_key())?
            .adaptive_kkt_override?;
        if !tau.is_finite() || tau <= 0.0 {
            return None;
        }
        let ceiling = pirls_config.convergence_tolerance;
        let floor =
            (self.config.reml_convergence_tolerance / ADAPTIVE_KKT_FLOOR_REML_DIVISOR).min(ceiling);
        if !(floor > 0.0 && ceiling >= floor) {
            return None;
        }
        let tau = tau.clamp(floor, ceiling);
        Some(pirls::AdaptiveKktTolerance {
            eta: 1.0,
            floor: tau,
            ceiling: tau,
            outer_grad_norm: tau,
        })
    }

    pub(crate) fn update_hypergradient_budget_after_outer_eval(
        &self,
        rho: &Array1<f64>,
        gradient: &Array1<f64>,
        ift_residual_energy: Option<f64>,
    ) {
        if rho.iter().any(|v| !v.is_finite()) || gradient.iter().any(|v| !v.is_finite()) {
            return;
        }

        let mut budgets = hypergradient_budgets().lock().unwrap();
        let state = budgets
            .entry(self.hypergradient_owner_key())
            .or_insert_with(HyperGradientRuntimeState::new);
        let (e_linear, sigma_sq, k, current_floor) = {
            let trace = match state.trace_state.lock() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            let linear_residual_norm = trace.last_linear_residual_norm.unwrap_or(0.0).max(0.0);
            (
                // Convert CG residual norm to 1/2*norm^2 so the channel is an energy like e_inner.
                0.5 * linear_residual_norm * linear_residual_norm,
                trace.last_probe_sigma_sq.unwrap_or(0.0).max(0.0),
                trace.last_probe_count,
                trace.monotone_probe_floor,
            )
        };
        let e_inner = ift_residual_energy.unwrap_or(0.0).max(0.0);
        state.budget.push(HyperGradHistoryEntry {
            rho: rho.clone(),
            g_outer: gradient.clone(),
            e_inner,
            e_linear,
            sigma_sq,
            k,
        });

        let sensitivity_estimate = state.budget.reestimate_sensitivities();
        let sensitivity_stable = state.budget.sensitivities_stable();
        let force_engage = state.budget.history.len() >= HGB_WARMUP_ITERS_MAX;
        if !state.budget.warmup_engaged
            && state.budget.history.len() >= HGB_WARMUP_ITERS_MIN
            && (sensitivity_stable || force_engage)
        {
            if sensitivity_stable {
                log::info!(
                    "[HGB] engage after {} iters (sensitivity stable)",
                    state.budget.history.len()
                );
            } else {
                log::info!(
                    "[HGB] engage after {} iters (max warmup reached)",
                    state.budget.history.len()
                );
            }
            state.budget.warmup_engaged = true;
        }

        if !state.budget.warmup_engaged {
            state.adaptive_kkt_override = None;
            match state.trace_state.lock() {
                Ok(mut trace) => trace.solve_rel_tol_override = None,
                Err(poisoned) => {
                    let mut trace = poisoned.into_inner();
                    trace.solve_rel_tol_override = None;
                }
            }
            return;
        }

        let [s_inner, s_linear, s_trace] = if let Some(sensitivities) = sensitivity_estimate {
            sensitivities
        } else {
            // Routine per-evaluation fallback (fires whenever the cross-channel
            // sensitivity estimate is unavailable, i.e. nearly every early
            // iteration) — not an anomaly. Keep it for `debug` tracing but off
            // the default `warn` stream so it does not re-create the #1688
            // firehose.
            log::debug!("[HGB] sensitivity_unavailable falling_back_to_per_channel");
            [S_INNER_INIT, S_LINEAR_INIT, S_TRACE_INIT]
        };

        let previous_grad_norm = state.budget.previous_gradient_norm().max(1e-12);
        state.budget.target_mse = (HGB_TARGET_FRACTION * previous_grad_norm).powi(2);
        let (eps2_inner, eps2_linear, eps2_trace, floor_active) = state
            .budget
            .allocate_with_sensitivities(s_inner, s_linear, s_trace);

        let ceiling = self.config.pirls_convergence_tolerance;
        let pirls_floor =
            (self.config.reml_convergence_tolerance / ADAPTIVE_KKT_FLOOR_REML_DIVISOR).min(ceiling);
        let tau_raw = eps2_inner.sqrt() / s_inner;
        let tau_inner = if pirls_floor > 0.0 && ceiling >= pirls_floor {
            tau_raw.clamp(pirls_floor, ceiling)
        } else {
            tau_raw
        }
        .max(0.0);
        state.adaptive_kkt_override =
            (tau_inner.is_finite() && tau_inner > 0.0).then_some(tau_inner);

        let rel_tol = eps2_linear.sqrt() / s_linear;
        let k_target = if eps2_trace > 0.0 && sigma_sq.is_finite() && sigma_sq > 0.0 {
            (sigma_sq / eps2_trace).ceil().clamp(0.0, usize::MAX as f64) as usize
        } else {
            current_floor
        };
        let raised_floor = current_floor.max(k_target);
        {
            let mut trace = match state.trace_state.lock() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            trace.solve_rel_tol_override =
                (rel_tol.is_finite() && rel_tol > 0.0).then_some(rel_tol);
            if raised_floor > trace.monotone_probe_floor {
                trace.monotone_probe_floor = raised_floor;
            }
        }

        let active = ["i", "l", "t"]
            .iter()
            .zip(floor_active.iter())
            .filter_map(|(name, active)| active.then_some(*name))
            .collect::<Vec<_>>()
            .join(",");
        log::info!(
            "[HGB] target_mse={:.3e} s_i={:.3e} s_l={:.3e} s_t={:.3e} eps²_i={:.3e} eps²_l={:.3e} eps²_t={:.3e} τ={:.3e} rtol={:.3e} k={} floor_active=[{}]",
            state.budget.target_mse,
            s_inner,
            s_linear,
            s_trace,
            eps2_inner,
            eps2_linear,
            eps2_trace,
            tau_inner,
            rel_tol,
            k_target,
            active,
        );
    }

    pub(crate) fn apply_inner_polish_step_to_warm_start(
        &self,
        bundle: &EvalShared,
        solution_beta: &Array1<f64>,
        polish_step: &Array1<f64>,
    ) {
        if !self.warm_start_enabled.load(Ordering::Relaxed)
            || solution_beta.len() != polish_step.len()
        {
            return;
        }
        let polish_norm_squared = polish_step.dot(polish_step);
        let beta_norm_squared = solution_beta.dot(solution_beta);
        if !polish_norm_squared.is_finite()
            || !beta_norm_squared.is_finite()
            || polish_norm_squared > Self::POLISH_NORM_RATIO * beta_norm_squared
        {
            log::info!(
                "[POLISH-SKIP] reason=large_step polish_norm²={} beta_norm²={}",
                polish_norm_squared,
                beta_norm_squared
            );
            return;
        }
        let polished_solution_beta = solution_beta - polish_step;
        let pirls_result = bundle.pirls_result.as_ref();
        let beta_original = match pirls_result.coordinate_frame {
            pirls::PirlsCoordinateFrame::OriginalSparseNative => {
                if self.active_constraint_free_basis(pirls_result).is_some() {
                    return;
                }
                polished_solution_beta
            }
            pirls::PirlsCoordinateFrame::TransformedQs => {
                if self.active_constraint_free_basis(pirls_result).is_some()
                    || polished_solution_beta.len() != self.p
                {
                    return;
                }
                polished_solution_beta
            }
        };
        if beta_original.len() != self.p || beta_original.iter().any(|v| !v.is_finite()) {
            return;
        }
        // The unified evaluator already paid for w = H^{-1}r to form the IFT
        // residual correction, so β - w is a free one-step inner polish. Store
        // it as the next warm start and keep the IFT cache's β-dependent rhs
        // precompute in sync with the polished coefficient vector.
        self.warm_start_beta
            .write()
            .unwrap()
            .replace(Coefficients::new(beta_original.clone()));
        if self.ift_warm_start_cache.read().unwrap().is_some() {
            let lambda_s_beta_blocks = {
                use rayon::prelude::*;
                let blocks: Vec<ndarray::Array1<f64>> = self
                    .canonical_penalties
                    .par_iter()
                    .map(|cp| {
                        let r = &cp.col_range;
                        let beta_block = beta_original.slice(s![r.start..r.end]);
                        let centered = &beta_block - &cp.prior_mean;
                        cp.local.dot(&centered)
                    })
                    .collect();
                (!blocks.is_empty()).then_some(blocks)
            };
            if let Some(cache) = self.ift_warm_start_cache.write().unwrap().as_mut() {
                cache.beta_original = beta_original.clone();
                cache.lambda_s_beta_blocks = lambda_s_beta_blocks;
            }
        }
    }

    #[inline]
    pub(crate) fn large_n_efs_single_loop_lane(&self) -> bool {
        (self.x.nrows() as f64) * (self.x.ncols() as f64) > LARGE_N_EFS_THRESHOLD
    }

    #[inline]
    pub(crate) fn efs_single_loop_cap_active(&self) -> bool {
        decode_efs_single_loop_cap(self.outer_inner_cap.load(Ordering::Relaxed)).is_some()
    }

    pub(crate) fn record_efs_single_loop_bias(
        &self,
        rho: &Array1<f64>,
        diagnostics: super::reml_outer_engine::EfsSingleLoopDiagnostics,
    ) -> Result<(), EstimationError> {
        if !self.efs_single_loop_cap_active() {
            return Ok(());
        }

        let owner = self as *const _ as usize;
        let mut state = EFS_SINGLE_LOOP_BIAS_GUARD.lock().unwrap();
        if state.owner != owner {
            state.owner = owner;
            state.consecutive = 0;
        }

        if diagnostics.bias_proxy >= EFS_SINGLE_LOOP_BIAS_THRESHOLD {
            state.consecutive = state.consecutive.saturating_add(1);
        } else {
            state.consecutive = 0;
        }

        log::info!(
            "[EFS-single-loop] bias_proxy={:.3e} gradient_residual={:.3e} inner_residual={:.3e} \
             |g|={:.3e} |step|inf={:.3e} consecutive={}/{} rho[..4]=[{}]",
            diagnostics.bias_proxy,
            diagnostics.gradient_residual,
            diagnostics.inner_residual,
            diagnostics.gradient_norm,
            diagnostics.step_inf_norm,
            state.consecutive,
            EFS_SINGLE_LOOP_BIAS_CONSECUTIVE_LIMIT,
            rho.iter()
                .take(4)
                .map(|v| format!("{v:.3}"))
                .collect::<Vec<_>>()
                .join(","),
        );

        if state.consecutive >= EFS_SINGLE_LOOP_BIAS_CONSECUTIVE_LIMIT {
            state.consecutive = 0;
            return Err(EstimationError::RemlOptimizationFailed(format!(
                "{} EFS single-loop bias guard fired: bias_proxy={:.3e} \
                 threshold={:.3e} consecutive_limit={} rho_dim={}",
                crate::rho_optimizer::EFS_FIRST_ORDER_FALLBACK_MARKER,
                diagnostics.bias_proxy,
                EFS_SINGLE_LOOP_BIAS_THRESHOLD,
                EFS_SINGLE_LOOP_BIAS_CONSECUTIVE_LIMIT,
                rho.len(),
            )));
        }

        Ok(())
    }

    pub(crate) fn analytic_outer_hessian_enabled(&self) -> bool {
        if self.large_n_efs_single_loop_lane() {
            log::info!(
                "[EFS-single-loop] large-n lane engaged: n={} p={} n*p={:.3e} threshold={:.3e}; \
                 declining analytic outer Hessian so the EFS fixed-point route runs first",
                self.x.nrows(),
                self.x.ncols(),
                (self.x.nrows() as f64) * (self.x.ncols() as f64),
                LARGE_N_EFS_THRESHOLD,
            );
            return false;
        }
        // The Tierney-Kadane fallback gate is no longer needed: the analytic
        // TK value, first ρ-derivative, AND second ρ-derivative paths are
        // implemented in `tierney_kadane_terms`, which now populates the
        // `hessian` field whenever the caller requests `ValueGradientHessian`.
        // The earlier gate (Firth + non-identity link → return false) was
        // kept during the manual conflict merge that landed the TK Hessian
        // implementation; it is now stale and was suppressing the analytic
        // path that was actually in place.
        //
        // Large-scale fallback: analytic outer Hessian is only safe when
        // the unified evaluator can express it as a matrix-free Hv operator
        // (`prefer_outer_hessian_operator`). Whenever that path is
        // unavailable at large scale, the dense `O(K²·n·p²)` LAML pairwise
        // assembly would run instead — route to BFGS.
        let n_obs = self.x.nrows();
        let p_dim = self.x.ncols();
        let k_outer = self.canonical_penalties.len();
        let operator_path_available =
            super::reml_outer_engine::prefer_outer_hessian_operator(n_obs, p_dim, k_outer);
        if n_obs > 50_000 && !operator_path_available {
            log::info!(
                "[standard-GAM] declining analytic outer Hessian for \
                 n={n_obs} p={p_dim} k={k_outer} (matrix-free operator \
                 path unavailable, dense LAML pairwise assembly is \
                 O(k²·n·p²)); routing to BFGS"
            );
            return false;
        }
        // Canonical-logit Firth fits have an exact Tierney-Kadane Hessian, but
        // its skewness correction contains an O(n²) row-pair contraction.  Keep
        // that exact curvature for small separation-rescue fits, where it is
        // cheap and useful; for ordinary multi-thousand-row binomial GAMs keep
        // the same Firth objective and analytic gradient while routing outer
        // curvature to BFGS instead of spending minutes on one Hessian probe
        // (#1575).
        if reml_robust_jeffreys_link(&self.config).is_some()
            && self.tk_correction_is_canonical_logit()
            && !Self::firth_tk_exact_hessian_scale_allows(n_obs, p_dim)
        {
            log::info!(
                "[standard-GAM] declining canonical-logit Firth exact outer Hessian for \
                 n={n_obs} p={p_dim} (row-pair TK Hessian work n²·p exceeds budget); \
                 routing to analytic-gradient BFGS"
            );
            return false;
        }
        // The analytic outer Hessian for a Firth fit folds in the Tierney-Kadane
        // curvature, whose c/d/e/f derivative arrays are implemented only for the
        // canonical Binomial Logit jet. #758 widened Firth to other Binomial
        // inverse links (Probit, CLogLog, SAS, …); those fits have no analytic TK
        // Hessian, so decline the analytic path and let BFGS drive the outer loop
        // off the (link-general) plain-Laplace gradient. Non-Firth fits are
        // unaffected — they never use the TK correction.
        if reml_robust_jeffreys_link(&self.config).is_some()
            && !self.tk_correction_is_canonical_logit()
        {
            return false;
        }
        true
    }

    pub(crate) fn firth_tk_exact_hessian_scale_allows(n_obs: usize, p_coeff: usize) -> bool {
        // Separate from `firth_problem_scale_allows`, which gates the O(n·p²)
        // dense Firth operator used by the inner solve and gradient.  The exact
        // TK Hessian-only path is O(n²·p) after the #1575 matvec hoist and needs
        // its own budget.
        const FIRTH_TK_EXACT_HESSIAN_MAX_ROW_PAIR_WORK: usize = 10_000_000;
        n_obs.saturating_mul(n_obs).saturating_mul(p_coeff)
            <= FIRTH_TK_EXACT_HESSIAN_MAX_ROW_PAIR_WORK
    }

    /// Whether the Tierney-Kadane outer correction (its value, ρ-gradient, and
    /// ρ-Hessian) applies to this fit. It is implemented only for canonical
    /// Binomial Logit Firth fits because its c/d/e/f derivative arrays consume
    /// the logit 5th-derivative jet (`logit_inverse_link_jet5`). Non-logit Firth
    /// fits skip the TK refinement and use plain Laplace REML, which is
    /// link-general; logit fits keep the full higher-order correction.
    pub(crate) fn tk_correction_is_canonical_logit(&self) -> bool {
        let spec = reml_spec(&self.config.likelihood);
        matches!(spec.response, ResponseFamily::Binomial)
            && matches!(spec.link, InverseLink::Standard(StandardLink::Logit))
            && self.runtime_mixture_link_state.is_none()
    }

    pub(crate) fn sparse_exact_beta_original(&self, pirls_result: &PirlsResult) -> Array1<f64> {
        match pirls_result.coordinate_frame {
            pirls::PirlsCoordinateFrame::OriginalSparseNative => {
                pirls_result.beta_transformed.as_ref().clone()
            }
            pirls::PirlsCoordinateFrame::TransformedQs => pirls_result
                .reparam_result
                .qs
                .dot(pirls_result.beta_transformed.as_ref()),
        }
    }

    pub(crate) fn bundle_matrix_in_original_basis(
        &self,
        pirls_result: &PirlsResult,
        matrix: &Array2<f64>,
    ) -> Array2<f64> {
        match pirls_result.coordinate_frame {
            pirls::PirlsCoordinateFrame::OriginalSparseNative => matrix.clone(),
            pirls::PirlsCoordinateFrame::TransformedQs => {
                let qs = &pirls_result.reparam_result.qs;
                let tmp = gam_linalg::faer_ndarray::fast_ab(qs, matrix);
                gam_linalg::faer_ndarray::fast_abt(&tmp, qs)
            }
        }
    }

    pub(crate) fn last_ridge_used(&self) -> Option<f64> {
        self.cache_manager
            .current_eval_bundle
            .read()
            .unwrap()
            .as_ref()
            .map(|bundle| bundle.ridge_passport.delta)
    }

    pub(crate) fn dense_penalty_logdet_derivs(
        &self,
        rho: &Array1<f64>,
        e_for_logdet: &Array2<f64>,
        penalty_roots: &[Array2<f64>],
        ridge_passport: RidgePassport,
        penalty_subspace: Option<&PenaltySubspace>,
        bundle: &EvalShared,
        mode: super::reml_outer_engine::EvalMode,
        free_basis: Option<&Array2<f64>>,
    ) -> Result<(usize, super::reml_outer_engine::PenaltyLogdetDerivs), EstimationError> {
        let logdet_s_start = std::time::Instant::now();
        let lambdas = rho.mapv(f64::exp);
        let ridge = ridge_passport.penalty_logdet_ridge();

        // Active-constraint projection consistency (#1380). When an active
        // shape/box constraint reduces the smooth to a constraint-free subspace
        // `Z` (columns = `null(A_active)`), the LAML cost's `½(log|H| − log|S|₊)`
        // pair must be evaluated on ONE common subspace. The Hessian side is
        // already projected (`h_for_operator = ZᵀHZ`, see `build_dense_assembly`),
        // but the penalty side below is assembled from the FULL-width canonical
        // penalties `Σ λ_k S_k` (rank/value/derivatives over the unprojected
        // p-space). That mismatch makes `log|S|₊` grow like `(rank_full/2)·Σρ`
        // while the projected `log|H|` grows only like `(dim(Z)/2)·Σρ`, so the
        // pair `½(log|H| − log|S|₊) → −((rank_full − dim Z)/2)·Σρ` decreases
        // without bound as λ → ∞ — the REML objective then rails λ to its
        // ceiling and the convex/concave smooth collapses to the flat linear
        // corner (EDF pinned, R² ≈ 0) even on a clean convex signal an
        // unconstrained `s(x)` recovers at R² ≈ 0.99.
        //
        // Fix: when `Z` is active, project each per-component penalty root onto
        // `Z` (`R_k ← R_k·Z`, so `S_k ↦ Zᵀ S_k Z`) and form the penalty logdet,
        // rank, and exact ρ-derivatives from the SAME projected components. Both
        // halves of the LAML pair then live in `range(Z)`; `log|S|₊` and
        // `log|H|` grow at the matched rate and the objective regains the proper
        // interior minimum in λ. The unconstrained / no-active-constraint path
        // (`free_basis = None`) is byte-for-byte unchanged.
        if let Some(z) = free_basis
            && !self.canonical_penalties.is_empty()
            && self.canonical_penalties.len() == rho.len()
        {
            // Frame consistency (#509 second face / completes #1380 & #1654).
            // The active-constraint free basis `z` is built from
            // `pr.linear_constraints_transformed` / `pr.beta_transformed`, so it
            // lives in the TRANSFORMED (post-Qs, and post box-reparam `T`) PIRLS
            // frame — the same frame as the Hessian side `h_for_operator = ZᵀHZ`
            // and `e_for_logdet = E_transformed·Z`. The penalty roots projected
            // here MUST live in that same transformed frame, otherwise
            // `Zᵀ S Z` mixes two coordinate systems and `log|S|₊` (and its
            // ρ-derivatives) desync from the projected `log|H|` whenever the
            // reparameterization is non-orthogonal. `self.canonical_penalties`
            // are the ORIGINAL-frame (pre-Qs) block-local roots; for a
            // shape-constrained smooth under the box reparameterization the
            // cumulative-sum `T` and the stabilizing `Qs` are both non-orthogonal,
            // so the original-frame projection produced a wrong outer REML
            // gradient (verified analytic ≠ central-difference), the Arc
            // trust-region rejected every step, and the monotone fit parked at
            // its under-smoothed seed. Use the TRANSFORMED-frame canonical roots
            // (`reparam_result.canonical_transformed`, the same per-component
            // roots every other transformed-frame penalty assembly reads) so both
            // halves of the LAML pair live in `range(Z)` of one frame. The roots
            // are orthogonal-invariant when `Qs = I` (no reparameterization), so
            // this is a no-op for the unconstrained / non-reparameterized paths.
            let transformed = &bundle.pirls_result.reparam_result.canonical_transformed;
            let projected_roots: Vec<Array2<f64>> = if transformed.len() == rho.len() {
                transformed
                    .iter()
                    .map(|penalty| penalty.full_width_root().dot(z))
                    .collect()
            } else {
                self.canonical_penalties
                    .iter()
                    .map(|penalty| penalty.full_width_root().dot(z))
                    .collect()
            };
            let (value, penalty_rank, det1, det2_full) = self
                .structural_penalty_logdet_value_and_derivatives(
                    &projected_roots,
                    &lambdas,
                    ridge,
                )?;
            log::info!(
                "[STAGE] logdet S (Z-projected) rho_dim={} penalty_rank={} elapsed={:.3}s",
                rho.len(),
                penalty_rank,
                logdet_s_start.elapsed().as_secs_f64(),
            );
            let det2 = if mode == super::reml_outer_engine::EvalMode::ValueGradientHessian {
                Some(det2_full)
            } else {
                None
            };
            return Ok((
                penalty_rank,
                super::reml_outer_engine::PenaltyLogdetDerivs {
                    value,
                    first: det1,
                    second: det2,
                },
            ));
        }
        // Value, rank, and ρ-derivatives of `log|Σ λ_k S_k|₊` ALL come from one
        // [`PenaltyPseudologdet`] (one eigendecomposition, one positive
        // eigenspace) so the analytic gradient differentiates exactly the value
        // the cost reports. A previous split — value from the structural-rank
        // `fixed_subspace_penalty_rank_and_logdet_from_subspace` (top-`rank`
        // eigenvalues) but derivatives from the eigenvalue-thresholded
        // `PenaltyPseudologdet` — let the two range over different eigenspaces
        // whenever a penalty eigenvalue sat near the ridge/noise band, which
        // sign-/scale-corrupted the GLM ρ-gradient against FD while the cost
        // stayed FD-consistent (#901: the canonical-empty Gaussian path was
        // immune because there BOTH value and derivative use the same
        // threshold). The penalty logdet is orthogonal-invariant, so the
        // original-basis canonical penalties give the same result as
        // transformed-basis roots.
        let (penalty_rank, log_det_s, det1, det2_full) = if let Some(ref kron) =
            self.kronecker_penalty_system
            && self.kronecker_factored.is_some()
            && kron.num_penalties() == rho.len()
        {
            let (logdet, rank, det1, det2) =
                kron.logdet_rank_and_derivatives(lambdas.as_slice().unwrap(), ridge);
            (rank, logdet, det1, det2)
        } else if !self.canonical_penalties.is_empty()
            && self.canonical_penalties.len() == rho.len()
        {
            let (value, rank, det1, det2) =
                self.structural_penalty_logdet_value_and_derivatives_block_local(&lambdas, bundle)?;
            (rank, value, det1, det2)
        } else if !penalty_roots.is_empty() {
            let (value, rank, det1, det2) = self.structural_penalty_logdet_value_and_derivatives(
                penalty_roots,
                &lambdas,
                ridge,
            )?;
            (rank, value, det1, det2)
        } else {
            // No Kronecker system, no canonical penalties (or a length mismatch),
            // and no penalty roots. This branch carries the combined penalty
            // `log|Σ λ_k S_k|₊` value+rank from the subspace eigensystem of the
            // assembled `E` (the eigenvalues of `EᵀE = Σ λ_k S_k`), but the
            // per-component matrices `S_k` are NOT available here, so the exact
            // ρ-derivatives `∂/∂ρ_k log|ΣλS|₊ = λ_k·tr((ΣλS)⁺ S_k)` (and the
            // second derivative) cannot be formed component-wise from this
            // object. The three branches above each own a per-component
            // representation (Kronecker marginal grid, `canonical_penalties`,
            // explicit roots) and produce the exact `det1`/`det2` from the SAME
            // positive eigenspace as the value.
            //
            // REACHABILITY: the cost path keeps `rho.len()` in lockstep with
            // `canonical_penalties.len()` (the smoothing-parameter coordinate is
            // one λ per canonical penalty), so any penalized fit (`rho.len() > 0`)
            // routes through the `canonical_penalties` branch above, where the
            // exact derivatives ARE formed. `canonical_penalties` is empty ONLY
            // when there are no smoothing parameters at all, i.e. `rho.len() == 0`
            // — and then the derivative arrays are genuinely empty (`(0,)`/`(0,0)`),
            // which is exact, not a desync. Returning identically-zero
            // `rho.len()`-sized derivatives for a ρ-dependent cost term would be
            // the #901 objective↔gradient desync bug class, so this branch must
            // NEVER reach the outer optimizer with `rho.len() > 0` while reporting
            // a zero gradient. Fail loud instead of silently mis-optimizing ρ if a
            // future penalty configuration ever lands a penalized fit here without
            // a per-component representation.
            let owned_subspace;
            let subspace = if let Some(penalty_subspace) = penalty_subspace {
                penalty_subspace
            } else {
                owned_subspace = self.compute_penalty_subspace(e_for_logdet, ridge_passport)?;
                &owned_subspace
            };
            let (rank, value) = self.fixed_subspace_penalty_rank_and_logdet_from_subspace(subspace);
            if !rho.is_empty() {
                crate::bail_invalid_estim!(
                    "penalty log|Σλ S|₊ ρ-derivatives unavailable: rho_dim={} but no Kronecker \
                     system, no canonical penalties, and no penalty roots provide a per-component \
                     S_k representation. The combined EᵀE subspace eigensystem yields the value \
                     and rank but cannot form the exact ρ-gradient λ_k·tr((ΣλS)⁺ S_k) — refusing \
                     to feed the outer REML optimizer an identically-zero gradient for a \
                     ρ-dependent cost term (#901 objective↔gradient desync). penalty_rank={}",
                    rho.len(),
                    rank
                );
            }
            (
                rank,
                value,
                Array1::zeros(rho.len()),
                Array2::zeros((rho.len(), rho.len())),
            )
        };
        log::info!(
            "[STAGE] logdet S rho_dim={} penalty_rank={} elapsed={:.3}s",
            rho.len(),
            penalty_rank,
            logdet_s_start.elapsed().as_secs_f64(),
        );

        let det2 = if mode == super::reml_outer_engine::EvalMode::ValueGradientHessian {
            Some(det2_full)
        } else {
            None
        };
        Ok((
            penalty_rank,
            super::reml_outer_engine::PenaltyLogdetDerivs {
                value: log_det_s,
                first: det1,
                second: det2,
            },
        ))
    }

    pub(crate) fn tk_shared_intermediates<S>(
        x_dense: &Array2<f64>,
        z: &Array2<f64>,
        c_array: &Array1<f64>,
        context: &str,
        h_inv_solve: &S,
    ) -> Result<TkSharedIntermediates, EstimationError>
    where
        S: Fn(&Array1<f64>) -> Result<Array1<f64>, EstimationError>,
    {
        let n = x_dense.nrows();
        let active_blocks = Self::tk_active_blocks(c_array);
        use rayon::prelude::*;
        let h_diag_vec: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|i| {
                let val = x_dense.row(i).dot(&z.column(i));
                if !val.is_finite() {
                    crate::bail_invalid_estim!(
                        "{context} produced non-finite leverage at row {i}: {val}"
                    );
                }
                Ok(val)
            })
            .collect::<Result<_, _>>()?;
        let h_diag = Array1::from(h_diag_vec);
        let m_vec = c_array * &h_diag;
        let x_m = gam_linalg::faer_ndarray::fast_atv(x_dense, &m_vec);
        let y = h_inv_solve(&x_m)?;
        Ok(TkSharedIntermediates {
            h_diag,
            x_m,
            y,
            active_blocks,
        })
    }

    pub(crate) fn tk_scalar_from_shared(
        x_dense: &Array2<f64>,
        z: &Array2<f64>,
        d_array: &Array1<f64>,
        shared: &TkSharedIntermediates,
        gram: &mut Array2<f64>,
    ) -> Result<f64, EstimationError> {
        let q_term = -0.125
            * d_array
                .iter()
                .zip(shared.h_diag.iter())
                .map(|(&d_i, &h_i)| d_i * h_i * h_i)
                .sum::<f64>();
        let t2_term = 0.125 * shared.x_m.dot(&shared.y);

        let mut t1_sum = 0.0_f64;
        for (j_block_idx, j_block) in shared.active_blocks.iter().enumerate() {
            let j0 = j_block.start;
            let j1 = j_block.end;
            for i_block in &shared.active_blocks[..=j_block_idx] {
                let i0 = i_block.start;
                let i1 = i_block.end;
                Self::tk_fill_gram_block(x_dense, z, i0, i1, j0, j1, gram);
                let mut block_sum = 0.0_f64;
                for &(bi, ci) in &i_block.entries {
                    for &(bj, cj) in &j_block.entries {
                        let kij = gram[[bi, bj]];
                        block_sum += ci * cj * kij * kij * kij;
                    }
                }
                t1_sum += if i0 == j0 { block_sum } else { 2.0 * block_sum };
            }
        }

        let value = q_term + t1_sum / 12.0 + t2_term;
        if !value.is_finite() {
            crate::bail_invalid_estim!(
                "Tierney-Kadane correction produced non-finite value: {value}"
            );
        }
        Ok(value)
    }

    pub(crate) fn tk_active_blocks(c_array: &Array1<f64>) -> Vec<TkActiveBlock> {
        let n = c_array.len();
        let mut blocks = Vec::with_capacity(n.div_ceil(TK_BLOCK_SIZE));
        for start in (0..n).step_by(TK_BLOCK_SIZE) {
            let end = (start + TK_BLOCK_SIZE).min(n);
            let entries = c_array
                .slice(s![start..end])
                .iter()
                .enumerate()
                .filter_map(|(offset, &value)| (value != 0.0).then_some((offset, value)))
                .collect::<Vec<_>>();
            if !entries.is_empty() {
                blocks.push(TkActiveBlock {
                    start,
                    end,
                    entries,
                });
            }
        }
        blocks
    }

    pub(crate) fn tk_active_weighted_trace(
        active_blocks: &[TkActiveBlock],
        x_vk: &Array1<f64>,
        lev_p: &Array1<f64>,
    ) -> f64 {
        let mut trace = 0.0;
        for block in active_blocks {
            for &(offset, c_i) in &block.entries {
                let i = block.start + offset;
                trace += c_i * x_vk[i] * lev_p[i];
            }
        }
        trace
    }

    pub(crate) fn tk_fill_gram_block(
        x_dense: &Array2<f64>,
        z: &Array2<f64>,
        i0: usize,
        i1: usize,
        j0: usize,
        j1: usize,
        gram: &mut Array2<f64>,
    ) {
        let rows = i1 - i0;
        let cols = j1 - j0;
        assert!(rows <= gram.nrows());
        assert!(cols <= gram.ncols());
        let x_block = x_dense.slice(s![i0..i1, ..]);
        let z_block = z.slice(s![.., j0..j1]);
        let mut target = gram.slice_mut(s![..rows, ..cols]);
        ndarray::linalg::general_mat_mul(1.0, &x_block, &z_block, 0.0, &mut target);
    }

    pub(crate) fn tk_fill_gram_block_entries_scalar(
        x_dense: &Array2<f64>,
        z: &Array2<f64>,
        i_block: &TkActiveBlock,
        j_block: &TkActiveBlock,
        gram: &mut Array2<f64>,
    ) {
        let p = x_dense.ncols();
        for &(bi, _) in &i_block.entries {
            let ii = i_block.start + bi;
            for &(bj, _) in &j_block.entries {
                let jj = j_block.start + bj;
                gram[[bi, bj]] = (0..p)
                    .map(|col| x_dense[[ii, col]] * z[[col, jj]])
                    .sum::<f64>();
            }
        }
    }

    pub(crate) fn tk_gradient_from_shared(
        x_dense: &Array2<f64>,
        z: &Array2<f64>,
        c_array: &Array1<f64>,
        d_array: &Array1<f64>,
        e_array: &Array1<f64>,
        tk_penalties: &[gam_terms::construction::CanonicalPenalty],
        lambdas: &[f64],
        ext_drifts: &[Array2<f64>],
        ext_eta_fixed: &[Option<Array1<f64>>],
        ext_x_fixed: &[Option<Array2<f64>>],
        x_vks: &[Array1<f64>],
        beta_dirs: &[Array1<f64>],
        firth_op: Option<&super::FirthDenseOperator>,
        shared: &TkSharedIntermediates,
        gram: &mut Array2<f64>,
    ) -> Result<Array1<f64>, EstimationError> {
        let n = x_dense.nrows();
        let p = x_dense.ncols();
        let k = tk_penalties.len();
        let total_k = k + ext_drifts.len();
        if x_vks.len() != total_k {
            crate::bail_invalid_estim!(
                "Tierney-Kadane correction internal gradient arity mismatch: {} response modes for {} coordinates",
                x_vks.len(),
                total_k
            );
        }
        if beta_dirs.len() != total_k {
            crate::bail_invalid_estim!(
                "Tierney-Kadane correction internal beta-direction arity mismatch: {} beta directions for {} coordinates",
                beta_dirs.len(),
                total_k
            );
        }
        let x_y = gam_linalg::faer_ndarray::fast_av(x_dense, &shared.y);

        let mut diag_combined = Array1::<f64>::zeros(n);
        ndarray::Zip::from(&mut diag_combined)
            .and(d_array)
            .and(&shared.h_diag)
            .and(c_array)
            .and(&x_y)
            .par_for_each(|o, &d, &h, &c, &xy| *o = d * h - c * xy);
        let chunk_len = (n
            / (rayon::current_num_threads()
                .saturating_mul(TK_CHUNK_OVERSUBSCRIBE)
                .max(1)))
        .clamp(TK_BLOCK_SIZE, TK_CHUNK_MAX_ROWS);
        let chunks = n.div_ceil(chunk_len);
        let mut p_total = (0..chunks)
            .into_par_iter()
            .fold(
                || Array2::<f64>::zeros((p, p)),
                |mut local, chunk_idx| {
                    let i0 = chunk_idx * chunk_len;
                    let i1 = (i0 + chunk_len).min(n);
                    for i in i0..i1 {
                        let wi = diag_combined[i];
                        if wi == 0.0 {
                            continue;
                        }
                        for a in 0..p {
                            let wa = wi * z[[a, i]];
                            for b in a..p {
                                let val = wa * z[[b, i]];
                                local[[a, b]] += val;
                                if a != b {
                                    local[[b, a]] += val;
                                }
                            }
                        }
                    }
                    local
                },
            )
            .reduce(
                || Array2::<f64>::zeros((p, p)),
                |mut left, right| {
                    left += &right;
                    left
                },
            );
        p_total.mapv_inplace(|v| 0.25 * v);
        for a in 0..p {
            for b in 0..p {
                p_total[[a, b]] -= 0.125 * shared.y[a] * shared.y[b];
            }
        }

        let active_pairs: Vec<(usize, usize)> = (0..shared.active_blocks.len())
            .flat_map(|j_block_idx| {
                (0..=j_block_idx).map(move |i_block_idx| (i_block_idx, j_block_idx))
            })
            .collect();
        let active_total = active_pairs
            .par_iter()
            .fold(
                || Array2::<f64>::zeros((p, p)),
                |mut local, &(i_block_idx, j_block_idx)| {
                    let i_block = &shared.active_blocks[i_block_idx];
                    let j_block = &shared.active_blocks[j_block_idx];
                    let sym_factor = if i_block_idx == j_block_idx { 1.0 } else { 2.0 };
                    for &(bi, ci) in &i_block.entries {
                        let ii = i_block.start + bi;
                        for &(bj, cj) in &j_block.entries {
                            let jj = j_block.start + bj;
                            let gij = (0..p)
                                .map(|col| x_dense[[ii, col]] * z[[col, jj]])
                                .sum::<f64>();
                            let weight = ci * cj * gij * gij;
                            let scale = -0.25 * weight * sym_factor;
                            if ii == jj {
                                for a in 0..p {
                                    let za = z[[a, ii]];
                                    for b in a..p {
                                        let val = scale * za * z[[b, ii]];
                                        local[[a, b]] += val;
                                        if a != b {
                                            local[[b, a]] += val;
                                        }
                                    }
                                }
                            } else {
                                let half_scale = 0.5 * scale;
                                for a in 0..p {
                                    let z_ii_a = z[[a, ii]];
                                    let z_jj_a = z[[a, jj]];
                                    for b in 0..p {
                                        local[[a, b]] += half_scale
                                            * (z_ii_a * z[[b, jj]] + z_jj_a * z[[b, ii]]);
                                    }
                                }
                            }
                        }
                    }
                    local
                },
            )
            .reduce(
                || Array2::<f64>::zeros((p, p)),
                |mut left, right| {
                    left += &right;
                    left
                },
            );
        p_total += &active_total;

        let xp = gam_linalg::faer_ndarray::fast_ab(x_dense, &p_total);
        let mut lev_p = Array1::<f64>::zeros(n);
        ndarray::Zip::from(&mut lev_p)
            .and(xp.rows())
            .and(x_dense.rows())
            .par_for_each(|o, xp_row, x_row| *o = xp_row.dot(&x_row));

        let mut gradient = Array1::<f64>::zeros(total_k);
        // The dominant `O(n²·p)` direct term for all `k` canonical directions
        // shares one gram assembly (the row-pair block is direction-independent):
        // fill each block once and reuse it, cutting the per-eval cost from
        // `O(k·n²·p)` to `O(n²·p)` bit-identically (#1575).
        let canonical_direct = Self::tk_direct_gradient_canonical_batched(
            x_dense, z, c_array, d_array, e_array, x_vks, k, shared, gram,
        )?;
        for idx in 0..k {
            let cp = &tk_penalties[idx];
            let r = &cp.col_range;
            let p_block = p_total.slice(s![r.start..r.end, r.start..r.end]);
            let rk_p = cp.root.dot(&p_block);
            let trace_ak_p = lambdas[idx]
                * (0..cp.rank())
                    .map(|row| rk_p.row(row).dot(&cp.root.row(row)))
                    .sum::<f64>();
            let correction_trace =
                Self::tk_active_weighted_trace(&shared.active_blocks, &x_vks[idx], &lev_p);
            let firth_trace =
                Self::tk_firth_beta_hessian_trace(firth_op, &beta_dirs[idx], &p_total)?;
            let direct = canonical_direct[idx];
            gradient[idx] = trace_ak_p - correction_trace + firth_trace + direct;
        }
        let ext_values = (0..ext_drifts.len())
            .into_par_iter()
            .map(|extra_idx| -> Result<(usize, f64), EstimationError> {
                let drift = &ext_drifts[extra_idx];
                if drift.raw_dim() != p_total.raw_dim() {
                    crate::bail_invalid_estim!(
                        "Tierney-Kadane ext penalty drift shape mismatch: expected {}x{}, got {}x{}",
                        p,
                        p,
                        drift.nrows(),
                        drift.ncols()
                    );
                }
                let mut trace_ak_p = 0.0;
                for row in 0..p {
                    for col in 0..p {
                        trace_ak_p += drift[[row, col]] * p_total[[col, row]];
                    }
                }
                let x_vk_idx = k + extra_idx;
                let correction_trace =
                    Self::tk_active_weighted_trace(&shared.active_blocks, &x_vks[x_vk_idx], &lev_p);
                let firth_trace =
                    Self::tk_firth_beta_hessian_trace(firth_op, &beta_dirs[x_vk_idx], &p_total)?;
                let mut eta_total = x_vks[x_vk_idx].mapv(|value| -value);
                if let Some(eta_fixed) = ext_eta_fixed
                    .get(extra_idx)
                    .and_then(|value| value.as_ref())
                {
                    if eta_fixed.len() != n {
                        crate::bail_invalid_estim!(
                            "Tierney-Kadane ext fixed eta length mismatch: expected {}, got {}",
                            n,
                            eta_fixed.len()
                        );
                    }
                    eta_total += eta_fixed;
                }
                let x_fixed = ext_x_fixed.get(extra_idx).and_then(|value| value.as_ref());
                if let Some(x_fixed) = x_fixed
                    && x_fixed.raw_dim() != x_dense.raw_dim() {
                        crate::bail_invalid_estim!(
                            "Tierney-Kadane ext fixed design shape mismatch: expected {}x{}, got {}x{}",
                            x_dense.nrows(),
                            x_dense.ncols(),
                            x_fixed.nrows(),
                            x_fixed.ncols()
                        );
                    }
                let mut local_gram = Array2::<f64>::zeros((TK_BLOCK_SIZE, TK_BLOCK_SIZE));
                let direct = Self::tk_direct_gradient_from_cd_and_design(
                    x_dense,
                    z,
                    c_array,
                    d_array,
                    e_array,
                    &eta_total,
                    x_fixed,
                    shared,
                    &mut local_gram,
                    false,
                )?;
                Ok((x_vk_idx, trace_ak_p - correction_trace + firth_trace + direct))
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (idx, value) in ext_values {
            gradient[idx] = value;
        }

        for g in gradient.iter_mut() {
            if !g.is_finite() {
                crate::bail_invalid_estim!(
                    "Tierney-Kadane correction produced a non-finite gradient entry"
                );
            }
        }
        Ok(gradient)
    }

    pub(crate) fn tk_firth_beta_hessian_trace(
        firth_op: Option<&super::FirthDenseOperator>,
        beta_dir: &Array1<f64>,
        p_total: &Array2<f64>,
    ) -> Result<f64, EstimationError> {
        let Some(firth_op) = firth_op else {
            return Ok(0.0);
        };
        if beta_dir.len() != p_total.nrows() {
            crate::bail_invalid_estim!(
                "Tierney-Kadane Firth beta-direction length mismatch: expected {}, got {}",
                p_total.nrows(),
                beta_dir.len()
            );
        }
        let deta = gam_linalg::faer_ndarray::fast_av(&firth_op.x_dense, beta_dir);
        let dir = firth_op.direction_from_deta(deta);
        let hphi = firth_op.hphi_direction(&dir);
        if hphi.raw_dim() != p_total.raw_dim() {
            crate::bail_invalid_estim!(
                "Tierney-Kadane Firth Hessian derivative shape mismatch: expected {}x{}, got {}x{}",
                p_total.nrows(),
                p_total.ncols(),
                hphi.nrows(),
                hphi.ncols()
            );
        }

        let mut trace = 0.0;
        for row in 0..hphi.nrows() {
            for col in 0..hphi.ncols() {
                trace -= hphi[[row, col]] * p_total[[col, row]];
            }
        }
        Ok(trace)
    }

    /// Direct analytic derivative of the TK scalar through the per-row
    /// curvature carriers and design rows, with `H⁻¹` held fixed.
    ///
    /// For
    ///   V_TK = -1/8 Σ d_i h_i^2 + 1/12 Σ c_i c_j K_ij^3 + 1/8 qᵀH⁻¹q,
    /// where `h_i = K_ii`, `K_ij = x_iᵀH⁻¹x_j`, and `q = Xᵀ(c ⊙ h)`,
    /// the direct terms are:
    ///   c' = d ⊙ η',  d' = e ⊙ η',
    ///   h'_i = 2 x'ᵢᵀH⁻¹x_i,
    ///   K'_ij = x'ᵢᵀH⁻¹x_j + x'ⱼᵀH⁻¹x_i,
    ///   q' = Xᵀ(c'⊙h + c⊙h') + X'ᵀ(c⊙h).
    /// The remaining `H`-drift part of dV_TK/dθ is assembled by
    /// `tk_gradient_from_shared`.
    /// Batched direct TK ρ-gradient for the `k` canonical smoothing directions,
    /// sharing the dominant `O(n²·p)` gram assembly.
    ///
    /// The per-direction [`tk_direct_gradient_from_cd_and_design`] refills the
    /// row-pair gram block `Gᵢⱼ = xᵢᵀH⁻¹xⱼ` (`tk_fill_gram_block`) for EVERY
    /// canonical direction, yet that block depends only on `(x_dense, z)` — it is
    /// identical across directions (only the per-row `c/c'` weights differ). For
    /// the default-ON binomial/logit Firth path this gram assembly is the
    /// dominant `O(n²·p)` cost of every outer gradient evaluation, so refilling
    /// it `k` times is a `k×` waste (#1575). Here each gram block is filled ONCE
    /// and consumed by all `k` directions, cutting the dominant term from
    /// `O(k·n²·p)` to `O(n²·p) + O(k·n²)` with no accuracy change.
    ///
    /// The result is BIT-IDENTICAL to calling
    /// `tk_direct_gradient_from_cd_and_design` per direction (canonical case:
    /// `x_fixed = None`, dense kernels): each direction accumulates `c_term`
    /// over the SAME `(j_block outer, i_block ≤ j_block)` order and the SAME
    /// entry order, reading the identical `tk_fill_gram_block` values. The active
    /// block set is the UNION over directions (a row participates if `c_array` OR
    /// any direction's `c'` is nonzero); a row absent from a given direction's
    /// own active set has `c = c' = 0` there and contributes exactly `0.0` to its
    /// `c_term`, and `+= sym·0.0` at the union's extra block-pairs leaves the
    /// per-direction running sum unchanged — so `direct[idx]` equals the
    /// per-direction value byte-for-byte. Locked by
    /// `tk_direct_gradient_canonical_batched_matches_per_direction_bit_identical_1575`.
    pub(crate) fn tk_direct_gradient_canonical_batched(
        x_dense: &Array2<f64>,
        z: &Array2<f64>,
        c_array: &Array1<f64>,
        d_array: &Array1<f64>,
        e_array: &Array1<f64>,
        x_vks: &[Array1<f64>],
        k: usize,
        shared: &TkSharedIntermediates,
        gram: &mut Array2<f64>,
    ) -> Result<Vec<f64>, EstimationError> {
        let n = x_dense.nrows();

        // Per-direction working weights and the cheap (non-gram) gradient terms.
        // These replicate the canonical-path arithmetic of
        // `tk_direct_gradient_from_cd_and_design` (x_fixed = None ⇒ h' = 0, so
        // the `-0.25·Σ d·h·h'` term and the `X'ᵀ(c⊙h)` / `c⊙h'` contributions
        // vanish identically) term-for-term, in the identical association
        // `d_term + c_term + q_term`.
        let mut c_primes: Vec<Array1<f64>> = Vec::with_capacity(k);
        let mut d_terms = vec![0.0_f64; k];
        let mut q_terms = vec![0.0_f64; k];
        for idx in 0..k {
            let eta_total = x_vks[idx].mapv(|value| -value);
            let mut c_prime = Array1::<f64>::zeros(n);
            let mut d_prime = Array1::<f64>::zeros(n);
            ndarray::Zip::from(&mut c_prime)
                .and(d_array)
                .and(&eta_total)
                .par_for_each(|out, &d, &eta| *out = d * eta);
            ndarray::Zip::from(&mut d_prime)
                .and(e_array)
                .and(&eta_total)
                .par_for_each(|out, &e, &eta| *out = e * eta);
            // q' = Xᵀ(c'⊙h)  (h' = 0 for canonical penalties).
            let q_weight_prime = &c_prime * &shared.h_diag;
            let q_prime = gam_linalg::faer_ndarray::fast_atv(x_dense, &q_weight_prime);
            q_terms[idx] = 0.25 * q_prime.dot(&shared.y);
            d_terms[idx] = -0.125
                * d_prime
                    .iter()
                    .zip(shared.h_diag.iter())
                    .map(|(&dp, &h)| dp * h * h)
                    .sum::<f64>();
            c_primes.push(c_prime);
        }

        // Union active-block structure over all directions (see doc comment):
        // filling each gram block once and letting every direction sum over the
        // union is bit-identical to per-direction blocks.
        let mut blocks: Vec<TkActiveBlock> = Vec::with_capacity(n.div_ceil(TK_BLOCK_SIZE));
        for start in (0..n).step_by(TK_BLOCK_SIZE) {
            let end = (start + TK_BLOCK_SIZE).min(n);
            let mut entries: Vec<(usize, f64)> = Vec::new();
            for offset in 0..(end - start) {
                let row = start + offset;
                let active = c_array[row] != 0.0 || c_primes.iter().any(|cp| cp[row] != 0.0);
                if active {
                    entries.push((offset, 0.0));
                }
            }
            if !entries.is_empty() {
                blocks.push(TkActiveBlock { start, end, entries });
            }
        }

        let mut c_terms = vec![0.0_f64; k];
        for (j_block_idx, j_block) in blocks.iter().enumerate() {
            let j0 = j_block.start;
            let j1 = j_block.end;
            for i_block in &blocks[..=j_block_idx] {
                let i0 = i_block.start;
                let i1 = i_block.end;
                // Fill the gram block ONCE; consume it for every direction.
                Self::tk_fill_gram_block(x_dense, z, i0, i1, j0, j1, gram);
                let sym_factor = if i0 == j0 { 1.0 } else { 2.0 };
                for idx in 0..k {
                    let cp = &c_primes[idx];
                    let mut block_sum = 0.0_f64;
                    for &(bi, _) in &i_block.entries {
                        let ii = i0 + bi;
                        let ci = c_array[ii];
                        let cpi = cp[ii];
                        for &(bj, _) in &j_block.entries {
                            let jj = j0 + bj;
                            let cj = c_array[jj];
                            let gij = gram[[bi, bj]];
                            let cpj = cp[jj];
                            let c_direct = (cpi * cj + ci * cpj) * gij * gij * gij / 12.0;
                            block_sum += c_direct;
                        }
                    }
                    c_terms[idx] += sym_factor * block_sum;
                }
            }
        }

        let mut out = Vec::with_capacity(k);
        for idx in 0..k {
            let value = d_terms[idx] + c_terms[idx] + q_terms[idx];
            if !value.is_finite() {
                crate::bail_invalid_estim!(
                    "Tierney-Kadane direct c/d derivative produced non-finite value: {value}"
                );
            }
            out.push(value);
        }
        Ok(out)
    }

    pub(crate) fn tk_direct_gradient_from_cd_and_design(
        x_dense: &Array2<f64>,
        z: &Array2<f64>,
        c_array: &Array1<f64>,
        d_array: &Array1<f64>,
        e_array: &Array1<f64>,
        eta_total: &Array1<f64>,
        x_fixed: Option<&Array2<f64>>,
        shared: &TkSharedIntermediates,
        gram: &mut Array2<f64>,
        use_dense_kernels: bool,
    ) -> Result<f64, EstimationError> {
        let n = x_dense.nrows();
        if eta_total.len() != n || e_array.len() != n {
            crate::bail_invalid_estim!(
                "Tierney-Kadane direct derivative length mismatch: n={}, eta={}, e={}",
                n,
                eta_total.len(),
                e_array.len()
            );
        }

        let mut c_prime = Array1::<f64>::zeros(n);
        let mut d_prime = Array1::<f64>::zeros(n);
        ndarray::Zip::from(&mut c_prime)
            .and(d_array)
            .and(eta_total)
            .par_for_each(|out, &d, &eta| *out = d * eta);
        ndarray::Zip::from(&mut d_prime)
            .and(e_array)
            .and(eta_total)
            .par_for_each(|out, &e, &eta| *out = e * eta);

        let mut h_prime = Array1::<f64>::zeros(n);
        let mut design_q_prime = Array1::<f64>::zeros(x_dense.ncols());
        let has_design_deriv = x_fixed.is_some();
        if let Some(x_theta) = x_fixed {
            let ch = c_array * &shared.h_diag;
            design_q_prime += &gam_linalg::faer_ndarray::fast_atv(x_theta, &ch);
            ndarray::Zip::from(&mut h_prime)
                .and(x_theta.rows())
                .and(z.columns())
                .par_for_each(|o, xr, zc| *o = 2.0 * xr.dot(&zc));
        }

        let q_weight_prime = &(&c_prime * &shared.h_diag) + &(c_array * &h_prime);
        let q_prime = gam_linalg::faer_ndarray::fast_atv(x_dense, &q_weight_prime) + design_q_prime;
        let q_term_prime = 0.25 * q_prime.dot(&shared.y);

        let d_term_prime = -0.125
            * d_prime
                .iter()
                .zip(shared.h_diag.iter())
                .map(|(&dp, &h)| dp * h * h)
                .sum::<f64>()
            - 0.25
                * d_array
                    .iter()
                    .zip(shared.h_diag.iter())
                    .zip(h_prime.iter())
                    .map(|((&d, &h), &hp)| d * h * hp)
                    .sum::<f64>();

        let direct_blocks = Self::tk_cd_direct_active_blocks(c_array, &c_prime);
        let mut c_term_prime = 0.0_f64;
        // Hoist per-block-pair scratch outside the double loop. Both buffers
        // are sized at TK_BLOCK_SIZE × TK_BLOCK_SIZE; per-iteration we take
        // sub-views matching the actual (rows, cols) of the current block.
        let mut block_scratch = Array2::<f64>::zeros((TK_BLOCK_SIZE, TK_BLOCK_SIZE));
        let mut reverse_scratch = Array2::<f64>::zeros((TK_BLOCK_SIZE, TK_BLOCK_SIZE));
        for (j_block_idx, j_block) in direct_blocks.iter().enumerate() {
            let j0 = j_block.start;
            let j1 = j_block.end;
            for i_block in &direct_blocks[..=j_block_idx] {
                let i0 = i_block.start;
                let i1 = i_block.end;
                if use_dense_kernels {
                    Self::tk_fill_gram_block(x_dense, z, i0, i1, j0, j1, gram);
                } else {
                    Self::tk_fill_gram_block_entries_scalar(x_dense, z, i_block, j_block, gram);
                }

                let design_gram_active = if has_design_deriv && use_dense_kernels {
                    let x_theta = x_fixed.expect("design derivative checked above");
                    let rows = i1 - i0;
                    let cols = j1 - j0;
                    let mut block = block_scratch.slice_mut(s![..rows, ..cols]);
                    let x_theta_i = x_theta.slice(s![i0..i1, ..]);
                    let z_j = z.slice(s![.., j0..j1]);
                    ndarray::linalg::general_mat_mul(1.0, &x_theta_i, &z_j, 0.0, &mut block);
                    let mut reverse = reverse_scratch.slice_mut(s![..cols, ..rows]);
                    let x_theta_j = x_theta.slice(s![j0..j1, ..]);
                    let z_i = z.slice(s![.., i0..i1]);
                    ndarray::linalg::general_mat_mul(1.0, &x_theta_j, &z_i, 0.0, &mut reverse);
                    true
                } else {
                    false
                };

                let mut block_sum = 0.0_f64;
                for &(bi, _) in &i_block.entries {
                    let ii = i0 + bi;
                    let ci = c_array[ii];
                    let cpi = c_prime[ii];
                    for &(bj, _) in &j_block.entries {
                        let jj = j0 + bj;
                        let cj = c_array[jj];
                        let gij = gram[[bi, bj]];
                        let cpj = c_prime[jj];
                        let c_direct = (cpi * cj + ci * cpj) * gij * gij * gij / 12.0;
                        let k_direct = if design_gram_active {
                            let kp = block_scratch[[bi, bj]] + reverse_scratch[[bj, bi]];
                            0.25 * ci * cj * gij * gij * kp
                        } else if let Some(x_theta) = x_fixed {
                            let kp = (0..x_dense.ncols())
                                .map(|col| {
                                    x_theta[[ii, col]] * z[[col, jj]]
                                        + x_theta[[jj, col]] * z[[col, ii]]
                                })
                                .sum::<f64>();
                            0.25 * ci * cj * gij * gij * kp
                        } else {
                            0.0
                        };
                        block_sum += c_direct + k_direct;
                    }
                }
                let sym_factor = if i0 == j0 { 1.0 } else { 2.0 };
                c_term_prime += sym_factor * block_sum;
            }
        }

        let value = d_term_prime + c_term_prime + q_term_prime;
        if !value.is_finite() {
            crate::bail_invalid_estim!(
                "Tierney-Kadane direct c/d derivative produced non-finite value: {value}"
            );
        }
        Ok(value)
    }

    pub(crate) fn tk_cd_direct_active_blocks(
        c_array: &Array1<f64>,
        c_prime: &Array1<f64>,
    ) -> Vec<TkActiveBlock> {
        let n = c_array.len();
        let mut blocks = Vec::with_capacity(n.div_ceil(TK_BLOCK_SIZE));
        for start in (0..n).step_by(TK_BLOCK_SIZE) {
            let end = (start + TK_BLOCK_SIZE).min(n);
            let mut entries = Vec::new();
            for offset in 0..(end - start) {
                let idx = start + offset;
                if c_array[idx] != 0.0 || c_prime[idx] != 0.0 {
                    entries.push((offset, 0.0));
                }
            }
            if !entries.is_empty() {
                blocks.push(TkActiveBlock {
                    start,
                    end,
                    entries,
                });
            }
        }
        blocks
    }

    pub(crate) fn tk_penalty_dense(
        cp: &gam_terms::construction::CanonicalPenalty,
        lambda: f64,
        p: usize,
    ) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((p, p));
        let r = &cp.col_range;
        for a in 0..cp.block_dim() {
            for b in 0..cp.block_dim() {
                let mut val = 0.0;
                for row in 0..cp.rank() {
                    val += cp.root[[row, a]] * cp.root[[row, b]];
                }
                out[[r.start + a, r.start + b]] = lambda * val;
            }
        }
        out
    }

    pub(crate) fn tk_xt_diag_x(x_dense: &Array2<f64>, diag: &Array1<f64>) -> Array2<f64> {
        let mut weighted = Array2::<f64>::zeros(x_dense.raw_dim());
        Self::xt_diag_x_dense_into(x_dense, diag, &mut weighted)
    }

    pub(crate) fn tk_hessian_rho_canonical_logit<S>(
        x_dense: &Array2<f64>,
        c_array: &Array1<f64>,
        d_array: &Array1<f64>,
        e_array: &Array1<f64>,
        f_array: &Array1<f64>,
        tk_penalties: &[gam_terms::construction::CanonicalPenalty],
        lambdas: &[f64],
        beta: &Array1<f64>,
        firth_op: Option<&super::FirthDenseOperator>,
        h_inv_solve: &S,
    ) -> Result<Array2<f64>, EstimationError>
    where
        S: Fn(&Array1<f64>) -> Result<Array1<f64>, EstimationError>,
    {
        let n = x_dense.nrows();
        let p = x_dense.ncols();
        let k = tk_penalties.len();
        if k == 0 {
            return Ok(Array2::zeros((0, 0)));
        }
        if c_array.len() != n || d_array.len() != n || e_array.len() != n || f_array.len() != n {
            crate::bail_invalid_estim!(
                "Tierney-Kadane Hessian derivative arrays have inconsistent lengths"
            );
        }

        let mut k_mat = Array2::<f64>::zeros((p, p));
        for col in 0..p {
            let mut rhs = Array1::<f64>::zeros(p);
            rhs[col] = 1.0;
            let sol = h_inv_solve(&rhs)?;
            k_mat.column_mut(col).assign(&sol);
        }
        gam_linalg::matrix::symmetrize_in_place(&mut k_mat);

        let mut a_mats = Vec::with_capacity(k);
        let mut v = Vec::with_capacity(k);
        let mut eta_i = Vec::with_capacity(k);
        for idx in 0..k {
            let a = Self::tk_penalty_dense(&tk_penalties[idx], lambdas[idx], p);
            let rhs = a.dot(beta);
            let vi = h_inv_solve(&rhs)?;
            let bi = vi.mapv(|value| -value);
            let ei = gam_linalg::faer_ndarray::fast_av(x_dense, &bi);
            a_mats.push(a);
            v.push(vi);
            eta_i.push(ei);
        }

        // Each per-penalty Firth direction depends only on eta_i[idx] (the
        // β-direction δη for penalty idx), so it is constant across both the
        // h_i loop below and every (i,j) pair in the second-derivative loop.
        // Building it once here avoids the O(k²) redundant rebuilds that
        // hphisecond_direction_apply otherwise triggered (each rebuild is an
        // O(n·r²) reduced-Gram), which dominate the Firth outer-Hessian cost
        // for binomial/logit REML (#1575). This is exact: direction_from_deta
        // is a pure function of (op, eta_i[idx]).
        //
        // The k builds are independent and each pays two O(n·r²) reduced-Gram
        // GEMMs (`reducedweighted_gram` + `reduced_diag_gram`), so — as with the
        // h_i / h_ij loops below — fan them across the Rayon pool when there is
        // more than one direction AND more than one thread, with the
        // `with_nested_parallel` guard pinning each build's faer GEMMs to
        // `Par::Seq` (no rayon×faer oversubscription). Index-ordered collection
        // keeps the Vec identical to the serial build; at fixture scale the
        // inner GEMMs are already `Par::Seq`, so the bits are unchanged (#1575).
        let firth_dir_i: Vec<super::FirthDirection> = match firth_op {
            Some(op) if k > 1 && rayon::current_num_threads() > 1 => {
                use rayon::prelude::*;
                eta_i
                    .par_iter()
                    .map(|e| {
                        gam_problem::with_nested_parallel(|| op.direction_from_deta(e.clone()))
                    })
                    .collect()
            }
            Some(op) => eta_i
                .iter()
                .map(|e| op.direction_from_deta(e.clone()))
                .collect(),
            None => Vec::new(),
        };

        // First-derivative blocks H'[idx] — k independent full-data passes, each
        // dominated (Firth path) by the O(n·r²·p) `hphi_direction` reduced-Gram
        // apply.
        //
        // When there are several independent passes AND more than one thread, fan
        // them across Rayon with the nested-BLAS guard (`with_nested_parallel`
        // pins each pass's faer GEMMs to `Par::Seq`), spreading the passes over
        // cores without rayon×faer oversubscription. With only one pass (k=1)
        // there is nothing to spread and pinning the inner GEMMs to `Par::Seq`
        // would instead STRIP the faer-level parallelism the serial path enjoys,
        // so we fall back to the plain serial loop, leaving the inner GEMMs free
        // to use the global pool. Either way the assembled blocks are identical:
        // index-ordered collection in the parallel arm, and at fixture scale the
        // inner GEMMs are already `Par::Seq` so the bits are unchanged (#1575).
        let fan_units = k > 1 && rayon::current_num_threads() > 1;
        let compute_h_i = |idx: usize| -> Array2<f64> {
            let diag = c_array * &eta_i[idx];
            let mut h = &a_mats[idx] + &Self::tk_xt_diag_x(x_dense, &diag);
            if let Some(op) = firth_op {
                h -= &op.hphi_direction(&firth_dir_i[idx]);
            }
            gam_linalg::matrix::symmetrize_in_place(&mut h);
            h
        };
        let h_i: Vec<Array2<f64>> = if fan_units {
            use rayon::prelude::*;
            (0..k)
                .into_par_iter()
                .map(|idx| gam_problem::with_nested_parallel(|| compute_h_i(idx)))
                .collect()
        } else {
            (0..k).map(compute_h_i).collect()
        };

        // The mixed second directional derivative D²H_φ[u,v] is evaluated for
        // every (i,j) penalty pair below against the SAME identity rhs. Its
        // single-index sub-blocks therefore have only k distinct values; cache
        // them once here so the O(k²) pair loop reuses them instead of rebuilding
        // an O(n·r²·p) reduced Hadamard-Gram for each pair. Exact / bit-identical
        // to per-pair hphisecond_direction_apply(.., &eye) (#1575).
        let firth_second_eye_cache = firth_op.map(|op| op.tk_second_direction_eye_cache(&firth_dir_i));

        let mut beta_ij: Vec<Vec<Array1<f64>>> = (0..k)
            .map(|_| (0..k).map(|_| Array1::<f64>::zeros(p)).collect())
            .collect();
        let mut h_ij: Vec<Vec<Array2<f64>>> = (0..k)
            .map(|_| (0..k).map(|_| Array2::<f64>::zeros((p, p))).collect())
            .collect();
        for i in 0..k {
            for j in 0..=i {
                let mut rhs = h_i[j].dot(&v[i]);
                rhs += &a_mats[i].dot(&v[j]);
                if i == j {
                    rhs -= &a_mats[i].dot(beta);
                }
                let bij = h_inv_solve(&rhs)?;
                beta_ij[i][j] = bij.clone();
                beta_ij[j][i] = bij;
            }
        }
        // Mixed second-derivative blocks H''[i,j] for every upper-triangle pair.
        // Each pair is an INDEPENDENT full-data pass — its cost is dominated, for
        // the default-ON binomial/logit Firth path, by ~5 O(n·r²·p) reduced
        // Hadamard-Gram applies inside `hphisecond_direction_apply_eye_cached`
        // (#1575). The pairs share only immutable state (`op`, the cached
        // directions/eye-blocks, `beta_ij`, the derivative arrays), so we fan the
        // `k(k+1)/2` pairs across the Rayon pool; the `with_nested_parallel` guard
        // pins each pair's faer GEMMs to `Par::Seq`, spreading the pairs over
        // cores without rayon×faer oversubscription. The cheap reduction back into
        // `h_ij` stays serial in index order, so the result is identical to the
        // original double `for` loop. At the small regression-fixture scales the
        // inner GEMMs are already `Par::Seq` (below faer's flop threshold), so
        // forcing seq there changes nothing and the assembled blocks are
        // bit-for-bit unchanged; the win is purely on the large-n perf path.
        //
        // As with `h_i` above, fan out only when there is more than one pair AND
        // more than one thread; a single pair (k=1) runs serially so the inner
        // GEMMs keep the global faer pool rather than being pinned to `Par::Seq`.
        let h_pairs: Vec<(usize, usize)> =
            (0..k).flat_map(|i| (0..=i).map(move |j| (i, j))).collect();
        let compute_h_pair = |&(i, j): &(usize, usize)| -> Array2<f64> {
            let eta_ij = gam_linalg::faer_ndarray::fast_av(x_dense, &beta_ij[i][j]);
            let diag = c_array * &eta_ij + &(d_array * &(&eta_i[i] * &eta_i[j]));
            let mut h = Self::tk_xt_diag_x(x_dense, &diag);
            if i == j {
                h += &a_mats[i];
            }
            if let Some(op) = firth_op {
                let dir_ij = op.direction_from_deta(eta_ij);
                h -= &op.hphi_direction(&dir_ij);
                // Reuse the per-penalty directions built once above and the
                // single-index second-derivative sub-blocks cached once above,
                // instead of rebuilding them per (i,j) pair (#1575).
                let cache = firth_second_eye_cache.as_ref().expect(
                    "firth second-direction eye cache present when firth_op is Some",
                );
                h -= &op.hphisecond_direction_apply_eye_cached(cache, &firth_dir_i, i, j);
            }
            gam_linalg::matrix::symmetrize_in_place(&mut h);
            h
        };
        let fan_pairs = h_pairs.len() > 1 && rayon::current_num_threads() > 1;
        let h_blocks: Vec<Array2<f64>> = if fan_pairs {
            use rayon::prelude::*;
            h_pairs
                .par_iter()
                .map(|pair| gam_problem::with_nested_parallel(|| compute_h_pair(pair)))
                .collect()
        } else {
            h_pairs.iter().map(compute_h_pair).collect()
        };
        for (&(i, j), h) in h_pairs.iter().zip(h_blocks.into_iter()) {
            h_ij[i][j] = h.clone();
            h_ij[j][i] = h;
        }

        let mut k_i = Vec::with_capacity(k);
        for i in 0..k {
            k_i.push(-k_mat.dot(&h_i[i]).dot(&k_mat));
        }
        let mut k_ij: Vec<Vec<Array2<f64>>> = (0..k)
            .map(|_| (0..k).map(|_| Array2::<f64>::zeros((p, p))).collect())
            .collect();
        for i in 0..k {
            for j in 0..=i {
                let kij = k_mat.dot(&h_i[j]).dot(&k_mat).dot(&h_i[i]).dot(&k_mat)
                    + k_mat.dot(&h_i[i]).dot(&k_mat).dot(&h_i[j]).dot(&k_mat)
                    - k_mat.dot(&h_ij[i][j]).dot(&k_mat);
                k_ij[i][j] = kij.clone();
                k_ij[j][i] = kij;
            }
        }

        #[derive(Clone)]
        struct Jet {
            pub(crate) v: f64,
            pub(crate) g: Array1<f64>,
            pub(crate) h: Array2<f64>,
        }
        impl Jet {
            pub(crate) fn constant(v: f64, k: usize) -> Self {
                Self {
                    v,
                    g: Array1::zeros(k),
                    h: Array2::zeros((k, k)),
                }
            }
            pub(crate) fn add(&self, other: &Self) -> Self {
                Self {
                    v: self.v + other.v,
                    g: &self.g + &other.g,
                    h: &self.h + &other.h,
                }
            }
            pub(crate) fn scale(&self, a: f64) -> Self {
                Self {
                    v: self.v * a,
                    g: self.g.mapv(|x| x * a),
                    h: self.h.mapv(|x| x * a),
                }
            }
            pub(crate) fn mul(&self, other: &Self) -> Self {
                let k = self.g.len();
                let mut h = &self.h * other.v + &other.h * self.v;
                for i in 0..k {
                    for j in 0..k {
                        h[[i, j]] += self.g[i] * other.g[j] + other.g[i] * self.g[j];
                    }
                }
                Self {
                    v: self.v * other.v,
                    g: &self.g * other.v + &other.g * self.v,
                    h,
                }
            }
            pub(crate) fn square(&self) -> Self {
                self.mul(self)
            }
            pub(crate) fn cube(&self) -> Self {
                self.mul(self).mul(self)
            }
        }

        // K xⱼ, Kᵢ xⱼ and Kᵢⱼ xⱼ depend only on the row j, yet the O(n²)
        // skewness double loop below recomputed each of them afresh for every
        // outer row i (and the per-row `hdiag` jet computes the very same
        // matvecs). For the default-ON binomial/logit Firth path this exact TK
        // Hessian runs at n up to FIRTH_MAX_OBSERVATIONS (20_000), so that inner
        // O(p²) matvec, repeated n² times, dominates the whole evaluation.
        //
        // Hoist the row-local matvecs once here (n gemvs each), so the diagonal
        // jet and every (i,j) inner iteration reduce to an O(p) `xᵢ · (K xⱼ)`
        // dot — turning the dominant term from O(n²·(1+k+k²)·p²) into
        // O(n²·(1+k+k²)·p) plus an O(n·(1+k+k²)·p²) precompute. This is exact:
        // each cached vector is the identical `Matrix::dot` matvec the inline
        // code performed, and the irow-outer / jrow-inner accumulation order is
        // preserved verbatim, so `total` is assembled bit-for-bit unchanged
        // (the k=4 determinism oracle covers it) (#1575).
        let rows: Vec<Array1<f64>> = (0..n).map(|r| x_dense.row(r).to_owned()).collect();
        let kmat_x: Vec<Array1<f64>> = rows.iter().map(|xr| k_mat.dot(xr)).collect();
        let ki_x: Vec<Vec<Array1<f64>>> = (0..k)
            .map(|a| rows.iter().map(|xr| k_i[a].dot(xr)).collect())
            .collect();
        // `kmat_x` (n·p) and `ki_x` (k·n·p, with n·p ≤ FIRTH_MAX_LINEAR_WORK)
        // are always cheap to hold, but the mixed cache `kij_x` is k²·n·p and
        // the Firth gate does NOT bound k (the smooth count), so a many-smooth
        // model could blow memory the original inline loop never allocated.
        // Materialize it only when it fits a fixed budget; otherwise fall back
        // to recomputing `k_ij[a][b]·xⱼ` inline (the original arithmetic, so
        // still bit-identical — just without the O(n²)→O(n) reuse on the mixed
        // blocks). 64M f64 ≈ 512 MB ceiling.
        const TK_KIJ_CACHE_MAX_ELEMS: usize = 64 * 1024 * 1024;
        let kij_x: Option<Vec<Vec<Vec<Array1<f64>>>>> =
            if k.saturating_mul(k).saturating_mul(n).saturating_mul(p) <= TK_KIJ_CACHE_MAX_ELEMS {
                Some(
                    (0..k)
                        .map(|a| {
                            (0..k)
                                .map(|b| rows.iter().map(|xr| k_ij[a][b].dot(xr)).collect())
                                .collect()
                        })
                        .collect(),
                )
            } else {
                None
            };
        // xᵢ · (Kᵢⱼ xⱼ): read the cached row-local matvec when present, else
        // recompute it inline. Both forms call the identical `Matrix::dot`, so
        // the scalar is bit-for-bit the same either way (#1575).
        let kij_bilinear = |a: usize, b: usize, xi: &Array1<f64>, jrow: usize| -> f64 {
            match &kij_x {
                Some(cache) => xi.dot(&cache[a][b][jrow]),
                None => xi.dot(&k_ij[a][b].dot(&rows[jrow])),
            }
        };

        let mut hdiag = Vec::with_capacity(n);
        for row in 0..n {
            let x = &rows[row];
            let mut jet = Jet::constant(x.dot(&kmat_x[row]), k);
            for a in 0..k {
                jet.g[a] = x.dot(&ki_x[a][row]);
            }
            for a in 0..k {
                for b in 0..k {
                    jet.h[[a, b]] = kij_bilinear(a, b, x, row);
                }
            }
            hdiag.push(jet);
        }
        let mut cjet = Vec::with_capacity(n);
        let mut djet = Vec::with_capacity(n);
        for row in 0..n {
            let mut eta = Jet::constant(0.0, k);
            for a in 0..k {
                eta.g[a] = eta_i[a][row];
                for b in 0..k {
                    eta.h[[a, b]] = x_dense.row(row).dot(&beta_ij[a][b]);
                }
            }
            let mut c = Jet::constant(c_array[row], k);
            let mut d = Jet::constant(d_array[row], k);
            for a in 0..k {
                c.g[a] = d_array[row] * eta.g[a];
                d.g[a] = e_array[row] * eta.g[a];
                for b in 0..k {
                    c.h[[a, b]] = d_array[row] * eta.h[[a, b]] + e_array[row] * eta.g[a] * eta.g[b];
                    d.h[[a, b]] = e_array[row] * eta.h[[a, b]] + f_array[row] * eta.g[a] * eta.g[b];
                }
            }
            cjet.push(c);
            djet.push(d);
        }

        let mut total = Jet::constant(0.0, k);
        for row in 0..n {
            total = total.add(&djet[row].mul(&hdiag[row].square()).scale(-0.125));
        }
        for irow in 0..n {
            let xi = &rows[irow];
            for jrow in 0..n {
                // K xⱼ, Kᵢ xⱼ, Kᵢⱼ xⱼ are the hoisted row-local matvecs; the
                // remaining work is the O(p) dot xᵢ · (· xⱼ), evaluated in the
                // identical irow-outer/jrow-inner order as before (#1575).
                let mut kg = Jet::constant(xi.dot(&kmat_x[jrow]), k);
                for a in 0..k {
                    kg.g[a] = xi.dot(&ki_x[a][jrow]);
                }
                for a in 0..k {
                    for b in 0..k {
                        kg.h[[a, b]] = kij_bilinear(a, b, xi, jrow);
                    }
                }
                let term = cjet[irow]
                    .mul(&cjet[jrow])
                    .mul(&kg.cube())
                    .scale(1.0 / 12.0);
                total = total.add(&term);
            }
        }
        let mut qjets: Vec<Jet> = (0..p).map(|_| Jet::constant(0.0, k)).collect();
        for row in 0..n {
            let wh = cjet[row].mul(&hdiag[row]);
            for col in 0..p {
                qjets[col] = qjets[col].add(&wh.scale(x_dense[[row, col]]));
            }
        }
        for a in 0..p {
            for b in 0..p {
                let mut kj = Jet::constant(k_mat[[a, b]], k);
                for i in 0..k {
                    kj.g[i] = k_i[i][[a, b]];
                }
                for i in 0..k {
                    for j in 0..k {
                        kj.h[[i, j]] = k_ij[i][j][[a, b]];
                    }
                }
                total = total.add(&qjets[a].mul(&kj).mul(&qjets[b]).scale(0.125));
            }
        }
        if total.h.iter().any(|v| !v.is_finite()) {
            crate::bail_invalid_estim!(
                "Tierney-Kadane analytic Hessian produced a non-finite entry"
            );
        }
        Ok(total.h)
    }

    pub(crate) fn tierney_kadane_analytic_core<S>(
        &self,
        x_dense: &Array2<f64>,
        z: &Array2<f64>,
        c_array: &Array1<f64>,
        d_array: &Array1<f64>,
        e_array: &Array1<f64>,
        f_array: &Array1<f64>,
        tk_penalties: &[gam_terms::construction::CanonicalPenalty],
        lambdas: &[f64],
        ext_coords: &[super::reml_outer_engine::HyperCoord],
        beta: &Array1<f64>,
        firth_op: Option<&super::FirthDenseOperator>,
        compute_gradient: bool,
        compute_hessian: bool,
        h_inv_solve: &S,
    ) -> Result<super::atoms::TierneyKadaneAtom, EstimationError>
    where
        S: Fn(&Array1<f64>) -> Result<Array1<f64>, EstimationError>,
    {
        let p = x_dense.ncols();
        let k = tk_penalties.len();

        let shared = Self::tk_shared_intermediates(
            x_dense,
            z,
            c_array,
            "Tierney-Kadane correction",
            h_inv_solve,
        )?;
        let mut gram = Array2::<f64>::zeros((TK_BLOCK_SIZE, TK_BLOCK_SIZE));
        let value = Self::tk_scalar_from_shared(x_dense, z, d_array, &shared, &mut gram)?;
        if !compute_gradient {
            return Ok(super::atoms::TierneyKadaneAtom::from_terms(
                TkCorrectionTerms {
                    value,
                    gradient: None,
                    hessian: None,
                },
            ));
        }

        let mut x_vks: Vec<Array1<f64>> = Vec::with_capacity(k + ext_coords.len());
        let mut beta_dirs: Vec<Array1<f64>> = Vec::with_capacity(k + ext_coords.len());
        for idx in 0..k {
            let cp = &tk_penalties[idx];
            let r = &cp.col_range;
            let beta_block = beta.slice(s![r.start..r.end]);
            let centered = &beta_block - &cp.prior_mean;
            let r_beta = cp.root.dot(&centered);
            let mut s_k_beta = Array1::<f64>::zeros(p);
            for a in 0..cp.block_dim() {
                s_k_beta[r.start + a] = (0..cp.rank())
                    .map(|row| cp.root[[row, a]] * r_beta[row])
                    .sum::<f64>();
            }
            let a_k_beta = &s_k_beta * lambdas[idx];
            let v_k = h_inv_solve(&a_k_beta)?;
            x_vks.push(gam_linalg::faer_ndarray::fast_av(x_dense, &v_k));
            beta_dirs.push(v_k.mapv(|value| -value));
        }
        let mut ext_drifts = Vec::with_capacity(ext_coords.len());
        let mut ext_eta_fixed = Vec::with_capacity(ext_coords.len());
        let mut ext_x_fixed = Vec::with_capacity(ext_coords.len());
        for coord in ext_coords {
            let drift = coord.drift.materialize();
            if drift.ncols() != beta.len() || drift.nrows() != beta.len() {
                crate::bail_invalid_estim!(
                    "Tierney-Kadane ext drift shape mismatch: expected {}x{}, got {}x{}",
                    beta.len(),
                    beta.len(),
                    drift.nrows(),
                    drift.ncols()
                );
            }
            if coord.g.len() != beta.len() {
                crate::bail_invalid_estim!(
                    "Tierney-Kadane ext mode RHS length mismatch: expected {}, got {}",
                    beta.len(),
                    coord.g.len()
                );
            }
            let beta_theta = h_inv_solve(&coord.g)?;
            x_vks.push(gam_linalg::faer_ndarray::fast_av(x_dense, &beta_theta));
            beta_dirs.push(beta_theta.mapv(|value| -value));
            ext_drifts.push(drift);
            ext_eta_fixed.push(coord.tk_eta_fixed.clone());
            ext_x_fixed.push(coord.tk_x_fixed.clone());
        }

        let gradient = Self::tk_gradient_from_shared(
            x_dense,
            z,
            c_array,
            d_array,
            e_array,
            tk_penalties,
            lambdas,
            &ext_drifts,
            &ext_eta_fixed,
            &ext_x_fixed,
            &x_vks,
            &beta_dirs,
            firth_op,
            &shared,
            &mut gram,
        )?;
        let hessian = if compute_hessian {
            Some(Self::tk_hessian_rho_canonical_logit(
                x_dense,
                c_array,
                d_array,
                e_array,
                f_array,
                tk_penalties,
                lambdas,
                beta,
                firth_op,
                h_inv_solve,
            )?)
        } else {
            None
        };
        Ok(super::atoms::TierneyKadaneAtom::from_terms(
            TkCorrectionTerms {
                value,
                gradient: Some(gradient),
                hessian,
            },
        ))
    }

    pub(crate) fn tierney_kadane_terms(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        mode: super::reml_outer_engine::EvalMode,
        ext_coords: &[super::reml_outer_engine::HyperCoord],
    ) -> Result<super::atoms::TierneyKadaneAtom, EstimationError> {
        if reml_is_gaussian_identity(&self.config.likelihood) {
            return Ok(super::atoms::TierneyKadaneAtom::from_terms(
                TkCorrectionTerms {
                    value: 0.0,
                    gradient: None,
                    hessian: None,
                },
            ));
        }
        if reml_robust_jeffreys_link(&self.config).is_none() {
            return Ok(super::atoms::TierneyKadaneAtom::from_terms(
                TkCorrectionTerms {
                    value: 0.0,
                    gradient: None,
                    hessian: None,
                },
            ));
        }
        // The TK correction's c/d/e/f derivative arrays use the logit
        // 5th-derivative jet and are implemented only for canonical Binomial
        // Logit Firth fits. #758 widened Firth to other Binomial inverse links;
        // those fits skip the higher-order TK refinement (zero correction) and
        // fall back to plain Laplace REML rather than erroring inside
        // `hessian_cdef_arrays`. The Firth/Jeffreys bias reduction itself lives in
        // the inner PIRLS solve, so it is fully retained — only the outer
        // marginal-likelihood refinement is dropped for non-logit links.
        if !self.tk_correction_is_canonical_logit() {
            return Ok(super::atoms::TierneyKadaneAtom::from_terms(
                TkCorrectionTerms {
                    value: 0.0,
                    gradient: None,
                    hessian: None,
                },
            ));
        }

        let compute_gradient = compute_gradient_for_tk(mode);
        let zero_correction = || {
            super::atoms::TierneyKadaneAtom::from_terms(TkCorrectionTerms {
                value: 0.0,
                gradient: if compute_gradient {
                    Some(Array1::zeros(rho.len() + ext_coords.len()))
                } else {
                    None
                },
                hessian: if mode == super::reml_outer_engine::EvalMode::ValueGradientHessian {
                    Some(Array2::zeros((
                        rho.len() + ext_coords.len(),
                        rho.len() + ext_coords.len(),
                    )))
                } else {
                    None
                },
            })
        };

        // The outer Firth gate (`firth_problem_scale_allows`) disables the dense
        // Firth operator at the same problem scale that makes the TK refinement's
        // dense calculus infeasible. When that gate trips, the inner PIRLS solve
        // logs `jeffreys_logdet=none` and falls back to plain Laplace REML, and
        // both `bundle.firth_dense_operator` and `bundle.firth_dense_operator_original`
        // are `None`. The TK refinement is a higher-order correction on top of the
        // Firth/Jeffreys-augmented Laplace expansion; without the operator it has
        // nothing to refine. Mirror the gate here so large-model fits silently drop
        // the refinement rather than erroring out — matching the established skip
        // pattern for non-canonical-logit links above.
        //
        // The Firth gate is strictly tighter than the TK dense-work caps used by
        // the non-Gaussianity audit (`TK_MAX_*`): `firth_problem_scale_allows`
        // already bounds `n ≤ FIRTH_MAX_OBSERVATIONS (20_000)`,
        // `p ≤ FIRTH_MAX_COEFFICIENTS (256)` and `n·p ≤ FIRTH_MAX_LINEAR_WORK (2e6)`,
        // each of which is below the corresponding TK cap. Passing this gate
        // therefore guarantees the dense calculus below is affordable, so no
        // separate TK size check is needed here.
        let n_x = self.x().nrows();
        let p_x = self.x().ncols();
        if !super::firth_problem_scale_allows(n_x, p_x) {
            return Ok(zero_correction());
        }

        let pirls_result = bundle.pirls_result.as_ref();
        let (c_array, d_array, e_array, f_array) = self.hessian_cdef_arrays(pirls_result)?;
        if let Some(idx) = c_array.iter().position(|v| !v.is_finite()) {
            crate::bail_invalid_estim!(
                "Tierney-Kadane correction received non-finite c derivative at row {idx}: {}",
                c_array[idx]
            );
        }
        if let Some(idx) = d_array.iter().position(|v| !v.is_finite()) {
            crate::bail_invalid_estim!(
                "Tierney-Kadane correction received non-finite d derivative at row {idx}: {}",
                d_array[idx]
            );
        }
        if c_array.is_empty() || d_array.is_empty() {
            return Ok(zero_correction());
        }

        if let Some(sparse) = bundle.sparse_exact.as_ref() {
            let x_dense = self
                .x()
                .try_to_dense_arc("frozen-curvature TK correction requires dense design access")
                .map_err(EstimationError::InvalidInput)?;
            let xt = x_dense.t().to_owned();
            let z_mat =
                gam_linalg::sparse_exact::solve_sparse_spdmulti(sparse.factor.as_ref(), &xt)?;
            let factor_ref = sparse.factor.clone();
            let h_inv_solve = |rhs: &Array1<f64>| -> Result<Array1<f64>, EstimationError> {
                Ok(gam_linalg::sparse_exact::solve_sparse_spd(
                    &factor_ref,
                    rhs,
                )?)
            };
            let lambdas: Vec<f64> = rho.iter().map(|r| r.exp()).collect();
            let beta = self.sparse_exact_beta_original(pirls_result);
            let firth_op = if let Some(jeffreys_link) = reml_robust_jeffreys_link(&self.config) {
                if let Some(cached) = bundle.firth_dense_operator_original.as_ref() {
                    Some(cached.clone())
                } else {
                    Some(std::sync::Arc::new(
                        Self::build_firth_dense_operator_for_link(
                            &jeffreys_link,
                            x_dense.as_ref(),
                            &pirls_result.final_eta.to_owned(),
                            self.weights,
                        )?,
                    ))
                }
            } else {
                None
            };
            return self.tierney_kadane_analytic_core(
                x_dense.as_ref(),
                &z_mat,
                &c_array,
                &d_array,
                &e_array,
                &f_array,
                &self.canonical_penalties,
                &lambdas,
                ext_coords,
                &beta,
                firth_op.as_deref(),
                compute_gradient,
                mode == super::reml_outer_engine::EvalMode::ValueGradientHessian,
                &h_inv_solve,
            );
        }

        let free_basis_opt = self.active_constraint_free_basis(pirls_result);
        let use_original_basis = matches!(
            pirls_result.coordinate_frame,
            pirls::PirlsCoordinateFrame::TransformedQs
        ) && free_basis_opt.is_none();
        // `firth_bias_reduction == true` here: the early return above bails
        // out otherwise, so h_total is the only TK-relevant operator.
        let h_tk_source = bundle.h_total.as_ref();
        let h_tk_eval = if use_original_basis {
            self.bundle_matrix_in_original_basis(pirls_result, h_tk_source)
        } else if let Some(z) = free_basis_opt.as_ref() {
            Self::projectwith_basis(h_tk_source, z)
        } else {
            h_tk_source.clone()
        };
        let x_eff_dense = if use_original_basis {
            self.x()
                .try_to_dense_arc("Tierney-Kadane correction requires dense original design access")
                .map_err(EstimationError::InvalidInput)?
                .as_ref()
                .clone()
        } else if let Some(z) = free_basis_opt.as_ref() {
            pirls_result.x_transformed.to_dense().dot(z)
        } else {
            pirls_result.x_transformed.to_dense()
        };

        let xt = x_eff_dense.t().to_owned();
        let p = x_eff_dense.ncols();
        let n = x_eff_dense.nrows();
        enum HFactor {
            Cholesky(gam_linalg::faer_ndarray::FaerCholeskyFactor),
            Eigh {
                evals: Array1<f64>,
                evecs: Array2<f64>,
            },
        }
        let h_factor = if let Ok(chol) = h_tk_eval.cholesky(Side::Lower) {
            HFactor::Cholesky(chol)
        } else if let Ok((evals, evecs)) = h_tk_eval.eigh(Side::Lower) {
            // Smallest eigenvalue at or below this floor means the effective
            // Hessian failed positive-definiteness (Cholesky already declined),
            // so the Tierney–Kadane Laplace correction is undefined here.
            const TK_HESSIAN_PD_EIGENVALUE_FLOOR: f64 = 1e-12;
            if let Some((idx, ev)) = evals
                .iter()
                .enumerate()
                .find(|(_, ev)| **ev <= TK_HESSIAN_PD_EIGENVALUE_FLOOR)
            {
                crate::bail_invalid_estim!(
                    "Tierney-Kadane correction requires a positive definite Hessian; eigenvalue {idx} is {ev}"
                );
            }
            HFactor::Eigh { evals, evecs }
        } else {
            crate::bail_invalid_estim!(
                "Tierney-Kadane correction could not factor the effective Hessian"
            );
        };

        let z_mat = match &h_factor {
            HFactor::Cholesky(chol) => {
                let mut solved = xt.clone();
                chol.solve_mat_in_place(&mut solved);
                solved
            }
            HFactor::Eigh { evals, evecs } => {
                let mut solved = Array2::<f64>::zeros((p, n));
                for m in 0..evals.len() {
                    let ev = evals[m];
                    let u = evecs.column(m);
                    let coeffs = xt.t().dot(&u).mapv(|v| v / ev);
                    for row in 0..p {
                        let u_row = u[row];
                        for col in 0..n {
                            solved[[row, col]] += u_row * coeffs[col];
                        }
                    }
                }
                solved
            }
        };

        let h_inv_solve = move |rhs: &Array1<f64>| -> Result<Array1<f64>, EstimationError> {
            match &h_factor {
                HFactor::Cholesky(chol) => Ok(chol.solvevec(rhs)),
                HFactor::Eigh { evals, evecs } => {
                    let mut sol = Array1::<f64>::zeros(rhs.len());
                    for m in 0..evals.len() {
                        let u = evecs.column(m);
                        let coeff = u.dot(rhs) / evals[m];
                        for row in 0..sol.len() {
                            sol[row] += coeff * u[row];
                        }
                    }
                    Ok(sol)
                }
            }
        };

        let p_eff = x_eff_dense.ncols();
        let tk_penalties: Vec<gam_terms::construction::CanonicalPenalty> = if use_original_basis {
            self.canonical_penalties.as_ref().clone()
        } else if let Some(z) = free_basis_opt.as_ref() {
            pirls_result
                .reparam_result
                .canonical_transformed
                .iter()
                .map(|cp| {
                    gam_terms::construction::CanonicalPenalty::from_dense_root(cp.root.dot(z), p_eff)
                })
                .collect()
        } else {
            pirls_result.reparam_result.canonical_transformed.clone()
        };
        let lambdas: Vec<f64> = rho.iter().map(|r| r.exp()).collect();
        let beta = if use_original_basis {
            self.sparse_exact_beta_original(pirls_result)
        } else if let Some(z) = free_basis_opt.as_ref() {
            z.t().dot(pirls_result.beta_transformed.as_ref())
        } else {
            pirls_result.beta_transformed.as_ref().clone()
        };
        let firth_op = if let Some(jeffreys_link) = reml_robust_jeffreys_link(&self.config) {
            Some(std::sync::Arc::new(
                Self::build_firth_dense_operator_for_link(
                    &jeffreys_link,
                    &x_eff_dense,
                    &pirls_result.final_eta.to_owned(),
                    self.weights,
                )?,
            ))
        } else {
            None
        };

        self.tierney_kadane_analytic_core(
            &x_eff_dense,
            &z_mat,
            &c_array,
            &d_array,
            &e_array,
            &f_array,
            &tk_penalties,
            &lambdas,
            ext_coords,
            &beta,
            firth_op.as_deref(),
            compute_gradient,
            mode == super::reml_outer_engine::EvalMode::ValueGradientHessian,
            &h_inv_solve,
        )
    }

    pub(crate) fn validate_tk_ext_coords(
        &self,
        mode: super::reml_outer_engine::EvalMode,
        ext_coords: &[super::reml_outer_engine::HyperCoord],
    ) -> Result<(), EstimationError> {
        if reml_is_gaussian_identity(&self.config.likelihood)
            || reml_robust_jeffreys_link(&self.config).is_none()
            || !compute_gradient_for_tk(mode)
        {
            return Ok(());
        }
        for (idx, coord) in ext_coords.iter().enumerate() {
            if coord.tk_eta_fixed.is_none() || coord.tk_x_fixed.is_none() {
                crate::bail_invalid_estim!(
                    "Tierney-Kadane external gradient coordinate {idx} is missing analytic fixed-beta design/eta derivative carriers"
                );
            }
        }
        Ok(())
    }

    pub(crate) fn apply_theta_correction_atom_to_result<A>(
        &self,
        mut result: super::reml_outer_engine::RemlLamlResult,
        correction: &A,
    ) -> Result<super::reml_outer_engine::RemlLamlResult, EstimationError>
    where
        A: super::atoms::ThetaCorrectionProjection + ?Sized,
    {
        result.cost += correction.cost();
        if let Some(correction_hess) = correction.hessian() {
            crate::objective_base::add_rho_block_dense_to_hessian(
                &mut result.hessian,
                correction_hess,
            )
            .map_err(EstimationError::InvalidInput)?;
        }
        if let (Some(ref mut grad), Some(correction_grad)) =
            (result.gradient.as_mut(), correction.gradient())
        {
            if correction_grad.len() == grad.len() {
                **grad += correction_grad;
            } else {
                // The unified evaluator returns one gradient entry per
                // (ρ, ext_coord) coordinate; theta-only correction atoms must
                // emit exactly the same coordinate layout. An arity mismatch
                // means the correction and evaluator were assembled against
                // different coordinate sets, which would yield a structurally
                // inconsistent total gradient and is therefore rejected
                // outright instead of silently zero-padding or truncating.
                crate::bail_invalid_estim!(
                    "{} gradient coordinate count mismatch: evaluator produced {} entries, correction atom produced {}; this indicates the correction term and the unified evaluator were assembled against different coordinate sets",
                    correction.name(),
                    grad.len(),
                    correction_grad.len()
                );
            }
        }
        Ok(result)
    }

    /// Build the inverse link from the runtime link state, mirroring the
    /// dispatch in `hessian_cde_arrays`. Used by the #784 block-local sampled
    /// marginalization to evaluate μ at displaced η.
    pub(crate) fn runtime_inverse_link(&self) -> InverseLink {
        let link_function = self.config.link_function();
        if let Some(state) = self.runtime_mixture_link_state.clone() {
            InverseLink::Mixture(state)
        } else if let Some(state) = self.runtime_sas_link_state {
            if matches!(link_function, LinkFunction::BetaLogistic) {
                InverseLink::BetaLogistic(state)
            } else {
                InverseLink::Sas(state)
            }
        } else {
            InverseLink::Standard(
                StandardLink::try_from(link_function)
                    .expect("state-bearing link without runtime state in runtime_inverse_link"),
            )
        }
    }

    /// Adaptive, block-local Laplace-to-sampling fallback for the inner
    /// marginalization loop (issue #784).
    ///
    /// The unified evaluator summarizes the coefficient posterior by its Laplace
    /// (Gaussian) moments. This method audits that summary per curvature
    /// direction and, where the Gaussian approximation is *not* trustworthy,
    /// replaces it with a sampling-based block marginal — keeping the cheap
    /// Laplace summary everywhere else:
    ///
    /// 1. Run the directional cubic non-Gaussianity diagnostic on the observed
    ///    penalized Hessian + the third-derivative weights `solve_c_array`,
    ///    yielding per-eigendirection standardized skewness `γ_r`.
    /// 2. Convert `γ_r` into a block-local activation set via the auto-derived
    ///    threshold `τ(n_eff)` (no flag). The flagged eigenvectors span the
    ///    curvature-heavy subspace `V_b`.
    /// 3. Importance-sample the true block marginal against the local Laplace
    ///    Gaussian (reusing the whitening) and return the additive correction
    ///    `Δ_b` to the marginal log-likelihood, together with its consistent
    ///    ρ-gradient, so the outer REML/LAML stays consistent.
    ///
    /// Returns `TkCorrectionTerms` whose `value` is added to the REML cost.
    /// Because `Δ_b` is added to the *marginal log-likelihood* it is subtracted
    /// from the cost, so the returned `value` is `−Δ_b` (likewise the gradient).
    /// The gradient is laid out over the ρ coordinates and zero-extended over
    /// external coordinates to match the unified evaluator's coordinate set in
    /// `apply_tk_to_result`.
    ///
    /// A no-op (zeros) is returned for Gaussian-identity fits (Laplace is
    /// exact), when no direction trips the threshold, or when the importance
    /// estimate is not trustworthy (low ESS) — in which case the plain Laplace
    /// summary is retained rather than splicing in a noisy correction.
    ///
    /// # Outer-consistency / continuity
    ///
    /// The threshold-based activation is technically discontinuous in ρ (a
    /// direction can cross `τ` as ρ moves). That discontinuity is harmless for
    /// the outer REML by construction: a direction crosses the threshold only
    /// at `|γ_r| ≈ τ = sqrt((24/5)/n_eff) = O(n^{−1/2})`, so its contribution to
    /// `Δ_b` is `O(γ_r²) = O(1/n)` — the same order as the Laplace floor error
    /// that the criterion already carries and below the inner KKT tolerance
    /// band. The correction value therefore vanishes continuously as a
    /// direction approaches the threshold, so the spliced objective is
    /// continuous to leading order and does not bias ρ selection.
    /// Per-bundle-cached wrapper around [`Self::block_local_sampled_correction_compute`].
    ///
    /// The block-local correction is a deterministic function of this bundle's
    /// converged inner state and ρ alone (mode-invariant, Hessian-free), but the
    /// outer loop evaluates the objective at one ρ up to three times (value,
    /// value+gradient, value+gradient+Hessian) sharing the SAME `bundle`. The
    /// expensive engaged path (dense O(p³) eigendecomposition plus the
    /// fixed-seed O(draws·n·m) importance sampler) therefore reran 2–3× per
    /// outer iteration. Hoist it onto `bundle.block_local_correction` so it is
    /// computed exactly once per inner solution and every consumer at that ρ
    /// reads the identical value+gradient (exact hoist — #784, #1082). Keyed on
    /// `n_ext`, which is fixed for a fit, so one cell suffices.
    pub(crate) fn block_local_sampled_correction(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        n_ext: usize,
    ) -> Result<TkCorrectionTerms, EstimationError> {
        if let Some((cached_ext, terms)) = bundle.block_local_correction.get()
            && *cached_ext == n_ext
        {
            return Ok((**terms).clone());
        }
        let terms = self.block_local_sampled_correction_compute(rho, bundle, n_ext)?;
        // First writer wins; a racing writer built from identical inputs, so
        // either stored object is correct. A `set` that loses the race (cell
        // already filled) is fine — both terms are equal — so the `Err` is
        // discarded by returning the freshly computed `terms` either way.
        match bundle
            .block_local_correction
            .set((n_ext, std::sync::Arc::new(terms.clone())))
        {
            Ok(()) => Ok(terms),
            Err(_) => Ok(terms),
        }
    }

    fn block_local_sampled_correction_compute(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        n_ext: usize,
    ) -> Result<TkCorrectionTerms, EstimationError> {
        // #1521 trait-inversion: the #784 importance-sampling correction and its
        // eigen-diagnostic live UP in the gam-inference `hmc_io` tier; gam-solve
        // calls them through the neutral `gam_problem` sampler contract instead
        // of a back-edge into the inference SCC. The pure threshold math
        // (`laplace_trustworthiness_from_skewness`) moved down outright.
        use gam_problem::laplace_sampler_contract::laplace_trustworthiness_from_skewness;

        let n_rho = self.canonical_penalties.len();
        let zero = || TkCorrectionTerms {
            value: 0.0,
            gradient: Some(Array1::zeros(n_rho + n_ext)),
            hessian: None,
        };

        // Laplace is exact for the Gaussian-identity model: nothing to correct.
        if reml_is_gaussian_identity(&self.config.likelihood) {
            return Ok(zero());
        }
        // The penalty-score channel needs one λ per canonical penalty.
        if rho.len() != n_rho || n_rho == 0 {
            return Ok(zero());
        }

        let pirls_result = bundle.pirls_result.as_ref();
        // Operate in the transformed basis, where `h_total`, `solve_c_array`,
        // `final_eta`, `finalweights`, `beta_transformed` and `x_transformed`
        // are all mutually consistent.
        let h_total = bundle.h_total.as_ref();
        let c_weights = &pirls_result.solve_c_array.to_owned();
        let x_design = &pirls_result.x_transformed;
        let p = h_total.nrows();
        if p == 0 || c_weights.len() != x_design.nrows() {
            return Ok(zero());
        }

        // Problem-scale gate. The non-Gaussianity diagnostic costs an O(p³)
        // dense eigendecomposition plus O(n·p) cubic contractions, and the
        // sampler adds O(draws · n · m) deviance work. At large scale that is
        // prohibitive on every inner evaluation, and the Laplace floor error is
        // already O(1/n) → negligible there, so the correction would be a
        // no-op anyway. Mirror the established TK scale caps: skip the audit
        // entirely above them and retain the (asymptotically exact) plain
        // Laplace summary.
        let n_obs = x_design.nrows();
        let dense_work = n_obs.saturating_mul(p);
        if n_obs > TK_MAX_OBSERVATIONS || p > TK_MAX_COEFFICIENTS || dense_work > TK_MAX_DENSE_WORK
        {
            return Ok(zero());
        }

        // Resolve the injected gam-inference sampler. When the sampler tier is
        // not linked / registered, decline the correction (zero contribution) —
        // the same safe no-op as every other decline branch here.
        let Some(sampler) = gam_problem::laplace_sampler_contract::laplace_marginal_sampler() else {
            return Ok(zero());
        };

        // Step 1: per-direction skewness diagnostic γ_r.
        let (max_abs, directional) = sampler
            .directional_cubic_diagnostic(h_total, x_design, c_weights, false)
            .map_err(EstimationError::InvalidInput)?;
        if !max_abs.is_finite() || max_abs == 0.0 {
            return Ok(zero());
        }

        // Step 2: auto-derived, block-local activation. `n_eff` is the number of
        // observations carrying curvature; using it (not the raw n) keeps the
        // verdict tied to the actual information content.
        let n_eff = c_weights.iter().filter(|&&c| c != 0.0).count() as f64;
        let verdict = laplace_trustworthiness_from_skewness(&directional, n_eff);
        if !verdict.fallback_required() {
            return Ok(zero());
        }

        // External (ψ) hyper-coordinates present: the exact gradient of the
        // realized estimator along ψ requires the field motion of `X(ψ)`,
        // `S(ψ)` and the reparameterized basis — moments this seam does not
        // yet carry. A spliced value whose ψ-gradient entries are zeroed (or
        // truncated) is an objective↔gradient desync (#901, the #752/#748
        // bug class); per the gradient exactness contract on
        // `block_sampled_marginal_correction`, the correct response is to
        // DECLINE the splice — value AND gradient together — rather than
        // approximate.
        if n_ext > 0 {
            log::info!(
                "[#784] block-local fallback declined: {n_ext} external (ψ) coordinate(s) \
                 present and the ψ-exact gradient channels are not implemented; splicing a \
                 ψ-truncated gradient would desync objective and gradient (#901)"
            );
            return Ok(zero());
        }
        // The exact score channel relies on the exponential-family unit-
        // deviance identity dD/dμ = −2w(y−μ)/V(μ), which does not hold for
        // the Beta pseudo-family parameterization. Decline rather than splice
        // a gradient that is not the derivative of the spliced value.
        if matches!(
            reml_spec(&self.config.likelihood).response,
            ResponseFamily::Beta { .. }
        ) {
            log::info!(
                "[#784] block-local fallback declined: Beta family has no exponential-family \
                 score identity for the exact gradient channels"
            );
            return Ok(zero());
        }

        // Build the block subspace V_b from the flagged H-eigenvectors.
        let sym_h = (h_total + &h_total.t()) * 0.5;
        let (evals, evecs) = sym_h.eigh(Side::Lower).map_err(|e| {
            EstimationError::InvalidInput(format!(
                "#784 block-local fallback eigendecomposition failed: {e}"
            ))
        })?;
        let mut block_cols: Vec<usize> = Vec::new();
        for &r in &verdict.untrustworthy_directions {
            if r < evals.len() && evals[r] > 0.0 {
                block_cols.push(r);
            }
        }
        if block_cols.is_empty() {
            return Ok(zero());
        }
        let m = block_cols.len();
        let mut block_vecs = Array2::<f64>::zeros((p, m));
        let mut block_lambdas = Array1::<f64>::zeros(m);
        for (j, &r) in block_cols.iter().enumerate() {
            block_vecs.column_mut(j).assign(&evecs.column(r));
            block_lambdas[j] = evals[r];
        }

        // Penalty scores S_k β̂ (canonical basis) and λ_k = e^{ρ_k}.
        // Computed once per inner solution on the eval bundle and reused
        // across every assemble call sharing this bundle: β̂ =
        // `pirls_result.beta_transformed` is fixed inside the bundle's
        // `Arc<PirlsResult>` and `self.canonical_penalties` is fixed for the
        // `RemlState`, so the vectors are ρ- and mode-invariant (exact hoist,
        // identical values for every consumer).
        let penalty_scores = bundle.canonical_penalty_scores_at_mode(&self.canonical_penalties)?;
        let lambdas: Vec<f64> = rho.iter().map(|&r| r.exp()).collect();

        // Dispersion φ used to turn deviance into log-likelihood.
        let phi = match reml_spec(&self.config.likelihood).response {
            ResponseFamily::Gaussian => 1.0,
            _ => reml_fixed_glm_dispersion(&self.config.likelihood),
        };
        let phi = if phi.is_finite() && phi > 0.0 {
            phi
        } else {
            1.0
        };

        let x_dense = x_design
            .try_to_dense_arc("#784 block-local fallback requires dense design access")
            .map_err(EstimationError::InvalidInput)?;

        let target = Gam784BlockTarget {
            x_transformed: x_dense.as_ref(),
            block_vecs,
            block_lambdas,
            eta_hat: pirls_result.final_eta.to_owned(),
            weights_obs: pirls_result.finalweights.to_owned(),
            y: self.y.to_owned(),
            prior_weights: self.weights.to_owned(),
            likelihood: self.config.likelihood.clone(),
            inverse_link: self.runtime_inverse_link(),
            phi,
            penalty_scores,
            lambdas,
            base_deviance: pirls_result.deviance,
        };

        let sampled = sampler
            .block_sampled_marginal_correction(&target)
            .map_err(EstimationError::InvalidInput)?;

        // Trust gate: an importance estimate with too few effective draws is
        // noisier than the Laplace error it is meant to correct, so we keep the
        // plain Laplace summary rather than splicing in Monte-Carlo jitter.
        let min_ess = (sampled.n_draws as f64 * MIN_IMPORTANCE_ESS_FRACTION).max(1.0);
        if sampled.importance_ess < min_ess {
            log::info!(
                "[#784] block-local fallback declined: importance ESS {:.1} < {:.1} \
                 (m={m} dirs, max|γ|={:.3}, τ={:.3})",
                sampled.importance_ess,
                min_ess,
                verdict.max_abs_skewness,
                verdict.threshold,
            );
            return Ok(zero());
        }

        log::info!(
            "[#784] block-local sampled marginalization ENGAGED: m={m} curvature-heavy dirs, \
             max|γ|={:.3}, τ={:.3}, Δ_b={:.4e}, ESS={:.1}/{}",
            verdict.max_abs_skewness,
            verdict.threshold,
            sampled.value,
            sampled.importance_ess,
            sampled.n_draws,
        );

        // `Δ_b` is added to the marginal log-likelihood ⇒ subtracted from the
        // REML cost. The gradient ∂Δ_b/∂ρ likewise enters the cost with a
        // negative sign.
        //
        // ── Exact gradient channels (b)–(d) ─────────────────────────────
        // The explicit channel `−sampled.rho_gradient` (channel (a)) is NOT
        // the total ρ-derivative of the realized estimator: with fixed-seed
        // draws `t_s = z_s/√λ_r(ρ)`, the value also moves through the block
        // eigenvalues (draw rescale, (b)), the block eigenvectors (frame
        // rotation, (c)), and the mode β̂ (mode motion, (d)). Splicing (a)
        // alone is the #752/#748/#901 objective↔gradient desync. The four
        // channels are assembled here per the gradient exactness contract on
        // `block_sampled_marginal_correction`, contracting the sampler's
        // self-normalized moments against fields this evaluator already owns:
        //
        //   d(cost)/dρ_j = E_p[dΔF/dρ_j]
        //                = (a) E_p[∂ΔF/∂ρ_j]
        //                + (b)+(c) tr(Ḣ_j · (Q_b + Q_c))
        //                + (d) g_dᵀ · dβ̂/dρ_j,
        //
        // with the TOTAL drift `Ḣ_j = λ_j S_j − C[v_j]`,
        // `C[v] = Xᵀ diag(c ⊙ Xv) X`, the IFT mode response
        // `dβ̂/dρ_j = −v_j = −H⁻¹ λ_j S_j β̂`, and
        //
        //   Q_b = Σ_r (M_r/λ_r) u_r u_rᵀ                       (rank m)
        //   Q_c = sym( Σ_r Σ_{q≠r} u_q (R̃_{q r}/(λ_r − σ_q)) u_rᵀ )
        //   M_r = E_p[(∂ΔF/∂t)_r · (−½ t_r)],   R̃ = Uᵀ E_p[t_r ∂ΔF/∂δ].
        //
        // Eigenvalue near-degeneracies `λ_r ≈ σ_q` are genuine
        // non-differentiability points of the eigenframe; the splice is
        // declined there rather than clamped.
        let Some(moments) = sampled.moments.as_ref() else {
            // m > 0 is guaranteed above, so absent moments means every draw
            // carried zero weight — nothing trustworthy to splice.
            return Ok(zero());
        };
        let x = x_dense.as_ref();
        let n_rows = x.nrows();
        let xv = x.dot(&target.block_vecs); // n × m
        let ngs_base = target.base_neg_score();

        // σ²_i = E_p[s_i²] and the shared n×m intermediates.
        let xv_ett = xv.dot(&moments.e_tt); // n × m
        let sigma2 = (&xv_ett * &xv).sum_axis(ndarray::Axis(1)); // n
        let mut w_xv_ett = xv_ett.clone();
        for i in 0..n_rows {
            let w_i = target.weights_obs[i];
            w_xv_ett.row_mut(i).mapv_inplace(|v| v * w_i);
        }

        // Channel (d) moment: g_d = E_p[∂ΔF/∂β̂]
        //   = Xᵀ(E_p[ngs_disp] − ngs_base) + Σ_k λ_k S_k (V_b E_p[t])
        //     − ½ Xᵀ(c ⊙ E_p[s²]).
        let delta_mean = target.block_vecs.dot(&moments.e_t); // p
        let mut g_d = x.t().dot(&(&moments.e_neg_score - &ngs_base));
        for (pen, &lam) in self.canonical_penalties.iter().zip(target.lambdas.iter()) {
            g_d.scaled_add(lam, &transformed_penalty_matvec(pen, &delta_mean));
        }
        g_d.scaled_add(-0.5, &x.t().dot(&(c_weights * &sigma2)));

        // Channel (c) moment: R[:,r] = E_p[t_r · ∂ΔF/∂δ]
        //   = Xᵀ E_p[t_r ngs_disp] + (Σ_k λ_k S_k β̂) E_p[t_r] − Xᵀ W X V_b E_p[t tᵀ][:,r].
        let mut pen_score_total = Array1::<f64>::zeros(p);
        for (score, &lam) in target.penalty_scores.iter().zip(target.lambdas.iter()) {
            pen_score_total.scaled_add(lam, score);
        }
        let mut r_mat = x.t().dot(&moments.e_t_neg_score); // p × m
        for r in 0..m {
            r_mat
                .column_mut(r)
                .scaled_add(moments.e_t[r], &pen_score_total);
        }
        r_mat -= &x.t().dot(&w_xv_ett);

        // Channel (b) moment: M_r = E_p[(∂ΔF/∂t)_r (−½ t_r)] via
        // ∂ΔF/∂t = (XV)ᵀ ngs_disp + V_bᵀ(Σλ_k S_k β̂) − (XV)ᵀ(W ⊙ s).
        let xvt_etngs = xv.t().dot(&moments.e_t_neg_score); // m × m
        let pterm = target.block_vecs.t().dot(&pen_score_total); // m
        let xvt_w_xv_ett = xv.t().dot(&w_xv_ett); // m × m
        let mut m_vec = Array1::<f64>::zeros(m);
        for r in 0..m {
            m_vec[r] =
                -0.5 * (xvt_etngs[(r, r)] + pterm[r] * moments.e_t[r] - xvt_w_xv_ett[(r, r)]);
        }

        // Eigenframe assembly. `block_vecs` are the `block_cols` columns of
        // `evecs`, so `Q_b`/`Q_c` are built from the same spectrum as the
        // draws — one source of truth for "the direction λ_r".
        if evals.iter().any(|&s| !(s.is_finite() && s > 0.0)) {
            log::info!(
                "[#784] block-local fallback declined: H_pen has a non-positive eigenvalue; \
                 the IFT mode response is undefined"
            );
            return Ok(zero());
        }
        let spectral_scale = evals.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        let degeneracy_tol = 1e-10 * spectral_scale.max(f64::MIN_POSITIVE);
        let r_tilde = evecs.t().dot(&r_mat); // p × m
        let mut g_mat = Array2::<f64>::zeros((p, m));
        for (jr, &col_r) in block_cols.iter().enumerate() {
            let lam_r = target.block_lambdas[jr];
            for q in 0..p {
                if q == col_r {
                    continue;
                }
                let gap = lam_r - evals[q];
                if gap.abs() < degeneracy_tol {
                    log::info!(
                        "[#784] block-local fallback declined: eigenvalue near-degeneracy \
                         |λ_r − σ_q| = {:.3e} < {degeneracy_tol:.3e} — the eigenframe is not \
                         differentiable on this stratum",
                        gap.abs(),
                    );
                    return Ok(zero());
                }
                g_mat[(q, jr)] = r_tilde[(q, jr)] / gap;
            }
        }
        let q_c_raw = evecs.dot(&g_mat).dot(&target.block_vecs.t()); // p × p
        let mut q_mat = 0.5 * (&q_c_raw + &q_c_raw.t());
        for jr in 0..m {
            let u_r = target.block_vecs.column(jr);
            let scale = m_vec[jr] / target.block_lambdas[jr];
            for a in 0..p {
                for b in 0..p {
                    q_mat[(a, b)] += scale * u_r[a] * u_r[b];
                }
            }
        }

        // rowq_i = x_iᵀ Q x_i (for tr(C[v] Q) = Σ_i (c ⊙ Xv)_i rowq_i).
        let xq = x.dot(&q_mat); // n × p
        let rowq = (&xq * x).sum_axis(ndarray::Axis(1)); // n

        // Per-coordinate contraction.
        let mut gradient = Array1::<f64>::zeros(n_rho + n_ext);
        for j in 0..n_rho.min(sampled.rho_gradient.len()) {
            let lam_j = target.lambdas[j];
            let a_j = target.penalty_scores[j].mapv(|v| lam_j * v); // λ_j S_j β̂
            // v_j = H⁻¹ a_j through the same eigendecomposition as Q.
            let uta = evecs.t().dot(&a_j);
            let v_j = evecs.dot(&(&uta / &evals));
            // tr(A_j Q) = λ_j Σ_c (S_j Q[:,c])_c.
            let mut tr_sq = 0.0_f64;
            for c in 0..p {
                let s_col = transformed_penalty_matvec(
                    &self.canonical_penalties[j],
                    &q_mat.column(c).to_owned(),
                );
                tr_sq += s_col[c];
            }
            // tr(C[v_j] Q) = Σ_i c_i (X v_j)_i rowq_i.
            let xv_j = gam_linalg::faer_ndarray::fast_av(x, &v_j);
            let mut tr_cq = 0.0_f64;
            for i in 0..n_rows {
                tr_cq += c_weights[i] * xv_j[i] * rowq[i];
            }
            let trace_j = lam_j * tr_sq - tr_cq;
            let mode_j = -v_j.dot(&g_d);
            gradient[j] = -sampled.rho_gradient[j] + trace_j + mode_j;
        }
        Ok(TkCorrectionTerms {
            value: -sampled.value,
            gradient: Some(gradient),
            hessian: None,
        })
    }

    pub(super) fn should_compute_hot_diagnostics(&self, eval_idx: u64) -> bool {
        // Keep expensive diagnostics out of the hot path unless they can
        // be surfaced. This has zero effect on optimization math.
        // Emit on the first eval and then once every this-many evals so a long
        // outer optimization leaves a periodic trace without flooding the log.
        const HOT_DIAGNOSTIC_EVAL_INTERVAL: u64 = 200;
        (log::log_enabled!(log::Level::Info) || log::log_enabled!(log::Level::Warn))
            && (eval_idx == 1 || eval_idx.is_multiple_of(HOT_DIAGNOSTIC_EVAL_INTERVAL))
    }

    pub(crate) fn invalidate_link_dependent_state(&self) {
        self.cache_manager.clear_eval_and_factor_caches();
        self.cache_manager.pirls_cache.write().unwrap().clear();
        // Under a link change the previous link's working-weight
        // curvature differs from the new link's, so the previous
        // solve's β / H_pen / Cholesky factor / damping / iter-count
        // / IFT residual / tangent-line history are all calibrated to
        // the wrong geometry. Wipe in lockstep — see
        // `clear_warm_start_predictor_state` and
        // `clear_warm_start_adaptive_signals` for the per-slot
        // staleness arguments.
        self.clear_warm_start_predictor_state();
        self.clear_warm_start_adaptive_signals();
    }

    /// Wipe every RwLock-guarded slot used by the warm-start
    /// predictors (tangent-line and IFT). Called on link change, on
    /// surface reset, on outer-seed restart, on failed PIRLS solve,
    /// and any other event where the cached `(β, ρ, H_pen, qs,
    /// factor)` is no longer calibrated to the geometry the next
    /// solve will see.
    ///
    /// Single source of truth: any new warm-start RwLock added to
    /// `RemlObjectiveState` MUST be wiped here, otherwise the
    /// warm-start machinery can leak stale state across the
    /// invalidation boundary and produce β-drift regressions
    /// (caught by `tests/warm_start_quality_regression.rs`).
    pub(crate) fn clear_warm_start_predictor_state(&self) {
        self.warm_start_beta.write().unwrap().take();
        self.warm_start_rho.write().unwrap().take();
        self.prev_warm_start_beta.write().unwrap().take();
        self.prev_warm_start_rho.write().unwrap().take();
        self.ift_warm_start_cache.write().unwrap().take();
        self.ift_cached_factor.write().unwrap().take();
        self.clear_ift_mode_response_cache();
    }

    pub(crate) fn ift_mode_response_cache_key(&self) -> usize {
        self as *const Self as usize
    }

    pub(crate) fn pending_joint_ift_theta(&self) -> Option<Array1<f64>> {
        latest_outer_theta_for_ift()
    }

    pub(crate) fn clear_joint_ift_mode_response_cache(&self) {
        if let Some(caches) = IFT_JOINT_MODE_RESPONSE_CACHES.get() {
            caches
                .lock()
                .unwrap()
                .remove(&self.ift_mode_response_cache_key());
        }
    }

    pub(crate) fn clear_ift_mode_response_cache(&self) {
        if let Some(caches) = IFT_MODE_RESPONSE_CACHES.get() {
            caches
                .lock()
                .unwrap()
                .remove(&self.ift_mode_response_cache_key());
        }
    }

    pub(crate) fn mode_response_cols_for_warm_start(
        &self,
        bundle: &EvalShared,
        cols: &Array2<f64>,
    ) -> Option<Array2<f64>> {
        if cols.ncols() == 0 || cols.nrows() != self.p {
            return None;
        }
        if bundle.backend_kind() == GeometryBackendKind::SparseExactSpd {
            return Some(cols.clone());
        }
        match bundle.pirls_result.coordinate_frame {
            pirls::PirlsCoordinateFrame::OriginalSparseNative => Some(cols.clone()),
            pirls::PirlsCoordinateFrame::TransformedQs
                if self
                    .active_constraint_free_basis(bundle.pirls_result.as_ref())
                    .is_none() =>
            {
                // `build_auto_assembly` routes this case through
                // `build_dense_original_assembly`, so evaluator columns are
                // already in the same original basis as the warm-start β.
                Some(cols.clone())
            }
            _ => None,
        }
    }

    pub(crate) fn store_ift_mode_response_cache_from_result(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        result: &super::reml_outer_engine::RemlLamlResult,
    ) {
        let rho_cols = result
            .rho_mode_response_cols
            .as_ref()
            .and_then(|cols| self.mode_response_cols_for_warm_start(bundle, cols));
        let ext_cols = result
            .ext_mode_response_cols
            .as_ref()
            .and_then(|cols| self.mode_response_cols_for_warm_start(bundle, cols));
        if rho_cols.is_none() && ext_cols.is_none() {
            if result.gradient.is_none() {
                return;
            }
            self.clear_ift_mode_response_cache();
            self.clear_joint_ift_mode_response_cache();
            return;
        }
        let rho_col_count = rho_cols.as_ref().map_or(0, Array2::ncols);
        let ext_col_count = ext_cols.as_ref().map_or(0, Array2::ncols);
        ift_mode_response_caches().lock().unwrap().insert(
            self.ift_mode_response_cache_key(),
            IftModeResponseRuntimeCache {
                rho: rho.clone(),
                rho_mode_response_cols: rho_cols.clone(),
                ext_mode_response_cols: ext_cols.clone(),
            },
        );
        log::debug!(
            "[IFT-CACHE] outcome=mode_response_store rho_cols={} ext_cols={} p={}",
            rho_col_count,
            ext_col_count,
            self.p,
        );

        let Some(theta) = self.pending_joint_ift_theta() else {
            self.clear_joint_ift_mode_response_cache();
            return;
        };
        if theta.len() <= rho.len() || theta.len() != rho.len() + ext_col_count {
            self.clear_joint_ift_mode_response_cache();
            return;
        }
        let Some(rho_cols_ref) = rho_cols.as_ref() else {
            self.clear_joint_ift_mode_response_cache();
            return;
        };
        let Some(ext_cols_ref) = ext_cols.as_ref() else {
            self.clear_joint_ift_mode_response_cache();
            return;
        };
        if rho_cols_ref.nrows() != self.p
            || rho_cols_ref.ncols() != rho.len()
            || ext_cols_ref.nrows() != self.p
            || ext_cols_ref.ncols() != ext_col_count
        {
            self.clear_joint_ift_mode_response_cache();
            return;
        }
        let active_constraints = self
            .active_constraint_free_basis(bundle.pirls_result.as_ref())
            .is_some();
        if active_constraints {
            self.clear_joint_ift_mode_response_cache();
            log::info!(
                "[IFT-REJECTED] reason=active_constraints joint_dim={}",
                theta.len()
            );
            return;
        }
        let beta_original = match bundle.pirls_result.coordinate_frame {
            pirls::PirlsCoordinateFrame::OriginalSparseNative => {
                bundle.pirls_result.beta_transformed.as_ref().clone()
            }
            pirls::PirlsCoordinateFrame::TransformedQs => bundle
                .pirls_result
                .reparam_result
                .qs
                .dot(bundle.pirls_result.beta_transformed.as_ref()),
        };
        if beta_original.len() != self.p || beta_original.iter().any(|v| !v.is_finite()) {
            self.clear_joint_ift_mode_response_cache();
            return;
        }
        let mut mode_response_cols = Array2::<f64>::zeros((self.p, theta.len()));
        mode_response_cols
            .slice_mut(s![.., ..rho.len()])
            .assign(rho_cols_ref);
        mode_response_cols
            .slice_mut(s![.., rho.len()..])
            .assign(ext_cols_ref);
        if mode_response_cols.iter().any(|v| !v.is_finite()) {
            self.clear_joint_ift_mode_response_cache();
            return;
        }
        ift_joint_mode_response_caches().lock().unwrap().insert(
            self.ift_mode_response_cache_key(),
            IftJointModeResponseRuntimeCache {
                theta,
                rho_dim: rho.len(),
                beta_original,
                mode_response_cols,
                active_constraints,
            },
        );
        log::debug!(
            "[IFT-CACHE] outcome=joint_mode_response_store rho_cols={} ext_cols={} p={}",
            rho_col_count,
            ext_col_count,
            self.p,
        );
    }

    pub(crate) fn cached_ift_rho_mode_response_cols(
        &self,
        cache: &super::IftWarmStartCache,
    ) -> Option<Array2<f64>> {
        let guard = ift_mode_response_caches().lock().unwrap();
        let cached = guard.get(&self.ift_mode_response_cache_key())?;
        if cached.rho.len() != cache.rho.len()
            || cached
                .rho
                .iter()
                .zip(cache.rho.iter())
                .any(|(&a, &b)| a.to_bits() != b.to_bits())
        {
            return None;
        }
        if cached
            .ext_mode_response_cols
            .as_ref()
            .is_some_and(|cols| cols.nrows() != self.p)
        {
            return None;
        }
        let cols = cached.rho_mode_response_cols.as_ref()?;
        if cols.nrows() != self.p || cols.ncols() != cache.rho.len() {
            return None;
        }
        Some(cols.clone())
    }

    pub(crate) fn predict_warm_start_beta_joint_ift_with_outcome(
        &self,
        new_rho: &Array1<f64>,
        max_dtheta_cap: f64,
    ) -> Option<(Coefficients, IftPredictionOutcome)> {
        let theta = self.pending_joint_ift_theta()?;
        let cache = {
            let guard = ift_joint_mode_response_caches().lock().unwrap();
            guard.get(&self.ift_mode_response_cache_key())?.clone()
        };
        if cache.active_constraints {
            log::info!(
                "[IFT-REJECTED] reason=active_constraints joint_dim={}",
                cache.theta.len(),
            );
            return None;
        }
        if !joint_ift_cache_matches_theta(&cache, &theta, new_rho)
            || cache.beta_original.len() != self.p
            || cache.mode_response_cols.nrows() != self.p
            || cache.mode_response_cols.ncols() != cache.theta.len()
        {
            return None;
        }

        let mut max_abs_dtheta = 0.0_f64;
        let dtheta: Array1<f64> = theta
            .iter()
            .zip(cache.theta.iter())
            .map(|(&new_value, &old_value)| {
                let d = new_value - old_value;
                if !d.is_finite() {
                    return f64::INFINITY;
                }
                if d.abs() > max_abs_dtheta {
                    max_abs_dtheta = d.abs();
                }
                d
            })
            .collect();
        if !max_abs_dtheta.is_finite() || max_abs_dtheta > max_dtheta_cap {
            log::info!(
                "[IFT-REJECTED] reason=large_dtheta max_dtheta={:.3e} cap={:.3e} joint_dim={}",
                max_abs_dtheta,
                max_dtheta_cap,
                cache.theta.len(),
            );
            return None;
        }
        if dtheta.iter().all(|d| d.abs() <= IFT_WARM_START_DRHO_EPS) {
            log::info!(
                "[IFT-NOOP] reason=all_dtheta_below_eps max_dtheta={:.3e} joint_dim={}",
                max_abs_dtheta,
                cache.theta.len(),
            );
            return Some((
                Coefficients::new(cache.beta_original),
                IftPredictionOutcome::Noop,
            ));
        }

        let solution_original = cache.mode_response_cols.dot(&dtheta);
        if !solution_original.iter().all(|v| v.is_finite()) {
            log::info!(
                "[IFT-REJECTED] reason=non_finite_solution max_dtheta={:.3e} joint_dim={}",
                max_abs_dtheta,
                cache.theta.len(),
            );
            return None;
        }

        let mut predicted = cache.beta_original;
        for (target, &correction) in predicted.iter_mut().zip(solution_original.iter()) {
            *target -= correction;
        }
        if !predicted.iter().all(|v| v.is_finite()) {
            log::info!(
                "[IFT-REJECTED] reason=non_finite_predicted max_dtheta={:.3e} joint_dim={}",
                max_abs_dtheta,
                cache.theta.len(),
            );
            return None;
        }
        log::info!(
            "[IFT-CACHE] outcome=joint_mode_response_hit joint_dim={} rho_dim={} p={}",
            cache.theta.len(),
            cache.rho_dim,
            self.p,
        );
        log::debug!(
            "[warm-start] joint IFT prediction reused mode responses: max|Δθ|={:.3e}, ‖Δβ‖={:.3e}",
            max_abs_dtheta,
            solution_original.dot(&solution_original).sqrt(),
        );
        Some((
            Coefficients::new(predicted),
            IftPredictionOutcome::Predicted,
        ))
    }

    pub(crate) fn joint_ift_cache_matches_pending_theta(&self, new_rho: &Array1<f64>) -> bool {
        let Some(theta) = self.pending_joint_ift_theta() else {
            return false;
        };
        let guard = ift_joint_mode_response_caches().lock().unwrap();
        let Some(cache) = guard.get(&self.ift_mode_response_cache_key()) else {
            return false;
        };
        joint_ift_cache_matches_theta(cache, &theta, new_rho)
    }

    /// Wipe every atomic-bit-packed signal used by the adaptive
    /// policies (inner-cap schedule margin, IFT |Δρ| cap, LM-λ
    /// hint clamp). Called from the same invalidation paths as
    /// `clear_warm_start_predictor_state`, plus from
    /// `execute_pirls_if_needed`'s failure branch.
    ///
    /// Single source of truth: any new adaptive-policy atomic added
    /// to `RemlObjectiveState` MUST be wiped here. The β-drift
    /// regression tests will catch correctness bugs from a stale
    /// β / H, but stale POLICY signals only surface as performance
    /// regressions (the inner solve takes longer than it should),
    /// which the bench runner's [PHASE summary] line catches via
    /// the `pirls_conv_*` and `ift_*` fields.
    pub(crate) fn clear_warm_start_adaptive_signals(&self) {
        self.last_inner_iters.store(0, Ordering::Relaxed);
        self.last_inner_converged.store(false, Ordering::Relaxed);
        self.last_pirls_lm_lambda.store(0, Ordering::Relaxed);
        // Use the NaN sentinel (not literal 0) so a residual of exactly
        // 0.0 — possible if β_predicted matched β_converged element-wise
        // — is not confused with "no signal yet".
        self.last_ift_prediction_residual
            .store(IFT_RESIDUAL_NO_SIGNAL_BITS, Ordering::Relaxed);
        // Same NaN-sentinel discipline as last_ift_prediction_residual:
        // a recorded gain ratio of exactly 0 (degenerate but possible
        // for noise-floor steps) must not collide with "no signal yet".
        self.last_pirls_accept_rho
            .store(IFT_RESIDUAL_NO_SIGNAL_BITS, Ordering::Relaxed);
        self.clear_ift_quality_runtime_state();
    }

    pub(crate) fn set_link_states(
        &mut self,
        mixture_link_state: Option<gam_problem::MixtureLinkState>,
        sas_link_state: Option<SasLinkState>,
    ) {
        let changed = self.runtime_mixture_link_state != mixture_link_state
            || self.runtime_sas_link_state != sas_link_state;
        if !changed {
            return;
        }
        self.runtime_mixture_link_state = mixture_link_state;
        self.runtime_sas_link_state = sas_link_state;
        *self.persistent_warm_start_key.write().unwrap() = None;
        self.persistent_warm_start_loaded
            .store(false, Ordering::Relaxed);
        self.invalidate_link_dependent_state();
    }

    /// Returns the eta-derivative carriers (c = dW/deta, d = d^2W/deta^2) for
    /// the exact Hessian surface that PIRLS accepted at the mode.
    ///
    /// For canonical links these coincide with the Fisher arrays; for
    /// non-canonical links they carry the clamped **observed-information**
    /// curvature used on the PIRLS left-hand side, including the residual-
    /// dependent corrections:
    ///   c_obs = c_Fisher + h'*B - (y-mu)*B_eta
    ///   d_obs = d_Fisher + h''*B + 2*h'*B_eta - (y-mu)*B_etaeta
    ///
    /// These flow into the outer REML gradient's C[v] correction term and
    /// the outer Hessian's Q[v_k, v_l] correction term. Using the observed
    /// versions is required for the exact Laplace approximation; Fisher
    /// versions would yield a PQL-type surrogate. See response.md Section 3.
    pub(crate) fn hessian_cd_arrays(
        &self,
        pirls_result: &PirlsResult,
    ) -> Result<(Array1<f64>, Array1<f64>), EstimationError> {
        Ok((
            pirls_result.solve_c_array.to_owned(),
            pirls_result.solve_d_array.to_owned(),
        ))
    }

    pub(crate) fn hessian_surface_arrays(
        &self,
        pirls_result: &PirlsResult,
    ) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), EstimationError> {
        let (c_array, d_array) = self.hessian_cd_arrays(pirls_result)?;
        Ok(crate::pirls::outer_hessian_curvature_arrays(
            pirls_result.final_weights_signed(),
            pirls_result.solve_weights_psd(),
            &c_array,
            &d_array,
            &pirls_result.final_eta.to_owned(),
            &self.config.link_kind,
        ))
    }

    /// Returns the (c, d, e) per-row mode-curvature carriers required for
    /// the analytic Tierney–Kadane c/d derivative propagation.
    ///
    /// `c` and `d` are the same observed-information arrays returned by
    /// [`hessian_cd_arrays`] (cᵢ = ∂Wᵢ/∂ηᵢ, dᵢ = ∂²Wᵢ/∂ηᵢ²); `e` is the
    /// next term in the η-Taylor expansion, eᵢ = ∂³Wᵢ/∂ηᵢ³, so that
    /// ∂c/∂θ |_β = d ⊙ ∂η/∂θ and ∂d/∂θ |_β = e ⊙ ∂η/∂θ along any chain
    /// rule that flows through η.  This is precisely what
    /// `tk_direct_gradient_from_cd_and_design` consumes.
    ///
    /// For canonical Logit on a binomial likelihood, W = h'(η)²/V(μ)
    /// reduces to W = h'(η) (because V(μ) = h'(η) for the canonical
    /// pairing μ(1−μ)), so ∂³W/∂η³ = h''''(η) and the closed-form 5-jet
    /// (`mixture_link::logit_inverse_link_jet5`) gives that exactly. We
    /// clamp to zero in saturated tails where the jet is dominated by
    /// floating-point noise (matching the existing `c_array`/`d_array`
    /// saturation handling).
    ///
    /// For non-canonical Bernoulli links (Probit, CLogLog, SAS,
    /// BetaLogistic, Mixture) and non-Bernoulli families with
    /// non-trivial observed corrections we use the analytic
    /// [`pirls::e_obs_from_jets`] formula. It expresses
    ///   ∂³W_obs/∂η³ = W_F''' + h₃ T₁ + 3 h₂ T₂ + 3 h₁ T₃ − (y−μ) T₄
    /// where T = h₁/(φV), T_k = ∂^k T/∂η^k, and W_F = h₁ T. Everything
    /// is closed-form in h₁..h₅ (inverse-link jet plus
    /// `inverse_link_pdf{third,fourth}_derivative_for_inverse_link`)
    /// and the variance jet V, V₁..V₄. The production path is fully
    /// analytic.
    pub(crate) fn hessian_cde_arrays(
        &self,
        pirls_result: &PirlsResult,
    ) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), EstimationError> {
        let c_array = pirls_result.solve_c_array.to_owned();
        let d_array = pirls_result.solve_d_array.to_owned();
        let n = d_array.len();
        let mut e_array = Array1::<f64>::zeros(n);

        let inverse_link = self.runtime_inverse_link();

        // Use the same saturation contract as PIRLS observed-Hessian
        // assembly.  If PIRLS evaluated W_obs at a clamped eta, the
        // derivative wrt the unclamped eta is zero; higher curvature carriers
        // must be zero on that branch as well.

        // Canonical-Logit fast path: W = h'(η), so ∂³W/∂η³ = h''''(η)
        // taken from the dedicated 5-jet (no variance-jet machinery).
        // Mixture links advertise `link_function() == Logit` but are
        // non-canonical; route them through the general path below.
        let canonical_logit = {
            let spec = reml_spec(&pirls_result.likelihood);
            matches!(spec.response, ResponseFamily::Binomial)
                && matches!(spec.link, InverseLink::Standard(StandardLink::Logit))
        } && self.runtime_mixture_link_state.is_none();

        if canonical_logit {
            // Canonical Logit fast path: per-row 5-jet evaluation, no
            // cross-row dependency. At large-scale n the 5-jet exp/log work
            // is the dominant cost.
            use rayon::prelude::*;
            let final_eta = &pirls_result.final_eta;
            let weights = &self.weights;
            let e_s = e_array.as_slice_mut().expect("e_array must be contiguous");
            e_s.par_iter_mut().enumerate().for_each(|(i, e_o)| {
                let eta_raw = final_eta[i];
                if pirls::eta_clamp_active(&inverse_link, eta_raw) {
                    *e_o = 0.0;
                } else {
                    let jet = crate::mixture_link::logit_inverse_link_jet5(eta_raw);
                    *e_o = weights[i].max(0.0) * jet.d4;
                }
            });
            return Ok((c_array, d_array, e_array));
        }

        // General observed-information path for non-canonical Bernoulli
        // links and other GLM families that support the observed Hessian
        // surface (Probit, CLogLog, SAS, BetaLogistic, Mixture, GammaLog).
        let likelihood = &pirls_result.likelihood;
        let weight_family = pirls::weight_family_for_glm_likelihood(likelihood);
        let phi = reml_fixed_glm_dispersion(likelihood);
        let dmu_deta = &pirls_result.solve_dmu_deta;
        let d2mu_deta2 = &pirls_result.solve_d2mu_deta2;
        let d3mu_deta3 = &pirls_result.solve_d3mu_deta3;
        if dmu_deta.len() != n || d2mu_deta2.len() != n || d3mu_deta3.len() != n {
            crate::bail_invalid_estim!(
                "Tierney-Kadane e_obs requires populated solve_*mu_deta arrays (n={}, dmu={}, d2mu={}, d3mu={}); ensure PIRLS rehydration ran",
                n,
                dmu_deta.len(),
                d2mu_deta2.len(),
                d3mu_deta3.len(),
            );
        }
        let mu = &pirls_result.solvemu;
        if mu.len() != n {
            crate::bail_invalid_estim!(
                "Tierney-Kadane e_obs requires solvemu populated (n={}, len={})",
                n,
                mu.len(),
            );
        }
        // Noncanonical / GammaLog observed-information path: each row's
        // e_i depends only on (eta[i], mu[i], priorweights[i], y[i], dmu/d2/d3
        // jets at row i, and the inverse-link's higher-order pdf derivatives
        // evaluated at eta_used). No carrier across rows. The h4/h5 lookup
        // calls return Result, so we propagate first error via try_for_each.
        use rayon::prelude::*;
        let final_eta = &pirls_result.final_eta;
        let weights = &self.weights;
        let y_view = &self.y;
        let inverse_link_ref = &inverse_link;
        let e_s = e_array.as_slice_mut().expect("e_array must be contiguous");
        e_s.par_iter_mut()
            .enumerate()
            .try_for_each(|(i, e_o)| -> Result<(), EstimationError> {
                let eta_raw = final_eta[i];
                if pirls::eta_clamp_active(inverse_link_ref, eta_raw) {
                    *e_o = 0.0;
                    return Ok(());
                }
                let h1 = dmu_deta[i];
                let h2 = d2mu_deta2[i];
                let h3 = d3mu_deta3[i];
                let h4 = crate::mixture_link::inverse_link_pdfthird_derivative_for_inverse_link(
                    inverse_link_ref,
                    eta_raw,
                )?;
                let h5 = crate::mixture_link::inverse_link_pdffourth_derivative_for_inverse_link(
                    inverse_link_ref,
                    eta_raw,
                )?;
                if !h1.is_finite()
                    || !h2.is_finite()
                    || !h3.is_finite()
                    || !h4.is_finite()
                    || !h5.is_finite()
                {
                    *e_o = 0.0;
                    return Ok(());
                }
                let mu_i = mu[i];
                let vj = pirls::variance_jet_for_weight_family(weight_family, mu_i);
                if !(vj.v.is_finite() && vj.v > 0.0) {
                    *e_o = 0.0;
                    return Ok(());
                }
                let pw = weights[i].max(0.0);
                let y_i = y_view[i];
                let e_i = pirls::e_obs_from_jets(y_i, mu_i, h1, h2, h3, h4, h5, vj, phi, pw);
                *e_o = if e_i.is_finite() { e_i } else { 0.0 };
                Ok(())
            })?;

        Ok((c_array, d_array, e_array))
    }

    pub(crate) fn hessian_cdef_arrays(
        &self,
        pirls_result: &PirlsResult,
    ) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>), EstimationError> {
        let (c_array, d_array, e_array) = self.hessian_cde_arrays(pirls_result)?;
        let canonical_logit = {
            let spec = reml_spec(&pirls_result.likelihood);
            matches!(spec.response, ResponseFamily::Binomial)
                && matches!(spec.link, InverseLink::Standard(StandardLink::Logit))
        } && self.runtime_mixture_link_state.is_none();
        if !canonical_logit {
            crate::bail_invalid_estim!(
                "Tierney-Kadane outer Hessian is implemented for canonical Binomial Logit Firth fits only"
            );
        }
        let mut f_array = Array1::<f64>::zeros(e_array.len());
        use rayon::prelude::*;
        let final_eta = &pirls_result.final_eta;
        let weights = &self.weights;
        let f_s = f_array.as_slice_mut().expect("f_array must be contiguous");
        f_s.par_iter_mut().enumerate().for_each(|(i, f_o)| {
            let eta_raw = final_eta[i];
            let eta_used = eta_raw.clamp(-ETA_OVERFLOW_CLAMP, ETA_OVERFLOW_CLAMP);
            if eta_raw != eta_used {
                *f_o = 0.0;
            } else {
                let jet = crate::mixture_link::logit_inverse_link_jet5(eta_used);
                *f_o = weights[i].max(0.0) * jet.d5;
            }
        });
        Ok((c_array, d_array, e_array, f_array))
    }

    /// Compute soft prior cost without needing workspace.
    ///
    /// The `log cosh` bound is evaluated at the weight-anchored coordinate
    /// `ρ̃ = ρ − log g(w)` (see [`rho_weight_anchor`](Self::rho_weight_anchor))
    /// so the selected λ̂ stays exactly invariant to a global prior-weight
    /// rescale `w → c·w` (issue #877). The pure-REML optimum drifts by
    /// `ρ̂ → ρ̂ + log c`; a barrier on *raw* ρ would then exert a different
    /// (asymmetric) pull at the shifted optimum, breaking the invariance. The
    /// anchor `log g(c·w) = log c + log g(w)` drifts identically to ρ̂, so the
    /// barrier's view ρ̃ — hence its cost, gradient and curvature — is identical
    /// at the rescaled optimum. With all weights 1 the anchor is exactly 0, so
    /// unweighted fits stay byte-identical.
    /// Build the soft numerical-guard ρ prior as a single atom (#931).
    ///
    /// The `log cosh` barrier's value, gradient, and diagonal Hessian were
    /// previously three separate `compute_soft_prior{cost,grad,hess}` functions
    /// that each independently re-derived the anchor, the `a = sharpness/bound`
    /// scale, and the `tanh` argument — the canonical objective↔gradient desync
    /// surface. [`SoftRhoGuardPriorAtom::evaluate_anchored`] evaluates the
    /// antiderivative chain (`log cosh → tanh → 1 − tanh²`) ONCE per coordinate,
    /// so cost, gradient, and curvature are projections of one computation and
    /// cannot drift.
    ///
    /// Evaluated at the weight-anchored coordinate `ρ̃ = ρ − rho_weight_anchor`
    /// (issue #877): the anchor is ρ-independent, so `d/dρ = d/dρ̃` and only the
    /// argument shifts.
    pub(crate) fn soft_rho_guard_prior_atom(
        &self,
        rho: &Array1<f64>,
    ) -> super::atoms::SoftRhoGuardPriorAtom {
        super::atoms::SoftRhoGuardPriorAtom::evaluate_anchored(
            rho,
            RHO_SOFT_PRIOR_WEIGHT,
            RHO_SOFT_PRIOR_SHARPNESS,
            RHO_BOUND,
            self.rho_weight_anchor(),
        )
    }

    // Gamma(a, b) precision hyperprior identity.
    //
    // For one penalty block, lambda = exp(rho) and
    // p(lambda) proportional to lambda^(a-1) exp(-b lambda). REML/LAML
    // smoothing-parameter optimization targets the mode in lambda, expressed
    // on the rho coordinate for numerical stability. Therefore no
    // change-of-variables Jacobian is added here; the hyperprior contributes
    //
    //     -log p(exp(rho)) = b exp(rho) - (a - 1) rho + constant.
    //
    // The likelihood/penalty part already contributes the usual
    // nu_p/2 log(lambda) - lambda (beta-mu_p)' S_p (beta-mu_p)/2
    // terms through the assembled
    // penalty coordinates. Adding the term below gives the conditional MAP
    // score
    //
    //     d/dlambda: (a - 1 + nu_p/2)/lambda
    //         - (b + (beta-mu_p)' S_p (beta-mu_p)/2) = 0,
    //
    // hence lambda* = (a - 1 + nu_p/2)
    //     / (b + (beta-mu_p)' S_p (beta-mu_p)/2), reducing to the existing
    // MacKay/Tipping fixed point when (a, b) = (1, 0).
    /// Evaluate the configured ρ prior through the shared
    /// [`rho_prior_eval`](crate::rho_prior_eval) engine under the REML/LAML
    /// invalid-prior policy
    /// ([`Saturate`](crate::rho_prior_eval::InvalidPriorPolicy::Saturate)): a
    /// malformed prior folds into the objective as `+inf` cost (and `NaN`
    /// gradient/Hessian) so the outer optimizer steps away from it. The math
    /// itself is shared with custom-family handling.
    pub(crate) fn evaluate_configured_rho_prior(
        &self,
        rho: &Array1<f64>,
    ) -> crate::rho_prior_eval::RhoPriorEval {
        let effective = self.effective_rho_prior();
        // Evaluate the prior at the weight-anchored coordinate `ρ̃ = ρ − log g(w)`
        // (see [`rho_weight_anchor`](Self::rho_weight_anchor)) so the selected λ̂
        // is exactly invariant to a global weight rescale `w → c·w` (issue #877).
        // The anchor is a ρ-independent constant, so d/dρ̃ = d/dρ: the returned
        // gradient and Hessian are already correct w.r.t. ρ. For unweighted fits
        // the anchor is 0 and `rho_eff` aliases `rho` (byte-identical behaviour).
        let anchor = self.rho_weight_anchor();
        let rho_anchored = (anchor != 0.0).then(|| rho.mapv(|r| r - anchor));
        let rho_eff: &Array1<f64> = rho_anchored.as_ref().unwrap_or(rho);
        let mut eval = crate::rho_prior_eval::evaluate(
            effective.as_ref(),
            rho_eff,
            crate::rho_prior_eval::InvalidPriorPolicy::Saturate,
        )
        .expect("Saturate policy never errors");
        // FIRTH-DEFAULT SELF-GATE (strict zero-downside). The shared engine
        // evaluated every firth-default-filled coordinate as a plain PC term,
        // whose gradient carries the persistent `+1/2` Occam pull that perturbs
        // even a well-identified λ by O(1/n). Replace those coordinates'
        // contribution with the SELF-GATED one-sided barrier
        // `firth_default_barrier_terms`, which is byte-identically flat on the
        // identified side (ρ ≥ −2 ln upper) and only a convex wall against the
        // λ → 0 degeneracy below it — so a clean/well-conditioned fit stays
        // byte-identical to plain REML. Explicitly-configured priors (including a
        // user-supplied PenalizedComplexity) are left exactly as the engine
        // produced them. If the saturating policy already flagged a malformed
        // prior (non-finite cost), leave it untouched so the repulsion stands.
        if eval.cost.is_finite() {
            let mask = firth_default_coord_mask(&self.rho_prior, rho.len());
            if mask.iter().any(|&d| d) {
                let theta = crate::rho_prior_eval::pc_prior_rate(
                    FIRTH_DEFAULT_PC_UPPER,
                    FIRTH_DEFAULT_PC_TAIL_PROB,
                );
                let mut hess = eval
                    .hessian
                    .take()
                    .unwrap_or_else(|| Array2::<f64>::zeros((rho.len(), rho.len())));
                for (idx, &is_default) in mask.iter().enumerate() {
                    if !is_default {
                        continue;
                    }
                    let r = rho_eff[idx];
                    // Remove the plain PC contribution the engine added for this
                    // defaulted coordinate, then add the self-gated barrier.
                    let (pc_c, pc_g, pc_h) = crate::rho_prior_eval::pc_prior_terms(theta, r);
                    let (b_c, b_g, b_h) = crate::rho_prior_eval::firth_default_barrier_terms(
                        theta,
                        FIRTH_DEFAULT_PC_UPPER,
                        r,
                    );
                    eval.cost += b_c - pc_c;
                    eval.gradient[idx] += b_g - pc_g;
                    hess[[idx, idx]] += b_h - pc_h;
                }
                eval.hessian = hess.iter().any(|&v| v != 0.0).then_some(hess);
            }
        }
        eval
    }

    /// Emit the configured ρ-prior as a criterion atom after every REML/LAML
    /// prior policy has been applied. Callers project cost, gradient, and
    /// Hessian from this one object instead of making separate evaluator calls.
    pub(crate) fn configured_rho_prior_atom(
        &self,
        rho: &Array1<f64>,
    ) -> super::atoms::ConfiguredRhoPriorAtom {
        super::atoms::ConfiguredRhoPriorAtom {
            eval: self.evaluate_configured_rho_prior(rho),
        }
    }

    /// Resolve the *effective* outer prior on the log-precision ρ, applying the
    /// (unconditional) firth-general default-hyperprior policy.
    ///
    /// An *unset* prior (the `Flat` sentinel, i.e. the caller did not configure a
    /// hyperprior on a coordinate) is replaced by the weakly-informative
    /// penalized-complexity
    /// (PC) prior [`firth_default_pc_prior`], turning the outer REML point into a
    /// proper marginal posterior over λ. A PC prior is reparameterization-
    /// invariant and shrinks only toward the simpler (more-smoothing) base model,
    /// so it removes the λ→0 degeneracy without ever walling off complexity the
    /// data actually buys. Any explicitly configured prior (Normal, Gamma, an
    /// already-PC coordinate, ...) is respected unchanged — the default only
    /// fills `Flat` holes.
    ///
    /// The default is *weakly* informative by construction: its calibrated rate
    /// `θ = −ln(tail_prob)/upper` is small, so its O(1) cost/gradient is
    /// dominated by the O(n) REML curvature wherever λ is well identified,
    /// leaving clean λ-selection unbiased (the zero-downside / information-limit
    /// reduction to plain REML).
    pub(crate) fn effective_rho_prior(&self) -> std::borrow::Cow<'_, RhoPrior> {
        resolve_effective_rho_prior(&self.rho_prior)
    }

    /// ½·Σᵢ log(wᵢ) over the positive-weight rows — the per-observation
    /// Gaussian normalization constant that the log-likelihood drops.
    ///
    /// `Var(yᵢ) = φ/wᵢ` under inverse-variance prior weights, so the full
    /// weighted-Gaussian normalization is `½ Σ log(2π φ/wᵢ) =
    /// (n/2) log(2πφ) − ½ Σ log wᵢ`; the `calculate_loglikelihood_omitting_constants`
    /// helper omits the `−½ Σ log wᵢ` piece. The `ProfiledGaussian` REML cost
    /// adds it back (`InnerSolution::gaussian_weight_log_sum_half`) so the
    /// objective VALUE is exactly invariant to a global prior-weight rescale
    /// `w → c·w`: the invariance-preserving `λ → c·λ` otherwise inflates the cost
    /// value by `(n/2) log c`. This term only restores the *value*; it is a
    /// ρ-independent constant and so cannot move the argmin. The *argmin*
    /// invariance of the selected λ̂ — the substance of issue #877 — is restored
    /// separately by [`rho_weight_anchor`](Self::rho_weight_anchor), which
    /// evaluates the configured ρ-prior at the weight-anchored coordinate.
    /// With all weights 1 this is exactly 0. Summed over the SAME positive-weight rows
    /// counted in `n_observations` (zero-weight rows are dropped; `log(0)` is
    /// undefined).
    pub(crate) fn gaussian_weight_log_sum_half(&self) -> f64 {
        // #1033: `½·Σ log wᵢ` is a pure function of the fit-invariant `weights`
        // view (never reassigned by `reset_surface`), so memoize the O(n)
        // reduction once per fit. Every subsequent in-window κ-trial eval reads
        // the cached scalar instead of re-scanning n rows — the per-trial eval
        // then touches only k-dim objects.
        *self
            .gaussian_weight_log_sum_half_cache
            .get_or_init(|| self.gaussian_weight_log_sum_half_uncached())
    }

    fn gaussian_weight_log_sum_half_uncached(&self) -> f64 {
        0.5 * self
            .weights
            .iter()
            .filter(|&&wi| wi > 0.0)
            .map(|&wi| wi.ln())
            .sum::<f64>()
    }

    /// Weighted null deviance `D₀ = Σ wᵢ(yᵢ − ȳ_w)²` of the Gaussian response,
    /// used as the *relative* reference scale for the smooth penalized-deviance
    /// floor (`InnerSolution::dp_floor_scale`, see
    /// [`crate::estimate::smooth_floor_dp`]).
    ///
    /// `D₀` is the deviance of the intercept-only model — the natural upper
    /// bound for the penalized deviance `D_p` — and is the right scale at which
    /// to judge "is `D_p` essentially zero?". It is computed once per fit from
    /// the data alone (no ρ dependence), so flooring `D_p` at a fixed fraction
    /// of it cannot perturb the ρ-gradient, only guard the `log φ̂` singularity
    /// at a perfect fit. Crucially it is exactly quadratic in the response, so
    /// under a rescale `y → a·y` it transforms as `D₀ → a²·D₀` in lockstep with
    /// `D_p`, which is what makes the profiled Gaussian REML criterion — and
    /// hence the selected `λ̂`, the EDF, and `ŝ(x)/a` — exactly scale-equivariant
    /// (issue #1127).
    ///
    /// Falls back to `1.0` (the historical absolute floor) when the weighted
    /// null deviance is not a usable positive scale — a degenerate constant
    /// response, where `D_p` is genuinely ~0 and equivariance is vacuous
    /// (`a·const` is still constant).
    pub(crate) fn gaussian_dp_floor_scale(&self) -> f64 {
        // #1033: the weighted null deviance `D₀` depends only on the
        // fit-invariant `(y, weights)` views (never reassigned by
        // `reset_surface`), so memoize the two O(n) passes once per fit. Every
        // in-window κ-trial eval then reads the cached scalar rather than
        // re-scanning n rows.
        *self
            .gaussian_dp_floor_scale_cache
            .get_or_init(|| self.gaussian_dp_floor_scale_uncached())
    }

    fn gaussian_dp_floor_scale_uncached(&self) -> f64 {
        let mut sw = 0.0_f64;
        let mut swy = 0.0_f64;
        for (&yi, &wi) in self.y.iter().zip(self.weights.iter()) {
            if wi > 0.0 && yi.is_finite() {
                sw += wi;
                swy += wi * yi;
            }
        }
        if sw <= 0.0 {
            return 1.0;
        }
        let ybar = swy / sw;
        let mut d0 = 0.0_f64;
        for (&yi, &wi) in self.y.iter().zip(self.weights.iter()) {
            if wi > 0.0 && yi.is_finite() {
                let r = yi - ybar;
                d0 += wi * r * r;
            }
        }
        if d0.is_finite() && d0 > 0.0 { d0 } else { 1.0 }
    }

    /// Geometric-mean log-weight anchor `log g(w) = (1/n₊)·Σ log wᵢ` over the
    /// positive-weight rows — **but only for a profiled-dispersion family**
    /// (Gaussian-identity). Fixed-dispersion families return exactly `0`.
    ///
    /// The configured outer ρ-prior is evaluated at the weight-anchored
    /// coordinate `ρ̃ = ρ − log g(w)` so that the selected λ̂ is *exactly*
    /// invariant to a global prior-weight rescale `w → c·w` (issue #877). Under
    /// inverse-variance weights the penalized Hessian is `XᵀWX + λS`, so the
    /// pure-REML optimum drifts by `ρ̂ → ρ̂ + log c` while the fit (β̂, EDF,
    /// predictions) is unchanged. A prior on *raw* ρ (e.g. the default
    /// `Normal{0, sd}`) would then pull the optimum back by `log(c)/sd²`,
    /// breaking the invariance — the exact defect #877 reports (λ̂ ratio 810×
    /// not 1000×). Anchoring removes it: `log g(c·w) = log c + log g(w)` drifts
    /// identically to ρ̂, so the prior's view `ρ̃` — hence its cost, gradient and
    /// curvature — is identical at the rescaled optimum. The
    /// [`gaussian_weight_log_sum_half`](Self::gaussian_weight_log_sum_half) cost
    /// term keeps the objective *value* invariant; this keeps its *argmin*
    /// invariant. With all weights 1 the anchor is exactly 0, so unweighted fits
    /// (the overwhelming majority) stay byte-identical.
    ///
    /// # Why this is gated on profiled dispersion (issue #893)
    ///
    /// The drift `ρ̂ → ρ̂ + log c` and the fit-invariance it compensates are a
    /// *profiled*-dispersion phenomenon: only when the scale φ̂ absorbs the
    /// weight magnitude (`Var = φ/w`, φ profiled) does a global rescale leave
    /// β̂/EDF fixed, so that the only correct response is to slide λ̂ — and the
    /// prior must slide with it. For a **fixed-dispersion** family (φ ≡ 1:
    /// Poisson, binomial, …) the weight has a *different* meaning entirely: a
    /// uniform prior weight `w = c` is exact `c`-fold row replication, and
    /// genuinely more data. Two encodings of the same data — one row with
    /// weight `c` vs `c` literal copies — share an identical penalized deviance
    /// `D_p` and identical `XᵀWX`, hence an identical fixed-dispersion LAML
    /// objective `V(ρ)` and an identical pure-LAML optimum `ρ̂₀` (there is no
    /// `log c` drift; the data and Occam terms are weighted differently, but
    /// *identically* between the two encodings). They differ only in their
    /// per-row log-weight mean — `log c` for the weighted row, `0` for the
    /// replicated copies. Applying the geometric-mean anchor would therefore
    /// evaluate the regularizing prior at *different* `ρ̃` for the two encodings
    /// (`ρ − log c` vs `ρ`), pulling λ̂ apart and making the weighted encoding
    /// systematically over-smooth (issue #893). The encoding-invariant choice
    /// is anchor `0`: the prior then acts identically on both encodings (it does
    /// not need to track the optimum — being O(1) against the O(Σwᵢ) data
    /// curvature, it is negligible either way; what matters is that it is the
    /// *same* for the two encodings). Concretely the only weight summary that is
    /// invariant under `w=c ↔ c-fold replication` is the total weight `Σwᵢ`, not
    /// the per-row geometric mean `log g(w)` or the row count `n₊`.
    pub(crate) fn rho_weight_anchor(&self) -> f64 {
        // Fixed-dispersion families: no profiled scale to absorb the weight
        // magnitude, so there is no `log c` optimum drift to compensate and a
        // nonzero anchor would break the `w=c ⇔ c-fold replication` equivalence
        // in λ-selection (issue #893). Only Gaussian-identity is profiled here
        // (see the `ProfiledGaussian` vs `Fixed` split in `build_*_context`).
        if !reml_is_gaussian_identity(&self.config.likelihood) {
            return 0.0;
        }
        let mut sum = 0.0;
        let mut count = 0usize;
        for &wi in self.weights.iter() {
            if wi > 0.0 {
                sum += wi.ln();
                count += 1;
            }
        }
        if count == 0 { 0.0 } else { sum / count as f64 }
    }

    /// Returns the effective Hessian and the ridge value used (if any).
    /// Uses the same Hessian matrix in both cost and gradient calculations.
    ///
    /// PIRLS folds any stabilization ridge directly into the penalized objective:
    ///   l_p(beta; rho) = l(beta) - 0.5 * beta^T (S_lambda + ridge I) beta.
    /// Therefore the curvature used in LAML is
    ///   H_eff = X' W_H X + S_lambda + ridge I,
    /// and adding another ridge here places the Laplace expansion on a different surface.
    ///
    /// IMPORTANT: W_H here is the **final observed-information** weight for
    /// non-canonical links (or Fisher weight for canonical links, where the two
    /// coincide). PIRLS may rehydrate final Laplace curvature arrays after compacting
    /// the accepted state; reusing `stabilizedhessian_transformed` would then evaluate
    /// log|H_eff| on the accepted-step weights while the hyper-gradient differentiates
    /// the final curvature arrays. Rebuilding from `finalweights` keeps the REML value
    /// and hyper-derivative on the same Hessian surface.
    ///
    pub(super) fn effectivehessian(
        &self,
        pr: &PirlsResult,
    ) -> Result<(Array2<f64>, RidgePassport), EstimationError> {
        // Use the same stabilized H = X' W X + S + δI that PIRLS built,
        // where W = `solver_weights = max(W_obs, floor(W_F))` keeps H PD.
        // The OUTER analytic operator constructs ∂H/∂ψ from the same
        // floored W (via `outer_hessian_curvature_arrays`), so log|H| and
        // its trace gradient live on a single, consistent surface.
        let h = &pr.stabilizedhessian_transformed;
        if h.factorize().is_ok() {
            return Ok((h.to_dense(), pr.ridge_passport));
        }

        Err(EstimationError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        })
    }

    pub(crate) fn newwith_offset<X>(
        y: ArrayView1<'a, f64>,
        x: X,
        weights: ArrayView1<'a, f64>,
        offset: ArrayView1<'_, f64>,
        canonical_penalties: Vec<gam_terms::construction::CanonicalPenalty>,
        p: usize,
        config: &'a RemlConfig,
        nullspace_dims: Option<Vec<usize>>,
        coefficient_lower_bounds: Option<Array1<f64>>,
        linear_constraints: Option<crate::pirls::LinearInequalityConstraints>,
    ) -> Result<Self, EstimationError>
    where
        X: Into<DesignMatrix>,
    {
        Self::newwith_offset_shared(
            y,
            x,
            weights,
            offset,
            Arc::new(canonical_penalties),
            p,
            Arc::new(config.clone()),
            nullspace_dims,
            coefficient_lower_bounds,
            linear_constraints,
        )
    }

    pub(crate) fn newwith_offset_shared<X>(
        y: ArrayView1<'a, f64>,
        x: X,
        weights: ArrayView1<'a, f64>,
        offset: ArrayView1<'_, f64>,
        canonical_penalties: Arc<Vec<gam_terms::construction::CanonicalPenalty>>,
        p: usize,
        config: Arc<RemlConfig>,
        nullspace_dims: Option<Vec<usize>>,
        coefficient_lower_bounds: Option<Array1<f64>>,
        linear_constraints: Option<crate::pirls::LinearInequalityConstraints>,
    ) -> Result<Self, EstimationError>
    where
        X: Into<DesignMatrix>,
    {
        let x = x.into();

        // Single-shot structural redundancy diagnostic. Fires exactly once
        // per RemlState construction — see
        // `gam_terms::construction::report_penalty_pair_redundancy` for the
        // logging policy. O(k² · block_dim²); negligible at fit-prep.
        gam_terms::construction::report_penalty_pair_redundancy(canonical_penalties.as_ref());

        let expected_len = canonical_penalties.len();
        let nullspace_dims = match nullspace_dims {
            Some(dims) => {
                if dims.len() != expected_len {
                    crate::bail_invalid_estim!(
                        "nullspace_dims length {} does not match penalties {}",
                        dims.len(),
                        expected_len
                    );
                }
                dims
            }
            None => vec![0; expected_len],
        };

        let balanced_penalty_root =
            create_balanced_penalty_root_from_canonical(&canonical_penalties, p)?;
        let reparam_invariant =
            precompute_reparam_invariant_from_canonical(&canonical_penalties, p)?;

        let sparse_penalty_block_count =
            sparse_penalty_block_count_from_canonical(canonical_penalties.as_ref(), p)?;

        let runtime_mixture_link_state = config.link_kind.mixture_state().cloned();
        let runtime_sas_link_state = config.link_kind.sas_state().copied();

        Ok(Self {
            y,
            x,
            weights,
            offset: offset.to_owned(),
            canonical_penalties,
            balanced_penalty_root,
            reparam_invariant,
            sparse_penalty_block_count,
            p,
            config,
            runtime_mixture_link_state,
            runtime_sas_link_state,
            nullspace_dims,
            coefficient_lower_bounds,
            linear_constraints,
            penalty_shrinkage_floor: None,
            rho_prior: RhoPrior::Flat,
            cache_manager: EvalCacheManager::new(),
            arena: RemlArena::new(),
            warm_start_beta: RwLock::new(None),
            warm_start_rho: RwLock::new(None),
            prev_warm_start_beta: RwLock::new(None),
            prev_warm_start_rho: RwLock::new(None),
            warm_start_enabled: AtomicBool::new(true),
            screening_max_inner_iterations: Arc::new(AtomicUsize::new(0)),
            outer_inner_cap: Arc::new(AtomicUsize::new(0)),
            last_inner_iters: Arc::new(AtomicUsize::new(0)),
            last_inner_converged: Arc::new(AtomicBool::new(false)),
            ift_warm_start_cache: RwLock::new(None),
            last_pirls_lm_lambda: Arc::new(AtomicU64::new(0)),
            frozen_negbin_theta: Arc::new(AtomicU64::new(0)),
            frozen_tweedie_phi: Arc::new(AtomicU64::new(0)),
            frozen_gamma_shape: Arc::new(AtomicU64::new(0)),
            last_ift_prediction_residual: Arc::new(AtomicU64::new(IFT_RESIDUAL_NO_SIGNAL_BITS)),
            last_pirls_accept_rho: Arc::new(AtomicU64::new(IFT_RESIDUAL_NO_SIGNAL_BITS)),
            ift_cached_factor: RwLock::new(None),
            kronecker_penalty_system: None,
            kronecker_factored: None,
            gaussian_fixed_cache: RwLock::new(None),
            gaussian_psi_gram_deriv: RwLock::new(None),
            glm_psi_gram_deriv: RwLock::new(None),
            glm_first_step_gram: RwLock::new(None),
            flat_glm_first_step_gram: RwLock::new(None),
            alo_frozen_nuisance: RwLock::new(None),
            alo_provably_inactive: RwLock::new(None),
            persistent_warm_start_key: RwLock::new(None),
            persistent_latent_values_fingerprint: None,
            persistent_latent_values_cache: RwLock::new(PersistentLatentValuesCache::default()),
            analytic_penalty_registry_fingerprint: 0,
            persistent_warm_start_loaded: AtomicBool::new(false),
            persistent_warm_start_store_suppression: AtomicUsize::new(0),
            alo_stabilization_suppression: AtomicUsize::new(0),
            persistent_warm_start_disk_enabled: AtomicBool::new(false),
            gaussian_weight_log_sum_half_cache: std::sync::OnceLock::new(),
            gaussian_dp_floor_scale_cache: std::sync::OnceLock::new(),
        })
    }

    pub(in crate::estimate) fn reset_surface<X>(
        &mut self,
        x: X,
        canonical_penalties: Arc<Vec<gam_terms::construction::CanonicalPenalty>>,
        p: usize,
        nullspace_dims: Vec<usize>,
        coefficient_lower_bounds: Option<Array1<f64>>,
        linear_constraints: Option<crate::pirls::LinearInequalityConstraints>,
        kronecker_penalty_system: Option<gam_terms::smooth::KroneckerPenaltySystem>,
        kronecker_factored: Option<gam_terms::basis::KroneckerFactoredBasis>,
    ) -> Result<(), EstimationError>
    where
        X: Into<DesignMatrix>,
    {
        let expected_len = canonical_penalties.len();
        if nullspace_dims.len() != expected_len {
            crate::bail_invalid_estim!(
                "nullspace_dims length {} does not match penalties {}",
                nullspace_dims.len(),
                expected_len
            );
        }

        let balanced_penalty_root =
            create_balanced_penalty_root_from_canonical(&canonical_penalties, p)?;
        let reparam_invariant =
            precompute_reparam_invariant_from_canonical(&canonical_penalties, p)?;
        let sparse_penalty_block_count =
            sparse_penalty_block_count_from_canonical(canonical_penalties.as_ref(), p)?;

        self.x = x.into();
        self.canonical_penalties = canonical_penalties;
        self.balanced_penalty_root = balanced_penalty_root;
        self.reparam_invariant = reparam_invariant;
        self.sparse_penalty_block_count = sparse_penalty_block_count;
        self.p = p;
        self.nullspace_dims = nullspace_dims;
        self.coefficient_lower_bounds = coefficient_lower_bounds;
        self.linear_constraints = linear_constraints;
        self.kronecker_penalty_system = kronecker_penalty_system;
        self.kronecker_factored = kronecker_factored;
        // The Gaussian-fixed cache is keyed to (X, y, w, offset); replacing the
        // design invalidates it. The new surface will repopulate it on demand.
        *self.gaussian_fixed_cache.write().unwrap() = None;
        // The conditioned-frame ψ-gram derivative is keyed to the same design;
        // a new design invalidates it. The installing trial repopulates it.
        *self.gaussian_psi_gram_deriv.write().unwrap() = None;
        // The GLM frozen-W conditioned-frame ψ-gram derivative is keyed to the
        // same design + ψ; a new design invalidates it. The installing trial
        // repopulates it.
        *self.glm_psi_gram_deriv.write().unwrap() = None;
        // The frozen-W GLM first-step Gram is keyed to the same design + ψ; a
        // new design invalidates it. The installing trial repopulates it.
        *self.glm_first_step_gram.write().unwrap() = None;
        // The flat-warm-start GLM first-step Gram is keyed to the previous
        // surface's design and warm β; a surface reset invalidates both.
        *self.flat_glm_first_step_gram.write().unwrap() = None;
        *self.alo_frozen_nuisance.write().unwrap() = None;
        *self.persistent_warm_start_key.write().unwrap() = None;
        self.persistent_warm_start_loaded
            .store(false, Ordering::Relaxed);
        self.persistent_warm_start_store_suppression
            .store(0, Ordering::Relaxed);
        self.cache_manager.clear_eval_and_factor_caches();
        self.cache_manager.pirls_cache.write().unwrap().clear();
        // The new surface has a different design / penalty system /
        // working-weight curvature, so every warm-start signal calibrated
        // to the previous surface is now stale. Wipe in lockstep — see
        // the helpers' doc-comments for the per-slot staleness arguments.
        self.clear_warm_start_predictor_state();
        self.clear_warm_start_adaptive_signals();
        self.reset_hypergradient_budget_controller();
        // The λ-search frozen NB θ (#1082) is computed from the seed fit on the
        // PREVIOUS design; a new surface (different X / penalties) must re-freeze
        // it from its own seed. `0` = "not yet frozen".
        self.frozen_negbin_theta.store(0, Ordering::Relaxed);
        // The λ-search frozen Tweedie φ (#1477) is likewise computed from the
        // seed fit on the PREVIOUS design; re-freeze it from the new surface's
        // own seed. `0` = "not yet frozen".
        self.frozen_tweedie_phi.store(0, Ordering::Relaxed);
        // The λ-search frozen Gamma shape (#1074) is likewise computed from the
        // seed fit on the PREVIOUS design; re-freeze it from the new surface's
        // own seed. `0` = "not yet frozen".
        self.frozen_gamma_shape.store(0, Ordering::Relaxed);
        Ok(())
    }

    /// #1033: refresh ONLY the canonical-penalty surface in k-space, keeping the
    /// realized n×k design `self.x` (and the externally-installed Gaussian Gram
    /// cache) untouched.
    ///
    /// On the certified κ-loop fast path the design coordinate ψ does NOT move
    /// the rows we re-stream — the n-free `PsiGramTensor` serves `XᵀWX(ψ)/XᵀWz(ψ)`
    /// — but for a spatial smooth ψ ALSO moves the penalty matrix `S(ψ)` (the
    /// Duchon/Matérn Hilbert scale is built as a function of the length scale).
    /// `reset_surface` is the only place the canonical penalty surface
    /// (`balanced_penalty_root` / `reparam_invariant` / `sparse_penalty_block_count`)
    /// is rebuilt, and the fast path skips it — so without this the inner solve
    /// would pair `XᵀWX(ψ_new)` with the STALE `S(ψ_old)` and converge to the
    /// wrong β̂ / κ-optimum. This re-keys `S(ψ_new)` from the supplied canonical
    /// penalties (a k×k object built from the basis centers, not the data rows,
    /// so the refresh stays n-free) and re-runs exactly the three k-space penalty
    /// derivations `reset_surface` runs — nothing design- or n-shaped.
    ///
    /// It does NOT touch `self.x`, the Gaussian-fixed Gram cache, or the
    /// conditioned-frame ψ-gram derivative: those are re-keyed to ψ_new by the
    /// tensor lane independently. It DOES invalidate the eval/factor/PIRLS caches
    /// (the penalized Hessian `H_λ = QsᵀXᵀWXQs + S(ψ)` changed), so the next inner
    /// solve refactors against the correct penalty.
    pub(crate) fn refresh_canonical_penalty_surface(
        &mut self,
        canonical_penalties: Arc<Vec<gam_terms::construction::CanonicalPenalty>>,
        nullspace_dims: Vec<usize>,
    ) -> Result<(), EstimationError> {
        if canonical_penalties.len() != self.canonical_penalties.len() {
            crate::bail_invalid_estim!(
                "refresh_canonical_penalty_surface: penalty count changed ({} → {}) — the \
                 fast path requires a fixed penalty topology",
                self.canonical_penalties.len(),
                canonical_penalties.len()
            );
        }
        if nullspace_dims.len() != canonical_penalties.len() {
            crate::bail_invalid_estim!(
                "refresh_canonical_penalty_surface: nullspace_dims length {} does not match \
                 penalties {}",
                nullspace_dims.len(),
                canonical_penalties.len()
            );
        }
        let p = self.p;
        let balanced_penalty_root =
            create_balanced_penalty_root_from_canonical(&canonical_penalties, p)?;
        let reparam_invariant =
            precompute_reparam_invariant_from_canonical(&canonical_penalties, p)?;
        let sparse_penalty_block_count =
            sparse_penalty_block_count_from_canonical(canonical_penalties.as_ref(), p)?;

        self.canonical_penalties = canonical_penalties;
        self.balanced_penalty_root = balanced_penalty_root;
        self.reparam_invariant = reparam_invariant;
        self.sparse_penalty_block_count = sparse_penalty_block_count;
        self.nullspace_dims = nullspace_dims;
        // The penalized Hessian / logdet depend on S(ψ); a new penalty
        // invalidates every memoized factorization and eval. The design-keyed
        // Gaussian Gram cache and its ψ-derivative are NOT cleared here: those
        // are re-keyed to ψ_new by the tensor lane (`install_psi_gram_statistics`).
        self.cache_manager.clear_eval_and_factor_caches();
        self.cache_manager.pirls_cache.write().unwrap().clear();
        Ok(())
    }

    /// Inject Kronecker penalty system metadata for tensor-product smooth terms.
    ///
    /// When set, the REML evaluator will use O(∏q_j) logdet instead of O(p³)
    /// eigendecomposition.  Also stores the full factored basis so that P-IRLS
    /// can use factored reparameterization (Qs = U_1 ⊗ ... ⊗ U_d).
    pub(crate) fn set_kronecker_penalty_system(
        &mut self,
        system: gam_terms::smooth::KroneckerPenaltySystem,
    ) {
        self.kronecker_penalty_system = Some(system);
    }

    /// Inject the full Kronecker factored basis for P-IRLS factored reparameterization.
    pub(crate) fn set_kronecker_factored(
        &mut self,
        factored: gam_terms::basis::KroneckerFactoredBasis,
    ) {
        self.kronecker_factored = Some(factored);
    }

    /// Sets the shrinkage floor for penalized block eigenvalues.
    /// The ridge magnitude will be `epsilon * max_balanced_eigenvalue` (rho-independent).
    /// This prevents barely-penalized directions from causing pathological
    /// non-Gaussianity in the posterior. Typical value: `Some(1e-6)`.
    pub(crate) fn set_penalty_shrinkage_floor(&mut self, floor: Option<f64>) {
        self.penalty_shrinkage_floor = floor;
        *self.persistent_warm_start_key.write().unwrap() = None;
        self.persistent_warm_start_loaded
            .store(false, Ordering::Relaxed);
    }

    pub(crate) fn set_rho_prior(&mut self, prior: RhoPrior) {
        self.rho_prior = prior;
        *self.persistent_warm_start_key.write().unwrap() = None;
        self.persistent_warm_start_loaded
            .store(false, Ordering::Relaxed);
    }

    pub(in crate::estimate) fn set_analytic_penalty_registry_fingerprint(
        &mut self,
        fingerprint: u64,
    ) {
        if self.analytic_penalty_registry_fingerprint == fingerprint {
            return;
        }
        self.analytic_penalty_registry_fingerprint = fingerprint;
        *self.persistent_warm_start_key.write().unwrap() = None;
        self.persistent_warm_start_loaded
            .store(false, Ordering::Relaxed);
    }

    pub(crate) fn set_persistent_latent_values_fingerprint(&mut self, fingerprint: u64) {
        self.persistent_latent_values_fingerprint = Some(fingerprint);
    }

    /// Creates a sanitized cache key from rho values.
    /// Returns None if any component is NaN, in which case caching is skipped.
    /// Maps -0.0 to 0.0 to ensure consistency in caching.
    pub(super) fn rhokey_sanitized(&self, rho: &Array1<f64>) -> Option<Vec<u64>> {
        EvalCacheManager::sanitized_rhokey(rho)
    }

    pub(super) fn prepare_eval_bundlewithkey(
        &self,
        rho: &Array1<f64>,
        key: Option<Vec<u64>>,
    ) -> Result<EvalShared, EstimationError> {
        // #1575 observability: count every genuine (cache-missing) full-n inner
        // P-IRLS solve. Callers funnel cache hits through `obtain_eval_bundle*`,
        // which only reaches here on a miss, so this counts exactly the
        // expensive solves the #1575 slowdown is measured in. Relaxed ordering
        // is sufficient — this is a monotone diagnostic counter, never read on
        // the math path.
        self.arena
            .inner_pirls_solve_count
            .fetch_add(1, Ordering::Relaxed);
        let decision = self.select_reml_geometry(rho);
        match decision.geometry {
            RemlGeometry::SparseExactSpd => {
                match self.prepare_sparse_eval_bundlewithkey(rho, key.clone()) {
                    Ok(bundle) => {
                        log::info!(
                            "[reml-geometry] sparse_exact_spd reason={} p={} nnz_x={} nnz_h_est={} density_h_est={}",
                            decision.reason,
                            decision.p,
                            decision.nnz_x,
                            decision
                                .nnz_h_upper_est
                                .map(|v| v.to_string())
                                .unwrap_or_else(|| "na".to_string()),
                            decision
                                .density_h_upper_est
                                .map(|v| format!("{v:.4}"))
                                .unwrap_or_else(|| "na".to_string()),
                        );
                        Ok(bundle)
                    }
                    Err(err) => {
                        log::warn!(
                            "[reml-geometry] sparse_exact_spd failed ({}); falling back to dense spectral",
                            err
                        );
                        self.prepare_dense_eval_bundlewithkey(rho, key)
                    }
                }
            }
            RemlGeometry::DenseSpectral => self.prepare_dense_eval_bundlewithkey(rho, key),
        }
    }

    pub(crate) fn obtain_eval_bundle(
        &self,
        rho: &Array1<f64>,
    ) -> Result<EvalShared, EstimationError> {
        let key = self.rhokey_sanitized(rho);
        if let Some(existing) = self.cache_manager.cached_eval_bundle(&key) {
            return Ok(existing.clone());
        }
        let bundle = self.prepare_eval_bundlewithkey(rho, key)?;
        self.cache_manager.store_eval_bundle(bundle.clone());
        Ok(bundle)
    }

    /// Fixes audit answer C for design-moving ext-coords: when the realized
    /// design has been rebuilt from a full outer vector `(rho, psi/z)`, the
    /// PIRLS bundle cache key must include those ext-coords rather than rho
    /// alone.
    pub(crate) fn obtain_eval_bundle_for_outer_theta(
        &self,
        rho: &Array1<f64>,
        theta: &Array1<f64>,
    ) -> Result<EvalShared, EstimationError> {
        let key = self.rhokey_sanitized(theta);
        if let Some(existing) = self.cache_manager.cached_eval_bundle(&key) {
            return Ok(existing.clone());
        }
        let bundle = self.prepare_eval_bundlewithkey(rho, key)?;
        self.cache_manager.store_eval_bundle(bundle.clone());
        Ok(bundle)
    }

    pub(crate) fn objective_innerhessian(
        &self,
        rho: &Array1<f64>,
    ) -> Result<Array2<f64>, EstimationError> {
        let bundle = self.obtain_eval_bundle(rho)?;
        if let Some(sparse) = bundle.sparse_exact.as_ref() {
            let h = gam_linalg::sparse_exact::assemble_sparse_factor_h_dense(&sparse.factor)?;
            if h.nrows() != self.p || h.ncols() != self.p {
                crate::bail_invalid_estim!(
                    "sparse exact objective inner Hessian shape {}x{} != {}x{}",
                    h.nrows(),
                    h.ncols(),
                    self.p,
                    self.p
                );
            }
            return Ok(h);
        }
        Ok(bundle.h_total.as_ref().clone())
    }

    pub(crate) fn previous_outer_gradient_norm(
        &self,
        current_key: &Option<Vec<u64>>,
    ) -> Option<f64> {
        let guard = self.cache_manager.current_outer_eval.read().unwrap();
        let (cached_key, eval) = guard.as_ref()?;
        if current_key
            .as_ref()
            .is_some_and(|current_key| current_key == cached_key)
        {
            return None;
        }
        let norm = eval.gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
        (norm.is_finite() && norm >= 0.0).then_some(norm)
    }

    pub(crate) fn active_constraint_free_basis(&self, pr: &PirlsResult) -> Option<Array2<f64>> {
        let lin = pr.linear_constraints_transformed.as_ref()?;
        let beta_t = pr.beta_transformed.as_ref();
        let mut activerows: Vec<Array1<f64>> = Vec::new();
        for i in 0..lin.a.nrows() {
            let slack = lin.a.row(i).dot(beta_t) - lin.b[i];
            if slack <= ACTIVE_CONSTRAINT_SLACK_TOL {
                activerows.push(lin.a.row(i).to_owned());
            }
        }
        if activerows.is_empty() {
            return None;
        }

        let p_t = lin.a.ncols();
        let mut a_t = Array2::<f64>::zeros((p_t, activerows.len()));
        for (j, row) in activerows.iter().enumerate() {
            for k in 0..p_t {
                a_t[[k, j]] = row[k];
            }
        }

        let qrow = Self::orthonormalize_columns(&a_t, ORTHONORM_DROP_TOL); // basis for active row-space^T
        let rank = qrow.ncols();
        if rank == 0 {
            return None;
        }
        if rank >= p_t {
            return Some(Array2::<f64>::zeros((p_t, 0)));
        }

        // Build orthonormal basis for null(A_active) as complement of row-space.
        let mut z = Array2::<f64>::zeros((p_t, p_t - rank));
        let mut kept = 0usize;
        for j in 0..p_t {
            let mut v = Array1::<f64>::zeros(p_t);
            v[j] = 1.0;
            for t in 0..rank {
                let qt = qrow.column(t);
                let proj = qt.dot(&v);
                v -= &qt.mapv(|x| x * proj);
            }
            for t in 0..kept {
                let zt = z.column(t);
                let proj = zt.dot(&v);
                v -= &zt.mapv(|x| x * proj);
            }
            let nrm = v.dot(&v).sqrt();
            if nrm > ORTHONORM_DROP_TOL {
                z.column_mut(kept).assign(&v.mapv(|x| x / nrm));
                kept += 1;
                if kept == p_t - rank {
                    break;
                }
            }
        }
        Some(z.slice(ndarray::s![.., 0..kept]).to_owned())
    }

    /// Construct a `BarrierConfig` from linear inequality constraints `A β ≥ b`
    /// by extracting rows that represent simple coordinate bounds (β_j ≥ b_i).
    ///
    /// Delegates to `BarrierConfig::from_constraints` and logs a diagnostic
    /// barrier-curvature check at a test point near the bounds.
    pub(crate) fn barrier_config_from_constraints(
        constraints: &crate::pirls::LinearInequalityConstraints,
    ) -> Option<super::reml_outer_engine::BarrierConfig> {
        let config = super::reml_outer_engine::BarrierConfig::from_constraints(Some(constraints))?;
        // Diagnostic: check curvature significance at a test point near bounds.
        {
            // Place the diagnostic test point a small slack inside the feasible
            // side of each bound, and probe the barrier curvature at unit β
            // magnitude against a 5%-of-curvature significance threshold. These
            // only shape the emitted trace line, not the fit.
            const DIAGNOSTIC_BOUND_SLACK: f64 = 0.01;
            const DIAGNOSTIC_BETA_MAGNITUDE: f64 = 1.0;
            const DIAGNOSTIC_CURVATURE_REL_THRESHOLD: f64 = 0.05;
            let max_idx = config
                .constrained_indices
                .iter()
                .max()
                .copied()
                .unwrap_or(0);
            let mut beta_test = Array1::<f64>::zeros(max_idx + 1);
            for ((&idx, &rhs), &sign) in config
                .constrained_indices
                .iter()
                .zip(config.lower_bounds.iter())
                .zip(config.bound_signs.iter())
            {
                beta_test[idx] = (rhs + DIAGNOSTIC_BOUND_SLACK) / sign;
            }
            let significant = config.barrier_curvature_is_significant(
                &beta_test,
                DIAGNOSTIC_BETA_MAGNITUDE,
                DIAGNOSTIC_CURVATURE_REL_THRESHOLD,
            );
            log::trace!(
                "[barrier] curvature significant={significant} (tau={:.2e}, n_constrained={})",
                config.tau,
                config.constrained_indices.len(),
            );
        }
        Some(config)
    }

    pub(super) fn enforce_constraint_kkt(&self, pr: &PirlsResult) -> Result<(), EstimationError> {
        let Some(kkt) = pr.constraint_kkt.as_ref() else {
            return Ok(());
        };
        // On a genuinely degenerate boundary face (linearly-dependent active
        // rows), the active-row multipliers are non-unique and a strict 5e-6
        // stationarity check is unreachable by construction. The inner
        // active-set solver already certifies such iterates via its
        // `degenerate_boundary_ok` clause at the relaxed
        // `ACTIVE_SET_KKT_DEGENERATE_STATIONARITY_TOL` tolerance — without
        // matching that here, the outer startup gate would refuse a
        // legitimately converged constrained optimum. This relaxation is gated
        // strictly on `working_set_rank_deficient`; it does NOT fire for
        // `shape=concave`/`shape=convex`, whose active rows are independent
        // coordinate lower bounds `γ_j ≥ 0` (full rank). Those converge from a
        // strictly-interior cold seed (`project_point_strictly_into_feasible_cone`)
        // and are held to the strict tolerance — their cold-vs-warm cache
        // divergence (#873) was a seed problem, not a degeneracy. Primal / dual
        // / complementarity stay on their strict tolerances; only the
        // stationarity channel — the one mathematically unreachable on a
        // rank-deficient face — gets the matching relaxation.
        let stationarity_tol = if kkt.working_set_rank_deficient {
            crate::active_set::ACTIVE_SET_KKT_DEGENERATE_STATIONARITY_TOL
        } else {
            KKT_TOL_STAT
        };
        // Scale-invariant stationarity, matching the inner active-set solver's
        // own acceptance contract (`stationarity_rel` in
        // `solve_newton_direction_with_linear_constraints_impl`): the
        // stationarity residual `‖grad − Aᵀλ‖∞` is certified relative to the
        // gradient scale `‖grad‖∞`, not against a bare absolute floor. The
        // profiled-REML / least-squares gradient is O(n) in magnitude even at a
        // genuine constrained optimum (issue #879), so the residual bottoms out
        // at an absolute value (≈5.8e-5 on the n=400 #989 repro) that the fixed
        // `5e-6` gate can never meet — even though the inner solver already
        // converged on the relative ratio. We accept when EITHER the absolute
        // residual is below the gate OR the relative ratio
        // `stationarity / max(‖grad‖∞, 1)` is, so the outer gate stops on the
        // same point the solver does instead of spuriously aborting a reachable
        // constrained optimum (issue #989). `bounded()`, which solves via the
        // exact-interval path rather than this active-set gate, was unaffected —
        // hence the two documented ways to bound a coefficient disagreed.
        let stationarity_rel = kkt.stationarity / kkt.gradient_scale.max(1.0);
        let stationarity_ok =
            kkt.stationarity <= stationarity_tol || stationarity_rel <= stationarity_tol;
        if kkt.primal_feasibility > KKT_TOL_PRIMAL
            || kkt.dual_feasibility > KKT_TOL_DUAL
            || kkt.complementarity > KKT_TOL_COMP
            || !stationarity_ok
        {
            let mut worstrow_msg = String::new();
            if let Some(lin) = pr.linear_constraints_transformed.as_ref() {
                let mut worst = 0.0_f64;
                let mut worstrow = 0usize;
                for i in 0..lin.a.nrows() {
                    let slack = lin.a.row(i).dot(&pr.beta_transformed.0) - lin.b[i];
                    let viol = (-slack).max(0.0);
                    if viol > worst {
                        worst = viol;
                        worstrow = i;
                    }
                }
                if worst > 0.0 {
                    worstrow_msg = format!("; worstrow={} worstviolation={:.3e}", worstrow, worst);
                }
            }
            return Err(EstimationError::ParameterConstraintViolation(format!(
                "KKT residuals exceed tolerance: primal={:.3e}, dual={:.3e}, comp={:.3e}, stat={:.3e} (stat_rel={:.3e} vs tol={:.3e}{}; ‖grad‖∞={:.3e}); active={}/{}{}",
                kkt.primal_feasibility,
                kkt.dual_feasibility,
                kkt.complementarity,
                kkt.stationarity,
                stationarity_rel,
                stationarity_tol,
                if kkt.working_set_rank_deficient {
                    ", degenerate face"
                } else {
                    ""
                },
                kkt.gradient_scale,
                kkt.n_active,
                kkt.n_constraints,
                worstrow_msg
            )));
        }
        Ok(())
    }

    pub(super) fn projectwith_basis(matrix: &Array2<f64>, z: &Array2<f64>) -> Array2<f64> {
        let zt_m = gam_linalg::faer_ndarray::fast_atb(z, matrix);
        gam_linalg::faer_ndarray::fast_ab(&zt_m, z)
    }

    pub(super) fn compute_penalty_subspace(
        &self,
        e_transformed: &Array2<f64>,
        ridge_passport: RidgePassport,
    ) -> Result<PenaltySubspace, EstimationError> {
        let p = e_transformed.ncols();
        if e_transformed.nrows() == 0 || p == 0 {
            return Ok(PenaltySubspace {
                evals: Array1::zeros(p),
                rank: 0,
            });
        }
        let cached =
            self.cache_manager
                .cached_penalty_subspace(e_transformed, &ridge_passport, || {
                    let mut s_lambda = e_transformed.t().dot(e_transformed);
                    let ridge = ridge_passport.penalty_logdet_ridge();
                    if ridge > 0.0 {
                        for i in 0..p {
                            s_lambda[[i, i]] += ridge;
                        }
                    }
                    let (evals, _) = s_lambda
                        .eigh(Side::Lower)
                        .map_err(EstimationError::EigendecompositionFailed)?;
                    let rank = if self.canonical_penalties.is_empty() {
                        positive_penalty_rank_and_logdet(evals.as_slice().unwrap()).0
                    } else {
                        self.canonical_penalties
                            .iter()
                            .map(gam_terms::construction::CanonicalPenalty::rank)
                            .sum::<usize>()
                            .min(p)
                    };
                    Ok(PenaltySubspace { evals, rank })
                })?;
        Ok(PenaltySubspace {
            evals: cached.evals.clone(),
            rank: cached.rank,
        })
    }

    pub(crate) fn fixed_subspace_penalty_rank_and_logdet_from_subspace(
        &self,
        penalty_subspace: &PenaltySubspace,
    ) -> (usize, f64) {
        if penalty_subspace.rank == 0 || penalty_subspace.evals.is_empty() {
            return (0, 0.0);
        }

        if self.canonical_penalties.is_empty() {
            return positive_penalty_rank_and_logdet(penalty_subspace.evals.as_slice().unwrap());
        }

        // eigh returns eigenvalues sorted ascending; the structural null
        // directions live at the bottom of the spectrum. Sum the top
        // `structural_rank` log-eigenvalues for log|S|+.
        let p = penalty_subspace.evals.len();
        let evals_slice = penalty_subspace.evals.as_slice().unwrap();
        let log_det: f64 = evals_slice
            .iter()
            .skip(p.saturating_sub(penalty_subspace.rank))
            .filter_map(|&ev| if ev > 0.0 { Some(ev.ln()) } else { None })
            .sum();

        (penalty_subspace.rank, log_det)
    }

    /// Intrinsic pseudo-logdet of the penalized Hessian `H_pen = X'WX + Sλ
    /// (− H_φ)` over its own identified subspace `range(H_pen)`, together with
    /// the matching trace kernel `H_pen⁺` in spectral form: `u_s` holds the
    /// kept eigenvectors `U_H` (σ above the pseudo-determinant threshold) and
    /// `h_proj_inverse = diag(1/σ_a)`, so the [`PenaltySubspaceTrace`]
    /// identities reproduce exactly `tr(H_pen⁺ · A)` for ANY drift `A`.
    ///
    /// ## Why the identified quotient, not range(Sλ) (#901)
    ///
    /// The previous realization projected onto `range(Sλ)`: value
    /// `log|U_Sᵀ H U_S|₊` with kernel `(U_Sᵀ H U_S)⁻¹`. That is the wrong
    /// object. Split `H` over `range(S) ⊕ ker(S)` as `[[A, B], [Bᵀ, C]]`:
    /// the projected logdet is `log det A`, while the determinant on the
    /// identified quotient is `log det A + log det(C − BᵀA⁻¹B)`. The dropped
    /// Schur term is the curvature of likelihood-identified but penalty-null
    /// directions (the unpenalized trend/intercept block), and for non-Gaussian
    /// families it is θ-dependent through the GLM weights `W(β̂(θ))` — so both
    /// the projected VALUE and every ρ/ψ trace built on the projected kernel
    /// were genuinely wrong, not numerically noisy (sign-flipped ρ-gradients
    /// and ~1e5 κ-gradient blow-ups in the iso-κ Duchon FD drivers). Gaussian
    /// identity passes only because `c ≡ 0` keeps this path uninstalled.
    ///
    /// The correct LAML determinant term (mgcv's generalized determinant, and
    /// the dual of the #752 fix which moved the BMS joint path to
    /// `range(H+Sλ)`) is the intrinsic pseudo-logdet
    ///
    /// ```text
    ///   ½ log|H_pen|₊ = ½ Σ_{σ_a > thr} log σ_a ,
    /// ```
    ///
    /// dropping only the truly unidentified directions `ker(H_pen) =
    /// ker(X'WX) ∩ ker(Sλ)` — exactly the directions `½ log|Sλ|₊` also omits,
    /// keeping the LAML ratio consistent.
    ///
    /// ## Why one spectral kernel serves ρ AND ψ (moving subspaces included)
    ///
    /// On a constant-rank stratum the first derivative of a pseudo-logdet
    /// along ANY symmetric drift `Ḣ` is
    ///
    /// ```text
    ///   d log|H|₊ [Ḣ] = tr(H⁺ · Ḣ)
    /// ```
    ///
    /// — first-order eigenvector motion cancels (`u̇_aᵀ H u_a + u_aᵀ H u̇_a =
    /// 2 σ_a u_aᵀ u̇_a = 0` by normalization), so there is NO moving-subspace
    /// correction term to track. This is what kills the ψ = log κ bug class:
    /// `range(Sλ(ψ))` rotates with ψ and the old fixed-`U_S` kernel dropped
    /// the `dU_S/dψ` term that finite differences capture; `range(H_pen)` as
    /// the intrinsic spectral object needs no external basis bookkeeping at
    /// all. Likewise the GLM IFT correction `D_β H[v] = X' diag(c ⊙ X v) X`,
    /// which leaks onto `null(Sλ)` (intercept column), is now traced against
    /// the SAME object whose logdet the cost reports — value and gradient are
    /// one spectral decomposition and cannot drift apart (the structural cure
    /// for the objective↔gradient desync class, #752/#748/#808).
    ///
    /// Rank-change points (an eigenvalue crossing the threshold) are genuine
    /// non-differentiability points of the criterion; the domain is a union
    /// of constant-rank strata and FD probes are only meaningful within one.
    pub(super) fn intrinsic_hessian_pseudo_logdet_parts(
        h_total: &Array2<f64>,
    ) -> Result<(f64, Option<super::reml_outer_engine::PenaltySubspaceTrace>), EstimationError>
    {
        let p = h_total.ncols();
        if p == 0 {
            return Ok((0.0, None));
        }
        if h_total.nrows() != p {
            crate::bail_invalid_estim!(
                "intrinsic_hessian_pseudo_logdet_parts: H must be square, got {}x{}",
                h_total.nrows(),
                p
            );
        }

        // Symmetrize before eigh: H_pen is symmetric in exact arithmetic and
        // faer rejects visibly asymmetric input.
        let mut h_sym = h_total.clone();
        gam_linalg::matrix::symmetrize_in_place(&mut h_sym);
        let (h_evals, h_evecs) = h_sym
            .eigh(Side::Lower)
            .map_err(EstimationError::EigendecompositionFailed)?;
        let h_thr =
            super::reml_outer_engine::positive_eigenvalue_threshold(h_evals.as_slice().unwrap());
        let kept: Vec<usize> = (0..p).filter(|&j| h_evals[j] > h_thr).collect();
        if kept.is_empty() {
            // No positive curvature anywhere: nothing identified, nothing to
            // correct — mirrors the structurally-null-penalty contract.
            return Ok((0.0, None));
        }
        // Rank-guarded full log|H| (#1426 part A). When EVERY eigenvalue clears
        // the (relative, eigensolver-noise-calibrated) threshold, H is full
        // rank: there is no genuinely-null direction, so the pseudo-logdet is
        // *identically* the full logdet `Σ_j ln μ_j`. Computing it as the full
        // sum over all p eigenvalues here makes that equivalence explicit and
        // guarantees the LAML determinant pair `½(log|H| − log|S|₊)` never
        // orphans a small-but-real H direction that the penalty side keeps —
        // the #1426 Occam-pair-inversion hazard. This is behaviourally identical
        // to `exact_pseudo_logdet(.., h_thr)` whenever `kept.len() == p` (the
        // filtered sum already spans every eigenvalue), so it cannot perturb any
        // existing FD cert; it differs from the pseudo path ONLY in the
        // genuinely rank-deficient case, where `kept.len() < p` and we fall back
        // to the exact pseudo-logdet that #901 installed to tame the
        // (p − rank)·ln ε divergence over true-null directions.
        let log_det = if kept.len() == p {
            h_evals.iter().map(|&s| s.ln()).sum()
        } else {
            super::reml_outer_engine::exact_pseudo_logdet(h_evals.as_slice().unwrap(), h_thr)
        };

        // Spectral form of H_pen⁺: U_H (p × r) and diag(1/σ). In this basis
        // `h_proj_inverse = (U_Hᵀ H U_H)⁻¹ = diag(1/σ)` EXACTLY, so the two
        // historical readings of the kernel ("projected inverse" vs "range
        // block of the full pseudo-inverse") coincide and every
        // `PenaltySubspaceTrace` consumer — drift traces, the IFT
        // `bilinear_pseudo_inverse` cost correction, KKT-residual reduction,
        // projected leverages — operates on the one true `H_pen⁺`.
        let r = kept.len();
        let mut u_s = Array2::<f64>::zeros((p, r));
        let mut h_proj_inverse = Array2::<f64>::zeros((r, r));
        for (out_col, &src_col) in kept.iter().enumerate() {
            for row in 0..p {
                u_s[[row, out_col]] = h_evecs[[row, src_col]];
            }
            h_proj_inverse[[out_col, out_col]] = 1.0 / h_evals[src_col];
        }

        Ok((
            log_det,
            Some(super::reml_outer_engine::PenaltySubspaceTrace {
                u_s,
                h_proj_inverse,
            }),
        ))
    }

    pub(super) fn updatewarm_start_from(&self, pr: &PirlsResult) {
        if !self.warm_start_enabled.load(Ordering::Relaxed) {
            return;
        }
        match pr.status {
            pirls::PirlsStatus::Converged | pirls::PirlsStatus::StalledAtValidMinimum => {
                self.update_flat_glm_first_step_gram_from(pr);
                let frame_was_original = matches!(
                    pr.coordinate_frame,
                    pirls::PirlsCoordinateFrame::OriginalSparseNative
                );
                let beta_original = match pr.coordinate_frame {
                    pirls::PirlsCoordinateFrame::OriginalSparseNative => {
                        pr.beta_transformed.as_ref().clone()
                    }
                    pirls::PirlsCoordinateFrame::TransformedQs => {
                        pr.reparam_result.qs.dot(pr.beta_transformed.as_ref())
                    }
                };
                // Shift the (ρ, β) history one step before storing the new
                // pair, so the previous-pair slot always holds the
                // pre-most-recent solve. Used by
                // `predict_warm_start_beta_with_source` for tangent-line
                // extrapolation across outer iterations.
                {
                    let mut prev_beta_w = self.prev_warm_start_beta.write().unwrap();
                    let mut prev_rho_w = self.prev_warm_start_rho.write().unwrap();
                    let mut cur_beta_w = self.warm_start_beta.write().unwrap();
                    let mut cur_rho_w = self.warm_start_rho.write().unwrap();
                    *prev_beta_w = cur_beta_w.take();
                    *prev_rho_w = cur_rho_w.take();
                    cur_beta_w.replace(Coefficients::new(beta_original.clone()));
                }
                // IFT warm-start cache: stash β / H_pen / qs from this
                // solve so the next outer iter's predictor can apply
                // `dβ/dρ_k = -H^{-1}(e^{ρ_k} S_k(β-μ_k))` directly. ρ is
                // populated by `record_warm_start_rho` immediately
                // after this call returns; until then the slot holds
                // an empty `rho` placeholder that the predictor
                // detects and skips.
                // Precompute the per-penalty `S_k · (β_cur-μ_k)` block-local
                // mat-vecs once at cache-write time so the IFT
                // predictor's per-call rhs construction skips the
                // `O(block²)` mat-vec on every penalty. At large-scale
                // CTN this saves a few ms per predict call; combined
                // with the H_pen factor cache (commit ec18559d) the
                // predictor's per-call work is now dominated by the
                // back-solve plus the `O(p)` rhs accumulation.
                let lambda_s_beta_blocks: Option<Vec<ndarray::Array1<f64>>> = {
                    // Parallelize across penalties: each `S_k · (β_cur-μ_k)`
                    // mat-vec is independent of the others. At large-scale-
                    // scale CTN with p ≈ several thousand and ~10
                    // penalties, the serial precompute is ~250M flops
                    // per cache write — repeated 5-10× per fit as the
                    // outer optimizer accepts new ρ. Parallelizing
                    // across rayon's thread pool brings this down to
                    // (~250M / cores) flops per write, eliminating
                    // the precompute as a meaningful large-scale
                    // serial cost.
                    use rayon::prelude::*;
                    let blocks: Vec<ndarray::Array1<f64>> = self
                        .canonical_penalties
                        .par_iter()
                        .map(|cp| {
                            let r = &cp.col_range;
                            let beta_block = beta_original.slice(s![r.start..r.end]);
                            let centered = &beta_block - &cp.prior_mean;
                            cp.local.dot(&centered)
                        })
                        .collect();
                    if blocks.is_empty() {
                        None
                    } else {
                        Some(blocks)
                    }
                };
                {
                    let mut cache_w = self.ift_warm_start_cache.write().unwrap();
                    cache_w.replace(super::IftWarmStartCache {
                        beta_original,
                        rho: ndarray::Array1::zeros(0),
                        penalized_hessian_transformed: pr.penalized_hessian_transformed.clone(),
                        qs: pr.reparam_result.qs.clone(),
                        frame_was_original,
                        lambda_s_beta_blocks,
                    });
                }
                // The factor cache holds the Cholesky of the PREVIOUS
                // H_pen; replacing the IFT cache invalidates it. Drop
                // here so the next predict call lazily refactors the
                // new H_pen and stashes the new factor.
                self.ift_cached_factor.write().unwrap().take();
                self.clear_ift_mode_response_cache();
            }
            _ => {
                // On a failed solve, drop both the current pair AND the
                // history — the tangent prediction would be misleading.
                // Adaptive signals are wiped by `execute_pirls_if_needed`'s
                // failure branch (where `pirls_result.iteration` is
                // available for the schedule's geometric backoff
                // accounting), so we only clear predictor state here.
                self.clear_warm_start_predictor_state();
            }
        }
    }

    fn update_flat_glm_first_step_gram_from(&self, pr: &PirlsResult) {
        if self.config.firth_bias_reduction
            || matches!(
                reml_spec(&self.config.likelihood).response,
                ResponseFamily::Gaussian
            )
            || pr.cache_compacted
            || pr.reparam_result.s_transformed.nrows() != pr.penalized_hessian_transformed.nrows()
            || pr.reparam_result.s_transformed.ncols() != pr.penalized_hessian_transformed.ncols()
        {
            *self.flat_glm_first_step_gram.write().unwrap() = None;
            return;
        }

        let mut gram_transformed = pr.penalized_hessian_transformed.to_dense();
        gram_transformed -= &pr.reparam_result.s_transformed;
        gam_linalg::matrix::symmetrize_in_place(&mut gram_transformed);

        let mut gram_original = match pr.coordinate_frame {
            pirls::PirlsCoordinateFrame::OriginalSparseNative => gram_transformed,
            pirls::PirlsCoordinateFrame::TransformedQs => {
                let left = gam_linalg::faer_ndarray::fast_ab(&pr.reparam_result.qs, &gram_transformed);
                gam_linalg::faer_ndarray::fast_ab(&left, &pr.reparam_result.qs.t().to_owned())
            }
        };
        gam_linalg::matrix::symmetrize_in_place(&mut gram_original);
        if gram_original.nrows() == self.p
            && gram_original.ncols() == self.p
            && gram_original.iter().all(|value| value.is_finite())
        {
            *self.flat_glm_first_step_gram.write().unwrap() = Some(Arc::new(gram_original));
        } else {
            *self.flat_glm_first_step_gram.write().unwrap() = None;
        }
    }

    /// Record the ρ at which the most recent successful warm-start β was
    /// obtained. Called immediately after `updatewarm_start_from` succeeds
    /// (the caller knows the ρ that produced the inner solve).
    pub(super) fn record_warm_start_rho(&self, rho: &Array1<f64>) {
        if !self.warm_start_enabled.load(Ordering::Relaxed) {
            return;
        }
        self.warm_start_rho.write().unwrap().replace(rho.to_owned());
        // Stamp the IFT cache's ρ slot so the predictor can compute
        // Δρ = ρ_new − ρ_cache. Skipped if the slot was cleared in
        // between (e.g. a concurrent failure path); the predictor will
        // see an empty `rho` (length 0) and fall back to the
        // tangent-line / flat warm-start path.
        if let Some(cache) = self.ift_warm_start_cache.write().unwrap().as_mut() {
            cache.rho = rho.to_owned();
        }
    }

    /// Outer-loop [`gam_runtime::warm_start::Session`] for this fit, derived from the
    /// same realized-fit-context key as the inner beta record. Disjoint
    /// keyspace, so inner and outer payloads don't collide.
    ///
    /// Returns `None` if no platform cache directory is discoverable.
    pub(crate) fn outer_cache_session(
        &self,
    ) -> Option<std::sync::Arc<gam_runtime::warm_start::Session>> {
        if !self.warm_start_enabled.load(Ordering::Relaxed) {
            return None;
        }
        // Opt-in only: opening the cross-process outer-iterate session also
        // opens the shared on-disk `WarmStartStore` (dir/eviction scan). Skip
        // it unless `FitConfig::persist_warm_start_disk` enabled disk
        // persistence (#1082/#1114).
        if !self
            .persistent_warm_start_disk_enabled
            .load(Ordering::Relaxed)
        {
            return None;
        }
        let key = self.persistent_warm_start_cache_key()?;
        crate::persistent_warm_start::open_outer_session(&key)
    }

    /// Engage the cross-process ON-DISK warm-start layer. Called from the
    /// estimate constructor when `FitConfig::persist_warm_start_disk` requests
    /// cross-process / repeat-fit persistence. Until this is set the disk
    /// load/store/eviction-scan path is skipped entirely and only the
    /// in-memory warm start is used.
    pub(crate) fn enable_persistent_warm_start_disk(&self) {
        self.persistent_warm_start_disk_enabled
            .store(true, Ordering::Relaxed);
    }

    pub(crate) fn persistent_warm_start_cache_key(&self) -> Option<String> {
        if let Some(key) = self.persistent_warm_start_key.read().unwrap().clone() {
            return Some(key);
        }
        let mut hasher = Fingerprinter::new();
        hasher.write_str("gamfit-persistent-warm-start-v2");
        // Use the cache schema tag (NOT CARGO_PKG_VERSION) so routine
        // library version bumps don't invalidate users' on-disk
        // warm-start caches.
        hasher.write_str(&crate::persistent_warm_start::cache_schema_tag());
        hasher.write_usize(self.y.len());
        hasher.write_usize(self.p);
        hasher.write_str(&format!("{:?}", self.config.likelihood));
        hasher.write_str(&format!("{:?}", self.config.link_kind));
        hasher.write_f64(self.config.pirls_convergence_tolerance);
        hasher.write_f64(self.config.reml_convergence_tolerance);
        hasher.write_usize(self.config.max_iterations);
        hasher.write_bool(self.config.firth_bias_reduction);
        hasher.write_str(&format!("{:?}", self.runtime_mixture_link_state));
        hasher.write_str(&format!("{:?}", self.runtime_sas_link_state));
        match self.penalty_shrinkage_floor {
            Some(value) => {
                hasher.write_bool(true);
                hasher.write_f64(value);
            }
            None => hasher.write_bool(false),
        }
        hasher.write_str(&format!("{:?}", self.rho_prior));

        hash_array_view(&mut hasher, self.y);
        hash_array_view(&mut hasher, self.weights);
        hash_array_view(&mut hasher, self.offset.view());
        if hash_design_matrix(&mut hasher, &self.x).is_err() {
            return None;
        }
        hash_canonical_penalties(&mut hasher, self.canonical_penalties.as_ref());
        // Analytic penalties alter the realized fit even when X and canonical S match.
        hasher.write_u64(self.analytic_penalty_registry_fingerprint);
        hasher.write_usize(self.nullspace_dims.len());
        for &dim in &self.nullspace_dims {
            hasher.write_usize(dim);
        }
        match self.coefficient_lower_bounds.as_ref() {
            Some(bounds) => {
                hasher.write_bool(true);
                hash_array_view(&mut hasher, bounds.view());
            }
            None => hasher.write_bool(false),
        }
        match self.linear_constraints.as_ref() {
            Some(constraints) => {
                hasher.write_bool(true);
                hash_array2(&mut hasher, &constraints.a);
                hash_array_view(&mut hasher, constraints.b.view());
            }
            None => hasher.write_bool(false),
        }
        hasher.write_bool(self.kronecker_penalty_system.is_some());
        hasher.write_bool(self.kronecker_factored.is_some());
        let key = hasher.finish_hex();
        self.persistent_warm_start_key
            .write()
            .unwrap()
            .replace(key.clone());
        Some(key)
    }

    pub(crate) fn persistent_latent_values_cache_key(&self) -> Option<String> {
        let latent_fingerprint = self.persistent_latent_values_fingerprint?;
        self.persistent_warm_start_cache_key()
            .map(|key| format!("persistent-latent-values-v2:{key}:{latent_fingerprint:016x}"))
    }

    pub(crate) fn load_persistent_latent_values(
        &self,
        n_obs: usize,
        latent_dim: usize,
    ) -> Option<Array2<f64>> {
        if !self.warm_start_enabled.load(Ordering::Relaxed) {
            return None;
        }
        let key = self.persistent_latent_values_cache_key()?;
        self.persistent_latent_values_cache
            .write()
            .unwrap()
            .lookup(&key, n_obs, latent_dim)
    }

    pub(crate) fn store_persistent_latent_values(&self, values: &Array2<f64>) {
        if !self.warm_start_enabled.load(Ordering::Relaxed) {
            return;
        }
        let Some(key) = self.persistent_latent_values_cache_key() else {
            return;
        };
        self.persistent_latent_values_cache
            .write()
            .unwrap()
            .insert(key, values.clone());
    }

    pub(crate) fn load_persistent_warm_start_once(&self) {
        if !self.warm_start_enabled.load(Ordering::Relaxed) {
            return;
        }
        // Cross-process disk restore is opt-in. Without it, skip the
        // `persistent_store()` open + dir/eviction scan entirely — the
        // in-memory warm start fully serves an in-process fit (#1082/#1114).
        if !self
            .persistent_warm_start_disk_enabled
            .load(Ordering::Relaxed)
        {
            return;
        }
        if self
            .persistent_warm_start_loaded
            .swap(true, Ordering::Relaxed)
        {
            return;
        }
        if self.warm_start_beta.read().unwrap().is_some() {
            return;
        }
        let Some(key) = self.persistent_warm_start_cache_key() else {
            return;
        };
        let Some(record) = load_record(&key) else {
            return;
        };
        if !record.is_compatible(&key, self.y.len(), self.p) {
            return;
        }
        let rho_len = record.rho.len();
        if rho_len != self.canonical_penalties.len() {
            return;
        }
        {
            self.warm_start_beta
                .write()
                .unwrap()
                .replace(Coefficients::new(Array1::from_vec(record.beta)));
            self.warm_start_rho
                .write()
                .unwrap()
                .replace(Array1::from_vec(record.rho));
            *self.prev_warm_start_beta.write().unwrap() = record
                .prev_beta
                .map(|beta| Coefficients::new(Array1::from_vec(beta)));
            *self.prev_warm_start_rho.write().unwrap() = record.prev_rho.map(Array1::from_vec);
        }
        self.last_inner_iters
            .store(record.last_inner_iters, Ordering::Relaxed);
        self.last_inner_converged
            .store(record.last_inner_converged, Ordering::Relaxed);
        self.last_pirls_lm_lambda.store(
            record
                .last_pirls_lm_lambda
                .filter(|v| v.is_finite() && *v > 0.0)
                .map(f64::to_bits)
                .unwrap_or(0),
            Ordering::Relaxed,
        );
        self.last_ift_prediction_residual.store(
            finite_nonnegative_bits_or_no_signal(record.last_ift_prediction_residual),
            Ordering::Relaxed,
        );
        self.last_pirls_accept_rho.store(
            finite_nonnegative_bits_or_no_signal(record.last_pirls_accept_rho),
            Ordering::Relaxed,
        );
        log::info!("[warm-start-cache] restored persistent warm start key={key}");
    }

    pub(crate) fn store_persistent_warm_start(&self) {
        if !self.warm_start_enabled.load(Ordering::Relaxed) {
            return;
        }
        // Cross-process disk checkpoint is opt-in (a cache session was
        // attached). Without it, never touch the shared `WarmStartStore`:
        // `store_record` opens the store and pays an eviction/dir scan that is
        // O(cache entries) on a network FS, and a throwaway in-process fit
        // (CI-coverage replicate, posterior probe) writes a record nothing will
        // ever read — pure overhead that, accumulated across a refit loop,
        // dominated the profile (#1082/#1114).
        if !self
            .persistent_warm_start_disk_enabled
            .load(Ordering::Relaxed)
        {
            return;
        }
        if self
            .persistent_warm_start_store_suppression
            .load(Ordering::Relaxed)
            > 0
        {
            return;
        }
        // Disk persistence is a process-recovery checkpoint, not part of the
        // REML objective. The in-memory warm start is updated on every
        // successful PIRLS solve above; writing JSON/bin records here on every
        // outer trial puts filesystem eviction scans directly in the optimizer
        // hot loop. Checkpoint sparsely so long fits remain recoverable while
        // ordinary fits and posterior probes stay CPU-bound.
        let eval_count = *self.arena.cost_eval_count.read().unwrap();
        if eval_count % 1024 != 0 {
            return;
        }
        let Some(key) = self.persistent_warm_start_cache_key() else {
            return;
        };
        let Some(beta) = self.warm_start_beta.read().unwrap().as_ref().cloned() else {
            return;
        };
        let rho_opt: Option<Array1<f64>> = self.warm_start_rho.read().unwrap().clone();
        let Some(rho) = rho_opt else {
            return;
        };
        if beta.0.len() != self.p || rho.len() != self.canonical_penalties.len() {
            return;
        }
        let mut record = PersistentWarmStartRecord::new(key, self.y.len(), self.p);
        record.updated_unix_secs = record.created_unix_secs;
        record.rho = rho.to_vec();
        record.beta = beta.0.to_vec();
        record.prev_rho = self
            .prev_warm_start_rho
            .read()
            .unwrap()
            .as_ref()
            .map(|rho: &Array1<f64>| rho.to_vec());
        record.prev_beta = self
            .prev_warm_start_beta
            .read()
            .unwrap()
            .as_ref()
            .map(|coefficients| coefficients.0.to_vec());
        record.last_inner_iters = self.last_inner_iters.load(Ordering::Relaxed);
        record.last_inner_converged = self.last_inner_converged.load(Ordering::Relaxed);
        record.last_pirls_lm_lambda =
            finite_positive_from_bits(self.last_pirls_lm_lambda.load(Ordering::Relaxed));
        record.last_ift_prediction_residual =
            finite_nonnegative_from_bits(self.last_ift_prediction_residual.load(Ordering::Relaxed));
        record.last_pirls_accept_rho =
            finite_nonnegative_from_bits(self.last_pirls_accept_rho.load(Ordering::Relaxed));
        if let Err(err) = store_record(&record) {
            log::warn!("[warm-start-cache] failed to persist warm start: {err}");
        }
    }

    pub(crate) fn without_persistent_warm_start_store<T>(&self, f: impl FnOnce() -> T) -> T {
        struct StoreSuppressionGuard<'a>(&'a AtomicUsize);
        impl Drop for StoreSuppressionGuard<'_> {
            fn drop(&mut self) {
                self.0.fetch_sub(1, Ordering::Relaxed);
            }
        }

        self.persistent_warm_start_store_suppression
            .fetch_add(1, Ordering::Relaxed);
        let guard = StoreSuppressionGuard(&self.persistent_warm_start_store_suppression);
        let out = f();
        drop(guard);
        out
    }

    /// Run `f` with the Gaussian-identity ALO-stabilization augmentation
    /// disabled (#979). Used to evaluate the genuine LAML criterion during
    /// ρ-posterior certificate / NUTS sampling, where the optimizer-stability
    /// leverage barrier (#813/#821) is both inappropriate (it is not part of the
    /// marginal posterior, whose Laplace proposal uses the base REML Hessian)
    /// and ruinously expensive (its full ALO diagnostic suite would run on every
    /// leapfrog step). Re-entrant via a counter, like
    /// [`Self::without_persistent_warm_start_store`].
    pub(crate) fn without_alo_stabilization<T>(&self, f: impl FnOnce() -> T) -> T {
        struct AloSuppressionGuard<'a>(&'a AtomicUsize);
        impl Drop for AloSuppressionGuard<'_> {
            fn drop(&mut self) {
                self.0.fetch_sub(1, Ordering::Relaxed);
            }
        }

        self.alo_stabilization_suppression
            .fetch_add(1, Ordering::Relaxed);
        let guard = AloSuppressionGuard(&self.alo_stabilization_suppression);
        let out = f();
        drop(guard);
        out
    }

    /// Predict β at `new_rho` via the implicit-function-theorem first-order
    /// expansion
    /// `β_predict = β_cur − Σ_k Δρ_k · H_pen^{-1} · (e^{ρ_cur_k} S_k(β_cur-μ_k))`,
    /// surfacing the outcome (Predicted vs Noop) so callers can map directly
    /// onto `WarmStartPredictionSource` without re-deriving noop-ness via an
    /// O(p) array comparison against the cached β.
    ///
    /// Returns `None` (caller falls back to tangent-line / flat warm-start) when:
    /// * the IFT cache is empty (no prior converged solve, or one was invalidated),
    /// * ρ has been stamped yet (length 0 — see `record_warm_start_rho`),
    /// * Δρ is too aggressive for the linearized predictor to trust
    ///   (max |Δρ_k| > 2.0 — i.e. a single penalty has moved by more than e²
    ///   in λ-space, well outside the regime where the local linear Jacobian
    ///   is descriptive),
    /// * factorization or back-solve fails / produces non-finite output.
    ///
    /// The factor is cached at `self.ift_cached_factor` and reused across
    /// successive predict calls until the IFT cache itself is replaced (a
    /// new `updatewarm_start_from`) or invalidated (failed solve, link
    /// change, surface reset). At large scale (p ≈ several thousand)
    /// the dense Cholesky is O(p³)/3 — multiple seconds per refactor —
    /// so caching saves real wall time across the typical 5-10 IFT
    /// predict calls per outer fit.
    pub(crate) fn predict_warm_start_beta_ift_with_outcome(
        &self,
        new_rho: &Array1<f64>,
    ) -> Option<(Coefficients, IftPredictionOutcome)> {
        if !self.warm_start_enabled.load(Ordering::Relaxed) {
            return None;
        }
        // The NaN sentinel + the is_finite() check together cover three
        // cases in one expression: "no signal yet" (sentinel decodes to
        // NaN, which fails is_finite), "corrupted state" (any non-finite
        // or negative residual stored by mistake), and "real signal"
        // (finite non-negative residual → Some).
        let last_residual_bits = self.last_ift_prediction_residual.load(Ordering::Relaxed);
        let r = f64::from_bits(last_residual_bits);
        let last_residual = if r.is_finite() && r >= 0.0 {
            Some(r)
        } else {
            None
        };
        let current_ift_step_cap = self.ift_quality_step_cap(adaptive_ift_max_drho(last_residual));
        if self.joint_ift_cache_matches_pending_theta(new_rho) {
            return self
                .predict_warm_start_beta_joint_ift_with_outcome(new_rho, current_ift_step_cap);
        }
        let cache_guard = self.ift_warm_start_cache.read().unwrap();
        let cache = cache_guard.as_ref()?;
        // Early short-circuit: detect both the no-op case (every
        // |Δρ_k| below the numerical-noise floor → predictor reduces
        // to identity) AND the large-Δρ rejection case (|Δρ| exceeds
        // the adaptive cap → predictor would reject) BEFORE acquiring
        // the H_pen factor. The inner function detects both cases and
        // returns the same outcome, but only AFTER the factor cache
        // lookup — which on a miss pays a fresh O(p³)/3 Cholesky
        // (multiple seconds at large-scale p) for a prediction
        // that's about to be discarded.
        //
        // The dim-check guards mirror the inner function's so an
        // ambiguous case (rho-not-stamped, dim mismatch) falls through
        // and lets the inner function emit its precise rejection
        // marker, preserving the single-source-of-truth contract.
        if !cache.rho.is_empty() && cache.rho.len() == new_rho.len() {
            let mut max_abs_drho = 0.0_f64;
            let mut any_non_finite = false;
            for i in 0..cache.rho.len() {
                let d = new_rho[i] - cache.rho[i];
                if !d.is_finite() {
                    any_non_finite = true;
                    max_abs_drho = f64::INFINITY;
                    break;
                }
                if d.abs() > max_abs_drho {
                    max_abs_drho = d.abs();
                }
            }
            // No-op case: every |Δρ_k| below the eps floor.
            if !any_non_finite && max_abs_drho <= IFT_WARM_START_DRHO_EPS {
                // Same NOOP marker the inner function would emit, so the
                // bench runner aggregator's count is preserved across
                // both the early-out path and the post-factor path.
                log::info!(
                    "[IFT-NOOP] reason=all_drho_below_eps max_drho={:.3e} drho_dim={}",
                    max_abs_drho,
                    cache.rho.len(),
                );
                return Some((
                    Coefficients::new(cache.beta_original.clone()),
                    IftPredictionOutcome::Noop,
                ));
            }
            // Large-Δρ rejection: |Δρ| exceeds the adaptive cap.
            // `adaptive_ift_max_drho` reads the same `last_residual`
            // signal we already loaded above. Same marker as the inner
            // function emits so the rejection-rate aggregator
            // (`_IFT_REJECTED_PATTERN` in runner.py) is preserved
            // across both paths.
            let max_drho_cap = current_ift_step_cap;
            if !max_abs_drho.is_finite() || max_abs_drho > max_drho_cap {
                log::info!(
                    "[IFT-REJECTED] reason=large_drho max_drho={:.3e} cap={:.3e} drho_dim={}",
                    max_abs_drho,
                    max_drho_cap,
                    cache.rho.len(),
                );
                return None;
            }
        }
        if let Some(rho_mode_response_cols) = self.cached_ift_rho_mode_response_cols(cache) {
            if let Some(prediction) = predict_warm_start_beta_ift_from_mode_response_cols(
                cache,
                new_rho,
                self.p,
                last_residual,
                Some(current_ift_step_cap),
                &rho_mode_response_cols,
            ) {
                log::info!(
                    "[IFT-CACHE] outcome=mode_response_hit drho_dim={} p={}",
                    new_rho.len(),
                    self.p,
                );
                return Some(prediction);
            }
            log::debug!(
                "[IFT-CACHE] outcome=mode_response_fallback drho_dim={} p={}",
                new_rho.len(),
                self.p,
            );
        }
        // Get the cached factor or lazily compute it. Holding the
        // ift_warm_start_cache read lock while we acquire the factor's
        // lock is safe: cache invalidation (writer) takes the cache
        // write lock first, then the factor write lock, so a reader
        // holding the cache lock prevents the writer from advancing
        // and we never observe a factor that doesn't match the cached
        // matrix.
        let factor_arc: Arc<dyn gam_linalg::matrix::FactorizedSystem> = {
            let read_guard = self.ift_cached_factor.read().unwrap();
            if let Some(arc) = read_guard.as_ref() {
                // Cache hit: H_pen factor was already computed by an
                // earlier predict call at the same surface. The
                // [IFT-CACHE] marker validates that the factor cache
                // (commit ec18559d) is paying off at large scale —
                // every cache hit avoids a fresh O(p³)/3 Cholesky,
                // which is multiple seconds at p ≈ several thousand.
                log::info!(
                    "[IFT-CACHE] outcome=hit drho_dim={} p={}",
                    new_rho.len(),
                    self.p,
                );
                Arc::clone(arc)
            } else {
                // release-early-on-purpose: upgrade from read access to write access without deadlocking.
                drop(read_guard);
                let factorize_start = std::time::Instant::now();
                let new_factor = match cache.penalized_hessian_transformed.factorize() {
                    Ok(f) => f,
                    Err(_) => {
                        log::info!(
                            "[IFT-REJECTED] reason=hessian_factorize_failed_cached drho_dim={}",
                            new_rho.len(),
                        );
                        return None;
                    }
                };
                // Cache miss: paid the Cholesky once. Subsequent predict
                // calls at the same surface will hit the cache.
                log::info!(
                    "[IFT-CACHE] outcome=miss drho_dim={} p={} elapsed={:.3}s",
                    new_rho.len(),
                    self.p,
                    factorize_start.elapsed().as_secs_f64(),
                );
                let arc: Arc<dyn gam_linalg::matrix::FactorizedSystem> = Arc::from(new_factor);
                let mut write_guard = self.ift_cached_factor.write().unwrap();
                // Race window: another reader may have populated the
                // slot between our drop(read_guard) and the write lock
                // acquisition. Prefer the existing entry to keep the
                // single-factor invariant and avoid tearing.
                if let Some(existing) = write_guard.as_ref() {
                    Arc::clone(existing)
                } else {
                    *write_guard = Some(Arc::clone(&arc));
                    arc
                }
            }
        };
        predict_warm_start_beta_ift_inner_with_outcome(
            cache,
            self.canonical_penalties.as_ref(),
            new_rho,
            self.p,
            last_residual,
            Some(current_ift_step_cap),
            Some(factor_arc.as_ref()),
        )
    }

    /// Predict β at `new_rho` together with a tag identifying which
    /// predictor produced it (IFT first-order Jacobian or tangent-line
    /// extrapolation across the last two (ρ, β) pairs). Returns `None`
    /// when no predictor can be applied — the warm-start machinery is
    /// disabled, neither cache is populated, the ρ-step direction is
    /// degenerate, or the extrapolation step `α` exceeds the adaptive
    /// safety cap (default 1.5 — see `adaptive_tangent_alpha_cap` for
    /// the residual-driven policy that loosens to 2.0 when prior IFT
    /// predictions were excellent and tightens to 0.5 when the local
    /// linear approximation has been shown to collapse toward flat
    /// warm-start).
    ///
    /// Callers fall back to the stored `warm_start_beta` (`β(ρ_k)` —
    /// the standard "use last β as-is" warm start) on `None`. So this
    /// is strictly an improvement: when prediction is safe, we use it;
    /// otherwise we use the existing flat warm-start. The source tag
    /// drives per-predictor quality markers (the [IFT-QUALITY] /
    /// [TANGENT-QUALITY] chain in `execute_pirls_if_needed`) so each
    /// marker is attributed correctly to the predictor that actually
    /// produced the β.
    pub(crate) fn predict_warm_start_beta_with_source(
        &self,
        new_rho: &Array1<f64>,
    ) -> Option<(Coefficients, WarmStartPredictionSource)> {
        if !self.warm_start_enabled.load(Ordering::Relaxed) {
            return None;
        }
        if self.take_ift_quality_flat_override()
            && let Some(cur_beta) = self.warm_start_beta.read().unwrap().clone()
        {
            return Some((cur_beta, WarmStartPredictionSource::Flat));
        }
        // #1082 / #1033: fixed-design non-Gaussian outer trials can reuse the
        // previous converged data-fit Gram for the next first Fisher step only
        // when the seed is exactly the previous beta. Prefer that flat seed once
        // the Gram exists; IFT/tangent predictions would change W(eta) and force
        // the dense X'WX rebuild this cache is meant to retire.
        if !self.config.firth_bias_reduction
            && !matches!(
                reml_spec(&self.config.likelihood).response,
                ResponseFamily::Gaussian
            )
            && self.flat_glm_first_step_gram.read().unwrap().is_some()
            && let Some(cur_beta) = self.warm_start_beta.read().unwrap().clone()
        {
            return Some((cur_beta, WarmStartPredictionSource::Flat));
        }
        // Try the IFT-based predictor first. It uses the exact first-order
        // Jacobian of β(ρ) at the cached solve and beats the tangent-line
        // predictor whenever a converged H_pen is available — which is
        // every call after the first successful PIRLS solve.
        //
        // Source disambiguation: IFT returns Some(β_cur) on noop (all
        // Δρ_k below the per-component eps floor). The tangent-line
        // path makes the same distinction explicit by tagging Flat
        // (commit 1b77588b). The IFT inner predictor exposes the
        // outcome directly via `predict_warm_start_beta_ift_with_outcome`,
        // eliminating the O(p) array-comparison-against-cached-β that
        // the prior implementation needed to re-derive noop-ness here.
        // Map outcome → source enum without further work.
        if let Some((predicted, outcome)) = self.predict_warm_start_beta_ift_with_outcome(new_rho) {
            log::debug!("[warm-start] IFT prediction accepted");
            let source = match outcome {
                IftPredictionOutcome::Predicted => WarmStartPredictionSource::Ift,
                IftPredictionOutcome::Noop => WarmStartPredictionSource::Flat,
            };
            return Some((predicted, source));
        }
        let cur_beta = self.warm_start_beta.read().unwrap().clone()?;
        let cur_rho: Option<Array1<f64>> = self.warm_start_rho.read().unwrap().clone();
        let prev_beta: Option<Coefficients> = self.prev_warm_start_beta.read().unwrap().clone();
        let prev_rho: Option<Array1<f64>> = self.prev_warm_start_rho.read().unwrap().clone();
        let (cur_rho, prev_beta, prev_rho): (Array1<f64>, Coefficients, Array1<f64>) =
            match (cur_rho, prev_beta, prev_rho) {
                (Some(cr), Some(pb), Some(pr)) => (cr, pb, pr),
                // No history yet — first call after a fresh successful
                // solve (only one (ρ, β) pair stashed). Silent fallback
                // is the right behavior; emitting a marker here would
                // just be noise on every first call.
                _ => return Some((cur_beta, WarmStartPredictionSource::Flat)),
            };
        // Dimension mismatch paths below are bug signals: state-machine
        // inconsistency between (β, ρ) cache and current state. Emit
        // structured markers so the bench runner can detect non-zero
        // counts as a regression signal. The wipe helpers from commit
        // 6f7cbfc8 should have cleared these slots before the layout
        // shifted; non-zero count indicates a missed invalidation.
        if cur_rho.len() != new_rho.len() || cur_rho.len() != prev_rho.len() {
            log::info!(
                "[TANGENT-REJECTED] reason=rho_dim_mismatch new_rho_dim={} cur_rho_dim={} prev_rho_dim={}",
                new_rho.len(),
                cur_rho.len(),
                prev_rho.len(),
            );
            return Some((cur_beta, WarmStartPredictionSource::Flat));
        }
        if cur_beta.0.len() != prev_beta.0.len() {
            log::info!(
                "[TANGENT-REJECTED] reason=beta_dim_mismatch cur_beta_dim={} prev_beta_dim={}",
                cur_beta.0.len(),
                prev_beta.0.len(),
            );
            return Some((cur_beta, WarmStartPredictionSource::Flat));
        }
        // d_rho = ρ_k − ρ_{k-1}; step_rho = ρ_new − ρ_k.
        // Squared-norm floor (≈1e-12 in ‖Δρ‖) below which the previous ρ-step is
        // treated as a degenerate/zero-length direction the tangent predictor
        // cannot extrapolate along.
        const DEGENERATE_DRHO_NORM_SQ: f64 = 1e-24;
        let d_rho_norm_sq: f64 = cur_rho
            .iter()
            .zip(prev_rho.iter())
            .map(|(c, p)| (c - p) * (c - p))
            .sum();
        if !d_rho_norm_sq.is_finite() || d_rho_norm_sq <= DEGENERATE_DRHO_NORM_SQ {
            // Degenerate Δρ direction (the previous ρ-step had zero or
            // unfinite length). Diagnostic rather than bug: this fires
            // when the outer optimizer landed on a flat region or the
            // ρ history collapsed. Surfacing the case lets the bench
            // runner see flat-region traces.
            log::info!(
                "[TANGENT-REJECTED] reason=degenerate_drho d_rho_norm_sq={:.3e}",
                d_rho_norm_sq,
            );
            return Some((cur_beta, WarmStartPredictionSource::Flat));
        }
        let step_dot_d: f64 = new_rho
            .iter()
            .zip(cur_rho.iter())
            .zip(prev_rho.iter())
            .map(|((n, c), p)| (n - c) * (c - p))
            .sum();
        let alpha = step_dot_d / d_rho_norm_sq;
        if !alpha.is_finite() {
            // Non-finite α (NaN or Inf). Real bug signal — the
            // numerator or denominator overflowed. Should never happen
            // if d_rho_norm_sq passed the prior finiteness check.
            log::info!(
                "[TANGENT-REJECTED] reason=nonfinite_alpha step_dot_d={:.3e} d_rho_norm_sq={:.3e}",
                step_dot_d,
                d_rho_norm_sq,
            );
            return Some((cur_beta, WarmStartPredictionSource::Flat));
        }
        // Don't extrapolate against the previous step direction (α<0).
        // The upper cap is adaptive: the same IFT residual signal that
        // drives `adaptive_ift_max_drho` (commit 06888a1e) and the cap-
        // schedule margin (commit 4eb3686a) is the most direct proxy
        // for "how trustworthy is the local linear approximation" —
        // and the tangent-line predictor IS a local linear
        // approximation (just along the previous ρ-step direction
        // rather than the IFT Jacobian's full direction). When the
        // most recent IFT prediction was excellent, the tangent
        // approximation can be trusted for slightly larger α; when
        // it was poor, tighten.
        // Same NaN-sentinel discipline as the IFT predictor's reader
        // — see the comment there for the encoding rationale.
        let last_residual_bits = self.last_ift_prediction_residual.load(Ordering::Relaxed);
        let r = f64::from_bits(last_residual_bits);
        let last_residual = if r.is_finite() && r >= 0.0 {
            Some(r)
        } else {
            None
        };
        let alpha_cap = adaptive_tangent_alpha_cap(last_residual);
        if alpha <= 0.0 || alpha > alpha_cap {
            // Emit a structured reject marker so the bench runner can
            // count tangent-line rejections alongside IFT ones.
            // Tangent-line only fires when IFT returned None for
            // non-cache reasons (large Δρ, factor failed, etc.), so
            // this represents the "linear predictor stack failed
            // entirely → fall back to flat warm-start" case. Counting
            // the rate at large scale tells us how often the warm-
            // start is degenerating to flat after IFT rejects.
            let reason = if alpha <= 0.0 {
                "alpha_negative"
            } else {
                "alpha_above_cap"
            };
            log::info!(
                "[TANGENT-REJECTED] reason={} alpha={:.3e} cap={:.3e}",
                reason,
                alpha,
                alpha_cap,
            );
            return Some((cur_beta, WarmStartPredictionSource::Flat));
        }
        // Tangent noop short-circuit: when α is below the
        // numerical-noise floor, `c + α · (c − pp) ≈ c` to machine
        // precision and the per-coefficient mat-add is wasted work.
        // Symmetric to the IFT-NOOP path (commit d437aed1 / 52372fd5)
        // and tagged with the same `[TANGENT-NOOP]` marker so the
        // bench runner can distinguish "tangent-line ran with
        // negligible step" from "tangent-line produced a real
        // prediction" without inferring it from the alpha
        // distribution. Returns Flat source so the
        // [TANGENT-QUALITY] block in execute_pirls_if_needed
        // doesn't fire on a near-identity prediction (would
        // contaminate the residual percentile distribution with
        // ~zero residuals — same bug class as commit 52372fd5).
        const TANGENT_ALPHA_NOOP_EPS: f64 = 1e-12;
        if alpha.abs() <= TANGENT_ALPHA_NOOP_EPS {
            log::info!(
                "[TANGENT-NOOP] reason=alpha_below_eps alpha={:.3e} eps={:.3e}",
                alpha,
                TANGENT_ALPHA_NOOP_EPS,
            );
            return Some((cur_beta, WarmStartPredictionSource::Flat));
        }
        let mut predicted = cur_beta.0.clone();
        for ((p, c), pp) in predicted
            .iter_mut()
            .zip(cur_beta.0.iter())
            .zip(prev_beta.0.iter())
        {
            *p = c + alpha * (c - pp);
        }
        if !predicted.iter().all(|v: &f64| v.is_finite()) {
            log::info!(
                "[TANGENT-REJECTED] reason=non_finite_predicted alpha={:.3e} cap={:.3e}",
                alpha,
                alpha_cap,
            );
            return Some((cur_beta, WarmStartPredictionSource::Flat));
        }
        log::info!(
            "[TANGENT-PREDICT] alpha={:.3e} cap={:.3e} drho_step_norm_sq={:.3e} drho_prev_norm_sq={:.3e}",
            alpha,
            alpha_cap,
            step_dot_d.abs(),
            d_rho_norm_sq,
        );
        Some((
            Coefficients::new(predicted),
            WarmStartPredictionSource::TangentLine,
        ))
    }

    pub(crate) fn setwarm_start_original_beta(&self, beta_original: Option<ArrayView1<'_, f64>>) {
        if !self.warm_start_enabled.load(Ordering::Relaxed) {
            return;
        }
        // Whenever an external caller substitutes the warm-start β
        // (e.g. a multi-start seed selector), every other piece of
        // warm-start state was calibrated to the OLD β at the OLD ρ
        // and is now stale: the IFT cache's H_pen and qs were assembled
        // against the old β; the cached factor is the Cholesky of the
        // OLD H_pen; the ρ pair (warm_start_rho, prev_*) was the OLD β's
        // ρ-trajectory; the LM-λ hint was the damping that worked at
        // the OLD geometry; the inner-PIRLS feedback signals reflect
        // the OLD solve's convergence behavior; the IFT residual was
        // measured against the OLD predictor.
        //
        // Without this wipe, the IFT predictor would seed PIRLS with
        // `β_new − H_old^{-1} · (e^{ρ_old_k} S_k · (β_new-μ_k))` — using the
        // new β as a substitute for the old β in a Jacobian
        // calibrated against the old β. That is mathematically wrong
        // and would produce arbitrary predictions, not just degraded
        // ones. Similarly the LM-λ hint would clamp to a value
        // calibrated to a curvature that no longer applies.
        //
        // Wipe everything EXCEPT `warm_start_beta` only when an external
        // caller actually supplies a replacement β. Passing `None` means the
        // caller has no new seed; preserve the current warm-start state.
        if let Some(beta) = beta_original {
            self.clear_warm_start_predictor_state();
            self.clear_warm_start_adaptive_signals();
            if beta.len() == self.p {
                if !beta.iter().all(|v: &f64| v.is_finite()) {
                    // Caller supplied a β with NaN / Inf entries — would
                    // poison the next PIRLS solve immediately. Refuse
                    // (slot remains cleared) and log so the source of
                    // the bad seed is debuggable.
                    log::warn!(
                        "[warm-start] external β setter rejected non-finite seed (len={}); slot left empty",
                        beta.len(),
                    );
                    return;
                }
                self.warm_start_beta
                    .write()
                    .unwrap()
                    .replace(Coefficients::new(beta.to_owned()));
            } else {
                // Length mismatch — common bug when a caller forgets to
                // re-derive β under a basis transformation. Surface it
                // rather than silently dropping; the slot is already
                // cleared.
                log::warn!(
                    "[warm-start] external β setter rejected length mismatch: got {}, expected {}",
                    beta.len(),
                    self.p,
                );
            }
        }
    }

    pub(crate) fn current_original_basis_beta(&self) -> Option<Array1<f64>> {
        let beta_guard = self.warm_start_beta.read().ok()?;
        let beta = beta_guard.as_ref()?;
        if beta.0.len() == self.p && beta.0.iter().all(|v: &f64| v.is_finite()) {
            Some(beta.0.clone())
        } else {
            None
        }
    }

    pub(crate) fn reset_outer_seed_state(&self) {
        self.cache_manager.invalidate_eval_bundle();
        // Drop cross-call PIRLS LRU entries: cached β may have been computed under a coarsened inner cap, so reusing them on retry skips real work and bit-replays the prior attempt.
        self.cache_manager.pirls_cache.write().unwrap().clear();
        // The outer is restarting from a fresh seed — the previous
        // trajectory's warm-start signals are calibrated to a different
        // ρ-path and would mislead both predictors and the adaptive cap
        // policies. Wipe in lockstep so the first solve at the new
        // seed starts fully cold.
        self.clear_warm_start_predictor_state();
        self.clear_warm_start_adaptive_signals();
        // Inner-PIRLS iteration caps are cross-trajectory state: the
        // previous outer's first-order bridge writes `outer_inner_cap`
        // on every accepted gradient eval, and seed-screening writes
        // `screening_max_inner_iterations` during cascade probes. A
        // value left over from the prior trajectory would silently
        // shape the first PIRLS solve at the new seed under a cap the
        // new trajectory never chose. The next outer is responsible
        // for re-establishing any cap it wants via its own bridge or
        // screening hook; zeroing here is the safe baseline.
        self.outer_inner_cap.store(0, Ordering::Relaxed);
        self.screening_max_inner_iterations
            .store(0, Ordering::Relaxed);
        self.reset_hypergradient_budget_controller();
    }

    // Accessor methods for private fields
    pub(crate) fn x(&self) -> &DesignMatrix {
        &self.x
    }

    pub(crate) fn balanced_penalty_root(&self) -> &Array2<f64> {
        &self.balanced_penalty_root
    }

    /// Return a Gaussian-Identity `XᵀWX` / `XᵀW(y−offset)` cache when the
    /// outer-loop preconditions for the Identity short-circuit hold, building
    /// it lazily on the first call.  Returns `None` otherwise — callers must
    /// then route through the non-cached path.
    ///
    /// Preconditions: family `GaussianIdentity` + standard `Identity` link
    /// + no Firth bias reduction + no coefficient lower bounds + no linear
    /// inequality constraints.  These exactly match the gate in
    /// `solve_penalized_least_squares_implicit` so the cache is only ever
    /// consulted where it is mathematically equivalent to streaming the
    /// dense `XᵀWX` from scratch.
    /// True iff this state satisfies the static Gaussian-identity
    /// sufficient-statistic eligibility (the gate `gaussian_fixed_cache_if_eligible`
    /// applies before building or returning a cache).
    pub(crate) fn gaussian_fixed_cache_eligible(&self) -> bool {
        let spec = reml_spec(&self.config.likelihood);
        let family_ok = matches!(spec.response, ResponseFamily::Gaussian);
        let link_ok = matches!(
            self.config.link_kind,
            gam_problem::InverseLink::Standard(StandardLink::Identity)
        );
        family_ok
            && link_ok
            && !self.config.firth_bias_reduction
            && self.coefficient_lower_bounds.is_none()
            && self.linear_constraints.is_none()
    }

    /// Install an externally assembled Gaussian sufficient-statistic cache
    /// (#1033b): the certified ψ-Gram tensor assembles `XᵀWX(ψ)/XᵀWy(ψ)/yᵀWy`
    /// n-free per design-moving trial, and this hands it to the same slot the
    /// streamed builder fills, so every consumer (dense PLS fast path, sparse
    /// scatter, final accept-fit) picks it up via the read fast path.
    /// Refuses (returns false) when the state is statically ineligible or the
    /// shape disagrees with the current design — the caller then leaves the
    /// streamed path to do its usual work.
    pub(crate) fn install_gaussian_fixed_cache(
        &self,
        cache: Arc<crate::pirls::GaussianFixedCache>,
    ) -> bool {
        if !self.gaussian_fixed_cache_eligible()
            || cache.xtwx_orig.nrows() != self.p
            || cache.xtwx_orig.ncols() != self.p
            || cache.xtwy_orig.len() != self.p
        {
            return false;
        }
        *self.gaussian_fixed_cache.write().unwrap() = Some(cache);
        true
    }

    /// Install the frozen-weight first-Fisher-step data-fit Gram `XᵀWX` for the
    /// GLM design-moving ψ-sweep (#1111 / #1033 mechanism (c)). The Gram is in
    /// the conditioned (original / `x_fit`) column frame and must match the
    /// current `p`. Returns whether it was installed; a shape mismatch refuses
    /// (the GLM inner then restreams the first-iteration Gram as usual).
    ///
    /// NOT family-gated: this is the GLM lane's own slot. The caller
    /// ([`SpatialJointContext::eval_full`]) only installs it after certifying
    /// that the trial's converged working weight has not drifted past tolerance
    /// from the frozen snapshot, so the frozen-W Gram is a faithful stand-in for
    /// the trial's first-iteration `XᵀWX`.
    pub(crate) fn install_glm_first_step_gram(&self, gram: Arc<ndarray::Array2<f64>>) -> bool {
        if gram.nrows() != self.p || gram.ncols() != self.p {
            return false;
        }
        *self.glm_first_step_gram.write().unwrap() = Some(gram);
        true
    }

    /// Clear any installed frozen-W GLM first-step Gram. Called per-trial before
    /// (re)installing for the current ψ, so a stale previous-ψ Gram is never
    /// consumed when the current trial is out-of-window or has drifted.
    pub(crate) fn clear_glm_first_step_gram(&self) {
        *self.glm_first_step_gram.write().unwrap() = None;
    }

    /// The frozen-W GLM first-step Gram for the current in-window drift-OK trial,
    /// if installed. Consumed exactly once by the GLM inner P-IRLS first
    /// iteration; `None` keeps the streamed first-iteration Gram.
    pub(crate) fn glm_first_step_gram(&self) -> Option<Arc<ndarray::Array2<f64>>> {
        self.glm_first_step_gram
            .read()
            .unwrap()
            .as_ref()
            .map(Arc::clone)
    }

    pub(crate) fn flat_glm_first_step_gram(&self) -> Option<Arc<ndarray::Array2<f64>>> {
        self.flat_glm_first_step_gram
            .read()
            .unwrap()
            .as_ref()
            .map(Arc::clone)
    }

    /// Install the conditioned-frame exact ψ-derivative pair
    /// `(∂XᵀWX/∂ψ, ∂XᵀW(y−offset)/∂ψ)` for the single design-moving spatial
    /// hyperparameter (#1033b). Shapes must match the current `p` (k×k Gram
    /// derivative, k-vector rhs derivative) and the Gaussian-fixed cache must
    /// be eligible — the gradient lane that consumes this only fires for
    /// Gaussian-identity. Returns whether the pair was installed.
    pub(crate) fn install_gaussian_psi_gram_deriv(
        &self,
        deriv: Arc<(ndarray::Array2<f64>, ndarray::Array1<f64>)>,
    ) -> bool {
        if !self.gaussian_fixed_cache_eligible()
            || deriv.0.nrows() != self.p
            || deriv.0.ncols() != self.p
            || deriv.1.len() != self.p
        {
            return false;
        }
        *self.gaussian_psi_gram_deriv.write().unwrap() = Some(deriv);
        true
    }

    /// Clear the ψ-keyed Gaussian sufficient-statistics cache (`XᵀWX(ψ)`,
    /// `XᵀW(y−offset)(ψ)`) and its conditioned-frame ψ-derivative pair (#1033).
    ///
    /// Both slots are keyed to a SPECIFIC trial ψ from the certified ψ-Gram
    /// tensor. The slow path nulls them inside [`Self::reset_surface`], but the
    /// design-revision fast path skips `reset_surface`, so a trial that lands
    /// OFF the certified ψ-window (or otherwise cannot re-install) must clear
    /// the previous in-window ψ's Gram explicitly — otherwise the inner
    /// Gaussian PLS would read a STALE Gram keyed to the wrong ψ. With the slot
    /// cleared the inner solver restreams the exact Gram for this trial's
    /// design, as it does whenever no tensor is installed.
    pub(crate) fn clear_gaussian_fixed_cache(&self) {
        *self.gaussian_fixed_cache.write().unwrap() = None;
        *self.gaussian_psi_gram_deriv.write().unwrap() = None;
    }

    /// Clear ONLY the conditioned-frame Gaussian ψ-derivative pair (#1033),
    /// keeping the value-lane `gaussian_fixed_cache` intact. Used when a trial's
    /// ψ lies inside the certified VALUE window (so the n-free Gram is sound)
    /// but OUTSIDE the narrower certified GRADIENT sub-window: the value cache
    /// stays, but a derivative pair keyed to a prior in-sub-window ψ must be
    /// dropped so the gradient lane falls back to the exact ∂X/∂ψ slab for this
    /// trial instead of reading a stale derivative on the fast path.
    pub(crate) fn clear_gaussian_psi_gram_deriv(&self) {
        *self.gaussian_psi_gram_deriv.write().unwrap() = None;
    }

    /// Conditioned-frame exact ψ-derivative pair, when installed for the
    /// current in-window Gaussian trial (#1033b). `None` keeps the slab path.
    pub(crate) fn gaussian_psi_gram_deriv(
        &self,
    ) -> Option<Arc<(ndarray::Array2<f64>, ndarray::Array1<f64>)>> {
        self.gaussian_psi_gram_deriv
            .read()
            .unwrap()
            .as_ref()
            .map(Arc::clone)
    }

    /// Install the conditioned-frame exact ψ-derivative pair
    /// `(∂XᵀWX/∂ψ, ∂XᵀW(y−offset)/∂ψ)` for the single design-moving spatial
    /// hyperparameter in the GLM frozen-W lane (#1033 / #1111). Shapes must
    /// match the current `p` (k×k Gram derivative, k-vector rhs derivative).
    /// NOT family-gated (the GLM lane's own slot, like `glm_first_step_gram`):
    /// the caller only installs it after `gradient_pair_if_sound` certified the
    /// trial's working weight is within `GRADIENT_WEIGHT_DRIFT_RTOL` of the
    /// frozen snapshot. Returns whether the pair was installed; a shape
    /// mismatch refuses (the gradient then keeps the exact ∂X/∂ψ slab path).
    pub(crate) fn install_glm_psi_gram_deriv(
        &self,
        deriv: Arc<(ndarray::Array2<f64>, ndarray::Array1<f64>)>,
    ) -> bool {
        if deriv.0.nrows() != self.p || deriv.0.ncols() != self.p || deriv.1.len() != self.p {
            return false;
        }
        *self.glm_psi_gram_deriv.write().unwrap() = Some(deriv);
        true
    }

    /// Clear any installed GLM frozen-W conditioned-frame ψ-gram derivative.
    /// Called per-trial before (re)installing for the current ψ, so a stale
    /// previous-ψ derivative is never consumed when the current trial is
    /// out-of-window or has drifted.
    pub(crate) fn clear_glm_psi_gram_deriv(&self) {
        *self.glm_psi_gram_deriv.write().unwrap() = None;
    }

    /// Conditioned-frame exact ψ-derivative pair for the current in-window
    /// drift-OK GLM trial, when installed (#1033 / #1111). Consumed by the GLM
    /// ψ-gradient HyperCoord to serve `a_j` / `g_j` n-free (`B_j` stays the
    /// exact slab). `None` keeps the slab path.
    pub(crate) fn glm_psi_gram_deriv(
        &self,
    ) -> Option<Arc<(ndarray::Array2<f64>, ndarray::Array1<f64>)>> {
        self.glm_psi_gram_deriv
            .read()
            .unwrap()
            .as_ref()
            .map(Arc::clone)
    }

    /// Return the currently installed Gaussian sufficient-statistic cache
    /// without constructing one from `self.x`. The ψ fast path uses this to
    /// capture the exact slow-reset anchor cache, if the slow Gaussian solve
    /// already built it, while preserving the no-new-row-pass contract.
    pub(crate) fn installed_gaussian_fixed_cache(
        &self,
    ) -> Option<Arc<crate::pirls::GaussianFixedCache>> {
        self.gaussian_fixed_cache
            .read()
            .unwrap()
            .as_ref()
            .map(Arc::clone)
    }

    pub(crate) fn gaussian_fixed_cache_if_eligible(
        &self,
    ) -> Option<Arc<crate::pirls::GaussianFixedCache>> {
        // Static eligibility — these only depend on data the outer loop
        // never mutates, so the gate is correct once and stays correct.
        if !self.gaussian_fixed_cache_eligible() {
            return None;
        }
        // Fast path — already populated.
        {
            let guard = self.gaussian_fixed_cache.read().unwrap();
            if let Some(cache) = guard.as_ref() {
                return Some(Arc::clone(cache));
            }
        }
        // First-call construction. Re-check under the write lock so two
        // concurrent callers cannot both pay the O(N·p²) bill.
        let mut guard = self.gaussian_fixed_cache.write().unwrap();
        if let Some(cache) = guard.as_ref() {
            return Some(Arc::clone(cache));
        }
        let build_start = std::time::Instant::now();
        let weights_owned = self.weights.to_owned();
        // wz_minus_offset = (y - offset) elementwise; cache stores XᵀW(y−offset).
        let mut wz = self.y.to_owned();
        wz -= &self.offset;
        wz *= &weights_owned;
        let centered_weighted_y_sq = self
            .y
            .iter()
            .zip(self.offset.iter())
            .zip(weights_owned.iter())
            .map(|((&y, &offset), &w)| {
                let centered = y - offset;
                w * centered * centered
            })
            .sum::<f64>();
        let xtwx = match gam_linalg::matrix::LinearOperator::xt_diag_x_signed_op(
            &self.x,
            gam_linalg::matrix::SignedWeightsView::from_array(&weights_owned),
        ) {
            Ok(m) => m,
            Err(e) => {
                log::warn!("[gaussian-fixed-cache] disabling cache: failed to build XᵀWX: {e}");
                return None;
            }
        };
        let xtwy = self.x.transpose_vector_multiply(&wz);
        // Build the matching sparse-path XᵀWX as well, when the design
        // exposes a sparse form. The sparse REML path rebuilds
        // H = XᵀWX + Sλ + δI per outer eval, so caching the constant
        // XᵀWX contribution lets the inner assemble skip the SpGEMM.
        let xtwx_sparse_orig = if let Some(sparse_design) = self.x.as_sparse() {
            let sparse_start = std::time::Instant::now();
            match crate::pirls::SparseXtwxPrecomputed::build(sparse_design.as_ref(), &weights_owned)
            {
                Ok(precomp) => {
                    log::info!(
                        "[gaussian-fixed-cache] sparse XᵀWX nnz={} built in {:.3} ms",
                        precomp.xtwxvalues.len(),
                        sparse_start.elapsed().as_secs_f64() * 1e3
                    );
                    Some(Arc::new(precomp))
                }
                Err(e) => {
                    log::warn!(
                        "[gaussian-fixed-cache] sparse XᵀWX build failed; falling back: {e}"
                    );
                    None
                }
            }
        } else {
            None
        };
        let cache = Arc::new(crate::pirls::GaussianFixedCache {
            xtwx_orig: xtwx,
            xtwy_orig: xtwy,
            centered_weighted_y_sq,
            row_prediction_is_stale: false,
            xtwx_sparse_orig,
            // Exact (non-stale) path: rows are freshly realised from the design,
            // so there is no shared frozen-row bundle to attach (#1868).
            frozen_rows: None,
        });
        log::info!(
            "[gaussian-fixed-cache] built p={} n={} in {:.3} ms",
            self.p,
            self.y.len(),
            build_start.elapsed().as_secs_f64() * 1e3
        );
        *guard = Some(Arc::clone(&cache));
        Some(cache)
    }

    pub(crate) fn canonical_penalties(&self) -> &[gam_terms::construction::CanonicalPenalty] {
        &self.canonical_penalties
    }

    pub(super) fn prepare_dense_eval_bundlewithkey(
        &self,
        rho: &Array1<f64>,
        key: Option<Vec<u64>>,
    ) -> Result<EvalShared, EstimationError> {
        let pirls_result = self.execute_pirls_if_needed(rho)?;
        let (mut h_total, ridge_passport) = self.effectivehessian(pirls_result.as_ref())?;
        let mut firth_dense_operator: Option<Arc<FirthDenseOperator>> = None;
        if let Some(jeffreys_link) = reml_robust_jeffreys_link(&self.config) {
            let firth_n = pirls_result.x_transformed.nrows();
            let firth_p = pirls_result.x_transformed.ncols();
            if !super::firth_problem_scale_allows(firth_n, firth_p) {
                log::info!(
                    "disabling Firth bias reduction for large model (n={}, p={}, n*p={}, n*p^2={}): \
                     exact Firth operator is small-model-only",
                    firth_n,
                    firth_p,
                    firth_n.saturating_mul(firth_p),
                    firth_n.saturating_mul(firth_p).saturating_mul(firth_p),
                );
            } else {
                let x_dense = pirls_result
                .x_transformed
                .try_to_dense_arc(
                    "dense REML eval bundle requires dense transformed design for Firth operator",
                )
                .map_err(EstimationError::InvalidInput)?;
                let firth_build_start = std::time::Instant::now();
                let firth_op = Arc::new(Self::build_firth_dense_operator_for_link(
                    &jeffreys_link,
                    x_dense.as_ref(),
                    &pirls_result.final_eta.to_owned(),
                    self.weights,
                )?);
                log::debug!(
                    "[Firth-op] build n={} p={} r={} half_logdet={:.3e} elapsed={:.3}s",
                    firth_op.x_dense.nrows(),
                    firth_op.x_dense.ncols(),
                    firth_op.k_reduced.nrows(),
                    firth_op.half_log_det,
                    firth_build_start.elapsed().as_secs_f64(),
                );
                // Firth-adjusted inner Jacobian for implicit differentiation:
                //   H_total = Xᵀ W X + S - H_φ,
                //   H_φ     = ∇²_β Φ
                //          = 0.5 [ Xᵀ diag(w'' ⊙ h) X - Bᵀ P B ].
                // This keeps B_k/B_{kl} solves on the same objective surface as
                // the Firth-augmented stationarity system.
                //
                // Conceptually Φ is the identifiable-subspace Jeffreys term
                // obtained by evaluating W on a canonical orthonormal basis of
                // the transformed design column space. The hphi block below is
                // therefore the curvature of that basis-invariant penalty,
                // represented in the current transformed basis.
                let mut weighted_xtdx = Array2::<f64>::zeros((0, 0));
                let diag_term = Self::xt_diag_x_dense_into(
                    &firth_op.x_dense,
                    &(&firth_op.w2 * &firth_op.h_diag),
                    &mut weighted_xtdx,
                );
                let bpb = gam_linalg::faer_ndarray::fast_atb(&firth_op.b_base, &firth_op.p_b_base);
                let mut hphi = 0.5 * (diag_term - bpb);
                // Numerical symmetry guard.
                gam_linalg::matrix::symmetrize_in_place(&mut hphi);
                // Keep tiny numerical noise from making the solve surface less stable.
                if hphi.iter().all(|v| v.is_finite()) {
                    h_total -= &hphi;
                }
                firth_dense_operator = Some(firth_op);
            } // else (not too large for Firth)
        }

        // Add log-barrier Hessian diagonal for monotonicity-constrained coefficients.
        // This augments the penalized Hessian before the spectral decomposition so
        // that logdet, trace, and solve operations all reflect the barrier curvature.
        if let Some(ref lin) = pirls_result.linear_constraints_transformed
            && let Some(barrier_cfg) = Self::barrier_config_from_constraints(lin)
        {
            let beta_t = pirls_result.beta_transformed.as_ref();
            if let Err(e) = barrier_cfg.add_barrier_hessian_diagonal(&mut h_total, beta_t) {
                log::warn!("Barrier Hessian diagonal skipped: {e}");
            }
        }

        Ok(EvalShared {
            key,
            pirls_result,
            ridge_passport,
            geometry: RemlGeometry::DenseSpectral,
            h_total: Arc::new(h_total),
            sparse_exact: None,
            firth_dense_operator,
            firth_dense_operator_original: None,
            penalty_pseudologdet: std::sync::OnceLock::new(),
            penalty_scores_at_mode: std::sync::OnceLock::new(),
            block_local_correction: std::sync::OnceLock::new(),
        })
    }

    pub(super) fn prepare_sparse_eval_bundlewithkey(
        &self,
        rho: &Array1<f64>,
        key: Option<Vec<u64>>,
    ) -> Result<EvalShared, EstimationError> {
        let pirls_result = self.execute_pirls_if_needed(rho)?;
        if !matches!(
            pirls_result.coordinate_frame,
            pirls::PirlsCoordinateFrame::OriginalSparseNative
        ) {
            crate::bail_invalid_estim!(
                "sparse exact geometry requires sparse-native PIRLS coordinates"
            );
        }
        let ridge_passport = pirls_result.ridge_passport;
        let x_sparse = self.x().as_sparse().ok_or_else(|| {
            EstimationError::InvalidInput(
                "sparse exact geometry requires sparse original design".to_string(),
            )
        })?;
        self.sparse_penalty_block_count.ok_or_else(|| {
            EstimationError::InvalidInput(
                "sparse exact geometry requires block-separable penalties".to_string(),
            )
        })?;

        let lambdas = rho.mapv(f64::exp);
        let mut s_lambda = Array2::<f64>::zeros((self.p, self.p));
        for (k, cp) in self.canonical_penalties.iter().enumerate() {
            if k < lambdas.len() && lambdas[k] != 0.0 {
                cp.accumulate_weighted(&mut s_lambda, lambdas[k]);
            }
        }
        // Add log-barrier Hessian diagonal for monotonicity-constrained
        // coefficients (sparse path uses original coordinates).
        if let Some(ref lin) = self.linear_constraints
            && let Some(barrier_cfg) = Self::barrier_config_from_constraints(lin)
        {
            let beta_orig = self.sparse_exact_beta_original(pirls_result.as_ref());
            if let Err(e) = barrier_cfg.add_barrier_hessian_diagonal(&mut s_lambda, &beta_orig) {
                log::warn!("Sparse barrier Hessian diagonal skipped: {e}");
            }
        }

        let mut workspace = PirlsWorkspace::new(self.y.len(), self.p, 0, 0);
        // Gaussian-Identity fast path: reuse the per-RemlState `XᵀWX` cache
        // built once from constant weights. The outer REML loop never
        // mutates W for Gaussian, so the inner sparse assemble can scatter
        // these values directly instead of re-running the SpGEMM.
        let gaussian_cache = self.gaussian_fixed_cache_if_eligible();
        let precomputed_xtwx = gaussian_cache
            .as_ref()
            .and_then(|c| c.xtwx_sparse_orig.as_ref().map(|arc| arc.as_ref()));
        let (hessian_weights, _, _) = self.hessian_surface_arrays(pirls_result.as_ref())?;
        let sparse_system = assemble_and_factor_sparse_penalized_system(
            &mut workspace,
            x_sparse,
            &hessian_weights,
            &s_lambda,
            ridge_passport.delta,
            precomputed_xtwx,
        )?;
        let lambdas_slice = lambdas.as_slice().ok_or_else(|| {
            EstimationError::InvalidInput(
                "non-contiguous lambda storage in sparse penalty logdet".to_string(),
            )
        })?;
        let penalty_logdet = super::penalty_logdet::PenaltyPseudologdet::from_penalties(
            &self.canonical_penalties,
            lambdas_slice,
            ridge_passport.penalty_logdet_ridge(),
            self.p,
        )
        .map_err(EstimationError::InvalidInput)?;
        let penalty_rank = penalty_logdet.rank();
        let logdet_s_pos = penalty_logdet.value();
        let (det1_values, _) =
            penalty_logdet.rho_derivatives_from_penalties(&self.canonical_penalties, lambdas_slice);
        let firth_dense_operator_original = if let Some(jeffreys_link) =
            reml_robust_jeffreys_link(&self.config)
        {
            let firth_n = self.x().nrows();
            let firth_p = self.x().ncols();
            if !super::firth_problem_scale_allows(firth_n, firth_p) {
                log::info!(
                    "disabling Firth bias reduction for large model (n={}, p={}, n*p={}, n*p^2={}): \
                     exact Firth operator is small-model-only",
                    firth_n,
                    firth_p,
                    firth_n.saturating_mul(firth_p),
                    firth_n.saturating_mul(firth_p).saturating_mul(firth_p),
                );
                None
            } else {
                let x_dense = self
                    .x()
                    .try_to_dense_arc(
                        "sparse exact REML runtime requires dense design for Firth operator",
                    )
                    .map_err(EstimationError::InvalidInput)?;
                Some(Arc::new(Self::build_firth_dense_operator_for_link(
                    &jeffreys_link,
                    x_dense.as_ref(),
                    &pirls_result.final_eta.to_owned(),
                    self.weights,
                )?))
            }
        } else {
            None
        };

        Ok(EvalShared {
            key,
            pirls_result,
            ridge_passport,
            geometry: RemlGeometry::SparseExactSpd,
            h_total: Arc::new(Array2::zeros((0, 0))),
            sparse_exact: Some(Arc::new({
                let factor = Arc::new(sparse_system.factor);
                // Compute Takahashi selected inverse from simplicial factorization.
                // This precomputes H^{-1} entries on the filled pattern of L, enabling
                // O(nnz) trace computations instead of O(p) column solves.
                let sfactor =
                    gam_linalg::sparse_exact::factorize_simplicial(&sparse_system.h_sparse)?;
                let takahashi = Some(Arc::new(
                    gam_linalg::sparse_exact::TakahashiInverse::compute(&sfactor)?,
                ));
                SparseExactEvalData {
                    factor,
                    takahashi,
                    logdet_h: sparse_system.logdet_h,
                    logdet_s_pos,
                    penalty_rank,
                    det1_values: Arc::new(det1_values),
                }
            })),
            firth_dense_operator: None,
            firth_dense_operator_original,
            penalty_pseudologdet: std::sync::OnceLock::new(),
            penalty_scores_at_mode: std::sync::OnceLock::new(),
            block_local_correction: std::sync::OnceLock::new(),
        })
    }

    /// Runs the inner P-IRLS loop, caching the result.
    pub(super) fn execute_pirls_if_needed(
        &self,
        rho: &Array1<f64>,
    ) -> Result<Arc<PirlsResult>, EstimationError> {
        let use_cache = self
            .cache_manager
            .pirls_cache_enabled
            .load(Ordering::Relaxed);
        // Use sanitized key to handle NaN and -0.0 vs 0.0 issues
        let key_opt = self.rhokey_sanitized(rho);
        if use_cache
            && let Some(key) = &key_opt
            && let Some(cached) = self.cache_manager.pirls_cache.write().unwrap().get(key)
        {
            // Do not overwrite the current warm start from cache hits.
            // Line search / multi-eval outer loops revisit older rho keys and
            // replacing a recent nearby beta with an older cached mode can
            // materially slow subsequent PIRLS convergence.
            if cached.cache_compacted {
                let mut pirls_config = self.config.as_pirls_config();
                pirls_config.link_kind = self.runtime_inverse_link();
                return Ok(Arc::new(cached.rehydrate_after_reml_cache(
                    self.x(),
                    self.y,
                    self.weights,
                    self.offset.view(),
                    &pirls_config.link_kind,
                )?));
            }
            return Ok(cached);
        }

        // Detect whether we are running under outer-loop seed screening. While
        // screening is active the planner caps inner iterations to a small
        // value (~3); trial seeds are NOT expected to certify a stationary
        // mode under that cap. Partial fits whose objective (deviance +
        // penalty), β, Hessian proxy, and residual are all finite are
        // surfaced as `Ok` so the caller can rank them by an approximate
        // cost (`screening_residual_penalty`); KKT enforcement, the
        // pirls_cache LRU write, and the warm-start update are all
        // suppressed to keep screening-mode results out of cross-call
        // state. The single atomic load below feeds both the bool and the
        // iteration cap from one observation, avoiding a stale-value race
        // between the two derivations.
        let screening_cap = self.screening_max_inner_iterations.load(Ordering::Relaxed);
        let in_screening = screening_cap > 0;
        // The shallow (~3-iteration) screening cap ranks candidate ρ by the
        // partial-fit `min_penalized_deviance` proxy. That proxy is only a
        // ρ-comparable quality signal when the inner solve descends toward its
        // optimum at a rate that does not itself depend on ρ — true for the
        // unconstrained penalized least-squares / P-IRLS Newton solve, which
        // reaches (near) its minimizer within the cap at every ρ.
        //
        // It is FALSE for an inequality-constrained inner solve. With box /
        // linear shape constraints the inner solver is an active-set QP whose
        // traversal length varies with ρ: an under-smoothing ρ (the seed that
        // recovers genuine curvature) must walk OFF the cone vertex / linear
        // corner and RELEASE many curvature bounds before its penalized
        // deviance drops, while an over-smoothing ρ sits at the linear corner —
        // its constrained optimum — from the first iterate. Truncating both at
        // ~3 iterations therefore makes the under-smoothing seed's proxy look
        // far worse than its true converged cost, so screening systematically
        // ranks the flat over-smoothed corner first and the real fit launches
        // from it. REML then parks at the linear corner (EDF pinned to the
        // affine null, R²≈0) on a clean convex signal an unconstrained s(x)
        // recovers — the #1380 collapse, and exactly why it is a sharp SNR×n
        // cliff (stronger signal lets even a 3-iteration partial solve overtake
        // the flat seed). The fix is to let each constrained candidate ρ be
        // judged at its CONVERGED constrained optimum: keep `in_screening`'s
        // cache/KKT-suppression semantics (these results must still stay out of
        // cross-call state), but do not apply the iteration cap to the
        // active-set solve. The other (non-constrained) terms in the same fit
        // still converge fast, so the screening pass stays cheap.
        let screening_iteration_cap_applies = in_screening
            && self.coefficient_lower_bounds.is_none()
            && self.linear_constraints.is_none();
        // Outer-aware cap: a sibling atomic that only caps the inner Newton
        // iteration count. Unlike `screening_cap`, it does NOT suppress
        // cache writes / warm-start updates / KKT enforcement — it is purely
        // a budget. Driven by the outer optimizer to coarsen early-iter
        // inner solves when ρ is far from converged. Both caps are honored
        // jointly via `min` when both are nonzero.
        let raw_outer_cap = self.outer_inner_cap.load(Ordering::Relaxed);
        let efs_single_loop_cap = decode_efs_single_loop_cap(raw_outer_cap);
        let in_efs_single_loop = efs_single_loop_cap.is_some();
        let outer_cap = efs_single_loop_cap.unwrap_or(raw_outer_cap);

        // Run P-IRLS with original matrices to perform fresh reparameterization
        // The returned result will include the transformation matrix qs.
        //
        // Warm-start: try the tangent-line prediction first (uses last
        // two (ρ, β) pairs to extrapolate β at the new ρ; falls back to
        // the flat last-β when no history). Either path produces a
        // `Coefficients` value used as the inner Newton's seed.
        if !in_screening {
            self.load_persistent_warm_start_once();
        }
        let predicted_warm_start_with_source = if self.warm_start_enabled.load(Ordering::Relaxed) {
            self.predict_warm_start_beta_with_source(rho)
        } else {
            None
        };
        let predicted_warm_start = predicted_warm_start_with_source
            .as_ref()
            .map(|(c, _)| c.clone());
        let prediction_source = predicted_warm_start_with_source.as_ref().map(|(_, s)| *s);
        let pirls_result = {
            let warm_start_holder = self.warm_start_beta.read().unwrap();
            let fallback_warm_start_ref = if self.warm_start_enabled.load(Ordering::Relaxed) {
                warm_start_holder.as_ref()
            } else {
                None
            };
            let warm_start_ref = predicted_warm_start.as_ref().or(fallback_warm_start_ref);
            let mut pirls_config = self.config.as_pirls_config();
            let original_cap = pirls_config.max_iterations;
            if screening_iteration_cap_applies {
                pirls_config.max_iterations = pirls_config.max_iterations.min(screening_cap);
            }
            if outer_cap > 0 {
                pirls_config.max_iterations = pirls_config.max_iterations.min(outer_cap);
            }
            // Seed-screening prepass: rank candidate ρ by a CHEAP partial fit,
            // never a full PIRLS-to-production-tolerance solve. The iteration cap
            // above bounds the unconstrained case, but it is deliberately NOT
            // applied to the inequality-constrained active-set solve (#1380), and
            // the screening cascade's uncapped final stage
            // (`SEED_SCREENING_UNCAPPED`) lifts the cap entirely when every capped
            // stage collapsed — both then run the inner solve all the way to the
            // tight production inner tolerance just to score a starting basin,
            // which at large `n` is the dominant #1033/#1575/#1082 prepass cost.
            // Loosen the inner convergence tolerance to a coarse screening floor
            // so the solve terminates as soon as the penalized-deviance proxy is
            // resolved to ranking accuracy. The proxy stays ρ-comparable because
            // EVERY candidate is judged at the identical coarse tolerance, exactly
            // as it is judged at the identical iteration cap. Only ever loosens
            // (`max`), never tightens, so a caller already running a coarser inner
            // solve keeps it. The winning seed's real fit and every multistart
            // full solve run with `in_screening == false`, so the converged
            // optimum (and its bit-results) is byte-for-byte unchanged — this
            // touches only which seed the descent starts from.
            if in_screening {
                pirls_config.convergence_tolerance = pirls_config
                    .convergence_tolerance
                    .max(SEED_SCREENING_INNER_CONVERGENCE_TOLERANCE);
            }
            if pirls_config.max_iterations != original_cap
                || (in_screening
                    && pirls_config.convergence_tolerance
                        > self.config.pirls_convergence_tolerance)
            {
                log::debug!(
                    "[PIRLS cap] inner_max_iterations={} (full={} screening={} outer={}) inner_tol={:.1e} (full_tol={:.1e})",
                    pirls_config.max_iterations,
                    original_cap,
                    if screening_iteration_cap_applies {
                        screening_cap as i64
                    } else {
                        -1
                    },
                    if outer_cap > 0 { outer_cap as i64 } else { -1 },
                    pirls_config.convergence_tolerance,
                    self.config.pirls_convergence_tolerance,
                );
            }
            pirls_config.link_kind = self.runtime_inverse_link();
            // Negative-Binomial λ-search θ freeze (#1082). With θ estimated,
            // the inner solver re-derives θ from each outer iterate's warm-start
            // η, so the NB working response / deviance / penalty-logdet — and
            // thus the REML criterion — drift every outer evaluation, defeating
            // the projected-gradient convergence test and grinding the loop to
            // max_iter. Once the first non-screening solve has fixed a
            // data-driven θ (captured below into `frozen_negbin_theta`), pin
            // every subsequent λ-search inner solve to that value so
            // `F(ρ) = REML(ρ, θ_frozen)` is a stationary function of ρ. θ is
            // still ML-refreshed at the single final reported fit (the
            // `refine_dispersion_at_converged_eta = true` accept-fit in
            // `optimizer.rs`), exactly as the dispersion-at-converged-η contract
            // requires. No effect on non-NB or user-fixed-θ specs.
            if pirls_config.likelihood.negbin_theta_is_estimated() {
                let frozen_bits = self.frozen_negbin_theta.load(Ordering::Relaxed);
                if frozen_bits != 0 {
                    let frozen_theta = f64::from_bits(frozen_bits);
                    if frozen_theta.is_finite() && frozen_theta > 0.0 {
                        pirls_config.likelihood = pirls_config
                            .likelihood
                            .clone()
                            .with_negbin_theta_frozen_for_search(frozen_theta);
                    }
                }
            }
            // Tweedie λ-search φ freeze (#1477). The same drift mechanism as the
            // NB θ freeze above, with a sharper failure mode: the Tweedie LAML
            // `−ℓ(β̂)` omits the φ-dependent saddlepoint normalizer, so a φ
            // re-estimated from each outer iterate's warm-start η does not merely
            // make `F(ρ)` drift — it makes the criterion REWARD dispersion
            // inflation, railing a double-penalty null-space `λ` to the box bound
            // and shipping a boundary blow-up (#1477). Pin every λ-search inner
            // solve to the first converged solve's Pearson φ so
            // `F(ρ) = REML(ρ, φ_frozen)` is stationary in ρ; φ is still refreshed
            // at the single final reported fit. No effect on non-Tweedie or
            // user-fixed-φ specs.
            if pirls_config.likelihood.tweedie_phi_is_estimated() {
                let frozen_bits = self.frozen_tweedie_phi.load(Ordering::Relaxed);
                if frozen_bits != 0 {
                    let frozen_phi = f64::from_bits(frozen_bits);
                    if frozen_phi.is_finite() && frozen_phi > 0.0 {
                        pirls_config.likelihood = pirls_config
                            .likelihood
                            .clone()
                            .with_tweedie_phi_frozen_for_search(frozen_phi);
                    }
                }
            }
            // Gamma λ-search shape freeze (#1074). Same drift mechanism as the NB
            // θ and Tweedie φ freezes above: with the shape `k` estimated, the
            // inner solver re-derives it from each outer iterate's warm-start η,
            // so `k` — and through it BOTH the Gamma curvature `H = k·XᵀX + λS`
            // and the data-fit `−ℓ = k·½D` (the `k`-saturated normalizer is
            // dropped, #359) — jumps with ρ. The realized REML cost then develops
            // deterministic spikes (a flat warm-start η at a just-rejected
            // over-smoothed trial gives a small `k`, the fitted-surface η at the
            // neighbor a ~2× larger one), the analytic outer gradient (which
            // holds `k` fixed) can never match the cost's `k(ρ)` motion, the
            // projected gradient floors well above tolerance, and the ARC descent
            // stalls and rails λ to the over-smoothed corner (the #1074 te/Gamma
            // tensor under-recovery). Pin every λ-search inner solve to the first
            // converged solve's MLE `k` so `F(ρ) = REML(ρ, k_frozen)` is
            // stationary in ρ; `k` is still refreshed at the single final
            // reported fit. No effect on non-Gamma or user-fixed-shape specs.
            if pirls_config.likelihood.scale.gamma_shape_is_estimated() {
                let frozen_bits = self.frozen_gamma_shape.load(Ordering::Relaxed);
                if frozen_bits != 0 {
                    let frozen_shape = f64::from_bits(frozen_bits);
                    if frozen_shape.is_finite() && frozen_shape > 0.0 {
                        pirls_config.likelihood = pirls_config
                            .likelihood
                            .clone()
                            .with_gamma_shape_frozen_for_search(frozen_shape);
                    }
                }
            }
            // Levenberg-Marquardt damping warm-start. Read the cached
            // λ from the previous successful PIRLS solve at this
            // surface (0 = no hint), and seed the inner solver. The
            // PIRLS layer applies a final safety clamp (defense in
            // depth); this layer pre-clamps adaptively based on the
            // previous solve's halving history, plus a quality signal
            // that the static clamp couldn't see.
            //
            // The principle: the cached λ encodes the curvature regime
            // the previous fit settled into. A Newton-friendly fit
            // (converged in ≤2 iters, no halving) leaves λ near the
            // 1e-9 floor; the next fit at a nearby ρ is likely
            // Newton-friendly too, so we allow the hint down to that
            // floor. A hard fit (many iters, possibly hit cap) leaves
            // λ in the heavy-damping regime; the next fit needs to
            // preserve that signal, so we allow the hint up to ~1.0.
            // The default (1-9 iters, converged) matches the historical
            // static clamp [1e-6, 1e-3].
            //
            // Each regime keeps the cached value's geometry information;
            // none of them throw it away. The Madsen rejection
            // trajectory (commit d37626e6) handles "wrong starting λ"
            // gracefully, so the cost of a slightly mis-tuned regime
            // is just a few extra rejections — far less than the cost
            // of throwing away the cache entirely.
            let cached_lambda_bits = self.last_pirls_lm_lambda.load(Ordering::Relaxed);
            if cached_lambda_bits != 0 {
                let cached_lambda = f64::from_bits(cached_lambda_bits);
                let last_iters = self.last_inner_iters.load(Ordering::Relaxed);
                let last_converged = self.last_inner_converged.load(Ordering::Relaxed);
                pirls_config.initial_lm_lambda =
                    adaptive_lm_lambda_hint(cached_lambda, last_iters, last_converged);
            }
            let adaptive_kkt_tolerance = if !in_screening {
                if let Some(override_tol) = self.hypergradient_adaptive_kkt_override(&pirls_config)
                {
                    Some(override_tol)
                } else if let Some(outer_grad_norm) = self.previous_outer_gradient_norm(&key_opt) {
                    // Ceiling is pinned to the tight inner tolerance. Loosening
                    // it to a fixed 1e-6 ceiling (#1575) made every inner solve
                    // uniformly coarse so the outer REML gradient was inaccurate
                    // and the optimizer stalled, declaring convergence at a
                    // non-stationary point (‖g‖≈0.38). The adaptive schedule
                    // still tightens monotonically to `floor` as ‖g_outer‖ → 0;
                    // with the ceiling at the tight tolerance the clamp can only
                    // tighten, preserving the genuine REML stationary point.
                    let ceiling = pirls_config.convergence_tolerance;
                    let floor = (self.config.reml_convergence_tolerance
                        / ADAPTIVE_KKT_FLOOR_REML_DIVISOR)
                        .min(ceiling);
                    (floor > 0.0 && ceiling >= floor).then_some(pirls::AdaptiveKktTolerance {
                        eta: ADAPTIVE_KKT_ETA,
                        floor,
                        ceiling,
                        outer_grad_norm,
                    })
                } else {
                    None
                }
            } else {
                None
            };
            // Gaussian + Identity outer REML reuses a precomputed XᵀWX and
            // XᵀW(y − offset) across every inner solve; for other families /
            // links this returns None and the inner solver falls back to the
            // streaming GEMM.
            let cache_handle = self.gaussian_fixed_cache_if_eligible();
            // #1111 / #1033 mechanism (c): the frozen-W first-Fisher-step Gram
            // (installed by the spatial GLM ψ-trial when it covers ψ and the
            // working weight has not drifted) serves the GLM inner P-IRLS first
            // iteration's XᵀWX n-free.
            let staged_glm_first_step_handle = self.glm_first_step_gram();
            let flat_glm_first_step_handle = if staged_glm_first_step_handle.is_none()
                && !in_screening
                && !self.config.firth_bias_reduction
                && warm_start_ref.is_some()
                && (predicted_warm_start.is_none()
                    || matches!(prediction_source, Some(WarmStartPredictionSource::Flat)))
            {
                self.flat_glm_first_step_gram()
            } else {
                None
            };
            let glm_first_step_handle = staged_glm_first_step_handle.or(flat_glm_first_step_handle);
            let problem = pirls::PirlsProblem {
                x: &self.x,
                offset: self.offset.view(),
                y: self.y,
                priorweights: self.weights,
                covariate_se: None,
                gaussian_fixed_cache: cache_handle.as_deref(),
                glm_first_step_gram: glm_first_step_handle.as_deref(),
            };
            let penalty = pirls::PenaltyConfig {
                canonical_penalties: &self.canonical_penalties,
                balanced_penalty_root: Some(&self.balanced_penalty_root),
                reparam_invariant: Some(&self.reparam_invariant),
                p: self.p,
                coefficient_lower_bounds: self.coefficient_lower_bounds.as_ref(),
                linear_constraints_original: self.linear_constraints.as_ref(),
                penalty_shrinkage_floor: self.penalty_shrinkage_floor,
                kronecker_factored: self.kronecker_factored.as_ref(),
            };
            let pirls_start = std::time::Instant::now();
            let result = pirls::fit_model_for_fixed_rho_with_adaptive_kkt(
                LogSmoothingParamsView::new(rho.view()),
                problem,
                penalty,
                &pirls_config,
                warm_start_ref,
                adaptive_kkt_tolerance,
                // REML cost eval: never re-profile the family dispersion (Gamma
                // shape / Beta precision) against the trial λ's residuals — that
                // would couple the scale to λ and bias selection (#678, #769).
                false,
            );
            let pirls_elapsed = pirls_start.elapsed();
            if let Ok((ref res, ref wm)) = result {
                // Surface every non-screening PIRLS call at INFO so CI logs
                // reveal exactly which inner solves dominate wall-clock at
                // large scale — the main signal for "what's the slow path"
                // when an outer BFGS / line-search step blows past the
                // 2400 s job budget.
                log::info!(
                    "[STAGE] inner pirls solve iters={} status={:?} max_eta={:.1} jeffreys_logdet={} elapsed={:.3}s",
                    wm.iterations,
                    res.status,
                    res.max_abs_eta,
                    res.jeffreys_logdet()
                        .map(|v| format!("{v:.3e}"))
                        .unwrap_or_else(|| "none".to_string()),
                    pirls_elapsed.as_secs_f64(),
                );
            }
            result
        };

        if let Err(e) = &pirls_result {
            if in_screening {
                // Seed-screening intentionally caps inner iterations very low,
                // so trial-point failures here are routine and ranked — not
                // bugs.
                log::debug!("[seed-screen] P-IRLS rejected candidate: {e:?}");
            } else {
                log::warn!("[GAM COST]   -> P-IRLS INNER LOOP FAILED. Error: {e:?}");
            }
            // Keep the previous successful warm start even when a trial point
            // fails. Outer line search commonly probes unstable candidates and
            // then returns to nearby feasible rho values where the prior warm
            // beta remains the best initializer.
        }

        let (pirls_result, _) = pirls_result?; // Propagate error if it occurred
        let pirls_result = Arc::new(pirls_result);
        // Capture the data-driven NB θ from the first converged non-screening
        // λ-search solve and freeze it for the rest of the search (#1082). The
        // first solve still estimated θ from the seed η (this branch only runs
        // when no frozen value exists yet), so the captured value is the same
        // ML θ the legacy estimated path would have used at the seed — we simply
        // stop letting it drift on subsequent outer evaluations. Screening
        // solves use a tiny inner budget and a partial mode, so they are never
        // the source of the frozen value.
        if !in_screening
            && pirls_result.likelihood.negbin_theta_is_estimated()
            && self.frozen_negbin_theta.load(Ordering::Relaxed) == 0
            && matches!(
                pirls_result.status,
                pirls::PirlsStatus::Converged | pirls::PirlsStatus::StalledAtValidMinimum
            )
            && let Some(theta) = pirls_result.likelihood.negbin_theta()
            && theta.is_finite()
            && theta > 0.0
        {
            self.frozen_negbin_theta
                .store(theta.to_bits(), Ordering::Relaxed);
            log::info!(
                "[OUTER] negative-binomial λ-search θ frozen at {theta:.6e} (#1082); \
                 outer REML criterion now stationary in ρ"
            );
        }
        // Capture the data-driven Tweedie φ from the first converged non-screening
        // λ-search solve and freeze it for the rest of the search (#1477), exactly
        // as for the NB θ above. The first solve estimated φ from the seed η via
        // the Pearson moment estimator (this branch only runs when no frozen value
        // exists yet), so the captured value is the seed-fit φ the estimated path
        // would have used — we simply stop letting it drift (and reward dispersion
        // inflation) on subsequent outer evaluations.
        if !in_screening
            && pirls_result.likelihood.tweedie_phi_is_estimated()
            && self.frozen_tweedie_phi.load(Ordering::Relaxed) == 0
            && matches!(
                pirls_result.status,
                pirls::PirlsStatus::Converged | pirls::PirlsStatus::StalledAtValidMinimum
            )
            && let Some(phi) = pirls_result.likelihood.fixed_phi()
            && phi.is_finite()
            && phi > 0.0
        {
            self.frozen_tweedie_phi.store(phi.to_bits(), Ordering::Relaxed);
            log::info!(
                "[OUTER] tweedie λ-search φ frozen at {phi:.6e} (#1477); \
                 outer LAML criterion now stationary in ρ"
            );
        }
        // Capture the data-driven Gamma shape `k = 1/φ` from the first converged
        // non-screening λ-search solve and freeze it for the rest of the search
        // (#1074), exactly as for the NB θ and Tweedie φ above. The first solve
        // estimated `k` from the seed η via the converged-η Gamma-shape MLE (this
        // branch only runs when no frozen value exists yet), so the captured
        // value is the seed-fit `k` the estimated path would have used — we
        // simply stop letting it drift with each warm-start η (which makes both
        // the curvature `H = k·XᵀX + λS` and the data-fit `k·½D` jump with ρ and
        // rails λ to the over-smoothed corner) on subsequent outer evaluations.
        if !in_screening
            && pirls_result.likelihood.scale.gamma_shape_is_estimated()
            && self.frozen_gamma_shape.load(Ordering::Relaxed) == 0
            && matches!(
                pirls_result.status,
                pirls::PirlsStatus::Converged | pirls::PirlsStatus::StalledAtValidMinimum
            )
            && let Some(shape) = pirls_result.likelihood.gamma_shape()
            && shape.is_finite()
            && shape > 0.0
        {
            self.frozen_gamma_shape.store(shape.to_bits(), Ordering::Relaxed);
            log::info!(
                "[OUTER] gamma λ-search shape frozen at {shape:.6e} (#1074); \
                 outer REML criterion now stationary in ρ"
            );
        }
        // Under seed screening the inner solver is intentionally given a tiny
        // iteration budget, so KKT stationarity will not be satisfied at the
        // partial mode. Skip the certificate so the seed can still be ranked
        // by an approximate cost; the actual fit (full inner budget) will
        // certify KKT later.
        if !in_screening && !in_efs_single_loop {
            self.enforce_constraint_kkt(pirls_result.as_ref())?;
        }

        // Check the status returned by the P-IRLS routine.
        match pirls_result.status {
            pirls::PirlsStatus::Converged | pirls::PirlsStatus::StalledAtValidMinimum => {
                if !in_screening {
                    if let Some(predicted) = predicted_warm_start.as_ref() {
                        let converged_original = match pirls_result.coordinate_frame {
                            pirls::PirlsCoordinateFrame::OriginalSparseNative => {
                                pirls_result.beta_transformed.as_ref().clone()
                            }
                            pirls::PirlsCoordinateFrame::TransformedQs => pirls_result
                                .reparam_result
                                .qs
                                .dot(pirls_result.beta_transformed.as_ref()),
                        };
                        if matches!(prediction_source, Some(WarmStartPredictionSource::Ift))
                            && predicted.0.len() == converged_original.len()
                        {
                            let mut diff_sq = 0.0_f64;
                            let mut conv_sq = 0.0_f64;
                            for (p_val, c_val) in predicted.0.iter().zip(converged_original.iter())
                            {
                                let d = c_val - p_val;
                                diff_sq += d * d;
                                conv_sq += c_val * c_val;
                            }
                            let conv_norm = conv_sq.sqrt();
                            let pred_residual = diff_sq.sqrt();
                            let quality = pred_residual / (1.0 + conv_norm);
                            if quality.is_finite() && quality >= 0.0 {
                                let last_residual_bits =
                                    self.last_ift_prediction_residual.load(Ordering::Relaxed);
                                let r = f64::from_bits(last_residual_bits);
                                let last_residual = if r.is_finite() && r >= 0.0 {
                                    Some(r)
                                } else {
                                    None
                                };
                                let current_cap =
                                    self.ift_quality_step_cap(adaptive_ift_max_drho(last_residual));
                                let cap_predicted = self
                                    .record_ift_prediction_quality(quality, current_cap)
                                    .unwrap_or(current_cap);
                                log::info!(
                                    "[IFT-QUALITY] quality={:.3e} ift={:.3e} pred_residual={:.3e} cap_predicted={:.3e}",
                                    quality,
                                    current_cap,
                                    pred_residual,
                                    cap_predicted,
                                );
                                self.last_ift_prediction_residual
                                    .store(quality.to_bits(), Ordering::Relaxed);
                            }
                        }
                    }
                    self.updatewarm_start_from(pirls_result.as_ref());
                    // Record the ρ that produced this β so the next call
                    // can extrapolate. Only meaningful after a successful
                    // solve (updatewarm_start_from clears history on
                    // failure); recording here is harmless on failure
                    // because the warm_start_beta will be None.
                    self.record_warm_start_rho(rho);
                    // Inner-PIRLS feedback signal for the adaptive cap
                    // schedule: record actual iteration count and
                    // converged flag. Only written for non-screening
                    // solves so the screening 3-iter cap doesn't
                    // poison the schedule's history.
                    self.last_inner_iters
                        .store(pirls_result.iteration, Ordering::Relaxed);
                    self.last_inner_converged.store(true, Ordering::Relaxed);
                    // Persist the converged λ_LM so the next PIRLS call
                    // at this surface can seed near the right damping.
                    // Only stash finite-positive values; failure modes
                    // (NaN, ≤0) fall through to the cold default.
                    if pirls_result.final_lm_lambda.is_finite()
                        && pirls_result.final_lm_lambda > 0.0
                    {
                        self.last_pirls_lm_lambda
                            .store(pirls_result.final_lm_lambda.to_bits(), Ordering::Relaxed);
                    }
                    // Persist the accepted gain ratio so the cap
                    // schedule can adapt to inner Newton model fidelity
                    // alongside `last_iters` and `last_converged`. The
                    // LM accept-branch invariant (rho > 0 is necessary
                    // for acceptance) means a finite-positive value is
                    // the only "meaningful signal" outcome; anything
                    // else (None, NaN) leaves the NaN sentinel in
                    // place so the schedule falls back to the
                    // default margin.
                    if let Some(rho) = pirls_result.final_accept_rho
                        && rho.is_finite()
                        && rho >= 0.0
                    {
                        self.last_pirls_accept_rho
                            .store(rho.to_bits(), Ordering::Relaxed);
                    }
                    self.store_persistent_warm_start();
                    // Cache only if key is valid (not NaN).
                    if use_cache && let Some(key) = key_opt {
                        self.cache_manager
                            .pirls_cache
                            .write()
                            .unwrap()
                            .insert(key, Arc::new(pirls_result.compact_for_reml_cache()));
                    }
                }
                Ok(pirls_result)
            }
            pirls::PirlsStatus::Unstable => {
                // The fit was unstable. This is where we throw our specific, user-friendly error.
                // Pass the diagnostic info into the error
                Err(EstimationError::PerfectSeparationDetected {
                    iteration: pirls_result.iteration,
                    max_abs_eta: pirls_result.max_abs_eta,
                })
            }
            pirls::PirlsStatus::MaxIterationsReached
            | pirls::PirlsStatus::LmStepSearchExhausted => {
                let kind = match pirls_result.status {
                    pirls::PirlsStatus::LmStepSearchExhausted => "LM step search exhausted",
                    _ => "max iterations reached",
                };
                // DESIGN INTENT: EFS single-loop intentionally accepts partial PIRLS
                // (capped to EFS_SINGLE_LOOP_PIRLS_SWEEPS) and exposes the resulting
                // (uncertified) gradient to the EFS update. This deviates from the
                // WS3b hard rule that uncertified inner states must NOT produce
                // derivative-bearing samples; the deviation is mitigated by the
                // bias_proxy guard (max(gradient_residual, inner_residual) > 0.10
                // for K=3 consecutive iters triggers fallback to the standard
                // two-loop driver). This is the bam (Wood 2015) tradeoff: tolerate
                // uncertified inner accuracy at large-scale n to amortize per-outer-iter
                // cost, then bail to the certified path if the EFS surrogate drifts.
                if in_efs_single_loop
                    && pirls_result.deviance.is_finite()
                    && pirls_result.stable_penalty_term.is_finite()
                    && pirls_result.gradient_natural_scale.is_finite()
                    && pirls_result.lastgradient_norm.is_finite()
                    && pirls_result
                        .beta_transformed
                        .0
                        .iter()
                        .all(|v| v.is_finite())
                {
                    log::info!(
                        "[EFS-single-loop] accepted partial PIRLS sweep: {kind} \
                         (cap={} |g_beta|={:.3e} r_g={:.3e} iter={})",
                        efs_single_loop_cap.unwrap_or(outer_cap),
                        pirls_result.lastgradient_norm,
                        pirls_result.relative_gradient_norm(),
                        pirls_result.iteration,
                    );
                    self.updatewarm_start_from(pirls_result.as_ref());
                    self.record_warm_start_rho(rho);
                    self.last_inner_iters
                        .store(pirls_result.iteration, Ordering::Relaxed);
                    self.last_inner_converged.store(false, Ordering::Relaxed);
                    return Ok(pirls_result);
                }
                // Seed screening's purpose is to rank candidate seeds by an
                // approximate cost. Requiring full KKT-style convergence under
                // a 3-iteration cap would discard all informative seeds, so
                // when the partial state is finite — objective (deviance +
                // penalty), β coordinates, residual gradient norm, and the
                // gradient-natural-scale (‖score‖ + ‖Sβ‖ + ridge·‖β‖, which
                // covers the penalty and ridge contributions to H and
                // detects state-blowup pathologies upstream of an explicit
                // O(p²) Hessian-eigenspectrum walk) — we surface the result
                // as Ok and let downstream cost computation derive a finite
                // approximate score. The result is not cached, the warm
                // start is not updated, and KKT is not enforced (the actual
                // fit at full inner budget will).
                // Order the predicates so the O(1) scalar finiteness checks
                // run before the O(p) β-vector walk: short-circuit means a
                // pathological partial fit (NaN deviance, NaN
                // gradient_natural_scale, etc.) is rejected without paying
                // the linear scan over coefficients.
                if in_screening
                    && pirls_result.deviance.is_finite()
                    && pirls_result.stable_penalty_term.is_finite()
                    && pirls_result.gradient_natural_scale.is_finite()
                    && pirls_result.lastgradient_norm.is_finite()
                    && pirls_result
                        .beta_transformed
                        .0
                        .iter()
                        .all(|v| v.is_finite())
                {
                    log::debug!(
                        "[seed-screen] partial-fit accepted for ranking: {kind} (|g| {:.3e}, r_g {:.3e}, iter {})",
                        pirls_result.lastgradient_norm,
                        pirls_result.relative_gradient_norm(),
                        pirls_result.iteration
                    );
                    return Ok(pirls_result);
                }
                if in_screening {
                    log::debug!(
                        "[seed-screen] P-IRLS rejected: {kind} (gradient norm {:.3e}, iter {})",
                        pirls_result.lastgradient_norm,
                        pirls_result.iteration
                    );
                } else {
                    log::error!(
                        "P-IRLS could not certify a valid minimum: {kind} (gradient norm {:.3e}, iter {})",
                        pirls_result.lastgradient_norm,
                        pirls_result.iteration
                    );
                    // Adaptive-cap feedback: cap was hit. Geometric
                    // backoff on the next outer iter's cap. Only write
                    // outside screening so the 3-iter screening cap
                    // does not corrupt the production schedule history.
                    self.last_inner_iters
                        .store(pirls_result.iteration, Ordering::Relaxed);
                    self.last_inner_converged.store(false, Ordering::Relaxed);
                    // A failed solve says nothing useful about the
                    // right damping for the next call; reset the LM
                    // hint to "no signal" so the next solve cold-starts.
                    self.last_pirls_lm_lambda.store(0, Ordering::Relaxed);
                    // A failed solve also invalidates the IFT residual
                    // signal — there's no meaningful "predicted vs
                    // converged" datum to feed back. Reset so the next
                    // predict call uses the default 2.0 cap. NaN
                    // sentinel rather than literal 0 — see
                    // `IFT_RESIDUAL_NO_SIGNAL_BITS`.
                    self.last_ift_prediction_residual
                        .store(IFT_RESIDUAL_NO_SIGNAL_BITS, Ordering::Relaxed);
                    self.clear_ift_quality_runtime_state();
                }
                Err(EstimationError::PirlsDidNotConverge {
                    max_iterations: pirls_result.iteration,
                    last_change: pirls_result.lastgradient_norm,
                })
            }
        }
    }

    /// Stateless inner P-IRLS fit at `rho` for the smoothing-correction
    /// sigma-point cubature path.
    ///
    /// This is the cubature analogue of [`execute_pirls_if_needed`] with
    /// every form of cross-call state removed: no PIRLS-cache lookup, no
    /// cache insert, no warm-start I/O (neither read nor write), no
    /// adaptive LM-lambda hint (cold-starts at `pirls_config.initial_lm_lambda`),
    /// no screening / outer-cap reads, no adaptive-KKT outer-grad lookup,
    /// no IFT-quality / accept-rho / last-iter / last-converged feedback
    /// writes, no persistent warm-start load/store. The KKT certificate
    /// is still enforced on the converged mode because the cubature
    /// integrand consumes (H⁻¹, β̂) and downstream linear algebra
    /// (inversion, basis remap) demands a certified minimum.
    ///
    /// Two motivations:
    /// 1. The current `compute_smoothing_correction_auto` Rayon path uses
    ///    [`AtomicFlagGuard`] swaps on `pirls_cache_enabled` and
    ///    `warm_start_enabled` to disable the most contention-prone writes
    ///    on the hot path. Process-wide atomic flips serialize unrelated
    ///    REML evaluations that race the cubature window and contaminate
    ///    their feedback signals. A stateless callee lets cubature run
    ///    concurrently with other PIRLS evaluations without that coupling.
    /// 2. The forthcoming GPU sigma-point executor needs to drive many
    ///    PIRLS fits in flight on independent CUDA streams; a callee that
    ///    threads no mutable cross-call state makes those fits provably
    ///    independent and stream-safe.
    ///
    /// The math (problem, penalty config, link kind, basis, KKT,
    /// failure-classification → `EstimationError` mapping) is bit-identical
    /// to the non-screening / non-EFS branch of `execute_pirls_if_needed`.
    pub(crate) fn execute_pirls_stateless_for_cubature(
        &self,
        rho: &Array1<f64>,
    ) -> Result<Arc<PirlsResult>, EstimationError> {
        let mut pirls_config = self.config.as_pirls_config();
        pirls_config.link_kind = self.runtime_inverse_link();
        // Pin the same λ-search-frozen NB θ the outer loop converged under
        // (#1082), so the rho-uncertainty sigma-point criterion is evaluated on
        // the identical stationary surface F(ρ) = REML(ρ, θ_frozen) rather than
        // re-estimating θ at each off-trajectory σ-point.
        if pirls_config.likelihood.negbin_theta_is_estimated() {
            let frozen_bits = self.frozen_negbin_theta.load(Ordering::Relaxed);
            if frozen_bits != 0 {
                let frozen_theta = f64::from_bits(frozen_bits);
                if frozen_theta.is_finite() && frozen_theta > 0.0 {
                    pirls_config.likelihood = pirls_config
                        .likelihood
                        .clone()
                        .with_negbin_theta_frozen_for_search(frozen_theta);
                }
            }
        }
        // Pin the same λ-search-frozen Tweedie φ the outer loop converged under
        // (#1477), so the rho-uncertainty sigma-point criterion is evaluated on
        // the identical stationary surface F(ρ) = REML(ρ, φ_frozen) rather than
        // re-estimating φ at each off-trajectory σ-point.
        if pirls_config.likelihood.tweedie_phi_is_estimated() {
            let frozen_bits = self.frozen_tweedie_phi.load(Ordering::Relaxed);
            if frozen_bits != 0 {
                let frozen_phi = f64::from_bits(frozen_bits);
                if frozen_phi.is_finite() && frozen_phi > 0.0 {
                    pirls_config.likelihood = pirls_config
                        .likelihood
                        .clone()
                        .with_tweedie_phi_frozen_for_search(frozen_phi);
                }
            }
        }
        // Pin the same λ-search-frozen Gamma shape the outer loop converged under
        // (#1074), so the rho-uncertainty sigma-point criterion is evaluated on
        // the identical stationary surface F(ρ) = REML(ρ, k_frozen) rather than
        // re-estimating `k` at each off-trajectory σ-point.
        if pirls_config.likelihood.scale.gamma_shape_is_estimated() {
            let frozen_bits = self.frozen_gamma_shape.load(Ordering::Relaxed);
            if frozen_bits != 0 {
                let frozen_shape = f64::from_bits(frozen_bits);
                if frozen_shape.is_finite() && frozen_shape > 0.0 {
                    pirls_config.likelihood = pirls_config
                        .likelihood
                        .clone()
                        .with_gamma_shape_frozen_for_search(frozen_shape);
                }
            }
        }

        // Gaussian + Identity outer REML reuses a precomputed XᵀWX and
        // XᵀW(y − offset) across every inner solve; for other families /
        // links this returns None and the inner solver falls back to the
        // streaming GEMM. Reading this cache is non-mutating (the cache
        // belongs to the surface, not to a particular outer iteration), so
        // it is safe to reuse here for parity with `execute_pirls_if_needed`.
        let cache_handle = self.gaussian_fixed_cache_if_eligible();
        let glm_first_step_handle = self.glm_first_step_gram();
        let problem = pirls::PirlsProblem {
            x: &self.x,
            offset: self.offset.view(),
            y: self.y,
            priorweights: self.weights,
            covariate_se: None,
            gaussian_fixed_cache: cache_handle.as_deref(),
            glm_first_step_gram: glm_first_step_handle.as_deref(),
        };
        let penalty = pirls::PenaltyConfig {
            canonical_penalties: &self.canonical_penalties,
            balanced_penalty_root: Some(&self.balanced_penalty_root),
            reparam_invariant: Some(&self.reparam_invariant),
            p: self.p,
            coefficient_lower_bounds: self.coefficient_lower_bounds.as_ref(),
            linear_constraints_original: self.linear_constraints.as_ref(),
            penalty_shrinkage_floor: self.penalty_shrinkage_floor,
            kronecker_factored: self.kronecker_factored.as_ref(),
        };

        let pirls_start = std::time::Instant::now();
        let result = pirls::fit_model_for_fixed_rho_with_adaptive_kkt(
            LogSmoothingParamsView::new(rho.view()),
            problem,
            penalty,
            &pirls_config,
            // No warm start: sigma points are off the outer trajectory and
            // a stale warm start would couple parallel sigma fits.
            None,
            // No adaptive-KKT outer-grad lookup: the outer-grad state is
            // owned by the production trajectory and the sigma points are
            // not on it.
            None,
            // Sigma-point cubature eval: Gamma scale refinement stays OFF (only
            // the final reported fit refines — see #678).
            false,
        );
        let pirls_elapsed = pirls_start.elapsed();
        if let Ok((ref res, ref wm)) = result {
            log::info!(
                "[STAGE] sigma-cubature pirls solve iters={} status={:?} max_eta={:.1} elapsed={:.3}s",
                wm.iterations,
                res.status,
                res.max_abs_eta,
                pirls_elapsed.as_secs_f64(),
            );
        }
        let (pirls_result, _) = result?;
        let pirls_result = Arc::new(pirls_result);

        // Enforce KKT on the converged mode; the cubature integrand
        // consumes (H⁻¹, β̂) and demands a certified minimum.
        self.enforce_constraint_kkt(pirls_result.as_ref())?;

        match pirls_result.status {
            pirls::PirlsStatus::Converged | pirls::PirlsStatus::StalledAtValidMinimum => {
                Ok(pirls_result)
            }
            pirls::PirlsStatus::Unstable => Err(EstimationError::PerfectSeparationDetected {
                iteration: pirls_result.iteration,
                max_abs_eta: pirls_result.max_abs_eta,
            }),
            pirls::PirlsStatus::MaxIterationsReached
            | pirls::PirlsStatus::LmStepSearchExhausted => {
                Err(EstimationError::PirlsDidNotConverge {
                    max_iterations: pirls_result.iteration,
                    last_change: pirls_result.lastgradient_norm,
                })
            }
        }
    }
}

/// Coarse inner-PIRLS convergence tolerance used ONLY while the outer
/// seed-screening prepass ranks candidate ρ (`in_screening == true`). The
/// prepass needs a ρ-comparable penalized-deviance proxy, not a converged fit:
/// a coarse `1e-3` inner solve resolves the basin ranking at a small fraction
/// of the Newton iterations a tight production tolerance (`~1e-8`) demands,
/// which is the dominant prepass cost at large `n` (#1033/#1575/#1082) on the
/// two paths the screening iteration cap does NOT bound — the inequality-
/// constrained active-set solve (#1380, cap deliberately disabled) and the
/// cascade's uncapped final stage. It is applied with `max`, so it can only
/// loosen a tighter production tolerance and never tightens a coarser one, and
/// it is scoped to screening solves alone: the winning seed's real fit and
/// every multistart full solve run with `in_screening == false`, so the
/// converged REML/LAML optimum and its bit-results are unchanged.
pub(crate) const SEED_SCREENING_INNER_CONVERGENCE_TOLERANCE: f64 = 1e-3;

/// Default cap on |Δρ_k| beyond which the IFT linear predictor rejects.
/// Δρ = log(λ_new / λ_old); 2.0 corresponds to a 7.4× change in λ along
/// any single penalty direction — well outside the regime where the local
/// first-order Jacobian dβ/dρ is faithful. The cap is now ADAPTIVE: see
/// `adaptive_ift_max_drho` for the empirical-quality-driven loosening /
/// tightening policy. This default is used when no IFT-quality history
/// is available yet (the first PIRLS solve at a fresh surface).
pub(crate) const IFT_WARM_START_DEFAULT_MAX_DRHO: f64 = 2.0;

/// Shared relative-residual tier breakpoints for the warm-start linear
/// predictors. `r = ‖β_converged − β_predicted‖ / ‖β_converged‖` from the
/// previous IFT prediction classifies the local linearization quality into
/// five bands; both `adaptive_ift_max_drho` and `adaptive_tangent_alpha_cap`
/// key off the SAME breakpoints so their caps move in lockstep (the two
/// predictors share one quality signal). One step per decade of residual keeps
/// the policy stable under noise.
pub(crate) const IFT_RESIDUAL_TIER_EXCELLENT: f64 = 0.01;

pub(crate) const IFT_RESIDUAL_TIER_VERY_GOOD: f64 = 0.05;

pub(crate) const IFT_RESIDUAL_TIER_OK: f64 = 0.20;

pub(crate) const IFT_RESIDUAL_TIER_MARGINAL: f64 = 0.50;

/// Adaptive |Δρ| cap for the IFT predictor, driven by the residual of
/// the previous IFT prediction (see `last_ift_prediction_residual`).
///
/// `last_residual = ‖β_converged − β_predicted‖ / ‖β_converged‖`:
/// - `r < 0.01` → linearization was excellent; allow Δρ up to **4.0**
///   (54× λ-step). At this regime the local Jacobian is faithful and
///   tightening to 2.0 leaves performance on the table at large-scale Δρ
///   magnitudes (where outer optimizers commonly take ρ-jumps of ~1).
/// - `0.01 ≤ r < 0.05` → very good; modest expansion to **3.0**.
/// - `0.05 ≤ r < 0.20` → ok; default **2.0** (the original constant).
/// - `0.20 ≤ r < 0.50` → marginal; tighten to **1.0** (2.7× λ-step).
/// - `r ≥ 0.50` → poor; tighten to **0.5** (1.6× λ-step). The IFT
///   prediction is not paying off at large-scale Δρ; fall back to flat
///   warm-start (β_cur) for any non-trivial outer step.
/// - `None` → no signal yet (first PIRLS solve at this surface) → default 2.0.
///
/// The thresholds are deliberately one-step-per-decade-of-residual so
/// the policy remains stable under noise; small fluctuations in
/// `last_residual` don't whipsaw the cap.
pub(crate) fn adaptive_ift_max_drho(last_residual: Option<f64>) -> f64 {
    let Some(r) = last_residual else {
        return IFT_WARM_START_DEFAULT_MAX_DRHO;
    };
    // Defensive: NaN or negative residuals indicate corrupted state
    // (bit-pattern unpacking from the atomic encountered an unwritten
    // slot, or norm-ratio division produced a non-physical value).
    // Fall back to the documented default rather than committing to
    // a tier based on garbage. Note: INFINITY is not rejected here —
    // it represents a catastrophic prediction (predicted_norm finite
    // but ‖Δβ‖ overflowed), so falling through to the catch-all 0.5
    // (tightest cap) is the right policy.
    if r.is_nan() || r < 0.0 {
        return IFT_WARM_START_DEFAULT_MAX_DRHO;
    }
    match r {
        r if r < IFT_RESIDUAL_TIER_EXCELLENT => 4.0,
        r if r < IFT_RESIDUAL_TIER_VERY_GOOD => 3.0,
        r if r < IFT_RESIDUAL_TIER_OK => 2.0,
        r if r < IFT_RESIDUAL_TIER_MARGINAL => 1.0,
        _ => 0.5,
    }
}

/// Default upper cap on the tangent-line predictor's α (extrapolation
/// fraction beyond the previous ρ-step). 1.5 means "we permit at most
/// 50% extrapolation past the last step length"; the original hardcoded
/// constant from commit dcacf9ee. Used when no IFT-quality history is
/// available (typically right after the first successful PIRLS solve
/// at a fresh surface — IFT cache exists but tangent-line history
/// pair is being assembled). Adaptive policy in
/// `adaptive_tangent_alpha_cap` adjusts this based on the IFT
/// residual signal when present.
pub(crate) const TANGENT_ALPHA_DEFAULT_CAP: f64 = 1.5;

/// Adaptive α-cap for the tangent-line predictor, sharing the IFT
/// residual signal as a proxy for "how trustworthy is the local linear
/// approximation". The tangent line IS a local linear approximation
/// (just along the previous ρ-step direction rather than the IFT
/// Jacobian's full direction), so the same residual that gates
/// `adaptive_ift_max_drho` informs the right cap here too:
///
/// - `r < 0.01`  → linearization excellent → α_cap = 2.0 (1.5×
///                  the default; permits a full step beyond the
///                  previous one)
/// - `r < 0.05`  → very good → α_cap = 1.75
/// - `r < 0.20`  → ok → α_cap = 1.5 (the original constant)
/// - `r < 0.50`  → marginal → α_cap = 1.0 (no extrapolation past
///                  the previous step length)
/// - `r ≥ 0.50`  → poor → α_cap = 0.5 (only HALF the previous step;
///                  the linear approximation has been shown to
///                  collapse toward flat warm-start at this surface)
/// - `None` (no signal yet) → default 1.5
///
/// Same tier-stable, monotone-non-increasing-in-residual shape as
/// `adaptive_ift_max_drho`; the two predictors share a single quality
/// signal so their caps move together.
pub(crate) fn adaptive_tangent_alpha_cap(last_residual: Option<f64>) -> f64 {
    let Some(r) = last_residual else {
        return TANGENT_ALPHA_DEFAULT_CAP;
    };
    if r.is_nan() || r < 0.0 {
        return TANGENT_ALPHA_DEFAULT_CAP;
    }
    match r {
        r if r < IFT_RESIDUAL_TIER_EXCELLENT => 2.0,
        r if r < IFT_RESIDUAL_TIER_VERY_GOOD => 1.75,
        r if r < IFT_RESIDUAL_TIER_OK => 1.5,
        r if r < IFT_RESIDUAL_TIER_MARGINAL => 1.0,
        _ => 0.5,
    }
}

/// What the IFT predictor's inner computation actually did with the
/// β it returned. Surfaced by the inner predictor so callers don't
/// need to re-derive noop-ness via O(p) array comparison against
/// the cached β. Production callers map this onto the higher-level
/// `WarmStartPredictionSource` enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum IftPredictionOutcome {
    /// The predictor computed a real Δβ (`any_active=true`) and the
    /// returned Coefficients differs from the cached β.
    Predicted,
    /// All Δρ_k were below the per-component eps floor, so the
    /// predictor returned the cached β unchanged. The
    /// [IFT-NOOP] marker has already been emitted at the inner
    /// predictor's noop branch; callers should NOT re-emit a
    /// quality probe on the returned β.
    Noop,
}

/// Tag identifying which branch of the warm-start linear-predictor
/// stack actually produced the β returned by
/// `predict_warm_start_beta_with_source`. Used by the caller in
/// `execute_pirls_if_needed` to emit per-predictor quality markers
/// (`[IFT-QUALITY]` vs `[TANGENT-QUALITY]`) so the bench runner's
/// residual percentile aggregations are correctly attributed instead
/// of mistakenly bucketing tangent-line predictions into IFT stats.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WarmStartPredictionSource {
    /// The IFT predictor produced a non-identity β (the cache was
    /// populated and Δρ stayed within the adaptive |Δρ| cap).
    Ift,
    /// The tangent-line predictor produced a non-identity β (the IFT
    /// predictor returned None for non-cache reasons, so the
    /// fallback chain reached tangent-line and it accepted).
    TangentLine,
    /// Both predictors fell through (no history, dim mismatch,
    /// large Δρ, non-finite intermediate, etc.) — the returned
    /// β is the cached `warm_start_beta` unchanged. Same outcome
    /// as the original "use last β as-is" warm start.
    Flat,
}

/// Below this magnitude in any Δρ_k, the per-component IFT contribution is
/// numerically negligible and skipped (saves one back-solve per inactive
/// component).
pub(crate) const IFT_WARM_START_DRHO_EPS: f64 = 1e-12;

/// Bit-pattern sentinel used by `RemlObjectiveState::last_ift_prediction_residual`
/// to encode "no signal yet" unambiguously. Decodes via `f64::from_bits`
/// to a quiet NaN — readers detect the sentinel via NaN's
/// self-inequality (NaN.is_finite() == false), so no value-comparison
/// trickery is needed.
///
/// The original `0` sentinel collided with `f64::to_bits(0.0) == 0`,
/// which would have made a true residual of exactly 0 (mathematically
/// possible if β_predicted matched β_converged element-wise) silently
/// indistinguishable from "predictor never reported". Any standard
/// quiet-NaN bit pattern works; we use the one Rust's `f64::NAN`
/// canonicalizes to (mantissa MSB = 1, sign = 0, exponent = all 1s).
///
/// `pub(crate)` so the bridge's `InnerProgressFeedback::snapshot` (in
/// `crate::rho_optimizer`) can reference the same constant —
/// both ends of this atomic must use the same sentinel discipline,
/// otherwise a residual-of-zero round-trips correctly through the
/// writer but the reader treats it as "no signal" and silently
/// drops the adaptive cap-margin signal.
pub(crate) const IFT_RESIDUAL_NO_SIGNAL_BITS: u64 = 0x7ff8_0000_0000_0000;

/// Adaptive clamp for the `initial_lm_lambda` warm-start hint passed
/// from `execute_pirls_if_needed` into PIRLS. Selects one of three
/// principled regimes based on the previous PIRLS solve's halving
/// history:
///
/// * Newton-friendly  (last_converged AND last_iters in 1..=2):
///   well-conditioned local geometry; allow the cached λ down to the
///   LM-internal floor. Range: [1e-9, 1e-3].
/// * Hard fit  (NOT last_converged OR last_iters ≥ 10):
///   previous solve hit the cap or needed many iters; preserve the
///   heavy-damping signal up to gradient-descent. Range: [1e-3, 1.0].
/// * Default  (everything else, incl. unset feedback):
///   matches the historical static `[1e-6, 1e-3]` clamp.
///
/// Returns `None` for non-finite or non-positive `cached_lambda`,
/// matching the historical contract that pathological cache entries
/// fall through to the cold default. Also returns `None` if the
/// caller hasn't recorded any feedback yet (`last_iters == 0` AND
/// `!last_converged`) — that combination signals "no signal" via
/// `clear_warm_start_adaptive_signals`, and the cold default is the
/// safer choice than seeding from a stale-but-finite cache slot.
pub(crate) fn adaptive_lm_lambda_hint(
    cached_lambda: f64,
    last_iters: usize,
    last_converged: bool,
) -> Option<f64> {
    // Iteration-count boundaries that classify the previous PIRLS solve's
    // conditioning regime.
    const NEWTON_FRIENDLY_MAX_ITERS: usize = 2;
    const HARD_FIT_MIN_ITERS: usize = 10;
    // Per-regime adaptive clamp bands for the cached LM damping hint (see the
    // doc comment): Newton-friendly relaxes down to the LM-internal floor, a
    // hard fit preserves the heavy-damping signal up to gradient-descent, and
    // the default reproduces the historical static `[1e-6, 1e-3]` clamp.
    const NEWTON_LAMBDA_FLOOR: f64 = 1e-9;
    const NEWTON_LAMBDA_CEILING: f64 = 1e-3;
    const HARD_FIT_LAMBDA_FLOOR: f64 = 1e-3;
    const HARD_FIT_LAMBDA_CEILING: f64 = 1.0;
    const DEFAULT_LAMBDA_FLOOR: f64 = 1e-6;
    const DEFAULT_LAMBDA_CEILING: f64 = 1e-3;
    if !cached_lambda.is_finite() || cached_lambda <= 0.0 {
        return None;
    }
    if last_iters == 0 && !last_converged {
        return None;
    }
    let (floor, ceiling) =
        if last_converged && (1..=NEWTON_FRIENDLY_MAX_ITERS).contains(&last_iters) {
            (NEWTON_LAMBDA_FLOOR, NEWTON_LAMBDA_CEILING)
        } else if !last_converged || last_iters >= HARD_FIT_MIN_ITERS {
            (HARD_FIT_LAMBDA_FLOOR, HARD_FIT_LAMBDA_CEILING)
        } else {
            (DEFAULT_LAMBDA_FLOOR, DEFAULT_LAMBDA_CEILING)
        };
    Some(cached_lambda.clamp(floor, ceiling))
}

pub(crate) fn predict_warm_start_beta_ift_inner_with_outcome(
    cache: &super::IftWarmStartCache,
    canonical_penalties: &[gam_terms::construction::CanonicalPenalty],
    new_rho: &Array1<f64>,
    p: usize,
    last_ift_residual: Option<f64>,
    max_drho_cap_override: Option<f64>,
    factor_override: Option<&dyn gam_linalg::matrix::FactorizedSystem>,
) -> Option<(Coefficients, IftPredictionOutcome)> {
    // Cache populated but ρ not yet stamped (happens between
    // updatewarm_start_from and record_warm_start_rho on the first solve).
    // This is the *expected* race window during normal operation — silent
    // None is correct; emitting a marker would just be noise on every
    // first call.
    if cache.rho.is_empty() {
        return None;
    }
    let k = cache.rho.len();
    // Dimension consistency check. A mismatch here is a state-machine
    // bug signal — the cache was populated under a different penalty
    // layout than the predictor is being asked to use. Emit a structured
    // [IFT-REJECTED] marker so the bench runner counts these. If the
    // count is ever non-zero in production, we have a real inconsistency
    // in the cache invalidation chain; reset_surface, link change, and
    // similar should have wiped the cache before the layout shifted.
    if new_rho.len() != k {
        log::info!(
            "[IFT-REJECTED] reason=rho_dim_mismatch new_rho_dim={} cache_rho_dim={}",
            new_rho.len(),
            k,
        );
        return None;
    }
    if canonical_penalties.len() != k {
        log::info!(
            "[IFT-REJECTED] reason=penalty_dim_mismatch penalties_dim={} cache_rho_dim={}",
            canonical_penalties.len(),
            k,
        );
        return None;
    }
    if cache.beta_original.len() != p {
        log::info!(
            "[IFT-REJECTED] reason=beta_dim_mismatch cache_beta_dim={} expected_p={}",
            cache.beta_original.len(),
            p,
        );
        return None;
    }

    // Δρ guard: reject in toto if any single component exceeds the cap.
    // We do not partially clip: the directions we drop matter just as much
    // as the directions we keep, and a clipped predictor is harder to
    // reason about than a clean fallback.
    let mut max_abs_drho = 0.0_f64;
    let upper_bounds = latest_outer_rho_upper_bounds_for_ift();
    let upper_active = |idx: usize| -> bool {
        let upper = upper_bounds
            .as_ref()
            .and_then(|bounds| bounds.get(idx))
            .copied()
            .unwrap_or(RHO_BOUND);
        upper.is_finite() && cache.rho[idx] >= upper - 1.0e-8
    };

    let drho: Array1<f64> = (0..k)
        .map(|i| {
            if upper_active(i) {
                return 0.0;
            }
            let d = new_rho[i] - cache.rho[i];
            if !d.is_finite() {
                return f64::INFINITY;
            }
            if d.abs() > max_abs_drho {
                max_abs_drho = d.abs();
            }
            d
        })
        .collect();
    let max_drho_cap = max_drho_cap_override
        .filter(|cap| cap.is_finite() && *cap > 0.0)
        .unwrap_or_else(|| adaptive_ift_max_drho(last_ift_residual));
    if !max_abs_drho.is_finite() || max_abs_drho > max_drho_cap {
        // Emit a structured reject marker so the bench runner can
        // count predictor rejections alongside accepts (the
        // `[IFT-QUALITY]` markers count only the accepts). The
        // accept/reject ratio at large scale tells us how often
        // the warm-start machinery is actually delivering, vs
        // falling through to flat warm-start at every accepted
        // outer iter.
        log::info!(
            "[IFT-REJECTED] reason=large_drho max_drho={:.3e} cap={:.3e} drho_dim={}",
            max_abs_drho,
            max_drho_cap,
            k,
        );
        return None;
    }

    // Build Σ_k Δρ_k · e^{ρ_cur_k} · S_k · (β_cur-μ_k) in the ORIGINAL basis.
    // Aggregating into a single rhs lets us factor H once and back-solve
    // exactly once, instead of k times. Mathematically:
    //   Σ_k Δρ_k H^{-1} v_k = H^{-1} (Σ_k Δρ_k v_k)
    // since H^{-1} is linear in its rhs.
    let beta_cur = &cache.beta_original;
    let mut rhs_original = Array1::<f64>::zeros(p);
    let mut any_active = false;
    // Use the precomputed `S_k · (β_cur-μ_k)` blocks (cache write-time
    // hook) when available. Falls back to recomputing the local
    // mat-vec when the cache predates this commit's writer hook
    // (None) or the precomputation length disagrees with the
    // current canonical_penalties (defensive: guards against
    // racing replace events).
    let precomputed_ok = match &cache.lambda_s_beta_blocks {
        Some(blocks) => blocks.len() == canonical_penalties.len(),
        None => false,
    };
    for (idx, cp) in canonical_penalties.iter().enumerate() {
        let dr = drho[idx];
        if dr.abs() <= IFT_WARM_START_DRHO_EPS {
            continue;
        }
        any_active = true;
        let r = &cp.col_range;
        // S_k · (β_cur-μ_k) is block-local: only β_cur[r] and μ_k contribute,
        // and the result lives in r as well.
        let scale = dr * cache.rho[idx].exp();
        let mut rhs_slice = rhs_original.slice_mut(s![r.start..r.end]);
        if precomputed_ok {
            // SAFE: precomputed_ok guarantees Some + length match.
            let blocks = cache.lambda_s_beta_blocks.as_ref().unwrap();
            let sb_block = &blocks[idx];
            // Defensive: a fresh canonical_penalty may have a different
            // col_range size than the cache's precomputation captured.
            // If so, fall through to the recompute path for this
            // index. (Should not happen in practice — the cache is
            // invalidated whenever the penalty layout changes.)
            if sb_block.len() == rhs_slice.len() {
                for (target, src) in rhs_slice.iter_mut().zip(sb_block.iter()) {
                    *target += scale * *src;
                }
                continue;
            }
        }
        let beta_block = beta_cur.slice(s![r.start..r.end]);
        let centered = &beta_block - &cp.prior_mean;
        let sb_block = cp.local.dot(&centered);
        for (target, src) in rhs_slice.iter_mut().zip(sb_block.iter()) {
            *target += scale * *src;
        }
    }

    if !any_active {
        // All Δρ are below the eps floor — predictor reduces to identity.
        // Returning the cached β here is equivalent to the flat warm-start
        // and slightly better than the tangent-line predictor (which the
        // caller would otherwise try), since the IFT cache reflects a
        // genuinely converged solve.
        //
        // Emit a structured no-op marker so the bench runner's accept /
        // reject ratio (commit fec27c97) can distinguish:
        //   accept  → predictor produced a non-identity β prediction
        //   reject  → predictor fell through to caller's tangent-line / flat
        //   noop    → predictor returned identity (this branch). The outer
        //             made an effectively-zero ρ-step; warm-start was free.
        // Without this marker, "noop" calls would inflate the accept count
        // because the predictor returns Some(β) — masking how often the
        // outer is actually exercising the linearization.
        log::info!(
            "[IFT-NOOP] reason=all_drho_below_eps max_drho={:.3e} drho_dim={}",
            max_abs_drho,
            k,
        );
        return Some((
            Coefficients::new(beta_cur.clone()),
            IftPredictionOutcome::Noop,
        ));
    }

    if !rhs_original.iter().all(|v| v.is_finite()) {
        log::info!(
            "[IFT-REJECTED] reason=non_finite_rhs max_drho={:.3e} drho_dim={}",
            max_abs_drho,
            k,
        );
        return None;
    }

    // Convert rhs to the basis H_pen lives in, solve, convert back.
    let solve_in_original = cache.frame_was_original;
    let rhs_in_h_basis = if solve_in_original {
        rhs_original
    } else {
        // rhs_tfd = qs^T · rhs_original. Dimension check: qs is p×p.
        if cache.qs.nrows() != p || cache.qs.ncols() != p {
            log::info!(
                "[IFT-REJECTED] reason=qs_dim_mismatch qs_dim={}x{} expected_p={}",
                cache.qs.nrows(),
                cache.qs.ncols(),
                p,
            );
            return None;
        }
        cache.qs.t().dot(&rhs_original)
    };

    // Use the caller's pre-built factor when present (the method-level
    // wrapper threads in a cached Cholesky); otherwise factorize inline.
    let owned_factor;
    let factor_ref: &dyn gam_linalg::matrix::FactorizedSystem = match factor_override {
        Some(f) => f,
        None => {
            owned_factor = match cache.penalized_hessian_transformed.factorize() {
                Ok(f) => f,
                Err(_) => {
                    log::info!(
                        "[IFT-REJECTED] reason=hessian_factorize_failed max_drho={:.3e} drho_dim={}",
                        max_abs_drho,
                        k,
                    );
                    return None;
                }
            };
            owned_factor.as_ref()
        }
    };
    let solution_in_h_basis = match factor_ref.solve(&rhs_in_h_basis) {
        Ok(u) => u,
        Err(_) => {
            log::info!(
                "[IFT-REJECTED] reason=hessian_solve_failed max_drho={:.3e} drho_dim={}",
                max_abs_drho,
                k,
            );
            return None;
        }
    };
    let solution_original = if solve_in_original {
        solution_in_h_basis
    } else {
        cache.qs.dot(&solution_in_h_basis)
    };

    if !solution_original.iter().all(|v| v.is_finite()) {
        log::info!(
            "[IFT-REJECTED] reason=non_finite_solution max_drho={:.3e} drho_dim={}",
            max_abs_drho,
            k,
        );
        return None;
    }

    // β_predict = β_cur − H^{-1} · (Σ_k Δρ_k e^{ρ_k} S_k(β_cur-μ_k)).
    // (The sign convention: dβ/dρ_k = −H^{-1}(e^{ρ_k}S_k(β-μ_k)), so
    //  Δβ = −Σ_k Δρ_k H^{-1}(e^{ρ_k}S_k(β-μ_k)) = −solution_original.)
    let mut predicted = beta_cur.clone();
    for (target, &correction) in predicted.iter_mut().zip(solution_original.iter()) {
        *target -= correction;
    }
    if !predicted.iter().all(|v| v.is_finite()) {
        log::info!(
            "[IFT-REJECTED] reason=non_finite_predicted max_drho={:.3e} drho_dim={}",
            max_abs_drho,
            k,
        );
        return None;
    }
    log::debug!(
        "[warm-start] IFT prediction: max|Δρ|={:.3e}, ‖rhs‖={:.3e}, ‖Δβ‖={:.3e}",
        max_abs_drho,
        rhs_in_h_basis.dot(&rhs_in_h_basis).sqrt(),
        solution_original.dot(&solution_original).sqrt(),
    );
    Some((
        Coefficients::new(predicted),
        IftPredictionOutcome::Predicted,
    ))
}

pub(crate) fn predict_warm_start_beta_ift_from_mode_response_cols(
    cache: &super::IftWarmStartCache,
    new_rho: &Array1<f64>,
    p: usize,
    last_ift_residual: Option<f64>,
    max_drho_cap_override: Option<f64>,
    rho_mode_response_cols: &Array2<f64>,
) -> Option<(Coefficients, IftPredictionOutcome)> {
    if cache.rho.is_empty() {
        return None;
    }
    let k = cache.rho.len();
    if new_rho.len() != k {
        log::info!(
            "[IFT-REJECTED] reason=rho_dim_mismatch new_rho_dim={} cache_rho_dim={}",
            new_rho.len(),
            k,
        );
        return None;
    }
    if cache.beta_original.len() != p {
        log::info!(
            "[IFT-REJECTED] reason=beta_dim_mismatch cache_beta_dim={} expected_p={}",
            cache.beta_original.len(),
            p,
        );
        return None;
    }
    if rho_mode_response_cols.nrows() != p || rho_mode_response_cols.ncols() != k {
        return None;
    }

    let mut max_abs_drho = 0.0_f64;
    let upper_bounds = latest_outer_rho_upper_bounds_for_ift();
    let upper_active = |idx: usize| -> bool {
        let upper = upper_bounds
            .as_ref()
            .and_then(|bounds| bounds.get(idx))
            .copied()
            .unwrap_or(RHO_BOUND);
        upper.is_finite() && cache.rho[idx] >= upper - 1.0e-8
    };

    let drho: Array1<f64> = (0..k)
        .map(|i| {
            if upper_active(i) {
                return 0.0;
            }
            let d = new_rho[i] - cache.rho[i];
            if !d.is_finite() {
                return f64::INFINITY;
            }
            if d.abs() > max_abs_drho {
                max_abs_drho = d.abs();
            }
            d
        })
        .collect();
    let max_drho_cap = max_drho_cap_override
        .filter(|cap| cap.is_finite() && *cap > 0.0)
        .unwrap_or_else(|| adaptive_ift_max_drho(last_ift_residual));
    if !max_abs_drho.is_finite() || max_abs_drho > max_drho_cap {
        log::info!(
            "[IFT-REJECTED] reason=large_drho max_drho={:.3e} cap={:.3e} drho_dim={}",
            max_abs_drho,
            max_drho_cap,
            k,
        );
        return None;
    }

    if drho.iter().all(|d| d.abs() <= IFT_WARM_START_DRHO_EPS) {
        log::info!(
            "[IFT-NOOP] reason=all_drho_below_eps max_drho={:.3e} drho_dim={}",
            max_abs_drho,
            k,
        );
        return Some((
            Coefficients::new(cache.beta_original.clone()),
            IftPredictionOutcome::Noop,
        ));
    }

    let solution_original = rho_mode_response_cols.dot(&drho);
    if !solution_original.iter().all(|v| v.is_finite()) {
        return None;
    }

    let mut predicted = cache.beta_original.clone();
    for (target, &correction) in predicted.iter_mut().zip(solution_original.iter()) {
        *target -= correction;
    }
    if !predicted.iter().all(|v| v.is_finite()) {
        log::info!(
            "[IFT-REJECTED] reason=non_finite_predicted max_drho={:.3e} drho_dim={}",
            max_abs_drho,
            k,
        );
        return None;
    }
    log::debug!(
        "[warm-start] IFT prediction reused mode responses: max|Δρ|={:.3e}, ‖Δβ‖={:.3e}",
        max_abs_drho,
        solution_original.dot(&solution_original).sqrt(),
    );
    Some((
        Coefficients::new(predicted),
        IftPredictionOutcome::Predicted,
    ))
}

#[cfg(test)]
mod firth_hessian_direction_reuse_tests {
    use super::*;
    use gam_problem::{InverseLink, StandardLink};
    use gam_terms::construction::CanonicalPenalty;
    use ndarray::{Array1, Array2};

    // Small deterministic logit design used to exercise the Firth outer-Hessian
    // direction path that #1575 made redundant-free.
    fn synthetic_logit_setup() -> (
        Array2<f64>,
        Array1<f64>,
        super::super::FirthDenseOperator,
        Vec<CanonicalPenalty>,
        Vec<f64>,
    ) {
        let n = 24usize;
        let p = 6usize;
        let mut x = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let t = (i as f64) / (n as f64 - 1.0);
            x[[i, 0]] = 1.0;
            x[[i, 1]] = t;
            x[[i, 2]] = t * t;
            x[[i, 3]] = (3.0 * t).sin();
            x[[i, 4]] = (2.0 * t).cos();
            x[[i, 5]] = (t - 0.5).abs();
        }
        // A bounded β keeps η finite so the logit weights are strictly positive
        // (the regime the Firth reduced-Fisher operator is built for).
        let beta = Array1::from(vec![0.2_f64, -0.4, 0.3, 0.1, -0.2, 0.15]);
        let eta = x.dot(&beta);
        let op = super::super::FirthDenseOperator::build_for_link(
            &InverseLink::Standard(StandardLink::Logit),
            &x,
            &eta,
        )
        .expect("firth operator");

        // Two block-local canonical penalties (k = 2) over disjoint coordinate
        // ranges, mirroring how distinct smooths map onto coefficient blocks.
        let mut root_a = Array2::<f64>::zeros((2, p));
        root_a[[0, 1]] = 1.0;
        root_a[[1, 2]] = 1.0;
        let mut root_b = Array2::<f64>::zeros((2, p));
        root_b[[0, 3]] = 1.0;
        root_b[[1, 4]] = 1.0;
        let penalties = vec![
            CanonicalPenalty::from_dense_root(root_a, p),
            CanonicalPenalty::from_dense_root(root_b, p),
        ];
        let lambdas = vec![0.7_f64, 1.3_f64];
        (x, beta, op, penalties, lambdas)
    }

    // The #1575 fix replaces per-(i,j)-pair rebuilds of the per-penalty Firth
    // directions with a single precomputed reuse. This locks in the invariant
    // that makes the substitution exact: a reused FirthDirection feeds
    // hphisecond_direction_apply to the same bits as a freshly rebuilt one.
    #[test]
    fn reused_firth_direction_matches_freshly_rebuilt_second_derivative() {
        let (x, _beta, op, _pen, _lam) = synthetic_logit_setup();
        let p = x.ncols();
        // Two distinct β-directions, exactly as eta_i[i]/eta_i[j] would be.
        let deta_i = x.dot(&Array1::from(vec![0.5, -0.1, 0.2, 0.0, 0.3, -0.4]));
        let deta_j = x.dot(&Array1::from(vec![-0.2, 0.4, -0.3, 0.1, 0.0, 0.25]));

        // Reuse path (what the fixed loop does): build each direction once.
        let dir_i_once = op.direction_from_deta(deta_i.clone());
        let dir_j_once = op.direction_from_deta(deta_j.clone());
        let eye = Array2::<f64>::eye(p);
        let reused = op.hphisecond_direction_apply(&dir_i_once, &dir_j_once, &eye);

        // Rebuild path (the pre-fix loop body): build fresh directions per use.
        let fresh = op.hphisecond_direction_apply(
            &op.direction_from_deta(deta_i.clone()),
            &op.direction_from_deta(deta_j.clone()),
            &eye,
        );

        assert_eq!(
            reused, fresh,
            "reusing a precomputed FirthDirection must be bit-identical to rebuilding it"
        );
        // Sanity: the operator is doing real work (not returning zeros).
        assert!(
            reused.iter().any(|v| v.abs() > 0.0),
            "second directional derivative should be non-trivial"
        );
        assert!(
            reused.iter().all(|v| v.is_finite()),
            "second directional derivative must be finite"
        );
    }

    // Drives the full changed function end-to-end and checks the Firth outer
    // Hessian it produces is symmetric and finite. Exercises the precompute-
    // -and-reuse path for every (i,j) pair (k=2 -> 3 pairs, each reusing the
    // 2 precomputed directions).
    #[test]
    fn tk_hessian_rho_canonical_logit_firth_is_symmetric_and_finite() {
        let (x, beta, op, penalties, lambdas) = synthetic_logit_setup();
        let n = x.nrows();
        // Tierney-Kadane derivative arrays for the canonical logit working
        // model; any smooth bounded values suffice to exercise the assembly.
        let c_array = Array1::from_elem(n, 0.05_f64);
        let d_array = Array1::from_elem(n, -0.02_f64);
        let e_array = Array1::from_elem(n, 0.01_f64);
        let f_array = Array1::from_elem(n, -0.005_f64);

        // An explicit SPD H to invert: Xᵀ diag(w) X + ridge, solved via faer.
        let p = x.ncols();
        let xtwx = RemlState::tk_xt_diag_x(&x, &op.pirls_hat_diag());
        let mut h = xtwx;
        for d in 0..p {
            h[[d, d]] += 1.0;
        }
        let h_solver = h.clone();
        let h_inv_solve = move |rhs: &Array1<f64>| -> Result<Array1<f64>, EstimationError> {
            let sol = gam_linalg::utils::solve_symmetric_vector_with_floor(&h_solver, rhs, 1e-10)
                .expect("well-conditioned SPD solve");
            Ok(sol)
        };

        let hess = RemlState::tk_hessian_rho_canonical_logit(
            &x,
            &c_array,
            &d_array,
            &e_array,
            &f_array,
            &penalties,
            &lambdas,
            &beta,
            Some(&op),
            &h_inv_solve,
        )
        .expect("tk hessian");

        let k = penalties.len();
        assert_eq!(hess.dim(), (k, k));
        assert!(
            hess.iter().all(|v| v.is_finite()),
            "Firth outer Hessian must be finite: {hess:?}"
        );
        for i in 0..k {
            for j in 0..k {
                assert!(
                    (hess[[i, j]] - hess[[j, i]]).abs() <= 1e-9,
                    "Firth outer Hessian must be symmetric: ({i},{j}) {} vs {}",
                    hess[[i, j]],
                    hess[[j, i]]
                );
            }
        }
    }

    // A wider logit design with k=4 penalties, exercising the Rayon-fanned
    // first-derivative (h_i), eye-cache, and O(k²) second-derivative pair loops
    // of `tk_hessian_rho_canonical_logit` (#1575). With 4 penalties the pair loop
    // alone has 10 upper-triangle passes spread across the pool.
    fn synthetic_logit_setup_k4() -> (
        Array2<f64>,
        Array1<f64>,
        super::super::FirthDenseOperator,
        Vec<CanonicalPenalty>,
        Vec<f64>,
    ) {
        let n = 96usize;
        let p = 9usize;
        let mut x = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let t = (i as f64) / (n as f64 - 1.0);
            x[[i, 0]] = 1.0;
            x[[i, 1]] = t;
            x[[i, 2]] = t * t;
            x[[i, 3]] = (3.0 * t).sin();
            x[[i, 4]] = (2.0 * t).cos();
            x[[i, 5]] = (t - 0.5).abs();
            x[[i, 6]] = (5.0 * t).sin();
            x[[i, 7]] = (4.0 * t).cos();
            x[[i, 8]] = t * t * t;
        }
        let beta = Array1::from(vec![0.2_f64, -0.4, 0.3, 0.1, -0.2, 0.15, 0.05, -0.1, 0.12]);
        let eta = x.dot(&beta);
        let op = super::super::FirthDenseOperator::build_for_link(
            &InverseLink::Standard(StandardLink::Logit),
            &x,
            &eta,
        )
        .expect("firth operator");

        // Four block-local canonical penalties (k = 4) over disjoint coordinate
        // ranges (cols 1..2, 3..4, 5..6, 7..8).
        let mut penalties = Vec::with_capacity(4);
        for (a, b) in [(1usize, 2usize), (3, 4), (5, 6), (7, 8)] {
            let mut root = Array2::<f64>::zeros((2, p));
            root[[0, a]] = 1.0;
            root[[1, b]] = 1.0;
            penalties.push(CanonicalPenalty::from_dense_root(root, p));
        }
        let lambdas = vec![0.7_f64, 1.3, 0.5, 1.1];
        (x, beta, op, penalties, lambdas)
    }

    fn tk_hessian_for_k4_setup(
        x: &Array2<f64>,
        beta: &Array1<f64>,
        op: &super::super::FirthDenseOperator,
        penalties: &[CanonicalPenalty],
        lambdas: &[f64],
    ) -> Array2<f64> {
        let n = x.nrows();
        let p = x.ncols();
        let c_array = Array1::from_elem(n, 0.05_f64);
        let d_array = Array1::from_elem(n, -0.02_f64);
        let e_array = Array1::from_elem(n, 0.01_f64);
        let f_array = Array1::from_elem(n, -0.005_f64);
        let xtwx = RemlState::tk_xt_diag_x(x, &op.pirls_hat_diag());
        let mut h = xtwx;
        for d in 0..p {
            h[[d, d]] += 1.0;
        }
        let h_solver = h.clone();
        let h_inv_solve = move |rhs: &Array1<f64>| -> Result<Array1<f64>, EstimationError> {
            let sol = gam_linalg::utils::solve_symmetric_vector_with_floor(&h_solver, rhs, 1e-10)
                .expect("well-conditioned SPD solve");
            Ok(sol)
        };
        RemlState::tk_hessian_rho_canonical_logit(
            x, &c_array, &d_array, &e_array, &f_array, penalties, lambdas, beta, Some(op),
            &h_inv_solve,
        )
        .expect("tk hessian")
    }

    // The #1575 Rayon fan-out of the first-derivative / eye-cache / pair loops
    // must not perturb the assembled Firth outer Hessian: repeated evaluation is
    // BIT-IDENTICAL (no race / mis-indexed write-back), and the result is
    // symmetric and finite. Determinism across repeats is the load-bearing guard
    // — a parallel write-back bug would surface as run-to-run drift or asymmetry.
    #[test]
    fn tk_hessian_rho_canonical_logit_firth_is_deterministic_under_parallel_fanout_k4() {
        let (x, beta, op, penalties, lambdas) = synthetic_logit_setup_k4();
        let k = penalties.len();
        // The fan-out this test guards is gated on `rayon::current_num_threads() > 1`
        // (the `Some(op) if k > 1 && current_num_threads() > 1`, `fan_units`, and
        // `fan_pairs` branches in this module). On a single-core runner or under
        // `RAYON_NUM_THREADS=1` that gate is false and the assembly silently takes
        // the SERIAL path — so without pinning a multi-threaded pool the
        // determinism guard would be vacuous (it would "pass" while exercising none
        // of the parallel write-back it is named for). Run every assembly inside a
        // dedicated >1-thread pool so the parallel path is guaranteed regardless of
        // the ambient environment.
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("build a 4-thread pool for the fan-out determinism guard");
        assert!(
            pool.current_num_threads() > 1,
            "the #1575 fan-out guard requires a >1-thread pool to be non-vacuous"
        );
        let run = || pool.install(|| tk_hessian_for_k4_setup(&x, &beta, &op, &penalties, &lambdas));
        let first = run();
        assert_eq!(first.dim(), (k, k));
        assert!(
            first.iter().all(|v| v.is_finite()),
            "Firth outer Hessian must be finite: {first:?}"
        );
        // Symmetric to working precision (each (i,j) and (j,i) cell is filled from
        // the same pair pass, so this also pins the index-ordered write-back).
        for i in 0..k {
            for j in 0..k {
                assert!(
                    (first[[i, j]] - first[[j, i]]).abs() <= 1e-9,
                    "Firth outer Hessian must be symmetric: ({i},{j}) {} vs {}",
                    first[[i, j]],
                    first[[j, i]]
                );
            }
        }
        // Re-run several times: the Rayon-fanned assembly must return BYTE-FOR-BYTE
        // the same matrix every time (the inner faer GEMMs are pinned to Par::Seq
        // inside with_nested_parallel, and the reduction is index-ordered).
        for rep in 0..4 {
            let again = run();
            assert_eq!(
                first.mapv(|v| v.to_bits()),
                again.mapv(|v| v.to_bits()),
                "parallel-fanned Firth outer Hessian must be bit-identical across runs (rep {rep})"
            );
        }
        // Sanity: the off-diagonal mixed second derivatives are non-trivial, so the
        // O(k²) pair loop is doing real work (not returning zeros).
        assert!(
            (0..k).any(|i| (0..k).any(|j| i != j && first[[i, j]].abs() > 0.0)),
            "k=4 Firth outer Hessian should have non-zero mixed second derivatives"
        );
    }

    // #1575: the batched canonical direct TK ρ-gradient shares one `O(n²·p)` gram
    // assembly across all `k` directions. This locks the invariant that makes the
    // substitution a pure perf win: `tk_direct_gradient_canonical_batched[idx]` is
    // BIT-IDENTICAL to the per-direction `tk_direct_gradient_from_cd_and_design`
    // it replaces. Uses n > TK_BLOCK_SIZE (multiple blocks incl. an off-diagonal
    // sym-factor-2 pair) and a scattered `c = 0` set (with one direction's `c'`
    // also vanishing there) so the union active-block path is exercised where a
    // direction's OWN active set is strictly smaller than the union.
    #[test]
    fn tk_direct_gradient_canonical_batched_matches_per_direction_bit_identical_1575() {
        let n = 200usize;
        let p = 4usize;
        let k = 3usize;
        let mut x_dense = Array2::<f64>::zeros((n, p));
        let mut z = Array2::<f64>::zeros((p, n));
        let mut c_array = Array1::<f64>::zeros(n);
        let mut d_array = Array1::<f64>::zeros(n);
        let mut e_array = Array1::<f64>::zeros(n);
        for i in 0..n {
            let t = (i as f64) / (n as f64);
            for j in 0..p {
                x_dense[[i, j]] = (0.3 + j as f64 * 0.11) * ((t * (j as f64 + 1.3)).sin() + 0.5);
                z[[j, i]] = 0.2 * ((t * (j as f64 + 0.7)).cos() - 0.1 * j as f64);
            }
            // Scattered c = 0 rows so a direction whose c' also vanishes there is
            // absent from that direction's own active blocks (union ≠ per-dir).
            c_array[i] = if i % 17 == 0 {
                0.0
            } else {
                0.15 * (t - 0.4).sin() + 0.2
            };
            d_array[i] = 0.1 * (t * 2.1).cos() - 0.05;
            e_array[i] = 0.07 * (t * 1.3).sin() + 0.03;
        }
        let mut x_vks: Vec<Array1<f64>> = Vec::with_capacity(k);
        for idx in 0..k {
            let mut v = Array1::<f64>::zeros(n);
            for i in 0..n {
                let t = (i as f64) / (n as f64);
                v[i] = if idx == 0 && i % 17 == 0 {
                    0.0
                } else {
                    0.4 * ((t * (idx as f64 + 1.1)).sin() + 0.2 * idx as f64)
                };
            }
            x_vks.push(v);
        }

        let solve =
            |rhs: &Array1<f64>| -> Result<Array1<f64>, EstimationError> { Ok(rhs.clone()) };
        let shared = RemlState::tk_shared_intermediates(
            &x_dense,
            &z,
            &c_array,
            "batched-bit-identity test",
            &solve,
        )
        .expect("shared TK intermediates");

        // Per-direction reference (the pre-#1575 path).
        let mut gram_ref = Array2::<f64>::zeros((TK_BLOCK_SIZE, TK_BLOCK_SIZE));
        let mut reference = Vec::with_capacity(k);
        for idx in 0..k {
            let eta_total = x_vks[idx].mapv(|value| -value);
            let direct = RemlState::tk_direct_gradient_from_cd_and_design(
                &x_dense,
                &z,
                &c_array,
                &d_array,
                &e_array,
                &eta_total,
                None,
                &shared,
                &mut gram_ref,
                true,
            )
            .expect("per-direction direct gradient");
            reference.push(direct);
        }

        // Batched (the #1575 shared-gram fast path).
        let mut gram_batched = Array2::<f64>::zeros((TK_BLOCK_SIZE, TK_BLOCK_SIZE));
        let batched = RemlState::tk_direct_gradient_canonical_batched(
            &x_dense,
            &z,
            &c_array,
            &d_array,
            &e_array,
            &x_vks,
            k,
            &shared,
            &mut gram_batched,
        )
        .expect("batched direct gradient");

        assert_eq!(batched.len(), k, "batched must return one value per direction");
        for idx in 0..k {
            assert!(
                reference[idx].is_finite() && reference[idx].abs() > 0.0,
                "reference direct[{idx}] should be a non-trivial finite value: {}",
                reference[idx]
            );
            assert_eq!(
                batched[idx].to_bits(),
                reference[idx].to_bits(),
                "batched direct[{idx}] ({}) is not bit-identical to the per-direction \
                 value ({})",
                batched[idx],
                reference[idx]
            );
        }
    }

    // #1575 measurement: at the `n` where the default-ON binomial/logit Firth path
    // pays its dominant `O(n²·p)` direct-gradient cost, the shared-gram batched
    // path is materially faster than refilling the gram `k` times — while staying
    // bit-identical. Reports a same-build, same-box before/after ratio (wall time
    // on a shared CI box is noisy, so the ASSERTION is bit-identity, not a time
    // bound; the printed ratio is the measurement).
    #[test]
    fn tk_direct_gradient_batched_faster_than_per_direction_bit_identical_1575() {
        let n = 2000usize;
        let p = 4usize;
        let k = 3usize;
        let mut x_dense = Array2::<f64>::zeros((n, p));
        let mut z = Array2::<f64>::zeros((p, n));
        let mut c_array = Array1::<f64>::zeros(n);
        let mut d_array = Array1::<f64>::zeros(n);
        let mut e_array = Array1::<f64>::zeros(n);
        for i in 0..n {
            let t = (i as f64) / (n as f64);
            for j in 0..p {
                x_dense[[i, j]] = (0.3 + j as f64 * 0.11) * ((t * (j as f64 + 1.3)).sin() + 0.5);
                z[[j, i]] = 0.2 * ((t * (j as f64 + 0.7)).cos() - 0.1 * j as f64);
            }
            c_array[i] = 0.15 * (t - 0.4).sin() + 0.2;
            d_array[i] = 0.1 * (t * 2.1).cos() - 0.05;
            e_array[i] = 0.07 * (t * 1.3).sin() + 0.03;
        }
        let x_vks: Vec<Array1<f64>> = (0..k)
            .map(|idx| {
                Array1::from_shape_fn(n, |i| {
                    let t = (i as f64) / (n as f64);
                    0.4 * ((t * (idx as f64 + 1.1)).sin() + 0.2 * idx as f64)
                })
            })
            .collect();

        let solve =
            |rhs: &Array1<f64>| -> Result<Array1<f64>, EstimationError> { Ok(rhs.clone()) };
        let shared =
            RemlState::tk_shared_intermediates(&x_dense, &z, &c_array, "perf test", &solve)
                .expect("shared TK intermediates");

        let reps = 20usize;
        let mut gram = Array2::<f64>::zeros((TK_BLOCK_SIZE, TK_BLOCK_SIZE));

        // Time the per-direction (pre-#1575) path.
        let t_ref = std::time::Instant::now();
        for _ in 0..reps {
            for idx in 0..k {
                let eta_total = x_vks[idx].mapv(|value| -value);
                std::hint::black_box(
                    RemlState::tk_direct_gradient_from_cd_and_design(
                        &x_dense, &z, &c_array, &d_array, &e_array, &eta_total, None, &shared,
                        &mut gram, true,
                    )
                    .expect("per-direction direct"),
                );
            }
        }
        let dt_ref = t_ref.elapsed().as_secs_f64();

        // Time the batched (#1575) path.
        let t_bat = std::time::Instant::now();
        for _ in 0..reps {
            std::hint::black_box(
                RemlState::tk_direct_gradient_canonical_batched(
                    &x_dense, &z, &c_array, &d_array, &e_array, &x_vks, k, &shared, &mut gram,
                )
                .expect("batched direct"),
            );
        }
        let dt_bat = t_bat.elapsed().as_secs_f64();

        eprintln!(
            "#1575 tk_direct_gradient n={n} p={p} k={k} reps={reps}: \
             per_direction={dt_ref:.4}s  batched={dt_bat:.4}s  speedup={:.2}x",
            dt_ref / dt_bat.max(1e-12)
        );

        // Final values for the bit-identity assertion.
        let mut ref_last = vec![0.0_f64; k];
        for idx in 0..k {
            let eta_total = x_vks[idx].mapv(|value| -value);
            ref_last[idx] = RemlState::tk_direct_gradient_from_cd_and_design(
                &x_dense, &z, &c_array, &d_array, &e_array, &eta_total, None, &shared, &mut gram,
                true,
            )
            .expect("per-direction direct");
        }
        let batched_last = RemlState::tk_direct_gradient_canonical_batched(
            &x_dense, &z, &c_array, &d_array, &e_array, &x_vks, k, &shared, &mut gram,
        )
        .expect("batched direct");

        for idx in 0..k {
            assert_eq!(
                batched_last[idx].to_bits(),
                ref_last[idx].to_bits(),
                "batched direct[{idx}] not bit-identical to per-direction at n={n}"
            );
        }
    }
}
