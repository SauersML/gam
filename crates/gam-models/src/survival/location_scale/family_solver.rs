use super::*;
use opt::{BacktrackConfig, RidgeSchedule, backtracking_line_search, constants, escalate_ridge};
use std::convert::Infallible;

impl SurvivalLocationScaleFamily {
    /// Recompute every block's linear predictor `η_b = D_b · β_b + o_b` from
    /// the joint coefficient vector `theta` (block-concatenated) and the block
    /// specs, returning freshly populated [`ParameterBlockState`]s.
    ///
    /// This mirrors the static-geometry branch of the inner solver's
    /// `refresh_all_block_etas`: in the reduced constant-scale parametric-AFT
    /// regime there is no link-wiggle and no monotone time-wiggle, so the
    /// family geometry is static and `solver_design()`/`solver_offset()`
    /// (the stacked `[entry; exit; deriv]` channels) map β to η directly. Each
    /// block's β is passed through `post_update_block_beta` so the time-warp
    /// monotonicity constraints are validated exactly as the coupled path does.
    pub(crate) fn parametric_aft_states_from_theta(
        &self,
        theta: &Array1<f64>,
        specs: &[ParameterBlockSpec],
    ) -> Result<Vec<ParameterBlockState>, String> {
        let offsets = self.joint_block_offsets();
        if theta.len() != *offsets.last().unwrap_or(&0) {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "parametric-AFT direct MLE theta length mismatch: got {}, expected {}",
                    theta.len(),
                    offsets.last().copied().unwrap_or(0)
                ),
            }
            .into());
        }
        let mut states = Vec::with_capacity(specs.len());
        for (b, spec) in specs.iter().enumerate() {
            let beta = theta.slice(s![offsets[b]..offsets[b + 1]]).to_owned();
            let eta = spec.solver_design().matrixvectormultiply(&beta) + spec.solver_offset();
            states.push(ParameterBlockState { beta, eta });
        }
        // Validate (and, for any family that projects, project) each block's β
        // against its constraints — the time block's monotone-derivative guard.
        for b in 0..specs.len() {
            let raw = states[b].beta.clone();
            let projected = self.post_update_block_beta(&states, b, &specs[b], raw)?;
            if projected != states[b].beta {
                states[b].beta.assign(&projected);
                states[b].eta = specs[b]
                    .solver_design()
                    .matrixvectormultiply(&states[b].beta)
                    + specs[b].solver_offset();
            }
        }
        Ok(states)
    }

    /// Direct, robust maximum-likelihood fit of the fully reduced constant-scale
    /// parametric AFT (affine time-warp + location intercept/covariates +
    /// constant log-σ).
    ///
    /// In this regime every block is UNPENALIZED — there are no smoothing
    /// parameters and the REML/LAML outer search is vacuous — so the coupled
    /// exact-joint REML machinery is the wrong tool (issue #736/#735/#721): it
    /// runs an outer ρ search around an inner per-block trust-region Newton that
    /// oscillates and never certifies stationarity on this tiny unpenalized
    /// likelihood. Instead we run a damped, line-searched joint Newton directly
    /// on the negative log-likelihood `−ℓ(θ)`, converging in a handful of
    /// iterations exactly like `survreg`/`lifelines`.
    ///
    /// The step is `δ = H⁻¹ g` with `g = ∇ℓ` (the block-concatenated
    /// log-likelihood gradient) and `H = −∇²ℓ` (the exact joint Hessian, all
    /// cross-blocks included). When `H` is not positive definite at the current
    /// iterate we add Levenberg damping `τ·I` (escalating geometrically) until
    /// the Cholesky factorization succeeds, giving a guaranteed ascent
    /// direction. The step length is first capped to keep the monotone
    /// time-warp feasible (`max_feasible_step_size`) and then Armijo-backtracked
    /// on `−ℓ`, so the time derivative stays `≥ guard` at every observed time
    /// and `ℓ` increases monotonically.
    ///
    /// # Convergence criterion
    ///
    /// Stationarity is certified by the **Newton decrement**
    /// `λ²(θ) = gᵀH⁻¹g = g·δ ≥ 0`, whose half is a second-order estimate of the
    /// log-likelihood gap `ℓ(θ*) − ℓ(θ) ≈ ½λ²` (equivalently, `λ²` is the squared
    /// Mahalanobis distance from `θ` to the optimum in the observed-information
    /// metric). The fit stops when `½λ² ≤ obj_tol`.
    ///
    /// A raw gradient-norm test is NOT used here: `g = ∇ℓ` is a SUM over the `n`
    /// observations, so at the true MLE its attainable sup-norm floor in double
    /// precision grows like `n·ε`, and an absolute gradient tolerance therefore
    /// spuriously fails to converge on perfectly benign data with rising
    /// frequency as `n` grows (gam#2112). The decrement `gᵀH⁻¹g` divides the
    /// n-scaled gradient by the n-scaled curvature, so it is invariant to the
    /// sample size and to any affine reparameterization: a single fixed `obj_tol`
    /// certifies stationarity uniformly across `n`, and its own round-off floor
    /// (`~ n·ε²·κ(H)`) stays vanishingly far below any usable tolerance.
    ///
    /// If, near the optimum, the damped-Newton ascent direction admits no
    /// Armijo-sufficient step (`ℓ` cannot be increased to numerical precision)
    /// while `½λ²` is already below [`REDUCED_AFT_NEWTON_STALL_TOL`], the iterate
    /// is accepted as the numerical MLE; a large decrement at such a stall is a
    /// genuine curvature-model failure and is surfaced as an error.
    ///
    /// Returns the converged block states, the log-likelihood at the MLE, and
    /// the joint negative-log-likelihood Hessian `H` (the observed information),
    /// whose inverse is the conditional covariance the caller assembles.
    ///
    /// `obj_tol` is the objective-suboptimality tolerance on `½λ²` described
    /// above (floored by the caller at [`REDUCED_AFT_OBJ_TOL_FLOOR`]).
    pub(crate) fn fit_parametric_aft_direct_mle(
        &self,
        specs: &[ParameterBlockSpec],
        max_iter: usize,
        obj_tol: f64,
    ) -> Result<(Vec<ParameterBlockState>, f64, Array2<f64>), String> {
        use gam_linalg::faer_ndarray::FaerCholesky;

        self.validate_joint_specs(
            specs,
            "SurvivalLocationScaleFamily direct parametric-AFT MLE",
        )?;
        let offsets = self.joint_block_offsets();
        let p_total = *offsets.last().unwrap_or(&0);
        if p_total == 0 {
            return Err(SurvivalLocationScaleError::InvalidConfiguration {
                reason: "direct parametric-AFT MLE has no free coefficients".to_string(),
            }
            .into());
        }

        // Cold-start θ from the block specs' (feasible) initial β, falling back
        // to zeros. `parametric_aft_states_from_theta` re-validates feasibility.
        let mut theta = Array1::<f64>::zeros(p_total);
        for (b, spec) in specs.iter().enumerate() {
            if let Some(beta0) = spec.initial_beta.as_ref() {
                if beta0.len() != offsets[b + 1] - offsets[b] {
                    return Err(SurvivalLocationScaleError::DimensionMismatch {
                        reason: format!(
                            "direct parametric-AFT MLE block {b} initial_beta length {} != block width {}",
                            beta0.len(),
                            offsets[b + 1] - offsets[b]
                        ),
                    }
                    .into());
                }
                theta
                    .slice_mut(s![offsets[b]..offsets[b + 1]])
                    .assign(beta0);
            }
        }

        let mut states = self.parametric_aft_states_from_theta(&theta, specs)?;
        // Resync θ to any constraint projection the state builder applied.
        for (b, state) in states.iter().enumerate() {
            theta
                .slice_mut(s![offsets[b]..offsets[b + 1]])
                .assign(&state.beta);
        }
        let mut ll = self.log_likelihood_only(&states)?;
        if !ll.is_finite() {
            return Err(SurvivalLocationScaleError::NumericalFailure {
                reason: format!(
                    "direct parametric-AFT MLE: non-finite initial log-likelihood {ll}"
                ),
            }
            .into());
        }

        // Newton iterations on −ℓ(θ).
        let mut converged = false;
        let mut last_grad_norm = f64::INFINITY;
        let mut last_newton_decrement = f64::INFINITY;
        for _ in 0..max_iter {
            let (ll_now, block_gradients) =
                self.evaluate_log_likelihood_and_block_gradients(&states)?;
            ll = ll_now;
            // Concatenate the block log-likelihood gradients g = ∇ℓ.
            let mut g = Array1::<f64>::zeros(p_total);
            if block_gradients.len() != specs.len() {
                return Err(SurvivalLocationScaleError::DimensionMismatch {
                    reason: format!(
                        "direct parametric-AFT MLE gradient block count mismatch: gradients={}, specs={}",
                        block_gradients.len(),
                        specs.len()
                    ),
                }
                .into());
            }
            for (b, gb) in block_gradients.iter().enumerate() {
                if gb.len() != offsets[b + 1] - offsets[b] {
                    return Err(SurvivalLocationScaleError::DimensionMismatch {
                        reason: format!(
                            "direct parametric-AFT MLE block {b} gradient length {} != block width {}",
                            gb.len(),
                            offsets[b + 1] - offsets[b]
                        ),
                    }
                    .into());
                }
                g.slice_mut(s![offsets[b]..offsets[b + 1]]).assign(gb);
            }
            if !g.iter().all(|v| v.is_finite()) {
                return Err(SurvivalLocationScaleError::NumericalFailure {
                    reason: "direct parametric-AFT MLE: non-finite gradient".to_string(),
                }
                .into());
            }
            // The step `H δ = g` is solved on a consistent (objective, gradient,
            // Hessian) triple for EVERY residual distribution: `g = ∇ℓ` above is
            // the block-gradient reduction
            // (`evaluate_log_likelihood_and_block_gradients`) and `H = −∇²ℓ` below
            // is the packed 24-pair coefficient lowering; both are pinned to the
            // ONE single-sourced `sls_row_nll` program to ≤1e-9 by the analytic
            // oracles (`survival_ls_block_gradient_matches_single_sourced_tower_932`
            // for the gradient across Gaussian/Gumbel/Logistic on the every-channel
            // time-varying shape;
            // `survival_ls_time_varying_joint_hessian_matches_single_sourced_tower_932`
            // for the Hessian). This closes gam#1110, where an earlier hand block
            // gradient diverged from the jet for the logit (log-logistic) residual
            // and pinned the `age` location coefficient to its cold-start 0: the
            // oracle now forbids any dropped cross-channel term from reappearing.
            // Retained for diagnostics only — the stopping test is the Newton
            // decrement computed below, NOT this raw summed-gradient sup-norm
            // (whose attainable floor scales with `n`; see the doc comment /
            // gam#2112).
            let grad_norm = g.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
            last_grad_norm = grad_norm;

            // H = −∇²ℓ (positive (semi)definite near the optimum). The exact
            // joint Hessian assembly returns it directly, symmetrized.
            let h = self.exact_newton_joint_hessian(&states)?.ok_or_else(|| {
                SurvivalLocationScaleError::NumericalFailure {
                    reason: "direct parametric-AFT MLE: joint Hessian assembly failed".to_string(),
                }
            })?;
            if !h.iter().all(|v| v.is_finite()) {
                return Err(SurvivalLocationScaleError::NumericalFailure {
                    reason: "direct parametric-AFT MLE: non-finite joint Hessian".to_string(),
                }
                .into());
            }

            // Newton direction δ solving H δ = g (ascent on ℓ). When H is not
            // positive definite, escalate Levenberg damping τ·I until the
            // Cholesky factorization succeeds, guaranteeing an ascent direction.
            let h_scale = h
                .diag()
                .iter()
                .fold(0.0_f64, |acc, &v| acc.max(v.abs()))
                .max(1.0);
            let try_damped = |tau: f64| -> Option<Array1<f64>> {
                let mut damped = h.clone();
                if tau > 0.0 {
                    for i in 0..p_total {
                        damped[[i, i]] += tau;
                    }
                }
                damped
                    .cholesky(faer::Side::Lower)
                    .ok()
                    .map(|chol| chol.solvevec(&g))
            };
            // Bare (undamped) Newton solve first; on failure escalate τ
            // geometrically across the damping span [INITIAL, MAX]·h_scale —
            // the trial count is that span's decade count, INCLUSIVE of the
            // final τ ≈ MAX·h_scale. The pre-primitive loop compared its
            // FP-accumulated τ chain against the single product `MAX·h_scale`
            // and, for ~40% of h_scale values, dropped the final decade by one
            // ulp; the deterministic count realizes the documented cap for
            // every h_scale.
            let damping_trials = (LEVENBERG_MAX_DAMPING_REL / LEVENBERG_INITIAL_DAMPING_REL)
                .log10()
                .ceil() as usize
                + 1;
            let delta = match try_damped(0.0) {
                Some(delta) => delta,
                None => escalate_ridge(
                    RidgeSchedule {
                        initial: LEVENBERG_INITIAL_DAMPING_REL * h_scale,
                        growth: LEVENBERG_DAMPING_GROWTH,
                        max_escalations: damping_trials,
                    },
                    try_damped,
                )
                .map(|success| success.value)
                .map_err(|_| SurvivalLocationScaleError::NumericalFailure {
                    reason:
                        "direct parametric-AFT MLE: Hessian not factorizable even with maximal damping"
                            .to_string(),
                })?,
            };
            if !delta.iter().all(|v| v.is_finite()) {
                return Err(SurvivalLocationScaleError::NumericalFailure {
                    reason: "direct parametric-AFT MLE: non-finite Newton step".to_string(),
                }
                .into());
            }

            // Affine-invariant stationarity test: the Newton decrement
            //   λ² = gᵀH⁻¹g = g·δ ≥ 0,   ℓ(θ*) − ℓ(θ) ≈ ½λ².
            // Because δ = H⁻¹g divides the n-scaled gradient by the n-scaled
            // curvature, ½λ² (the estimated log-likelihood gap, equivalently the
            // squared Mahalanobis distance to θ* in the observed-information
            // metric) is invariant to the sample size — so a single `obj_tol`
            // certifies stationarity uniformly across `n`, unlike the raw
            // summed-gradient sup-norm whose floor grows like n·ε (gam#2112). If
            // Levenberg damping was active (τ>0, only ever off the optimum) δ is
            // the damped solve, so g·δ under-estimates the true decrement — a
            // conservative (never-early) test there, since a genuinely stationary
            // iterate has τ=0.
            let newton_decrement = g.dot(&delta);
            last_newton_decrement = newton_decrement;
            if 0.5 * newton_decrement <= obj_tol {
                converged = true;
                break;
            }

            // Cap the step to keep the monotone time-warp feasible: the family's
            // per-block feasibility barrier reports the largest α that keeps the
            // derivative guard satisfied (only the time block constrains it).
            let mut alpha = 1.0_f64;
            for (b, spec_offset) in offsets.iter().take(specs.len()).enumerate() {
                let block_delta = delta.slice(s![*spec_offset..offsets[b + 1]]).to_owned();
                if let Some(a_max) = self.max_feasible_step_size(&states, b, &block_delta)? {
                    alpha = alpha.min(a_max);
                }
            }

            // Armijo backtracking on −ℓ along the (feasibility-capped) Newton
            // ascent direction. `g·δ > 0` because δ is an ascent direction, so a
            // sufficient-increase condition on ℓ is well posed. The directional
            // derivative is exactly the Newton decrement computed above.
            let directional = newton_decrement;
            const MIN_ALPHA: f64 = 1e-12;
            // The pre-migration loop halved from the feasibility-capped α
            // while `alpha >= MIN_ALPHA`; count those trials by the same
            // halving recurrence (exact, unlike a log — zero trials when α₀
            // already sits below the floor, leaving the search exhausted).
            let max_steps = {
                let mut n = 0_usize;
                let mut a = alpha;
                while a >= MIN_ALPHA {
                    n += 1;
                    a *= 0.5;
                }
                n
            };
            // A trial whose block-state rebuild or likelihood evaluation errors
            // is INVALID (`Ok(None)`): halve without consulting the Armijo test.
            let accepted = match backtracking_line_search::<_, Infallible>(
                BacktrackConfig {
                    initial_step: alpha,
                    max_steps,
                    ..BacktrackConfig::default()
                },
                |alpha| {
                    let trial_theta = &theta + &(alpha * &delta);
                    let Ok(cand_states) =
                        self.parametric_aft_states_from_theta(&trial_theta, specs)
                    else {
                        return Ok(None);
                    };
                    let Ok(cand_ll) = self.log_likelihood_only(&cand_states) else {
                        return Ok(None);
                    };
                    Ok(Some((cand_ll, (trial_theta, cand_states))))
                },
                |alpha, cand_ll| {
                    cand_ll.is_finite()
                        && cand_ll >= ll + constants::ARMIJO_C1 * alpha * directional
                },
            ) {
                Ok(result) => result,
                Err(never) => match never {},
            };
            match accepted.map(|step| (step.payload.0, step.payload.1, step.value)) {
                Some((new_theta, new_states, new_ll)) => {
                    theta = new_theta;
                    states = new_states;
                    ll = new_ll;
                }
                // The damped-Newton ascent direction admits no Armijo-sufficient
                // step: ℓ can no longer be increased to numerical precision. If
                // the Newton decrement is also small, this is the numerical MLE —
                // a near-stationary iterate whose step no longer improves ℓ
                // (gam#2112) — so accept it. This is the correct terminal state
                // of a maximizer: an iterate at which the objective cannot be
                // increased IS the optimum, even if the raw (n-scaled) gradient
                // has not reached an absolute floor. A LARGE decrement here means
                // the quadratic model is badly wrong (ill-conditioning / a bad
                // local curvature model), not an MLE; that stays a structured
                // error, keeping the reduced-AFT route consistent with the coupled
                // location-scale solvers rather than handing an unconverged /
                // possibly indefinite Hessian to downstream linear algebra, where
                // panic=abort builds can terminate the CLI.
                None => {
                    if 0.5 * newton_decrement <= REDUCED_AFT_NEWTON_STALL_TOL {
                        converged = true;
                        break;
                    }
                    return Err(SurvivalLocationScaleError::NumericalFailure {
                        reason: format!(
                            "direct parametric-AFT MLE: line search failed before convergence \
                             (½·Newton-decrement {half_decrement:.6e} > tolerance {obj_tol:.6e}; \
                             gradient sup-norm {grad_norm:.6e})",
                            half_decrement = 0.5 * newton_decrement
                        ),
                    }
                    .into());
                }
            }
        }

        if !converged {
            return Err(SurvivalLocationScaleError::NumericalFailure {
                reason: format!(
                    "direct parametric-AFT MLE: failed to converge after {max_iter} Newton iterations \
                     (last ½·Newton-decrement {half_decrement:.6e} > tolerance {obj_tol:.6e}; \
                     last gradient sup-norm {last_grad_norm:.6e})",
                    half_decrement = 0.5 * last_newton_decrement
                ),
            }
            .into());
        }

        // Observed information at the MLE: the joint negative-log-likelihood
        // Hessian. This is the conditional precision; its inverse is the
        // covariance the caller lifts to the raw coordinate system.
        let h_final = self.exact_newton_joint_hessian(&states)?.ok_or_else(|| {
            SurvivalLocationScaleError::NumericalFailure {
                reason: "direct parametric-AFT MLE: final joint Hessian assembly failed"
                    .to_string(),
            }
        })?;
        if !h_final.iter().all(|v| v.is_finite()) {
            return Err(SurvivalLocationScaleError::NumericalFailure {
                reason: "direct parametric-AFT MLE: non-finite final joint Hessian".to_string(),
            }
            .into());
        }
        Ok((states, ll, h_final))
    }

    /// Compute the log-scale shift needed to keep CLogLog survival
    /// derivatives finite.  Returns `L >= 0` such that `exp(u - L) <= exp(500)`
    /// for all row linear predictors `u`.  For non-CLogLog links, returns 0.
    pub(crate) fn hessian_deriv_log_rescale(&self, block_states: &[ParameterBlockState]) -> f64 {
        if !matches!(
            self.inverse_link,
            InverseLink::Standard(StandardLink::CLogLog)
        ) {
            return 0.0;
        }
        let dynamic = match self.build_dynamic_geometry(block_states) {
            Ok(d) => d,
            Err(_) => return 0.0,
        };
        let mut max_u = f64::NEG_INFINITY;
        for i in 0..self.n {
            if self.w[i] <= 0.0 {
                continue;
            }
            let u0 = dynamic.h_entry[i] + dynamic.q_entry[i];
            let u1 = dynamic.h_exit[i] + dynamic.q_exit[i];
            max_u = max_u.max(u0).max(u1);
        }
        // Shift so the largest exp(u - L) ~ exp(500), well within f64 range.
        (max_u - 500.0).max(0.0)
    }

    /// Rescaled joint Hessian for logdet computation.  Returns
    /// `(H_scaled, L)` where `H_scaled = exp(-L) * H_exact` and
    /// `logdet(H_exact) = logdet(H_scaled) + p * L`.
    pub(crate) fn exact_newton_joint_hessian_rescaled(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<(Array2<f64>, f64)>, String> {
        let log_scale = self.hessian_deriv_log_rescale(block_states);
        if log_scale == 0.0 {
            return Ok(self
                .exact_newton_joint_hessian(block_states)?
                .map(|h| (h, 0.0)));
        }
        let dynamic = self.build_dynamic_geometry(block_states)?;
        if self.x_link_wiggle.is_some() {
            // #932: the link-wiggle joint Hessian is single-sourced through the
            // §13 warp kernel (`sls_row_nll_wiggle`) instead of the bespoke
            // `assemble_h_wiggle`; non-wiggle rows are untouched below.
            return Ok(Some((
                super::row_kernel::survival_ls_wiggle_joint_hessian_dense(
                    self, &dynamic, log_scale,
                )?,
                log_scale,
            )));
        }
        let dense = self
            .survival_ls_coefficient_hessian(
                &dynamic,
                log_scale,
                None,
                SlsCoefficientHessianTarget::DenseFull,
            )?
            .into_dense_full()?;
        Ok(Some((dense, log_scale)))
    }

    pub(crate) fn exact_newton_joint_hessian_directional_derivative_rescaled(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
        log_rescale: f64,
    ) -> Result<Option<Array2<f64>>, String> {
        let dynamic = self.build_dynamic_geometry(block_states)?;
        self.exact_newton_joint_hessian_directional_derivative_rescaled_from_parts(
            d_beta_flat,
            &dynamic,
            log_rescale,
        )
    }

    /// `_from_parts` variant of
    /// [`Self::exact_newton_joint_hessian_directional_derivative_rescaled`]
    /// that receives the precomputed dynamic geometry instead of recomputing it
    /// on every call. This is the workspace-friendly entry point used by
    /// `SurvivalLocationScaleExactNewtonJointHessianWorkspace` to avoid the
    /// ~300 redundant `build_dynamic_geometry` sweeps the outer Hessian pair loop would
    /// otherwise trigger per evaluation.
    pub(crate) fn exact_newton_joint_hessian_directional_derivative_rescaled_from_parts(
        &self,
        d_beta_flat: &Array1<f64>,
        dynamic: &SurvivalDynamicGeometry,
        deriv_log_scale: f64,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_directional_derivative_rescaled_from_parts_masked(
            d_beta_flat,
            dynamic,
            deriv_log_scale,
            None,
        )
    }

    /// HT-mask-aware variant of
    /// [`Self::exact_newton_joint_hessian_directional_derivative_rescaled_from_parts`].
    /// `None` is byte-identical to the full-row expression at every site. The
    /// mask scales each completed nonlinear row contribution exactly once.
    pub(crate) fn exact_newton_joint_hessian_directional_derivative_rescaled_from_parts_masked(
        &self,
        d_beta_flat: &Array1<f64>,
        dynamic: &SurvivalDynamicGeometry,
        deriv_log_scale: f64,
        row_mask: Option<&Array1<f64>>,
    ) -> Result<Option<Array2<f64>>, String> {
        let offsets = self.joint_block_offsets();
        let p_total = *offsets
            .last()
            .ok_or_else(|| "missing joint block offsets".to_string())?;
        if d_beta_flat.len() != p_total {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "joint d_beta length mismatch: got {}, expected {p_total}",
                    d_beta_flat.len()
                ),
            }
            .into());
        }

        if self.row_kernel_directional_supported() {
            let kernel = self.survival_ls_row_kernel_rescaled(dynamic, deriv_log_scale);
            let rows = row_set_from_survival_mask(row_mask, self.n);
            return crate::row_kernel::row_kernel_directional_derivative(
                &kernel,
                &rows,
                d_beta_flat
                    .as_slice()
                    .ok_or_else(|| "joint d_beta must be contiguous".to_string())?,
            )
            .map(Some);
        }

        // #932: single-source the link-wiggle FIRST directional derivative
        // `D_dir H` through the §13 warp kernel. The βw-dependent Jacobian is
        // carried by the kernel's `JᵀHJ` pullback, so the contracted third
        // (`OneSeed<KW>`) reproduces the bespoke hand assembly that used to live
        // here, by single-source construction: it is one more jet order of the
        // SAME §13 warp NLL whose Order2 joint Hessian is oracle-pinned to the
        // bespoke `assemble_h_wiggle`. This branch is reached ONLY for wiggle
        // rows — `row_kernel_directional_supported()` is `x_link_wiggle.is_none()`,
        // so the non-wiggle case already returned above — and the convention
        // matches that base path (same `row_kernel_directional_derivative`).
        let rows = row_set_from_survival_mask(row_mask, self.n);
        let d = d_beta_flat
            .as_slice()
            .ok_or_else(|| "joint d_beta must be contiguous".to_string())?;
        Ok(Some(
            super::row_kernel::survival_ls_wiggle_directional_derivative_dense(
                self,
                dynamic,
                deriv_log_scale,
                &rows,
                d,
            )?,
        ))
    }

    pub(crate) fn evaluate_log_likelihood_and_block_gradients(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<(f64, Vec<Array1<f64>>), String> {
        self.evaluate_log_likelihood_and_block_gradients_masked(block_states, None)
    }

    /// HT-mask-aware variant of
    /// [`Self::evaluate_log_likelihood_and_block_gradients`]. `None` is
    /// byte-identical to the pre-refactor implementation. `Some(m)`
    /// multiplies each row's likelihood contribution and per-row partial
    /// derivative contributions by `m[i]` before aggregation: the
    /// downstream `X.t().dot(...)` / `transpose_vector_multiply` calls
    /// then automatically produce the HT-weighted gradient.
    pub(crate) fn evaluate_log_likelihood_and_block_gradients_masked(
        &self,
        block_states: &[ParameterBlockState],
        row_mask: Option<&Array1<f64>>,
    ) -> Result<(f64, Vec<Array1<f64>>), String> {
        let n = self.n;
        let dynamic = self.build_dynamic_geometry(block_states)?;
        let mut ll = 0.0;

        let mut grad_time_eta_h0 = Array1::<f64>::zeros(n);
        let mut grad_time_eta_h1 = Array1::<f64>::zeros(n);
        let mut grad_time_eta_d = Array1::<f64>::zeros(n);
        let mut d1_q0 = Array1::<f64>::zeros(n);
        let mut d1_q1 = Array1::<f64>::zeros(n);
        let mut d1_qdot = Array1::<f64>::zeros(n);

        // HT mask lookup: returns m[i] if mask is Some(m) else 1.0. For
        // f64 multiplication, `x * 1.0 == x` exactly (IEEE 754), so the
        // None path is byte-identical to the pre-refactor expression.
        let mask_at = |i: usize| -> f64 { row_mask.map_or(1.0, |m| m[i]) };
        if n >= Self::EVALUATE_PARALLEL_ROW_THRESHOLD && rayon::current_num_threads() > 1 {
            const CHUNK: usize = 1024;
            let d1_q0_s = d1_q0
                .as_slice_memory_order_mut()
                .expect("zeros is contiguous");
            let d1_q1_s = d1_q1
                .as_slice_memory_order_mut()
                .expect("zeros is contiguous");
            let d1_qdot_s = d1_qdot
                .as_slice_memory_order_mut()
                .expect("zeros is contiguous");
            let g_h0_s = grad_time_eta_h0
                .as_slice_memory_order_mut()
                .expect("zeros is contiguous");
            let g_h1_s = grad_time_eta_h1
                .as_slice_memory_order_mut()
                .expect("zeros is contiguous");
            let g_d_s = grad_time_eta_d
                .as_slice_memory_order_mut()
                .expect("zeros is contiguous");
            let ll_partials: Vec<f64> = d1_q0_s
                .par_chunks_mut(CHUNK)
                .zip(d1_q1_s.par_chunks_mut(CHUNK))
                .zip(d1_qdot_s.par_chunks_mut(CHUNK))
                .zip(g_h0_s.par_chunks_mut(CHUNK))
                .zip(g_h1_s.par_chunks_mut(CHUNK))
                .zip(g_d_s.par_chunks_mut(CHUNK))
                .enumerate()
                .map(
                    |(chunk_idx, (((((d1q0_c, d1q1_c), d1qd_c), gh0_c), gh1_c), gd_c))|
                     -> Result<f64, String> {
                        let start = chunk_idx * CHUNK;
                        let mut acc = 0.0_f64;
                        for local in 0..d1q0_c.len() {
                            let i = start + local;
                            let state = self.row_predictor_state(
                                dynamic.h_entry[i],
                                dynamic.h_exit[i],
                                dynamic.hdot_exit[i],
                                dynamic.q_entry[i],
                                dynamic.q_exit[i],
                                dynamic.qdot_exit[i],
                            );
                            if let Some(row) = self.row_derivatives(i, state)? {
                                let w = mask_at(i);
                                acc += row.ll * w;
                                d1q0_c[local] = row.d1_q0 * w;
                                d1q1_c[local] = row.d1_q1 * w;
                                d1qd_c[local] = row.d1_qdot1 * w;
                                gh0_c[local] = row.grad_time_eta_h0 * w;
                                gh1_c[local] = row.grad_time_eta_h1 * w;
                                gd_c[local] = row.grad_time_eta_d * w;
                            }
                        }
                        Ok(acc)
                    },
                )
                .collect::<Result<Vec<f64>, String>>()?;
            ll = gam_linalg::pairwise_reduce::pairwise_sum(&ll_partials);
        } else {
            for i in 0..n {
                let state = self.row_predictor_state(
                    dynamic.h_entry[i],
                    dynamic.h_exit[i],
                    dynamic.hdot_exit[i],
                    dynamic.q_entry[i],
                    dynamic.q_exit[i],
                    dynamic.qdot_exit[i],
                );
                let Some(row) = self.row_derivatives(i, state)? else {
                    continue;
                };
                let w = mask_at(i);
                ll += row.ll * w;
                d1_q0[i] = row.d1_q0 * w;
                d1_q1[i] = row.d1_q1 * w;
                d1_qdot[i] = row.d1_qdot1 * w;
                grad_time_eta_h0[i] = row.grad_time_eta_h0 * w;
                grad_time_eta_h1[i] = row.grad_time_eta_h1 * w;
                grad_time_eta_d[i] = row.grad_time_eta_d * w;
            }
        }

        let grad_time = dynamic.time_jac_entry.t().dot(&grad_time_eta_h0)
            + dynamic.time_jac_exit.t().dot(&grad_time_eta_h1)
            + dynamic.time_jac_deriv.t().dot(&grad_time_eta_d);

        let mut scratch = Array1::<f64>::zeros(n);

        let grad_t = if let (Some(x_t_entry), Some(x_t_deriv)) = (
            self.x_threshold_entry.as_ref(),
            self.x_threshold_deriv.as_ref(),
        ) {
            // grad_exit[i] = d1_q1[i] * dq_t_exit[i] + d1_qdot[i] * dqdot_t[i]
            ndarray::Zip::from(&mut scratch)
                .and(&d1_q1)
                .and(&dynamic.dq_t_exit)
                .and(&d1_qdot)
                .and(&dynamic.dqdot_t)
                .for_each(|s, &a, &b, &c, &d| *s = a * b + c * d);
            let mut out = self.x_threshold.transpose_vector_multiply(&scratch);
            // grad_entry[i] = d1_q0[i] * dq_t_entry[i]
            ndarray::Zip::from(&mut scratch)
                .and(&d1_q0)
                .and(&dynamic.dq_t_entry)
                .for_each(|s, &a, &b| *s = a * b);
            out = out + x_t_entry.transpose_vector_multiply(&scratch);
            // grad_deriv[i] = d1_qdot[i] * dqdot_td[i]
            ndarray::Zip::from(&mut scratch)
                .and(&d1_qdot)
                .and(&dynamic.dqdot_td)
                .for_each(|s, &a, &b| *s = a * b);
            out + x_t_deriv.transpose_vector_multiply(&scratch)
        } else {
            // combined[i] = d1_q1[i]*dq_t_exit[i] + d1_q0[i]*dq_t_entry[i] + d1_qdot[i]*dqdot_t[i]
            ndarray::Zip::from(&mut scratch)
                .and(&d1_q1)
                .and(&dynamic.dq_t_exit)
                .and(&d1_q0)
                .and(&dynamic.dq_t_entry)
                .for_each(|s, &a, &b, &c, &d| *s = a * b + c * d);
            ndarray::Zip::from(&mut scratch)
                .and(&d1_qdot)
                .and(&dynamic.dqdot_t)
                .for_each(|s, &a, &b| *s += a * b);
            self.x_threshold.transpose_vector_multiply(&scratch)
        };

        let grad_ls = if let (Some(x_ls_entry), Some(x_ls_deriv)) = (
            self.x_log_sigma_entry.as_ref(),
            self.x_log_sigma_deriv.as_ref(),
        ) {
            ndarray::Zip::from(&mut scratch)
                .and(&d1_q1)
                .and(&dynamic.dq_ls_exit)
                .and(&d1_qdot)
                .and(&dynamic.dqdot_ls)
                .for_each(|s, &a, &b, &c, &d| *s = a * b + c * d);
            let mut out = self.x_log_sigma.transpose_vector_multiply(&scratch);
            ndarray::Zip::from(&mut scratch)
                .and(&d1_q0)
                .and(&dynamic.dq_ls_entry)
                .for_each(|s, &a, &b| *s = a * b);
            out = out + x_ls_entry.transpose_vector_multiply(&scratch);
            ndarray::Zip::from(&mut scratch)
                .and(&d1_qdot)
                .and(&dynamic.dqdot_lsd)
                .for_each(|s, &a, &b| *s = a * b);
            out + x_ls_deriv.transpose_vector_multiply(&scratch)
        } else {
            ndarray::Zip::from(&mut scratch)
                .and(&d1_q1)
                .and(&dynamic.dq_ls_exit)
                .and(&d1_q0)
                .and(&dynamic.dq_ls_entry)
                .for_each(|s, &a, &b, &c, &d| *s = a * b + c * d);
            ndarray::Zip::from(&mut scratch)
                .and(&d1_qdot)
                .and(&dynamic.dqdot_ls)
                .for_each(|s, &a, &b| *s += a * b);
            self.x_log_sigma.transpose_vector_multiply(&scratch)
        };

        let mut block_gradients = vec![grad_time, grad_t, grad_ls];
        if let (Some(xw_exit), Some(xw_entry), Some(xw_qdot)) = (
            dynamic.wiggle_basis_exit.as_ref(),
            dynamic.wiggle_basis_entry.as_ref(),
            dynamic.wiggle_qdot_basis_exit.as_ref(),
        ) {
            let gradw =
                xw_exit.t().dot(&d1_q1) + xw_entry.t().dot(&d1_q0) + xw_qdot.t().dot(&d1_qdot);
            block_gradients.push(gradw);
        }

        Ok((ll, block_gradients))
    }

    /// Build the [`BlockEffectiveJacobian`] for block `block_idx` given the
    /// realised block specs.
    ///
    /// Survival location-scale has three linear outputs per row:
    ///   - output 0: η_time       ← time_transform block (block 0)
    ///   - output 1: η_threshold  ← threshold block (block 1)
    ///   - output 2: η_log_sigma  ← log_sigma block (block 2)
    ///
    /// The optional linkwiggle block (block 3) modulates the inverse link
    /// nonlinearly and has an all-zero effective linear Jacobian.
    ///
    /// The stacked Jacobian for block k has shape `(3 * n, p_k)`.
    pub fn block_effective_jacobian(
        specs: &[ParameterBlockSpec],
        block_idx: usize,
    ) -> Result<Box<dyn BlockEffectiveJacobian>, String> {
        crate::block_layout::block_jacobian::AdditiveWiggleBlockLayout {
            family: "SurvivalLocationScaleFamily",
            n_outputs: 3,
            additive_blocks: &[
                Self::BLOCK_TIME,
                Self::BLOCK_THRESHOLD,
                Self::BLOCK_LOG_SIGMA,
            ],
            wiggle_block: Some(Self::BLOCK_LINK_WIGGLE),
        }
        .block_effective_jacobian(specs, block_idx)
    }
}

/// Per-subject 3×3 channel Hessian W_i for survival location-scale.
///
/// The three output channels are:
///   0. η_time   (time-transform, shared entry/exit predictor shift)
///   1. η_thr    (threshold block — shifts both u0 and u1 identically)
///   2. η_ls     (log-scale block — enters the inverse link)
///
/// The full W_i is the second derivative of the row NLL
/// `ρ_i(η_time, η_thr, η_ls)` at the current pilot β:
///
/// ```text
/// W_i[a, b] = ∂²ρ_i / ∂η_a ∂η_b
/// ```
///
/// These are the same second-order scalars computed by
/// `SurvivalLocationScaleFamily::row_derivatives_rescaled` but arranged
/// into the per-channel output-space matrix instead of the per-block
/// raw-coefficient space.
///
/// When the cross-channel curvature is unavailable (e.g. at the
/// canonicalize step before any pilot β is known), the identity metric
/// is used instead — see [`Self::identity`].
pub struct SurvivalLocationScaleChannelHessian {
    /// Row-major `(n × 3 × 3)` PSD-clamped per-subject Hessian.
    pub(crate) h: ndarray::Array3<f64>,
}

impl SurvivalLocationScaleChannelHessian {
    /// Number of output channels for SLS (always 3).
    pub const K: usize = 3;

    /// Construct from a pre-computed `(n × 3 × 3)` tensor.
    /// No PSD clamping is applied — caller is responsible for ensuring PSD.
    pub fn from_full(h: ndarray::Array3<f64>) -> Self {
        assert_eq!(
            h.shape()[1],
            Self::K,
            "SurvivalLocationScaleChannelHessian: expected K={} channels, got {}",
            Self::K,
            h.shape()[1],
        );
        assert_eq!(
            h.shape()[2],
            Self::K,
            "SurvivalLocationScaleChannelHessian: expected K={} channels, got {}",
            Self::K,
            h.shape()[2],
        );
        Self { h }
    }

    /// Structural identity metric: W_i = I₃ for every subject.
    ///
    /// Used at the canonicalize step where no pilot β is available. The
    /// identity metric gives the structurally correct rank answer (a block
    /// with zero Jacobian contributes no information regardless of the
    /// curvature).
    pub fn identity(n: usize) -> Self {
        let mut h = ndarray::Array3::<f64>::zeros((n, Self::K, Self::K));
        for i in 0..n {
            for c in 0..Self::K {
                h[[i, c, c]] = 1.0;
            }
        }
        Self { h }
    }
}

impl FamilyChannelHessian for SurvivalLocationScaleChannelHessian {
    fn n_outputs(&self) -> usize {
        Self::K
    }

    fn n_subjects(&self) -> usize {
        self.h.shape()[0]
    }

    fn fill_subject(&self, i: usize, out: &mut [f64]) {
        assert_eq!(out.len(), Self::K * Self::K);
        let k = Self::K;
        for a in 0..k {
            for b in 0..k {
                out[a * k + b] = self.h[[i, a, b]];
            }
        }
    }

    fn evaluate_full(&self) -> ndarray::Array3<f64> {
        self.h.clone()
    }
}

/// Public entry point for building a [`BlockEffectiveJacobian`] for one block of
/// the survival location-scale model.
///
/// This thin wrapper exposes the otherwise-private `SurvivalLocationScaleFamily`
/// associated function so integration tests and downstream crates can verify the
/// block Jacobian contract without depending on the internal struct.
///
/// See [`SurvivalLocationScaleFamily::block_effective_jacobian`] for the full
/// contract.
pub fn survival_location_scale_block_effective_jacobian(
    specs: &[ParameterBlockSpec],
    block_idx: usize,
) -> Result<Box<dyn BlockEffectiveJacobian>, String> {
    SurvivalLocationScaleFamily::block_effective_jacobian(specs, block_idx)
}

/// Observed vs expected information: The survival location-scale family uses
/// `BlockWorkingSet::ExactNewton` which provides the actual gradient and Hessian
/// (-nabla^2 log L) from the survival likelihood. This is the **observed** Hessian
/// by construction, which is the correct quantity for the outer REML Laplace
/// approximation (see response.md Section 3). No Fisher surrogate is used here.
//
// WS4a-survival-LS staged outer-score subsampling is enabled through
// Horvitz-Thompson row reweighting. The log-likelihood override streams the
// sampled rows through `exact_row_kernel` and multiplies each row contribution by
// `WeightedOuterRow.weight`. The joint-Hessian and ψ workspaces carry a shared
// row mask into the `_masked` assembly variants, where every row-additive
// `Xᵀ diag(W) Y`, `Xᵀ w`, and dot-product site multiplies the final per-row
// contribution by `mask[i]`. This deliberately masks after each row's nonlinear
// survival derivative algebra has produced the final row coefficient, preserving
// the invariant E[Σ_i (mask_i / π_i) contribution_i] = full-data sum.
impl CustomFamily for SurvivalLocationScaleFamily {
    // Survival location-scale fits keep the self-limiting Jeffreys/Firth
    // curvature active for their under-identification regime. The trait default
    // flipped to OFF in gam#1395 (flat-prior exact-Newton objective); opt in.
    fn joint_jeffreys_term_required(&self) -> bool {
        true
    }

    /// Batched all-axes first beta-directional derivative of the joint Jeffreys
    /// information, building the per-row joint quantities and dynamic geometry
    /// ONCE and sweeping every canonical axis from them.
    ///
    /// The trait default fans `p` independent per-axis calls across the Rayon
    /// pool, and each call re-enters
    /// `exact_newton_joint_hessian_directional_derivative` → `…_rescaled`,
    /// which rebuilds `collect_joint_quantities_rescaled` (the per-row
    /// third-order survival jet) and `build_dynamic_geometry` from scratch — so
    /// the same `O(n)` quantity build is recomputed `p` times per all-axes
    /// object. The Jeffreys/Firth conditioning gate keeps this object live on
    /// every armed inner-Newton cycle and outer Hessian eval (the
    /// under-identified constant-scale time ridge of #1389 keeps it armed), so
    /// the redundant rebuild is a real, repeated cost. Because
    /// `has_explicit_joint_hessian()` is unconditionally true, the default's
    /// per-axis `…_with_specs` chain reduces EXACTLY to
    /// `exact_newton_joint_hessian_directional_derivative_rescaled_from_parts`
    /// with `log_rescale = hessian_deriv_log_rescale(block_states)`, so reusing a
    /// single `(q, dynamic)` across the axis sweep is bit-identical to the
    /// default while paying the quantity build once instead of `p` times.
    fn joint_jeffreys_information_directional_derivative_all_axes_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Vec<Array2<f64>>>, String> {
        // Match the trait default's canonical-axis count (`Σ block design ncols`);
        // for this family it equals the joint block-offset width that
        // `…_from_parts` validates `d_beta` against.
        let p_total = specs.iter().map(|spec| spec.design.ncols()).sum::<usize>();
        if p_total == 0 {
            return Ok(None);
        }
        let log_rescale = self.hessian_deriv_log_rescale(block_states);
        let dynamic = self.build_dynamic_geometry(block_states)?;

        // Base (non-wiggle) path: sweep every canonical axis through the batched
        // all-axes dispatcher. The dispatcher routes to
        // `SurvivalLsRowKernel::directional_derivative_all_axes_dense_override`,
        // which builds the special-function-heavy per-row NLL derivative stack
        // (`row_nll_inputs` → `exact_row_kernel_rescaled`) ONCE and reuses it for
        // every one of the `p_total` axes. The previous per-axis loop ran
        // `p_total` independent single-direction sweeps, each rebuilding that
        // per-row stack from scratch — an `O(p_total · n · special-fn)` cost the
        // inner-Newton Jeffreys term and the outer-REML Jeffreys drift pay on
        // every joint evaluation. The dispatcher's override is bit-for-bit equal
        // to this loop (same kernel, same `RowSet::All` reduction); the wiggle
        // branch keeps the bespoke per-axis dense path the dispatcher does not
        // cover.
        if self.row_kernel_directional_supported() {
            let kernel = self.survival_ls_row_kernel_rescaled(&dynamic, log_rescale);
            let rows = crate::row_kernel::RowSet::All;
            let axes =
                crate::row_kernel::row_kernel_directional_derivative_all_axes(&kernel, &rows)?;
            return Ok(Some(axes));
        }

        let mut axes = Vec::with_capacity(p_total);
        for a in 0..p_total {
            let mut e_a = Array1::<f64>::zeros(p_total);
            e_a[a] = 1.0;
            match self.exact_newton_joint_hessian_directional_derivative_rescaled_from_parts(
                &e_a,
                &dynamic,
                log_rescale,
            )? {
                Some(m) => axes.push(m),
                None => return Ok(None),
            }
        }
        Ok(Some(axes))
    }

    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    /// Declare the per-block output channel so the pre-fit identifiability
    /// audit routes **channel-aware** (`audit_identifiability_channel_aware`)
    /// instead of the flat n-row Euclidean stack.
    ///
    /// The survival location-scale row NLL `ρ_i(η_time, η_thr, η_ls)` has THREE
    /// output channels (see `SurvivalLocationScaleChannelHessian`):
    ///   - channel 0 — `η_time` (time-transform predictor shift), and the
    ///     link-wiggle correction anchors here (it perturbs the inverse link
    ///     applied on the time/location side; cf. `AdditiveWiggleBlockLayout`
    ///     in `block_effective_jacobian`, which anchors the wiggle at output 0),
    ///   - channel 1 — `η_thr` (threshold / **location** predictor),
    ///   - channel 2 — `η_ls`  (log-σ / **scale** predictor, entering the
    ///     inverse link multiplicatively).
    ///
    /// Without this assignment the flat audit stacks every block's design into
    /// one n-row Euclidean space, so the threshold's **location intercept**
    /// (a `ones` column on channel 1) and the log-σ block's **scale intercept**
    /// (a `ones` column on channel 2) look like two copies of the same constant
    /// and the joint RRQR reports a (spurious) rank deficiency. The audit then
    /// drops one of them by gauge priority, collapsing a genuine free parameter
    /// and pinning a time-invariant covariate's coefficient to exactly 0
    /// (gam#1110: `gam_a_age = 0`). Both intercepts are separately identifiable
    /// — they live on orthogonal likelihood channels — and the channel-aware
    /// audit recognises this, returning a clean (identity-gauge) verdict with no
    /// column surgery, so every block keeps its raw width (no #1068 z-lift /
    /// fixed-col / monotonicity desync) and the location/scale parameters
    /// recover to the survreg/lifelines MLE.
    ///
    /// `wire_output_channels` installs an `AdditiveBlockJacobian` on each block
    /// from this assignment (the blocks carry no explicit `jacobian_callback`);
    /// that callback feeds ONLY the audit — the inner exact-Newton solve maps
    /// β→η through `solver_design()` (the stacked `[exit; entry; deriv]`
    /// operator), which never reads `jacobian_callback`, so the channel wiring
    /// is invisible to the fit itself. This mirrors the survival marginal-slope
    /// family, which wires its own multi-output Jacobian for the same reason.
    fn output_channel_assignment(&self, specs: &[ParameterBlockSpec]) -> Option<Vec<usize>> {
        Some(
            specs
                .iter()
                .map(|spec| match spec.name.as_str() {
                    "time_transform" => 0,
                    "threshold" => 1,
                    "log_sigma" => 2,
                    // The link-wiggle / time-wiggle corrections perturb the
                    // time/location-side inverse link; anchor them on the time
                    // channel exactly as `block_effective_jacobian` does
                    // (`AdditiveWiggleBlockLayout::wiggle_block` → output 0).
                    _ => 0,
                })
                .collect(),
        )
    }

    fn coefficient_hessian_cost(&self, specs: &[crate::custom_family::ParameterBlockSpec]) -> u64 {
        // Survival location-scale couples its blocks (threshold/time/log-σ
        // and any link/time wiggles) through the survival likelihood: every
        // row contributes a dense outer-product over (Σ p_b) coefficients.
        // At large scale the joint outer evaluator routes the coefficient
        // Hessian through its matrix-free HVP path; the cost remains an honest
        // dense-assembly diagnostic, while exact outer derivative order is now
        // driven by the explicit outer-HVP capability below rather than by a
        // first-order downgrade gate.
        crate::custom_family::joint_coupled_coefficient_hessian_cost(self.n as u64, specs)
    }

    fn outer_hyper_hessian_hvp_available(
        &self,
        specs: &[crate::custom_family::ParameterBlockSpec],
    ) -> bool {
        self.validate_joint_specs(
            specs,
            "SurvivalLocationScaleFamily outer hyper Hessian HVP availability",
        )
        .is_ok()
    }

    fn outer_hyper_hessian_dense_available(
        &self,
        specs: &[crate::custom_family::ParameterBlockSpec],
    ) -> bool {
        let p_total: usize = specs.iter().map(|spec| spec.design.ncols()).sum();
        !crate::custom_family::use_joint_matrix_free_path(p_total, self.n)
    }

    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let (ll, block_gradients) =
            self.evaluate_log_likelihood_and_block_gradients(block_states)?;

        // Every non-wiggle block is a view of the same packed 24-pair row
        // program lowering. Cross-block groups are never materialized for this
        // target. Link-wiggle geometry has a beta-dependent Jacobian and uses its
        // canonical runtime-sized row program, then slices the same dense result.
        let dynamic = self.build_dynamic_geometry(block_states)?;
        let block_hessians = if self.x_link_wiggle.is_some() {
            let dense =
                super::row_kernel::survival_ls_wiggle_joint_hessian_dense(self, &dynamic, 0.0)?;
            let offsets = self.joint_block_offsets();
            offsets
                .windows(2)
                .map(|bounds| {
                    dense
                        .slice(s![bounds[0]..bounds[1], bounds[0]..bounds[1]])
                        .to_owned()
                })
                .collect()
        } else {
            self.survival_ls_coefficient_hessian(
                &dynamic,
                0.0,
                None,
                SlsCoefficientHessianTarget::BlockDiagonal,
            )?
            .into_block_diagonal()?
        };
        if block_hessians.len() != block_gradients.len() {
            return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
                "SurvivalLocationScaleFamily evaluate block count mismatch: gradients={}, hessians={}",
                block_gradients.len(),
                block_hessians.len()
            ) }.into());
        }
        let blockworking_sets = block_gradients
            .into_iter()
            .zip(block_hessians)
            .map(|(gradient, hessian)| BlockWorkingSet::ExactNewton {
                gradient,
                hessian: SymmetricMatrix::Dense(hessian),
            })
            .collect();
        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets,
        })
    }

    fn log_likelihood_only(&self, block_states: &[ParameterBlockState]) -> Result<f64, String> {
        // Fast path for backtracking line search: compute only the scalar
        // log-likelihood, skipping all gradient/Hessian/derivative assembly.
        let n = self.n;
        let dynamic = self.build_dynamic_geometry(block_states)?;

        let row_log_likelihood = |i: usize| -> Result<f64, String> {
            let state = self.row_predictor_state(
                dynamic.h_entry[i],
                dynamic.h_exit[i],
                dynamic.hdot_exit[i],
                dynamic.q_entry[i],
                dynamic.q_exit[i],
                dynamic.qdot_exit[i],
            );
            Ok(self
                .exact_row_kernel(i, state)?
                .map_or(0.0, SurvivalExactRowKernel::log_likelihood))
        };

        const PARALLEL_LOG_LIKELIHOOD_ROW_THRESHOLD: usize = 1024;
        const LOG_LIKELIHOOD_CHUNK_ROWS: usize = 1024;
        if n < PARALLEL_LOG_LIKELIHOOD_ROW_THRESHOLD {
            let mut ll = 0.0;
            for i in 0..n {
                ll += row_log_likelihood(i)?;
            }
            return Ok(ll);
        }

        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let chunk_sums: Vec<Result<f64, String>> = (0..n.div_ceil(LOG_LIKELIHOOD_CHUNK_ROWS))
            .into_par_iter()
            .map(|chunk_idx| {
                let start = chunk_idx * LOG_LIKELIHOOD_CHUNK_ROWS;
                let end = (start + LOG_LIKELIHOOD_CHUNK_ROWS).min(n);
                let mut ll = 0.0;
                for i in start..end {
                    ll += row_log_likelihood(i)?;
                }
                Ok(ll)
            })
            .collect();

        let mut ll = 0.0;
        for chunk_sum in chunk_sums {
            ll += chunk_sum?;
        }
        Ok(ll)
    }

    fn log_likelihood_only_with_options(
        &self,
        block_states: &[ParameterBlockState],
        options: &BlockwiseFitOptions,
    ) -> Result<f64, String> {
        let Some(subsample) = options.outer_score_subsample.as_ref() else {
            return self.log_likelihood_only(block_states);
        };
        let n = self.n;
        let dynamic = self.build_dynamic_geometry(block_states)?;
        let mut ll = 0.0;
        for row in subsample.rows.as_ref() {
            let i = row.index;
            if i >= n {
                return Err(SurvivalLocationScaleError::DimensionMismatch {
                    reason: format!(
                        "SurvivalLocationScaleFamily outer subsample row index {i} out of bounds for n={n}"
                    ),
                }
                .into());
            }
            let state = self.row_predictor_state(
                dynamic.h_entry[i],
                dynamic.h_exit[i],
                dynamic.hdot_exit[i],
                dynamic.q_entry[i],
                dynamic.q_exit[i],
                dynamic.qdot_exit[i],
            );
            ll += row.weight
                * self
                    .exact_row_kernel(i, state)?
                    .map_or(0.0, SurvivalExactRowKernel::log_likelihood);
        }
        Ok(ll)
    }

    fn exact_newton_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        d_beta: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let dims = self.joint_block_dims();
        if block_idx >= dims.len() {
            return Ok(None);
        }
        if d_beta.len() != dims[block_idx] {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "block {block_idx} d_beta length mismatch: got {}, expected {}",
                    d_beta.len(),
                    dims[block_idx]
                ),
            }
            .into());
        }
        let offsets = self.joint_block_offsets();
        let mut d_beta_flat = Array1::<f64>::zeros(*offsets.last().unwrap());
        d_beta_flat
            .slice_mut(s![offsets[block_idx]..offsets[block_idx + 1]])
            .assign(d_beta);
        // The block-level directional derivative must differentiate the
        // UNRESCALED Hessian (from exact_newton_joint_hessian / evaluate()),
        // not the rescaled one used in the outer curvature path.  Pass
        // log_rescale = 0 so quantities match what evaluate() returns.
        let d_joint = self
            .exact_newton_joint_hessian_directional_derivative_rescaled(
                block_states,
                &d_beta_flat,
                0.0,
            )?
            .ok_or_else(|| {
                "missing survival location-scale exact joint directional Hessian".to_string()
            })?;
        Ok(Some(
            d_joint
                .slice(s![
                    offsets[block_idx]..offsets[block_idx + 1],
                    offsets[block_idx]..offsets[block_idx + 1]
                ])
                .to_owned(),
        ))
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        let dynamic = self.build_dynamic_geometry(block_states)?;
        if self.x_link_wiggle.is_some() {
            return Ok(Some(
                super::row_kernel::survival_ls_wiggle_joint_hessian_dense(self, &dynamic, 0.0)?,
            ));
        }
        let dense = self
            .survival_ls_coefficient_hessian(
                &dynamic,
                0.0,
                None,
                SlsCoefficientHessianTarget::DenseFull,
            )?
            .into_dense_full()?;
        Ok(Some(dense))
    }

    fn exact_newton_joint_gradient_evaluation(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<ExactNewtonJointGradientEvaluation>, String> {
        let (log_likelihood, block_gradients) =
            self.evaluate_log_likelihood_and_block_gradients(block_states)?;
        if block_gradients.len() != specs.len() {
            return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
                "SurvivalLocationScaleFamily joint gradient block count mismatch: gradients={}, specs={}",
                block_gradients.len(),
                specs.len()
            ) }.into());
        }

        let total_p = specs.iter().map(|spec| spec.design.ncols()).sum::<usize>();
        let mut gradient = Array1::<f64>::zeros(total_p);
        let mut offset = 0usize;
        for (block_idx, (block_gradient, spec)) in
            block_gradients.iter().zip(specs.iter()).enumerate()
        {
            let width = spec.design.ncols();
            if block_gradient.len() != width {
                return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
                    "SurvivalLocationScaleFamily joint gradient length mismatch for block {block_idx}: got {}, expected {}",
                    block_gradient.len(),
                    width
                ) }.into());
            }
            gradient
                .slice_mut(s![offset..offset + width])
                .assign(block_gradient);
            offset += width;
        }

        Ok(Some(ExactNewtonJointGradientEvaluation {
            log_likelihood,
            gradient,
        }))
    }

    fn has_explicit_joint_hessian(&self) -> bool {
        true
    }

    fn exact_newton_outer_curvature(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<ExactNewtonOuterCurvature>, String> {
        Ok(self
            .exact_newton_joint_hessian_rescaled(block_states)?
            .map(|(hessian, log_scale)| {
                let p = hessian.nrows();
                ExactNewtonOuterCurvature {
                    hessian,
                    rho_curvature_scale: (-log_scale).exp(),
                    hessian_logdet_correction: p as f64 * log_scale,
                }
            }))
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        // The trait method uses the full rescale for the outer curvature path.
        self.exact_newton_joint_hessian_directional_derivative_rescaled(
            block_states,
            d_beta_flat,
            self.hessian_deriv_log_rescale(block_states),
        )
    }

    fn exact_newton_joint_psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        hyper_layout: &CustomFamilyHyperLayout,
        psi_index: usize,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        if hyper_layout.family_axis_count() != 0 {
            return Err("SurvivalLocationScaleFamily does not declare family-owned hyper axes"
                .to_string());
        }
        self.exact_newton_joint_psi_terms_masked(
            block_states,
            specs,
            hyper_layout.design_derivative_blocks(),
            psi_index,
            None,
        )
    }

    fn exact_newton_joint_psisecond_order_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        hyper_layout: &CustomFamilyHyperLayout,
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        if hyper_layout.family_axis_count() != 0 {
            return Err("SurvivalLocationScaleFamily does not declare family-owned hyper axes"
                .to_string());
        }
        let derivative_blocks = hyper_layout.design_derivative_blocks();
        if block_states.len() != self.expected_blocks()
            || derivative_blocks.len() != self.expected_blocks()
        {
            return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
                "SurvivalLocationScaleFamily joint psi second-order terms expect {} states and derivative blocks, got {} / {}",
                self.expected_blocks(),
                block_states.len(),
                derivative_blocks.len()
            ) }.into());
        }
        self.validate_joint_specs(
            specs,
            "SurvivalLocationScaleFamily joint psi second-order terms",
        )?;
        let psi_dim = derivative_blocks.iter().map(Vec::len).sum::<usize>();
        if psi_i >= psi_dim || psi_j >= psi_dim {
            return Ok(None);
        }
        Ok(None)
    }

    fn exact_newton_joint_psi_workspace(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        hyper_layout: &CustomFamilyHyperLayout,
    ) -> Result<Option<Arc<dyn ExactNewtonJointPsiWorkspace>>, String> {
        if hyper_layout.family_axis_count() != 0 {
            return Err("SurvivalLocationScaleFamily does not declare family-owned hyper axes"
                .to_string());
        }
        let derivative_blocks = hyper_layout.design_derivative_blocks();
        if block_states.len() != self.expected_blocks()
            || specs.len() != self.expected_blocks()
            || derivative_blocks.len() != self.expected_blocks()
        {
            return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
                "SurvivalLocationScaleFamily joint psi workspace expects {} states, specs, and derivative blocks, got {} / {} / {}",
                self.expected_blocks(),
                block_states.len(),
                specs.len(),
                derivative_blocks.len()
            ) }.into());
        }
        Ok(Some(Arc::new(SurvivalExactNewtonJointPsiWorkspace::new(
            self.clone(),
            block_states.to_vec(),
            specs.to_vec(),
            derivative_blocks.to_vec(),
        )?)))
    }

    fn exact_newton_joint_psi_workspace_with_options(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        hyper_layout: &CustomFamilyHyperLayout,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Arc<dyn ExactNewtonJointPsiWorkspace>>, String> {
        if hyper_layout.family_axis_count() != 0 {
            return Err("SurvivalLocationScaleFamily does not declare family-owned hyper axes"
                .to_string());
        }
        let derivative_blocks = hyper_layout.design_derivative_blocks();
        if block_states.len() != self.expected_blocks()
            || specs.len() != self.expected_blocks()
            || derivative_blocks.len() != self.expected_blocks()
        {
            return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
                "SurvivalLocationScaleFamily joint psi workspace expects {} states, specs, and derivative blocks, got {} / {} / {}",
                self.expected_blocks(),
                block_states.len(),
                specs.len(),
                derivative_blocks.len()
            ) }.into());
        }
        let mut workspace = SurvivalExactNewtonJointPsiWorkspace::new(
            self.clone(),
            block_states.to_vec(),
            specs.to_vec(),
            derivative_blocks.to_vec(),
        )?;
        if let Some(subsample) = options.outer_score_subsample.as_ref() {
            workspace.apply_outer_subsample(subsample.rows.as_ref());
        }
        Ok(Some(Arc::new(workspace)))
    }

    fn exact_newton_joint_psihessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        hyper_layout: &CustomFamilyHyperLayout,
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if hyper_layout.family_axis_count() != 0 {
            return Err("SurvivalLocationScaleFamily does not declare family-owned hyper axes"
                .to_string());
        }
        let derivative_blocks = hyper_layout.design_derivative_blocks();
        if block_states.len() != self.expected_blocks()
            || derivative_blocks.len() != self.expected_blocks()
        {
            return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
                "SurvivalLocationScaleFamily joint psi Hessian directional derivative expects {} states and derivative blocks, got {} / {}",
                self.expected_blocks(),
                block_states.len(),
                derivative_blocks.len()
            ) }.into());
        }
        self.validate_joint_specs(
            specs,
            "SurvivalLocationScaleFamily joint psi Hessian directional derivative",
        )?;
        let p_total = *self
            .joint_block_offsets()
            .last()
            .ok_or_else(|| "missing joint block offsets".to_string())?;
        if d_beta_flat.len() != p_total {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "joint psi Hessian directional derivative d_beta length mismatch: got {}, expected {p_total}",
                    d_beta_flat.len()
                ),
            }
            .into());
        }
        let psi_dim = derivative_blocks.iter().map(Vec::len).sum::<usize>();
        if psi_index >= psi_dim {
            return Ok(None);
        }
        Ok(None)
    }

    fn exact_newton_joint_hessiansecond_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        crate::block_layout::block_count::validate_block_count::<SurvivalLocationScaleError>(
            "SurvivalLocationScaleFamily joint Hessian second directional derivative",
            self.expected_blocks(),
            block_states.len(),
        )?;
        let p_total = *self
            .joint_block_offsets()
            .last()
            .ok_or_else(|| "missing joint block offsets".to_string())?;
        if d_beta_u_flat.len() != p_total || d_beta_v_flat.len() != p_total {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "joint Hessian second directional derivative length mismatch: got {} / {}, expected {p_total}",
                    d_beta_u_flat.len(),
                    d_beta_v_flat.len()
                ),
            }
            .into());
        }
        let log_rescale = self.hessian_deriv_log_rescale(block_states);
        let dynamic = self.build_dynamic_geometry(block_states)?;
        if self.x_link_wiggle.is_some() {
            // #932: single-source the wiggle SECOND directional derivative via
            // the §13 warp kernel (`TwoSeed<KW>`). Previously this returned
            // `None` (the wiggle carve-out provided no second-directional
            // curvature); now wiggle rows get the exact ε,δ-Hessian channel of
            // the same single-sourced row NLL, matching the non-wiggle base path.
            return Ok(Some(
                super::row_kernel::survival_ls_wiggle_second_directional_derivative_dense(
                    self,
                    &dynamic,
                    log_rescale,
                    &crate::row_kernel::RowSet::All,
                    d_beta_u_flat.as_slice().ok_or_else(|| {
                        "joint Hessian second directional u must be contiguous".to_string()
                    })?,
                    d_beta_v_flat.as_slice().ok_or_else(|| {
                        "joint Hessian second directional v must be contiguous".to_string()
                    })?,
                )?,
            ));
        }
        let kernel = self.survival_ls_row_kernel_rescaled(&dynamic, log_rescale);
        crate::row_kernel::row_kernel_second_directional_derivative(
            &kernel,
            &crate::row_kernel::RowSet::All,
            d_beta_u_flat.as_slice().ok_or_else(|| {
                "joint Hessian second directional u must be contiguous".to_string()
            })?,
            d_beta_v_flat.as_slice().ok_or_else(|| {
                "joint Hessian second directional v must be contiguous".to_string()
            })?,
        )
        .map(Some)
    }

    fn block_linear_constraints(
        &self,
        _: &[ParameterBlockState],
        block_idx: usize,
        spec: &ParameterBlockSpec,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        if block_idx == Self::BLOCK_LINK_WIGGLE {
            return Ok(monotone_wiggle_nonnegative_constraints(spec.design.ncols()));
        }
        if block_idx != Self::BLOCK_TIME {
            return Ok(None);
        }
        Ok(self.time_linear_constraints.clone())
    }

    fn max_feasible_step_size(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        delta: &Array1<f64>,
    ) -> Result<Option<f64>, String> {
        if block_idx == Self::BLOCK_TIME {
            return self.max_feasible_time_step(&block_states[Self::BLOCK_TIME].beta, delta);
        }
        if block_idx == Self::BLOCK_LINK_WIGGLE {
            return self
                .max_feasible_link_wiggle_step(&block_states[Self::BLOCK_LINK_WIGGLE].beta, delta);
        }
        Ok(None)
    }

    fn joint_trust_metric_block_floor(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
    ) -> Result<Option<Array1<f64>>, String> {
        // Scale-aware trust-metric floor for the coupled smooth-scale fit
        // (issue #1569). The free scale predictor `η_σ` enters the likelihood
        // through the standardized index `u = inv_sigma·(h − η_t)` with
        // `inv_sigma = exp(−η_σ)`, so `∂u/∂η_t = −inv_sigma`: the LOCATION
        // (threshold) and LOG-σ channels — but NOT the flexible time baseline
        // `h`, whose `∂u/∂h = 1` is scale-free — carry an `exp(−η_σ)` factor in
        // their gradient and an `exp(−2 η_σ)` factor in their likelihood-Hessian
        // diagonal. When the scale predictor drives some rows to small σ (large
        // `exp(−η_σ)`), a location/log-σ coefficient loading mostly on the
        // large-σ rows is METRIC-STARVED relative to one loading on the small-σ
        // rows; the affine-covariant Moré–Sorensen step then over-reaches on the
        // starved coordinate, the gain ratio never justifies growing the radius,
        // and the inner solve grinds. We floor each scale-coupled block's metric
        // entries at `SCALE_COUPLED_TRUST_METRIC_FLOOR_REL × (block max metric)`,
        // capping the `exp(−η_σ)`-induced metric condition number so no
        // coordinate is starved. The floor is derived ENTIRELY from the
        // scale-coupled Hessian diagonal (no knob); `max(D_i, floor_i)` can only
        // tighten the metric and self-vanishes at the KKT fixed point.
        let offsets = self.joint_block_offsets();
        if offsets.len() < 2 {
            return Ok(None);
        }
        let p_total = *offsets
            .last()
            .ok_or_else(|| "missing joint block offsets".to_string())?;
        // Lower only the joint likelihood diagonal from the scale-stabilized
        // packed coefficients. The uniform `exp(−L)` rescale cancels in the
        // relative floor `fraction × max(diag)`, so the floor is scale-invariant.
        let log_scale = self.hessian_deriv_log_rescale(block_states);
        let dynamic = self.build_dynamic_geometry(block_states)?;
        let h_diagonal = if self.x_link_wiggle.is_some() {
            super::row_kernel::survival_ls_wiggle_joint_hessian_dense(self, &dynamic, log_scale)?
                .diag()
                .to_owned()
        } else {
            self.survival_ls_coefficient_hessian(
                &dynamic,
                log_scale,
                None,
                SlsCoefficientHessianTarget::DiagonalOnly,
            )?
            .into_diagonal_only()?
        };
        if h_diagonal.len() != p_total {
            return Ok(None);
        }
        let mut floor = Array1::<f64>::zeros(p_total);
        let mut any = false;
        // Floor every scale-coupled block: the LOCATION (threshold) and LOG-σ
        // channels carry the `exp(−η_σ)` factor; the flexible time baseline does
        // NOT, so it is deliberately excluded.
        for &block in &[Self::BLOCK_THRESHOLD, Self::BLOCK_LOG_SIGMA] {
            if block + 1 >= offsets.len() {
                continue;
            }
            let (start, end) = (offsets[block], offsets[block + 1]);
            if end <= start {
                continue;
            }
            let max_diag = (start..end)
                .map(|j| h_diagonal[j].abs())
                .filter(|v| v.is_finite())
                .fold(0.0_f64, f64::max);
            if !(max_diag.is_finite() && max_diag > 0.0) {
                continue;
            }
            let floor_value = SCALE_COUPLED_TRUST_METRIC_FLOOR_REL * max_diag;
            if !(floor_value.is_finite() && floor_value > 0.0) {
                continue;
            }
            for j in start..end {
                floor[j] = floor_value;
            }
            any = true;
        }
        if any { Ok(Some(floor)) } else { Ok(None) }
    }

    fn post_update_block_beta(
        &self,
        _: &[ParameterBlockState],
        block_idx: usize,
        block_spec: &ParameterBlockSpec,
        beta: Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        assert!(!block_spec.name.is_empty());
        if block_idx == Self::BLOCK_TIME
            && let Some(constraints) = self.time_linear_constraints.as_ref()
        {
            validate_linear_constraints("time post-update", &beta, constraints)?;
        } else if block_idx == Self::BLOCK_LINK_WIGGLE && self.x_link_wiggle.is_some() {
            for j in 0..beta.len() {
                let tol = CONSTRAINT_NONNEGATIVITY_REL_TOL * beta[j].abs().max(1.0);
                if !beta[j].is_finite() || beta[j] < -tol {
                    return Err(SurvivalLocationScaleError::ConstraintViolation {
                        reason: format!(
                            "survival location-scale link-wiggle post-update violates represented nonnegativity at coefficient {j}: value={:.3e}, tol={:.3e}",
                            beta[j], tol
                        ),
                    }
                    .into());
                }
            }
        }
        Ok(beta)
    }

    fn exact_newton_joint_hessian_workspace(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        self.validate_joint_specs(specs, "SurvivalLocationScaleFamily joint Hessian workspace")?;
        // The wrapper owns the precomputed survival quantities/dynamic geometry
        // and routes non-wiggle Hessian derivative calls through the RowKernel
        // engine. Link-wiggle still stays on the existing family algebra because
        // its row design depends on beta and is outside the fixed-Jacobian
        // RowKernel contract.
        Ok(Some(Arc::new(
            SurvivalLocationScaleExactNewtonJointHessianWorkspace::new(
                self.clone(),
                block_states.to_vec(),
            )?,
        )))
    }

    fn exact_newton_joint_hessian_workspace_with_options(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        self.validate_joint_specs(
            specs,
            "SurvivalLocationScaleFamily joint Hessian workspace with options",
        )?;
        // See the non-options workspace constructor above. The HT row mask is
        // threaded into the supported RowKernel derivative paths by
        // `row_set_from_survival_mask`.
        let mut workspace = SurvivalLocationScaleExactNewtonJointHessianWorkspace::new(
            self.clone(),
            block_states.to_vec(),
        )?;
        if let Some(subsample) = options.outer_score_subsample.as_ref() {
            workspace.apply_outer_subsample(subsample.rows.as_ref());
        } else {
            workspace.clear_outer_subsample();
        }
        Ok(Some(Arc::new(workspace)))
    }

    fn outer_derivative_subsample_capable(&self) -> bool {
        true
    }

    // Inherent `exact_newton_joint_psi_terms_masked` is defined in the
    // `impl SurvivalLocationScaleFamily` block below. It is invoked directly
    // by both this trait method and the ψ workspace's `first_order_terms`
    // override to thread the Horvitz-Thompson row mask through the staged
    // outer-score subsample.
}

impl SurvivalLocationScaleFamily {
    /// HT-mask-aware variant of [`Self::exact_newton_joint_psi_terms`].
    ///
    /// Lives in an inherent impl (not the `impl CustomFamily` trait impl)
    /// because the trait does not declare a `_masked` signature. The survival
    /// ψ workspace overrides `first_order_terms` to invoke this directly with
    /// the workspace's `row_mask`, so the trait dispatch stays on the
    /// pre-refactor `exact_newton_joint_psi_terms` (full data) while staged
    /// outer subsampling threads the HT mask through this side door.
    pub(crate) fn exact_newton_joint_psi_terms_masked(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        row_mask: Option<&Array1<f64>>,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        if specs.len() != self.expected_blocks()
            || derivative_blocks.len() != self.expected_blocks()
        {
            return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
                "SurvivalLocationScaleFamily joint psi terms expect {} specs and derivative blocks, got {} and {}",
                self.expected_blocks(),
                specs.len(),
                derivative_blocks.len()
            ) }.into());
        }
        let Some(dir) =
            self.exact_newton_joint_psi_direction(block_states, derivative_blocks, psi_index)?
        else {
            return Ok(None);
        };
        let z_t_exit_psi = &dir.z_t_exit_psi;
        let z_t_entry_psi = &dir.z_t_entry_psi;
        let z_ls_exit_psi = &dir.z_ls_exit_psi;
        let z_ls_entry_psi = &dir.z_ls_entry_psi;
        let q = self.collect_joint_quantities(block_states)?;
        let dynamic = self.build_dynamic_geometry(block_states)?;
        let offsets = self.joint_block_offsets();
        let p_total = *offsets
            .last()
            .ok_or_else(|| "missing joint block offsets".to_string())?;

        let x_threshold_exit_cow = self.x_threshold.to_dense_cow();
        let x_threshold_exit = &*x_threshold_exit_cow;
        let x_threshold_entry_cow = self
            .x_threshold_entry
            .as_ref()
            .map(DesignMatrix::to_dense_cow);
        let x_threshold_entry = x_threshold_entry_cow
            .as_ref()
            .map_or(x_threshold_exit, |c| &**c);
        let x_log_sigma_exit_cow = self.x_log_sigma.to_dense_cow();
        let x_log_sigma_exit = &*x_log_sigma_exit_cow;
        let x_log_sigma_entry_cow = self
            .x_log_sigma_entry
            .as_ref()
            .map(DesignMatrix::to_dense_cow);
        let x_log_sigma_entry = x_log_sigma_entry_cow
            .as_ref()
            .map_or(x_log_sigma_exit, |c| &**c);
        let xw_cow = self.x_link_wiggle.as_ref().map(DesignMatrix::to_dense_cow);
        let xw = xw_cow.as_deref();
        let x_t_exit_map = first_psi_linear_map(
            dir.x_t_exit_action.as_ref(),
            dir.x_t_exit_psi.as_ref(),
            self.n,
            x_threshold_exit.ncols(),
        );
        let x_t_entry_map = first_psi_linear_map(
            dir.x_t_entry_action.as_ref(),
            dir.x_t_entry_psi.as_ref(),
            self.n,
            x_threshold_entry.ncols(),
        );
        let x_ls_exit_map = first_psi_linear_map(
            dir.x_ls_exit_action.as_ref(),
            dir.x_ls_exit_psi.as_ref(),
            self.n,
            x_log_sigma_exit.ncols(),
        );
        let x_ls_entry_map = first_psi_linear_map(
            dir.x_ls_entry_action.as_ref(),
            dir.x_ls_entry_psi.as_ref(),
            self.n,
            x_log_sigma_entry.ncols(),
        );

        let dq_t_entry = q.dq_t_entry.as_ref().unwrap_or(&q.dq_t);
        let dq_ls_entry = q.dq_ls_entry.as_ref().unwrap_or(&q.dq_ls);
        let d2q_tls_entry = q.d2q_tls_entry.as_ref().unwrap_or(&q.d2q_tls);
        let d2q_ls_entry = q.d2q_ls_entry.as_ref().unwrap_or(&q.d2q_ls);
        let d3q_tls_ls_entry = q.d3q_tls_ls_entry.as_ref().unwrap_or(&q.d3q_tls_ls);
        let d3q_ls_entry = q.d3q_ls_entry.as_ref().unwrap_or(&q.d3q_ls);

        let q0_psi = &(dq_t_entry * z_t_entry_psi) + &(dq_ls_entry * z_ls_entry_psi);
        let q1_psi = &(&q.dq_t * z_t_exit_psi) + &(&q.dq_ls * z_ls_exit_psi);
        let dq_t_entry_psi = d2q_tls_entry * z_ls_entry_psi;
        let dq_t_exit_psi = &q.d2q_tls * z_ls_exit_psi;
        let dq_ls_entry_psi = d2q_tls_entry * z_t_entry_psi + d2q_ls_entry * z_ls_entry_psi;
        let dq_ls_exit_psi = &q.d2q_tls * z_t_exit_psi + &q.d2q_ls * z_ls_exit_psi;
        let d2q_tls_entry_psi = d3q_tls_ls_entry * z_ls_entry_psi;
        let d2q_tls_exit_psi = &q.d3q_tls_ls * z_ls_exit_psi;
        let d2q_ls_entry_psi = d3q_tls_ls_entry * z_t_entry_psi + d3q_ls_entry * z_ls_entry_psi;
        let d2q_ls_exit_psi = &q.d3q_tls_ls * z_t_exit_psi + &q.d3q_ls * z_ls_exit_psi;

        let objective_psi = if let Some(m) = row_mask {
            (&(&q.d1_q0 * &q0_psi) * m).sum() + (&(&q.d1_q1 * &q1_psi) * m).sum()
        } else {
            q.d1_q0.dot(&q0_psi) + q.d1_q1.dot(&q1_psi)
        };

        let mut score_psi = Array1::<f64>::zeros(p_total);
        let time_row_entry = -&q.d2_q0 * &q0_psi;
        let time_row_exit = -&q.d2_q1 * &q1_psi;
        let time_score = dynamic
            .time_jac_entry
            .t()
            .dot(&*mask_row_vec(&time_row_entry, row_mask))
            + dynamic
                .time_jac_exit
                .t()
                .dot(&*mask_row_vec(&time_row_exit, row_mask));
        score_psi
            .slice_mut(s![offsets[0]..offsets[1]])
            .assign(&time_score);

        let threshold_score_row_exit = &q.d1_q1 * &q.dq_t;
        let threshold_score_row_entry = &q.d1_q0 * dq_t_entry;
        let d_threshold_score_row_exit = &q.d2_q1 * &q1_psi * &q.dq_t + &q.d1_q1 * &dq_t_exit_psi;
        let d_threshold_score_row_entry =
            &q.d2_q0 * &q0_psi * dq_t_entry + &q.d1_q0 * &dq_t_entry_psi;
        let threshold_score = x_t_exit_map
            .transpose_mul(mask_row_vec(&threshold_score_row_exit, row_mask).view())
            + x_threshold_exit
                .t()
                .dot(&*mask_row_vec(&d_threshold_score_row_exit, row_mask))
            + x_t_entry_map
                .transpose_mul(mask_row_vec(&threshold_score_row_entry, row_mask).view())
            + x_threshold_entry
                .t()
                .dot(&*mask_row_vec(&d_threshold_score_row_entry, row_mask));
        score_psi
            .slice_mut(s![offsets[1]..offsets[2]])
            .assign(&threshold_score);

        let log_sigma_score_row_exit = &q.d1_q1 * &q.dq_ls;
        let log_sigma_score_row_entry = &q.d1_q0 * dq_ls_entry;
        let d_log_sigma_score_row_exit = &q.d2_q1 * &q1_psi * &q.dq_ls + &q.d1_q1 * &dq_ls_exit_psi;
        let d_log_sigma_score_row_entry =
            &q.d2_q0 * &q0_psi * dq_ls_entry + &q.d1_q0 * &dq_ls_entry_psi;
        let log_sigma_score = x_ls_exit_map
            .transpose_mul(mask_row_vec(&log_sigma_score_row_exit, row_mask).view())
            + x_log_sigma_exit
                .t()
                .dot(&*mask_row_vec(&d_log_sigma_score_row_exit, row_mask))
            + x_ls_entry_map
                .transpose_mul(mask_row_vec(&log_sigma_score_row_entry, row_mask).view())
            + x_log_sigma_entry
                .t()
                .dot(&*mask_row_vec(&d_log_sigma_score_row_entry, row_mask));
        score_psi
            .slice_mut(s![offsets[2]..offsets[3]])
            .assign(&log_sigma_score);

        if let (Some(xw_dense), Some(w_offset)) = (xw, offsets.get(3).copied()) {
            let wiggle_row = &q.d2_q0 * &q0_psi + &q.d2_q1 * &q1_psi;
            let wiggle_score = xw_dense.t().dot(&*mask_row_vec(&wiggle_row, row_mask));
            score_psi
                .slice_mut(s![w_offset..offsets[4]])
                .assign(&wiggle_score);
        }

        let h_time_time = mxtwxd(&dynamic.time_jac_entry, &(-&q.d3_q0 * &q0_psi), row_mask)
            + mxtwxd(&dynamic.time_jac_exit, &(-&q.d3_q1 * &q1_psi), row_mask);

        let h_tt_entry = -(&q.d2_q0 * &dq_t_entry.mapv(|v| safe_product(v, v)));
        let h_tt_exit = -(&q.d2_q1 * &q.dq_t.mapv(|v| safe_product(v, v)));
        let dh_tt_entry = -(&q.d3_q0 * &q0_psi * &dq_t_entry.mapv(|v| safe_product(v, v))
            + &(2.0 * &q.d2_q0 * dq_t_entry * &dq_t_entry_psi));
        let dh_tt_exit = -(&q.d3_q1 * &q1_psi * &q.dq_t.mapv(|v| safe_product(v, v))
            + &(2.0 * &q.d2_q1 * &q.dq_t * &dq_t_exit_psi));

        let h_ll_entry =
            -(&q.d2_q0 * &dq_ls_entry.mapv(|v| safe_product(v, v)) + &(&q.d1_q0 * d2q_ls_entry));
        let h_ll_exit =
            -(&q.d2_q1 * &q.dq_ls.mapv(|v| safe_product(v, v)) + &(&q.d1_q1 * &q.d2q_ls));
        let dh_ll_entry = -(&q.d3_q0 * &q0_psi * &dq_ls_entry.mapv(|v| safe_product(v, v))
            + &(2.0 * &q.d2_q0 * dq_ls_entry * &dq_ls_entry_psi)
            + &(&q.d2_q0 * &q0_psi * d2q_ls_entry)
            + &(&q.d1_q0 * &d2q_ls_entry_psi));
        let dh_ll_exit = -(&q.d3_q1 * &q1_psi * &q.dq_ls.mapv(|v| safe_product(v, v))
            + &(2.0 * &q.d2_q1 * &q.dq_ls * &dq_ls_exit_psi)
            + &(&q.d2_q1 * &q1_psi * &q.d2q_ls)
            + &(&q.d1_q1 * &d2q_ls_exit_psi));

        let h_tl_entry = -(&q.d2_q0 * &(dq_t_entry * dq_ls_entry) + &(&q.d1_q0 * d2q_tls_entry));
        let h_tl_exit = -(&q.d2_q1 * &(&q.dq_t * &q.dq_ls) + &(&q.d1_q1 * &q.d2q_tls));
        let dh_tl_entry = -(&q.d3_q0 * &q0_psi * &(dq_t_entry * dq_ls_entry)
            + &(&q.d2_q0 * &(&dq_t_entry_psi * dq_ls_entry + dq_t_entry * &dq_ls_entry_psi))
            + &(&q.d2_q0 * &q0_psi * d2q_tls_entry)
            + &(&q.d1_q0 * &d2q_tls_entry_psi));
        let dh_tl_exit = -(&q.d3_q1 * &q1_psi * &(&q.dq_t * &q.dq_ls)
            + &(&q.d2_q1 * &(&dq_t_exit_psi * &q.dq_ls + &q.dq_t * &dq_ls_exit_psi))
            + &(&q.d2_q1 * &q1_psi * &q.d2q_tls)
            + &(&q.d1_q1 * &d2q_tls_exit_psi));

        let h_h0_t = &q.d2_q0 * dq_t_entry;
        let h_h1_t = &q.d2_q1 * &q.dq_t;
        let dh_h0_t = &q.d3_q0 * &q0_psi * dq_t_entry + &q.d2_q0 * &dq_t_entry_psi;
        let dh_h1_t = &q.d3_q1 * &q1_psi * &q.dq_t + &q.d2_q1 * &dq_t_exit_psi;

        let h_h0_ls = &q.d2_q0 * dq_ls_entry;
        let h_h1_ls = &q.d2_q1 * &q.dq_ls;
        let dh_h0_ls = &q.d3_q0 * &q0_psi * dq_ls_entry + &q.d2_q0 * &dq_ls_entry_psi;
        let dh_h1_ls = &q.d3_q1 * &q1_psi * &q.dq_ls + &q.d2_q1 * &dq_ls_exit_psi;
        let h_tw_entry = -(&q.d2_q0 * dq_t_entry);
        let h_tw_exit = -(&q.d2_q1 * &q.dq_t);
        let dh_tw_entry = -(&q.d3_q0 * &q0_psi * dq_t_entry + &q.d2_q0 * &dq_t_entry_psi);
        let dh_tw_exit = -(&q.d3_q1 * &q1_psi * &q.dq_t + &q.d2_q1 * &dq_t_exit_psi);
        let h_lw_entry = -(&q.d2_q0 * dq_ls_entry);
        let h_lw_exit = -(&q.d2_q1 * &q.dq_ls);
        let dh_lw_entry = -(&q.d3_q0 * &q0_psi * dq_ls_entry + &q.d2_q0 * &dq_ls_entry_psi);
        let dh_lw_exit = -(&q.d3_q1 * &q1_psi * &q.dq_ls + &q.d2_q1 * &dq_ls_exit_psi);

        if dir.x_t_exit_action.is_some()
            || dir.x_t_entry_action.is_some()
            || dir.x_ls_exit_action.is_some()
            || dir.x_ls_entry_action.is_some()
        {
            // HT-mask helper. Each per-row pair weight (h_*, dh_*, ±d3·q_psi)
            // is multiplied by the mask before being moved into the deferred
            // operator. `None` is a zero-cost passthrough.
            let mw = |arr: Array1<f64>| -> Array1<f64> {
                match row_mask {
                    Some(m) => &arr * m,
                    None => arr,
                }
            };
            let mut channels = vec![
                CustomFamilyJointDesignChannel::new(
                    offsets[0]..offsets[1],
                    shared_dense_arc(&self.x_time_entry),
                    None,
                ),
                CustomFamilyJointDesignChannel::new(
                    offsets[0]..offsets[1],
                    shared_dense_arc(&self.x_time_exit),
                    None,
                ),
                CustomFamilyJointDesignChannel::new(
                    offsets[1]..offsets[2],
                    shared_dense_arc(x_threshold_exit),
                    dir.x_t_exit_action.clone(),
                ),
                CustomFamilyJointDesignChannel::new(
                    offsets[1]..offsets[2],
                    shared_dense_arc(x_threshold_entry),
                    dir.x_t_entry_action.clone(),
                ),
                CustomFamilyJointDesignChannel::new(
                    offsets[2]..offsets[3],
                    shared_dense_arc(x_log_sigma_exit),
                    dir.x_ls_exit_action.clone(),
                ),
                CustomFamilyJointDesignChannel::new(
                    offsets[2]..offsets[3],
                    shared_dense_arc(x_log_sigma_entry),
                    dir.x_ls_entry_action.clone(),
                ),
            ];
            let mut pairs = vec![
                CustomFamilyJointDesignPairContribution::new(
                    0,
                    0,
                    mw(Array1::zeros(self.x_time_entry.nrows())),
                    mw(-&q.d3_q0 * &q0_psi),
                ),
                CustomFamilyJointDesignPairContribution::new(
                    1,
                    1,
                    mw(Array1::zeros(self.x_time_exit.nrows())),
                    mw(-&q.d3_q1 * &q1_psi),
                ),
                CustomFamilyJointDesignPairContribution::new(
                    2,
                    2,
                    mw(h_tt_exit.clone()),
                    mw(dh_tt_exit.clone()),
                ),
                CustomFamilyJointDesignPairContribution::new(
                    3,
                    3,
                    mw(h_tt_entry.clone()),
                    mw(dh_tt_entry.clone()),
                ),
                CustomFamilyJointDesignPairContribution::new(
                    4,
                    4,
                    mw(h_ll_exit.clone()),
                    mw(dh_ll_exit.clone()),
                ),
                CustomFamilyJointDesignPairContribution::new(
                    5,
                    5,
                    mw(h_ll_entry.clone()),
                    mw(dh_ll_entry.clone()),
                ),
                CustomFamilyJointDesignPairContribution::new(
                    2,
                    4,
                    mw(h_tl_exit.clone()),
                    mw(dh_tl_exit.clone()),
                ),
                CustomFamilyJointDesignPairContribution::new(
                    4,
                    2,
                    mw(h_tl_exit.clone()),
                    mw(dh_tl_exit.clone()),
                ),
                CustomFamilyJointDesignPairContribution::new(
                    3,
                    5,
                    mw(h_tl_entry.clone()),
                    mw(dh_tl_entry.clone()),
                ),
                CustomFamilyJointDesignPairContribution::new(
                    5,
                    3,
                    mw(h_tl_entry.clone()),
                    mw(dh_tl_entry.clone()),
                ),
                CustomFamilyJointDesignPairContribution::new(
                    0,
                    3,
                    mw(h_h0_t.clone()),
                    mw(dh_h0_t.clone()),
                ),
                CustomFamilyJointDesignPairContribution::new(
                    3,
                    0,
                    mw(h_h0_t.clone()),
                    mw(dh_h0_t.clone()),
                ),
                CustomFamilyJointDesignPairContribution::new(
                    1,
                    2,
                    mw(h_h1_t.clone()),
                    mw(dh_h1_t.clone()),
                ),
                CustomFamilyJointDesignPairContribution::new(
                    2,
                    1,
                    mw(h_h1_t.clone()),
                    mw(dh_h1_t.clone()),
                ),
                CustomFamilyJointDesignPairContribution::new(
                    0,
                    5,
                    mw(h_h0_ls.clone()),
                    mw(dh_h0_ls.clone()),
                ),
                CustomFamilyJointDesignPairContribution::new(
                    5,
                    0,
                    mw(h_h0_ls.clone()),
                    mw(dh_h0_ls.clone()),
                ),
                CustomFamilyJointDesignPairContribution::new(
                    1,
                    4,
                    mw(h_h1_ls.clone()),
                    mw(dh_h1_ls.clone()),
                ),
                CustomFamilyJointDesignPairContribution::new(
                    4,
                    1,
                    mw(h_h1_ls.clone()),
                    mw(dh_h1_ls.clone()),
                ),
            ];
            if let (Some(xw_dense), Some(w_offset)) = (xw, offsets.get(3).copied()) {
                channels.push(CustomFamilyJointDesignChannel::new(
                    w_offset..offsets[4],
                    shared_dense_arc(xw_dense),
                    None,
                ));
                let w_idx = channels.len() - 1;
                let zero_w = Array1::zeros(xw_dense.nrows());
                pairs.push(CustomFamilyJointDesignPairContribution::new(
                    w_idx,
                    w_idx,
                    mw(zero_w.clone()),
                    mw(-&q.d3_q0 * &q0_psi - &q.d3_q1 * &q1_psi),
                ));
                pairs.push(CustomFamilyJointDesignPairContribution::new(
                    2,
                    w_idx,
                    mw(h_tw_exit.clone()),
                    mw(dh_tw_exit.clone()),
                ));
                pairs.push(CustomFamilyJointDesignPairContribution::new(
                    w_idx,
                    2,
                    mw(h_tw_exit.clone()),
                    mw(dh_tw_exit.clone()),
                ));
                pairs.push(CustomFamilyJointDesignPairContribution::new(
                    3,
                    w_idx,
                    mw(h_tw_entry.clone()),
                    mw(dh_tw_entry.clone()),
                ));
                pairs.push(CustomFamilyJointDesignPairContribution::new(
                    w_idx,
                    3,
                    mw(h_tw_entry.clone()),
                    mw(dh_tw_entry.clone()),
                ));
                pairs.push(CustomFamilyJointDesignPairContribution::new(
                    4,
                    w_idx,
                    mw(h_lw_exit.clone()),
                    mw(dh_lw_exit.clone()),
                ));
                pairs.push(CustomFamilyJointDesignPairContribution::new(
                    w_idx,
                    4,
                    mw(h_lw_exit.clone()),
                    mw(dh_lw_exit.clone()),
                ));
                pairs.push(CustomFamilyJointDesignPairContribution::new(
                    5,
                    w_idx,
                    mw(h_lw_entry.clone()),
                    mw(dh_lw_entry.clone()),
                ));
                pairs.push(CustomFamilyJointDesignPairContribution::new(
                    w_idx,
                    5,
                    mw(h_lw_entry.clone()),
                    mw(dh_lw_entry.clone()),
                ));
                pairs.push(CustomFamilyJointDesignPairContribution::new(
                    0,
                    w_idx,
                    mw(zero_w.clone()),
                    mw(&q.d3_q0 * &q0_psi),
                ));
                pairs.push(CustomFamilyJointDesignPairContribution::new(
                    w_idx,
                    0,
                    mw(zero_w.clone()),
                    mw(&q.d3_q0 * &q0_psi),
                ));
                pairs.push(CustomFamilyJointDesignPairContribution::new(
                    1,
                    w_idx,
                    mw(zero_w.clone()),
                    mw(&q.d3_q1 * &q1_psi),
                ));
                pairs.push(CustomFamilyJointDesignPairContribution::new(
                    w_idx,
                    1,
                    mw(zero_w),
                    mw(&q.d3_q1 * &q1_psi),
                ));
            }
            return Ok(Some(ExactNewtonJointPsiTerms {
                objective_psi,
                score_psi,
                hessian_psi: Array2::zeros((0, 0)),
                hessian_psi_operator: Some(std::sync::Arc::new(CustomFamilyJointPsiOperator::new(
                    p_total, channels, pairs,
                ))),
            }));
        }
        let mut hessian_psi = Array2::<f64>::zeros((p_total, p_total));
        assign_symmetric_block(&mut hessian_psi, offsets[0], offsets[0], &h_time_time);
        let h_threshold_threshold =
            mxtwx_psi(
                x_t_exit_map,
                h_tt_exit.view(),
                CustomFamilyPsiLinearMapRef::Dense(x_threshold_exit),
                row_mask,
            )? + mxtwx_psi(
                CustomFamilyPsiLinearMapRef::Dense(x_threshold_exit),
                h_tt_exit.view(),
                x_t_exit_map,
                row_mask,
            )? + mxtwx(x_threshold_exit, &dh_tt_exit, x_threshold_exit, row_mask)?
                + mxtwx_psi(
                    x_t_entry_map,
                    h_tt_entry.view(),
                    CustomFamilyPsiLinearMapRef::Dense(x_threshold_entry),
                    row_mask,
                )?
                + mxtwx_psi(
                    CustomFamilyPsiLinearMapRef::Dense(x_threshold_entry),
                    h_tt_entry.view(),
                    x_t_entry_map,
                    row_mask,
                )?
                + mxtwx(x_threshold_entry, &dh_tt_entry, x_threshold_entry, row_mask)?;
        assign_symmetric_block(
            &mut hessian_psi,
            offsets[1],
            offsets[1],
            &h_threshold_threshold,
        );
        let h_log_sigma_log_sigma =
            mxtwx_psi(
                x_ls_exit_map,
                h_ll_exit.view(),
                CustomFamilyPsiLinearMapRef::Dense(x_log_sigma_exit),
                row_mask,
            )? + mxtwx_psi(
                CustomFamilyPsiLinearMapRef::Dense(x_log_sigma_exit),
                h_ll_exit.view(),
                x_ls_exit_map,
                row_mask,
            )? + mxtwx(x_log_sigma_exit, &dh_ll_exit, x_log_sigma_exit, row_mask)?
                + mxtwx_psi(
                    x_ls_entry_map,
                    h_ll_entry.view(),
                    CustomFamilyPsiLinearMapRef::Dense(x_log_sigma_entry),
                    row_mask,
                )?
                + mxtwx_psi(
                    CustomFamilyPsiLinearMapRef::Dense(x_log_sigma_entry),
                    h_ll_entry.view(),
                    x_ls_entry_map,
                    row_mask,
                )?
                + mxtwx(x_log_sigma_entry, &dh_ll_entry, x_log_sigma_entry, row_mask)?;
        assign_symmetric_block(
            &mut hessian_psi,
            offsets[2],
            offsets[2],
            &h_log_sigma_log_sigma,
        );
        let h_threshold_log_sigma =
            mxtwx_psi(
                x_t_exit_map,
                h_tl_exit.view(),
                CustomFamilyPsiLinearMapRef::Dense(x_log_sigma_exit),
                row_mask,
            )? + mxtwx_psi(
                CustomFamilyPsiLinearMapRef::Dense(x_threshold_exit),
                h_tl_exit.view(),
                x_ls_exit_map,
                row_mask,
            )? + mxtwx(x_threshold_exit, &dh_tl_exit, x_log_sigma_exit, row_mask)?
                + mxtwx_psi(
                    x_t_entry_map,
                    h_tl_entry.view(),
                    CustomFamilyPsiLinearMapRef::Dense(x_log_sigma_entry),
                    row_mask,
                )?
                + mxtwx_psi(
                    CustomFamilyPsiLinearMapRef::Dense(x_threshold_entry),
                    h_tl_entry.view(),
                    x_ls_entry_map,
                    row_mask,
                )?
                + mxtwx(x_threshold_entry, &dh_tl_entry, x_log_sigma_entry, row_mask)?;
        assign_symmetric_block(
            &mut hessian_psi,
            offsets[1],
            offsets[2],
            &h_threshold_log_sigma,
        );
        let h_time_threshold = mxtwx(&self.x_time_entry, &dh_h0_t, x_threshold_entry, row_mask)?
            + mxtwx_psi(
                CustomFamilyPsiLinearMapRef::Dense(&self.x_time_entry),
                h_h0_t.view(),
                x_t_entry_map,
                row_mask,
            )?
            + mxtwx(&self.x_time_exit, &dh_h1_t, x_threshold_exit, row_mask)?
            + mxtwx_psi(
                CustomFamilyPsiLinearMapRef::Dense(&self.x_time_exit),
                h_h1_t.view(),
                x_t_exit_map,
                row_mask,
            )?;
        assign_symmetric_block(&mut hessian_psi, offsets[0], offsets[1], &h_time_threshold);
        let h_time_log_sigma = mxtwx(&self.x_time_entry, &dh_h0_ls, x_log_sigma_entry, row_mask)?
            + mxtwx_psi(
                CustomFamilyPsiLinearMapRef::Dense(&self.x_time_entry),
                h_h0_ls.view(),
                x_ls_entry_map,
                row_mask,
            )?
            + mxtwx(&self.x_time_exit, &dh_h1_ls, x_log_sigma_exit, row_mask)?
            + mxtwx_psi(
                CustomFamilyPsiLinearMapRef::Dense(&self.x_time_exit),
                h_h1_ls.view(),
                x_ls_exit_map,
                row_mask,
            )?;
        assign_symmetric_block(&mut hessian_psi, offsets[0], offsets[2], &h_time_log_sigma);

        if let (Some(xw_dense), Some(w_offset)) = (xw, offsets.get(3).copied()) {
            let h_ww = -(&q.d3_q0 * &q0_psi + &q.d3_q1 * &q1_psi);
            let h_wiggle_wiggle = mxtwx(xw_dense, &h_ww, xw_dense, row_mask)?;
            assign_symmetric_block(&mut hessian_psi, w_offset, w_offset, &h_wiggle_wiggle);
            let h_threshold_wiggle = mxtwx_psi(
                x_t_exit_map,
                h_tw_exit.view(),
                CustomFamilyPsiLinearMapRef::Dense(xw_dense),
                row_mask,
            )? + mxtwx(x_threshold_exit, &dh_tw_exit, xw_dense, row_mask)?
                + mxtwx_psi(
                    x_t_entry_map,
                    h_tw_entry.view(),
                    CustomFamilyPsiLinearMapRef::Dense(xw_dense),
                    row_mask,
                )?
                + mxtwx(x_threshold_entry, &dh_tw_entry, xw_dense, row_mask)?;
            assign_symmetric_block(&mut hessian_psi, offsets[1], w_offset, &h_threshold_wiggle);
            let h_log_sigma_wiggle = mxtwx_psi(
                x_ls_exit_map,
                h_lw_exit.view(),
                CustomFamilyPsiLinearMapRef::Dense(xw_dense),
                row_mask,
            )? + mxtwx(x_log_sigma_exit, &dh_lw_exit, xw_dense, row_mask)?
                + mxtwx_psi(
                    x_ls_entry_map,
                    h_lw_entry.view(),
                    CustomFamilyPsiLinearMapRef::Dense(xw_dense),
                    row_mask,
                )?
                + mxtwx(x_log_sigma_entry, &dh_lw_entry, xw_dense, row_mask)?;
            assign_symmetric_block(&mut hessian_psi, offsets[2], w_offset, &h_log_sigma_wiggle);
            let h_time_wiggle =
                mxtwx(
                    &self.x_time_entry,
                    &(&q.d3_q0 * &q0_psi),
                    xw_dense,
                    row_mask,
                )? + mxtwx(&self.x_time_exit, &(&q.d3_q1 * &q1_psi), xw_dense, row_mask)?;
            assign_symmetric_block(&mut hessian_psi, offsets[0], w_offset, &h_time_wiggle);
        }

        Ok(Some(ExactNewtonJointPsiTerms {
            objective_psi,
            score_psi,
            hessian_psi,
            hessian_psi_operator: None,
        }))
    }
}

pub(crate) struct SurvivalExactNewtonJointPsiWorkspace {
    pub(crate) family: SurvivalLocationScaleFamily,
    pub(crate) block_states: Vec<ParameterBlockState>,
    pub(crate) specs: Vec<ParameterBlockSpec>,
    pub(crate) derivative_blocks: Vec<Vec<CustomFamilyBlockPsiDerivative>>,
    pub(crate) row_mask: Option<Arc<Array1<f64>>>,
}

impl SurvivalExactNewtonJointPsiWorkspace {
    pub(crate) fn new(
        family: SurvivalLocationScaleFamily,
        block_states: Vec<ParameterBlockState>,
        specs: Vec<ParameterBlockSpec>,
        derivative_blocks: Vec<Vec<CustomFamilyBlockPsiDerivative>>,
    ) -> Result<Self, String> {
        Ok(Self {
            family,
            block_states,
            specs,
            derivative_blocks,
            row_mask: None,
        })
    }

    pub(crate) fn apply_outer_subsample(
        &mut self,
        rows: &[crate::outer_subsample::WeightedOuterRow],
    ) {
        let n = self.family.n;
        let mut mask = Array1::<f64>::zeros(n);
        for r in rows {
            if r.index < n {
                mask[r.index] = r.weight;
            }
        }
        self.row_mask = Some(Arc::new(mask));
    }
}

impl ExactNewtonJointPsiWorkspace for SurvivalExactNewtonJointPsiWorkspace {
    fn first_order_terms(
        &self,
        psi_index: usize,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        self.family.exact_newton_joint_psi_terms_masked(
            &self.block_states,
            &self.specs,
            &self.derivative_blocks,
            psi_index,
            self.row_mask.as_deref(),
        )
    }

    fn second_order_terms(
        &self,
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        let psi_dim = self.derivative_blocks.iter().map(Vec::len).sum::<usize>();
        if psi_i >= psi_dim || psi_j >= psi_dim {
            return Ok(None);
        }
        Ok(None)
    }

    fn hessian_directional_derivative(
        &self,
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<gam_problem::DriftDerivResult>, String> {
        let p_total = *self
            .family
            .joint_block_offsets()
            .last()
            .ok_or_else(|| "missing joint block offsets".to_string())?;
        if d_beta_flat.len() != p_total {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "joint psi workspace Hessian directional derivative d_beta length mismatch: got {}, expected {p_total}",
                    d_beta_flat.len()
                ),
            }
            .into());
        }
        let psi_dim = self.derivative_blocks.iter().map(Vec::len).sum::<usize>();
        if psi_index >= psi_dim {
            return Ok(None);
        }
        Ok(self
            .family
            .exact_newton_joint_psihessian_directional_derivative(
                &self.block_states,
                &self.specs,
                &self.derivative_blocks,
                psi_index,
                d_beta_flat,
            )?
            .map(gam_problem::DriftDerivResult::Dense))
    }
}

/// Workspace caching the direction-independent state used by the survival
/// location-scale joint-Hessian directional derivative operators.
pub(crate) struct SurvivalLocationScaleExactNewtonJointHessianWorkspace {
    pub(crate) family: SurvivalLocationScaleFamily,
    pub(crate) dynamic: SurvivalDynamicGeometry,
    pub(crate) deriv_log_scale: f64,
    pub(crate) row_mask: Option<Arc<Array1<f64>>>,
}

impl SurvivalLocationScaleExactNewtonJointHessianWorkspace {
    pub(crate) fn new(
        family: SurvivalLocationScaleFamily,
        block_states: Vec<ParameterBlockState>,
    ) -> Result<Self, String> {
        let log_rescale = family.hessian_deriv_log_rescale(&block_states);
        let dynamic = family.build_dynamic_geometry(&block_states)?;
        Ok(Self {
            family,
            dynamic,
            deriv_log_scale: log_rescale,
            row_mask: None,
        })
    }

    pub(crate) fn apply_outer_subsample(
        &mut self,
        rows: &[crate::outer_subsample::WeightedOuterRow],
    ) {
        let n = self.family.n;
        let mut mask = Array1::<f64>::zeros(n);
        for r in rows {
            if r.index < n {
                mask[r.index] = r.weight;
            }
        }
        self.row_mask = Some(Arc::new(mask));
    }

    pub(crate) fn clear_outer_subsample(&mut self) {
        self.row_mask = None;
    }
}

impl ExactNewtonJointHessianWorkspace for SurvivalLocationScaleExactNewtonJointHessianWorkspace {
    fn warm_up_outer_caches_for_mode(
        &self,
        eval_mode: gam_problem::EvalMode,
    ) -> Result<(), String> {
        match eval_mode {
            gam_problem::EvalMode::ValueOnly
            | gam_problem::EvalMode::ValueAndGradient
            | gam_problem::EvalMode::ValueGradientHessian => Ok(()),
        }
    }

    fn directional_derivative(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family
            .exact_newton_joint_hessian_directional_derivative_rescaled_from_parts_masked(
                d_beta_flat,
                &self.dynamic,
                self.deriv_log_scale,
                self.row_mask.as_deref(),
            )
    }

    fn directional_derivative_operator(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn HyperOperator>>, String> {
        Ok(self
            .family
            .exact_newton_joint_hessian_directional_derivative_rescaled_from_parts_masked(
                d_beta_flat,
                &self.dynamic,
                self.deriv_log_scale,
                self.row_mask.as_deref(),
            )?
            .map(|matrix| Arc::new(DenseMatrixHyperOperator { matrix }) as Arc<dyn HyperOperator>))
    }

    fn second_directional_derivative(
        &self,
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let p_total = *self
            .family
            .joint_block_offsets()
            .last()
            .ok_or_else(|| "missing joint block offsets".to_string())?;
        if d_beta_u_flat.len() != p_total || d_beta_v_flat.len() != p_total {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "joint Hessian workspace second directional derivative length mismatch: got {} / {}, expected {p_total}",
                    d_beta_u_flat.len(),
                    d_beta_v_flat.len()
                ),
            }
            .into());
        }
        let rows = row_set_from_survival_mask(self.row_mask.as_deref(), self.family.n);
        if self.family.x_link_wiggle.is_some() {
            // #932: single-source the wiggle workspace SECOND directional
            // derivative through the §13 warp kernel (`TwoSeed<KW>`) — the
            // cached dynamic geometry already carries the wiggle geometry + βw, so no
            // `block_states` re-thread is needed. Previously returned `None`.
            return Ok(Some(
                super::row_kernel::survival_ls_wiggle_second_directional_derivative_dense(
                    &self.family,
                    &self.dynamic,
                    self.deriv_log_scale,
                    &rows,
                    d_beta_u_flat.as_slice().ok_or_else(|| {
                        "joint Hessian workspace second directional u must be contiguous"
                            .to_string()
                    })?,
                    d_beta_v_flat.as_slice().ok_or_else(|| {
                        "joint Hessian workspace second directional v must be contiguous"
                            .to_string()
                    })?,
                )?,
            ));
        }
        let kernel = self
            .family
            .survival_ls_row_kernel_rescaled(&self.dynamic, self.deriv_log_scale);
        crate::row_kernel::row_kernel_second_directional_derivative(
            &kernel,
            &rows,
            d_beta_u_flat.as_slice().ok_or_else(|| {
                "joint Hessian workspace second directional u must be contiguous".to_string()
            })?,
            d_beta_v_flat.as_slice().ok_or_else(|| {
                "joint Hessian workspace second directional v must be contiguous".to_string()
            })?,
        )
        .map(Some)
    }

    fn second_directional_derivative_operator(
        &self,
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn HyperOperator>>, String> {
        Ok(self
            .second_directional_derivative(d_beta_u_flat, d_beta_v_flat)?
            .map(|matrix| Arc::new(DenseMatrixHyperOperator { matrix }) as Arc<dyn HyperOperator>))
    }
}
