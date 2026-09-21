use super::*;
use gam_problem::ConstraintSet;
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
    /// `obj_tol` is the caller's objective-suboptimality tolerance on `½λ²`
    /// described above, raised to the objective's own rounding band.
    pub(crate) fn fit_parametric_aft_direct_mle(
        &self,
        specs: &[ParameterBlockSpec],
        max_iter: usize,
        obj_tol: f64,
    ) -> Result<(Vec<ParameterBlockState>, f64, Array2<f64>), SurvivalLocationScaleError> {
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
            // is the packed 27-pair coefficient lowering; both are pinned to the
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
            // The log-likelihood accumulates `n` rows and the decrement `p²`
            // products; a predicted gain inside that accumulation's rounding band
            // `γ_{n+p²}·|ℓ|` cannot be told from zero, so the caller's tolerance is
            // raised to it.
            let n_rows = states.iter().map(|state| state.eta.len()).max().unwrap_or(0);
            let objective_band =
                gam_linalg::roundoff::accumulation_growth(n_rows + p_total * p_total) * ll.abs();
            if 0.5 * newton_decrement <= obj_tol.max(objective_band) {
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
            let u0 = dynamic.hs_entry[i] + dynamic.q_entry[i];
            let u1 = dynamic.hs_exit[i] + dynamic.q_exit[i];
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

    /// Second coefficient-directional derivative of the joint Hessian in an
    /// explicitly selected numerical scale.
    ///
    /// The outer Laplace factorization differentiates the rescaled curvature,
    /// while the Jeffreys term values the unscaled observed information. Both
    /// consumers must use the same row kernel but must not silently share the
    /// outer rescale: differentiating `exp(-L)H` as though it were `H` changes
    /// the Jeffreys prior whenever `L` moves with the coefficients.
    pub(crate) fn exact_newton_joint_hessian_second_directional_derivative_rescaled(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
        log_rescale: f64,
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
                    d_beta_v_flat.len(),
                ),
            }
            .into());
        }
        let dynamic = self.build_dynamic_geometry(block_states)?;
        if self.x_link_wiggle.is_some() {
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
                            let state = self.row_predictor_state_at(&dynamic, i);
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
                let state = self.row_predictor_state_at(&dynamic, i);
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

        // The row partials are taken in the index channels (u0, u1, g). The
        // scale divides the time transform (#2695), so the raw time channels
        // map through `∂u0/∂h0 = s0`, `∂u1/∂h1 = s1`, `∂g/∂h1 = −eta_ls'` and
        // `∂g/∂hdot = 1`, with `s = e^{−eta_ls}` and `du1/dt = s1·g`.
        let mut time_row_exit = Array1::<f64>::zeros(n);
        ndarray::Zip::from(&mut grad_time_eta_h0)
            .and(&dynamic.inv_sigma_entry)
            .for_each(|g, &s| *g *= s);
        ndarray::Zip::from(&mut time_row_exit)
            .and(&grad_time_eta_h1)
            .and(&grad_time_eta_d)
            .and(&dynamic.inv_sigma_exit)
            .and(&dynamic.eta_ls_deriv_exit)
            .for_each(|out, &gu, &gg, &s, &lsd| *out = s * gu - lsd * gg);
        let grad_time = dynamic.time_jac_entry.t().dot(&grad_time_eta_h0)
            + dynamic.time_jac_exit.t().dot(&time_row_exit)
            + dynamic.time_jac_deriv.t().dot(&grad_time_eta_d);

        // #2342: stable combined index-derivative sum `S1 = d1_q0 + d1_q1` for
        // the far-tail rows whose entry (`d1_q0 = w·A′(u0)`) and exit
        // (`d1_q1`) hazard channels are each ~1e300 and (near-)opposite. The
        // naive coefficient contraction `d1_q1·dq_exit + d1_q0·dq_entry` sums
        // them at a SHARED coefficient and cancels catastrophically; the
        // regroup `S1·dq_exit + d1_q0·(dq_entry − dq_exit)` keeps every retained
        // quantity moderate (or honestly-huge times an exact/Sterbenz zero). We
        // compute `S1` cancellation-free (paired stacks) only where the gate
        // fires; every other row keeps today's op order bitwise. `δu` is the
        // Sterbenz-safe channel gap `(h1−h0)+(q1−q0)` — never `u1−u0`, which
        // rounds to 0 in the far tail.
        let mut paired_s1 = Array1::<f64>::zeros(n);
        let mut use_paired = vec![false; n];
        for i in 0..n {
            let u0 = dynamic.hs_entry[i] + dynamic.q_entry[i];
            if self.w[i] > 0.0
                && self.entry_active[i]
                && paired_stacks::paired_contraction_needs_regroup(&self.inverse_link, u0)
            {
                let u1 = dynamic.hs_exit[i] + dynamic.q_exit[i];
                let delta_u = (dynamic.hs_exit[i] - dynamic.hs_entry[i])
                    + (dynamic.q_exit[i] - dynamic.q_entry[i]);
                let w_eff = self.w[i] * mask_at(i);
                if let Some(sums) = paired_stacks::weighted_paired_index_sums(
                    &self.inverse_link,
                    u0,
                    u1,
                    delta_u,
                    self.y[i],
                    w_eff,
                ) {
                    paired_s1[i] = sums[0];
                    use_paired[i] = true;
                }
            }
        }

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
            // regrouped to S1·dq_exit + d1_q0·(dq_entry − dq_exit) on the far-tail
            // rows (#2342); byte-identical to the pre-#2342 Zip on every other row.
            for i in 0..n {
                let exit = dynamic.dq_t_exit[i];
                let entry = dynamic.dq_t_entry[i];
                scratch[i] = if use_paired[i] {
                    paired_s1[i] * exit + d1_q0[i] * (entry - exit)
                } else {
                    d1_q1[i] * exit + d1_q0[i] * entry
                };
            }
            ndarray::Zip::from(&mut scratch)
                .and(&d1_qdot)
                .and(&dynamic.dqdot_t)
                .for_each(|s, &a, &b| *s += a * b);
            self.x_threshold.transpose_vector_multiply(&scratch)
        };

        // The log-σ derivatives of the indices are the location channel's plus
        // the scaled time transform's (#2695): `∂u/∂eta_ls = ∂q/∂eta_ls − hs` and
        // `∂g/∂eta_ls' = ∂qdot/∂eta_ls' − h1`, while `∂g/∂eta_ls = ∂qdot/∂eta_ls`
        // (the warp slope's alone). The event log-density's linear `−eta_ls`
        // adds `−w·d` at exit.
        let ls_exit = &dynamic.dq_ls_exit - &dynamic.hs_exit;
        let ls_entry = &dynamic.dq_ls_entry - &dynamic.hs_entry;
        let log_scale_score = Array1::from_shape_fn(n, |i| {
            if self.w[i] > 0.0 {
                -self.w[i] * self.y[i] * mask_at(i)
            } else {
                0.0
            }
        });
        let grad_ls = if let (Some(x_ls_entry), Some(x_ls_deriv)) = (
            self.x_log_sigma_entry.as_ref(),
            self.x_log_sigma_deriv.as_ref(),
        ) {
            ndarray::Zip::from(&mut scratch)
                .and(&d1_q1)
                .and(&ls_exit)
                .and(&d1_qdot)
                .and(&dynamic.dqdot_ls)
                .and(&log_scale_score)
                .for_each(|s, &a, &b, &c, &d, &e| *s = a * b + c * d + e);
            let mut out = self.x_log_sigma.transpose_vector_multiply(&scratch);
            ndarray::Zip::from(&mut scratch)
                .and(&d1_q0)
                .and(&ls_entry)
                .for_each(|s, &a, &b| *s = a * b);
            out = out + x_ls_entry.transpose_vector_multiply(&scratch);
            ndarray::Zip::from(&mut scratch)
                .and(&d1_qdot)
                .and(&dynamic.dqdot_lsd)
                .and(&dynamic.h_exit)
                .for_each(|s, &a, &b, &h| *s = a * (b - h));
            out + x_ls_deriv.transpose_vector_multiply(&scratch)
        } else {
            // combined[i] = d1_q1[i]*ls_exit[i] + d1_q0[i]*ls_entry[i],
            // regrouped to S1·ls_exit + d1_q0·(ls_entry − ls_exit) on the
            // far-tail rows (#2342). For a shared (time-invariant) log-sigma
            // channel the location parts agree exactly, so the huge d1_q0
            // multiplies only the scaled time gap `hs_exit − hs_entry`: an
            // honestly-huge term of the gradient, never a cancelling pair. The
            // gap is taken channel by channel, since `hs` rounds away inside
            // `ls_entry` and `ls_exit` once `q ~ 1e150`.
            for i in 0..n {
                let exit = ls_exit[i];
                let entry = ls_entry[i];
                scratch[i] = if use_paired[i] {
                    let gap = (dynamic.dq_ls_entry[i] - dynamic.dq_ls_exit[i])
                        - (dynamic.hs_entry[i] - dynamic.hs_exit[i]);
                    paired_s1[i] * exit + d1_q0[i] * gap
                } else {
                    d1_q1[i] * exit + d1_q0[i] * entry
                };
            }
            ndarray::Zip::from(&mut scratch)
                .and(&d1_qdot)
                .and(&dynamic.dqdot_ls)
                .and(&log_scale_score)
                .for_each(|s, &a, &b, &e| *s += a * b + e);
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
impl crate::custom_family::JeffreysArming for SurvivalLocationScaleFamily {
    fn with_jeffreys_armed(
        &self,
        evidence: Option<&gam_problem::jeffreys_arming::JeffreysArmingEvidence>,
    ) -> Self {
        Self {
            jeffreys_armed: evidence.is_some(),
            ..self.clone()
        }
    }
}

impl CustomFamily for SurvivalLocationScaleFamily {
    // The self-limiting Jeffreys/Firth curvature bounds a direction the data do
    // not, but it is armed only when the unarmed fit proves it is needed (#979).
    fn joint_jeffreys_term_required(&self) -> bool {
        self.jeffreys_armed
    }

    /// Differentiate the same unscaled observed information returned by
    /// `joint_jeffreys_information_with_specs`.
    ///
    /// The outer Laplace Hessian is rescaled to protect its factorization, but
    /// that numerical representation is not the Jeffreys information itself.
    /// Inheriting the generic hook through
    /// `exact_newton_joint_hessian_directional_derivative` differentiated the
    /// rescaled matrix instead, desynchronizing Φ/H_Φ from their derivatives.
    fn joint_jeffreys_information_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.validate_joint_specs(
            specs,
            "SurvivalLocationScaleFamily joint Jeffreys directional derivative",
        )?;
        self.exact_newton_joint_hessian_directional_derivative_rescaled(
            block_states,
            d_beta_flat,
            0.0,
        )
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
    /// with `log_rescale = 0`: the Jeffreys value is the unscaled observed
    /// information, so its derivative batch must be unscaled too. Reusing a
    /// single dynamic geometry across the axis sweep remains bit-identical to
    /// the corresponding singular hook while paying the quantity build once.
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
        let log_rescale = 0.0;
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

    fn joint_jeffreys_information_second_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.validate_joint_specs(
            specs,
            "SurvivalLocationScaleFamily joint Jeffreys second directional derivative",
        )?;
        self.exact_newton_joint_hessian_second_directional_derivative_rescaled(
            block_states,
            d_beta_u_flat,
            d_beta_v_flat,
            0.0,
        )
    }

    /// All-axes `I''[u, e_a]` along one mode response `u`. The Jeffreys `H_Φ`
    /// drift asks for this object once per outer mode-response direction. The
    /// trait default calls the per-axis hook `p` times, and every call rebuilds
    /// the dynamic geometry and folds all rows serially. On the non-wiggle row
    /// kernel the geometry and kernel are built once and the `p` axes go through
    /// `row_kernel_second_directional_derivative_all_axes`: each axis is the same
    /// `row_kernel_second_directional_derivative` fold the per-axis hook runs, so
    /// the matrices are bit-identical (#2668, #2106). The link-wiggle lowering has
    /// no fixed-width row kernel and keeps the per-axis loop.
    fn joint_jeffreys_information_second_directional_all_axes_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_u_flat: &Array1<f64>,
    ) -> Result<Option<Vec<Array2<f64>>>, String> {
        self.validate_joint_specs(
            specs,
            "SurvivalLocationScaleFamily joint Jeffreys all-axes second directional derivative",
        )?;
        let p_total = specs.iter().map(|spec| spec.design.ncols()).sum::<usize>();
        if d_beta_u_flat.len() != p_total {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "joint Jeffreys all-axes second directional derivative: direction has {} entries, expected {p_total}",
                    d_beta_u_flat.len(),
                ),
            }
            .into());
        }
        if !self.row_kernel_directional_supported() {
            let mut axes = Vec::with_capacity(p_total);
            for a in 0..p_total {
                let mut axis = Array1::<f64>::zeros(p_total);
                axis[a] = 1.0;
                match self.exact_newton_joint_hessian_second_directional_derivative_rescaled(
                    block_states,
                    d_beta_u_flat,
                    &axis,
                    0.0,
                )? {
                    Some(matrix) => axes.push(matrix),
                    None => return Ok(None),
                }
            }
            return Ok(Some(axes));
        }
        crate::block_layout::block_count::validate_block_count::<SurvivalLocationScaleError>(
            "SurvivalLocationScaleFamily joint Jeffreys all-axes second directional derivative",
            self.expected_blocks(),
            block_states.len(),
        )?;
        let dynamic = self.build_dynamic_geometry(block_states)?;
        let kernel = self.survival_ls_row_kernel_rescaled(&dynamic, 0.0);
        crate::row_kernel::row_kernel_second_directional_derivative_all_axes(
            &kernel,
            &crate::row_kernel::RowSet::All,
            d_beta_u_flat.as_slice().ok_or_else(|| {
                "joint Jeffreys all-axes second directional u must be contiguous".to_string()
            })?,
        )
        .map(Some)
    }

    /// On the non-wiggle row kernel the family contracts `I''[δ, e_a]` against the Jeffreys
    /// drift's kernels in one pass (#2668). The link-wiggle lowering has no fixed-width row
    /// kernel and provides none.
    fn jeffreys_axis_contractions(
        &self,
    ) -> Option<&dyn crate::custom_family::JeffreysAxisContractions> {
        if self.row_kernel_directional_supported() {
            Some(self)
        } else {
            None
        }
    }

    /// On the non-wiggle row kernel the family forms the rotated first information rows from
    /// its projected channel rows (#2668). The link-wiggle lowering has no fixed-width row
    /// kernel and provides none.
    fn jeffreys_rotated_first_derivative(
        &self,
    ) -> Option<&dyn crate::custom_family::JeffreysRotatedFirstDerivative> {
        if self.row_kernel_directional_supported() {
            Some(self)
        } else {
            None
        }
    }

    /// `∇²_β tr(W · I(β))` for the unscaled observed information: the same
    /// fourth-order row contraction that
    /// [`Self::joint_jeffreys_information_second_directional_derivative_with_specs`]
    /// pulls back one direction pair at a time, contracted with `W` over every
    /// coefficient pair in a single row fold. Without it the exact Jeffreys
    /// completion assembles `p(p+1)/2` full-data `I''[e_a, e_b]` passes on every
    /// armed inner-Newton endgame cycle, and the outer mode response never receives
    /// the completion at all (#2668, #2106). The link-wiggle runtime lowering has no
    /// fixed-width row kernel and keeps the pairwise route.
    fn joint_jeffreys_information_contracted_trace_hessian_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        weight: &Array2<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.validate_joint_specs(
            specs,
            "SurvivalLocationScaleFamily joint Jeffreys contracted trace Hessian",
        )?;
        crate::block_layout::block_count::validate_block_count::<SurvivalLocationScaleError>(
            "SurvivalLocationScaleFamily joint Jeffreys contracted trace Hessian",
            self.expected_blocks(),
            block_states.len(),
        )?;
        if !self.row_kernel_directional_supported() {
            return Ok(None);
        }
        let dynamic = self.build_dynamic_geometry(block_states)?;
        let kernel = self.survival_ls_row_kernel_rescaled(&dynamic, 0.0);
        crate::row_kernel::row_kernel_contracted_trace_hessian(
            &kernel,
            &crate::row_kernel::RowSet::All,
            weight,
        )
        .map(Some)
    }

    /// The fused contraction exists exactly where the fixed-width row kernel
    /// supplies the second directional derivative it contracts.
    fn joint_jeffreys_information_contracted_trace_hessian_available(&self) -> bool {
        self.row_kernel_directional_supported()
    }

    /// The fifth likelihood derivative lowers through `sls_row_program`'s
    /// three-seed directional surface wherever every residual-distribution
    /// stack has a closed-form fifth derivative. The link-wiggle runtime
    /// lowering carries no fifth order, and neither do the parameterized links
    /// served by the generic pdf-jet dispatch.
    fn jeffreys_third_information_derivative(
        &self,
    ) -> Option<&dyn crate::custom_family::JeffreysThirdInformationDerivative> {
        if self.row_kernel_directional_supported()
            && Self::inverse_link_has_fifth_derivative_stacks(&self.inverse_link)
        {
            Some(self)
        } else {
            None
        }
    }

    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    /// Declare the per-block output channel so the pre-fit identifiability
    /// audit routes **channel-aware** (`audit_identifiability_channel_aware`)
    /// instead of the flat n-row Euclidean stack.
    ///
    /// The survival location-scale row NLL `ρ_i(η_time, η_thr, η_ls)` has THREE
    /// output channels:
    ///   - channel 0 — `η_time` (time-transform predictor shift), and the
    ///     link-wiggle correction anchors here (it perturbs the inverse link
    ///     applied on the time/location side),
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
                    // channel.
                    _ => 0,
                })
                .collect(),
        )
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

    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let (ll, block_gradients) =
            self.evaluate_log_likelihood_and_block_gradients(block_states)?;

        // Every non-wiggle block is a view of the same packed 27-pair row
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
            let state = self.row_predictor_state_at(&dynamic, i);
            Ok(self
                .exact_row_kernel(i, state)?
                .map_or(0.0, |kernel| kernel.log_likelihood_at(&state)))
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
            let state = self.row_predictor_state_at(&dynamic, i);
            ll += row.weight
                * self
                    .exact_row_kernel(i, state)?
                    .map_or(0.0, |kernel| kernel.log_likelihood_at(&state));
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
        let mut d_beta_flat = Array1::<f64>::zeros(
            *offsets
                .last()
                .expect("joint block offsets always end with the total joint width"),
        );
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
        if let Some(axis) = hyper_layout.family_axis(psi_index) {
            // The family-owned axes are the inverse-link shape parameters, in
            // the link's own parameter order (#2904).
            return self.link_param_joint_psi_terms(block_states, axis);
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
            return Err(
                "SurvivalLocationScaleFamily serves no second-order hyper terms while its \
                 inverse-link shape axes are present"
                    .to_string(),
            );
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
            return Err(
                "SurvivalLocationScaleFamily has no exact-psi workspace over its inverse-link \
                 shape axes"
                    .to_string(),
            );
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
            hyper_layout.clone(),
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
            return Err(
                "SurvivalLocationScaleFamily has no exact-psi workspace over its inverse-link \
                 shape axes"
                    .to_string(),
            );
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
            hyper_layout.clone(),
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
        self.exact_newton_joint_psihessian_directional_derivative_masked(
            block_states,
            specs,
            hyper_layout,
            psi_index,
            d_beta_flat,
            None,
        )
    }

    fn exact_newton_joint_hessiansecond_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let log_rescale = self.hessian_deriv_log_rescale(block_states);
        self.exact_newton_joint_hessian_second_directional_derivative_rescaled(
            block_states,
            d_beta_u_flat,
            d_beta_v_flat,
            log_rescale,
        )
    }

    fn block_linear_constraints(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        spec: &ParameterBlockSpec,
    ) -> Result<Option<ConstraintSet>, String> {
        // Every constraint below is sized from `spec`, so it only constrains
        // anything if `spec` really describes the block the solver is carrying
        // at `block_idx`.
        let Some(state) = block_states.get(block_idx) else {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "survival location-scale linear constraints requested for block {block_idx}, but only {} block states were supplied",
                    block_states.len()
                ),
            }
            .into());
        };
        if state.beta.len() != spec.design.ncols() {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "survival location-scale linear constraints for block {block_idx}: spec width {} does not match the carried coefficient width {}",
                    spec.design.ncols(),
                    state.beta.len()
                ),
            }
            .into());
        }
        if block_idx == Self::BLOCK_LINK_WIGGLE {
            return Ok(monotone_wiggle_nonnegative_constraints(spec.design.ncols()));
        }
        if block_idx != Self::BLOCK_TIME {
            return Ok(None);
        }
        Ok(self
            .time_linear_constraints
            .clone()
            .map(ConstraintSet::Dense))
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
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Array1<f64>>, String> {
        // The floor is returned in the packed joint coefficient space this
        // family lays out, so the caller's blocks must be the blocks that
        // layout describes — otherwise the returned vector would be applied to
        // a metric it does not index.
        self.validate_joint_specs(
            specs,
            "SurvivalLocationScaleFamily joint trust metric floor",
        )?;
        // Scale-aware trust-metric floor for the coupled smooth-scale fit
        // (issue #1569). The free scale predictor `η_σ` enters the likelihood
        // through the standardized index `u = inv_sigma·(h − η_t)` with
        // `inv_sigma = exp(−η_σ)`, so `∂u/∂h = inv_sigma` and
        // `∂u/∂η_t = −inv_sigma`: the TIME, LOCATION (threshold) and LOG-σ
        // channels all carry an `exp(−η_σ)` factor in their gradient and an
        // `exp(−2 η_σ)` factor in their likelihood-Hessian diagonal, since the
        // scale divides the whole residual (#2695). When the scale predictor
        // drives some rows to small σ (large `exp(−η_σ)`), a coefficient loading
        // mostly on the large-σ rows is METRIC-STARVED relative to one loading on the small-σ
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
        // Floor the LOCATION (threshold) and LOG-σ blocks. The time block carries
        // the same factor but is left unfloored: its metric range comes as much
        // from the warp basis as from σ, and flooring it would move the metric of
        // every constant-scale fit whose time block spans six decades.
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
        block_states: &[ParameterBlockState],
        block_idx: usize,
        block_spec: &ParameterBlockSpec,
        beta: Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        assert!(!block_spec.name.is_empty());
        // A post-update hook may change the coefficients' VALUES but never
        // their layout: the vector returned here replaces the block the solver
        // is carrying, so it must have that block's width.
        let Some(state) = block_states.get(block_idx) else {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "survival location-scale post-update for block {block_idx}, but only {} block states were supplied",
                    block_states.len()
                ),
            }
            .into());
        };
        if beta.len() != state.beta.len() {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "survival location-scale post-update for block {block_idx} received {} coefficients but the block carries {}",
                    beta.len(),
                    state.beta.len()
                ),
            }
            .into());
        }
        if block_idx == Self::BLOCK_TIME
            && let Some(constraints) = self.time_linear_constraints.as_ref()
        {
            validate_linear_constraints("time post-update", &beta, constraints)?;
        } else if block_idx == Self::BLOCK_LINK_WIGGLE && self.x_link_wiggle.is_some() {
            for j in 0..beta.len() {
                // The SAME derivation the time arm above uses, called rather
                // than copied (#2722): #1569's floor was landed on
                // `validate_linear_constraints` alone and this arm kept
                // rejecting at the unfloored relative tolerance, two orders
                // tighter than the gate its own consumers certify to. The
                // link-wiggle cone's rows are the unit identity, so its scaled
                // and raw slacks coincide and the contract band is the absolute
                // gate outright — a coefficient the QP left at `-6.6e-9` is
                // feasible to every downstream consumer and was a hard error
                // here.
                let tol = roundoff_feasible_slack_tol(beta[j].abs());
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

    // Inherent `exact_newton_joint_psi_terms_masked` is defined in the
    // `impl SurvivalLocationScaleFamily` block below. It is invoked directly
    // by both this trait method and the ψ workspace's `first_order_terms`
    // override to thread the Horvitz-Thompson row mask through the staged
    // outer-score subsample.
}

impl crate::custom_family::JeffreysThirdInformationDerivative for SurvivalLocationScaleFamily {
    /// Third beta-directional derivative of the same unscaled observed
    /// information returned by `joint_jeffreys_information_with_specs`, along
    /// every canonical axis: `{I'''[u, v, e_a]}`. Each row's fifth-order
    /// contraction with `(u, v)` is built once and pulled back per axis (#2677).
    fn third_directional_all_axes(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Vec<Array2<f64>>>, String> {
        self.validate_joint_specs(
            specs,
            "SurvivalLocationScaleFamily joint Jeffreys third directional derivative",
        )?;
        let dynamic = self.build_dynamic_geometry(block_states)?;
        let kernel = self.survival_ls_row_kernel_rescaled(&dynamic, 0.0);
        crate::row_kernel::row_kernel_third_directional_derivative_all_axes(
            &kernel,
            &crate::row_kernel::RowSet::All,
            d_beta_u_flat.as_slice().ok_or_else(|| {
                "joint Jeffreys third directional u must be contiguous".to_string()
            })?,
            d_beta_v_flat.as_slice().ok_or_else(|| {
                "joint Jeffreys third directional v must be contiguous".to_string()
            })?,
        )
        .map(Some)
    }
}

impl crate::custom_family::JeffreysRotatedFirstDerivative for SurvivalLocationScaleFamily {
    /// The rows `vec(sym(Uᵀ I'[e_a] U))` from each row's nine third contractions and its channel
    /// rows projected onto `U`, so the `p` dense axis matrices are never formed (#2668).
    fn first_directional_rotated_all_axes(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        basis: ndarray::ArrayView2<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        self.validate_joint_specs(
            specs,
            "SurvivalLocationScaleFamily joint Jeffreys rotated first directional derivative",
        )?;
        crate::block_layout::block_count::validate_block_count::<SurvivalLocationScaleError>(
            "SurvivalLocationScaleFamily joint Jeffreys rotated first directional derivative",
            self.expected_blocks(),
            block_states.len(),
        )?;
        let dynamic = self.build_dynamic_geometry(block_states)?;
        let kernel = self.survival_ls_row_kernel_rescaled(&dynamic, 0.0);
        kernel.directional_derivative_rotated_all_axes(basis)
    }
}

impl crate::custom_family::JeffreysAxisContractions for SurvivalLocationScaleFamily {
    /// `⟨I''[δ, e_a], K_b⟩` for every drift direction in one pass on the row kernel. The kernel
    /// products are formed once per row for the batch, and each direction contracts nine
    /// fourth-order rows, so no `p × p` axis matrix is materialized (#2668).
    fn second_directional_axis_contractions_each(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        directions: &[Array1<f64>],
        kernels: &dyn Fn() -> Vec<Array2<f64>>,
        consume: &mut dyn FnMut(usize, Array2<f64>) -> Result<(), String>,
    ) -> Result<(), String> {
        self.validate_joint_specs(
            specs,
            "SurvivalLocationScaleFamily joint Jeffreys all-axes second directional contractions",
        )?;
        crate::block_layout::block_count::validate_block_count::<SurvivalLocationScaleError>(
            "SurvivalLocationScaleFamily joint Jeffreys all-axes second directional contractions",
            self.expected_blocks(),
            block_states.len(),
        )?;
        let dynamic = self.build_dynamic_geometry(block_states)?;
        let kernel = self.survival_ls_row_kernel_rescaled(&dynamic, 0.0);
        kernel.second_directional_axis_contractions_each(directions, &kernels(), consume)
    }
}

impl SurvivalLocationScaleFamily {
    pub(crate) fn exact_newton_joint_psihessian_directional_derivative_masked(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        hyper_layout: &CustomFamilyHyperLayout,
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
        row_mask: Option<&Array1<f64>>,
    ) -> Result<Option<Array2<f64>>, String> {
        if let Some(axis) = hyper_layout.family_axis(psi_index) {
            // The family-owned axes are the inverse-link shape parameters, in the
            // link's own parameter order (#2904).
            return self.link_param_joint_psihessian_directional_derivative(
                block_states,
                axis,
                d_beta_flat
                    .as_slice()
                    .ok_or_else(|| "joint psi Hessian direction must be contiguous".to_string())?,
                &row_set_from_survival_mask(row_mask, self.n),
            );
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
        let Some(direction) =
            self.exact_newton_joint_psi_direction(block_states, derivative_blocks, psi_index)?
        else {
            return Ok(None);
        };
        let dynamic = self.build_dynamic_geometry(block_states)?;
        super::row_kernel::survival_ls_joint_psi_hessian_directional_derivative_dense(
            self,
            &dynamic,
            &direction,
            d_beta_flat.as_slice().ok_or_else(|| {
                "joint psi mixed Hessian direction must be contiguous".to_string()
            })?,
            row_mask,
        )
        .map(Some)
    }

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
        // Every explicit ψ term comes from the same row program as the value,
        // gradient and Hessian (the #736/#932 single-source contract), so the
        // scaled time transform (#2695), the event-rate channel and the NLL sign
        // are shared by construction. The non-wiggle design-action path keeps
        // `H_ψ` as a streamed operator; every other path assembles it dense.
        let dynamic = self.build_dynamic_geometry(block_states)?;
        let has_design_actions = dir.x_t_exit_action.is_some()
            || dir.x_t_entry_action.is_some()
            || dir.x_t_deriv_action.is_some()
            || dir.x_ls_exit_action.is_some()
            || dir.x_ls_entry_action.is_some()
            || dir.x_ls_deriv_action.is_some();
        let streamed_hessian = has_design_actions && self.x_link_wiggle.is_none();
        let (objective_psi, score_psi, dense_hessian) =
            super::row_kernel::survival_ls_joint_psi_first_order_terms(
                self,
                &dynamic,
                &dir,
                row_mask,
                !streamed_hessian,
            )?;
        if streamed_hessian {
            return Ok(Some(ExactNewtonJointPsiTerms {
                objective_psi,
                score_psi,
                hessian_psi: Array2::zeros((0, 0)),
                hessian_psi_operator: Some(super::row_kernel::survival_ls_joint_psi_hessian_operator(
                    self, &dynamic, &dir, row_mask,
                )?),
            }));
        }
        let hessian_psi = dense_hessian.ok_or_else(|| {
            String::from(SurvivalLocationScaleError::InternalInvariant {
                reason: "survival location-scale dense ψ Hessian was requested but not assembled"
                    .to_string(),
            })
        })?;
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
    pub(crate) hyper_layout: CustomFamilyHyperLayout,
    pub(crate) row_mask: Option<Arc<Array1<f64>>>,
}

impl SurvivalExactNewtonJointPsiWorkspace {
    pub(crate) fn new(
        family: SurvivalLocationScaleFamily,
        block_states: Vec<ParameterBlockState>,
        specs: Vec<ParameterBlockSpec>,
        hyper_layout: CustomFamilyHyperLayout,
    ) -> Result<Self, String> {
        Ok(Self {
            family,
            block_states,
            specs,
            hyper_layout,
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
            self.hyper_layout.design_derivative_blocks(),
            psi_index,
            self.row_mask.as_deref(),
        )
    }

    fn second_order_terms(
        &self,
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        let psi_dim = self.hyper_layout.design_axis_count();
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
        let psi_dim = self.hyper_layout.design_axis_count();
        if psi_index >= psi_dim {
            return Ok(None);
        }
        Ok(self
            .family
            .exact_newton_joint_psihessian_directional_derivative_masked(
                &self.block_states,
                &self.specs,
                &self.hyper_layout,
                psi_index,
                d_beta_flat,
                self.row_mask.as_deref(),
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

/// #2722: the two arms of [`SurvivalLocationScaleFamily::post_update_block_beta`]
/// must accept exactly the same round-off-feasible step.
///
/// #1569 floored the time arm's relative tolerance at the absolute gate the
/// downstream consumers (`check_linear_feasibility` /
/// `project_onto_linear_constraints`) certify to, and landed a regression test
/// for that arm alone. The link-wiggle arm — the `else if` of the very same
/// conditional — kept rejecting at the unfloored relative tolerance, two orders
/// TIGHTER than those consumers, so a cone-projected coefficient at the exact
/// slack #1569 measured for its sibling (`-6.6e-9`) was feasible to every
/// consumer in the pipeline and a hard error here.
///
/// Both arms now derive their tolerance from the single
/// [`roundoff_feasible_slack_tol`], and this test pins the symmetry: the same
/// step value is fed to both arms and both must return the same verdict, for a
/// round-off-feasible value AND for a genuine violation past the gate.
#[cfg(test)]
mod post_update_roundoff_floor_symmetry_2722_tests {
    use super::*;
    use ndarray::array;

    /// Minimal family carrying BOTH a time nonnegativity constraint system and a
    /// link-wiggle block, so a single instance can be asked for both arms'
    /// verdicts on the same value. Each block is one coefficient wide, and the
    /// time constraint system is the per-coordinate lower bound `β ≥ 0` — the
    /// identity row, i.e. structurally the SAME cone the link-wiggle arm checks
    /// coordinate-wise. Anything the two arms disagree about is therefore a
    /// difference in the tolerance rule, not in the geometry.
    fn family_with_both_arms_armed() -> SurvivalLocationScaleFamily {
        let dense = |values: Array2<f64>| {
            DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(values))
        };
        SurvivalLocationScaleFamily {
            n: 3,
            y: array![1.0, 0.0, 1.0],
            w: array![1.0, 0.8, 1.2],
            inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
            derivative_guard: 1e-8,
            x_time_entry: Arc::new(array![[1.0], [1.0], [1.0]]),
            x_time_exit: Arc::new(array![[1.2], [0.9], [1.4]]),
            x_time_deriv: Arc::new(array![[1.0], [1.0], [1.0]]),
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
            time_linear_constraints: lower_bound_constraints(&array![0.0]),
            x_threshold: dense(array![[1.0], [0.4], [-0.6]]),
            x_threshold_entry: None,
            x_threshold_deriv: None,
            x_log_sigma: dense(array![[1.0], [-0.3], [0.5]]),
            x_log_sigma_entry: None,
            x_log_sigma_deriv: None,
            x_link_wiggle: Some(dense(array![[0.5], [0.25], [0.75]])),
            wiggle_knots: None,
            wiggle_degree: None,
            location_log_time: None,
            entry_active: Arc::from(vec![true; 3]),
            policy: gam_runtime::resource::ResourcePolicy::default_library(),
            jeffreys_armed: true,
        }
    }

    fn one_wide_spec(name: &str) -> ParameterBlockSpec {
        ParameterBlockSpec {
            name: name.to_string(),
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                Array2::<f64>::zeros((3, 1)),
            )),
            offset: Array1::zeros(3),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: None,
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        }
    }

    /// The four one-coefficient block states the family expects (time,
    /// threshold, log-sigma, link-wiggle). Only the block widths matter to the
    /// post-update hook, which validates the proposal it is handed.
    fn one_wide_states() -> Vec<ParameterBlockState> {
        (0..4)
            .map(|_| ParameterBlockState {
                beta: array![0.0],
                eta: Array1::<f64>::zeros(9),
            })
            .collect()
    }

    /// Both arms' verdicts on the same proposed coefficient value.
    fn verdicts(value: f64) -> (Result<Array1<f64>, String>, Result<Array1<f64>, String>) {
        let family = family_with_both_arms_armed();
        let states = one_wide_states();
        let time = family.post_update_block_beta(
            &states,
            SurvivalLocationScaleFamily::BLOCK_TIME,
            &one_wide_spec("time"),
            array![value],
        );
        let link_wiggle = family.post_update_block_beta(
            &states,
            SurvivalLocationScaleFamily::BLOCK_LINK_WIGGLE,
            &one_wide_spec("link_wiggle"),
            array![value],
        );
        (time, link_wiggle)
    }

    #[test]
    fn both_arms_accept_the_same_roundoff_feasible_step_2722() {
        // The #1569 witness magnitude: inside the absolute downstream gate, so
        // every consumer in the pipeline calls it feasible.
        let (time, link_wiggle) = verdicts(-6.6e-9);
        assert!(
            time.is_ok(),
            "time arm rejected a round-off-feasible step the downstream gate accepts: {:?}",
            time.unwrap_err()
        );
        assert!(
            link_wiggle.is_ok(),
            "link-wiggle arm rejected the SAME step the time arm accepts (#2722): {:?}",
            link_wiggle.unwrap_err()
        );

        // Strictly interior: trivially accepted by both.
        let (time, link_wiggle) = verdicts(0.5);
        assert!(time.is_ok() && link_wiggle.is_ok());
    }

    #[test]
    fn both_arms_reject_the_same_genuine_violation_2722() {
        // An order of magnitude PAST the gate: the shared floor relaxes
        // round-off, not real violations, and it must do so symmetrically.
        let (time, link_wiggle) = verdicts(-1e-7);
        assert!(
            time.is_err(),
            "time arm accepted a genuine violation 10x past the downstream gate"
        );
        assert!(
            link_wiggle.is_err(),
            "link-wiggle arm accepted a genuine violation 10x past the downstream gate"
        );
    }

    /// The floor itself: at every scale the shared derivation returns a
    /// tolerance at least the absolute gate the consumers certify to, and grows
    /// with the problem scale beyond it. No arm can be tighter than the gate.
    #[test]
    fn the_shared_floor_is_never_tighter_than_the_consumer_gate_2722() {
        for scale in [0.0, 1e-12, 1.0, 1e3, 1e6] {
            let tol = roundoff_feasible_slack_tol(scale);
            assert!(
                tol >= MONOTONE_CONE_FEASIBILITY_GATE_TOL,
                "tolerance {tol:.3e} at scale {scale:.3e} is tighter than the downstream gate"
            );
        }
        assert!(
            roundoff_feasible_slack_tol(1e6) > MONOTONE_CONE_FEASIBILITY_GATE_TOL,
            "the relative term must still dominate at large problem scale"
        );
    }
}
