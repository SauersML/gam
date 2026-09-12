//! `CustomFamily` implementations for the latent-survival and latent-binary
//! families.
//!
//! Carved out of `survival.rs` under #2601 only because that file reached the
//! repository's 10,000-line ceiling and no build in the workspace could run.
//! The two `impl` blocks moved here verbatim; they are a natural unit (the
//! solver-facing trait surface of the two latent families) and they reference
//! the module's private machinery through `use super::*`, exactly as they did
//! when they were lexically inside it.

use super::*;

impl CustomFamily for LatentSurvivalFamily {
    // Latent survival fits keep the self-limiting Jeffreys/Firth curvature
    // active for their under-identification regime. The trait default flipped to
    // OFF in gam#1395 (flat-prior exact-Newton objective); opt back in here.
    fn joint_jeffreys_term_required(&self) -> bool {
        true
    }

    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    fn has_explicit_joint_hessian(&self) -> bool {
        true
    }

    /// Route the pre-fit identifiability audit CHANNEL-AWARE across the latent
    /// survival blocks. The time-transform baseline `q(a)`, the frailty mean
    /// `μ = Xβ`, and the learnable log-σ scale are three STRUCTURALLY DISTINCT
    /// outputs — block-diagonal entries of the true joint Jacobian
    /// `blkdiag(X_time, X_mean, X_logσ)`, full rank `Σ p_b`. The blocks are
    /// hand-built (no `jacobian_callback`), so without this the flat single-
    /// channel audit runs.
    ///
    /// That flat audit is fatal for a learnable scale: the `log_sigma` block's
    /// design is a single structural constant-of-ones column (`eta = design·β`
    /// broadcasts the scalar log-σ to every row — see `build_log_sigma_blockspec`).
    /// The flat RRQR sees that constant as a repeated shared basis and mistakes
    /// it for a cross-block ALIAS with the `mean` intercept, then — because the
    /// log-σ block carries the lowest gauge priority — attributes the drop to
    /// `log_sigma[0]` and demotes it below the rank tolerance. That both FREEZES
    /// the frailty scale at its seed (the only handle on σ is deleted) and leaves
    /// the reduced spec width one short of the family's raw joint Hessian, so the
    /// outer LAML logdet aborts with `joint exact-newton Hessian validation …:
    /// got (p+1)×(p+1), expected p×p`. The mean intercept and log-σ are NOT
    /// aliased: they parameterise the frailty MEAN and VARIANCE respectively and
    /// are jointly identified; the collinearity is an artefact of auditing a
    /// nonlinear scale channel as if it were a linear predictor. Placing each
    /// block on its own output channel makes the audit see the genuine
    /// block-diagonal structure. Mirrors
    /// `SurvivalLocationScaleFamily::output_channel_assignment`.
    fn output_channel_assignment(&self, specs: &[ParameterBlockSpec]) -> Option<Vec<usize>> {
        Some(
            specs
                .iter()
                .map(|spec| match spec.name.as_str() {
                    "time_transform" => 0,
                    "mean" => 1,
                    "log_sigma" => 2,
                    _ => 0,
                })
                .collect(),
        )
    }

    /// Engage the inner self-vanishing Levenberg–Marquardt μ on a full-rank but
    /// indefinite / ill-conditioned penalized joint Hessian, mirroring the
    /// sibling `SurvivalMarginalSlopeFamily`. Interval-censored rows contribute
    /// `ℓ = log[S(L) − S(R)]`, the log of a DIFFERENCE of two survival kernels:
    /// unlike the log-concave exact-event / right-censored contributions, its
    /// per-row Hessian is legitimately INDEFINITE away from the optimum, so the
    /// coupled exact-joint penalized Hessian on the constrained (monotone-cone)
    /// time block can be full-rank (`nullity == 0`) yet indefinite or severely
    /// ill-conditioned at the cold-start seed. The constrained-QP path already
    /// REFLECTS negative-curvature modes to `|λ|` (a convex modified-Newton
    /// model), but with this gate OFF it adds NO diagonal floor on a full-rank
    /// ill-conditioned reflected model, so the trust-region Newton oscillates on
    /// the near-singular mode and stalls out the inner budget before any KKT
    /// snapshot is taken ("exited the joint Newton path before convergence — no
    /// math snapshot"). Arming the gate adds the SAME self-vanishing μ
    /// (∝ the projected KKT residual `‖∇ℓ − Sβ + ∇Φ‖` → 0 at the fixed point) the
    /// marginal-slope survival inner relies on, so the step is a well-damped
    /// modified-Newton descent that converges, while the converged β̂ is the
    /// EXACT unconditioned optimum (μ → 0 there) — zero REML/LAML bias, exact
    /// gradient unchanged.
    fn levenberg_on_ill_conditioning(&self) -> bool {
        true
    }

    fn coefficient_hessian_cost(&self, specs: &[ParameterBlockSpec]) -> u64 {
        // `evaluate_exact_newton_joint_dense` builds a fully dense joint
        // Hessian over (Σ p_b)² across time, mean, and optional log-σ blocks
        // via per-row pullback of the latent-survival primary kernel.
        crate::custom_family::joint_coupled_coefficient_hessian_cost(
            self.event_target.len() as u64,
            specs,
        )
    }

    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let (ll, joint_gradient, hess_time, hess_mean, hess_log_sigma) =
            self.evaluate_exact_newton_block_diagonals(block_states)?;
        let block_ranges = self.joint_block_ranges();
        let mut blockworking_sets = vec![
            BlockWorkingSet::ExactNewton {
                gradient: joint_gradient.slice(s![block_ranges[0].clone()]).to_owned(),
                hessian: SymmetricMatrix::Dense(hess_time),
            },
            BlockWorkingSet::ExactNewton {
                gradient: joint_gradient.slice(s![block_ranges[1].clone()]).to_owned(),
                hessian: SymmetricMatrix::Dense(hess_mean),
            },
        ];
        if let (Some(range), Some(hessian)) = (block_ranges.get(2).cloned(), hess_log_sigma) {
            blockworking_sets.push(BlockWorkingSet::ExactNewton {
                gradient: joint_gradient.slice(s![range]).to_owned(),
                hessian: SymmetricMatrix::Dense(hessian),
            });
        }
        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets,
        })
    }

    /// The scalar the joint-Newton accept test evaluates at the TRIAL β.
    ///
    /// # This must be the same function as the gradient hook's, not a second
    /// implementation of it (#2714)
    ///
    /// The trust ratio's numerator is `old_objective − trial_objective`, and the
    /// two ends are produced by different hooks: `old_objective` is built from
    /// `current_log_likelihood`, which `load_joint_gradient_evaluation` reads off
    /// [`Self::exact_newton_joint_gradient_evaluation`], while `trial_objective`
    /// is built from here. Writing `b(β)` for the gap between them, and noting
    /// that the base point does not move across a backtracking ladder,
    ///
    /// ```text
    /// actual_reduction = −[ℓ(β+δ) − ℓ(β)] − b(β) + (penalty terms),
    /// ```
    ///
    /// so `b` is a CONSTANT of the ladder: shrinking the radius shrinks the
    /// bracket and leaves `b` alone, and `actual_reduction → −b` instead of
    /// `→ 0`. Below the radius where the true reduction falls under `|b|` the
    /// sign of `b` decides every attempt outright — `b > 0` refuses all of them,
    /// which is the `rejects[model,likelihood,objective,feasibility] = [0,0,2,0]`
    /// partition at trust radius `1e-12` this family was filed on.
    ///
    /// This used to sum `LatentSurvivalRowJet::evaluate(..).log_lik`, assembled
    /// by `LogKernelSumJet` over the RUNG basis, against a gradient hook summing
    /// the row program's `∂_a^j K₀`-basis value channel. Those are an exact
    /// integer change of basis in real arithmetic (`m^k K_k = (−1)^k (∂_a)_k K₀`)
    /// and two different quadratures in f64, so `b ≠ 0` by construction on any
    /// row whose term list reaches `k ≥ 1` — i.e. every exact-event row, which is
    /// most of a survival dataset. `k = 0` lists agreed anyway, because the
    /// tower's entry 0 is `log_values[0]` verbatim, which is why right-censored
    /// rows were silent about it.
    ///
    /// It now evaluates the SAME row expression through
    /// `latent_survival_row_primary_value` and sums it through the SAME
    /// deterministic reduction, so the two scalars are bit-identical rather than
    /// merely close, and `b ≡ 0`. It is not slower for it: the value backend
    /// builds the one kernel bundle the gradient hook builds and then evaluates a
    /// single term list instead of the `K + K(K+1)/2` normalised moments the
    /// order-two lift needs.
    fn log_likelihood_only(&self, block_states: &[ParameterBlockState]) -> Result<f64, String> {
        let weights = ValidatedLikelihoodWeights::new(&self.weights, "latent-survival")
            .map_err(String::from)?;
        let (q_entry, q_exit, qdot_exit, mu) = self.split_time_eta(block_states)?;
        let q_right = self.time_q_right(block_states)?;
        let sigma = self.latent_sd(block_states)?;
        let include_log_sigma = self.joint_slices().log_sigma.is_some();
        // The SAME chunked reduction `evaluate_exact_newton_joint_gradient_dense`
        // uses. A different summation order over the same per-row values would
        // reintroduce `b` at `n·ε`, which is small but is still a systematic
        // constant of the ladder rather than noise.
        let total = deterministic_latent_survival_row_reduction(
            self.event_target.len(),
            CompensatedRowSum::default,
            |row_idx, acc| {
                let wi = weights.at(row_idx);
                if wi == 0.0 {
                    return Ok(());
                }
                let row = self.build_row_at(
                    row_idx,
                    q_entry[row_idx],
                    q_exit[row_idx],
                    qdot_exit[row_idx],
                    q_right[row_idx],
                )?;
                let row_ll = latent_survival_row_primary_value(
                    &self.quadctx,
                    &row,
                    LatentSurvivalPrimaryPoint {
                        q_entry: q_entry[row_idx],
                        q_exit: q_exit[row_idx],
                        qdot_exit: qdot_exit[row_idx],
                        q_right: q_right[row_idx],
                        mu: mu[row_idx],
                        sigma,
                    },
                    include_log_sigma,
                )?;
                acc.add(checked_weighted_row_value(
                    wi,
                    row_ll,
                    row_idx,
                    "log likelihood",
                )?);
                Ok(())
            },
            |total_acc, chunk_acc| total_acc.add(chunk_acc.value()),
        )?;
        require_finite_likelihood_scalar(total.value(), "log likelihood")
    }

    fn block_linear_constraints(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        block_spec: &ParameterBlockSpec,
    ) -> Result<Option<ConstraintSet>, String> {
        assert!(!block_spec.name.is_empty());
        // Constraints are requested per block, so the index must address a
        // block that actually carries state.
        if block_idx >= block_states.len() {
            return Err(format!(
                "block_linear_constraints: block {block_idx} ({}) is outside the {} \
                 parameter blocks carrying state",
                block_spec.name,
                block_states.len()
            ));
        }
        if block_idx == Self::BLOCK_TIME {
            Ok(self
                .time_linear_constraints
                .clone()
                .map(ConstraintSet::Dense))
        } else {
            Ok(None)
        }
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        self.evaluate_exact_newton_joint_dense(block_states)
            .map(|(_, _, hessian)| Some(hessian))
    }

    fn exact_newton_joint_hessian_workspace(
        &self,
        block_states: &[ParameterBlockState],
        block_specs: &[ParameterBlockSpec],
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        // States and specs are parallel per-block arrays; a length mismatch
        // means the caller assembled an inconsistent parameter partition.
        if block_specs.len() != block_states.len() {
            return Err(format!(
                "exact_newton_joint_hessian_workspace: {} parameter-block specs for {} \
                 block states",
                block_specs.len(),
                block_states.len()
            ));
        }
        Ok(Some(Arc::new(LatentSurvivalHessianWorkspace::new(
            self.clone(),
            block_states.to_vec(),
        ))))
    }

    fn exact_newton_joint_gradient_evaluation(
        &self,
        block_states: &[ParameterBlockState],
        block_specs: &[ParameterBlockSpec],
    ) -> Result<Option<ExactNewtonJointGradientEvaluation>, String> {
        // Same parallel-array precondition as the workspace hook above.
        if block_specs.len() != block_states.len() {
            return Err(format!(
                "exact_newton_joint_gradient_evaluation: {} parameter-block specs for {} \
                 block states",
                block_specs.len(),
                block_states.len()
            ));
        }
        self.evaluate_exact_newton_joint_gradient_dense(block_states)
            .map(|(log_likelihood, gradient)| {
                Some(ExactNewtonJointGradientEvaluation {
                    log_likelihood,
                    gradient,
                })
            })
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_directional_derivative_dense(block_states, d_beta_flat)
            .map(Some)
    }

    fn exact_newton_joint_hessiansecond_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_second_directional_derivative_dense(
            block_states,
            d_beta_u_flat,
            d_beta_v_flat,
        )
        .map(Some)
    }

    /// One-pass Jeffreys completion (#2714). The latent-survival Jeffreys
    /// information is the observed joint Hessian (trait default), so the
    /// contracted second derivative is `∇²_β tr(W · H(β))`, the object the
    /// pairwise `H''[e_a, e_b]` fallback assembles in `p(p+1)/2` full-data
    /// passes. Gated exactly as that fallback's delegate is.
    fn joint_jeffreys_information_contracted_trace_hessian_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        weight: &Array2<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if !self.outer_default_trustworthy_for_joint_hessian(specs)
            && !self.joint_hessian_is_structurally_coupled(block_states)?
        {
            return Ok(None);
        }
        self.jeffreys_information_contracted_trace_hessian_dense(block_states, weight)
            .map(Some)
    }

    fn joint_jeffreys_information_contracted_trace_hessian_available(&self) -> bool {
        true
    }

    fn requires_joint_outer_hyper_path(&self) -> bool {
        true
    }
}

impl CustomFamily for LatentBinaryFamily {
    // Latent binary fits have a separation regime; keep the self-limiting
    // Jeffreys/Firth curvature active. The trait default flipped to OFF in
    // gam#1395 (flat-prior exact-Newton objective); opt back in here.
    fn joint_jeffreys_term_required(&self) -> bool {
        true
    }

    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    fn has_explicit_joint_hessian(&self) -> bool {
        true
    }

    /// Same self-vanishing Levenberg–Marquardt gate as
    /// [`LatentSurvivalFamily`]: the latent-binary deployment shares the
    /// constrained (monotone-cone) coupled time block, so a full-rank but
    /// ill-conditioned penalized joint Hessian at the cold-start seed must get
    /// the self-vanishing μ floor rather than oscillating the constrained-QP
    /// trust region into a snapshot-less stall. μ → 0 at the fixed point, so the
    /// converged β̂ is exact (no REML/LAML bias).
    fn levenberg_on_ill_conditioning(&self) -> bool {
        true
    }

    fn coefficient_hessian_cost(&self, specs: &[ParameterBlockSpec]) -> u64 {
        crate::custom_family::joint_coupled_coefficient_hessian_cost(
            self.event_target.len() as u64,
            specs,
        )
    }

    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let weights = ValidatedLikelihoodWeights::new(&self.weights, "latent-binary")
            .map_err(String::from)?;
        let (q_entry, q_exit, mu) = self.split_time_eta(block_states)?;
        let n = self.event_target.len();
        let p_time = self.x_time_exit.ncols();
        let p_mean = self.x_mean.ncols();

        let mut ll = CompensatedRowSum::default();
        let mut grad_time = Array1::<f64>::zeros(p_time);
        let mut hess_time = Array2::<f64>::zeros((p_time, p_time));
        let mut grad_mean = Array1::<f64>::zeros(p_mean);
        let mut hess_mean = Array2::<f64>::zeros((p_mean, p_mean));
        // Reusable 1-row buffer for x_mean so we avoid allocating a fresh
        // Array2<f64> on every iteration via try_row_chunk(i..i+1).
        let mut mean_row_buf = Array2::<f64>::zeros((1, p_mean));

        for i in 0..n {
            let wi = weights.at(i);
            if wi == 0.0 {
                continue;
            }
            if !(q_entry[i].is_finite() && q_exit[i].is_finite() && mu[i].is_finite()) {
                return Err(format!(
                    "latent-binary row {i} contains non-finite predictors: q_entry={}, q_exit={}, mu={}",
                    q_entry[i], q_exit[i], mu[i]
                ));
            }
            let row = self.build_right_censored_row_at(i, q_entry[i], q_exit[i])?;
            let survival_jet =
                LatentSurvivalRowJet::evaluate(&self.quadctx, &row, mu[i], self.latent_sd)
                    .map_err(|e| format!("LatentBinaryFamily row {i}: {e}"))?;
            let binary = binary_from_log_survival(survival_jet.log_lik, self.event_target[i])?;
            ll.add(checked_weighted_row_value(
                wi,
                binary.log_lik,
                i,
                "binary log likelihood",
            )?);

            self.x_mean
                .row_chunk_into(i..i + 1, mean_row_buf.view_mut())
                .map_err(|e| format!("LatentBinaryFamily row {i} mean row_chunk: {e}"))?;
            let mean_vec = mean_row_buf.row(0);
            let mean_grad_scale = checked_weighted_row_value(
                wi,
                binary.grad_scale * survival_jet.score,
                i,
                "binary mean gradient scale",
            )?;
            for j in 0..p_mean {
                grad_mean[j] += mean_grad_scale * mean_vec[j];
            }
            let mean_neg_hess = checked_weighted_row_value(
                wi,
                binary.neg_hess_scale * survival_jet.neg_hessian
                    + binary.outer_scale * survival_jet.score * survival_jet.score,
                i,
                "binary mean Hessian scale",
            )?;
            dense_outer_accumulate(&mut hess_mean, mean_neg_hess, mean_vec);

            let time_jet =
                latent_survival_time_jet(&self.quadctx, &row, 0.0, mu[i], self.latent_sd)?;
            let t_entry = self.x_time_entry.row(i);
            let t_exit = self.x_time_exit.row(i);
            let time_gradient_scale =
                checked_weighted_row_value(wi, binary.grad_scale, i, "binary time gradient scale")?;
            for j in 0..p_time {
                grad_time[j] += time_gradient_scale
                    * (time_jet.grad_entry * t_entry[j] + time_jet.grad_exit * t_exit[j]);
            }
            let entry_hessian_scale = checked_weighted_row_value(
                wi,
                binary.neg_hess_scale * time_jet.neg_hess_entry,
                i,
                "binary entry Hessian scale",
            )?;
            dense_outer_accumulate(&mut hess_time, entry_hessian_scale, t_entry);
            let exit_hessian_scale = checked_weighted_row_value(
                wi,
                binary.neg_hess_scale * time_jet.neg_hess_exit,
                i,
                "binary exit Hessian scale",
            )?;
            dense_outer_accumulate(&mut hess_time, exit_hessian_scale, t_exit);
            if binary.outer_scale != 0.0 {
                let entry_outer_scale = checked_weighted_row_value(
                    wi,
                    binary.outer_scale * time_jet.grad_entry * time_jet.grad_entry,
                    i,
                    "binary entry outer Hessian scale",
                )?;
                dense_outer_accumulate(&mut hess_time, entry_outer_scale, t_entry);
                let exit_outer_scale = checked_weighted_row_value(
                    wi,
                    binary.outer_scale * time_jet.grad_exit * time_jet.grad_exit,
                    i,
                    "binary exit outer Hessian scale",
                )?;
                dense_outer_accumulate(&mut hess_time, exit_outer_scale, t_exit);
                let cross_outer_scale = checked_weighted_row_value(
                    wi,
                    binary.outer_scale * time_jet.grad_entry * time_jet.grad_exit,
                    i,
                    "binary cross outer Hessian scale",
                )?;
                dense_symmetric_cross_accumulate(
                    &mut hess_time,
                    cross_outer_scale,
                    t_entry,
                    t_exit,
                );
            }
        }

        let ll = require_finite_likelihood_scalar(ll.value(), "binary log likelihood")?;
        require_finite_likelihood_vector(&grad_time, "binary time gradient")?;
        require_finite_likelihood_vector(&grad_mean, "binary mean gradient")?;
        require_finite_likelihood_matrix(&hess_time, "binary time Hessian")?;
        require_finite_likelihood_matrix(&hess_mean, "binary mean Hessian")?;
        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets: vec![
                BlockWorkingSet::ExactNewton {
                    gradient: grad_time,
                    hessian: SymmetricMatrix::Dense(hess_time),
                },
                BlockWorkingSet::ExactNewton {
                    gradient: grad_mean,
                    hessian: SymmetricMatrix::Dense(hess_mean),
                },
            ],
        })
    }

    /// Same contract as [`LatentSurvivalFamily::log_likelihood_only`], and the
    /// same repair (#2714): this is the trial end of the trust ratio's numerator
    /// and `evaluate_exact_newton_joint_dense` is the base end, so the row
    /// log-survival both of them start from has to be ONE function.
    ///
    /// Latent-binary rows are right-censored, whose term lists reach only
    /// `k = 0`, where the two bases agree bit-for-bit — so the gap here was not
    /// the basis but the composition: `unloaded_offset + num − den` associates
    /// differently from `(num − den) + unloaded_offset`, and `ln(m·q̇)` differently
    /// from `ln m + ln q̇`. One ulp per row is still a systematic constant of the
    /// backtracking ladder rather than noise, and there is no reason to carry two
    /// spellings of one row expression.
    fn log_likelihood_only(&self, block_states: &[ParameterBlockState]) -> Result<f64, String> {
        let weights = ValidatedLikelihoodWeights::new(&self.weights, "latent-binary")
            .map_err(String::from)?;
        let (q_entry, q_exit, mu) = self.split_time_eta(block_states)?;
        let mut ll = CompensatedRowSum::default();
        for i in 0..self.event_target.len() {
            let wi = weights.at(i);
            if wi == 0.0 {
                continue;
            }
            let row = self.build_right_censored_row_at(i, q_entry[i], q_exit[i])?;
            let row_log_survival = latent_survival_row_primary_value(
                &self.quadctx,
                &row,
                LatentSurvivalPrimaryPoint {
                    q_entry: q_entry[i],
                    q_exit: q_exit[i],
                    qdot_exit: 1.0,
                    q_right: q_exit[i],
                    mu: mu[i],
                    sigma: self.latent_sd,
                },
                false,
            )?;
            let binary_log_lik =
                binary_log_likelihood_from_log_survival(row_log_survival, self.event_target[i])?;
            ll.add(checked_weighted_row_value(
                wi,
                binary_log_lik,
                i,
                "binary log likelihood",
            )?);
        }
        require_finite_likelihood_scalar(ll.value(), "binary log likelihood")
    }

    fn block_linear_constraints(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        block_spec: &ParameterBlockSpec,
    ) -> Result<Option<ConstraintSet>, String> {
        assert!(!block_spec.name.is_empty());
        // Constraints are requested per block, so the index must address a
        // block that actually carries state.
        if block_idx >= block_states.len() {
            return Err(format!(
                "block_linear_constraints: block {block_idx} ({}) is outside the {} \
                 parameter blocks carrying state",
                block_spec.name,
                block_states.len()
            ));
        }
        if block_idx == Self::BLOCK_TIME {
            Ok(self
                .time_linear_constraints
                .clone()
                .map(ConstraintSet::Dense))
        } else {
            Ok(None)
        }
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        self.evaluate_exact_newton_joint_dense(block_states)
            .map(|(_, _, hessian)| Some(hessian))
    }

    fn exact_newton_joint_hessian_workspace(
        &self,
        block_states: &[ParameterBlockState],
        block_specs: &[ParameterBlockSpec],
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        // States and specs are parallel per-block arrays; a length mismatch
        // means the caller assembled an inconsistent parameter partition.
        if block_specs.len() != block_states.len() {
            return Err(format!(
                "exact_newton_joint_hessian_workspace: {} parameter-block specs for {} \
                 block states",
                block_specs.len(),
                block_states.len()
            ));
        }
        Ok(Some(Arc::new(LatentBinaryHessianWorkspace::new(
            self.clone(),
            block_states.to_vec(),
        ))))
    }

    fn exact_newton_joint_gradient_evaluation(
        &self,
        block_states: &[ParameterBlockState],
        block_specs: &[ParameterBlockSpec],
    ) -> Result<Option<ExactNewtonJointGradientEvaluation>, String> {
        // Same parallel-array precondition as the workspace hook above.
        if block_specs.len() != block_states.len() {
            return Err(format!(
                "exact_newton_joint_gradient_evaluation: {} parameter-block specs for {} \
                 block states",
                block_specs.len(),
                block_states.len()
            ));
        }
        self.evaluate_exact_newton_joint_dense(block_states)
            .map(|(log_likelihood, gradient, _)| {
                Some(ExactNewtonJointGradientEvaluation {
                    log_likelihood,
                    gradient,
                })
            })
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_directional_derivative_dense(block_states, d_beta_flat)
            .map(Some)
    }

    fn exact_newton_joint_hessiansecond_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_second_directional_derivative_dense(
            block_states,
            d_beta_u_flat,
            d_beta_v_flat,
        )
        .map(Some)
    }

    fn requires_joint_outer_hyper_path(&self) -> bool {
        true
    }
}
