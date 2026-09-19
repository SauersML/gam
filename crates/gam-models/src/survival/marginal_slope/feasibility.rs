//! Feasibility and monotonicity enforcement: maximum feasible time step,
//! the time-derivative and score-warp linear constraints, and validation
//! that a candidate beta keeps the time derivative / monotonicity feasible.

use super::*;

/// Render a shared barrier-step refusal into the marginal-slope error
/// vocabulary. A width disagreement is a dimension fault; everything else is a
/// statement about the derivative guard's geometry at the current iterate.
fn map_time_barrier_step_error(error: gam_problem::ContractFeasibleStepError) -> String {
    let reason = format!("survival marginal-slope time-block step: {error}");
    match error {
        gam_problem::ContractFeasibleStepError::Dimension { .. } => {
            SurvivalMarginalSlopeError::IncompatibleDimensions { reason }.into()
        }
        _ => SurvivalMarginalSlopeError::MonotonicityViolation { reason }.into(),
    }
}

impl SurvivalMarginalSlopeFamily {
    pub(crate) fn max_feasible_time_step(
        &self,
        beta: &Array1<f64>,
        delta: &Array1<f64>,
    ) -> Result<Option<f64>, String> {
        let Some(constraints) = self.effective_time_linear_constraints()? else {
            return Ok(None);
        };
        crate::marginal_slope_shared::feasible_step_fraction(&constraints, beta, delta)
            .map(Some)
            .map_err(map_time_barrier_step_error)
    }

    pub(crate) fn effective_time_linear_constraints(
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

    pub(crate) fn score_warp_linear_constraints(
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
        LinearInequalityConstraints::new(a, b)
    }

    pub(crate) fn validate_time_qd1_feasible(
        &self,
        beta: &Array1<f64>,
        label: &str,
    ) -> Result<(), String> {
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
        let feasibility_band = 4.0 * gam_solve::pirls::ACTIVE_SET_PRIMAL_FEASIBILITY_TOL;
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

    pub(crate) fn validate_exact_monotonicity(
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
}

// ── The follow-up-varying likelihood domain (gam#2765 / gam#2767) ───────────
//
// The row log-density carries `log η′₁`, so the family's domain is
//
//     η′₁(t) = q′(t)·c(t) + q(t)·c′(t) + b′(t)ᵀz > 0   at every EVENT row,
//
// with `c = √(1 + bᵀΣb)`. With a time-CONSTANT slope the last two terms are
// identically zero and `η′₁ = q′·c ≥ q′`, so the time block's own linear guard
// `q′ ≥ derivative_guard > 0` IMPLIES the domain — which is exactly why the
// solver's feasible set has always been that one polytope, and why a per-BLOCK
// `max_feasible_step_size` sufficed.
//
// A follow-up-varying slope breaks the implication: `q·c′` and `b′ᵀz` carry no
// sign, and they read the marginal and slope blocks. The condition was
// therefore placed in the likelihood DOMAIN — `+∞` outside, refusal at the row
// evaluator — rather than in the FEASIBLE SET. Measured on the #2765 acceptance
// fixture that costs the fit its convergence: the criterion is finite on one
// side of a surface the outer BFGS cannot see, every probe across it is refused
// with no gradient information, the line search halves to nothing, and the outer
// runner reports `cost-stall STUCK (infeasible BFGS probes)` at `|g| = 5.4e1`
// against its own `1.5e0` escape threshold.
//
// The rule below puts the condition back where a solver can use it: a JOINT
// fraction-to-boundary limit, evaluated on the SAME witness the row program
// admits on, so "the step limiter says feasible" and "the row program accepts"
// are the same arithmetic and cannot disagree.
impl SurvivalMarginalSlopeFamily {
    /// `min η′₁` over the rows whose likelihood contains `log η′₁` — the EVENT
    /// rows — at the given coefficients.
    ///
    /// `None` on the time-constant frame: there the quantity is implied by the
    /// time block's linear guard and this rule has nothing to add.
    ///
    /// The witness is read through [`rigid_row_admission_witnesses`], the same
    /// call the row evaluator's own admission uses, at primaries built by the
    /// same [`rigid_row_kernel_primaries`]. That is deliberate and load-bearing:
    /// a step limiter that computed `η′₁` its own way would be a second copy of
    /// the model, and the two copies would eventually disagree about which side
    /// of the boundary a coefficient is on.
    pub(crate) fn follow_up_domain_margin(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<f64>, String> {
        if !self.slope_is_follow_up_varying() {
            return Ok(None);
        }
        let margin = (0..self.n)
            .into_par_iter()
            .map(|row| -> Result<f64, String> {
                // A censored row's density has no `log η′₁` factor, so it
                // imposes no domain condition. Mirrors the `di != 0.0` gate in
                // `validate_rigid_row_admission` exactly.
                if self.event[row] == 0.0 {
                    return Ok(f64::INFINITY);
                }
                let inputs = rigid_row_inputs(
                    self,
                    block_states,
                    row,
                    "survival marginal-slope follow-up domain margin",
                )?;
                let primaries = rigid_row_kernel_primaries::<
                    DYNAMIC_SLOPE_PRIMARIES,
                    DynamicSlopeGeometry,
                >(self, block_states, row)?;
                let [_, _, adjusted_derivative] = rigid_row_admission_witnesses::<
                    DYNAMIC_SLOPE_PRIMARIES,
                    DynamicSlopeGeometry,
                >(&primaries, &inputs);
                // A NaN witness is not "a large margin": report it as the worst
                // possible value so a step that produces one is never taken.
                Ok(if adjusted_derivative.is_nan() {
                    f64::NEG_INFINITY
                } else {
                    adjusted_derivative
                })
            })
            .try_reduce(|| f64::INFINITY, |a, b| Ok(a.min(b)))?;
        Ok(Some(margin))
    }

    /// The largest `α ∈ [0, 1]` for which `β + α·δ` is strictly inside the
    /// follow-up-varying likelihood domain, or `None` when the rule does not
    /// apply (time-constant slope, or the whole step is already inside).
    ///
    /// # Why there is no safety fraction
    ///
    /// The usual interior-point retreat exists because the limiter and the
    /// objective compute the boundary differently, so a step taken to the
    /// limiter's boundary can land outside the objective's. Here they are the
    /// same function evaluated by the same code at the same primaries, so the
    /// endpoint this returns is admitted by the row program bit for bit. What
    /// remains — a step that lands legally but very close, where `−log η′₁` is
    /// large — is not a hazard the limiter should price: the objective there is
    /// finite and worse, and the trust region rejects it on its own terms. That
    /// is the mechanism that is supposed to decide step length, and inventing a
    /// second one would shorten steps the criterion is happy with.
    ///
    /// This mirrors [`apply_feasible_step_boundary_backoff`]'s finding on the
    /// linear guard (gam#2695): a multiplicative retreat is a proportionality
    /// the geometry does not have, and it can walk a coefficient toward a face
    /// it never reaches.
    ///
    /// # Search
    ///
    /// `η′₁(α)` is smooth but not monotone in `α`, so the answer is found by
    /// bisecting the bracket `[feasible, infeasible]` and returning the
    /// FEASIBLE end — a point the rule has actually evaluated, never an
    /// interpolated one. The bracket stops once it is finer than the step's own
    /// representability, `α·‖δ‖∞ ≲ ε·‖β‖∞`, below which `β + α·δ` is not a
    /// different point in f64 and no smaller `α` can change the answer.
    pub(crate) fn max_feasible_follow_up_joint_step(
        &self,
        block_states: &[ParameterBlockState],
        delta: &Array1<f64>,
    ) -> Result<Option<f64>, String> {
        if !self.slope_is_follow_up_varying() {
            return Ok(None);
        }
        let Some(base_margin) = self.follow_up_domain_margin(block_states)? else {
            return Ok(None);
        };
        let margin_at = |alpha: f64| -> Result<f64, String> {
            let states = self.displaced_block_states(block_states, delta, alpha)?;
            Ok(self
                .follow_up_domain_margin(&states)?
                .expect("the follow-up frame is what this rule is gated on"))
        };

        if margin_at(1.0)? > 0.0 {
            return Ok(None);
        }
        if !(base_margin > 0.0) {
            // The rule has no bracket: the CURRENT iterate is already outside
            // the domain, which is a statement about how it got there and not
            // about this step. Decline rather than invent an answer — the row
            // program refuses at `β` itself and names the row.
            log::debug!(
                "[survival-marginal-slope/follow-up-domain] declining a joint step limit: the \
                 base iterate is already outside the domain (min η′₁ = {base_margin:.6e})"
            );
            return Ok(None);
        }

        let step_scale = delta.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        let beta_scale = block_states
            .iter()
            .flat_map(|state| state.beta.iter())
            .fold(0.0_f64, |acc, v| acc.max(v.abs()))
            .max(1.0);
        // Below this the displaced coefficients are not a different f64 point,
        // so no finer bracket can change which side of the boundary they sit on.
        let resolution = if step_scale > 0.0 {
            (beta_scale * f64::EPSILON) / step_scale
        } else {
            return Ok(None);
        };

        let mut feasible = 0.0_f64;
        let mut infeasible = 1.0_f64;
        while infeasible - feasible > resolution {
            let midpoint = 0.5 * (feasible + infeasible);
            if midpoint <= feasible || midpoint >= infeasible {
                break;
            }
            if margin_at(midpoint)? > 0.0 {
                feasible = midpoint;
            } else {
                infeasible = midpoint;
            }
        }
        log::debug!(
            "[survival-marginal-slope/follow-up-domain] joint step limited to α={feasible:.6e} \
             (base min η′₁ = {base_margin:.6e}, |δ|∞ = {step_scale:.6e})"
        );
        Ok(Some(feasible))
    }

    /// `(β + α·δ, η + α·Xδ)` for every block: the displaced coefficients AND the
    /// linear predictors that must travel with them.
    ///
    /// The predictors are updated INCREMENTALLY rather than rebuilt. Each
    /// block's `η = X·β + offset` is affine in `β` with a `β`-independent
    /// offset, so `η(β + αδ) = η(β) + α·(X·δ)` exactly — which is what lets
    /// this run without the block SPECS, which the joint feasibility hook is
    /// not handed. Leaving `η` stale instead would silently evaluate the
    /// domain at a mixture of two coefficient vectors: the marginal index `q`
    /// reads `states[1].eta` and the slope's exit channel reads
    /// `states[2].eta`, so a step that moved either block would be graded at
    /// the OLD one.
    ///
    /// The follow-up-varying frame refuses every optional block (score-warp,
    /// link-deviation, CTN influence absorber, time-wiggle, frailty) at
    /// construction, so the layout here is exactly `[time, marginal,
    /// slope]` and each design is one the family itself owns.
    pub(crate) fn displaced_block_states(
        &self,
        block_states: &[ParameterBlockState],
        delta: &Array1<f64>,
        alpha: f64,
    ) -> Result<Vec<ParameterBlockState>, String> {
        let designs: [&DesignMatrix; 3] = [
            &self.design_exit,
            &self.marginal_design,
            self.slope_layout.coefficient_design(),
        ];
        if block_states.len() != designs.len() {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "a follow-up-varying survival marginal-slope fit carries exactly {} \
                     coefficient blocks (time, marginal, slope); got {}",
                    designs.len(),
                    block_states.len(),
                ),
            }
            .into());
        }
        let total: usize = block_states.iter().map(|state| state.beta.len()).sum();
        if delta.len() != total {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival marginal-slope joint feasibility step has {} entries for {total} \
                     coefficients",
                    delta.len(),
                ),
            }
            .into());
        }
        let mut offset = 0usize;
        let mut out = Vec::with_capacity(block_states.len());
        for (state, design) in block_states.iter().zip(designs) {
            let width = state.beta.len();
            let block_delta = delta.slice(s![offset..offset + width]).to_owned();
            let mut moved = state.clone();
            moved.beta = &state.beta + &(&block_delta * alpha);
            let eta_step = design.matrixvectormultiply(&block_delta);
            if eta_step.len() != state.eta.len() {
                return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                    reason: format!(
                        "survival marginal-slope joint feasibility predictor step has {} rows \
                         against a block predictor of {}",
                        eta_step.len(),
                        state.eta.len(),
                    ),
                }
                .into());
            }
            moved.eta = &state.eta + &(&eta_step * alpha);
            out.push(moved);
            offset += width;
        }
        Ok(out)
    }


    /// Bring a WARM-START coefficient vector back inside the follow-up-varying
    /// likelihood domain, by retreating the slope block toward its own
    /// origin.
    ///
    /// # Why a seed can be outside a domain it was inside
    ///
    /// The domain condition `η′₁ > 0` reads the baseline chart through `q′` and
    /// `q`. The outer search moves that chart (ρ and the parametric-baseline ψ)
    /// while the warm start carries β forward unchanged, so a β that was
    /// interior at the previous θ can be exterior at the next one — through no
    /// property of β at all. The evaluation then refuses the whole trial point,
    /// and because it refuses rather than returns a value the outer line search
    /// gets no information from it: measured on the #2765 acceptance fixture the
    /// BFGS halves its step six times, every probe refuses, and the runner
    /// halts with `cost-stall (infeasible BFGS probes)` at `|g| = 3.9e1` against
    /// its own `1.5e0` escape threshold.
    ///
    /// Restoring feasibility is the inner solve's job, not the caller's, and it
    /// is the exact analogue of what the time block already does: its warm start
    /// is projected onto `q′ ≥ derivative_guard` before it is used
    /// (`project_onto_linear_constraints` in the block builder). This is the
    /// slope block's half of the same contract.
    ///
    /// # Why the origin is the retreat target
    ///
    /// At `β_g = 0` the block's rate channel is `X_ġ · 0 = 0` exactly — the
    /// layout's exit-derivative design carries no offset — so `ḃ ≡ 0`, both
    /// follow-up terms of `η′₁` vanish identically, and
    /// `η′₁ = q′·c ≥ q′ ≥ derivative_guard > 0` because `c = √(1 + bᵀΣb) ≥ 1`.
    /// The retreat therefore has a PROVABLY interior endpoint, which is what
    /// makes the bisection below terminate rather than merely usually succeed.
    /// It is also the block's own cold start, so the worst case is "start this
    /// evaluation's slope block from scratch", not "start it somewhere
    /// arbitrary".
    ///
    /// # Why the retreat stops at the guard, not at the boundary
    ///
    /// The time block's seed contract is `q′ ≥ derivative_guard`, not `q′ > 0`,
    /// and this is the same contract for `η′₁`: the retreat stops at the
    /// smallest fraction with `min η′₁ ≥ derivative_guard`. The smallest
    /// fraction with `min η′₁ > 0` is interior by the bisection's resolution
    /// only, and the likelihood's `−log η′₁` then hands the inner solve a seed
    /// whose gradient scales as `1/η′₁` and whose curvature scales as `1/η′₁²`.
    /// Measured on docs/marginal-slope.md:205 (#2627): every seed started at
    /// `min η′₁ = 1.9e-14`, with gradient `6e13` and curvature `4e27`, and every
    /// one was refused at the reduced-face first-order KKT check before the
    /// solver started.
    ///
    /// The time seed meets its guard only to the active-set solver's primal
    /// tolerance, so the origin's `η′₁ = q′·c ≥ q′` can sit a rounding below
    /// the guard. The retreat asks for no more than the origin provides, which
    /// keeps its endpoint provably reachable. This moves a seed only: the step
    /// limiter keeps the `η′₁ > 0` domain, because a guard margin there would be
    /// a constraint on the whole search that the likelihood does not have.
    ///
    /// Returns the retreat fraction actually applied (`0.0` when the seed
    /// already met the guard, in which case nothing is written).
    pub(crate) fn retreat_seed_into_follow_up_domain(
        &self,
        blocks: &mut [ParameterBlockSpec],
    ) -> Result<f64, String> {
        if !self.slope_is_follow_up_varying() {
            return Ok(0.0);
        }
        let states: Vec<ParameterBlockState> = blocks
            .iter()
            .map(|block| {
                let width = block.design.ncols();
                let beta = block
                    .initial_beta
                    .as_ref()
                    .filter(|seed| seed.len() == width)
                    .cloned()
                    .unwrap_or_else(|| Array1::<f64>::zeros(width));
                let eta = block.solver_design().matrixvectormultiply(&beta)
                    + block.solver_offset();
                ParameterBlockState { beta, eta }
            })
            .collect();
        let Some(margin) = self.follow_up_domain_margin(&states)? else {
            return Ok(0.0);
        };
        if margin >= self.derivative_guard {
            return Ok(0.0);
        }
        // The retreat direction: the slope block's own coefficients, pointing
        // at the origin. Every other block is held, because they are not what
        // made the domain condition non-trivial.
        let slope = 2usize;
        let total: usize = states.iter().map(|state| state.beta.len()).sum();
        let mut direction = Array1::<f64>::zeros(total);
        let start: usize = states[..slope]
            .iter()
            .map(|state| state.beta.len())
            .sum();
        let width = states[slope].beta.len();
        for column in 0..width {
            direction[start + column] = -states[slope].beta[column];
        }
        let margin_at = |fraction: f64| -> Result<f64, String> {
            let moved = self.displaced_block_states(&states, &direction, fraction)?;
            Ok(self
                .follow_up_domain_margin(&moved)?
                .expect("the follow-up frame is what this rule is gated on"))
        };
        let origin_margin = margin_at(1.0)?;
        if !(origin_margin > 0.0) {
            // The origin is interior whenever the time block's own guard holds,
            // so reaching here means the time seed is itself infeasible. That is
            // a different defect and this rule must not paper over it: leave the
            // seed alone and let the row evaluator refuse and name its row.
            log::warn!(
                "[survival-marginal-slope/follow-up-domain] the slope origin does not \
                 restore the domain (min η′₁ = {origin_margin:.6e} there, {margin:.6e} at the \
                 seed); the time block's derivative guard must be violated at this theta",
            );
            return Ok(0.0);
        }
        let target = self.derivative_guard.min(origin_margin);
        // Smallest retreat that reaches the target. `interior` is always a
        // fraction this rule has EVALUATED, never an interpolated one.
        let mut exterior = 0.0_f64;
        let mut interior = 1.0_f64;
        let seed_scale = states[slope]
            .beta
            .iter()
            .fold(0.0_f64, |acc, v| acc.max(v.abs()));
        let resolution = if seed_scale > 0.0 {
            f64::EPSILON * seed_scale.max(1.0) / seed_scale
        } else {
            return Ok(0.0);
        };
        while interior - exterior > resolution {
            let midpoint = 0.5 * (exterior + interior);
            if midpoint <= exterior || midpoint >= interior {
                break;
            }
            if margin_at(midpoint)? >= target {
                interior = midpoint;
            } else {
                exterior = midpoint;
            }
        }
        let retreated = &states[slope].beta * (1.0 - interior);
        log::info!(
            "[survival-marginal-slope/follow-up-domain] warm-start slope seed was short of \
             the derivative guard (min η′₁ = {margin:.6e}, guard {:.6e}); retreated {:.4}% \
             toward the block origin",
            self.derivative_guard,
            100.0 * interior,
        );
        blocks[slope].initial_beta = Some(retreated);
        Ok(interior)
    }

}
