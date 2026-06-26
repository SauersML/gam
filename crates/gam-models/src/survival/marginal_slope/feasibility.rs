//! Feasibility and monotonicity enforcement: maximum feasible time step,
//! the time-derivative and score-warp linear constraints, and validation
//! that a candidate beta keeps the time derivative / monotonicity feasible.

use super::*;

impl SurvivalMarginalSlopeFamily {
    pub(crate) fn max_feasible_time_step(
        &self,
        beta: &Array1<f64>,
        delta: &Array1<f64>,
    ) -> Result<Option<f64>, String> {
        let Some(constraints) = self.effective_time_linear_constraints()? else {
            return Ok(None);
        };
        crate::marginal_slope_shared::feasible_step_fraction(
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
