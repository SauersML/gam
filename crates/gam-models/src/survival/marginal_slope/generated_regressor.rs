//! The Murphy–Topel generated-regressor seam for the survival marginal-slope
//! family (gam#2768).
//!
//! When the automatic conditional location-scale gate fires, the fit consumes a
//! score `ζ = (z − m̂(C))/√v̂(C)` whose first stage `θ₁ = (mean_coeffs,
//! var_coeffs)` was *estimated from the same data*. The joint solve treats ζ as
//! KNOWN, so `covariance_conditional` is the naive second-stage covariance
//! `H_β⁻¹`. The honest two-stage covariance is
//!
//! ```text
//!     V_β = V_β^naive + (H_β⁻¹ G) V₁ (H_β⁻¹ G)ᵀ ,    G = ∂(score_β)/∂θ₁ ,
//! ```
//!
//! which is PSD, so the naive covariance is always too *narrow* — and per
//! gam#2718 publishing it uncorrected is inadmissible, because the intervals
//! come out too tight and are indistinguishable on the wire from corrected ones.
//!
//! [`LatentZConditionalCalibration::generated_regressor_correction`] owns every
//! family-independent piece (`V₁`, `∂ζ/∂θ₁`, the congruence). The only thing a
//! family owes it is the per-row channel
//! `s_i = ∂(score_β,i)/∂ζ_i`, and that is what this module supplies for the
//! survival rigid kernel: [`rigid_row_primary_mixed_in_z`] gives the mixed
//! derivative in PRIMARY coordinates, and the same
//! `accumulate_dynamic_q_core_gradient` the production gradient uses scatters it
//! into the β frame — so the sensitivity cannot drift from the score it is a
//! derivative of.

use super::*;

use crate::bms::LatentZConditionalCalibration;

impl SurvivalMarginalSlopeFamily {
    /// `s = ∂(score_β)/∂ζ` at the converged fit: an `n × p_β` matrix whose row
    /// `i` is the derivative of row `i`'s contribution to the β-score in that
    /// row's latent score.
    ///
    /// Rigid, scalar-score only. Both restrictions are refusals rather than
    /// silent zeroes:
    ///
    /// * with a score-warp or link-deviation block active, `ζ_i` reaches the row
    ///   likelihood a second time through the flex basis evaluated *at* `ζ_i`,
    ///   so the mixed derivative is not the rigid one and a rigid `s` would
    ///   understate the correction;
    /// * with `K > 1` the row kernel sees only `z_sum = Σ_k z_k`, so a
    ///   per-coordinate `∂/∂ζ_k` is not separable from the shared-slope
    ///   summary — and a `K > 1` conditional calibration is already refused at
    ///   persistence, for the same reason it cannot be replayed.
    pub(crate) fn rigid_score_zeta_sensitivity(
        &self,
        block_states: &[ParameterBlockState],
        p_beta: usize,
    ) -> Result<Array2<f64>, String> {
        if self.score_warp.is_some() || self.link_dev.is_some() {
            return Err(
                "survival marginal-slope generated-regressor sensitivity is rigid-only: a \
                 score-warp / link-deviation block makes the row likelihood depend on the \
                 latent score a second time, through a basis evaluated at that score"
                    .to_string(),
            );
        }
        if self.score_dim() != 1 {
            return Err(format!(
                "survival marginal-slope generated-regressor sensitivity is scalar-score only; \
                 the shared-slope row kernel sees only z_sum = Σ_k z_k, so ∂/∂ζ_k is not \
                 separable at K={}",
                self.score_dim()
            ));
        }
        if let Some(law) = self.latent_law.as_ref() {
            // The declared law was compressed from the calibrated residual ζ
            // itself, so it moves with the first stage θ₁ through every row at
            // once; the per-row mixed derivative ∂/∂ζ_i is not the whole of
            // ∂(score_β)/∂θ₁, and a correction built from it alone would
            // understate the first-stage uncertainty exactly as gam#2484 names
            // for the Bernoulli family.
            return Err(format!(
                "survival marginal-slope generated-regressor sensitivity is defined for the \
                 standard-normal latent measure only; the row index is anchored on a declared \
                 {} latent law built from the calibrated residual, so the law itself moves with \
                 the first stage and the per-row mixed derivative does not account for that \
                 channel (gam#2484)",
                match &law.kind {
                    crate::bms::LatentMeasureKind::GlobalEmpirical { grid } =>
                        format!("global-empirical ({} nodes)", grid.nodes.len()),
                    crate::bms::LatentMeasureKind::LocalEmpirical { grids, .. } =>
                        format!("local-empirical ({} grids)", grids.len()),
                    crate::bms::LatentMeasureKind::StandardNormal => "standard-normal".to_string(),
                }
            ));
        }
        let slices = block_slices(self, block_states);
        if slices.total != p_beta {
            return Err(format!(
                "survival marginal-slope generated-regressor frame mismatch: covariance width \
                 {p_beta} != block layout total {}",
                slices.total
            ));
        }
        let rows = (0..self.n)
            .into_par_iter()
            .map(|row| -> Result<(usize, Array1<f64>), String> {
                let mut sensitivity = Array1::<f64>::zeros(p_beta);
                if self.weights[row] <= 0.0 {
                    // A zero-weight row contributes nothing to the score, so it
                    // contributes nothing to the score's derivative either.
                    return Ok((row, sensitivity));
                }
                let inputs = rigid_row_inputs(
                    self,
                    block_states,
                    row,
                    "survival marginal-slope generated-regressor sensitivity",
                )?;
                let mixed = in_slope_frame!(self, P, Frame, {
                    let primaries =
                        rigid_row_kernel_primaries::<P, Frame>(self, block_states, row)?;
                    rigid_row_primary_mixed_in_z::<P, Frame>(&primaries, &inputs)
                        .map(|mixed| Array1::from_vec(mixed.to_vec()))
                })?;
                let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
                // The production gradient accumulator, fed the mixed vector
                // instead of the primary gradient. It applies the same `-=` the
                // score does, so the result is `∂(score)/∂ζ` and not
                // `∂(∇NLL)/∂ζ` — the sign the correction's congruence expects,
                // and in any case one it is quadratic in.
                let mixed_view = mixed.view();
                self.accumulate_dynamic_q_core_gradient(
                    row,
                    &slices,
                    &q_geom,
                    mixed_view,
                    &mut sensitivity,
                )?;
                Ok((row, sensitivity))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let mut out = Array2::<f64>::zeros((self.n, p_beta));
        for (row, sensitivity) in rows {
            out.row_mut(row).assign(&sensitivity);
        }
        Ok(out)
    }
}

/// Apply the Murphy–Topel correction in place to a fitted survival
/// marginal-slope covariance, or declare why it was withheld.
///
/// `raw_z` is the score BEFORE the calibration (what the first stage regressed)
/// and `a_block` is the conditioning span the calibration was fit against — the
/// two inputs `generated_regressor_correction` needs beside `s` and the naive
/// covariance.
pub(crate) fn apply_survival_generated_regressor_correction(
    family: &SurvivalMarginalSlopeFamily,
    calibration: &LatentZConditionalCalibration,
    fit: &mut UnifiedFitResult,
    raw_z: ArrayView1<'_, f64>,
    a_block: ArrayView2<'_, f64>,
) -> Result<(), String> {
    let Some(naive) = fit.covariance_conditional.clone() else {
        // No covariance was computed, so there is nothing to correct and
        // nothing to withhold. `Some` on the declined field must always mean "a
        // covariance existed and was taken away".
        return Ok(());
    };
    if naive.nrows() != naive.ncols() {
        return Err(format!(
            "survival marginal-slope generated-regressor: covariance_conditional must be \
             square, got {}×{}",
            naive.nrows(),
            naive.ncols()
        ));
    }
    let p_beta = naive.nrows();
    let sensitivity =
        match family.rigid_score_zeta_sensitivity(&fit.block_states, p_beta) {
            Ok(sensitivity) => sensitivity,
            Err(reason) => {
                // The correction is unavailable for this fit's shape. Withhold
                // the covariance and SAY SO on the fit's own payload — the
                // uncorrected one omits exactly the first-stage uncertainty the
                // correction exists to add, so it would ship intervals that are
                // too narrow and indistinguishable from corrected ones
                // (gam#2718). The point estimates are unaffected and published.
                withhold_covariance(fit, &reason);
                return Ok(());
            }
        };
    let correction = calibration.generated_regressor_correction(
        sensitivity.view(),
        raw_z,
        a_block,
        naive.view(),
    )?;
    // gam#2943: the correction applies to the one published covariance store,
    // whose standard errors derive from it (#2955).
    fit.add_coefficient_covariance_correction(&correction)
        .map_err(|err| format!("survival marginal-slope generated-regressor: {err}"))?;
    log::info!(
        "[survival-marginal-slope latent-z] Murphy–Topel generated-regressor SE correction \
         applied: p_beta={p_beta} theta1_dim={} max_diag_inflation={:.3e}",
        calibration.theta1_dim(),
        (0..p_beta)
            .map(|i| correction[[i, i]])
            .fold(0.0_f64, f64::max),
    );
    Ok(())
}

fn withhold_covariance(fit: &mut UnifiedFitResult, reason: &str) {
    fit.covariance_conditional = None;
    fit.covariance_corrected = None;
    if let Some(inference) = fit.inference.as_mut() {
        inference.factorized_standard_errors = None;
    }
    let declined = gam_solve::estimate::CovarianceDeclined::
        SurvivalMarginalSlopeGeneratedRegressorSensitivityUnavailable {
            unavailable_channel: reason.to_string(),
        };
    log::warn!("[survival-marginal-slope latent-z] {}", declined.explain());
    fit.artifacts.covariance_declined = Some(declined);
}
