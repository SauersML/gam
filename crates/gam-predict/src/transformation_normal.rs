use super::*;
use crate::input::{TRANSFORMATION_NORMAL_BAND_Z_MAX, TRANSFORMATION_NORMAL_BAND_Z_NODES};

/// Predictor for transformation-normal (CTM) models.
///
/// The response-scale conditional mean `E[Y|x]` is precomputed in
/// `build_predict_input_for_model` (issue #1612) and stored in the PredictInput
/// offset. `E[Y|x] = E_{Z~N(0,1)}[h⁻¹(Z|x)]` is a function of the covariates
/// alone, so prediction is covariate-only and does not require the outcome
/// column. This predictor passes the precomputed value through unchanged as both
/// the linear predictor and the mean: eta = mean = E[Y|x].
///
/// ## Uncertainty contract
///
/// * **Epistemic (coefficient) uncertainty is reported as unavailable, never
///   as zero.** Propagating `Cov(β)` into `E[Y|x]` requires the Jacobian of
///   the inverse transform `∂h⁻¹/∂β`, which needs the I-spline basis partials
///   that are not part of the persisted quantile grid. A zero SE claims exact
///   knowledge of `E[Y|x]` the posterior does not have, so the point paths
///   return `None` SEs and `predict_full_uncertainty` errors instead of
///   emitting zero-width intervals.
/// * **Observation (predictive) intervals are exact response-scale quantiles.**
///   The CTM predictive is `Y|x = h⁻¹(Z|x)` with `Z ~ N(0,1)`, so the
///   `p`-quantile of `Y|x` is `h⁻¹(Φ⁻¹(p)|x)`. The input builder tabulates
///   `h⁻¹` on a fixed latent-z ladder (`PredictInput::auxiliary_matrix`);
///   the band interpolates that ladder. Adding standard-normal quantiles to
///   `E[Y|x]` directly would be off by exactly the (row-dependent) scale of
///   `h⁻¹` — for `h(y) = 10·y` the true 95% band is `±0.196`, not `±1.96`.
pub struct TransformationNormalPredictor {
    pub covariance: Option<Array2<f64>>,
}

/// The secant `dy/dz` of the ladder across cell `j`.
#[inline]
fn ladder_secant(ladder_row: ndarray::ArrayView1<'_, f64>, j: usize, step: f64) -> f64 {
    (ladder_row[j + 1] - ladder_row[j]) / step
}

/// `dy/dz` at ladder node `j`, by the shape-preserving (PCHIP / Fritsch-Carlson)
/// rule: the harmonic mean of the two neighbouring secants in the interior, and
/// the one-sided three-point estimate limited to `3·d` at the two ends.
///
/// The harmonic mean is what makes the resulting cubic monotone — it vanishes
/// wherever the data have an extremum and never exceeds three times either
/// neighbouring secant — which matters here because the interpolated object is a
/// *quantile function*: a band whose interpolation could overshoot would report
/// a lower limit above its own upper limit.
///
/// Slopes are computed from the ladder rather than stored because `PredictInput`
/// carries the ladder as a plain `n × m` matrix, and `m = 65` per row is already
/// the memory budget for this path.
fn ladder_slope(ladder_row: ndarray::ArrayView1<'_, f64>, j: usize, step: f64) -> f64 {
    let m = ladder_row.len();
    let one_sided = |near: f64, far: f64| {
        let estimate = 0.5 * (3.0 * near - far);
        if estimate <= 0.0 {
            0.0
        } else if estimate > 3.0 * near {
            3.0 * near
        } else {
            estimate
        }
    };
    if m == 2 {
        return ladder_secant(ladder_row, 0, step);
    }
    if j == 0 {
        return one_sided(
            ladder_secant(ladder_row, 0, step),
            ladder_secant(ladder_row, 1, step),
        );
    }
    if j == m - 1 {
        return one_sided(
            ladder_secant(ladder_row, m - 2, step),
            ladder_secant(ladder_row, m - 3, step),
        );
    }
    let before = ladder_secant(ladder_row, j - 1, step);
    let after = ladder_secant(ladder_row, j, step);
    if before <= 0.0 || after <= 0.0 {
        return 0.0;
    }
    2.0 / (1.0 / before + 1.0 / after)
}

/// Interpolate one row of the tabulated response-quantile ladder
/// `Q[j] = h⁻¹(z_j | x)` at an arbitrary latent value `z`. The ladder nodes are
/// the fixed even grid from `transformation_normal_band_z_nodes`.
///
/// Two things this must not do, both of which it used to (gam#2600):
///
/// * **Clamp past the ends.** A requested level with `|Φ⁻¹(p)| > z_max` used to
///   return the outermost tabulated quantile, so every band beyond 99.994 % was
///   the same interval. Since the CTN transform is affine past the fitted
///   support, `h⁻¹` is affine in `z` out there and the end slope IS the exact
///   continuation; where `h⁻¹(±z_max)` still falls inside the support it is a
///   first-order one, which is strictly better than a constant.
/// * **Interpolate a curved quantile function with a chord.** `Q` is `h⁻¹`
///   sampled every `2·z_max/(m−1) = 0.125` in the latent, and `h⁻¹` is as curved
///   as the response is skewed — for a lognormal response `d²y/dz² = y`, so a
///   chord carries `O(Δz²·y/8) ≈ 2e-3·y`, i.e. two parts in a thousand of the
///   reported limit. The shape-preserving cubic through the same nodes is third
///   order and, being monotone, keeps the band ordered.
fn ladder_quantile(ladder_row: ndarray::ArrayView1<'_, f64>, z: f64) -> f64 {
    let m = ladder_row.len();
    assert_eq!(
        m, TRANSFORMATION_NORMAL_BAND_Z_NODES,
        "quantile ladder row must be tabulated on the fixed even z grid"
    );
    let z_max = TRANSFORMATION_NORMAL_BAND_Z_MAX;
    let step = 2.0 * z_max / ((m - 1) as f64);
    let t = (z + z_max) / step;
    if t <= 0.0 {
        return ladder_row[0] + (z + z_max) * ladder_slope(ladder_row, 0, step);
    }
    if t >= (m - 1) as f64 {
        return ladder_row[m - 1] + (z - z_max) * ladder_slope(ladder_row, m - 1, step);
    }
    let j = t.floor() as usize;
    let frac = t - j as f64;
    let (q0, q1) = (ladder_row[j], ladder_row[j + 1]);
    let secant = ladder_secant(ladder_row, j, step);
    if !(secant > 0.0) {
        // A cell the transform could not separate: nothing to shape-preserve.
        return q0 + frac * (q1 - q0);
    }
    let m0 = ladder_slope(ladder_row, j, step);
    let m1 = ladder_slope(ladder_row, j + 1, step);
    let (t2, t3) = (frac * frac, frac * frac * frac);
    (2.0 * t3 - 3.0 * t2 + 1.0) * q0
        + (t3 - 2.0 * t2 + frac) * step * m0
        + (-2.0 * t3 + 3.0 * t2) * q1
        + (t3 - t2) * step * m1
}

impl PredictionTransform for TransformationNormalPredictor {

    fn response_family(&self) -> ResponseFamily {
        // Only the *latent* `h(y)` is Gaussian. The generic family observation
        // band must never be built from this (its σ lives in latent units);
        // the predictor supplies its own response-scale band from the
        // quantile ladder in `predict_posterior_mean`.
        ResponseFamily::Gaussian
    }

    fn response_jacobian_rows(&self, pass: PredictPass) -> ResponseInterval {
        match pass {
            // `response` is the identity here (the offset already carries the
            // response-scale conditional mean), so there is no link to
            // transform or delta-propagate through in either pass: an η
            // interval already IS the response interval.
            PredictPass::FullUncertainty | PredictPass::PosteriorMean => {
                ResponseInterval::IdentityEta
            }
        }
    }
    fn point_state(&self, input: &PredictInput) -> Result<LinearState, EstimationError> {
        // The offset carries the precomputed response-scale conditional mean
        // `E[Y|x]`. No covariance-propagated SE exists for this quantity (see
        // the struct-level uncertainty contract), so the SEs are `None` —
        // reporting zero would claim certainty the posterior does not have.
        let h = input.offset.clone();
        Ok(LinearState {
            eta: h.clone(),
            mean: h,
            eta_se: None,
            mean_se: None,
            covariance_source: InferenceCovarianceMode::Conditional,
        })
    }

    fn response(&self, eta: &Array1<f64>) -> Result<Array1<f64>, EstimationError> {
        Ok(eta.clone())
    }

    fn bounds(&self) -> ResponseBounds {
        ResponseBounds::UNBOUNDED
    }

}

impl PredictableModel for TransformationNormalPredictor {
    fn predict_plugin_response(
        &self,
        input: &PredictInput,
    ) -> Result<PredictResult, EstimationError> {
        predict_plugin_response_generic(self, input)
    }

    fn predict_with_uncertainty(
        &self,
        input: &PredictInput,
    ) -> Result<PredictionWithSE, EstimationError> {
        // The CTM predictor reports no covariance-derived SEs on the point path;
        // it passes through the precomputed E[Y|x] offset as eta and mean.
        let h = input.offset.clone();
        Ok(PredictionWithSE {
            eta: h.clone(),
            mean: h,
            eta_se: None,
            mean_se: None,
        })
    }

    fn predict_full_uncertainty(
        &self,
        input: &PredictInput,
        fit: &UnifiedFitResult,
        options: &PredictUncertaintyOptions,
    ) -> Result<PredictUncertaintyResult, EstimationError> {
        if !fit.log_likelihood.is_finite() {
            return Err(EstimationError::InvalidInput(
                "transformation-normal predict received a fit with a non-finite log-likelihood"
                    .to_string(),
            ));
        }
        Err(EstimationError::InvalidInput(format!(
            "transformation-normal models cannot report coefficient-uncertainty intervals \
             (level {} requested for {} rows): propagating the coefficient covariance through \
             the inverse transform h⁻¹ requires the I-spline basis Jacobian, which is not part \
             of the persisted quantile grid. Use predict_posterior_mean for the point E[Y|x] \
             and its response-scale observation (predictive) interval.",
            options.confidence_level,
            input.offset.len(),
        )))
    }

    fn predict_posterior_mean(
        &self,
        input: &PredictInput,
        fit: &UnifiedFitResult,
        options: &PosteriorMeanOptions,
    ) -> Result<PredictPosteriorMeanResult, EstimationError> {
        // The posterior mean is read entirely off the persisted quantile
        // ladder; the fit contributes no coefficient state here, but a
        // non-finite fitted log-likelihood marks a corrupted payload.
        if !fit.log_likelihood.is_finite() {
            return Err(EstimationError::InvalidInput(
                "transformation-normal predict received a fit with a non-finite log-likelihood"
                    .to_string(),
            ));
        }
        let h = input.offset.clone();
        let n = h.len();
        let mut result = PredictPosteriorMeanResult {
            eta: h.clone(),
            // The result struct requires an SE array; epistemic uncertainty is
            // unavailable (see the struct-level contract), so no credible
            // bounds or mean SE are emitted below and this array is inert.
            eta_standard_error: Array1::zeros(n),
            mean: h,
            mean_standard_error: None,
            mean_lower: None,
            mean_upper: None,
            observation_lower: None,
            observation_upper: None,
            point_covariance_source: InferenceCovarianceMode::Conditional,
            uncertainty_covariance_source: None,
        };
        if options.include_observation_interval
            && let Some(level) = options.confidence_level
        {
            let z = crate::interval_policy::validated_central_z(level)?;
            let ladder = input.auxiliary_matrix.as_ref().ok_or_else(|| {
                EstimationError::InvalidInput(
                    "transformation-normal prediction input is missing the response-scale \
                     quantile ladder (auxiliary_matrix)"
                        .to_string(),
                )
            })?;
            if ladder.nrows() != n || ladder.ncols() != TRANSFORMATION_NORMAL_BAND_Z_NODES {
                return Err(EstimationError::InvalidInput(format!(
                    "transformation-normal quantile ladder shape mismatch: expected {}x{}, got {}x{}",
                    n,
                    TRANSFORMATION_NORMAL_BAND_Z_NODES,
                    ladder.nrows(),
                    ladder.ncols()
                )));
            }
            // Equal-tailed response-scale predictive band: the p-quantile of
            // `Y|x = h⁻¹(Z|x)` is `h⁻¹(Φ⁻¹(p)|x)`, interpolated from the
            // tabulated ladder. `h⁻¹` is monotone increasing, so the band is
            // ordered by construction.
            let lower = Array1::from_shape_fn(n, |i| ladder_quantile(ladder.row(i), -z));
            let upper = Array1::from_shape_fn(n, |i| ladder_quantile(ladder.row(i), z));
            result.observation_lower = Some(lower);
            result.observation_upper = Some(upper);
        }
        Ok(result)
    }

    fn n_blocks(&self) -> usize {
        1
    }
    fn block_roles(&self) -> Vec<BlockRole> {
        vec![BlockRole::Mean]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    /// The ladder a lognormal response actually produces: `h(y) = ln y`, so
    /// `h⁻¹(z) = exp(z)` — smooth, strongly curved, and known in closed form, so
    /// every deviation below is interpolation error and nothing else.
    fn exp_ladder() -> Array1<f64> {
        let m = TRANSFORMATION_NORMAL_BAND_Z_NODES;
        let z_max = TRANSFORMATION_NORMAL_BAND_Z_MAX;
        Array1::from_shape_fn(m, |j| {
            (-z_max + 2.0 * z_max * (j as f64) / ((m - 1) as f64)).exp()
        })
    }

    #[test]
    fn band_ladder_does_not_clamp_past_its_own_ends_2600() {
        // A requested level past `z_max` used to return the outermost tabulated
        // quantile, so every band beyond 99.994 % was the same interval.
        let ladder = exp_ladder();
        let z_max = TRANSFORMATION_NORMAL_BAND_Z_MAX;
        let end = ladder[ladder.len() - 1];
        let mut previous = end;
        for &z in &[4.5_f64, 5.0, 6.0] {
            let value = ladder_quantile(ladder.view(), z);
            assert!(
                value > previous,
                "the ladder saturated past its end: q({z}) = {value} <= {previous}"
            );
            previous = value;
        }
        let start = ladder[0];
        let mut previous = start;
        for &z in &[-4.5_f64, -5.0, -6.0] {
            let value = ladder_quantile(ladder.view(), z);
            assert!(
                value < previous,
                "the ladder saturated past its start: q({z}) = {value} >= {previous}"
            );
            previous = value;
        }
        // The continuation is the ladder's own end slope, exactly.
        let step = 2.0 * z_max / ((ladder.len() - 1) as f64);
        let slope = ladder_slope(ladder.view(), ladder.len() - 1, step);
        let far = ladder_quantile(ladder.view(), z_max + 1.5);
        assert!(
            (far - (end + 1.5 * slope)).abs() < 1e-9,
            "the exterior continuation is not affine at the end slope: {far} vs {}",
            end + 1.5 * slope
        );
    }

    #[test]
    fn band_ladder_interpolation_is_shape_preserving_and_beats_the_chord_2600() {
        let ladder = exp_ladder();
        let m = ladder.len();
        let z_max = TRANSFORMATION_NORMAL_BAND_Z_MAX;
        let step = 2.0 * z_max / ((m - 1) as f64);
        let (mut shaped, mut chord) = (0.0_f64, 0.0_f64);
        let mut previous = f64::NEG_INFINITY;
        for k in 0..=4000 {
            let z = -z_max + 2.0 * z_max * (k as f64) / 4000.0;
            let truth = z.exp();
            let value = ladder_quantile(ladder.view(), z);
            // Monotonicity: the interpolated quantile function IS the band, so
            // an overshoot here is a lower limit above its own upper limit.
            assert!(
                value > previous,
                "the interpolated ladder is not monotone at z={z}: {value} <= {previous}"
            );
            previous = value;
            shaped = shaped.max((value - truth).abs() / truth);
            let t = (z + z_max) / step;
            let j = (t.floor() as usize).min(m - 2);
            let frac = t - (j as f64);
            let linear = ladder[j] + frac * (ladder[j + 1] - ladder[j]);
            chord = chord.max((linear - truth).abs() / truth);
        }
        eprintln!(
            "#2600 ladder: max relative band error  shape-preserving={shaped:.3e}  chord={chord:.3e}"
        );
        assert!(
            shaped * 10.0 < chord,
            "the shape-preserving ladder is not decisively tighter than the chord it replaces: \
             {shaped:.6e} vs {chord:.6e}"
        );
        assert!(
            shaped < 1.0e-3,
            "the reported band carries {shaped:.6e} relative interpolation error"
        );
    }

    #[test]
    fn band_ladder_reproduces_its_own_nodes_exactly_2600() {
        let ladder = exp_ladder();
        let m = ladder.len();
        let z_max = TRANSFORMATION_NORMAL_BAND_Z_MAX;
        for j in 0..m {
            let z = -z_max + 2.0 * z_max * (j as f64) / ((m - 1) as f64);
            let value = ladder_quantile(ladder.view(), z);
            assert!(
                (value - ladder[j]).abs() <= 1e-12 * ladder[j].abs().max(1.0),
                "node {j} is not reproduced: {value} vs {}",
                ladder[j]
            );
        }
    }
}
