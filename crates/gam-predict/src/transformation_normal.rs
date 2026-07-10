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

/// Interpolate one row of the tabulated response-quantile ladder
/// `Q[j] = h⁻¹(z_j | x)` at an arbitrary latent value `z`. The ladder nodes are
/// the fixed even grid from `transformation_normal_band_z_nodes`; values beyond
/// the ladder clamp to the outermost tabulated quantile (the same endpoint
/// clamping the grid inversion itself applies at the response support).
fn ladder_quantile(ladder_row: ndarray::ArrayView1<'_, f64>, z: f64) -> f64 {
    let m = ladder_row.len();
    assert_eq!(
        m, TRANSFORMATION_NORMAL_BAND_Z_NODES,
        "quantile ladder row must be tabulated on the fixed even z grid"
    );
    let z_max = TRANSFORMATION_NORMAL_BAND_Z_MAX;
    let t = ((z + z_max) / (2.0 * z_max)) * ((m - 1) as f64);
    if t <= 0.0 {
        return ladder_row[0];
    }
    if t >= (m - 1) as f64 {
        return ladder_row[m - 1];
    }
    let j = t.floor() as usize;
    let frac = t - j as f64;
    ladder_row[j] + frac * (ladder_row[j + 1] - ladder_row[j])
}

impl PredictionTransform for TransformationNormalPredictor {
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
            covariance_corrected_used: false,
        })
    }

    fn response(&self, eta: &Array1<f64>) -> Result<Array1<f64>, EstimationError> {
        Ok(eta.clone())
    }

    fn response_jacobian_rows(&self, _: PredictPass) -> ResponseInterval {
        ResponseInterval::IdentityEta
    }

    fn bounds(&self) -> ResponseBounds {
        ResponseBounds::UNBOUNDED
    }

    fn response_family(&self) -> ResponseFamily {
        // Only the *latent* `h(y)` is Gaussian. The generic family observation
        // band must never be built from this (its σ lives in latent units);
        // the predictor supplies its own response-scale band from the
        // quantile ladder in `predict_posterior_mean`.
        ResponseFamily::Gaussian
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
        _input: &PredictInput,
        _fit: &UnifiedFitResult,
        _options: &PredictUncertaintyOptions,
    ) -> Result<PredictUncertaintyResult, EstimationError> {
        Err(EstimationError::InvalidInput(
            "transformation-normal models cannot report coefficient-uncertainty intervals: \
             propagating the coefficient covariance through the inverse transform h⁻¹ requires \
             the I-spline basis Jacobian, which is not part of the persisted quantile grid. \
             Use predict_posterior_mean for the point E[Y|x] and its response-scale \
             observation (predictive) interval."
                .to_string(),
        ))
    }

    fn predict_posterior_mean(
        &self,
        input: &PredictInput,
        _fit: &UnifiedFitResult,
        options: &PosteriorMeanOptions,
    ) -> Result<PredictPosteriorMeanResult, EstimationError> {
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
