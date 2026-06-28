use super::*;

/// Predictor for transformation-normal (CTM) models.
///
/// The response-scale conditional mean `E[Y|x]` is precomputed in
/// `build_predict_input_for_model` (issue #1612) and stored in the PredictInput
/// offset. `E[Y|x] = E_{Z~N(0,1)}[h⁻¹(Z|x)]` is a function of the covariates
/// alone, so prediction is covariate-only and does not require the outcome
/// column. This predictor passes the precomputed value through unchanged as both
/// the linear predictor and the mean: eta = mean = E[Y|x].
pub struct TransformationNormalPredictor {
    pub covariance: Option<Array2<f64>>,
}

impl PredictionTransform for TransformationNormalPredictor {
    fn point_state(&self, input: &PredictInput) -> Result<LinearState, EstimationError> {
        // The offset carries the precomputed response-scale conditional mean
        // `E[Y|x]`; the standard error is zero (the η endpoints coincide with the
        // offset), so the engine's identity-η path reproduces
        // `eta = mean = bounds = E[Y|x]`. The zero SEs make the posterior-mean
        // bounds collapse onto the point; whether bounds are produced at all is
        // gated upstream by fit covariance.
        let h = input.offset.clone();
        let zeros = Array1::zeros(h.len());
        Ok(LinearState {
            eta: h.clone(),
            mean: h,
            eta_se: Some(zeros.clone()),
            mean_se: Some(zeros),
            covariance_corrected_used: false,
        })
    }

    fn linear_state(
        &self,
        input: &PredictInput,
        fit: &UnifiedFitResult,
        pass: PredictPass,
        covariance_mode: InferenceCovarianceMode,
    ) -> Result<LinearState, EstimationError> {
        // Both passes share the same identity state (η = mean = h, zero SE). The
        // provenance flag reflects whether the requested covariance mode would
        // resolve to the smoothing-corrected covariance: corrected is used iff
        // the caller asked for it and the fit carries it.
        let mut state = self.point_state(input)?;
        if matches!(pass, PredictPass::FullUncertainty) {
            let corrected_requested =
                !matches!(covariance_mode, InferenceCovarianceMode::Conditional);
            state.covariance_corrected_used =
                corrected_requested && fit.covariance_corrected.is_some();
        }
        Ok(state)
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
        // The transformed scale `h(y)` is modelled as Gaussian; the identity-η
        // observation band uses the residual SD.
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
        input: &PredictInput,
        fit: &UnifiedFitResult,
        options: &PredictUncertaintyOptions,
    ) -> Result<PredictUncertaintyResult, EstimationError> {
        predict_full_uncertainty_generic(self, input, fit, options)
    }

    fn predict_posterior_mean(
        &self,
        input: &PredictInput,
        fit: &UnifiedFitResult,
        options: &PosteriorMeanOptions,
    ) -> Result<PredictPosteriorMeanResult, EstimationError> {
        // Bounds are only defined once a fit covariance exists; the SE is zero,
        // so the engine's identity-η path collapses the bounds onto `h`.
        let has_fit_covariance =
            fit.covariance_corrected.is_some() || fit.covariance_conditional.is_some();
        let bound_level = has_fit_covariance
            .then_some(options.confidence_level)
            .flatten();
        let bounded_options = PosteriorMeanOptions {
            confidence_level: bound_level,
            ..*options
        };
        predict_posterior_mean_generic(self, input, fit, &bounded_options)
    }

    fn n_blocks(&self) -> usize {
        1
    }
    fn block_roles(&self) -> Vec<BlockRole> {
        vec![BlockRole::Mean]
    }
}
