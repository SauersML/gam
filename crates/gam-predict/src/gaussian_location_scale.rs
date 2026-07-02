use super::*;

/// Gaussian location-scale predictor: two blocks (mean + log-sigma).
///
/// Predicts `mean = X_mu @ beta_mu` (identity link on mean) and
/// `sigma = sigma_floor + exp(X_noise @ beta_noise + offset_noise)`.
///
/// `sigma_floor` is the scaled-response σ floor (`LOGB_SIGMA_FLOOR`) and
/// `response_scale` maps the fitted scaled-response σ back to the original
/// response units.
pub struct GaussianLocationScalePredictor {
    pub beta_mu: Array1<f64>,
    pub beta_noise: Array1<f64>,
    pub sigma_floor: f64,
    pub response_scale: f64,
    pub covariance: Option<Array2<f64>>,
    pub link_wiggle: Option<SavedLinkWiggleRuntime>,
}

impl GaussianLocationScalePredictor {
    /// Compute σ on the scaled-response likelihood scale and then restore it to
    /// raw response units. Both the positive floor and the exp(η) component are
    /// scale parameters, so both must be multiplied by the saved response scale.
    fn compute_sigma(
        &self,
        design_noise: &DesignMatrix,
        offset_noise: Option<&Array1<f64>>,
    ) -> Result<Array1<f64>, EstimationError> {
        let mut eta_noise = design_noise.dot(&self.beta_noise);
        if let Some(offset_noise) = offset_noise {
            if offset_noise.len() != eta_noise.len() {
                return Err(EstimationError::InvalidInput(format!(
                    "gaussian location-scale noise offset length mismatch: expected {}, got {}",
                    eta_noise.len(),
                    offset_noise.len()
                )));
            }
            eta_noise += offset_noise;
        }
        let floor = self.sigma_floor;
        let response_scale = self.response_scale;
        Ok(eta_noise.mapv(|eta| {
            response_scale
                * gam::families::sigma_link::logb_sigma_from_eta_with_floor_scalar(floor, eta)
        }))
    }

    fn eta_standard_error_from_backend(
        &self,
        input: &PredictInput,
        backend: &PredictionCovarianceBackend<'_>,
        eta_len: usize,
        p_mu: usize,
        p_sigma: usize,
        p_w: usize,
    ) -> Result<Array1<f64>, EstimationError> {
        let p_total = p_mu + p_sigma + p_w;
        if backend.nrows() != p_total {
            return Err(EstimationError::InvalidInput(format!(
                "gaussian location-scale covariance mismatch: expected parameter dimension {}, got {}",
                p_total,
                backend.nrows()
            )));
        }
        if let Some(runtime) = self.link_wiggle.as_ref() {
            let eta_base = input.design.dot(&self.beta_mu) + &input.offset;
            link_wiggle_eta_se_from_backend(
                backend,
                eta_len,
                &input.design,
                &eta_base,
                runtime,
                LinkWiggleGradientLayout {
                    p_main: p_mu,
                    p_total,
                    wiggle_col_start: p_mu + p_sigma,
                },
                "gaussian location-scale covariance mismatch",
            )
        } else {
            padded_design_standard_errors_from_backend(
                &input.design,
                backend,
                0,
                p_sigma + p_w,
                "gaussian location-scale posterior mean",
            )
        }
    }
}

impl GaussianLocationScalePredictor {
    /// Identity-link plug-in: η = X_μ β_μ (+ wiggle), mean == η.
    fn plugin_eta(&self, input: &PredictInput) -> Result<Array1<f64>, EstimationError> {
        let eta_base = input.design.dot(&self.beta_mu) + &input.offset;
        if let Some(runtime) = self.link_wiggle.as_ref() {
            runtime.apply(&eta_base).map_err(EstimationError::from)
        } else {
            Ok(eta_base)
        }
    }
}

impl PredictionTransform for GaussianLocationScalePredictor {
    fn point_state(&self, input: &PredictInput) -> Result<LinearState, EstimationError> {
        let eta_base = input.design.dot(&self.beta_mu) + &input.offset;
        let eta = if let Some(runtime) = self.link_wiggle.as_ref() {
            runtime.apply(&eta_base).map_err(EstimationError::from)?
        } else {
            eta_base
        };
        // Gaussian identity link: mean == eta.
        let mean = eta.clone();
        let (eta_se, mean_se) = if let Some(covariance) = self.covariance.as_ref() {
            let p_mu = self.beta_mu.len();
            let p_sigma = self.beta_noise.len();
            let p_w = self.link_wiggle.as_ref().map_or(0, |w| w.beta.len());
            let backend = PredictionCovarianceBackend::from_dense(covariance.view());
            let eta_se = self.eta_standard_error_from_backend(
                input,
                &backend,
                eta.len(),
                p_mu,
                p_sigma,
                p_w,
            )?;
            (Some(eta_se.clone()), Some(eta_se))
        } else {
            (None, None)
        };
        Ok(LinearState {
            eta,
            mean,
            eta_se,
            mean_se,
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
        // Both fit-backed passes share the identity-link state: mean == eta and
        // the mean SE equals the η SE, computed from the fit-backed backend.
        let eta = self.plugin_eta(input)?;
        let p_mu = self.beta_mu.len();
        let p_sigma = self.beta_noise.len();
        let p_w = self.link_wiggle.as_ref().map_or(0, |w| w.beta.len());
        let p_total = p_mu + p_sigma + p_w;
        // Full uncertainty honors the requested covariance mode; posterior-mean
        // integration uses the conditional posterior.
        let (backend, covariance_corrected_used) = match pass {
            PredictPass::FullUncertainty => {
                fit.select_uncertainty_backend(p_total, covariance_mode, "gaussian location-scale")?
            }
            PredictPass::PosteriorMean => (
                require_posterior_mean_backend(
                    fit,
                    self.covariance.as_ref(),
                    p_total,
                    "gaussian location-scale posterior mean",
                )?,
                false,
            ),
        };
        let eta_se =
            self.eta_standard_error_from_backend(input, &backend, eta.len(), p_mu, p_sigma, p_w)?;
        let mean = eta.clone();
        Ok(LinearState {
            eta,
            mean,
            eta_se: Some(eta_se.clone()),
            mean_se: Some(eta_se),
            covariance_corrected_used,
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
        ResponseFamily::Gaussian
    }

    fn observation_noise(
        &self,
        input: &PredictInput,
    ) -> Result<Option<Array1<f64>>, EstimationError> {
        let design_noise = input.design_noise.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput(
                "Gaussian location-scale prediction requires noise design matrix".to_string(),
            )
        })?;
        self.compute_sigma(design_noise, input.offset_noise.as_ref())
            .map(Some)
    }
}

impl PredictableModel for GaussianLocationScalePredictor {
    fn predict_plugin_response(
        &self,
        input: &PredictInput,
    ) -> Result<PredictResult, EstimationError> {
        let eta = self.plugin_eta(input)?;
        let mean = eta.clone();
        Ok(PredictResult { eta, mean })
    }

    fn predict_with_uncertainty(
        &self,
        input: &PredictInput,
    ) -> Result<PredictionWithSE, EstimationError> {
        predict_with_uncertainty_generic(self, input)
    }

    fn predict_noise_scale(
        &self,
        input: &PredictInput,
    ) -> Result<Option<Array1<f64>>, EstimationError> {
        let design_noise = input.design_noise.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput(
                "Gaussian location-scale prediction requires noise design matrix".to_string(),
            )
        })?;
        self.compute_sigma(design_noise, input.offset_noise.as_ref())
            .map(Some)
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
        predict_posterior_mean_generic(self, input, fit, options)
    }

    fn n_blocks(&self) -> usize {
        if self.link_wiggle.is_some() { 3 } else { 2 }
    }

    fn block_roles(&self) -> Vec<BlockRole> {
        if self.link_wiggle.is_some() {
            vec![BlockRole::Location, BlockRole::Scale, BlockRole::LinkWiggle]
        } else {
            vec![BlockRole::Location, BlockRole::Scale]
        }
    }
}
