use super::*;

fn bernoulli_eta_standard_error_from_covariance(
    predictor: &BernoulliMarginalSlopePredictor,
    input: &PredictInput,
    covariance: &Array2<f64>,
) -> Result<Array1<f64>, EstimationError> {
    let backend = PredictionCovarianceBackend::from_dense(covariance.view());
    bernoulli_eta_standard_error_from_backend(predictor, input, &backend)
}

fn bernoulli_eta_standard_error_from_backend(
    predictor: &BernoulliMarginalSlopePredictor,
    input: &PredictInput,
    backend: &PredictionCovarianceBackend<'_>,
) -> Result<Array1<f64>, EstimationError> {
    let theta = predictor.theta();
    linear_predictor_se_from_backend(backend, input.design.nrows(), |rows| {
        let chunk_input = slice_predict_input(input, rows).map_err(|e| e.to_string())?;
        let (_, grad) = predictor
            .final_eta_and_gradient_from_theta(&chunk_input, &theta, true)
            .map_err(|e| e.to_string())?;
        let grad = grad.ok_or_else(|| {
            "bernoulli marginal-slope analytic predictor gradient was not produced".to_string()
        })?;
        Ok(vec![grad])
    })
}

impl PredictionTransform for BernoulliMarginalSlopePredictor {
    fn point_state(&self, input: &PredictInput) -> Result<LinearState, EstimationError> {
        let eta = self.final_eta_from_theta(input, &self.theta())?;
        let mean = self.mean_from_eta(&eta)?;
        let (eta_se, mean_se) = if let Some(covariance) = self.covariance.as_ref() {
            let theta = self.theta();
            if covariance.nrows() != theta.len() || covariance.ncols() != theta.len() {
                return Err(EstimationError::InvalidInput(format!(
                    "bernoulli marginal-slope covariance dimension mismatch: expected {}x{}, got {}x{}",
                    theta.len(),
                    theta.len(),
                    covariance.nrows(),
                    covariance.ncols()
                )));
            }
            let eta_se = bernoulli_eta_standard_error_from_covariance(self, input, covariance)?;
            let mean_se = eta_se.clone() * self.mean_derivative_from_eta(&eta)?;
            (Some(eta_se), Some(mean_se))
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
        let eta = self.final_eta_from_theta(input, &self.theta())?;
        match pass {
            PredictPass::FullUncertainty => {
                // Select the covariance the caller requested (conditional vs.
                // smoothing-corrected) instead of always using the conditional
                // backend, and report which was used.
                let (backend, covariance_corrected_used) = fit.select_uncertainty_backend(
                    self.theta().len(),
                    covariance_mode,
                    "bernoulli marginal-slope",
                )?;
                let eta_se = bernoulli_eta_standard_error_from_backend(self, input, &backend)?;
                let mean = self.mean_from_eta(&eta)?;
                let mean_se = eta_se.clone() * self.mean_derivative_from_eta(&eta)?;
                Ok(LinearState {
                    eta,
                    mean,
                    eta_se: Some(eta_se),
                    mean_se: Some(mean_se),
                    covariance_corrected_used,
                })
            }
            PredictPass::PosteriorMean => {
                // Posterior-mean integration uses the conditional posterior.
                let backend = require_posterior_mean_backend(
                    fit,
                    self.covariance.as_ref(),
                    self.theta().len(),
                    "bernoulli marginal-slope posterior mean",
                )?;
                let eta_se = bernoulli_eta_standard_error_from_backend(self, input, &backend)?;
                let strategy = strategy_for_family(self.likelihood_family(), Some(&self.base_link));
                let quadctx = gam_solve::quadrature::QuadratureContext::new();
                let mean = Array1::from_iter(
                    eta.iter()
                        .zip(eta_se.iter())
                        .map(|(&eta_i, &se)| strategy.posterior_mean(&quadctx, eta_i, se))
                        .collect::<Result<Vec<_>, _>>()?,
                );
                Ok(LinearState {
                    eta,
                    mean,
                    eta_se: Some(eta_se),
                    mean_se: None,
                    covariance_corrected_used: false,
                })
            }
        }
    }

    fn response(&self, eta: &Array1<f64>) -> Result<Array1<f64>, EstimationError> {
        self.mean_from_eta(eta)
    }

    fn response_jacobian_rows(&self, _: PredictPass) -> ResponseInterval {
        ResponseInterval::TransformEta
    }

    fn bounds(&self) -> ResponseBounds {
        ResponseBounds::for_family(&self.likelihood_family().response)
    }

    fn response_family(&self) -> ResponseFamily {
        self.likelihood_family().response.clone()
    }
}

impl PredictableModel for BernoulliMarginalSlopePredictor {
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
        predict_with_uncertainty_generic(self, input)
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
        2 + usize::from(self.beta_score_warp.is_some()) + usize::from(self.beta_link_dev.is_some())
    }

    fn block_roles(&self) -> Vec<BlockRole> {
        let mut roles = vec![BlockRole::Location, BlockRole::Scale];
        if self.beta_score_warp.is_some() {
            roles.push(BlockRole::Mean);
        }
        if self.beta_link_dev.is_some() {
            roles.push(BlockRole::LinkWiggle);
        }
        roles
    }
}
