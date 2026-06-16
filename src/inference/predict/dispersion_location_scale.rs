use super::*;

pub struct DispersionLocationScalePredictor {
    pub beta_mu: Array1<f64>,
    pub beta_noise: Array1<f64>,
    /// Persisted location-scale likelihood: its `response` selects the
    /// mean–variance law and its link is the mean inverse link.
    pub likelihood: LikelihoodSpec,
    /// Resolved mean inverse link (log for NB/Gamma/Tweedie, logit for Beta).
    /// `None` falls back to the link carried by `likelihood`.
    pub inverse_link: Option<InverseLink>,
    pub covariance: Option<Array2<f64>>,
}

impl DispersionLocationScalePredictor {
    fn strategy(&self) -> ResolvedFamilyStrategy {
        strategy_for_family(self.likelihood.clone(), self.inverse_link.as_ref())
    }

    /// Mean linear predictor `eta_mu = X_mu @ beta_mu + offset`.
    fn eta_mean(&self, input: &PredictInput) -> Array1<f64> {
        input.design.dot(&self.beta_mu) + &input.offset
    }

    /// Log-precision linear predictor `eta_d = X_noise @ beta_noise +
    /// offset_noise`, mapped to `precision = exp(eta_d)`.
    fn precision(&self, input: &PredictInput) -> Result<Array1<f64>, EstimationError> {
        let design_noise = input.design_noise.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput(
                "dispersion location-scale prediction requires noise design matrix".to_string(),
            )
        })?;
        let mut eta_d = design_noise.dot(&self.beta_noise);
        if let Some(offset_noise) = input.offset_noise.as_ref() {
            if offset_noise.len() != eta_d.len() {
                return Err(EstimationError::InvalidInput(format!(
                    "dispersion location-scale noise offset length mismatch: expected {}, got {}",
                    eta_d.len(),
                    offset_noise.len()
                )));
            }
            eta_d += offset_noise;
        }
        // `exp(eta_d)` is the precision; floor it away from zero so the
        // mean–variance law below never divides by zero on underflow.
        Ok(eta_d.mapv(|v| v.exp().max(f64::MIN_POSITIVE)))
    }

    /// Observation-scale predictive standard deviation `sqrt(Var(y))` from the
    /// predicted mean and precision, per the family's mean–variance law:
    ///   NegativeBinomial  Var = mu + mu^2 / theta,   theta     = exp(eta_d)
    ///   Gamma             Var = mu^2 / nu,            nu        = exp(eta_d)
    ///   Beta              Var = mu (1 - mu)/(1 + phi),phi       = exp(eta_d)
    ///   Tweedie(p)        Var = phi mu^p,             1/phi     = exp(eta_d)
    fn noise_sd(&self, input: &PredictInput) -> Result<Array1<f64>, EstimationError> {
        let eta_mu = self.eta_mean(input);
        let mean = self.strategy().inverse_link_array(eta_mu.view())?;
        let precision = self.precision(input)?;
        if mean.len() != precision.len() {
            return Err(EstimationError::InvalidInput(format!(
                "dispersion location-scale mean/precision length mismatch: {} vs {}",
                mean.len(),
                precision.len()
            )));
        }
        let response = &self.likelihood.response;
        let variance = Array1::from_shape_fn(mean.len(), |i| {
            let mu = mean[i];
            let prec = precision[i];
            let var = match response {
                ResponseFamily::NegativeBinomial { .. } => mu + mu * mu / prec,
                ResponseFamily::Gamma => mu * mu / prec,
                ResponseFamily::Beta { .. } => mu * (1.0 - mu) / (1.0 + prec),
                ResponseFamily::Tweedie { p } => mu.powf(*p) / prec,
                // The dispersion location-scale class only routes the four
                // overdispersion mean families above; any other response is a
                // classification error upstream. Report a Gaussian-style scalar
                // variance (`1/precision`) rather than panic so a corrupt model
                // degrades gracefully instead of aborting prediction.
                _ => 1.0 / prec,
            };
            var.max(0.0)
        });
        Ok(variance.mapv(f64::sqrt))
    }

    /// Mean-scale point + (covariance-derived) η and mean standard errors. The η
    /// SE is the mean block's joint-covariance slice (the noise/log-precision
    /// columns do not enter the mean linear predictor); the mean SE is the
    /// delta-method propagation through the inverse link.
    fn state_from_backend(
        &self,
        input: &PredictInput,
        backend: &PredictionCovarianceBackend<'_>,
    ) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>), EstimationError> {
        let eta = self.eta_mean(input);
        let strategy = self.strategy();
        let (mean, dmu_deta) = inverse_link_mean_and_d1(&strategy, eta.view())?;
        // Mean block leads the coefficient layout `[mean | noise]`, so its
        // covariance slice needs no leading pad and is followed by the `p_d`
        // noise columns the mean linear predictor does not touch.
        let p_d = self.beta_noise.len();
        let eta_se = padded_design_standard_errors_from_backend(
            &input.design,
            backend,
            0,
            p_d,
            "dispersion location-scale",
        )?;
        let mean_se = delta_method_mean_se_from_d1(&dmu_deta, &eta_se);
        Ok((eta, mean, eta_se, mean_se))
    }
}

impl PredictionTransform for DispersionLocationScalePredictor {
    fn point_state(&self, input: &PredictInput) -> Result<LinearState, EstimationError> {
        let eta = self.eta_mean(input);
        let mean = self.strategy().inverse_link_array(eta.view())?;
        let (eta_se, mean_se) = if let Some(covariance) = self.covariance.as_ref() {
            let backend = PredictionCovarianceBackend::from_dense(covariance.view());
            let (_, _, eta_se, mean_se) = self.state_from_backend(input, &backend)?;
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
        let p_total = self.beta_mu.len() + self.beta_noise.len();
        let (backend, covariance_corrected_used) = match pass {
            PredictPass::FullUncertainty => fit.select_uncertainty_backend(
                p_total,
                covariance_mode,
                "dispersion location-scale",
            )?,
            PredictPass::PosteriorMean => (
                require_posterior_mean_backend(
                    fit,
                    self.covariance.as_ref(),
                    p_total,
                    "dispersion location-scale posterior mean",
                )?,
                false,
            ),
        };
        let (eta, plugin_mean, eta_se, mean_se) = self.state_from_backend(input, &backend)?;
        let mean = match pass {
            // Plug-in mean is correct for the symmetric-delta full-uncertainty
            // report (the point is the inverse link of the conditional η).
            PredictPass::FullUncertainty => plugin_mean,
            // The curved inverse link makes `E[g^{-1}(η)] ≠ g^{-1}(E[η])`, so the
            // posterior-mean point integrates the inverse link over the
            // conditional η posterior `η ~ N(eta, eta_se²)`.
            PredictPass::PosteriorMean => {
                let strategy = self.strategy();
                let quadctx = crate::quadrature::QuadratureContext::new();
                eta.iter()
                    .zip(eta_se.iter())
                    .map(|(&e, &se)| strategy.posterior_mean(&quadctx, e, se))
                    .collect::<Result<Array1<f64>, _>>()?
            }
        };
        Ok(LinearState {
            eta,
            mean,
            eta_se: Some(eta_se),
            mean_se: Some(mean_se),
            covariance_corrected_used,
        })
    }

    fn response(&self, eta: &Array1<f64>) -> Result<Array1<f64>, EstimationError> {
        self.strategy().inverse_link_array(eta.view())
    }

    fn response_jacobian_rows(&self, pass: PredictPass) -> ResponseInterval {
        match pass {
            // Full uncertainty reports a genuine η interval and a delta-method
            // response interval through the inverse link.
            PredictPass::FullUncertainty => ResponseInterval::SymmetricDelta,
            // Posterior-mean bounds transform the η endpoints through the
            // inverse link.
            PredictPass::PosteriorMean => ResponseInterval::TransformEta,
        }
    }

    fn bounds(&self) -> ResponseBounds {
        ResponseBounds::for_family(&self.likelihood.response)
    }

    fn response_family(&self) -> ResponseFamily {
        self.likelihood.response.clone()
    }

    fn observation_noise(
        &self,
        input: &PredictInput,
    ) -> Result<Option<Array1<f64>>, EstimationError> {
        self.noise_sd(input).map(Some)
    }
}

impl PredictableModel for DispersionLocationScalePredictor {
    fn predict_plugin_response(
        &self,
        input: &PredictInput,
    ) -> Result<PredictResult, EstimationError> {
        let eta = self.eta_mean(input);
        let mean = self.strategy().inverse_link_array(eta.view())?;
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
        self.noise_sd(input).map(Some)
    }

    fn predict_dispersion_scale(
        &self,
        input: &PredictInput,
    ) -> Result<Option<Array1<f64>>, EstimationError> {
        // Per-row precision `exp(eta_d(x))` mapped into the generative
        // NoiseModel's dispersion units (#1125): NB θ, Gamma shape and Beta φ
        // ARE the precision; Tweedie φ is its reciprocal.
        let precision = self.precision(input)?;
        let dispersion = match self.likelihood.response {
            ResponseFamily::Tweedie { .. } => precision.mapv(|pr| 1.0 / pr.max(f64::MIN_POSITIVE)),
            _ => precision,
        };
        Ok(Some(dispersion))
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
        2
    }

    fn block_roles(&self) -> Vec<BlockRole> {
        vec![BlockRole::Location, BlockRole::Scale]
    }
}

/// Binomial location-scale predictor: two blocks (threshold + log-sigma).
///
/// Predicts probabilities through the threshold-scale parameterisation:
///   eta_t = X_threshold @ beta_threshold + offset
///   eta_s = X_noise @ beta_noise + offset_noise
///   sigma = exp(eta_s)
///   q0    = -eta_t / sigma
///   prob  = inverse_link(q0)
///
/// Delta-method SEs propagate through the chain rule of q0 w.r.t. both
/// linear predictors.
