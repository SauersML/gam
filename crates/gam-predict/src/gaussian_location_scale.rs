use super::*;

/// Gaussian location-scale predictor: two blocks (mean + log-sigma).
///
/// Predicts `mean = X_mu @ beta_mu` (identity link on mean) and
/// `sigma = response_scale · sigma_floor + exp(X_noise @ beta_noise + offset_noise)`.
///
/// `sigma_floor` is the standardized-response σ floor (`LOGB_SIGMA_FLOOR`) and
/// `response_scale` maps it back to raw response units. The `exp(η)` term is
/// already in raw units (the persisted log-σ intercept is shifted by
/// `+ln(response_scale)` at fit time), so only the floor is scaled here — see
/// `GaussianLocationScalePredictor::compute_sigma`.
pub struct GaussianLocationScalePredictor {
    pub beta_mu: Array1<f64>,
    pub beta_noise: Array1<f64>,
    pub sigma_floor: f64,
    pub response_scale: f64,
    pub covariance: Option<Array2<f64>>,
    pub link_wiggle: Option<SavedLinkWiggleRuntime>,
}

/// `sqrt(E[(floor + exp(Z))²])` for `Z ~ N(mean, variance)`. Form square
/// roots of all three nonnegative moment terms before combining them so a
/// representable SD is not lost when its variance overflows or underflows.
fn shifted_lognormal_root_second_moment(mean: f64, variance: f64, floor: f64) -> f64 {
    let exponential_sd = (mean + variance).exp();
    let cross_sd = if floor == 0.0 {
        0.0
    } else {
        (0.5 * std::f64::consts::LN_2 + 0.5 * floor.ln() + 0.5 * mean + 0.25 * variance)
            .exp()
    };
    floor.hypot(exponential_sd).hypot(cross_sd)
}

impl GaussianLocationScalePredictor {
    /// Reconstruct σ in raw response units from the persisted Scale-block
    /// coefficients.
    ///
    /// The persisted `beta_noise` is already in RAW response units:
    /// `rescale_gaussian_location_scale_to_raw` shifts the log-σ block intercept
    /// by `+ln(response_scale)`, so `η_noise = η_internal + ln(response_scale)`
    /// and `exp(η_noise) = response_scale · exp(η_internal)` already carries one
    /// factor of the response scale. The soft floor is the only piece still in
    /// standardized units — it sits *outside* the exp and cannot ride the
    /// intercept shift — so it alone is multiplied by `response_scale` here:
    ///
    ///   σ_raw = response_scale · sigma_floor + exp(η_noise)
    ///         = response_scale · (sigma_floor + exp(η_internal))
    ///         = response_scale · σ_internal.
    ///
    /// This applies **exactly one** factor of `response_scale` to the σ surface,
    /// matching the fit-side reconstruction (the #1874 equivariance test and the
    /// FFI). Multiplying the whole `(sigma_floor + exp(η_noise))` by
    /// `response_scale` would double-count it on the exp term (`response_scale²`),
    /// breaking response-scale equivariance of the reported σ (#1874/#1928).
    fn compute_sigma(
        &self,
        design_noise: &DesignMatrix,
        offset_noise: Option<&Array1<f64>>,
    ) -> Result<Array1<f64>, EstimationError> {
        let eta_noise = self.eta_noise(design_noise, offset_noise)?;
        let scaled_floor = self.response_scale * self.sigma_floor;
        Ok(eta_noise.mapv(|eta| {
            gam_model_kernels::sigma_link::logb_sigma_from_eta_with_floor_scalar(scaled_floor, eta)
        }))
    }

    /// Log-σ linear predictor `η_s = X_noise β_noise + offset_noise`.
    fn eta_noise(
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
        Ok(eta_noise)
    }

    /// Predictive observation-noise SD `sqrt(E[σ²])`, integrating the log-σ
    /// posterior instead of plugging in its mode.
    ///
    /// The future response is `Y = μ(β) + σ(β_s)·Z` with `Z ~ N(0,1)`
    /// independent of the posterior, so by the law of total variance
    /// `Var(Y) = Var(μ̂) + E[σ²]` — the cross term vanishes because `E[Z] = 0`.
    /// The interval engine adds `Var(μ̂) = mean_se²` itself, so the noise term
    /// it must receive is `E[σ²]`, not the plug-in `σ(m̂)²` (audit finding 6):
    /// with `η_s ~ N(0, 1)` the exact `E[σ²] ≈ e² ≈ 7.39` while the plug-in
    /// reports `1`.
    ///
    /// With `σ = f + exp(η_s)` (`f` the scaled floor) and the per-row posterior
    /// `η_s ~ N(m, v)`, the lognormal moments give exactly
    ///   `E[σ²] = f² + 2·f·exp(m + v/2) + exp(2m + 2v)`,
    /// which reduces to the plug-in `σ(m)²` at `v = 0` (also the no-covariance
    /// degrade). The SD is evaluated without forming an overflowing squared
    /// moment; `+inf` is returned only when the SD itself is outside the range.
    fn integrated_noise_sd(&self, input: &PredictInput) -> Result<Array1<f64>, EstimationError> {
        let design_noise = input.design_noise.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput(
                "Gaussian location-scale prediction requires noise design matrix".to_string(),
            )
        })?;
        let eta_noise = self.eta_noise(design_noise, input.offset_noise.as_ref())?;
        let scaled_floor = self.response_scale * self.sigma_floor;
        let log_sigma_var = match self.covariance.as_ref() {
            Some(covariance) => {
                let backend = PredictionCovarianceBackend::from_dense(covariance.view());
                let p_mu = self.beta_mu.len();
                let p_w = self.link_wiggle.as_ref().map_or(0, |w| w.beta.len());
                // Coefficient layout is `[mean | scale | wiggle]`, so the log-σ
                // block sits after the `p_mu` mean columns.
                let se = padded_design_standard_errors_from_backend(
                    design_noise,
                    &backend,
                    p_mu,
                    p_w,
                    "gaussian location-scale log-sigma uncertainty",
                )?;
                se.mapv(|s| s * s)
            }
            None => Array1::zeros(eta_noise.len()),
        };
        Ok(Array1::from_shape_fn(eta_noise.len(), |i| {
            shifted_lognormal_root_second_moment(eta_noise[i], log_sigma_var[i], scaled_floor)
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
            covariance_source: InferenceCovarianceMode::Conditional,
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
        let (backend, covariance_source) = match pass {
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
                InferenceCovarianceMode::Conditional,
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
            covariance_source,
        })
    }

    fn response(&self, eta: &Array1<f64>) -> Result<Array1<f64>, EstimationError> {
        Ok(eta.clone())
    }

    fn response_jacobian_rows(&self, pass: PredictPass) -> ResponseInterval {
        match pass {
            // The response IS the linear predictor (identity link), so the
            // response interval is exactly the η interval on both passes —
            // there is no transform to delta-approximate either way. Spelled
            // out per pass so a new one has to state its policy here rather
            // than inherit this one silently.
            PredictPass::FullUncertainty | PredictPass::PosteriorMean => {
                ResponseInterval::IdentityEta
            }
        }
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
        // The predictive band needs `E[σ²]` under the log-σ posterior, not the
        // plug-in σ(m̂) — see `integrated_noise_sd`. The fitted σ *surface*
        // (plug-in) remains available through `predict_noise_scale`.
        self.integrated_noise_sd(input).map(Some)
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

#[cfg(test)]
mod noise_moment_tests {
    use super::shifted_lognormal_root_second_moment;

    #[test]
    fn gaussian_noise_sd_retains_extreme_finite_standard_deviations() {
        // A zero posterior variance makes the conditional noise SD exactly
        // floor+exp(mean), even when its square lies outside the float range.
        for mean in [-400.0_f64, 400.0] {
            for floor in [0.0, 0.25, (-401.0_f64).exp()] {
                let expected = floor + mean.exp();
                let actual = shifted_lognormal_root_second_moment(mean, 0.0, floor);
                assert!((actual / expected - 1.0).abs() < 1e-13);
            }
        }
    }

    #[test]
    fn gaussian_noise_sd_matches_lognormal_moments_with_uncertainty() {
        let mean = 0.3_f64;
        let variance = 0.4_f64;
        let floor = 0.2_f64;
        let expected = (floor * floor
            + 2.0 * floor * (mean + 0.5 * variance).exp()
            + (2.0 * mean + 2.0 * variance).exp())
        .sqrt();
        let actual = shifted_lognormal_root_second_moment(mean, variance, floor);
        assert!((actual / expected - 1.0).abs() < 1e-14);
        assert_eq!(
            shifted_lognormal_root_second_moment(800.0, 0.0, 0.0),
            f64::INFINITY
        );
    }
}
