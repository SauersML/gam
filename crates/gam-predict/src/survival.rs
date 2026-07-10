use super::*;

/// Survival location-scale predictor: two blocks (threshold + log-sigma).
///
/// Predicts survival probability via:
///   q0 = -eta_threshold * exp(-eta_log_sigma)
///   survival_prob = 1 - inverse_link(q0)
///
/// The "design" in `PredictInput` is the threshold design matrix, and
/// "design_noise" is the log-sigma design matrix. The time dimension
/// (x_time_exit) is handled externally and is not part of this predictor.
///
/// Both the inverse σ-link and the guarded q0 product are the canonical
/// `gam_model_kernels::sigma_link` kernels shared with the fit engine: exact
/// wherever the mathematical value is representable in `f64`, saturating only
/// at the representability boundary itself.
#[inline]
fn survival_inverse_sigma_from_eta_log_sigma(eta_log_sigma: f64) -> f64 {
    gam_model_kernels::sigma_link::exp_sigma_inverse_from_eta_scalar(eta_log_sigma)
}

#[inline]
fn survival_q0_and_inverse_sigma(eta_threshold: f64, eta_log_sigma: f64) -> (f64, f64) {
    (
        gam_model_kernels::sigma_link::survival_q0_from_eta(eta_threshold, eta_log_sigma),
        survival_inverse_sigma_from_eta_log_sigma(eta_log_sigma),
    )
}

#[inline]
fn survival_tail_value_from_failure_jet(
    inverse_link: &InverseLink,
    eta: f64,
    failure_jet: &InverseLinkJet,
) -> f64 {
    match inverse_link {
        InverseLink::Standard(gam_spec::StandardLink::Probit) => {
            if eta.is_nan() {
                f64::NAN
            } else if eta == f64::INFINITY {
                0.0
            } else if eta == f64::NEG_INFINITY {
                1.0
            } else {
                0.5 * statrs::function::erf::erfc(eta / std::f64::consts::SQRT_2)
            }
        }
        InverseLink::Standard(gam_spec::StandardLink::Logit) => 1.0 / (1.0 + eta.exp()),
        InverseLink::Standard(gam_spec::StandardLink::CLogLog) => (-(eta.exp())).exp(),
        _ => (1.0 - failure_jet.mu).clamp(0.0, 1.0),
    }
}

#[inline]
fn inverse_link_survival_tail_value_and_failure_density(
    inverse_link: &InverseLink,
    eta: f64,
) -> Result<(f64, f64), EstimationError> {
    let failure_jet =
        gam_solve::mixture_link::inverse_link_jet_for_inverse_link(inverse_link, eta)?;
    Ok((
        survival_tail_value_from_failure_jet(inverse_link, eta, &failure_jet).clamp(0.0, 1.0),
        failure_jet.d1,
    ))
}

pub struct SurvivalPredictor {
    pub beta_threshold: Array1<f64>,
    pub beta_log_sigma: Array1<f64>,
    pub covariance: Option<Array2<f64>>,
    pub inverse_link: InverseLink,
}

impl SurvivalPredictor {
    /// Build a `SurvivalPredictor` from a `UnifiedFitResult`, extracting betas
    /// from blocks by role: Threshold (or legacy Location/Mean) ->
    /// beta_threshold, Scale -> beta_log_sigma.
    pub(crate) fn from_unified(
        unified: &UnifiedFitResult,
        inverse_link: InverseLink,
    ) -> Result<Self, EstimationError> {
        let beta_threshold = unified
            .block_by_role(BlockRole::Threshold)
            .or_else(|| unified.block_by_role(BlockRole::Location))
            .or_else(|| unified.block_by_role(BlockRole::Mean))
            .map(|b| b.beta.clone())
            .ok_or_else(|| {
                EstimationError::InvalidInput("Survival model missing threshold block".to_string())
            })?;
        let beta_log_sigma = unified
            .block_by_role(BlockRole::Scale)
            .map(|b| b.beta.clone())
            .ok_or_else(|| {
                EstimationError::InvalidInput(
                    "Survival model missing scale (log-sigma) block".to_string(),
                )
            })?;
        Ok(Self {
            beta_threshold,
            beta_log_sigma,
            covariance: unified.covariance_conditional.clone(),
            inverse_link,
        })
    }

    /// Compute q0 = -eta_threshold * exp(-eta_log_sigma) and survival_prob = 1 - F(q0).
    fn compute_survival(
        &self,
        eta_threshold: &Array1<f64>,
        eta_log_sigma: &Array1<f64>,
    ) -> Result<Array1<f64>, EstimationError> {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let n = eta_threshold.len();
        let survival_prob: Result<Vec<f64>, EstimationError> = (0..n)
            .into_par_iter()
            .map(|i| {
                let (q0, _) = survival_q0_and_inverse_sigma(eta_threshold[i], eta_log_sigma[i]);
                let (survival, _) =
                    inverse_link_survival_tail_value_and_failure_density(&self.inverse_link, q0)?;
                Ok(survival)
            })
            .collect();
        Ok(Array1::from_vec(survival_prob?))
    }
}

impl SurvivalPredictor {
    /// Threshold and log-sigma linear predictors, validating that the noise
    /// design / offset are present.
    fn linear_predictors<'a>(
        &self,
        input: &'a PredictInput,
    ) -> Result<(Array1<f64>, Array1<f64>, &'a DesignMatrix), EstimationError> {
        let eta_threshold = input.design.dot(&self.beta_threshold) + &input.offset;
        let design_noise = input.design_noise.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput(
                "Survival prediction requires noise (log-sigma) design matrix".to_string(),
            )
        })?;
        let offset_noise = input.offset_noise.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput(
                "Survival prediction requires noise (log-sigma) offset".to_string(),
            )
        })?;
        let eta_log_sigma = design_noise.dot(&self.beta_log_sigma) + offset_noise;
        Ok((eta_threshold, eta_log_sigma, design_noise))
    }

    /// Delta-method response-scale SE of the survival probability from an
    /// explicit covariance `backend`, via the threshold/log-σ chain rule.
    /// Shared by the full-uncertainty state and the posterior-mean pass so both
    /// report a genuine probability-scale SE (never the threshold-scale η SE,
    /// which lives in different units).
    fn survival_mean_se_from_backend(
        &self,
        input: &PredictInput,
        backend: &PredictionCovarianceBackend<'_>,
        eta_threshold: &Array1<f64>,
        eta_log_sigma: &Array1<f64>,
        design_noise: &DesignMatrix,
    ) -> Result<Array1<f64>, EstimationError> {
        let n = eta_threshold.len();
        let p_t = self.beta_threshold.len();
        let p_s = self.beta_log_sigma.len();
        linear_predictor_se_from_backend(backend, n, |rows| {
            let x_t = design_row_chunk(&input.design, rows.clone())?;
            let x_s = design_row_chunk(design_noise, rows.clone())?;
            let eta_t_chunk = eta_threshold.slice(ndarray::s![rows.clone()]);
            let eta_ls_chunk = eta_log_sigma.slice(ndarray::s![rows.clone()]);
            let rows_in_chunk = eta_t_chunk.len();
            let mut grad = Array2::<f64>::zeros((rows_in_chunk, p_t + p_s));
            for i in 0..rows_in_chunk {
                let (q0, inv_sigma) =
                    survival_q0_and_inverse_sigma(eta_t_chunk[i], eta_ls_chunk[i]);
                let (_, failure_density) =
                    inverse_link_survival_tail_value_and_failure_density(&self.inverse_link, q0)
                        .map_err(|e| e.to_string())?;
                let dsurv_deta_t = failure_density * inv_sigma;
                let dsurv_deta_s = failure_density * q0;
                for j in 0..p_t {
                    grad[[i, j]] = dsurv_deta_t * x_t[[i, j]];
                }
                for j in 0..p_s {
                    grad[[i, p_t + j]] = dsurv_deta_s * x_s[[i, j]];
                }
            }
            Ok(vec![grad])
        })
    }

    /// Survival point + η/mean standard errors from an explicit covariance
    /// `backend` (so the caller can select conditional vs. smoothing-corrected
    /// covariance). The `covariance_corrected_used` flag is left `false`; the
    /// caller overrides it according to the backend it selected.
    fn state_from_backend(
        &self,
        input: &PredictInput,
        backend: &PredictionCovarianceBackend<'_>,
    ) -> Result<LinearState, EstimationError> {
        let (eta_threshold, eta_log_sigma, design_noise) = self.linear_predictors(input)?;
        let survival_prob = self.compute_survival(&eta_threshold, &eta_log_sigma)?;
        let p_s = self.beta_log_sigma.len();

        let eta_se = padded_design_standard_errors_from_backend(
            &input.design,
            backend,
            0,
            p_s,
            "survival threshold uncertainty",
        )?;

        // Delta-method SE for survival probability.
        let mean_se_vec = self.survival_mean_se_from_backend(
            input,
            backend,
            &eta_threshold,
            &eta_log_sigma,
            design_noise,
        )?;
        Ok(LinearState {
            eta: eta_threshold,
            mean: survival_prob,
            eta_se: Some(eta_se),
            mean_se: Some(mean_se_vec),
            covariance_corrected_used: false,
        })
    }

    /// Plug-in survival point + conditional-covariance η/mean standard errors,
    /// used by the `predict_with_uncertainty` point path. Returns zero-SE
    /// (no interval) state when the predictor carries no covariance.
    fn plugin_state_from_covariance(
        &self,
        input: &PredictInput,
    ) -> Result<LinearState, EstimationError> {
        if let Some(ref cov) = self.covariance {
            let backend = PredictionCovarianceBackend::from_dense(cov.view());
            self.state_from_backend(input, &backend)
        } else {
            let (eta_threshold, eta_log_sigma, _) = self.linear_predictors(input)?;
            let survival_prob = self.compute_survival(&eta_threshold, &eta_log_sigma)?;
            Ok(LinearState {
                eta: eta_threshold,
                mean: survival_prob,
                eta_se: None,
                mean_se: None,
                covariance_corrected_used: false,
            })
        }
    }
}

impl PredictionTransform for SurvivalPredictor {
    fn point_state(&self, input: &PredictInput) -> Result<LinearState, EstimationError> {
        self.plugin_state_from_covariance(input)
    }

    fn linear_state(
        &self,
        input: &PredictInput,
        fit: &UnifiedFitResult,
        pass: PredictPass,
        covariance_mode: InferenceCovarianceMode,
    ) -> Result<LinearState, EstimationError> {
        match pass {
            PredictPass::FullUncertainty => {
                // Select the covariance the caller requested and report which
                // was used, instead of always using the conditional covariance.
                let p_total = self.beta_threshold.len() + self.beta_log_sigma.len();
                let (backend, covariance_corrected_used) =
                    fit.select_uncertainty_backend(p_total, covariance_mode, "survival")?;
                let mut state = self.state_from_backend(input, &backend)?;
                state.covariance_corrected_used = covariance_corrected_used;
                Ok(state)
            }
            PredictPass::PosteriorMean => {
                let (eta_threshold, eta_log_sigma, design_noise) = self.linear_predictors(input)?;
                // The eta_se covers only the threshold block; the response-scale
                // `mean_se` below carries the full threshold + log-σ delta-method
                // propagation, so the collapsed-delta credible band is in genuine
                // survival-probability units.
                //
                // Validation target for this survival posterior-mean path:
                // compare against 50K Monte Carlo draws from N(beta_hat, V) for a
                // simple Weibull-style location-scale survival fit and require
                // agreement within ~0.005; as covariance -> 0, the integrated mean
                // must collapse to the point prediction.
                let p_t = self.beta_threshold.len();
                let p_s = self.beta_log_sigma.len();
                let p_total = p_t + p_s;
                let backend = require_posterior_mean_backend(
                    fit,
                    self.covariance.as_ref(),
                    p_total,
                    "survival posterior mean",
                )?;

                let eta_se = padded_design_standard_errors_from_backend(
                    &input.design,
                    &backend,
                    0,
                    p_s,
                    "survival posterior mean",
                )?;
                let (var_t, var_s, cov_ts) = project_two_block_linear_predictor_covariance(
                    &input.design,
                    design_noise,
                    &backend,
                    p_t,
                    p_s,
                    "survival posterior mean",
                )?;
                let quadctx = gam_solve::quadrature::QuadratureContext::new();
                let mean = Array1::from_vec(
                    (0..eta_threshold.len())
                        .map(|i| {
                            projected_bivariate_posterior_mean_result(
                                &quadctx,
                                [eta_threshold[i], eta_log_sigma[i]],
                                [
                                    [var_t[i].max(0.0), cov_ts[i]],
                                    [cov_ts[i], var_s[i].max(0.0)],
                                ],
                                |threshold, log_sigma| {
                                    let (q0, _) =
                                        survival_q0_and_inverse_sigma(threshold, log_sigma);
                                    let (survival, _) =
                                        inverse_link_survival_tail_value_and_failure_density(
                                            &self.inverse_link,
                                            q0,
                                        )?;
                                    Ok(survival)
                                },
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                );
                let mean_se = self.survival_mean_se_from_backend(
                    input,
                    &backend,
                    &eta_threshold,
                    &eta_log_sigma,
                    design_noise,
                )?;
                Ok(LinearState {
                    eta: eta_threshold,
                    mean,
                    eta_se: Some(eta_se),
                    mean_se: Some(mean_se),
                    covariance_corrected_used: false,
                })
            }
        }
    }

    fn response(&self, eta: &Array1<f64>) -> Result<Array1<f64>, EstimationError> {
        // The survival predictor never routes its interval through the response
        // map (it uses the delta-method response SE), so this is unreachable in
        // practice; kept total for trait completeness.
        self.compute_survival(eta, &Array1::zeros(eta.len()))
    }

    fn response_jacobian_rows(&self, pass: PredictPass) -> ResponseInterval {
        match pass {
            // Survival tail is a probability; the delta-method response interval
            // is μ ± z·SE(μ) clamped to [0, 1] with a genuine threshold-scale η
            // interval retained.
            PredictPass::FullUncertainty => ResponseInterval::SymmetricDelta,
            // The threshold-scale η interval is not directly meaningful on the
            // survival-probability scale, so it is collapsed onto the point
            // predictor and uncertainty flows through the delta-method response
            // SE (here the threshold-only eta_se).
            PredictPass::PosteriorMean => ResponseInterval::CollapsedDelta,
        }
    }

    fn bounds(&self) -> ResponseBounds {
        ResponseBounds::UNIT_PROBABILITY
    }

    fn response_family(&self) -> ResponseFamily {
        // Survival reports the survival probability `S(t)`; a fresh observation
        // at that horizon is the Bernoulli indicator `1{T > t}`, so the generic
        // drivers build the discrete `RoystonParmar` predictive set from
        // `family_observation_band` (marginal `P(Y = 1) = E[S]`).
        ResponseFamily::RoystonParmar
    }
}

impl PredictableModel for SurvivalPredictor {
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
        fit_result: &UnifiedFitResult,
        options: &PredictUncertaintyOptions,
    ) -> Result<PredictUncertaintyResult, EstimationError> {
        predict_full_uncertainty_generic(self, input, fit_result, options)
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
        vec![BlockRole::Threshold, BlockRole::Scale]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_sigma_at_zero_is_one() {
        let result = survival_inverse_sigma_from_eta_log_sigma(0.0);
        assert!((result - 1.0).abs() < 1e-15);
    }

    #[test]
    fn inverse_sigma_positive_eta_decreases() {
        let result = survival_inverse_sigma_from_eta_log_sigma(1.0);
        assert!((result - (-1.0_f64).exp()).abs() < 1e-15);
    }

    #[test]
    fn inverse_sigma_negative_eta_increases() {
        let result = survival_inverse_sigma_from_eta_log_sigma(-2.0);
        assert!((result - 2.0_f64.exp()).abs() < 1e-12);
    }

    #[test]
    fn inverse_sigma_large_positive_underflows_to_zero() {
        // exp(-1000) underflows naturally — the mathematically correct limit.
        let result = survival_inverse_sigma_from_eta_log_sigma(1000.0);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn inverse_sigma_is_exact_on_the_representable_range() {
        // exp(600) ≈ 3.77e260 is representable and must be returned exactly,
        // not rewritten by an artificial cap.
        let result = survival_inverse_sigma_from_eta_log_sigma(-600.0);
        assert_eq!(result, 600.0_f64.exp());
    }

    #[test]
    fn inverse_sigma_saturates_finite_at_representability_boundary() {
        // exp(1000) overflows binary64; the link saturates near f64::MAX
        // instead of returning +inf.
        let result = survival_inverse_sigma_from_eta_log_sigma(-1000.0);
        assert!(result.is_finite());
        assert!(result > 1e308);
    }

    #[test]
    fn q0_zero_threshold_returns_zero_and_inv_sigma() {
        let (q0, inv_sigma) = survival_q0_and_inverse_sigma(0.0, 1.0);
        assert_eq!(q0, 0.0);
        assert!((inv_sigma - (-1.0_f64).exp()).abs() < 1e-15);
    }

    #[test]
    fn q0_unit_threshold_and_zero_log_sigma() {
        // eta_threshold = 1.0, eta_log_sigma = 0.0 → inv_sigma = 1.0, q0 = -1.0
        let (q0, inv_sigma) = survival_q0_and_inverse_sigma(1.0, 0.0);
        assert!((inv_sigma - 1.0).abs() < 1e-15);
        assert!((q0 - (-1.0)).abs() < 1e-15);
    }

    #[test]
    fn q0_negative_threshold_positive_sign() {
        // eta_threshold = -2.0, eta_log_sigma = 0.0 → q0 = -(-2.0)*1.0 = 2.0
        let (q0, _) = survival_q0_and_inverse_sigma(-2.0, 0.0);
        assert!((q0 - 2.0).abs() < 1e-15);
    }

    #[test]
    fn q0_huge_finite_product_is_exact() {
        // |q0| = 1e300 is representable, so no saturation may engage.
        let (q0, _) = survival_q0_and_inverse_sigma(1e300, 0.0);
        assert_eq!(q0, -1e300);
        let (q0_neg, _) = survival_q0_and_inverse_sigma(-1e300, 0.0);
        assert_eq!(q0_neg, 1e300);
    }

    #[test]
    fn q0_saturates_only_past_f64_representability() {
        // ln(1e300) - (-100) ≈ 790.8 > ln(f64::MAX): the true product
        // overflows binary64 and saturates to ∓MAX.
        let (q0, _) = survival_q0_and_inverse_sigma(1e300, -100.0);
        assert_eq!(q0, -f64::MAX);
        let (q0_neg, _) = survival_q0_and_inverse_sigma(-1e300, -100.0);
        assert_eq!(q0_neg, f64::MAX);
    }

    #[test]
    fn q0_log_domain_rescues_saturated_inverse_sigma() {
        // exp(720) alone is unrepresentable, but 1e-9 · exp(720) ≈ e^699.3 is
        // finite; the product must be evaluated in the log domain.
        let (q0, inv_sigma) = survival_q0_and_inverse_sigma(1e-9, -720.0);
        assert!(!inv_sigma.is_infinite());
        assert!(q0.is_finite());
        let expected = -(1e-9_f64.ln() + 720.0).exp();
        assert!((q0 - expected).abs() <= 1e-10 * expected.abs());
    }

    #[test]
    fn survival_tail_probit_at_infinity_is_zero() {
        let link = InverseLink::Standard(gam_spec::StandardLink::Probit);
        let jet = InverseLinkJet {
            mu: 0.0,
            d1: 0.0,
            d2: 0.0,
            d3: 0.0,
        };
        let tail = survival_tail_value_from_failure_jet(&link, f64::INFINITY, &jet);
        assert_eq!(tail, 0.0);
    }

    #[test]
    fn survival_tail_probit_at_neg_infinity_is_one() {
        let link = InverseLink::Standard(gam_spec::StandardLink::Probit);
        let jet = InverseLinkJet {
            mu: 1.0,
            d1: 0.0,
            d2: 0.0,
            d3: 0.0,
        };
        let tail = survival_tail_value_from_failure_jet(&link, f64::NEG_INFINITY, &jet);
        assert_eq!(tail, 1.0);
    }

    #[test]
    fn survival_tail_logit_at_zero_is_half() {
        let link = InverseLink::Standard(gam_spec::StandardLink::Logit);
        let jet = InverseLinkJet {
            mu: 0.5,
            d1: 0.25,
            d2: 0.0,
            d3: 0.0,
        };
        let tail = survival_tail_value_from_failure_jet(&link, 0.0, &jet);
        // 1 / (1 + exp(0)) = 0.5
        assert!((tail - 0.5).abs() < 1e-15);
    }

    #[test]
    fn survival_tail_cloglog_at_zero() {
        let link = InverseLink::Standard(gam_spec::StandardLink::CLogLog);
        let jet = InverseLinkJet {
            mu: 0.0,
            d1: 0.0,
            d2: 0.0,
            d3: 0.0,
        };
        // exp(-exp(0)) = exp(-1)
        let tail = survival_tail_value_from_failure_jet(&link, 0.0, &jet);
        assert!((tail - (-1.0_f64).exp()).abs() < 1e-15);
    }
}
