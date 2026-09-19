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

/// How coefficient uncertainty enters the response-scale point of an anchored
/// marginal-slope prediction, `p = E_θ[Φ(η(θ))]` under `θ ~ N(θ̂, V)`.
///
/// The anchored linear predictor is `η = c(b)·q + s·b·z` (standard-normal
/// latent law) or `η = a(q, b) + s·b·z` with `a` the root of the calibration
/// equation (declared empirical law), where `q = X·β_q` and `b = W·β_b` are
/// affine in `θ`. A coefficient draw therefore moves `q`, `b` AND the anchor.
/// The variants name what each one does with that dependence; they exist so
/// the shortcuts stay reachable by name for comparison, not as alternatives a
/// caller should prefer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnchoredPosteriorIntegration {
    /// `Φ(η(θ̂))`: posterior-mean coefficients inserted into the nonlinear
    /// response; no coefficient uncertainty at all.
    PlugIn,
    /// `η ~ N(η̂, s²z²·Var(b))`, the anchor frozen at `θ̂`: only the direct
    /// `s·b·z` term carries uncertainty, then `Φ(η̂/√(1 + v))`. The Gaussian
    /// shortcut of a model whose intercept does not move.
    FrozenAnchorGaussian,
    /// `η ~ N(η̂, gᵀVg)` with `g = ∂η/∂θ|θ̂` the complete first-order
    /// sensitivity (anchor moved through its implicit-function derivative
    /// `∂a/∂θ = −F_θ/F_a`), then `Φ(η̂/√(1 + gᵀVg))`. Exact for a linear `η(θ)`;
    /// drops the curvature of `c(b)·q` and of `a(q, b)`. This is what the
    /// posterior-mean pass reported before the exact integration existed, and
    /// what the flexible (score-warp / link-deviation) and residual repair
    /// paths still report.
    LinearisedAnchorGaussian,
    /// Exact: `(q, b) ~ N₂((q̂, b̂), J V Jᵀ)` — exact because both are affine
    /// in `θ` — and `Φ(η(q, b))` is integrated over that bivariate law by
    /// adaptive Gauss–Hermite quadrature with the anchor re-solved at every
    /// node. Unavailable while a flexible runtime is active, where the anchor
    /// depends on the flex coefficient vectors and not on `(q, b)` alone, and
    /// while a residual repair block is present, where the anchor reads the
    /// whole residual coefficient vector through `b̃ᵀΣb̃` (gam#2924).
    ExactAnchor,
}

impl AnchoredPosteriorIntegration {
    /// The integration the posterior-mean pass runs for `predictor`: exact
    /// wherever the anchor is a function of `(q, b)`, first-order otherwise.
    pub fn default_for(predictor: &BernoulliMarginalSlopePredictor) -> Self {
        if predictor.has_flexible_runtime() || predictor.has_residual_repair() {
            Self::LinearisedAnchorGaussian
        } else {
            Self::ExactAnchor
        }
    }
}

/// The response-scale point `E_θ[Φ(η(θ))]` of a Bernoulli marginal-slope
/// prediction under the named `integration`, one entry per row of `input`.
/// The coefficient covariance is the conditional posterior the fit carries
/// (or the predictor's own copy), as for every posterior-mean pass.
pub fn bernoulli_marginal_slope_posterior_mean(
    predictor: &BernoulliMarginalSlopePredictor,
    input: &PredictInput,
    fit: &UnifiedFitResult,
    integration: AnchoredPosteriorIntegration,
) -> Result<Array1<f64>, EstimationError> {
    let theta = predictor.theta();
    let eta = predictor.final_eta_from_theta(input, &theta)?;
    let backend = || {
        require_posterior_mean_backend(
            fit,
            predictor.covariance.as_ref(),
            theta.len(),
            "bernoulli marginal-slope posterior mean",
        )
    };
    let strategy = strategy_for_family(predictor.likelihood_family(), Some(&predictor.base_link));
    let gaussian_eta_mean = |eta_se: &Array1<f64>| {
        PREDICT_QUADRATURE_CONTEXT.with(|quadctx| {
            eta.iter()
                .zip(eta_se.iter())
                .map(|(&eta_i, &se)| strategy.posterior_mean(quadctx, eta_i, se))
                .collect::<Result<Vec<_>, _>>()
                .map(Array1::from_vec)
        })
    };
    match integration {
        AnchoredPosteriorIntegration::PlugIn => predictor.mean_from_eta(&eta),
        AnchoredPosteriorIntegration::LinearisedAnchorGaussian => {
            let eta_se = bernoulli_eta_standard_error_from_backend(predictor, input, &backend()?)?;
            gaussian_eta_mean(&eta_se)
        }
        AnchoredPosteriorIntegration::FrozenAnchorGaussian
        | AnchoredPosteriorIntegration::ExactAnchor => {
            // Both need the coefficient posterior pushed onto the primaries:
            // with no flexible runtime `θ = [β_q | β_b]`, so the projection is
            // the two-block one the survival predictor already uses.
            let kernels = predictor.anchored_row_kernels(input)?;
            let (q, b) = predictor.anchored_primaries(input, &theta)?;
            let design_slope = input.design_noise.as_ref().ok_or_else(|| {
                EstimationError::InvalidInput(
                    "bernoulli marginal-slope prediction requires slope design".to_string(),
                )
            })?;
            let (var_q, var_b, cov_qb) = project_two_block_linear_predictor_covariance(
                &input.design,
                design_slope,
                &backend()?,
                predictor.beta_marginal.len(),
                predictor.beta_slope.len(),
                "bernoulli marginal-slope posterior mean",
            )?;
            if integration == AnchoredPosteriorIntegration::FrozenAnchorGaussian {
                let eta_se = Array1::from_iter(kernels.iter().zip(var_b.iter()).map(
                    |(kernel, &var_b_i)| {
                        let sz = kernel.probit_scale() * kernel.latent_z();
                        (sz * sz * var_b_i).sqrt()
                    },
                ));
                return gaussian_eta_mean(&eta_se);
            }
            // Exact: every quadrature node re-solves the anchor. Rows are
            // independent, and an empirical law costs a root solve per node,
            // so spread them across the pool.
            let rows: Result<Vec<f64>, EstimationError> = (0..eta.len())
                .into_par_iter()
                .map(|i| {
                    PREDICT_QUADRATURE_CONTEXT.with(|quadctx| {
                        projected_bivariate_posterior_mean_result(
                            quadctx,
                            [q[i], b[i]],
                            [[var_q[i], cov_qb[i]], [cov_qb[i], var_b[i]]],
                            |q_node, b_node| Ok(normal_cdf(kernels[i].eta(q_node, b_node)?)),
                        )
                    })
                })
                .collect();
            Ok(Array1::from_vec(rows?))
        }
    }
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
        let eta = self.final_eta_from_theta(input, &self.theta())?;
        match pass {
            PredictPass::FullUncertainty => {
                // Select the covariance the caller requested (conditional vs.
                // smoothing-corrected) instead of always using the conditional
                // backend, and report which was used.
                let (backend, covariance_source) = fit.select_uncertainty_backend(
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
                    covariance_source,
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
                // The point is the posterior-predictive probability of the
                // ANCHORED model: the anchor is re-solved at every coefficient
                // node, not linearised at θ̂ and pushed through the Gaussian
                // probit identity (`AnchoredPosteriorIntegration` names both).
                let mean = bernoulli_marginal_slope_posterior_mean(
                    self,
                    input,
                    fit,
                    AnchoredPosteriorIntegration::default_for(self),
                )?;
                // Response-scale delta-method SE: SE(μ) = |dμ/dη|·SE(η). The
                // η-scale SE alone lives on the link scale and must never be
                // reported as a probability-scale SE.
                let mean_se = eta_se.clone() * self.mean_derivative_from_eta(&eta)?;
                Ok(LinearState {
                    eta,
                    mean,
                    eta_se: Some(eta_se),
                    mean_se: Some(mean_se),
                    covariance_source: InferenceCovarianceMode::Conditional,
                })
            }
        }
    }

    fn response(&self, eta: &Array1<f64>) -> Result<Array1<f64>, EstimationError> {
        self.mean_from_eta(eta)
    }

    fn response_jacobian_rows(&self, pass: PredictPass) -> ResponseInterval {
        match pass {
            // Both passes push the η endpoints through the marginal-slope
            // response map: the response is the same smooth image of η either
            // way, so neither pass has its own delta-method row set. Spelled
            // out per pass so a new one has to state its policy here rather
            // than inherit this one silently.
            PredictPass::FullUncertainty | PredictPass::PosteriorMean => {
                ResponseInterval::TransformEta
            }
        }
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
        2 + usize::from(self.beta_residual.is_some())
            + usize::from(self.beta_score_warp.is_some())
            + usize::from(self.beta_link_dev.is_some())
    }

    fn block_roles(&self) -> Vec<BlockRole> {
        let mut roles = vec![BlockRole::Location, BlockRole::Scale];
        if self.beta_residual.is_some() {
            // The residual repair block reads the genome beside the score: a
            // mean-model read of the outcome, not a link or scale correction.
            roles.push(BlockRole::Mean);
        }
        if self.beta_score_warp.is_some() {
            roles.push(BlockRole::Mean);
        }
        if self.beta_link_dev.is_some() {
            roles.push(BlockRole::LinkWiggle);
        }
        roles
    }
}
