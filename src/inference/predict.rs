use crate::estimate::{EstimationError, FitResult, FittedLinkState, UnifiedFitResult};
use crate::families::strategy::{FamilyStrategy, strategy_for_family, strategy_from_fit};
use crate::linalg::utils::predict_gam_dimension_mismatch_message;
use crate::matrix::DesignMatrix;
use crate::mixture_link::{
    InverseLinkJet, beta_logistic_inverse_link_jetwith_param_partials,
    mixture_inverse_link_jetwith_rho_partials_into, sas_inverse_link_jetwith_param_partials,
};
use crate::probability::standard_normal_quantile;
use crate::types::InverseLink;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

pub(crate) fn se_from_covariance(cov: &Array2<f64>) -> Array1<f64> {
    let p = cov.nrows().min(cov.ncols());
    let mut se = Array1::<f64>::zeros(p);
    for i in 0..p {
        se[i] = cov[[i, i]].max(0.0).sqrt();
    }
    se
}

fn apply_family_inverse_link(
    eta: &Array1<f64>,
    family: crate::types::LikelihoodFamily,
    link_kind: Option<&InverseLink>,
) -> Result<Array1<f64>, EstimationError> {
    if matches!(family, crate::types::LikelihoodFamily::RoystonParmar) {
        return Err(EstimationError::InvalidInput(
            "prediction uncertainty for RoystonParmar is not available in predict_gam".to_string(),
        ));
    }
    strategy_for_family(family, link_kind).inverse_link_array(eta.view())
}

#[inline]
fn quadratic_form(cov: &Array2<f64>, grad: &[f64]) -> Result<f64, EstimationError> {
    if cov.nrows() != grad.len() || cov.ncols() != grad.len() {
        return Err(EstimationError::InvalidInput(format!(
            "covariance/gradient dimension mismatch: covariance is {}x{}, gradient length is {}",
            cov.nrows(),
            cov.ncols(),
            grad.len()
        )));
    }
    let mut acc = 0.0_f64;
    for i in 0..grad.len() {
        for j in 0..grad.len() {
            acc += grad[i] * cov[[i, j]] * grad[j];
        }
    }
    Ok(acc.max(0.0))
}

#[inline]
fn quadratic_form_from_jetmu(
    cov: &Array2<f64>,
    partials: &[InverseLinkJet],
) -> Result<f64, EstimationError> {
    let m = partials.len();
    if cov.nrows() != m || cov.ncols() != m {
        return Err(EstimationError::InvalidInput(format!(
            "covariance/mixture-gradient dimension mismatch: covariance is {}x{}, mixture gradient length is {}",
            cov.nrows(),
            cov.ncols(),
            m
        )));
    }
    let mut acc = 0.0_f64;
    for i in 0..m {
        let gi = partials[i].mu;
        for j in 0..m {
            acc += gi * cov[[i, j]] * partials[j].mu;
        }
    }
    Ok(acc.max(0.0))
}

fn linear_predictorvariance(
    x: &DesignMatrix,
    cov: &Array2<f64>,
) -> Result<Array1<f64>, EstimationError> {
    x.quadratic_form_diag(cov).map_err(|e| {
        EstimationError::InvalidInput(format!("failed to compute linear predictor variance: {e}"))
    })
}

pub struct PredictResult {
    pub eta: Array1<f64>,
    pub mean: Array1<f64>,
}

// ═══════════════════════════════════════════════════════════════════════════
//  PredictableModel trait — uniform prediction interface for all model types
// ═══════════════════════════════════════════════════════════════════════════

use crate::estimate::BlockRole;

/// Input to the prediction trait. Contains the design matrix and metadata
/// needed for point prediction + uncertainty quantification.
pub struct PredictInput {
    /// Design matrix for the primary (mean/location) block.
    pub design: DesignMatrix,
    /// Offset vector for the primary block.
    pub offset: Array1<f64>,
    /// Optional design matrix for the noise/scale block (GAMLSS/survival).
    pub design_noise: Option<DesignMatrix>,
    /// Optional offset vector for the noise/scale block.
    pub offset_noise: Option<Array1<f64>>,
}

/// Point prediction with optional standard errors on the linear predictor.
pub struct PredictionWithSE {
    /// Linear predictor η = Xβ + offset.
    pub eta: Array1<f64>,
    /// Response-scale prediction g⁻¹(η).
    pub mean: Array1<f64>,
    /// Standard error of η (if covariance available).
    pub eta_se: Option<Array1<f64>>,
    /// Standard error of the mean (delta-method, if covariance available).
    pub mean_se: Option<Array1<f64>>,
}

/// Trait for models that can produce predictions from new data.
///
/// Implemented by each model class (standard, GAMLSS, survival) to provide
/// a uniform prediction interface. Eliminates the match-dispatch pattern in
/// main.rs for predict, NUTS, and summary commands.
pub trait PredictableModel {
    /// Point prediction on the response scale.
    fn predict_response(&self, input: &PredictInput) -> Result<PredictResult, EstimationError>;

    /// Prediction with uncertainty quantification (SE on eta and mean scales).
    fn predict_with_uncertainty(
        &self,
        input: &PredictInput,
    ) -> Result<PredictionWithSE, EstimationError>;

    /// Full prediction with confidence/observation intervals.
    ///
    /// Delegates to `predict_gamwith_uncertainty` for standard models.
    /// Survival and location-scale models will override with domain-specific
    /// interval construction.
    fn predict_full_uncertainty(
        &self,
        input: &PredictInput,
        fit: &FitResult,
        options: &PredictUncertaintyOptions,
    ) -> Result<PredictUncertaintyResult, EstimationError>;

    /// Posterior-mean prediction with coefficient uncertainty propagation.
    ///
    /// For nonlinear links, returns E[g^{-1}(eta_tilde)] where eta_tilde ~ N(eta_hat, se^2).
    fn predict_posterior_mean(
        &self,
        input: &PredictInput,
        fit: &FitResult,
    ) -> Result<PredictPosteriorMeanResult, EstimationError>;

    /// Number of coefficient blocks in the model.
    fn n_blocks(&self) -> usize;

    /// Roles of each block.
    fn block_roles(&self) -> Vec<BlockRole>;
}

/// Standard (single-block) GAM predictor.
pub struct StandardPredictor {
    pub beta: Array1<f64>,
    pub family: crate::types::LikelihoodFamily,
    pub link_kind: Option<InverseLink>,
    pub covariance: Option<Array2<f64>>,
}

impl StandardPredictor {
    /// Build a `StandardPredictor` from a `UnifiedFitResult`, extracting beta
    /// from the first block and covariance from the unified result.
    pub(crate) fn from_unified(
        unified: &UnifiedFitResult,
        family: crate::types::LikelihoodFamily,
        link_kind: Option<InverseLink>,
    ) -> Self {
        let beta = unified
            .blocks
            .first()
            .map(|b| b.beta.clone())
            .unwrap_or_default();
        let covariance = unified.covariance_conditional.clone();
        Self {
            beta,
            family,
            link_kind,
            covariance,
        }
    }
}

impl PredictableModel for StandardPredictor {
    fn predict_response(&self, input: &PredictInput) -> Result<PredictResult, EstimationError> {
        predict_gam(
            input.design.clone(),
            self.beta.view(),
            input.offset.view(),
            self.family,
        )
    }

    fn predict_with_uncertainty(
        &self,
        input: &PredictInput,
    ) -> Result<PredictionWithSE, EstimationError> {
        let result = self.predict_response(input)?;
        let (eta_se, mean_se) = if let Some(ref cov) = self.covariance {
            let se = eta_standard_errors_from_design(&input.design, cov)?;
            let strategy = strategy_for_family(self.family, self.link_kind.as_ref());
            let mean_se = delta_method_mean_se(&result.eta, &se, &strategy)?;
            (Some(se), Some(mean_se))
        } else {
            (None, None)
        };
        Ok(PredictionWithSE {
            eta: result.eta,
            mean: result.mean,
            eta_se,
            mean_se,
        })
    }

    fn predict_full_uncertainty(
        &self,
        input: &PredictInput,
        fit: &FitResult,
        options: &PredictUncertaintyOptions,
    ) -> Result<PredictUncertaintyResult, EstimationError> {
        predict_gamwith_uncertainty(
            input.design.clone(),
            self.beta.view(),
            input.offset.view(),
            self.family,
            fit,
            options,
        )
    }

    fn predict_posterior_mean(
        &self,
        input: &PredictInput,
        fit: &FitResult,
    ) -> Result<PredictPosteriorMeanResult, EstimationError> {
        let cov = fit.beta_covariance().ok_or_else(|| {
            EstimationError::InvalidInput(
                "posterior-mean prediction requires beta covariance in fit result".to_string(),
            )
        })?;
        predict_gam_posterior_meanwith_fit(
            input.design.clone(),
            self.beta.view(),
            input.offset.view(),
            self.family,
            cov.view(),
            fit,
        )
    }

    fn n_blocks(&self) -> usize {
        1
    }

    fn block_roles(&self) -> Vec<BlockRole> {
        vec![BlockRole::Mean]
    }
}

/// Gaussian location-scale predictor: two blocks (mean + log-sigma).
///
/// Predicts `mean = X_mu @ beta_mu` (identity link on mean) and
/// `sigma = exp(X_noise @ beta_noise) * response_scale`.
pub struct GaussianLocationScalePredictor {
    pub beta_mu: Array1<f64>,
    pub beta_noise: Array1<f64>,
    pub response_scale: f64,
    pub covariance: Option<Array2<f64>>,
}

impl GaussianLocationScalePredictor {
    pub(crate) fn from_unified(
        unified: &UnifiedFitResult,
        response_scale: f64,
    ) -> Result<Self, EstimationError> {
        let beta_mu = unified
            .blocks
            .iter()
            .find(|b| matches!(b.role, BlockRole::Location | BlockRole::Mean))
            .map(|b| b.beta.clone())
            .ok_or_else(|| {
                EstimationError::InvalidInput(
                    "Gaussian location-scale model missing location/mean block".to_string(),
                )
            })?;
        let beta_noise = unified
            .blocks
            .iter()
            .find(|b| matches!(b.role, BlockRole::Scale))
            .map(|b| b.beta.clone())
            .ok_or_else(|| {
                EstimationError::InvalidInput(
                    "Gaussian location-scale model missing scale block".to_string(),
                )
            })?;
        Ok(Self {
            beta_mu,
            beta_noise,
            response_scale,
            covariance: unified.covariance_conditional.clone(),
        })
    }

    /// Compute sigma = exp(eta_noise) * response_scale for each observation.
    fn compute_sigma(&self, design_noise: &DesignMatrix) -> Array1<f64> {
        let eta_noise = design_noise.dot(&self.beta_noise);
        eta_noise.mapv(|eta| eta.exp() * self.response_scale)
    }
}

impl PredictableModel for GaussianLocationScalePredictor {
    fn predict_response(&self, input: &PredictInput) -> Result<PredictResult, EstimationError> {
        let eta = input.design.dot(&self.beta_mu) + &input.offset;
        // Gaussian identity link: mean = eta.
        let mean = eta.clone();
        Ok(PredictResult { eta, mean })
    }

    fn predict_with_uncertainty(
        &self,
        input: &PredictInput,
    ) -> Result<PredictionWithSE, EstimationError> {
        let result = self.predict_response(input)?;
        let design_noise = input.design_noise.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput(
                "Gaussian location-scale prediction requires noise design matrix".to_string(),
            )
        })?;
        let sigma = self.compute_sigma(design_noise);
        // For Gaussian LS, the "SE" on the mean scale is sigma (the distribution parameter).
        // This is an observation-level interval, not an estimator interval.
        Ok(PredictionWithSE {
            eta: result.eta,
            mean: result.mean,
            eta_se: Some(sigma.clone()),
            mean_se: Some(sigma),
        })
    }

    fn predict_full_uncertainty(
        &self,
        input: &PredictInput,
        fit: &FitResult,
        options: &PredictUncertaintyOptions,
    ) -> Result<PredictUncertaintyResult, EstimationError> {
        let _ = fit; // not needed for Gaussian LS intervals
        let pred = self.predict_with_uncertainty(input)?;
        let sigma = pred.mean_se.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput("missing sigma for Gaussian LS intervals".to_string())
        })?;
        let z = crate::probability::standard_normal_quantile(
            0.5 + options.confidence_level * 0.5,
        )
        .map_err(|e| EstimationError::InvalidInput(e))?;
        let eta_lower = &pred.eta - &sigma.mapv(|s| z * s);
        let eta_upper = &pred.eta + &sigma.mapv(|s| z * s);
        Ok(PredictUncertaintyResult {
            eta: pred.eta.clone(),
            mean: pred.mean.clone(),
            eta_standard_error: sigma.clone(),
            mean_standard_error: sigma.clone(),
            eta_lower: eta_lower.clone(),
            eta_upper: eta_upper.clone(),
            mean_lower: eta_lower,
            mean_upper: eta_upper,
            observation_lower: None,
            observation_upper: None,
            covariance_mode_requested: options.covariance_mode,
            covariance_corrected_used: false,
        })
    }

    fn predict_posterior_mean(
        &self,
        input: &PredictInput,
        fit: &FitResult,
    ) -> Result<PredictPosteriorMeanResult, EstimationError> {
        // Gaussian identity link: posterior mean = point estimate (no nonlinearity).
        let _ = fit;
        let result = self.predict_response(input)?;
        let design_noise = input.design_noise.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput(
                "Gaussian location-scale posterior mean requires noise design matrix".to_string(),
            )
        })?;
        let sigma = self.compute_sigma(design_noise);
        Ok(PredictPosteriorMeanResult {
            eta: result.eta,
            eta_standard_error: sigma,
            mean: result.mean,
        })
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
pub struct BinomialLocationScalePredictor {
    pub beta_threshold: Array1<f64>,
    pub beta_noise: Array1<f64>,
    pub covariance: Option<Array2<f64>>,
    pub inverse_link: InverseLink,
}

impl BinomialLocationScalePredictor {
    pub(crate) fn from_unified(
        unified: &UnifiedFitResult,
        inverse_link: InverseLink,
    ) -> Result<Self, EstimationError> {
        let beta_threshold = unified
            .blocks
            .iter()
            .find(|b| matches!(b.role, BlockRole::Location | BlockRole::Mean))
            .map(|b| b.beta.clone())
            .ok_or_else(|| {
                EstimationError::InvalidInput(
                    "Binomial location-scale model missing location/mean block".to_string(),
                )
            })?;
        let beta_noise = unified
            .blocks
            .iter()
            .find(|b| matches!(b.role, BlockRole::Scale))
            .map(|b| b.beta.clone())
            .ok_or_else(|| {
                EstimationError::InvalidInput(
                    "Binomial location-scale model missing scale block".to_string(),
                )
            })?;
        Ok(Self {
            beta_threshold,
            beta_noise,
            covariance: unified.covariance_conditional.clone(),
            inverse_link,
        })
    }

    /// Compute q0 = -eta_t / sigma for each observation, where
    /// eta_t is the threshold linear predictor and sigma = exp(eta_s).
    ///
    /// Returns (q0, sigma, eta_t).
    fn compute_q0_and_sigma(
        &self,
        input: &PredictInput,
    ) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), EstimationError> {
        let eta_t = input.design.dot(&self.beta_threshold) + &input.offset;
        let design_noise = input.design_noise.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput(
                "Binomial location-scale prediction requires noise design matrix".to_string(),
            )
        })?;
        let offset_noise = input
            .offset_noise
            .as_ref()
            .map_or_else(|| Array1::zeros(design_noise.nrows()), |o| o.clone());
        let eta_s = design_noise.dot(&self.beta_noise) + &offset_noise;
        let sigma = eta_s.mapv(f64::exp);
        let q0 = Array1::from_shape_fn(eta_t.len(), |i| -eta_t[i] / sigma[i]);
        Ok((q0, sigma, eta_t))
    }

    /// Apply the inverse link to q0 to get probabilities.
    fn apply_link(&self, q0: &Array1<f64>) -> Result<Array1<f64>, EstimationError> {
        let mut prob = Array1::zeros(q0.len());
        for i in 0..q0.len() {
            let jet = crate::solver::mixture_link::inverse_link_jet_for_inverse_link(
                &self.inverse_link,
                q0[i],
            )?;
            prob[i] = jet.mu.clamp(0.0, 1.0);
        }
        Ok(prob)
    }
}

impl PredictableModel for BinomialLocationScalePredictor {
    fn predict_response(&self, input: &PredictInput) -> Result<PredictResult, EstimationError> {
        let (q0, _sigma, eta_t) = self.compute_q0_and_sigma(input)?;
        let prob = self.apply_link(&q0)?;
        Ok(PredictResult {
            eta: eta_t,
            mean: prob,
        })
    }

    fn predict_with_uncertainty(
        &self,
        input: &PredictInput,
    ) -> Result<PredictionWithSE, EstimationError> {
        let (q0, sigma, eta_t) = self.compute_q0_and_sigma(input)?;
        let prob = self.apply_link(&q0)?;

        let mean_se = if let Some(ref cov) = self.covariance {
            let n = eta_t.len();
            let p_t = self.beta_threshold.len();
            let p_s = self.beta_noise.len();
            let p_total = p_t + p_s;

            if cov.nrows() != p_total || cov.ncols() != p_total {
                return Err(EstimationError::InvalidInput(format!(
                    "covariance dimension mismatch for binomial LS: expected {}x{}, got {}x{}",
                    p_total,
                    p_total,
                    cov.nrows(),
                    cov.ncols()
                )));
            }

            let design_noise = input.design_noise.as_ref().unwrap();
            let mut se = Array1::zeros(n);

            for i in 0..n {
                // Derivative of inverse link at q0: d(prob)/d(q0) = jet.d1
                let jet = crate::solver::mixture_link::inverse_link_jet_for_inverse_link(
                    &self.inverse_link,
                    q0[i],
                )?;
                let dphi = jet.d1;

                // q0 = -eta_t / sigma, sigma = exp(eta_s)
                // d(q0)/d(eta_t) = -1 / sigma
                // d(q0)/d(eta_s) = d(q0)/d(sigma) * d(sigma)/d(eta_s)
                //                = (eta_t / sigma^2) * sigma = eta_t / sigma
                // d(prob)/d(eta_t) = dphi * (-1 / sigma)
                // d(prob)/d(eta_s) = dphi * (eta_t / sigma)
                let dprob_deta_t = dphi * (-1.0 / sigma[i]);
                let dprob_deta_s = dphi * (eta_t[i] / sigma[i]);

                // Build gradient: [dprob/d(beta_t), dprob/d(beta_s)]
                let mut grad = Vec::with_capacity(p_total);
                for j in 0..p_t {
                    grad.push(dprob_deta_t * input.design.get(i, j));
                }
                for j in 0..p_s {
                    grad.push(dprob_deta_s * design_noise.get(i, j));
                }

                let var = quadratic_form(cov, &grad)?;
                se[i] = var.sqrt();
            }
            Some(se)
        } else {
            None
        };

        Ok(PredictionWithSE {
            eta: eta_t,
            mean: prob,
            eta_se: None,
            mean_se,
        })
    }

    fn predict_full_uncertainty(
        &self,
        input: &PredictInput,
        fit: &FitResult,
        options: &PredictUncertaintyOptions,
    ) -> Result<PredictUncertaintyResult, EstimationError> {
        let _ = fit; // not needed for binomial LS intervals
        let pred = self.predict_with_uncertainty(input)?;
        let z = standard_normal_quantile(0.5 + options.confidence_level * 0.5)
            .map_err(EstimationError::InvalidInput)?;

        let mean_se = pred
            .mean_se
            .as_ref()
            .cloned()
            .unwrap_or_else(|| Array1::zeros(pred.mean.len()));

        let mut mean_lower = &pred.mean - &mean_se.mapv(|s| z * s);
        let mut mean_upper = &pred.mean + &mean_se.mapv(|s| z * s);
        // Clamp probabilities to [0, 1].
        mean_lower.mapv_inplace(|v| v.clamp(0.0, 1.0));
        mean_upper.mapv_inplace(|v| v.clamp(0.0, 1.0));

        // For binomial LS, eta intervals on the threshold predictor are not
        // directly meaningful for response-scale inference. Provide the
        // response-scale SE as the primary uncertainty measure.
        Ok(PredictUncertaintyResult {
            eta: pred.eta.clone(),
            mean: pred.mean.clone(),
            eta_standard_error: mean_se.clone(),
            mean_standard_error: mean_se,
            eta_lower: pred.eta.clone(),
            eta_upper: pred.eta,
            mean_lower,
            mean_upper,
            observation_lower: None,
            observation_upper: None,
            covariance_mode_requested: options.covariance_mode,
            covariance_corrected_used: false,
        })
    }

    fn predict_posterior_mean(
        &self,
        input: &PredictInput,
        fit: &FitResult,
    ) -> Result<PredictPosteriorMeanResult, EstimationError> {
        // Full posterior mean with 2D quadrature over (eta_t, eta_s) uncertainty
        // is too complex to extract here. Return point prediction.
        let _ = fit;
        let (q0, sigma, eta_t) = self.compute_q0_and_sigma(input)?;
        let prob = self.apply_link(&q0)?;
        let eta_se = if let Some(ref cov) = self.covariance {
            let n = eta_t.len();
            let p_t = self.beta_threshold.len();
            let p_s = self.beta_noise.len();
            let p_total = p_t + p_s;
            if cov.nrows() == p_total && cov.ncols() == p_total {
                let design_noise = input.design_noise.as_ref().ok_or_else(|| {
                    EstimationError::InvalidInput(
                        "Binomial location-scale posterior mean requires noise design matrix"
                            .to_string(),
                    )
                })?;
                let mut se = Array1::zeros(n);
                for i in 0..n {
                    let jet = crate::solver::mixture_link::inverse_link_jet_for_inverse_link(
                        &self.inverse_link,
                        q0[i],
                    )?;
                    let dphi = jet.d1;
                    let dprob_deta_t = dphi * (-1.0 / sigma[i]);
                    let dprob_deta_s = dphi * (eta_t[i] / sigma[i]);
                    let mut grad = Vec::with_capacity(p_total);
                    for j in 0..p_t {
                        grad.push(dprob_deta_t * input.design.get(i, j));
                    }
                    for j in 0..p_s {
                        grad.push(dprob_deta_s * design_noise.get(i, j));
                    }
                    let var = quadratic_form(cov, &grad)?;
                    se[i] = var.sqrt();
                }
                se
            } else {
                Array1::zeros(n)
            }
        } else {
            Array1::zeros(eta_t.len())
        };
        Ok(PredictPosteriorMeanResult {
            eta: eta_t,
            eta_standard_error: eta_se,
            mean: prob,
        })
    }

    fn n_blocks(&self) -> usize {
        2
    }

    fn block_roles(&self) -> Vec<BlockRole> {
        vec![BlockRole::Location, BlockRole::Scale]
    }
}

/// Survival location-scale predictor: two blocks (threshold + log-sigma).
///
/// Predicts survival probability via:
///   q0 = -eta_threshold / exp(eta_log_sigma)
///   survival_prob = 1 - inverse_link(q0)
///
/// The "design" in `PredictInput` is the threshold design matrix, and
/// "design_noise" is the log-sigma design matrix. The time dimension
/// (x_time_exit) is handled externally and is not part of this predictor.
pub struct SurvivalPredictor {
    pub beta_threshold: Array1<f64>,
    pub beta_log_sigma: Array1<f64>,
    pub covariance: Option<Array2<f64>>,
    pub inverse_link: InverseLink,
}

impl SurvivalPredictor {
    /// Build a `SurvivalPredictor` from a `UnifiedFitResult`, extracting betas
    /// from blocks by role: Location/Mean → beta_threshold, Scale → beta_log_sigma.
    pub(crate) fn from_unified(
        unified: &UnifiedFitResult,
        inverse_link: InverseLink,
    ) -> Result<Self, EstimationError> {
        let beta_threshold = unified
            .blocks
            .iter()
            .find(|b| matches!(b.role, BlockRole::Location | BlockRole::Mean))
            .map(|b| b.beta.clone())
            .ok_or_else(|| {
                EstimationError::InvalidInput(
                    "Survival model missing location/mean (threshold) block".to_string(),
                )
            })?;
        let beta_log_sigma = unified
            .blocks
            .iter()
            .find(|b| matches!(b.role, BlockRole::Scale))
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

    /// Compute q0 = -eta_threshold / sigma and survival_prob = 1 - F(q0).
    fn compute_survival(
        &self,
        eta_threshold: &Array1<f64>,
        eta_log_sigma: &Array1<f64>,
    ) -> Result<Array1<f64>, EstimationError> {
        let n = eta_threshold.len();
        let strategy = strategy_for_family(
            crate::types::LikelihoodFamily::BinomialProbit,
            Some(&self.inverse_link),
        );
        let mut survival_prob = Array1::<f64>::zeros(n);
        for i in 0..n {
            let sigma = eta_log_sigma[i].exp();
            let q0 = -eta_threshold[i] / sigma;
            // survival = 1 - F(q0) = F(-q0)
            let jet = strategy.inverse_link_jet(-q0)?;
            survival_prob[i] = jet.mu.clamp(0.0, 1.0);
        }
        Ok(survival_prob)
    }
}

impl PredictableModel for SurvivalPredictor {
    fn predict_response(&self, input: &PredictInput) -> Result<PredictResult, EstimationError> {
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
        let survival_prob = self.compute_survival(&eta_threshold, &eta_log_sigma)?;
        Ok(PredictResult {
            eta: eta_threshold,
            mean: survival_prob,
        })
    }

    fn predict_with_uncertainty(
        &self,
        input: &PredictInput,
    ) -> Result<PredictionWithSE, EstimationError> {
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
        let survival_prob = self.compute_survival(&eta_threshold, &eta_log_sigma)?;

        let (eta_se, mean_se) = if let Some(ref cov) = self.covariance {
            let n = eta_threshold.len();
            let p_t = self.beta_threshold.len();
            let p_s = self.beta_log_sigma.len();

            // Eta SE for threshold linear predictor only.
            let cov_tt = cov.slice(ndarray::s![..p_t, ..p_t]).to_owned();
            let eta_se = eta_standard_errors_from_design(&input.design, &cov_tt)?;

            // Delta-method SE for survival probability.
            // Convert design matrices to dense for element access.
            let x_t_dense = input.design.to_dense();
            let x_s_dense = design_noise.to_dense();
            let strategy = strategy_for_family(
                crate::types::LikelihoodFamily::BinomialProbit,
                Some(&self.inverse_link),
            );
            let mut mean_se_vec = Array1::<f64>::zeros(n);
            for i in 0..n {
                let sigma = eta_log_sigma[i].exp();
                let q0 = -eta_threshold[i] / sigma;
                // surv = F(-q0), so d(surv)/d(q0) = -F'(-q0) = -phi(-q0)
                let jet = strategy.inverse_link_jet(-q0)?;
                let phi_neg_q0 = jet.d1; // F'(-q0) = phi(-q0)

                // d(q0)/d(eta_t) = -1/sigma
                // d(q0)/d(eta_s) = eta_t/sigma  (since q0 = -eta_t/sigma)
                // d(surv)/d(eta_t) = -phi_neg_q0 * (-1/sigma) = phi_neg_q0 / sigma
                // d(surv)/d(eta_s) = -phi_neg_q0 * (eta_t/sigma)
                let dsurv_deta_t = phi_neg_q0 / sigma;
                let dsurv_deta_s = -phi_neg_q0 * eta_threshold[i] / sigma;

                // Build combined gradient: [dsurv/d(beta_t), dsurv/d(beta_s)]
                let mut grad = Vec::with_capacity(p_t + p_s);
                for j in 0..p_t {
                    grad.push(dsurv_deta_t * x_t_dense[[i, j]]);
                }
                for j in 0..p_s {
                    grad.push(dsurv_deta_s * x_s_dense[[i, j]]);
                }

                let var = quadratic_form(cov, &grad)?;
                mean_se_vec[i] = var.sqrt();
            }
            (Some(eta_se), Some(mean_se_vec))
        } else {
            (None, None)
        };

        Ok(PredictionWithSE {
            eta: eta_threshold,
            mean: survival_prob,
            eta_se,
            mean_se,
        })
    }

    fn predict_full_uncertainty(
        &self,
        input: &PredictInput,
        fit: &FitResult,
        options: &PredictUncertaintyOptions,
    ) -> Result<PredictUncertaintyResult, EstimationError> {
        let _ = fit; // not needed for survival LS intervals
        let pred = self.predict_with_uncertainty(input)?;
        let z = crate::probability::standard_normal_quantile(
            0.5 + options.confidence_level * 0.5,
        )
        .map_err(|e| EstimationError::InvalidInput(e))?;

        let eta_se = pred.eta_se.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput(
                "Survival full uncertainty requires covariance (eta_se unavailable)".to_string(),
            )
        })?;
        let mean_se = pred.mean_se.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput(
                "Survival full uncertainty requires covariance (mean_se unavailable)".to_string(),
            )
        })?;

        let eta_lower = &pred.eta - &eta_se.mapv(|s| z * s);
        let eta_upper = &pred.eta + &eta_se.mapv(|s| z * s);
        let mut mean_lower = &pred.mean - &mean_se.mapv(|s| z * s);
        let mut mean_upper = &pred.mean + &mean_se.mapv(|s| z * s);
        // Clamp survival probabilities to [0, 1].
        mean_lower.mapv_inplace(|v| v.clamp(0.0, 1.0));
        mean_upper.mapv_inplace(|v| v.clamp(0.0, 1.0));

        Ok(PredictUncertaintyResult {
            eta: pred.eta,
            mean: pred.mean,
            eta_standard_error: eta_se.clone(),
            mean_standard_error: mean_se.clone(),
            eta_lower,
            eta_upper,
            mean_lower,
            mean_upper,
            observation_lower: None,
            observation_upper: None,
            covariance_mode_requested: options.covariance_mode,
            covariance_corrected_used: false,
        })
    }

    fn predict_posterior_mean(
        &self,
        input: &PredictInput,
        fit: &FitResult,
    ) -> Result<PredictPosteriorMeanResult, EstimationError> {
        // Exact posterior mean integration not yet implemented for survival.
        // Return point prediction as approximation.
        let _ = fit;
        let result = self.predict_response(input)?;
        let eta_se = if let Some(ref cov) = self.covariance {
            let p_t = self.beta_threshold.len();
            eta_standard_errors_from_design(
                &input.design,
                &cov.slice(ndarray::s![..p_t, ..p_t]).to_owned(),
            )?
        } else {
            Array1::zeros(result.eta.len())
        };
        Ok(PredictPosteriorMeanResult {
            eta: result.eta,
            eta_standard_error: eta_se,
            mean: result.mean,
        })
    }

    fn n_blocks(&self) -> usize {
        2
    }

    fn block_roles(&self) -> Vec<BlockRole> {
        vec![BlockRole::Location, BlockRole::Scale]
    }
}

/// Compute eta standard errors from design matrix and coefficient covariance.
fn eta_standard_errors_from_design(
    x: &DesignMatrix,
    cov: &Array2<f64>,
) -> Result<Array1<f64>, EstimationError> {
    let vars = x.quadratic_form_diag(cov).map_err(|e| {
        EstimationError::InvalidInput(format!("failed to compute linear predictor variance: {e}"))
    })?;
    Ok(vars.mapv(|v| v.max(0.0).sqrt()))
}

/// Delta-method standard errors on the mean scale.
fn delta_method_mean_se(
    eta: &Array1<f64>,
    eta_se: &Array1<f64>,
    strategy: &dyn FamilyStrategy,
) -> Result<Array1<f64>, EstimationError> {
    let n = eta.len();
    let mut mean_se = Array1::<f64>::zeros(n);
    for i in 0..n {
        let jet = strategy.inverse_link_jet(eta[i])?;
        mean_se[i] = (jet.d1 * eta_se[i]).abs();
    }
    Ok(mean_se)
}

pub struct PredictPosteriorMeanResult {
    pub eta: Array1<f64>,
    pub eta_standard_error: Array1<f64>,
    pub mean: Array1<f64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InferenceCovarianceMode {
    /// Use conditional posterior covariance only:
    ///   Var(beta | lambda_hat) ~= H_{rho_hat}^{-1}.
    Conditional,
    /// Prefer first-order smoothing-corrected covariance when available:
    ///   Var(beta) ~= H_{rho_hat}^{-1} + J Var(rho_hat) J^T.
    /// Falls back to conditional if correction is unavailable.
    ConditionalPlusSmoothingPreferred,
    /// Require the first-order smoothing-corrected covariance; error if unavailable.
    ConditionalPlusSmoothingRequired,
}

pub struct PredictUncertaintyOptions {
    /// Central interval level in (0, 1), e.g. 0.95.
    pub confidence_level: f64,
    /// Covariance mode used for eta/mean intervals.
    pub covariance_mode: InferenceCovarianceMode,
    /// Mean-scale interval construction method.
    pub mean_interval_method: MeanIntervalMethod,
    /// For Gaussian identity, also return observation intervals using
    /// Var(y_new | x) = Var(eta_hat) + sigma^2.
    pub includeobservation_interval: bool,
}

impl Default for PredictUncertaintyOptions {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
            covariance_mode: InferenceCovarianceMode::ConditionalPlusSmoothingPreferred,
            mean_interval_method: MeanIntervalMethod::TransformEta,
            includeobservation_interval: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MeanIntervalMethod {
    /// Interval on mean scale from delta-method SEs.
    Delta,
    /// Transform eta interval endpoints through inverse link.
    /// This is usually better behaved for nonlinear links.
    TransformEta,
}

pub struct PredictUncertaintyResult {
    pub eta: Array1<f64>,
    pub mean: Array1<f64>,
    pub eta_standard_error: Array1<f64>,
    pub mean_standard_error: Array1<f64>,
    pub eta_lower: Array1<f64>,
    pub eta_upper: Array1<f64>,
    pub mean_lower: Array1<f64>,
    pub mean_upper: Array1<f64>,
    /// Optional Gaussian observation interval bounds.
    pub observation_lower: Option<Array1<f64>>,
    pub observation_upper: Option<Array1<f64>>,
    /// Covariance mode requested by caller.
    pub covariance_mode_requested: InferenceCovarianceMode,
    /// True if smoothing-corrected covariance was used.
    pub covariance_corrected_used: bool,
}

pub struct CoefficientUncertaintyResult {
    pub estimate: Array1<f64>,
    pub standard_error: Array1<f64>,
    pub lower: Array1<f64>,
    pub upper: Array1<f64>,
    pub corrected: bool,
    pub covariance_mode_requested: InferenceCovarianceMode,
}

/// Generic engine prediction for external designs.
/// This API is domain-agnostic: callers provide only design matrix, coefficients, offset, and family.
pub fn predict_gam<X>(
    x: X,
    beta: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    family: crate::types::LikelihoodFamily,
) -> Result<PredictResult, EstimationError>
where
    X: Into<DesignMatrix>,
{
    let x = x.into();
    if let Some(message) =
        predict_gam_dimension_mismatch_message(x.nrows(), x.ncols(), beta.len(), offset.len())
    {
        return Err(EstimationError::InvalidInput(message));
    }
    if matches!(family, crate::types::LikelihoodFamily::RoystonParmar) {
        return Err(EstimationError::InvalidInput(
            "predict_gam does not support RoystonParmar; use survival prediction APIs".to_string(),
        ));
    }

    let mut eta = x.matrixvectormultiply(&beta.to_owned());
    eta += &offset;

    let mean = apply_family_inverse_link(&eta, family, None)?;

    Ok(PredictResult { eta, mean })
}

/// Nonlinear posterior-mean prediction with coefficient uncertainty propagation.
///
/// For nonlinear links, returns E[g^{-1}(eta_tilde)] where eta_tilde ~ N(eta_hat, se_eta^2).
/// For Gaussian identity, this equals the standard plug-in mean.
pub fn predict_gam_posterior_mean<X>(
    x: X,
    beta: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    family: crate::types::LikelihoodFamily,
    covariance: ArrayView2<'_, f64>,
) -> Result<PredictPosteriorMeanResult, EstimationError>
where
    X: Into<DesignMatrix>,
{
    // Posterior mean prediction under Gaussian coefficient uncertainty.
    //
    // For each row x_i we first propagate coefficient covariance to the linear
    // predictor:
    //
    //   eta_i       = x_i^T beta
    //   Var(eta_i)  = x_i^T Var(beta) x_i.
    //
    // For nonlinear links the desired prediction is not the plug-in
    // g^{-1}(eta_i), but the Gaussian-uncertainty average
    //
    //   E[g^{-1}(Eta_i)],   Eta_i ~ N(eta_i, Var(eta_i)).
    //
    // The key design choice is that prediction uses the same integrated-
    // expectation dispatcher as integrated PIRLS. That keeps fitting-time and
    // prediction-time uncertainty propagation on the same mathematical object:
    // if a link has an exact closed form or guarded special-function backend,
    // both paths use it; numerical failure is propagated rather than silently
    // degrading to a plug-in mean.
    let x = x.into();
    if x.ncols() != beta.len() {
        return Err(EstimationError::InvalidInput(format!(
            "predict_gam_posterior_mean dimension mismatch: X has {} columns but beta has length {}",
            x.ncols(),
            beta.len()
        )));
    }
    if x.nrows() != offset.len() {
        return Err(EstimationError::InvalidInput(format!(
            "predict_gam_posterior_mean dimension mismatch: X has {} rows but offset has length {}",
            x.nrows(),
            offset.len()
        )));
    }
    if covariance.nrows() != beta.len() || covariance.ncols() != beta.len() {
        return Err(EstimationError::InvalidInput(format!(
            "predict_gam_posterior_mean covariance dimension mismatch: expected {}x{}, got {}x{}",
            beta.len(),
            beta.len(),
            covariance.nrows(),
            covariance.ncols()
        )));
    }
    if matches!(family, crate::types::LikelihoodFamily::RoystonParmar) {
        return Err(EstimationError::InvalidInput(
            "predict_gam_posterior_mean does not support RoystonParmar; use survival prediction APIs"
                .to_string(),
        ));
    }

    let mut eta = x.matrixvectormultiply(&beta.to_owned());
    eta += &offset;

    let etavar = linear_predictorvariance(&x, &covariance.to_owned())?;
    let eta_standard_error = etavar.mapv(|v| v.max(0.0).sqrt());
    let quadctx = crate::quadrature::QuadratureContext::new();
    let strategy = strategy_for_family(family, None);
    let means: Result<Vec<f64>, EstimationError> = eta
        .iter()
        .zip(eta_standard_error.iter())
        .map(|(&e, &se)| strategy.posterior_mean(&quadctx, e, se))
        .collect();
    let mean = Array1::from_vec(means?);

    Ok(PredictPosteriorMeanResult {
        eta,
        eta_standard_error,
        mean,
    })
}

/// Nonlinear posterior-mean prediction with link-state support for SAS/mixture families.
///
/// This mirrors `predict_gam_posterior_mean`, but also uses `fit` metadata for
/// link families that require extra state (`BinomialSas`, `BinomialMixture`).
pub fn predict_gam_posterior_meanwith_fit<X>(
    x: X,
    beta: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    family: crate::types::LikelihoodFamily,
    covariance: ArrayView2<'_, f64>,
    fit: &FitResult,
) -> Result<PredictPosteriorMeanResult, EstimationError>
where
    X: Into<DesignMatrix>,
{
    let x = x.into();
    if x.ncols() != beta.len() {
        return Err(EstimationError::InvalidInput(format!(
            "predict_gam_posterior_meanwith_fit dimension mismatch: X has {} columns but beta has length {}",
            x.ncols(),
            beta.len()
        )));
    }
    if x.nrows() != offset.len() {
        return Err(EstimationError::InvalidInput(format!(
            "predict_gam_posterior_meanwith_fit dimension mismatch: X has {} rows but offset has length {}",
            x.nrows(),
            offset.len()
        )));
    }
    if covariance.nrows() != beta.len() || covariance.ncols() != beta.len() {
        return Err(EstimationError::InvalidInput(format!(
            "predict_gam_posterior_meanwith_fit covariance dimension mismatch: expected {}x{}, got {}x{}",
            beta.len(),
            beta.len(),
            covariance.nrows(),
            covariance.ncols()
        )));
    }
    if matches!(family, crate::types::LikelihoodFamily::RoystonParmar) {
        return Err(EstimationError::InvalidInput(
            "predict_gam_posterior_meanwith_fit does not support RoystonParmar; use survival prediction APIs"
                .to_string(),
        ));
    }

    let mut eta = x.matrixvectormultiply(&beta.to_owned());
    eta += &offset;
    let etavar = linear_predictorvariance(&x, &covariance.to_owned())?;
    let eta_standard_error = etavar.mapv(|v| v.max(0.0).sqrt());
    let quadctx = crate::quadrature::QuadratureContext::new();
    let strategy = strategy_from_fit(family, fit)?;
    let means: Result<Vec<f64>, EstimationError> = eta
        .iter()
        .zip(eta_standard_error.iter())
        .map(|(&e, &se)| strategy.posterior_mean(&quadctx, e, se))
        .collect();
    let mean = Array1::from_vec(means?);

    Ok(PredictPosteriorMeanResult {
        eta,
        eta_standard_error,
        mean,
    })
}

/// Prediction with coefficient uncertainty propagation.
///
/// The linear predictor variance uses:
/// Var(η_i) = x_i^T Var(β) x_i
///
/// Mean-scale SEs are delta-method approximations:
/// Var(μ_i) ≈ (dμ/dη)^2 Var(η_i)
///
/// Math note (logit family, Gaussian η posterior):
///
/// If η_i | D ≈ N(m_i, v_i), then the exact posterior predictive mean on the
/// probability scale is the logistic-normal integral
///
///   E[sigmoid(η_i)] = ∫ sigmoid(x) N(x; m_i, v_i) dx.
///
/// This does not reduce to an elementary closed form. Two exact representations
/// often used in the literature are:
///
/// 1) Theta/Appell-Lerch style representations (via Poisson summation / Mordell integrals).
/// 2) Absolutely convergent complex-error-function (Faddeeva) series obtained from
///    partial-fraction expansions of tanh/logistic.
///
/// A practical exact series form is:
///
///   E[sigmoid(η)] = 1/2
///                   - (sqrt(2π)/σ) * Σ_{n>=1} Im[ w((i a_n - μ)/(sqrt(2)σ)) ],
///   where a_n = (2n-1)π, σ = sqrt(v), and w is the Faddeeva function
///   w(z) = exp(-z^2) erfc(-i z).
///
/// The formulas above define the exact logistic-normal target moments under
/// Gaussian η uncertainty.
///
/// CLogLog note (exact target):
/// If p = 1 - exp(-exp(η)) and η ~ N(μ,σ²), then
///   E[p] = 1 - I(1),  E[p²] = 1 - 2I(1) + I(2),  Var(p) = I(2) - I(1)²
/// where I(λ) = E[exp(-λ exp(η))] is the lognormal Laplace transform.
/// This identity is exact, and highlights that the moments are determined by
/// the lognormal Laplace transform values at λ=1 and λ=2.
///
/// Exact analytic representation (Mellin-Barnes) for I(λ):
///   I(λ) = (1/(2πi)) ∫_{c-i∞}^{c+i∞} Γ(z) λ^{-z} exp(-μ z + 0.5 σ² z²) dz, c>0.
/// This Mellin-Barnes integral is mathematically exact.
pub fn predict_gamwith_uncertainty<X>(
    x: X,
    beta: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    family: crate::types::LikelihoodFamily,
    fit: &FitResult,
    options: &PredictUncertaintyOptions,
) -> Result<PredictUncertaintyResult, EstimationError>
where
    X: Into<DesignMatrix>,
{
    let x = x.into();
    if x.ncols() != beta.len() {
        return Err(EstimationError::InvalidInput(format!(
            "predict_gamwith_uncertainty dimension mismatch: X has {} columns but beta has length {}",
            x.ncols(),
            beta.len()
        )));
    }
    if x.nrows() != offset.len() {
        return Err(EstimationError::InvalidInput(format!(
            "predict_gamwith_uncertainty dimension mismatch: X has {} rows but offset has length {}",
            x.nrows(),
            offset.len()
        )));
    }
    if !(options.confidence_level.is_finite()
        && options.confidence_level > 0.0
        && options.confidence_level < 1.0)
    {
        return Err(EstimationError::InvalidInput(format!(
            "confidence_level must be in (0,1), got {}",
            options.confidence_level
        )));
    }

    let requested_mode = options.covariance_mode;
    // Covariance selection corresponds to approximation order:
    // - Conditional: uses only A(mu) = Hmu^{-1}
    // - Corrected: adds first-order Var(b(rho)) term J V_rho J^T
    let (cov, covariance_corrected_used) = match requested_mode {
        InferenceCovarianceMode::Conditional => (
            fit.beta_covariance().ok_or_else(|| {
                EstimationError::InvalidInput(
                    "fit result does not contain conditional covariance".to_string(),
                )
            })?,
            false,
        ),
        InferenceCovarianceMode::ConditionalPlusSmoothingPreferred => {
            if let Some(cov_corr) = fit.beta_covariance_corrected() {
                (cov_corr, true)
            } else if let Some(cov_base) = fit.beta_covariance() {
                (cov_base, false)
            } else {
                return Err(EstimationError::InvalidInput(
                    "fit result does not contain a usable posterior covariance".to_string(),
                ));
            }
        }
        InferenceCovarianceMode::ConditionalPlusSmoothingRequired => (
            fit.beta_covariance_corrected().ok_or_else(|| {
                EstimationError::InvalidInput(
                    "fit result does not contain smoothing-corrected covariance".to_string(),
                )
            })?,
            true,
        ),
    };

    if cov.nrows() != beta.len() || cov.ncols() != beta.len() {
        return Err(EstimationError::InvalidInput(format!(
            "covariance dimension mismatch: expected {}x{}, got {}x{}",
            beta.len(),
            beta.len(),
            cov.nrows(),
            cov.ncols()
        )));
    }

    let mut eta = x.matrixvectormultiply(&beta.to_owned());
    eta += &offset;
    let fitted_link_state = fit.fitted_link_state(family).ok();
    let mixture_state = match fitted_link_state.as_ref() {
        Some(FittedLinkState::Mixture { state, .. }) => Some(state.clone()),
        _ => None,
    };
    let sas_state = match fitted_link_state.as_ref() {
        Some(FittedLinkState::Sas { state, .. })
        | Some(FittedLinkState::BetaLogistic { state, .. }) => Some(*state),
        _ => None,
    };
    let link_kind = match fitted_link_state.as_ref() {
        Some(FittedLinkState::Standard(link)) => Some(InverseLink::Standard(*link)),
        Some(FittedLinkState::Sas { state, .. }) => Some(InverseLink::Sas(*state)),
        Some(FittedLinkState::BetaLogistic { state, .. }) => {
            Some(InverseLink::BetaLogistic(*state))
        }
        Some(FittedLinkState::Mixture { state, .. }) => Some(InverseLink::Mixture(state.clone())),
        None => None,
    };
    let strategy = strategy_for_family(family, link_kind.as_ref());
    let mean = apply_family_inverse_link(&eta, family, link_kind.as_ref())?;

    let etavar = linear_predictorvariance(&x, cov)?;
    let eta_standard_error = etavar.mapv(|v| v.max(0.0).sqrt());

    let z = standard_normal_quantile(0.5 + 0.5 * options.confidence_level)
        .map_err(EstimationError::InvalidInput)?;
    let eta_lower = &eta - &eta_standard_error.mapv(|s| z * s);
    let eta_upper = &eta + &eta_standard_error.mapv(|s| z * s);
    let quadctx = crate::quadrature::QuadratureContext::new();

    // Derivative of inverse link g^{-1}(η) used for delta-method:
    //   Var(μ_i) ≈ [d g^{-1}(η_i)/dη]^2 Var(η_i).
    //
    // For logit:
    //   g^{-1}(η)=sigmoid(η), dμ/dη=μ(1-μ).
    // If η itself is uncertain (η ~ N(m,v)), the exact predictive mean is
    // E[sigmoid(η)] (logistic-normal integral) as documented above.
    //
    // For cloglog:
    //   g^{-1}(η)=1-exp(-exp(η)), dμ/dη=exp(η)exp(-exp(η)).
    // With uncertain η the exact moments can be written via I(λ)=E[exp(-λexp(η))],
    // and:
    //   E[μ]   = 1 - I(1),
    //   E[μ²]  = 1 - 2I(1) + I(2),
    //   Var(μ) = I(2) - I(1)^2.
    // These identities characterize the exact cloglog moments under Gaussian η uncertainty.
    let mut mean_standard_error = Array1::<f64>::zeros(eta.len());
    let mut mix_partials = mixture_state
        .as_ref()
        .map(|state| {
            vec![
                InverseLinkJet {
                    mu: 0.0,
                    d1: 0.0,
                    d2: 0.0,
                    d3: 0.0,
                };
                state.rho.len()
            ]
        })
        .unwrap_or_default();
    for i in 0..eta.len() {
        let se_i = etavar[i].max(0.0).sqrt();
        let (_, mut meanvar) = strategy.posterior_meanvariance(&quadctx, eta[i], se_i)?;
        if matches!(family, crate::types::LikelihoodFamily::BinomialSas)
            && let Some(cov_theta) = fitted_link_state.as_ref().and_then(|s| match s {
                FittedLinkState::Sas { covariance, .. } => covariance.as_ref(),
                _ => None,
            })
        {
            let sas = sas_state.ok_or_else(|| {
                EstimationError::InvalidInput(
                    "BinomialSas uncertainty requires fitted sas_epsilon/sas_log_delta".to_string(),
                )
            })?;
            let jets = sas_inverse_link_jetwith_param_partials(eta[i], sas.epsilon, sas.log_delta);
            let g = [jets.djet_depsilon.mu, jets.djet_dlog_delta.mu];
            meanvar += quadratic_form(cov_theta, &g)?;
        }
        if matches!(family, crate::types::LikelihoodFamily::BinomialBetaLogistic)
            && let Some(cov_theta) = fitted_link_state.as_ref().and_then(|s| match s {
                FittedLinkState::BetaLogistic { covariance, .. } => covariance.as_ref(),
                _ => None,
            })
        {
            let sas = sas_state.ok_or_else(|| {
                EstimationError::InvalidInput(
                    "BinomialBetaLogistic uncertainty requires fitted parameters".to_string(),
                )
            })?;
            let jets = beta_logistic_inverse_link_jetwith_param_partials(
                eta[i],
                sas.log_delta,
                sas.epsilon,
            );
            let g = [jets.djet_depsilon.mu, jets.djet_dlog_delta.mu];
            meanvar += quadratic_form(cov_theta, &g)?;
        }
        if matches!(family, crate::types::LikelihoodFamily::BinomialMixture)
            && let Some(cov_theta) = fitted_link_state.as_ref().and_then(|s| match s {
                FittedLinkState::Mixture { covariance, .. } => covariance.as_ref(),
                _ => None,
            })
            && let Some(state) = mixture_state.as_ref()
        {
            if mix_partials.len() != state.rho.len() {
                mix_partials = vec![
                    InverseLinkJet {
                        mu: 0.0,
                        d1: 0.0,
                        d2: 0.0,
                        d3: 0.0,
                    };
                    state.rho.len()
                ];
            }
            let _ =
                mixture_inverse_link_jetwith_rho_partials_into(state, eta[i], &mut mix_partials);
            meanvar += quadratic_form_from_jetmu(cov_theta, &mix_partials)?;
        }
        mean_standard_error[i] = meanvar.max(0.0).sqrt();
    }

    let (mut mean_lower, mut mean_upper) = match options.mean_interval_method {
        MeanIntervalMethod::Delta => (
            &mean - &mean_standard_error.mapv(|s| z * s),
            &mean + &mean_standard_error.mapv(|s| z * s),
        ),
        MeanIntervalMethod::TransformEta => (
            apply_family_inverse_link(&eta_lower, family, link_kind.as_ref())?,
            apply_family_inverse_link(&eta_upper, family, link_kind.as_ref())?,
        ),
    };

    if matches!(
        family,
        crate::types::LikelihoodFamily::BinomialLogit
            | crate::types::LikelihoodFamily::BinomialProbit
            | crate::types::LikelihoodFamily::BinomialCLogLog
            | crate::types::LikelihoodFamily::BinomialSas
            | crate::types::LikelihoodFamily::BinomialBetaLogistic
            | crate::types::LikelihoodFamily::BinomialMixture
    ) {
        mean_lower.mapv_inplace(|v| v.clamp(0.0, 1.0));
        mean_upper.mapv_inplace(|v| v.clamp(0.0, 1.0));
    }

    let (observation_lower, observation_upper) = if options.includeobservation_interval
        && matches!(family, crate::types::LikelihoodFamily::GaussianIdentity)
    {
        let obsvar = fit.standard_deviation.max(0.0).powi(2);
        let obs_se = etavar.mapv(|v| (v + obsvar).max(0.0).sqrt());
        let lower = &eta - &obs_se.mapv(|s| z * s);
        let upper = &eta + &obs_se.mapv(|s| z * s);
        (Some(lower), Some(upper))
    } else {
        (None, None)
    };

    Ok(PredictUncertaintyResult {
        eta,
        mean,
        eta_standard_error,
        mean_standard_error,
        eta_lower,
        eta_upper,
        mean_lower,
        mean_upper,
        observation_lower,
        observation_upper,
        covariance_mode_requested: requested_mode,
        covariance_corrected_used,
    })
}

/// Coefficient-level uncertainty and confidence intervals.
pub fn coefficient_uncertainty(
    fit: &FitResult,
    confidence_level: f64,
    covariance_mode: InferenceCovarianceMode,
) -> Result<CoefficientUncertaintyResult, EstimationError> {
    coefficient_uncertaintywith_mode(fit, confidence_level, covariance_mode)
}

/// Coefficient-level uncertainty and confidence intervals with explicit covariance mode.
pub fn coefficient_uncertaintywith_mode(
    fit: &FitResult,
    confidence_level: f64,
    covariance_mode: InferenceCovarianceMode,
) -> Result<CoefficientUncertaintyResult, EstimationError> {
    if !(confidence_level.is_finite() && confidence_level > 0.0 && confidence_level < 1.0) {
        return Err(EstimationError::InvalidInput(format!(
            "confidence_level must be in (0,1), got {}",
            confidence_level
        )));
    }
    // Coefficient SEs are extracted from either:
    // - conditional covariance H^{-1}, or
    // - first-order corrected covariance H^{-1} + J V_rho J^T.
    let (se, corrected) = match covariance_mode {
        InferenceCovarianceMode::Conditional => (
            fit.beta_standard_errors().cloned().ok_or_else(|| {
                EstimationError::InvalidInput(
                    "fit result does not contain conditional coefficient standard errors"
                        .to_string(),
                )
            })?,
            false,
        ),
        InferenceCovarianceMode::ConditionalPlusSmoothingPreferred => {
            if let Some(se_corr) = fit.beta_standard_errors_corrected() {
                (se_corr.clone(), true)
            } else if let Some(se_base) = fit.beta_standard_errors() {
                (se_base.clone(), false)
            } else {
                return Err(EstimationError::InvalidInput(
                    "fit result does not contain coefficient standard errors".to_string(),
                ));
            }
        }
        InferenceCovarianceMode::ConditionalPlusSmoothingRequired => (
            fit.beta_standard_errors_corrected()
                .cloned()
                .ok_or_else(|| {
                    EstimationError::InvalidInput(
                        "fit result does not contain smoothing-corrected coefficient standard errors"
                            .to_string(),
                    )
                })?,
            true,
        ),
    };

    if se.len() != fit.beta.len() {
        return Err(EstimationError::InvalidInput(format!(
            "standard error length mismatch: beta has {}, se has {}",
            fit.beta.len(),
            se.len()
        )));
    }

    let z = standard_normal_quantile(0.5 + 0.5 * confidence_level)
        .map_err(EstimationError::InvalidInput)?;
    let lower = &fit.beta - &se.mapv(|s| z * s);
    let upper = &fit.beta + &se.mapv(|s| z * s);
    Ok(CoefficientUncertaintyResult {
        estimate: fit.beta.clone(),
        standard_error: se,
        lower,
        upper,
        corrected,
        covariance_mode_requested: covariance_mode,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::LinkFunction;
    use ndarray::{Array2, array};

    #[test]
    fn predict_posterior_mean_probit_matches_closed_form_reference() {
        let x = array![[1.0], [1.0]];
        let beta = array![0.7];
        let offset = array![0.0, 0.0];
        let covariance = Array2::from_diag(&array![0.25]);
        let out = predict_gam_posterior_mean(
            x,
            beta.view(),
            offset.view(),
            crate::types::LikelihoodFamily::BinomialProbit,
            covariance.view(),
        )
        .expect("predict posterior mean");
        let expected = crate::quadrature::probit_posterior_meanwith_deriv_exact(0.7, 0.5).mean;
        assert!((out.mean[0] - expected).abs() <= 1e-12);
        assert!((out.mean[1] - expected).abs() <= 1e-12);
    }

    #[test]
    fn predict_posterior_mean_logit_uses_integrated_dispatch() {
        let x = array![[1.0], [1.0]];
        let beta = array![0.4];
        let offset = array![0.0, 0.0];
        let covariance = Array2::from_diag(&array![0.16]);
        let out = predict_gam_posterior_mean(
            x,
            beta.view(),
            offset.view(),
            crate::types::LikelihoodFamily::BinomialLogit,
            covariance.view(),
        )
        .expect("predict posterior mean");
        let quadctx = crate::quadrature::QuadratureContext::new();
        let expected = crate::quadrature::integrated_inverse_link_mean_and_derivative(
            &quadctx,
            LinkFunction::Logit,
            0.4,
            0.4,
        )
        .expect("logit integrated inverse-link moments should evaluate")
        .mean;
        assert!((out.mean[0] - expected).abs() <= 1e-12);
        assert!((out.mean[1] - expected).abs() <= 1e-12);
    }
}
