use crate::estimate::{EstimationError, FittedLinkState, UnifiedFitResult};
use crate::families::lognormal_kernel::latent_cloglog_inverse_link_jet;
use crate::inference::generative::NoiseModel;
use crate::mixture_link::{InverseLinkJet, inverse_link_jet_for_family, mixture_inverse_link_jet};
use crate::quadrature::{
    IntegratedMomentsJet, QuadratureContext, cloglog_posterior_meanvariance,
    integrated_family_moments_jet, integrated_inverse_link_jetwith_state,
    integrated_inverse_link_mean_and_derivative, logit_posterior_meanvariance,
    normal_expectation_1d_adaptive, normal_expectation_1d_adaptive_pair,
    probit_posterior_meanvariance, survival_posterior_mean, survival_posterior_meanvariance,
};
use crate::types::{
    InverseLink, LikelihoodSpec, LinkFunction, ResponseFamily, StandardLink, is_valid_tweedie_power,
};
use ndarray::{Array1, ArrayView1};

/// Runtime family behavior carrier built from a `LikelihoodSpec` (response
/// distribution + parameterized inverse-link).
pub trait FamilyStrategy: std::fmt::Debug + Send + Sync {
    fn name(&self) -> &'static str;

    fn family(&self) -> LikelihoodSpec;

    fn link_function(&self) -> LinkFunction;

    fn inverse_link(&self, eta: f64) -> Result<f64, EstimationError>;

    fn inverse_link_array(&self, eta: ArrayView1<'_, f64>) -> Result<Array1<f64>, EstimationError>;

    fn inverse_link_jet(&self, eta: f64) -> Result<InverseLinkJet, EstimationError>;

    fn posterior_mean(
        &self,
        quadctx: &QuadratureContext,
        eta: f64,
        se_eta: f64,
    ) -> Result<f64, EstimationError>;

    fn posterior_meanvariance(
        &self,
        quadctx: &QuadratureContext,
        eta: f64,
        se_eta: f64,
    ) -> Result<(f64, f64), EstimationError>;

    fn simulate_noise(
        &self,
        mean: &Array1<f64>,
        gaussian_scale: Option<f64>,
    ) -> Result<NoiseModel, EstimationError>;

    fn integrated_moments(
        &self,
        quadctx: &QuadratureContext,
        eta: f64,
        se_eta: f64,
    ) -> Result<IntegratedMomentsJet, EstimationError>;
}

/// Default `FamilyStrategy` implementation: stores a `LikelihoodSpec`
/// (response distribution + parameterized inverse-link state).  Trait
/// methods dispatch on `spec.response` / `spec.link`: `inverse_link_*`
/// routes through the parameterized link; `posterior_*` integrates
/// `p(η) | η ~ N(eta, se_eta²)` via the appropriate exact / quadrature
/// path; `simulate_noise` extracts the dispersion parameter from
/// `gaussian_scale` (or rejects when the family needs one and it is
/// missing).
#[derive(Clone, Debug)]
pub struct ResolvedFamilyStrategy {
    spec: LikelihoodSpec,
}

/// Build a `LikelihoodSpec` from a response/link spec plus an optional
/// fitted `InverseLink` state. The supplied `InverseLink` is preferred;
/// when absent the original spec is retained unchanged.
fn spec_from_family(family: LikelihoodSpec, inverse_link: Option<&InverseLink>) -> LikelihoodSpec {
    if let Some(link) = inverse_link {
        return LikelihoodSpec {
            response: family.response,
            link: link.clone(),
        };
    }
    family
}

/// Construct a `ResolvedFamilyStrategy` from a family identifier and an
/// optional inverse-link state (cloned).  No validation is performed —
/// the strategy methods will return `EstimationError::InvalidInput`
/// later if they need state that this constructor did not supply.
#[inline]
pub fn strategy_for_family(
    family: LikelihoodSpec,
    inverse_link: Option<&InverseLink>,
) -> ResolvedFamilyStrategy {
    ResolvedFamilyStrategy {
        spec: spec_from_family(family, inverse_link),
    }
}

/// Construct a `ResolvedFamilyStrategy` directly from a `LikelihoodSpec`.
/// Mirrors `strategy_for_family` but takes the modern (response, link)
/// representation without any legacy-enum round-trip. The spec is cloned
/// into the resulting strategy.
#[inline]
pub fn strategy_for_spec(spec: &LikelihoodSpec) -> ResolvedFamilyStrategy {
    ResolvedFamilyStrategy { spec: spec.clone() }
}

/// Build a `ResolvedFamilyStrategy` from a fitted result, lifting the
/// fitted link state (`FittedLinkState`) into an `InverseLink` variant
/// suitable for predict-time evaluation.  Returns an error when the
/// recorded link state and the supplied `family` are mutually
/// inconsistent (propagated from `fit.fitted_link_state`).
pub fn strategy_from_fit(
    family: &LikelihoodSpec,
    fit: &UnifiedFitResult,
) -> Result<ResolvedFamilyStrategy, EstimationError> {
    let inverse_link = match fit.fitted_link_state(family)? {
        FittedLinkState::Standard(Some(link)) => Some(InverseLink::Standard(link)),
        FittedLinkState::Standard(None) => None,
        FittedLinkState::LatentCLogLog { state } => Some(InverseLink::LatentCLogLog(state)),
        FittedLinkState::Sas { state, .. } => Some(InverseLink::Sas(state)),
        FittedLinkState::BetaLogistic { state, .. } => Some(InverseLink::BetaLogistic(state)),
        FittedLinkState::Mixture { state, .. } => Some(InverseLink::Mixture(state)),
    };
    let spec = if let Some(link) = inverse_link {
        LikelihoodSpec::new(family.response.clone(), link)
    } else {
        family.clone()
    };
    Ok(strategy_for_spec(&spec))
}

impl ResolvedFamilyStrategy {
    #[inline]
    fn mixture_state(&self) -> Option<&crate::types::MixtureLinkState> {
        self.spec.link.mixture_state()
    }

    #[inline]
    fn sas_state(&self) -> Option<&crate::types::SasLinkState> {
        self.spec.link.sas_state()
    }

    #[inline]
    fn latent_cloglog_state(&self) -> Option<&crate::types::LatentCLogLogState> {
        self.spec.link.latent_cloglog_state()
    }

    #[inline]
    fn require_latent_cloglog_state(
        &self,
    ) -> Result<&crate::types::LatentCLogLogState, EstimationError> {
        self.latent_cloglog_state()
            .ok_or_else(|| missing_state(&self.spec, "latent cloglog"))
    }

    #[inline]
    fn require_sas_state(&self) -> Result<&crate::types::SasLinkState, EstimationError> {
        self.sas_state()
            .ok_or_else(|| missing_state(&self.spec, "SAS link"))
    }

    #[inline]
    fn require_mixture_state(&self) -> Result<&crate::types::MixtureLinkState, EstimationError> {
        self.mixture_state()
            .ok_or_else(|| missing_state(&self.spec, "mixture link"))
    }
}

#[cold]
fn missing_state(spec: &LikelihoodSpec, what: &str) -> EstimationError {
    EstimationError::InvalidInput(format!(
        "{} requires fitted {} state",
        spec.pretty_name(),
        what
    ))
}

/// Compute `(mean, variance)` of a Bernoulli probability `p(η)` integrated
/// against `η ~ N(eta, se_eta²)` via the joint `(p, p²)` adaptive Gauss-Hermite
/// rule. Both SAS and beta-logistic posterior-mean-variance branches share
/// this exact shape — only the probability kernel differs.
#[inline]
fn posterior_mv_from_prob_kernel<F>(
    quadctx: &QuadratureContext,
    eta: f64,
    se_eta: f64,
    prob: F,
) -> (f64, f64)
where
    F: Fn(f64) -> f64,
{
    let (m1, m2) = normal_expectation_1d_adaptive_pair(quadctx, eta, se_eta, |x| {
        let p = prob(x);
        (p, p * p)
    });
    (m1, (m2 - m1 * m1).max(0.0))
}

#[inline]
fn require_noise_parameter(
    spec: &LikelihoodSpec,
    parameter_name: &str,
    value: Option<f64>,
) -> Result<f64, EstimationError> {
    let value = value.ok_or_else(|| {
        EstimationError::InvalidInput(format!(
            "{} generative sampling requires fitted {parameter_name}",
            spec.pretty_name()
        ))
    })?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(EstimationError::InvalidInput(format!(
            "{} generative sampling requires finite {parameter_name}; got {value}",
            spec.pretty_name()
        )))
    }
}

#[inline]
fn require_positive_noise_parameter(
    spec: &LikelihoodSpec,
    parameter_name: &str,
    value: Option<f64>,
) -> Result<f64, EstimationError> {
    let value = require_noise_parameter(spec, parameter_name, value)?;
    if value > 0.0 {
        Ok(value)
    } else {
        Err(EstimationError::InvalidInput(format!(
            "{} generative sampling requires {parameter_name} > 0; got {value}",
            spec.pretty_name()
        )))
    }
}

impl FamilyStrategy for ResolvedFamilyStrategy {
    fn name(&self) -> &'static str {
        self.spec.name()
    }

    fn family(&self) -> LikelihoodSpec {
        self.spec.clone()
    }

    fn link_function(&self) -> LinkFunction {
        self.spec.link.link_function()
    }

    fn inverse_link(&self, eta: f64) -> Result<f64, EstimationError> {
        self.inverse_link_jet(eta).map(|jet| jet.mu)
    }

    fn inverse_link_array(&self, eta: ArrayView1<'_, f64>) -> Result<Array1<f64>, EstimationError> {
        let mut out = Array1::<f64>::zeros(eta.len());
        for i in 0..eta.len() {
            out[i] = self.inverse_link(eta[i])?;
        }
        Ok(out)
    }

    fn inverse_link_jet(&self, eta: f64) -> Result<InverseLinkJet, EstimationError> {
        inverse_link_jet_for_family(&self.spec, eta)
    }

    fn posterior_mean(
        &self,
        quadctx: &QuadratureContext,
        eta: f64,
        se_eta: f64,
    ) -> Result<f64, EstimationError> {
        match (&self.spec.response, &self.spec.link) {
            (ResponseFamily::Gaussian, _) => Ok(eta),
            (ResponseFamily::Binomial, InverseLink::Standard(_)) => {
                integrated_inverse_link_mean_and_derivative(
                    quadctx,
                    self.link_function(),
                    eta,
                    se_eta,
                )
                .map(|v| v.mean)
            }
            (ResponseFamily::Binomial, InverseLink::LatentCLogLog(_)) => {
                let state = self.require_latent_cloglog_state()?;
                latent_cloglog_inverse_link_jet(quadctx, eta, se_eta.hypot(state.latent_sd))
                    .map(|v| v.mean)
            }
            (ResponseFamily::Binomial, InverseLink::Sas(_))
            | (ResponseFamily::Binomial, InverseLink::BetaLogistic(_)) => {
                integrated_inverse_link_jetwith_state(
                    quadctx,
                    self.link_function(),
                    eta,
                    se_eta,
                    self.mixture_state(),
                    self.sas_state(),
                )
                .map(|v| v.mean)
            }
            (ResponseFamily::Binomial, InverseLink::Mixture(_)) => {
                let state = self.require_mixture_state()?;
                integrated_family_moments_jet(
                    quadctx,
                    &LikelihoodSpec::binomial_mixture(state.clone()),
                    eta,
                    se_eta,
                )
                .map(|v| v.mean)
            }
            (ResponseFamily::Poisson, _)
            | (ResponseFamily::Tweedie { .. }, _)
            | (ResponseFamily::NegativeBinomial { .. }, _)
            | (ResponseFamily::Gamma, _) => {
                // E[exp(η)] where η ~ N(eta, se²) = exp(eta + se²/2)  (log-normal MGF)
                Ok((eta + 0.5 * se_eta * se_eta).exp())
            }
            (ResponseFamily::Beta { .. }, _) => {
                Ok(logit_posterior_meanvariance(quadctx, eta, se_eta).0)
            }
            (ResponseFamily::RoystonParmar, _) => Ok(survival_posterior_mean(quadctx, eta, se_eta)),
        }
    }

    fn posterior_meanvariance(
        &self,
        quadctx: &QuadratureContext,
        eta: f64,
        se_eta: f64,
    ) -> Result<(f64, f64), EstimationError> {
        match (&self.spec.response, &self.spec.link) {
            (ResponseFamily::Gaussian, _) => Ok((eta, (se_eta * se_eta).max(0.0))),
            (ResponseFamily::Binomial, InverseLink::Standard(StandardLink::Logit)) => {
                Ok(logit_posterior_meanvariance(quadctx, eta, se_eta))
            }
            (ResponseFamily::Binomial, InverseLink::Standard(StandardLink::Probit)) => {
                Ok(probit_posterior_meanvariance(quadctx, eta, se_eta))
            }
            (ResponseFamily::Binomial, InverseLink::Standard(StandardLink::CLogLog)) => {
                Ok(cloglog_posterior_meanvariance(quadctx, eta, se_eta))
            }
            (ResponseFamily::Binomial, InverseLink::Standard(_)) => {
                // Other standard binomial links fall back to the logit path
                // (the legacy default for non-{logit,probit,cloglog} binomial
                // standard links).
                Ok(logit_posterior_meanvariance(quadctx, eta, se_eta))
            }
            (ResponseFamily::Binomial, InverseLink::LatentCLogLog(_)) => {
                let state = self.require_latent_cloglog_state()?;
                let total_sigma = se_eta.hypot(state.latent_sd);
                let m1 = latent_cloglog_inverse_link_jet(quadctx, eta, total_sigma)?.mean;
                let m2 = normal_expectation_1d_adaptive(quadctx, eta, se_eta, |x| {
                    latent_cloglog_inverse_link_jet(quadctx, x, state.latent_sd)
                        .map(|jet| {
                            let p = jet.mean;
                            p * p
                        })
                        .unwrap_or(f64::NAN)
                });
                Ok((m1, (m2 - m1 * m1).max(0.0)))
            }
            (ResponseFamily::Binomial, InverseLink::Sas(_)) => {
                let state = self.require_sas_state()?;
                Ok(posterior_mv_from_prob_kernel(quadctx, eta, se_eta, |x| {
                    crate::mixture_link::sas_inverse_link_jet(x, state.epsilon, state.log_delta).mu
                }))
            }
            (ResponseFamily::Binomial, InverseLink::BetaLogistic(_)) => {
                let state = self.require_sas_state()?;
                Ok(posterior_mv_from_prob_kernel(quadctx, eta, se_eta, |x| {
                    crate::mixture_link::beta_logistic_inverse_link_jet(
                        x,
                        state.log_delta,
                        state.epsilon,
                    )
                    .mu
                }))
            }
            (ResponseFamily::Binomial, InverseLink::Mixture(_)) => {
                let state = self.require_mixture_state()?;
                let m1 = integrated_family_moments_jet(
                    quadctx,
                    &LikelihoodSpec::binomial_mixture(state.clone()),
                    eta,
                    se_eta,
                )?
                .mean;
                let m2 = normal_expectation_1d_adaptive(quadctx, eta, se_eta, |x| {
                    let p = mixture_inverse_link_jet(state, x).mu;
                    p * p
                });
                Ok((m1, (m2 - m1 * m1).max(0.0)))
            }
            (ResponseFamily::Poisson, _)
            | (ResponseFamily::Tweedie { .. }, _)
            | (ResponseFamily::NegativeBinomial { .. }, _)
            | (ResponseFamily::Gamma, _) => {
                // Log-normal moments: E[exp(η)] = exp(μ + σ²/2),
                // Var[exp(η)] = exp(2μ + σ²)(exp(σ²) - 1)
                let s2 = se_eta * se_eta;
                let m1 = (eta + 0.5 * s2).exp();
                let m2 = (2.0 * eta + s2).exp() * (s2.exp() - 1.0);
                Ok((m1, m2.max(0.0)))
            }
            (ResponseFamily::Beta { .. }, _) => {
                Ok(logit_posterior_meanvariance(quadctx, eta, se_eta))
            }
            (ResponseFamily::RoystonParmar, _) => {
                Ok(survival_posterior_meanvariance(quadctx, eta, se_eta))
            }
        }
    }

    fn simulate_noise(
        &self,
        mean: &Array1<f64>,
        gaussian_scale: Option<f64>,
    ) -> Result<NoiseModel, EstimationError> {
        match &self.spec.response {
            ResponseFamily::Gaussian => {
                let sigma = require_noise_parameter(&self.spec, "Gaussian sigma", gaussian_scale)?;
                if sigma < 0.0 {
                    crate::bail_invalid_estim!(
                        "Gaussian Identity generative sampling requires Gaussian sigma >= 0; got {sigma}"
                    );
                }
                Ok(NoiseModel::Gaussian {
                    sigma: Array1::from_elem(mean.len(), sigma),
                })
            }
            ResponseFamily::Binomial => Ok(NoiseModel::Bernoulli),
            ResponseFamily::Poisson => Ok(NoiseModel::Poisson),
            ResponseFamily::Tweedie { p } => {
                let p = *p;
                if !is_valid_tweedie_power(p) {
                    crate::bail_invalid_estim!(
                        "Tweedie variance power must be finite and strictly between 1 and 2; got {p}"
                    );
                }
                Ok(NoiseModel::Tweedie {
                    p,
                    phi: require_positive_noise_parameter(
                        &self.spec,
                        "Tweedie dispersion phi",
                        gaussian_scale,
                    )?,
                })
            }
            ResponseFamily::NegativeBinomial { theta } => {
                let theta = *theta;
                if !(theta.is_finite() && theta > 0.0) {
                    crate::bail_invalid_estim!(
                        "negative-binomial theta must be finite and > 0; got {theta}"
                    );
                }
                Ok(NoiseModel::NegativeBinomial { theta })
            }
            ResponseFamily::Beta { phi } => {
                let phi = *phi;
                if !(phi.is_finite() && phi > 0.0) {
                    crate::bail_invalid_estim!(
                        "beta-regression phi must be finite and > 0; got {phi}"
                    );
                }
                Ok(NoiseModel::Beta { phi })
            }
            ResponseFamily::Gamma => Ok(NoiseModel::Gamma {
                shape: require_positive_noise_parameter(&self.spec, "Gamma shape", gaussian_scale)?,
            }),
            ResponseFamily::RoystonParmar => Err(EstimationError::InvalidInput(
                "RoystonParmar generative sampling is not exposed via generic family strategy"
                    .to_string(),
            )),
        }
    }

    fn integrated_moments(
        &self,
        quadctx: &QuadratureContext,
        eta: f64,
        se_eta: f64,
    ) -> Result<IntegratedMomentsJet, EstimationError> {
        if let Some(state) = self.latent_cloglog_state() {
            let jet = latent_cloglog_inverse_link_jet(quadctx, eta, se_eta.hypot(state.latent_sd))?;
            let mean = jet.mean;
            return Ok(IntegratedMomentsJet {
                mean,
                variance: (mean * (1.0 - mean)).max(1e-12),
                d1: jet.d1,
                d2: jet.d2,
                d3: jet.d3,
                mode: jet.mode,
            });
        }
        integrated_family_moments_jet(quadctx, &self.spec, eta, se_eta)
    }
}
