use crate::estimate::{EstimationError, FitResult, FittedLinkState};
use crate::inference::generative::NoiseModel;
use crate::mixture_link::{InverseLinkJet, inverse_link_jet_for_family, mixture_inverse_link_jet};
use crate::quadrature::{
    IntegratedMomentsJet, QuadratureContext, cloglog_posterior_mean_variance,
    integrated_family_moments_jet_with_state, integrated_inverse_link_jet_with_state,
    integrated_inverse_link_mean_and_derivative, logit_posterior_mean_variance,
    normal_expectation_1d_adaptive, normal_expectation_1d_adaptive_pair,
    probit_posterior_mean_variance,
};
use crate::types::{InverseLink, LikelihoodFamily, LinkFunction};
use ndarray::{Array1, ArrayView1};

/// Runtime family behavior carrier built from a stable family identifier plus
/// optional fitted inverse-link state.
pub trait FamilyStrategy: std::fmt::Debug + Send + Sync {
    fn name(&self) -> &'static str;

    fn family(&self) -> LikelihoodFamily;

    fn link_function(&self) -> LinkFunction;

    fn inverse_link(&self, eta: f64) -> Result<f64, EstimationError>;

    fn inverse_link_array(&self, eta: ArrayView1<'_, f64>) -> Result<Array1<f64>, EstimationError>;

    fn inverse_link_jet(&self, eta: f64) -> Result<InverseLinkJet, EstimationError>;

    fn posterior_mean(
        &self,
        quad_ctx: &QuadratureContext,
        eta: f64,
        se_eta: f64,
    ) -> Result<f64, EstimationError>;

    fn posterior_mean_variance(
        &self,
        quad_ctx: &QuadratureContext,
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
        quad_ctx: &QuadratureContext,
        eta: f64,
        se_eta: f64,
    ) -> Result<IntegratedMomentsJet, EstimationError>;
}

#[derive(Clone, Debug)]
pub struct ResolvedFamilyStrategy {
    family: LikelihoodFamily,
    inverse_link: Option<InverseLink>,
}

#[inline]
pub fn strategy_for_family(
    family: LikelihoodFamily,
    inverse_link: Option<&InverseLink>,
) -> ResolvedFamilyStrategy {
    ResolvedFamilyStrategy {
        family,
        inverse_link: inverse_link.cloned(),
    }
}

pub fn strategy_from_fit(
    family: LikelihoodFamily,
    fit: &FitResult,
) -> Result<ResolvedFamilyStrategy, EstimationError> {
    let inverse_link = match fit.fitted_link_state(family)? {
        FittedLinkState::Standard(link) => Some(InverseLink::Standard(link)),
        FittedLinkState::Sas { state, .. } => Some(InverseLink::Sas(state)),
        FittedLinkState::BetaLogistic { state, .. } => Some(InverseLink::BetaLogistic(state)),
        FittedLinkState::Mixture { state, .. } => Some(InverseLink::Mixture(state)),
    };
    Ok(strategy_for_family(family, inverse_link.as_ref()))
}

impl ResolvedFamilyStrategy {
    #[inline]
    fn mixture_state(&self) -> Option<&crate::types::MixtureLinkState> {
        self.inverse_link
            .as_ref()
            .and_then(InverseLink::mixture_state)
    }

    #[inline]
    fn sas_state(&self) -> Option<&crate::types::SasLinkState> {
        self.inverse_link.as_ref().and_then(InverseLink::sas_state)
    }
}

impl FamilyStrategy for ResolvedFamilyStrategy {
    fn name(&self) -> &'static str {
        match self.family {
            LikelihoodFamily::GaussianIdentity => "gaussian",
            LikelihoodFamily::BinomialLogit => "binomial-logit",
            LikelihoodFamily::BinomialProbit => "binomial-probit",
            LikelihoodFamily::BinomialCLogLog => "binomial-cloglog",
            LikelihoodFamily::BinomialSas => "binomial-sas",
            LikelihoodFamily::BinomialBetaLogistic => "binomial-beta-logistic",
            LikelihoodFamily::BinomialMixture => "binomial-blended-inverse-link",
            LikelihoodFamily::RoystonParmar => "royston-parmar",
        }
    }

    fn family(&self) -> LikelihoodFamily {
        self.family
    }

    fn link_function(&self) -> LinkFunction {
        if let Some(inverse_link) = &self.inverse_link {
            return inverse_link.link_function();
        }
        match self.family {
            LikelihoodFamily::GaussianIdentity => LinkFunction::Identity,
            LikelihoodFamily::BinomialLogit => LinkFunction::Logit,
            LikelihoodFamily::BinomialProbit => LinkFunction::Probit,
            LikelihoodFamily::BinomialCLogLog => LinkFunction::CLogLog,
            LikelihoodFamily::BinomialSas => LinkFunction::Sas,
            LikelihoodFamily::BinomialBetaLogistic => LinkFunction::BetaLogistic,
            LikelihoodFamily::BinomialMixture => LinkFunction::Logit,
            LikelihoodFamily::RoystonParmar => LinkFunction::Identity,
        }
    }

    fn inverse_link(&self, eta: f64) -> Result<f64, EstimationError> {
        if matches!(self.family, LikelihoodFamily::RoystonParmar) {
            return Ok(eta);
        }
        self.inverse_link_jet(eta).map(|jet| jet.mu)
    }

    fn inverse_link_array(&self, eta: ArrayView1<'_, f64>) -> Result<Array1<f64>, EstimationError> {
        if matches!(self.family, LikelihoodFamily::RoystonParmar) {
            return Ok(eta.to_owned());
        }
        let mut out = Array1::<f64>::zeros(eta.len());
        for i in 0..eta.len() {
            out[i] = self.inverse_link(eta[i])?;
        }
        Ok(out)
    }

    fn inverse_link_jet(&self, eta: f64) -> Result<InverseLinkJet, EstimationError> {
        inverse_link_jet_for_family(self.family, eta, self.mixture_state(), self.sas_state())
    }

    fn posterior_mean(
        &self,
        quad_ctx: &QuadratureContext,
        eta: f64,
        se_eta: f64,
    ) -> Result<f64, EstimationError> {
        match self.family {
            LikelihoodFamily::GaussianIdentity => Ok(eta),
            LikelihoodFamily::BinomialLogit
            | LikelihoodFamily::BinomialProbit
            | LikelihoodFamily::BinomialCLogLog => integrated_inverse_link_mean_and_derivative(
                quad_ctx,
                self.link_function(),
                eta,
                se_eta,
            )
            .map(|v| v.mean),
            LikelihoodFamily::BinomialSas | LikelihoodFamily::BinomialBetaLogistic => {
                integrated_inverse_link_jet_with_state(
                    quad_ctx,
                    self.link_function(),
                    eta,
                    se_eta,
                    self.mixture_state(),
                    self.sas_state(),
                )
                .map(|v| v.mean)
            }
            LikelihoodFamily::BinomialMixture => {
                let state = self.mixture_state().ok_or_else(|| {
                    EstimationError::InvalidInput(
                        "BinomialMixture posterior mean requires fitted mixture link state"
                            .to_string(),
                    )
                })?;
                Ok(normal_expectation_1d_adaptive(quad_ctx, eta, se_eta, |x| {
                    mixture_inverse_link_jet(state, x).mu
                }))
            }
            LikelihoodFamily::RoystonParmar => Err(EstimationError::InvalidInput(
                "RoystonParmar posterior mean is not exposed via generic family strategy"
                    .to_string(),
            )),
        }
    }

    fn posterior_mean_variance(
        &self,
        quad_ctx: &QuadratureContext,
        eta: f64,
        se_eta: f64,
    ) -> Result<(f64, f64), EstimationError> {
        match self.family {
            LikelihoodFamily::GaussianIdentity => Ok((eta, (se_eta * se_eta).max(0.0))),
            LikelihoodFamily::BinomialLogit => {
                Ok(logit_posterior_mean_variance(quad_ctx, eta, se_eta))
            }
            LikelihoodFamily::BinomialProbit => {
                Ok(probit_posterior_mean_variance(quad_ctx, eta, se_eta))
            }
            LikelihoodFamily::BinomialCLogLog => {
                Ok(cloglog_posterior_mean_variance(quad_ctx, eta, se_eta))
            }
            LikelihoodFamily::BinomialSas => {
                let state = self.sas_state().ok_or_else(|| {
                    EstimationError::InvalidInput(
                        "BinomialSas posterior mean requires fitted SAS link state".to_string(),
                    )
                })?;
                let (m1, m2) = normal_expectation_1d_adaptive_pair(quad_ctx, eta, se_eta, |x| {
                    let p = crate::mixture_link::sas_inverse_link_jet(
                        x,
                        state.epsilon,
                        state.log_delta,
                    )
                    .mu;
                    (p, p * p)
                });
                Ok((m1, (m2 - m1 * m1).max(0.0)))
            }
            LikelihoodFamily::BinomialBetaLogistic => {
                let state = self.sas_state().ok_or_else(|| {
                    EstimationError::InvalidInput(
                        "BinomialBetaLogistic posterior mean requires fitted link state"
                            .to_string(),
                    )
                })?;
                let (m1, m2) = normal_expectation_1d_adaptive_pair(quad_ctx, eta, se_eta, |x| {
                    let p = crate::mixture_link::beta_logistic_inverse_link_jet(
                        x,
                        state.log_delta,
                        state.epsilon,
                    )
                    .mu;
                    (p, p * p)
                });
                Ok((m1, (m2 - m1 * m1).max(0.0)))
            }
            LikelihoodFamily::BinomialMixture => {
                let state = self.mixture_state().ok_or_else(|| {
                    EstimationError::InvalidInput(
                        "BinomialMixture posterior mean requires fitted mixture link state"
                            .to_string(),
                    )
                })?;
                let (m1, m2) = normal_expectation_1d_adaptive_pair(quad_ctx, eta, se_eta, |x| {
                    let p = mixture_inverse_link_jet(state, x).mu;
                    (p, p * p)
                });
                Ok((m1, (m2 - m1 * m1).max(0.0)))
            }
            LikelihoodFamily::RoystonParmar => Err(EstimationError::InvalidInput(
                "RoystonParmar posterior mean is not exposed via generic family strategy"
                    .to_string(),
            )),
        }
    }

    fn simulate_noise(
        &self,
        mean: &Array1<f64>,
        gaussian_scale: Option<f64>,
    ) -> Result<NoiseModel, EstimationError> {
        match self.family {
            LikelihoodFamily::GaussianIdentity => {
                let sigma = gaussian_scale.unwrap_or(1.0).max(0.0);
                Ok(NoiseModel::Gaussian {
                    sigma: Array1::from_elem(mean.len(), sigma),
                })
            }
            LikelihoodFamily::BinomialLogit
            | LikelihoodFamily::BinomialProbit
            | LikelihoodFamily::BinomialCLogLog
            | LikelihoodFamily::BinomialSas
            | LikelihoodFamily::BinomialBetaLogistic
            | LikelihoodFamily::BinomialMixture => Ok(NoiseModel::Bernoulli),
            LikelihoodFamily::RoystonParmar => Err(EstimationError::InvalidInput(
                "RoystonParmar generative sampling is not exposed via generic family strategy"
                    .to_string(),
            )),
        }
    }

    fn integrated_moments(
        &self,
        quad_ctx: &QuadratureContext,
        eta: f64,
        se_eta: f64,
    ) -> Result<IntegratedMomentsJet, EstimationError> {
        integrated_family_moments_jet_with_state(
            quad_ctx,
            self.family,
            eta,
            se_eta,
            self.mixture_state(),
            self.sas_state(),
        )
    }
}
