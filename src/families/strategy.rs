use crate::estimate::{EstimationError, FittedLinkState, UnifiedFitResult};
use crate::families::lognormal_kernel::latent_cloglog_inverse_link_jet;
use crate::inference::generative::NoiseModel;
use crate::mixture_link::{InverseLinkJet, inverse_link_jet_for_family, mixture_inverse_link_jet};
use crate::quadrature::{
    IntegratedMomentsJet, QuadratureContext, cloglog_posterior_meanvariance,
    integrated_family_moments_jetwith_state, integrated_inverse_link_jetwith_state,
    integrated_inverse_link_mean_and_derivative, logit_posterior_meanvariance,
    normal_expectation_1d_adaptive, normal_expectation_1d_adaptive_pair,
    probit_posterior_meanvariance, survival_posterior_mean, survival_posterior_meanvariance,
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

    #[inline]
    fn latent_cloglog_state(&self) -> Option<&crate::types::LatentCLogLogState> {
        self.inverse_link
            .as_ref()
            .and_then(InverseLink::latent_cloglog_state)
    }

    #[inline]
    fn require_latent_cloglog_state(
        &self,
    ) -> Result<&crate::types::LatentCLogLogState, EstimationError> {
        self.latent_cloglog_state()
            .ok_or_else(|| missing_state(self.family, "latent cloglog"))
    }

    #[inline]
    fn require_sas_state(&self) -> Result<&crate::types::SasLinkState, EstimationError> {
        self.sas_state()
            .ok_or_else(|| missing_state(self.family, "SAS link"))
    }

    #[inline]
    fn require_mixture_state(&self) -> Result<&crate::types::MixtureLinkState, EstimationError> {
        self.mixture_state()
            .ok_or_else(|| missing_state(self.family, "mixture link"))
    }
}

#[cold]
fn missing_state(family: LikelihoodFamily, what: &str) -> EstimationError {
    EstimationError::InvalidInput(format!(
        "{} requires fitted {} state",
        family.pretty_name(),
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

impl FamilyStrategy for ResolvedFamilyStrategy {
    fn name(&self) -> &'static str {
        self.family.name()
    }

    fn family(&self) -> LikelihoodFamily {
        self.family
    }

    fn link_function(&self) -> LinkFunction {
        if let Some(inverse_link) = &self.inverse_link {
            return inverse_link.link_function();
        }
        self.family.link_function()
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
        if let Some(inverse_link) = &self.inverse_link {
            return crate::mixture_link::inverse_link_jet_for_inverse_link(inverse_link, eta);
        }
        inverse_link_jet_for_family(self.family, eta, self.mixture_state(), self.sas_state())
    }

    fn posterior_mean(
        &self,
        quadctx: &QuadratureContext,
        eta: f64,
        se_eta: f64,
    ) -> Result<f64, EstimationError> {
        match self.family {
            LikelihoodFamily::GaussianIdentity => Ok(eta),
            LikelihoodFamily::BinomialLogit
            | LikelihoodFamily::BinomialProbit
            | LikelihoodFamily::BinomialCLogLog => integrated_inverse_link_mean_and_derivative(
                quadctx,
                self.link_function(),
                eta,
                se_eta,
            )
            .map(|v| v.mean),
            LikelihoodFamily::BinomialLatentCLogLog => {
                let state = self.require_latent_cloglog_state()?;
                latent_cloglog_inverse_link_jet(quadctx, eta, se_eta.hypot(state.latent_sd))
                    .map(|v| v.mean)
            }
            LikelihoodFamily::BinomialSas | LikelihoodFamily::BinomialBetaLogistic => {
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
            LikelihoodFamily::BinomialMixture => {
                let state = self.require_mixture_state()?;
                integrated_family_moments_jetwith_state(
                    quadctx,
                    LikelihoodFamily::BinomialMixture,
                    eta,
                    se_eta,
                    Some(state),
                    None,
                )
                .map(|v| v.mean)
            }
            LikelihoodFamily::PoissonLog
            | LikelihoodFamily::Tweedie { .. }
            | LikelihoodFamily::NegativeBinomial { .. }
            | LikelihoodFamily::GammaLog => {
                // E[exp(η)] where η ~ N(eta, se²) = exp(eta + se²/2)  (log-normal MGF)
                Ok((eta + 0.5 * se_eta * se_eta).exp())
            }
            LikelihoodFamily::BetaLogit { .. } => {
                Ok(logit_posterior_meanvariance(quadctx, eta, se_eta).0)
            }
            LikelihoodFamily::RoystonParmar => Ok(survival_posterior_mean(quadctx, eta, se_eta)),
        }
    }

    fn posterior_meanvariance(
        &self,
        quadctx: &QuadratureContext,
        eta: f64,
        se_eta: f64,
    ) -> Result<(f64, f64), EstimationError> {
        match self.family {
            LikelihoodFamily::GaussianIdentity => Ok((eta, (se_eta * se_eta).max(0.0))),
            LikelihoodFamily::BinomialLogit => {
                Ok(logit_posterior_meanvariance(quadctx, eta, se_eta))
            }
            LikelihoodFamily::BinomialProbit => {
                Ok(probit_posterior_meanvariance(quadctx, eta, se_eta))
            }
            LikelihoodFamily::BinomialCLogLog => {
                Ok(cloglog_posterior_meanvariance(quadctx, eta, se_eta))
            }
            LikelihoodFamily::BinomialLatentCLogLog => {
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
            LikelihoodFamily::BinomialSas => {
                let state = self.require_sas_state()?;
                Ok(posterior_mv_from_prob_kernel(quadctx, eta, se_eta, |x| {
                    crate::mixture_link::sas_inverse_link_jet(x, state.epsilon, state.log_delta).mu
                }))
            }
            LikelihoodFamily::BinomialBetaLogistic => {
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
            LikelihoodFamily::BinomialMixture => {
                let state = self.require_mixture_state()?;
                let m1 = integrated_family_moments_jetwith_state(
                    quadctx,
                    LikelihoodFamily::BinomialMixture,
                    eta,
                    se_eta,
                    Some(state),
                    None,
                )?
                .mean;
                let m2 = normal_expectation_1d_adaptive(quadctx, eta, se_eta, |x| {
                    let p = mixture_inverse_link_jet(state, x).mu;
                    p * p
                });
                Ok((m1, (m2 - m1 * m1).max(0.0)))
            }
            LikelihoodFamily::PoissonLog
            | LikelihoodFamily::Tweedie { .. }
            | LikelihoodFamily::NegativeBinomial { .. }
            | LikelihoodFamily::GammaLog => {
                // Log-normal moments: E[exp(η)] = exp(μ + σ²/2),
                // Var[exp(η)] = exp(2μ + σ²)(exp(σ²) - 1)
                let s2 = se_eta * se_eta;
                let m1 = (eta + 0.5 * s2).exp();
                let m2 = (2.0 * eta + s2).exp() * (s2.exp() - 1.0);
                Ok((m1, m2.max(0.0)))
            }
            LikelihoodFamily::BetaLogit { .. } => {
                Ok(logit_posterior_meanvariance(quadctx, eta, se_eta))
            }
            LikelihoodFamily::RoystonParmar => {
                Ok(survival_posterior_meanvariance(quadctx, eta, se_eta))
            }
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
            | LikelihoodFamily::BinomialLatentCLogLog
            | LikelihoodFamily::BinomialSas
            | LikelihoodFamily::BinomialBetaLogistic
            | LikelihoodFamily::BinomialMixture => Ok(NoiseModel::Bernoulli),
            LikelihoodFamily::PoissonLog => Ok(NoiseModel::Poisson),
            LikelihoodFamily::Tweedie { p } => Ok(NoiseModel::Tweedie {
                p,
                phi: gaussian_scale.unwrap_or(1.0).max(1e-12),
            }),
            LikelihoodFamily::NegativeBinomial { theta } => {
                if !(theta.is_finite() && theta > 0.0) {
                    return Err(EstimationError::InvalidInput(format!(
                        "negative-binomial theta must be finite and > 0; got {theta}"
                    )));
                }
                Ok(NoiseModel::NegativeBinomial { theta })
            }
            LikelihoodFamily::BetaLogit { phi } => {
                if !(phi.is_finite() && phi > 0.0) {
                    return Err(EstimationError::InvalidInput(format!(
                        "beta-regression phi must be finite and > 0; got {phi}"
                    )));
                }
                Ok(NoiseModel::Beta { phi })
            }
            LikelihoodFamily::GammaLog => {
                // Callers should pass the fitted Gamma shape; default to 1 only
                // when older metadata does not expose it.
                Ok(NoiseModel::Gamma {
                    shape: gaussian_scale.unwrap_or(1.0).max(1e-6),
                })
            }
            LikelihoodFamily::RoystonParmar => Err(EstimationError::InvalidInput(
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
        integrated_family_moments_jetwith_state(
            quadctx,
            self.family,
            eta,
            se_eta,
            self.mixture_state(),
            self.sas_state(),
        )
    }
}
