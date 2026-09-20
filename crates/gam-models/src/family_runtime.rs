use crate::inference::generative::NoiseModel;
use crate::model_types::{EstimationError, FittedLinkState, UnifiedFitResult};
use crate::quadrature::{
    QuadratureContext, cloglog_posterior_meanvariance,
    integrated_family_moments_jet, integrated_inverse_link_jetwith_state,
    integrated_inverse_link_mean_and_derivative, logit_posterior_meanvariance,
    logit_posterior_meanwith_deriv,
    normal_expectation_1d_adaptive, normal_expectation_1d_adaptive_pair,
    probit_posterior_meanvariance, reciprocal_link_posterior_meanvariance, survival_posterior_mean, survival_posterior_meanvariance,
};
use crate::survival::lognormal_kernel::latent_cloglog_inverse_link_jet;
use gam_problem::{
    InverseLink, LikelihoodSpec, LinkFunction, ResponseFamily,
    StandardLink,
};
use gam_solve::mixture_link::{
    InverseLinkJet, inverse_link_jet_for_family_public, mixture_inverse_link_jet,
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

    /// `E[1 − g⁻¹(η)]` under `η ~ N(eta, se_eta²)` for a response whose mean is a
    /// probability (Binomial, Beta, Royston–Parmar's survival probability).
    ///
    /// The complement is integrated as its own function of `η`, never formed as
    /// `1 − E[g⁻¹(η)]`: where the mean rounds to one, `1 − mean` is exactly zero
    /// while the complement is a representable positive number, and the Bernoulli
    /// variance `μ(1 − μ)` an observation band is built from lives on it (#3140).
    fn posterior_complement_mean(
        &self,
        quadctx: &QuadratureContext,
        eta: f64,
        se_eta: f64,
    ) -> Result<f64, EstimationError>;

    fn simulate_noise(
        &self,
        mean: &Array1<f64>,
        gaussian_scale: Option<f64>,
    ) -> Result<NoiseModel, EstimationError>;

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
    fn mixture_state(&self) -> Option<&gam_problem::MixtureLinkState> {
        self.spec.link.mixture_state()
    }

    #[inline]
    fn sas_state(&self) -> Option<&gam_problem::SasLinkState> {
        self.spec.link.sas_state()
    }

    #[inline]
    fn latent_cloglog_state(&self) -> Option<&gam_problem::LatentCLogLogState> {
        self.spec.link.latent_cloglog_state()
    }

    #[inline]
    fn require_latent_cloglog_state(
        &self,
    ) -> Result<&gam_problem::LatentCLogLogState, EstimationError> {
        self.latent_cloglog_state()
            .ok_or_else(|| missing_state(&self.spec, "latent cloglog"))
    }

    #[inline]
    fn require_sas_state(&self) -> Result<&gam_problem::SasLinkState, EstimationError> {
        self.sas_state()
            .ok_or_else(|| missing_state(&self.spec, "SAS link"))
    }

    #[inline]
    fn require_mixture_state(&self) -> Result<&gam_problem::MixtureLinkState, EstimationError> {
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
        // Public response-scale surface: use the EXACT inverse-link jet so the
        // log link reports `exp(eta)` wherever IEEE-754 can represent it. The
        // shared solver derivative seam instead refuses eta outside its declared
        // inclusive [-700, 700] domain; PIRLS working-state conditioning remains
        // a separate concern. Within the solver domain both paths are the same
        // exact exponential (issue #963).
        inverse_link_jet_for_family_public(&self.spec, eta)
    }

    fn posterior_mean(
        &self,
        quadctx: &QuadratureContext,
        eta: f64,
        se_eta: f64,
    ) -> Result<f64, EstimationError> {
        match (&self.spec.response, &self.spec.link) {
            (ResponseFamily::StudentT { .. }, _) => Ok(eta),
            (
                ResponseFamily::Gaussian
                | ResponseFamily::Gamma
                | ResponseFamily::InverseGaussian
                | ResponseFamily::Poisson
                | ResponseFamily::Tweedie { .. }
                | ResponseFamily::NegativeBinomial { .. },
                InverseLink::Standard(StandardLink::Log),
            ) => {
                // E[exp(η)] where η ~ N(eta, se²) = exp(eta + se²/2)
                // (log-normal MGF). When the exponent exceeds the f64 range the
                // posterior mean genuinely overflows; `exp` then returns +inf,
                // which IS the correctly rounded value of the integral. Earlier
                // revisions substituted the plug-in `exp(η)` (or f64::MAX) here
                // to keep the FFI finite, silently turning an unbounded
                // posterior mean into an innocuous value (η = 0, se = 40 →
                // exponent 800 reported as 1). Honesty over convenience:
                // return the exact, possibly infinite, mean and let callers
                // decide how to present it.
                Ok((eta + 0.5 * se_eta * se_eta).exp())
            }
            (
                ResponseFamily::Gaussian
                | ResponseFamily::Gamma
                | ResponseFamily::InverseGaussian
                | ResponseFamily::Poisson
                | ResponseFamily::Tweedie { .. }
                | ResponseFamily::NegativeBinomial { .. },
                _,
            ) => {
                // Identity is the plug-in, sqrt is `eta² + se²`, and the
                // reciprocal links are the principal-value / positive-part
                // means of the shared dispatcher.
                integrated_inverse_link_mean_and_derivative(
                    quadctx,
                    self.link_function(),
                    eta,
                    se_eta,
                )
                .map(|v| v.mean)
            }
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
                let likelihood = gam_problem::GlmLikelihoodSpec::canonical(
                    LikelihoodSpec::binomial_mixture(state.clone()),
                );
                integrated_family_moments_jet(
                    quadctx,
                    &likelihood,
                    eta,
                    se_eta,
                )
                .map(|v| v.mean)
            }
            (ResponseFamily::Beta { .. }, _) => {
                logit_posterior_meanwith_deriv(eta, se_eta).map(|(mean, _)| mean)
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
            (ResponseFamily::StudentT { .. }, _) => Ok((eta, se_eta * se_eta)),
            (
                ResponseFamily::Gaussian
                | ResponseFamily::Gamma
                | ResponseFamily::InverseGaussian
                | ResponseFamily::Poisson
                | ResponseFamily::Tweedie { .. }
                | ResponseFamily::NegativeBinomial { .. },
                link,
            ) => match link {
                InverseLink::Standard(StandardLink::Identity) => Ok((eta, se_eta * se_eta)),
                InverseLink::Standard(StandardLink::Log) => Ok(lognormal_meanvariance(eta, se_eta)),
                InverseLink::Standard(StandardLink::Sqrt) => {
                    // μ = η² with η ~ N(m, s²): E[η²] = m² + s² and
                    // Var[η²] = E[η⁴] − E[η²]² = (m⁴ + 6m²s² + 3s⁴) − (m² + s²)²
                    // = 2s²(2m² + s²), exactly. The mean comes from the shared
                    // dispatcher, which also refuses a predictor outside `η > 0`.
                    let mean = integrated_inverse_link_mean_and_derivative(
                        quadctx,
                        LinkFunction::Sqrt,
                        eta,
                        se_eta,
                    )?
                    .mean;
                    let s2 = se_eta * se_eta;
                    Ok((mean, 2.0 * s2 * (2.0 * eta * eta + s2)))
                }
                InverseLink::Standard(
                    reciprocal @ (StandardLink::Inverse | StandardLink::InverseSquared),
                ) => reciprocal_link_posterior_meanvariance(
                    reciprocal.as_link_function(),
                    eta,
                    se_eta,
                ),
                other => Err(EstimationError::InvalidInput(format!(
                    "{} likelihood has no posterior variance for link {:?}",
                    self.spec.response.name(),
                    other
                ))),
            },
            (ResponseFamily::Binomial, InverseLink::Standard(StandardLink::Logit)) => {
                logit_posterior_meanvariance(eta, se_eta)
            }
            (ResponseFamily::Binomial, InverseLink::Standard(StandardLink::Probit)) => {
                Ok(probit_posterior_meanvariance(eta, se_eta))
            }
            (ResponseFamily::Binomial, InverseLink::Standard(StandardLink::CLogLog)) => {
                Ok(cloglog_posterior_meanvariance(quadctx, eta, se_eta))
            }
            (ResponseFamily::Binomial, InverseLink::Standard(_)) => {
                // Remaining standard binomial links (LogLog, Cauchit, ...):
                // integrate the family's ACTUAL inverse link through the shared
                // probability-kernel quadrature. The historical fallback
                // integrated the logistic kernel here, so LogLog/Cauchit
                // response moments were computed for the wrong link (at
                // se_eta = 0, η = 1: exact Cauchit mean 0.75, exact LogLog mean
                // exp(-exp(-1)) ≈ 0.6922, logistic 0.7311).
                Ok(posterior_mv_from_prob_kernel(quadctx, eta, se_eta, |x| {
                    inverse_link_jet_for_family_public(&self.spec, x)
                        .map(|jet| jet.mu)
                        .unwrap_or(f64::NAN)
                }))
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
                    gam_solve::mixture_link::sas_inverse_link_jet(x, state.epsilon, state.log_delta)
                        .expect("normal quadrature nodes must be finite")
                        .mu
                }))
            }
            (ResponseFamily::Binomial, InverseLink::BetaLogistic(_)) => {
                let state = self.require_sas_state()?;
                Ok(posterior_mv_from_prob_kernel(quadctx, eta, se_eta, |x| {
                    gam_solve::mixture_link::beta_logistic_inverse_link_jet(
                        x,
                        state.log_delta,
                        state.epsilon,
                    )
                    .mu
                }))
            }
            (ResponseFamily::Binomial, InverseLink::Mixture(_)) => {
                let state = self.require_mixture_state()?;
                let likelihood = gam_problem::GlmLikelihoodSpec::canonical(
                    LikelihoodSpec::binomial_mixture(state.clone()),
                );
                let m1 = integrated_family_moments_jet(
                    quadctx,
                    &likelihood,
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
            (ResponseFamily::Beta { .. }, _) => {
                logit_posterior_meanvariance(eta, se_eta)
            }
            (ResponseFamily::RoystonParmar, _) => {
                Ok(survival_posterior_meanvariance(quadctx, eta, se_eta))
            }
        }
    }

    fn posterior_complement_mean(
        &self,
        quadctx: &QuadratureContext,
        eta: f64,
        se_eta: f64,
    ) -> Result<f64, EstimationError> {
        match (&self.spec.response, &self.spec.link) {
            // The logistic is point-symmetric, `1 − σ(η) = σ(−η)`, so the complement's
            // posterior mean is the mean at the reflected centre. Beta's mean is the
            // logistic of η whatever its link tag, as `posterior_mean` integrates it.
            (ResponseFamily::Binomial, InverseLink::Standard(StandardLink::Logit))
            | (ResponseFamily::Beta { .. }, _) => {
                logit_posterior_meanwith_deriv(-eta, se_eta).map(|(mean, _)| mean)
            }
            (ResponseFamily::Binomial, InverseLink::Standard(StandardLink::Probit)) => {
                // E[Φ(η)] = Φ(eta / √(1 + se²)), so E[1 − Φ(η)] = Φ(−eta / √(1 + se²)).
                Ok(gam_math::probability::normal_cdf(
                    -eta / (1.0 + se_eta * se_eta).sqrt(),
                ))
            }
            // cloglog: 1 − μ = exp(−exp η), the survival term the survival path owns.
            (ResponseFamily::Binomial, InverseLink::Standard(StandardLink::CLogLog)) => {
                Ok(survival_posterior_mean(quadctx, eta, se_eta))
            }
            // The latent-cloglog kernel reports its mean but not the survival
            // output the exact complement needs (mixture_link.rs,
            // `inverse_link_complement_for_inverse_link`), so its complement is the
            // mean's, as the working response already takes it.
            (ResponseFamily::Binomial, InverseLink::LatentCLogLog(_)) => self
                .posterior_mean(quadctx, eta, se_eta)
                .map(|mean| 1.0 - mean),
            (ResponseFamily::Binomial, link) => {
                let spec = &self.spec;
                Ok(normal_expectation_1d_adaptive(quadctx, eta, se_eta, |x| {
                    inverse_link_jet_for_family_public(spec, x)
                        .map(|jet| {
                            gam_solve::mixture_link::inverse_link_complement_for_inverse_link(
                                link, x, jet.mu,
                            )
                        })
                        .unwrap_or(f64::NAN)
                }))
            }
            // S = exp(−exp η), so 1 − S = −expm1(−exp η), whose digits survive where
            // S rounds to one.
            (ResponseFamily::RoystonParmar, _) => {
                Ok(normal_expectation_1d_adaptive(quadctx, eta, se_eta, |x| {
                    -(-x.exp()).exp_m1()
                }))
            }
            (
                ResponseFamily::Gaussian
                | ResponseFamily::StudentT { .. }
                | ResponseFamily::Poisson
                | ResponseFamily::Tweedie { .. }
                | ResponseFamily::NegativeBinomial { .. }
                | ResponseFamily::Gamma
                | ResponseFamily::InverseGaussian,
                _,
            ) => Err(EstimationError::InvalidInput(format!(
                "{} has no probability-valued mean, so it has no posterior complement mean",
                self.spec.pretty_name()
            ))),
        }
    }

    fn simulate_noise(
        &self,
        mean: &Array1<f64>,
        gaussian_scale: Option<f64>,
    ) -> Result<NoiseModel, EstimationError> {
        // Thin adapter over the single canonical likelihood -> noise-model
        // mapping shared with generative inference, so simulation and
        // inference can never disagree on supported likelihoods or how
        // dispersion parameters are interpreted.
        NoiseModel::from_likelihood(&self.spec, mean.len(), gaussian_scale)
    }

}

#[cfg(test)]
mod log_link_public_jet_tests {
    use super::*;
    use gam_problem::LikelihoodSpec;
    use ndarray::Array1;

    /// The PUBLIC predict surface for a log-link family (Poisson/Gamma/Tweedie/
    /// NB) accepts representable eta beyond the solver derivative domain. This
    /// drives the exact funnel the predict path uses —
    /// `FamilyStrategy::inverse_link` / `inverse_link_array` /
    /// `inverse_link_jet` — and pins a finite eta the solver correctly refuses.
    #[test]
    fn public_predict_log_inverse_link_is_exact_exp_at_boundary() {
        let strategy = strategy_for_spec(&LikelihoodSpec::poisson_log());

        // eta = 705 is outside the solver derivative domain but exact exp(705)
        // is representable and therefore valid on this public surface.
        let exact = 705.0_f64.exp();
        assert!(exact.is_finite(), "exp(705) must be representable in f64");
        let jet = strategy.inverse_link_jet(705.0).expect("jet");
        assert_eq!(jet.mu, exact, "predict mean must be exact exp(705)");
        // All derivatives of exp are exp; the delta-method SE reads `d1`.
        assert_eq!(jet.d1, exact, "predict dmu/deta must be exact exp(705)");
        assert_eq!(jet.d2, exact);
        assert_eq!(jet.d3, exact);
        let historical_projection = 700.0_f64.exp();
        assert!(
            jet.mu > historical_projection * 100.0,
            "exact exp(705) must not regress to the historical exp(700) projection"
        );

        // Array entry point used by `predict_plugin_response`/`response`.
        let arr = strategy
            .inverse_link_array(Array1::from(vec![705.0]).view())
            .expect("array");
        assert_eq!(arr[0], exact, "inverse_link_array must be exact exp(705)");

        // eta = -720 is likewise valid on the public response transform.
        let exact_neg = (-720.0_f64).exp();
        let jet = strategy.inverse_link_jet(-720.0).expect("jet");
        assert_eq!(jet.mu, exact_neg, "predict mean must be exact exp(-720)");
        let historical_projection_neg = (-700.0_f64).exp();
        assert!(
            jet.mu < historical_projection_neg,
            "exact exp(-720) must not regress to the historical exp(-700) projection"
        );

        // True IEEE limits honored exactly on the public surface.
        let over = strategy.inverse_link_jet(710.0).expect("jet");
        assert!(over.mu.is_infinite() && over.mu > 0.0, "exp(710) -> +inf");
        let under = strategy.inverse_link_jet(-746.0).expect("jet");
        assert_eq!(under.mu, 0.0, "exp(-746) -> 0.0");
    }

    fn standard_strategy(response: ResponseFamily, link: StandardLink) -> ResolvedFamilyStrategy {
        strategy_for_spec(&LikelihoodSpec::new(response, InverseLink::Standard(link)))
    }

    /// A count family's posterior response moments are those of ITS inverse
    /// link: identity-, sqrt- and inverse-link Poisson/Tweedie/NB fits are
    /// generic EDM cells, and none of their means is the log-normal
    /// `exp(eta + se²/2)` the log link implies.
    #[test]
    fn count_family_posterior_moments_follow_the_link() {
        let quadctx = QuadratureContext::new();
        let (eta, se) = (5.0_f64, 0.3_f64);
        for response in [
            ResponseFamily::Poisson,
            ResponseFamily::Tweedie { p: 1.5 },
            ResponseFamily::NegativeBinomial {
                theta: 2.0,
                theta_fixed: true,
            },
        ] {
            let identity = standard_strategy(response.clone(), StandardLink::Identity);
            assert_eq!(identity.posterior_mean(&quadctx, eta, se).unwrap(), eta);
            assert_eq!(
                identity.posterior_meanvariance(&quadctx, eta, se).unwrap(),
                (eta, se * se)
            );

            let sqrt = standard_strategy(response.clone(), StandardLink::Sqrt);
            let (m1, m2) = normal_expectation_1d_adaptive_pair(&quadctx, eta, se, |x| {
                let mu = x * x;
                (mu, mu * mu)
            });
            let mean = sqrt.posterior_mean(&quadctx, eta, se).unwrap();
            let (mv_mean, variance) = sqrt.posterior_meanvariance(&quadctx, eta, se).unwrap();
            assert!((mean - (eta * eta + se * se)).abs() <= 1e-14 * mean);
            assert_eq!(mv_mean, mean);
            assert!(
                (mean - m1).abs() <= 1e-12 * m1,
                "sqrt mean {mean} vs quadrature {m1}"
            );
            let quadrature_variance = m2 - m1 * m1;
            assert!(
                (variance - quadrature_variance).abs() <= 1e-9 * quadrature_variance,
                "sqrt variance {variance} vs quadrature {quadrature_variance}"
            );
            assert!(sqrt.posterior_mean(&quadctx, -1.0, se).is_err());

            for link in [StandardLink::Inverse, StandardLink::InverseSquared] {
                let strategy = standard_strategy(response.clone(), link);
                let expected =
                    reciprocal_link_posterior_meanvariance(link.as_link_function(), eta, se)
                        .unwrap();
                assert_eq!(
                    strategy.posterior_meanvariance(&quadctx, eta, se).unwrap(),
                    expected
                );
                assert_eq!(
                    strategy.posterior_mean(&quadctx, eta, se).unwrap(),
                    expected.0
                );
            }

            let log = standard_strategy(response, StandardLink::Log);
            assert_eq!(
                log.posterior_mean(&quadctx, eta, se).unwrap(),
                (eta + 0.5 * se * se).exp()
            );
            assert_eq!(
                log.posterior_meanvariance(&quadctx, eta, se).unwrap(),
                lognormal_meanvariance(eta, se)
            );
        }
    }

    /// Sqrt-link Gaussian/Gamma/inverse-Gaussian fits have a posterior mean,
    /// so they have the posterior variance `2se²(2eta² + se²)` of `η²` too.
    #[test]
    fn sqrt_link_continuous_families_report_posterior_variance() {
        let quadctx = QuadratureContext::new();
        let (eta, se) = (2.0_f64, 0.25_f64);
        for response in [
            ResponseFamily::Gaussian,
            ResponseFamily::Gamma,
            ResponseFamily::InverseGaussian,
        ] {
            let strategy = standard_strategy(response, StandardLink::Sqrt);
            let (mean, variance) = strategy.posterior_meanvariance(&quadctx, eta, se).unwrap();
            let s2 = se * se;
            assert_eq!(mean, strategy.posterior_mean(&quadctx, eta, se).unwrap());
            assert!((mean - (eta * eta + s2)).abs() <= 1e-14 * mean);
            assert_eq!(variance, 2.0 * s2 * (2.0 * eta * eta + s2));
            assert!(strategy.posterior_meanvariance(&quadctx, 0.0, se).is_err());
        }
    }

}

/// Log-normal moments: `E[exp(η)] = exp(μ + σ²/2)`,
/// `Var[exp(η)] = exp(2μ + σ²)·expm1(σ²)`. `expm1` keeps the variance factor
/// exact for tiny `σ²` (`σ² = 1e-20`: `exp(σ²) − 1` rounds to 0, `expm1`
/// returns 1e-20), so small but nonzero posterior uncertainty is never
/// reported as exactly zero.
fn lognormal_meanvariance(eta: f64, se_eta: f64) -> (f64, f64) {
    let s2 = se_eta * se_eta;
    let m1 = (eta + 0.5 * s2).exp();
    let variance = (2.0 * eta + s2).exp() * s2.exp_m1();
    (m1, variance)
}
