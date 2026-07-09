use crate::survival::lognormal_kernel::latent_cloglog_inverse_link_jet;
use crate::inference::generative::NoiseModel;
use gam_solve::mixture_link::{
    InverseLinkJet, inverse_link_jet_for_family_public, mixture_inverse_link_jet,
};
use crate::model_types::{EstimationError, FittedLinkState, UnifiedFitResult};
use crate::quadrature::{
    IntegratedMomentsJet, QuadratureContext, cloglog_posterior_meanvariance,
    integrated_family_moments_jet, integrated_inverse_link_jetwith_state,
    integrated_inverse_link_mean_and_derivative, logit_posterior_meanvariance,
    normal_expectation_1d_adaptive, normal_expectation_1d_adaptive_pair,
    probit_posterior_meanvariance, survival_posterior_mean, survival_posterior_meanvariance,
};
use gam_problem::{
    InverseLink, LikelihoodScaleMetadata, LikelihoodSpec, LinkFunction, ResponseFamily,
    StandardLink,
};
use ndarray::{Array1, ArrayView1};

/// Floor on the Bernoulli posterior variance `p(1 - p)`. Keeps the reported
/// variance strictly positive when the integrated probability saturates at 0
/// or 1, so downstream weighting / standard-error code never divides by zero.
/// Matches the `PROB_EPS` floor used for the same `mean·(1 - mean)` variance
/// in `crate::inference::quadrature`.
const PROB_VARIANCE_FLOOR: f64 = 1e-12;

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
        // log link reports `exp(η)` (finite wherever representable) rather than
        // the solver's `η.clamp(−700, 700).exp()` conditioning value. This funnel
        // feeds `inverse_link`/`inverse_link_array` and the predict mean +
        // delta-method SE path; the solver/REML/PIRLS engines keep the clamped
        // jet. For `|η| ≤ 700` the two are byte-identical (issue #963).
        inverse_link_jet_for_family_public(&self.spec, eta)
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
                    // Binomial variance is pinned by the mean (φ ≡ 1); the mixture
                    // mean does not depend on `scale`, so pass the canonical
                    // unit-scale label explicitly.
                    LikelihoodScaleMetadata::FixedDispersion { phi: 1.0 },
                    eta,
                    se_eta,
                )
                .map(|v| v.mean)
            }
            (ResponseFamily::Poisson, _)
            | (ResponseFamily::Tweedie { .. }, _)
            | (ResponseFamily::NegativeBinomial { .. }, _)
            | (ResponseFamily::Gamma, _) => {
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
                    gam_solve::mixture_link::sas_inverse_link_jet(x, state.epsilon, state.log_delta).mu
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
                let m1 = integrated_family_moments_jet(
                    quadctx,
                    &LikelihoodSpec::binomial_mixture(state.clone()),
                    // Binomial variance is pinned by the mean (φ ≡ 1); only the
                    // integrated mean `m1` is read here, so pass the canonical
                    // unit-scale label explicitly.
                    LikelihoodScaleMetadata::FixedDispersion { phi: 1.0 },
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
                // Var[exp(η)] = exp(2μ + σ²)·expm1(σ²). `expm1` keeps the
                // variance factor exact for tiny σ² (σ² = 1e-20: exp(σ²) - 1
                // rounds to 0, expm1 returns 1e-20), so small but nonzero
                // posterior uncertainty is never reported as exactly zero.
                let s2 = se_eta * se_eta;
                let m1 = (eta + 0.5 * s2).exp();
                let m2 = (2.0 * eta + s2).exp() * s2.exp_m1();
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
        // Thin adapter over the single canonical likelihood -> noise-model
        // mapping shared with generative inference, so simulation and
        // inference can never disagree on supported likelihoods or how
        // dispersion parameters are interpreted.
        NoiseModel::from_likelihood(&self.spec, mean.len(), gaussian_scale)
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
                variance: (mean * (1.0 - mean)).max(PROB_VARIANCE_FLOOR),
                d1: jet.d1,
                d2: jet.d2,
                d3: jet.d3,
                mode: jet.mode,
            });
        }
        // The observation-model variance for Tweedie/Gamma depends on the
        // exponential-dispersion metadata (Tweedie φ, Gamma shape). The strategy
        // carries the (response, link) spec, so supply that spec's scale metadata
        // — for Gamma/Tweedie this is the estimated-dispersion variant (seeded at
        // the unit value, refined during fitting), never a silent hardcoded φ = 1
        // baked into the integrator (issue #953).
        integrated_family_moments_jet(
            quadctx,
            &self.spec,
            self.spec.default_scale_metadata(),
            eta,
            se_eta,
        )
    }
}

#[cfg(test)]
mod log_link_public_jet_tests {
    use super::*;
    use gam_solve::mixture_link::inverse_link_jet_for_family;
    use gam_problem::LikelihoodSpec;
    use ndarray::Array1;

    /// The PUBLIC predict surface for a log-link family (Poisson/Gamma/Tweedie/
    /// NB) must report the EXACT `exp(η)`, never the solver's
    /// `η.clamp(−700, 700).exp()` conditioning value (issue #963). This drives
    /// the exact funnel the predict path uses — `FamilyStrategy::inverse_link`
    /// / `inverse_link_array` / `inverse_link_jet` (the predict mean +
    /// delta-method SE source) — and pins the finite boundary η where the exact
    /// and clamped transforms diverge.
    #[test]
    fn public_predict_log_inverse_link_is_exact_exp_at_boundary() {
        let strategy = strategy_for_spec(&LikelihoodSpec::poisson_log());

        // η = 705: exact exp(705) ≈ 1.505e306 is finite; the solver clamp would
        // return exp(700) ≈ 1.014e304, wrong by exp(5) ≈ 148.
        let exact = 705.0_f64.exp();
        assert!(exact.is_finite(), "exp(705) must be representable in f64");
        let jet = strategy.inverse_link_jet(705.0).expect("jet");
        assert_eq!(jet.mu, exact, "predict mean must be exact exp(705)");
        // All derivatives of exp are exp; the delta-method SE reads `d1`.
        assert_eq!(jet.d1, exact, "predict dmu/deta must be exact exp(705)");
        assert_eq!(jet.d2, exact);
        assert_eq!(jet.d3, exact);
        let clamped = 700.0_f64.exp();
        assert!(
            jet.mu > clamped * 100.0,
            "exact exp(705) must exceed the clamped exp(700) by ~exp(5)"
        );

        // Array entry point used by `predict_plugin_response`/`response`.
        let arr = strategy
            .inverse_link_array(Array1::from(vec![705.0]).view())
            .expect("array");
        assert_eq!(arr[0], exact, "inverse_link_array must be exact exp(705)");

        // η = −720: exact underflows toward 0 (≈2.03e−313); the clamp would pin
        // it at exp(−700) ≈ 9.86e−305 (~4.85e8× too large).
        let exact_neg = (-720.0_f64).exp();
        let jet = strategy.inverse_link_jet(-720.0).expect("jet");
        assert_eq!(jet.mu, exact_neg, "predict mean must be exact exp(-720)");
        let clamped_neg = (-700.0_f64).exp();
        assert!(
            jet.mu < clamped_neg,
            "exact exp(-720) must be strictly below the clamped exp(-700)"
        );

        // True IEEE limits honored exactly on the public surface.
        let over = strategy.inverse_link_jet(710.0).expect("jet");
        assert!(over.mu.is_infinite() && over.mu > 0.0, "exp(710) -> +inf");
        let under = strategy.inverse_link_jet(-746.0).expect("jet");
        assert_eq!(under.mu, 0.0, "exp(-746) -> 0.0");
    }

    /// No regression: for in-range η (|η| ≤ 700 — where the solver clamp is
    /// inert) the public predict jet is BYTE-IDENTICAL to the pre-fix clamped
    /// jet across mu and all derivatives. Only the out-of-range tails change.
    #[test]
    fn public_predict_log_jet_byte_identical_to_clamped_in_range() {
        let spec = LikelihoodSpec::poisson_log();
        let strategy = strategy_for_spec(&spec);
        for &eta in &[
            -700.0, -300.0, -12.5, -1.0, -0.25, 0.0, 0.25, 1.0, 12.5, 300.0, 700.0,
        ] {
            let public_jet = strategy.inverse_link_jet(eta).expect("public jet");
            let clamped_jet = inverse_link_jet_for_family(&spec, eta).expect("clamped jet");
            assert_eq!(
                public_jet.mu.to_bits(),
                clamped_jet.mu.to_bits(),
                "mu must be byte-identical in range at eta={eta}"
            );
            assert_eq!(
                public_jet.d1.to_bits(),
                clamped_jet.d1.to_bits(),
                "d1 must be byte-identical in range at eta={eta}"
            );
            assert_eq!(public_jet.d2.to_bits(), clamped_jet.d2.to_bits());
            assert_eq!(public_jet.d3.to_bits(), clamped_jet.d3.to_bits());
        }
    }
}
