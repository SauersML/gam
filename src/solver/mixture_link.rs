use crate::estimate::EstimationError;
use crate::probability::{normal_cdf, normal_pdf};
use crate::types::{
    InverseLink, LikelihoodFamily, LinkComponent, LinkFunction, MixtureLinkSpec, MixtureLinkState,
    SasLinkSpec, SasLinkState,
};
use ndarray::Array1;
use statrs::function::beta::{beta_reg, ln_beta};
use statrs::function::gamma::digamma;

const SAS_U_CLAMP: f64 = 50.0;
const SAS_LOG_DELTA_BOUND: f64 = 12.0;
const BETA_LOGISTIC_U_EPS: f64 = 1e-12;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InverseLinkJet {
    pub mu: f64,
    pub d1: f64,
    pub d2: f64,
    pub d3: f64,
}

#[inline]
fn canonicalzero(v: f64) -> f64 {
    if v.abs() < 1e-12 { 0.0 } else { v }
}

#[inline]
fn canonicalize_jet(mut jet: InverseLinkJet) -> InverseLinkJet {
    jet.d1 = canonicalzero(jet.d1);
    jet.d2 = canonicalzero(jet.d2);
    jet.d3 = canonicalzero(jet.d3);
    jet
}

#[inline]
fn logistic_stable(eta: f64) -> f64 {
    if eta.is_nan() {
        f64::NAN
    } else if eta == f64::INFINITY {
        1.0
    } else if eta == f64::NEG_INFINITY {
        0.0
    } else if eta >= 0.0 {
        let z = (-eta).exp();
        1.0 / (1.0 + z)
    } else {
        let z = eta.exp();
        z / (1.0 + z)
    }
}

#[inline]
fn probit_jet(eta: f64) -> InverseLinkJet {
    // Exact probit semantics:
    //
    //   mu(eta) = Phi(eta),
    //   mu'     = phi(eta),
    //   mu''    = -eta * phi(eta),
    //   mu'''   = (eta^2 - 1) * phi(eta).
    //
    // `normal_cdf` now evaluates the exact special-function form
    // Phi(x) = 0.5 * erfc(-x / sqrt(2)), so the jet can and should use the
    // matching closed-form Gaussian identities directly.
    if eta.is_nan() {
        return InverseLinkJet {
            mu: f64::NAN,
            d1: f64::NAN,
            d2: f64::NAN,
            d3: f64::NAN,
        };
    }
    if eta == f64::INFINITY {
        return InverseLinkJet {
            mu: 1.0,
            d1: 0.0,
            d2: 0.0,
            d3: 0.0,
        };
    }
    if eta == f64::NEG_INFINITY {
        return InverseLinkJet {
            mu: 0.0,
            d1: 0.0,
            d2: 0.0,
            d3: 0.0,
        };
    }
    let x = eta;
    let phi = normal_pdf(x);
    InverseLinkJet {
        mu: normal_cdf(x),
        d1: phi,
        d2: -x * phi,
        d3: (x * x - 1.0) * phi,
    }
}

#[inline]
fn probit_pdfthird_derivative(eta: f64) -> f64 {
    // Since d1 = mu' = phi(eta), this returns
    //
    //   d³/deta³ d1 = mu'''' = -(eta³ - 3 eta) phi(eta).
    if eta.is_nan() {
        return f64::NAN;
    }
    if !eta.is_finite() {
        return 0.0;
    }
    let x = eta;
    let phi = normal_pdf(x);
    canonicalzero(-(x * x * x - 3.0 * x) * phi)
}

#[inline]
fn chain_inverse_link_jet(base: InverseLinkJet, z1: f64, z2: f64, z3: f64) -> InverseLinkJet {
    InverseLinkJet {
        mu: base.mu,
        d1: base.d1 * z1,
        d2: base.d2 * z1 * z1 + base.d1 * z2,
        d3: base.d3 * z1 * z1 * z1 + 3.0 * base.d2 * z1 * z2 + base.d1 * z3,
    }
}

#[inline]
fn component_inverse_link_pdfthird_derivative(component: LinkComponent, eta: f64) -> f64 {
    match component {
        LinkComponent::Probit => probit_pdfthird_derivative(eta),
        LinkComponent::Logit => {
            // Logistic link:
            //   mu'   = d1 = mu(1-mu)
            //   d1'   = d2 = d1(1-2mu)
            //   d2'   = d3 = d1(1-6mu+6mu²)
            //   d3'   = d4 = d1(1-14mu+36mu²-24mu³).
            if eta.is_nan() {
                return f64::NAN;
            }
            if !eta.is_finite() {
                return 0.0;
            }
            let mu = logistic_stable(eta);
            let d1 = mu * (1.0 - mu);
            d1 * (1.0 - 14.0 * mu + 36.0 * mu * mu - 24.0 * mu * mu * mu)
        }
        LinkComponent::CLogLog => {
            // CLogLog link:
            //   mu = 1 - exp(-t),  t = exp(eta),  d1 = t exp(-t).
            //
            // Repeated differentiation closes in the basis `d1 * poly(t)`:
            //   d2 = d1(-t + 1)
            //   d3 = d1(t² - 3t + 1)
            //   d4 = d1(-t³ + 6t² - 7t + 1).
            if eta.is_nan() {
                return f64::NAN;
            }
            if !eta.is_finite() {
                return 0.0;
            }
            let t = eta.exp();
            let d1 = t * (-t).exp();
            d1 * (1.0 - 7.0 * t + 6.0 * t * t - t * t * t)
        }
        LinkComponent::LogLog => {
            // LogLog link is the reflected cloglog family with `r = exp(-eta)`:
            //   mu = exp(-r), d1 = mu r,
            // and again higher derivatives are `d1 * poly(r)`:
            //   d2 = d1(r - 1)
            //   d3 = d1(r² - 3r + 1)
            //   d4 = d1(r³ - 6r² + 7r - 1).
            if eta.is_nan() {
                return f64::NAN;
            }
            if !eta.is_finite() {
                return 0.0;
            }
            let r = (-eta).exp();
            let d1 = (-r).exp() * r;
            d1 * (r * r * r - 6.0 * r * r + 7.0 * r - 1.0)
        }
        LinkComponent::Cauchit => {
            // Cauchit link:
            //   mu = 1/2 + atan(eta)/pi,
            //   d1 = 1 / [pi (1+eta²)].
            //
            // Differentiating three more times gives
            //
            //   d4 = 24 eta (1-eta²) / [pi (1+eta²)^4].
            if eta.is_nan() {
                return f64::NAN;
            }
            if !eta.is_finite() {
                return 0.0;
            }
            let denom = 1.0 + eta * eta;
            24.0 * eta * (1.0 - eta * eta) / (std::f64::consts::PI * denom.powi(4))
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MixtureJetWithRhoPartials {
    pub jet: InverseLinkJet,
    /// Partial derivatives wrt free logits rho_j, j in [0, K-2].
    /// Each entry stores derivatives of (mu, d1, d2, d3) wrt one rho_j.
    pub djet_drho: Vec<InverseLinkJet>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SasJetWithParamPartials {
    pub jet: InverseLinkJet,
    pub djet_depsilon: InverseLinkJet,
    pub djet_dlog_delta: InverseLinkJet,
}

#[derive(Clone, Debug, PartialEq)]
pub enum LinkParamPartials {
    Mixture(MixtureJetWithRhoPartials),
    Sas(SasJetWithParamPartials),
}

/// Trait-based inverse-link kernel interface.
///
/// Implementors provide pointwise inverse-link derivatives wrt `eta`:
/// `F(eta), F'(eta), F''(eta), F'''(eta)`.
/// Optionally they may expose parameter partials used by outer-loop optimization.
pub trait InverseLinkKernel {
    fn jet(&self, eta: f64) -> Result<InverseLinkJet, EstimationError>;

    fn param_partials(&self, _: f64) -> Result<Option<LinkParamPartials>, EstimationError> {
        Ok(None)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProbitLinkKernel;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LogitLinkKernel;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CLogLogLinkKernel;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LogLogLinkKernel;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CauchitLinkKernel;

impl SasLinkState {
    /// Construct SAS state from raw optimizer parameters using the same bounded
    /// transform used everywhere in fitting/evaluation.
    pub fn new(raw_epsilon: f64, raw_log_delta: f64) -> Result<Self, String> {
        if !raw_epsilon.is_finite() || !raw_log_delta.is_finite() {
            return Err("SAS link parameters must be finite".to_string());
        }
        Ok(Self {
            epsilon: raw_epsilon,
            log_delta: raw_log_delta,
            delta: sas_delta_from_raw_log_delta(raw_log_delta),
        })
    }
}

pub fn state_from_sasspec(spec: SasLinkSpec) -> Result<SasLinkState, String> {
    SasLinkState::new(spec.initial_epsilon, spec.initial_log_delta)
}

pub fn state_from_beta_logisticspec(spec: SasLinkSpec) -> Result<SasLinkState, String> {
    if !spec.initial_epsilon.is_finite() || !spec.initial_log_delta.is_finite() {
        return Err("Beta-Logistic link parameters must be finite".to_string());
    }
    let delta_raw = spec.initial_log_delta;
    Ok(SasLinkState {
        epsilon: spec.initial_epsilon,
        log_delta: delta_raw,
        delta: delta_raw.exp(),
    })
}

#[inline]
fn tanh_bound(value: f64, bound: f64) -> f64 {
    let b = bound.max(f64::EPSILON);
    b * (value / b).tanh()
}

#[inline]
fn tanh_bound_d1(value: f64, bound: f64) -> f64 {
    let b = bound.max(f64::EPSILON);
    let t = (value / b).tanh();
    1.0 - t * t
}

#[inline]
fn tanh_bound_d2(value: f64, bound: f64) -> f64 {
    let b = bound.max(f64::EPSILON);
    let t = (value / b).tanh();
    let s = 1.0 - t * t;
    -2.0 * t * s / b
}

#[inline]
fn tanh_bound_d3(value: f64, bound: f64) -> f64 {
    let b = bound.max(f64::EPSILON);
    let t = (value / b).tanh();
    let s = 1.0 - t * t;
    -2.0 * s * (1.0 - 3.0 * t * t) / (b * b)
}

#[inline]
fn tanh_bound_d4(value: f64, bound: f64) -> f64 {
    let b = bound.max(f64::EPSILON);
    let t = (value / b).tanh();
    let s = 1.0 - t * t;
    8.0 * t * s * (2.0 - 3.0 * t * t) / (b * b * b)
}

#[inline]
fn sas_effective_log_delta(raw_log_delta: f64) -> (f64, f64) {
    let ld_eff = tanh_bound(raw_log_delta, SAS_LOG_DELTA_BOUND);
    let dld_eff_draw = tanh_bound_d1(raw_log_delta, SAS_LOG_DELTA_BOUND);
    (ld_eff, dld_eff_draw)
}

#[inline]
fn sas_delta_from_raw_log_delta(raw_log_delta: f64) -> f64 {
    let (ld_eff, _) = sas_effective_log_delta(raw_log_delta);
    ld_eff.exp()
}

pub fn validate_mixturespec(spec: &MixtureLinkSpec) -> Result<(), String> {
    if spec.components.is_empty() {
        return Err("mixture link requires at least 1 component".to_string());
    }
    if spec.initial_rho.len() + 1 != spec.components.len() {
        return Err(format!(
            "mixture link rho length mismatch: expected {}, got {}",
            spec.components.len() - 1,
            spec.initial_rho.len()
        ));
    }
    for i in 0..spec.components.len() {
        for j in (i + 1)..spec.components.len() {
            if spec.components[i] == spec.components[j] {
                return Err("mixture link components must be unique".to_string());
            }
        }
    }
    Ok(())
}

pub fn softmax_last_fixedzero(rho: &Array1<f64>) -> Array1<f64> {
    let k = rho.len() + 1;
    let mut logits = Vec::with_capacity(k);
    let mut maxv = 0.0_f64;
    for &v in rho {
        maxv = maxv.max(v);
        logits.push(v);
    }
    maxv = maxv.max(0.0);
    logits.push(0.0);

    let mut sum = 0.0_f64;
    let mut exps = vec![0.0_f64; k];
    for i in 0..k {
        let e = (logits[i] - maxv).exp();
        exps[i] = e;
        sum += e;
    }
    if !sum.is_finite() || sum <= 0.0 {
        return Array1::from_elem(k, 1.0 / k as f64);
    }
    let inv = 1.0 / sum;
    Array1::from_iter(exps.into_iter().map(|v| v * inv))
}

/// Returns softmax weights and Jacobian wrt free logits (last logit fixed at zero).
/// Jacobian shape is (K, K-1): d pi_k / d rho_j.
pub fn softmaxwith_jacobian_last_fixedzero(
    rho: &Array1<f64>,
) -> (Array1<f64>, ndarray::Array2<f64>) {
    let pi = softmax_last_fixedzero(rho);
    let k = pi.len();
    let m = k.saturating_sub(1);
    let mut jac = ndarray::Array2::<f64>::zeros((k, m));
    for j in 0..m {
        let pi_j = pi[j];
        for kk in 0..k {
            let delta = if kk == j { 1.0 } else { 0.0 };
            jac[[kk, j]] = pi[kk] * (delta - pi_j);
        }
    }
    (pi, jac)
}

pub fn state_fromspec(spec: &MixtureLinkSpec) -> Result<MixtureLinkState, String> {
    validate_mixturespec(spec)?;
    let pi = softmax_last_fixedzero(&spec.initial_rho);
    Ok(MixtureLinkState {
        components: spec.components.clone(),
        rho: spec.initial_rho.clone(),
        pi,
    })
}

#[inline]
pub fn component_inverse_link_jet(component: LinkComponent, eta: f64) -> InverseLinkJet {
    canonicalize_jet(match component {
        LinkComponent::Logit => {
            if eta.is_nan() {
                return InverseLinkJet {
                    mu: f64::NAN,
                    d1: f64::NAN,
                    d2: f64::NAN,
                    d3: f64::NAN,
                };
            }
            let mu = logistic_stable(eta);
            let d1 = if eta.is_finite() {
                mu * (1.0 - mu)
            } else {
                0.0
            };
            let d2 = if eta.is_finite() {
                d1 * (1.0 - 2.0 * mu)
            } else {
                0.0
            };
            let d3 = if eta.is_finite() {
                d1 * (1.0 - 6.0 * d1)
            } else {
                0.0
            };
            InverseLinkJet { mu, d1, d2, d3 }
        }
        LinkComponent::Probit => probit_jet(eta),
        LinkComponent::CLogLog => {
            if eta.is_nan() {
                return InverseLinkJet {
                    mu: f64::NAN,
                    d1: f64::NAN,
                    d2: f64::NAN,
                    d3: f64::NAN,
                };
            }
            if eta == f64::INFINITY {
                return InverseLinkJet {
                    mu: 1.0,
                    d1: 0.0,
                    d2: 0.0,
                    d3: 0.0,
                };
            }
            if eta == f64::NEG_INFINITY {
                return InverseLinkJet {
                    mu: 0.0,
                    d1: 0.0,
                    d2: 0.0,
                    d3: 0.0,
                };
            }
            let t = eta.exp();
            let s = (-t).exp();
            let d1 = t * s;
            let d2 = -d1 * (t - 1.0);
            let d3 = d1 * (t * t - 3.0 * t + 1.0);
            InverseLinkJet {
                mu: 1.0 - s,
                d1,
                d2,
                d3,
            }
        }
        LinkComponent::LogLog => {
            if eta.is_nan() {
                return InverseLinkJet {
                    mu: f64::NAN,
                    d1: f64::NAN,
                    d2: f64::NAN,
                    d3: f64::NAN,
                };
            }
            if eta == f64::INFINITY {
                return InverseLinkJet {
                    mu: 1.0,
                    d1: 0.0,
                    d2: 0.0,
                    d3: 0.0,
                };
            }
            if eta == f64::NEG_INFINITY {
                return InverseLinkJet {
                    mu: 0.0,
                    d1: 0.0,
                    d2: 0.0,
                    d3: 0.0,
                };
            }
            let r = (-eta).exp();
            let mu = (-r).exp();
            let d1 = mu * r;
            let d2 = d1 * (r - 1.0);
            let d3 = d1 * (r * r - 3.0 * r + 1.0);
            InverseLinkJet { mu, d1, d2, d3 }
        }
        LinkComponent::Cauchit => {
            if eta.is_nan() {
                return InverseLinkJet {
                    mu: f64::NAN,
                    d1: f64::NAN,
                    d2: f64::NAN,
                    d3: f64::NAN,
                };
            }
            let den = 1.0 + eta * eta;
            let d1 = if eta.is_finite() {
                1.0 / (std::f64::consts::PI * den)
            } else {
                0.0
            };
            let d2 = if eta.is_finite() {
                -2.0 * eta / (std::f64::consts::PI * den * den)
            } else {
                0.0
            };
            let d3 = if eta.is_finite() {
                (6.0 * eta * eta - 2.0) / (std::f64::consts::PI * den * den * den)
            } else {
                0.0
            };
            InverseLinkJet {
                mu: 0.5 + eta.atan() / std::f64::consts::PI,
                d1,
                d2,
                d3,
            }
        }
    })
}

impl InverseLinkKernel for ProbitLinkKernel {
    #[inline]
    fn jet(&self, eta: f64) -> Result<InverseLinkJet, EstimationError> {
        Ok(component_inverse_link_jet(LinkComponent::Probit, eta))
    }
}

impl InverseLinkKernel for LogitLinkKernel {
    #[inline]
    fn jet(&self, eta: f64) -> Result<InverseLinkJet, EstimationError> {
        Ok(component_inverse_link_jet(LinkComponent::Logit, eta))
    }
}

impl InverseLinkKernel for CLogLogLinkKernel {
    #[inline]
    fn jet(&self, eta: f64) -> Result<InverseLinkJet, EstimationError> {
        Ok(component_inverse_link_jet(LinkComponent::CLogLog, eta))
    }
}

impl InverseLinkKernel for LogLogLinkKernel {
    #[inline]
    fn jet(&self, eta: f64) -> Result<InverseLinkJet, EstimationError> {
        Ok(component_inverse_link_jet(LinkComponent::LogLog, eta))
    }
}

impl InverseLinkKernel for CauchitLinkKernel {
    #[inline]
    fn jet(&self, eta: f64) -> Result<InverseLinkJet, EstimationError> {
        Ok(component_inverse_link_jet(LinkComponent::Cauchit, eta))
    }
}

impl InverseLinkKernel for LinkComponent {
    #[inline]
    fn jet(&self, eta: f64) -> Result<InverseLinkJet, EstimationError> {
        Ok(component_inverse_link_jet(*self, eta))
    }
}

impl InverseLinkKernel for LinkFunction {
    fn jet(&self, eta: f64) -> Result<InverseLinkJet, EstimationError> {
        match self {
            LinkFunction::Logit => LogitLinkKernel.jet(eta),
            LinkFunction::Probit => ProbitLinkKernel.jet(eta),
            LinkFunction::CLogLog => CLogLogLinkKernel.jet(eta),
            LinkFunction::Identity => Ok(InverseLinkJet {
                mu: eta,
                d1: 1.0,
                d2: 0.0,
                d3: 0.0,
            }),
            LinkFunction::Sas => Err(EstimationError::InvalidInput(
                "LinkFunction::Sas inverse-link requires explicit SAS link state".to_string(),
            )),
            LinkFunction::BetaLogistic => Err(EstimationError::InvalidInput(
                "LinkFunction::BetaLogistic inverse-link requires explicit Beta-Logistic link state"
                    .to_string(),
            )),
        }
    }
}

impl InverseLinkKernel for SasLinkState {
    fn jet(&self, eta: f64) -> Result<InverseLinkJet, EstimationError> {
        Ok(sas_inverse_link_jet(eta, self.epsilon, self.log_delta))
    }

    fn param_partials(&self, eta: f64) -> Result<Option<LinkParamPartials>, EstimationError> {
        Ok(Some(LinkParamPartials::Sas(
            sas_inverse_link_jetwith_param_partials(eta, self.epsilon, self.log_delta),
        )))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BetaLogisticKernel {
    pub delta: f64,
    pub epsilon: f64,
}

impl InverseLinkKernel for BetaLogisticKernel {
    fn jet(&self, eta: f64) -> Result<InverseLinkJet, EstimationError> {
        Ok(beta_logistic_inverse_link_jet(
            eta,
            self.delta,
            self.epsilon,
        ))
    }

    fn param_partials(&self, eta: f64) -> Result<Option<LinkParamPartials>, EstimationError> {
        Ok(Some(LinkParamPartials::Sas(
            beta_logistic_inverse_link_jetwith_param_partials(eta, self.delta, self.epsilon),
        )))
    }
}

impl InverseLinkKernel for MixtureLinkState {
    fn jet(&self, eta: f64) -> Result<InverseLinkJet, EstimationError> {
        Ok(mixture_inverse_link_jet(self, eta))
    }

    fn param_partials(&self, eta: f64) -> Result<Option<LinkParamPartials>, EstimationError> {
        Ok(Some(LinkParamPartials::Mixture(
            mixture_inverse_link_jetwith_rho_partials(self, eta),
        )))
    }
}

impl InverseLinkKernel for InverseLink {
    fn jet(&self, eta: f64) -> Result<InverseLinkJet, EstimationError> {
        match self {
            InverseLink::Standard(link_fn) => link_fn.jet(eta),
            InverseLink::Sas(state) => state.jet(eta),
            InverseLink::BetaLogistic(state) => BetaLogisticKernel {
                delta: state.log_delta,
                epsilon: state.epsilon,
            }
            .jet(eta),
            InverseLink::Mixture(state) => state.jet(eta),
        }
    }

    fn param_partials(&self, eta: f64) -> Result<Option<LinkParamPartials>, EstimationError> {
        match self {
            InverseLink::Standard(_) => Ok(None),
            InverseLink::Sas(state) => state.param_partials(eta),
            InverseLink::BetaLogistic(state) => BetaLogisticKernel {
                delta: state.log_delta,
                epsilon: state.epsilon,
            }
            .param_partials(eta),
            InverseLink::Mixture(state) => state.param_partials(eta),
        }
    }
}

/// Central family-aware inverse-link jet dispatch.
///
/// For `BinomialSas` and `BinomialMixture`, required state must be provided.
pub fn inverse_link_jet_for_inverse_link(
    link: &InverseLink,
    eta: f64,
) -> Result<InverseLinkJet, EstimationError> {
    link.jet(eta)
}

pub fn inverse_link_pdfthird_derivative_for_inverse_link(
    link: &InverseLink,
    eta: f64,
) -> Result<f64, EstimationError> {
    // This dispatch returns the fourth eta-derivative of the inverse-link CDF,
    // equivalently the third derivative of the inverse-link density
    //
    //   f(eta) = d/deta mu(eta).
    //
    // It is used downstream as the `f'''` input in
    //
    //   d³/deta³ log f = f'''/f - 3 f'f''/f² + 2(f')³/f³.
    //
    // Mixture links preserve linearity:
    //
    //   mu = sum_j pi_j mu_j
    //   => f''' = sum_j pi_j f_j'''
    //
    // because the mixture weights `pi_j` are constant with respect to `eta`.
    match link {
        InverseLink::Standard(LinkFunction::Identity) => Ok(0.0),
        InverseLink::Standard(LinkFunction::Probit) => Ok(probit_pdfthird_derivative(eta)),
        InverseLink::Standard(LinkFunction::Logit) => Ok(
            component_inverse_link_pdfthird_derivative(LinkComponent::Logit, eta),
        ),
        InverseLink::Standard(LinkFunction::CLogLog) => Ok(
            component_inverse_link_pdfthird_derivative(LinkComponent::CLogLog, eta),
        ),
        InverseLink::Standard(LinkFunction::Sas) => {
            Ok(sas_inverse_link_pdfthird_derivative(eta, 0.0, 0.0))
        }
        InverseLink::Sas(state) => Ok(sas_inverse_link_pdfthird_derivative(
            eta,
            state.epsilon,
            state.log_delta,
        )),
        InverseLink::Standard(LinkFunction::BetaLogistic) => Ok(
            beta_logistic_inverse_link_pdfthird_derivative(eta, 0.0, 0.0),
        ),
        InverseLink::BetaLogistic(state) => Ok(beta_logistic_inverse_link_pdfthird_derivative(
            eta,
            state.log_delta,
            state.epsilon,
        )),
        InverseLink::Mixture(state) => Ok(state
            .components
            .iter()
            .zip(state.pi.iter())
            .map(|(&component, &weight)| {
                weight * component_inverse_link_pdfthird_derivative(component, eta)
            })
            .sum()),
    }
}

#[inline]
pub fn inverse_link_param_partials_for_inverse_link(
    link: &InverseLink,
    eta: f64,
) -> Result<Option<LinkParamPartials>, EstimationError> {
    link.param_partials(eta)
}

pub fn inverse_link_jet_for_link_function(
    link: LinkFunction,
    eta: f64,
    mixture_link_state: Option<&MixtureLinkState>,
    sas_link_state: Option<&SasLinkState>,
) -> Result<InverseLinkJet, EstimationError> {
    if let Some(state) = mixture_link_state {
        return state.jet(eta);
    }
    if let Some(sas) = sas_link_state {
        return match link {
            LinkFunction::BetaLogistic => BetaLogisticKernel {
                delta: sas.log_delta,
                epsilon: sas.epsilon,
            }
            .jet(eta),
            _ => sas.jet(eta),
        };
    }
    link.jet(eta)
}

pub fn inverse_link_jet_for_family(
    family: LikelihoodFamily,
    eta: f64,
    mixture_link_state: Option<&MixtureLinkState>,
    sas_link_state: Option<&SasLinkState>,
) -> Result<InverseLinkJet, EstimationError> {
    match family {
        LikelihoodFamily::GaussianIdentity => inverse_link_jet_for_link_function(
            LinkFunction::Identity,
            eta,
            mixture_link_state,
            sas_link_state,
        ),
        LikelihoodFamily::BinomialLogit => inverse_link_jet_for_link_function(
            LinkFunction::Logit,
            eta,
            mixture_link_state,
            sas_link_state,
        ),
        LikelihoodFamily::BinomialProbit => inverse_link_jet_for_link_function(
            LinkFunction::Probit,
            eta,
            mixture_link_state,
            sas_link_state,
        ),
        LikelihoodFamily::BinomialCLogLog => inverse_link_jet_for_link_function(
            LinkFunction::CLogLog,
            eta,
            mixture_link_state,
            sas_link_state,
        ),
        LikelihoodFamily::BinomialSas => inverse_link_jet_for_link_function(
            LinkFunction::Sas,
            eta,
            mixture_link_state,
            sas_link_state,
        )
        .map_err(|_| {
            EstimationError::InvalidInput(
                "BinomialSas inverse-link requires SAS link state".to_string(),
            )
        }),
        LikelihoodFamily::BinomialBetaLogistic => inverse_link_jet_for_link_function(
            LinkFunction::BetaLogistic,
            eta,
            mixture_link_state,
            sas_link_state,
        )
        .map_err(|_| {
            EstimationError::InvalidInput(
                "BinomialBetaLogistic inverse-link requires Beta-Logistic link state".to_string(),
            )
        }),
        LikelihoodFamily::BinomialMixture => {
            let state = mixture_link_state.ok_or_else(|| {
                EstimationError::InvalidInput(
                    "BinomialMixture inverse-link requires mixture link state".to_string(),
                )
            })?;
            inverse_link_jet_for_link_function(
                LinkFunction::Logit,
                eta,
                Some(state),
                sas_link_state,
            )
        }
        LikelihoodFamily::RoystonParmar => Err(EstimationError::InvalidInput(
            "RoystonParmar inverse-link jet is not defined in mixture-link dispatcher".to_string(),
        )),
    }
}

#[inline]
pub fn mixture_inverse_link_jet(state: &MixtureLinkState, eta: f64) -> InverseLinkJet {
    let mut mu = 0.0_f64;
    let mut d1 = 0.0_f64;
    let mut d2 = 0.0_f64;
    let mut d3 = 0.0_f64;
    let k = state.components.len().min(state.pi.len());
    for i in 0..k {
        let jet = component_inverse_link_jet(state.components[i], eta);
        let w = state.pi[i];
        mu += w * jet.mu;
        d1 += w * jet.d1;
        d2 += w * jet.d2;
        d3 += w * jet.d3;
    }
    InverseLinkJet { mu, d1, d2, d3 }
}

/// Computes mixture jet and exact partial derivatives wrt free softmax logits.
///
/// Uses identities:
///   d mu     / d rho_j = pi_j (mu_j     - mu)
///   d mu'    / d rho_j = pi_j (mu_j'    - mu')
///   d mu''   / d rho_j = pi_j (mu_j''   - mu'')
///   d mu'''  / d rho_j = pi_j (mu_j'''  - mu''')
pub fn mixture_inverse_link_jetwith_rho_partials(
    state: &MixtureLinkState,
    eta: f64,
) -> MixtureJetWithRhoPartials {
    let k = state.components.len().min(state.pi.len());
    let m = k.saturating_sub(1);
    let mut djet_drho = vec![
        InverseLinkJet {
            mu: 0.0,
            d1: 0.0,
            d2: 0.0,
            d3: 0.0,
        };
        m
    ];
    let jet = mixture_inverse_link_jetwith_rho_partials_into(state, eta, &mut djet_drho);
    MixtureJetWithRhoPartials { jet, djet_drho }
}

/// Computes mixture jet and writes exact rho partial jets into `out` (length >= K-1).
/// This avoids heap allocation in hot loops.
pub fn mixture_inverse_link_jetwith_rho_partials_into(
    state: &MixtureLinkState,
    eta: f64,
    out: &mut [InverseLinkJet],
) -> InverseLinkJet {
    let k = state.components.len().min(state.pi.len());
    let m = k.saturating_sub(1);
    assert!(
        out.len() >= m,
        "rho-partial output buffer too small: got {}, need {}",
        out.len(),
        m
    );
    let mut mixed = InverseLinkJet {
        mu: 0.0,
        d1: 0.0,
        d2: 0.0,
        d3: 0.0,
    };
    for i in 0..k {
        let jet_i = component_inverse_link_jet(state.components[i], eta);
        let w = state.pi[i];
        mixed.mu += w * jet_i.mu;
        mixed.d1 += w * jet_i.d1;
        mixed.d2 += w * jet_i.d2;
        mixed.d3 += w * jet_i.d3;
        // Cache the first K-1 component jets directly in the output buffer so
        // we don't recompute them in the partial loop.
        if i < m {
            out[i] = jet_i;
        }
    }
    for j in 0..m {
        let pi_j = state.pi[j];
        let cj = out[j];
        out[j] = InverseLinkJet {
            mu: pi_j * (cj.mu - mixed.mu),
            d1: pi_j * (cj.d1 - mixed.d1),
            d2: pi_j * (cj.d2 - mixed.d2),
            d3: pi_j * (cj.d3 - mixed.d3),
        };
    }
    mixed
}

#[inline]
fn logistic_uwith_derivatives(eta: f64) -> (f64, f64) {
    let u = logistic_stable(eta);
    let u_clamped = u.clamp(BETA_LOGISTIC_U_EPS, 1.0 - BETA_LOGISTIC_U_EPS);
    let clamp_active = !eta.is_finite() || u_clamped != u;
    let du = if clamp_active {
        0.0
    } else {
        u_clamped * (1.0 - u_clamped)
    };
    let u = u_clamped;
    (u, du)
}

/// Beta-Logistic inverse-link jet for:
///   u = logistic(eta)
///   a = exp(delta - epsilon), b = exp(delta + epsilon)
///   mu = I_u(a, b)
pub fn beta_logistic_inverse_link_jet(eta: f64, delta: f64, epsilon: f64) -> InverseLinkJet {
    let (u, du) = logistic_uwith_derivatives(eta);
    let a = (delta - epsilon).exp();
    let b = (delta + epsilon).exp();
    let mu = beta_reg(a, b, u);
    if du == 0.0 {
        return InverseLinkJet {
            mu,
            d1: 0.0,
            d2: 0.0,
            d3: 0.0,
        };
    }
    let log_d1 = a * u.ln() + b * (1.0 - u).ln() - ln_beta(a, b);
    let d1 = log_d1.exp();
    let t = a * (1.0 - u) - b * u;
    let d2 = d1 * t;
    let d3 = d1 * (t * t - (a + b) * du);
    InverseLinkJet { mu, d1, d2, d3 }
}

pub fn beta_logistic_inverse_link_pdfthird_derivative(eta: f64, delta: f64, epsilon: f64) -> f64 {
    // Beta-logistic link:
    //
    //   u = logistic(eta),
    //   d1 = C * u^a (1-u)^b,
    //   t  = a(1-u) - b u,
    //   c  = a + b,
    //
    // so
    //
    //   d2 = d1 * t
    //   d3 = d1 * (t² - c u')
    //
    // with `u' = u(1-u)`.
    //
    // Differentiate once more:
    //
    //   d4 = d/deta[d1 (t² - c u')]
    //      = d1' (t² - c u') + d1 (2 t t' - c u'')
    //      = d1 [ t(t² - c u') - 2 c t u' - c u'' ]
    //      = d1 [ t³ - 3 c t u' - c u'' ],
    //
    // since `t' = -c u'`.
    let (u, du) = logistic_uwith_derivatives(eta);
    if du == 0.0 {
        return 0.0;
    }
    let a = (delta - epsilon).exp();
    let b = (delta + epsilon).exp();
    let log_d1 = a * u.ln() + b * (1.0 - u).ln() - ln_beta(a, b);
    let d1 = log_d1.exp();
    let c = a + b;
    let t = a * (1.0 - u) - b * u;
    let u2 = du * (1.0 - 2.0 * u);
    d1 * (t * t * t - 3.0 * c * t * du - c * u2)
}

pub fn beta_logistic_inverse_link_jetwith_param_partials(
    eta: f64,
    delta: f64,
    epsilon: f64,
) -> SasJetWithParamPartials {
    let (u, du) = logistic_uwith_derivatives(eta);
    let a = (delta - epsilon).exp();
    let b = (delta + epsilon).exp();
    let mu = beta_reg(a, b, u);
    if du == 0.0 {
        let zero = InverseLinkJet {
            mu,
            d1: 0.0,
            d2: 0.0,
            d3: 0.0,
        };
        return SasJetWithParamPartials {
            jet: zero,
            djet_depsilon: InverseLinkJet {
                mu: 0.0,
                d1: 0.0,
                d2: 0.0,
                d3: 0.0,
            },
            djet_dlog_delta: InverseLinkJet {
                mu: 0.0,
                d1: 0.0,
                d2: 0.0,
                d3: 0.0,
            },
        };
    }
    let log_d1 = a * u.ln() + b * (1.0 - u).ln() - ln_beta(a, b);
    let d1 = log_d1.exp();
    let t = a * (1.0 - u) - b * u;
    let d2 = d1 * t;
    let k = t * t - (a + b) * du;
    let d3 = d1 * k;
    let jet = InverseLinkJet { mu, d1, d2, d3 };

    let psi_a = digamma(a);
    let psi_b = digamma(b);
    let psi_ab = digamma(a + b);
    let la = u.ln() - psi_a + psi_ab;
    let lb = (1.0 - u).ln() - psi_b + psi_ab;

    let partials_for = |a_p: f64, b_p: f64, dmu: f64| -> InverseLinkJet {
        let logd1_p = a_p * la + b_p * lb;
        let d1_p = d1 * logd1_p;
        let t_p = a_p * (1.0 - u) - b_p * u;
        let d2_p = d1_p * t + d1 * t_p;
        let k_p = 2.0 * t * t_p - (a_p + b_p) * du;
        let d3_p = d1_p * k + d1 * k_p;
        InverseLinkJet {
            mu: dmu,
            d1: d1_p,
            d2: d2_p,
            d3: d3_p,
        }
    };

    let mu_only = |d: f64, e: f64| -> f64 {
        let aa = (d - e).exp();
        let bb = (d + e).exp();
        beta_reg(aa, bb, u).clamp(BETA_LOGISTIC_U_EPS, 1.0 - BETA_LOGISTIC_U_EPS)
    };
    let h_delta = 1e-6 * (1.0 + delta.abs());
    let h_epsilon = 1e-6 * (1.0 + epsilon.abs());
    let dmu_ddelta =
        (mu_only(delta + h_delta, epsilon) - mu_only(delta - h_delta, epsilon)) / (2.0 * h_delta);
    let dmu_depsilon = (mu_only(delta, epsilon + h_epsilon) - mu_only(delta, epsilon - h_epsilon))
        / (2.0 * h_epsilon);
    let djet_ddelta = partials_for(a, b, dmu_ddelta);
    let djet_depsilon = partials_for(-a, b, dmu_depsilon);
    SasJetWithParamPartials {
        jet,
        djet_depsilon,
        djet_dlog_delta: djet_ddelta,
    }
}

/// SAS inverse-link jet for:
///   mu(eta) = Phi(sinh(delta * asinh(eta) - epsilon)),
///   delta = exp(B * tanh(log_delta / B)), B = SAS_LOG_DELTA_BOUND.
pub fn sas_inverse_link_jet(eta: f64, epsilon: f64, log_delta: f64) -> InverseLinkJet {
    let delta_id = sas_delta_from_raw_log_delta(log_delta);
    if epsilon.abs() < 1e-12 && (delta_id - 1.0).abs() < 1e-12 {
        return component_inverse_link_jet(LinkComponent::Probit, eta);
    }
    let e = if eta.is_finite() { eta } else { 0.0 };
    let a = e.asinh();
    let delta = delta_id;
    let u_raw = delta * a - epsilon;
    let u = tanh_bound(u_raw, SAS_U_CLAMP);
    let g1 = tanh_bound_d1(u_raw, SAS_U_CLAMP);
    let g2 = tanh_bound_d2(u_raw, SAS_U_CLAMP);
    let g3 = tanh_bound_d3(u_raw, SAS_U_CLAMP);
    let s = u.sinh();
    let c = u.cosh();
    let z = s;
    let q = e.hypot(1.0);
    let inv_q = 1.0 / q;
    let inv_q2 = inv_q * inv_q;
    let inv_q3 = inv_q2 * inv_q;
    let inv_q5 = inv_q3 * inv_q2;
    let r1 = delta * inv_q;
    let r2 = -delta * e * inv_q3;
    let r3 = delta * (2.0 * e * e - 1.0) * inv_q5;
    let u1 = g1 * r1;
    let u2 = g2 * r1 * r1 + g1 * r2;
    let u3 = g3 * r1 * r1 * r1 + 3.0 * g2 * r1 * r2 + g1 * r3;
    let z1 = c * u1;
    let z2 = s * u1 * u1 + c * u2;
    let z3 = c * u1 * u1 * u1 + 3.0 * s * u1 * u2 + c * u3;
    let base = probit_jet(z);
    chain_inverse_link_jet(base, z1, z2, z3)
}

pub fn sas_inverse_link_pdfthird_derivative(eta: f64, epsilon: f64, log_delta: f64) -> f64 {
    // SAS link with bounded latent transform:
    //
    //   a  = asinh(eta),
    //   u  = tanh_bound(delta * a - epsilon),
    //   z  = sinh(u),
    //   mu = Phi(z).
    //
    // Write:
    //
    //   z1 = z'
    //   z2 = z''
    //   z3 = z'''
    //   z4 = z''''.
    //
    // Since `mu' = phi(z) z1`, repeated differentiation factors through the
    // standard normal Hermite-polynomial identities:
    //
    //   mu''   = phi(z) [ z2 - z z1² ]
    //
    //   mu'''  = phi(z) [ z3 - 3 z z1 z2 + (z² - 1) z1³ ]
    //          = phi(z) k3
    //
    //   mu'''' = phi(z) [ k4 - z z1 k3 ],
    //
    // where `k4` is the derivative of `k3` after collecting like terms. The
    // code below computes `u1..u4`, then `z1..z4`, then `k3` and `k4`, exactly
    // matching that chain.
    //
    // The needed fourth derivative of `u(eta)` is obtained from the nested
    // composition `u(eta) = g(r(eta))` with
    //   g = tanh_bound, r = delta * asinh(eta) - epsilon:
    //
    //   u4 = g'''' r1^4 + 6 g''' r1² r2 + 3 g'' r2² + 4 g'' r1 r3 + g' r4,
    //
    // which is the standard scalar Faà di Bruno expansion for order four.
    let e = if eta.is_finite() { eta } else { 0.0 };
    let a = e.asinh();
    let delta = sas_delta_from_raw_log_delta(log_delta);
    let u_raw = delta * a - epsilon;
    let u = tanh_bound(u_raw, SAS_U_CLAMP);
    let g1 = tanh_bound_d1(u_raw, SAS_U_CLAMP);
    let g2 = tanh_bound_d2(u_raw, SAS_U_CLAMP);
    let g3 = tanh_bound_d3(u_raw, SAS_U_CLAMP);
    let g4 = tanh_bound_d4(u_raw, SAS_U_CLAMP);
    let s = u.sinh();
    let c = u.cosh();
    let z = s;
    let base = probit_jet(z);
    let q = e.hypot(1.0);
    let inv_q = 1.0 / q;
    let inv_q2 = inv_q * inv_q;
    let inv_q3 = inv_q2 * inv_q;
    let inv_q5 = inv_q3 * inv_q2;
    let inv_q7 = inv_q5 * inv_q2;
    let r1 = delta * inv_q;
    let r2 = -delta * e * inv_q3;
    let r3 = delta * (2.0 * e * e - 1.0) * inv_q5;
    let r4 = delta * e * (9.0 - 6.0 * e * e) * inv_q7;
    let u1 = g1 * r1;
    let u2 = g2 * r1 * r1 + g1 * r2;
    let u3 = g3 * r1 * r1 * r1 + 3.0 * g2 * r1 * r2 + g1 * r3;
    let u4 = g4 * r1.powi(4)
        + 6.0 * g3 * r1 * r1 * r2
        + 3.0 * g2 * r2 * r2
        + 4.0 * g2 * r1 * r3
        + g1 * r4;
    let z1 = c * u1;
    let z2 = s * u1 * u1 + c * u2;
    let z3 = c * u1 * u1 * u1 + 3.0 * s * u1 * u2 + c * u3;
    let z4 =
        s * u1.powi(4) + 6.0 * c * u1 * u1 * u2 + 3.0 * s * u2 * u2 + 4.0 * s * u1 * u3 + c * u4;
    let base4 = probit_pdfthird_derivative(z);
    let out = base4 * z1.powi(4)
        + 6.0 * base.d3 * z1 * z1 * z2
        + 3.0 * base.d2 * z2 * z2
        + 4.0 * base.d2 * z1 * z3
        + base.d1 * z4;
    canonicalzero(out)
}

pub fn sas_inverse_link_jetwith_param_partials(
    eta: f64,
    epsilon: f64,
    log_delta: f64,
) -> SasJetWithParamPartials {
    let e = if eta.is_finite() { eta } else { 0.0 };
    let a = e.asinh();
    let (ld_eff, dld_eff_draw) = sas_effective_log_delta(log_delta);
    let delta = ld_eff.exp();
    let ddelta_draw = delta * dld_eff_draw;
    let u_raw = delta * a - epsilon;
    let u = tanh_bound(u_raw, SAS_U_CLAMP);
    let g1 = tanh_bound_d1(u_raw, SAS_U_CLAMP);
    let g2 = tanh_bound_d2(u_raw, SAS_U_CLAMP);
    let g3 = tanh_bound_d3(u_raw, SAS_U_CLAMP);
    let g4 = tanh_bound_d4(u_raw, SAS_U_CLAMP);
    let s = u.sinh();
    let c = u.cosh();
    let z = s;
    let q = e.hypot(1.0);
    let inv_q = 1.0 / q;
    let inv_q2 = inv_q * inv_q;
    let inv_q3 = inv_q2 * inv_q;
    let inv_q5 = inv_q3 * inv_q2;
    let a1 = inv_q;
    let a2 = -e * inv_q3;
    let a3 = (2.0 * e * e - 1.0) * inv_q5;
    let r1 = delta * a1;
    let r2 = delta * a2;
    let r3 = delta * a3;
    let u1 = g1 * r1;
    let u2 = g2 * r1 * r1 + g1 * r2;
    let u3 = g3 * r1 * r1 * r1 + 3.0 * g2 * r1 * r2 + g1 * r3;
    let z1 = c * u1;
    let z2 = s * u1 * u1 + c * u2;
    let z3 = c * u1 * u1 * u1 + 3.0 * s * u1 * u2 + c * u3;

    let base = probit_jet(z);
    let jet = chain_inverse_link_jet(base, z1, z2, z3);

    // Generic chain for parameter t:
    // u_t, u1_t, u2_t, u3_t -> z_t,z1_t,z2_t,z3_t -> mu_t,d1_t,d2_t,d3_t
    let param_partials = |u_t: f64, u1_t: f64, u2_t: f64, u3_t: f64| -> InverseLinkJet {
        let z_t = c * u_t;
        let z1_t = s * u_t * u1 + c * u1_t;
        let z2_t = c * u_t * u1 * u1 + 2.0 * s * u1 * u1_t + s * u_t * u2 + c * u2_t;
        let z3_t = s * u_t * u1 * u1 * u1
            + 3.0 * c * u1 * u1 * u1_t
            + 3.0 * c * u_t * u1 * u2
            + 3.0 * s * (u1_t * u2 + u1 * u2_t)
            + s * u_t * u3
            + c * u3_t;

        InverseLinkJet {
            mu: base.d1 * z_t,
            d1: base.d2 * z_t * z1 + base.d1 * z1_t,
            d2: base.d3 * z_t * z1 * z1
                + 2.0 * base.d2 * z1 * z1_t
                + base.d2 * z_t * z2
                + base.d1 * z2_t,
            d3: probit_pdfthird_derivative(z) * z_t * z1.powi(3)
                + 3.0 * base.d3 * z1 * z1 * z1_t
                + 3.0 * base.d3 * z_t * z1 * z2
                + 3.0 * base.d2 * (z1_t * z2 + z1 * z2_t)
                + base.d2 * z_t * z3
                + base.d1 * z3_t,
        }
    };

    // epsilon partials (raw_u_t = -1).
    let rt_eps = -1.0;
    let r1t_eps = 0.0;
    let r2t_eps = 0.0;
    let r3t_eps = 0.0;
    let u_eps = g1 * rt_eps;
    let u1_eps = g2 * rt_eps * r1 + g1 * r1t_eps;
    let u2_eps = g3 * rt_eps * r1 * r1 + 2.0 * g2 * r1 * r1t_eps + g2 * rt_eps * r2 + g1 * r2t_eps;
    let u3_eps = g4 * rt_eps * r1 * r1 * r1
        + 3.0 * g3 * r1 * r1 * r1t_eps
        + 3.0 * g3 * rt_eps * r1 * r2
        + 3.0 * g2 * (r1t_eps * r2 + r1 * r2t_eps)
        + g2 * rt_eps * r3
        + g1 * r3t_eps;
    let djet_depsilon = param_partials(u_eps, u1_eps, u2_eps, u3_eps);

    // raw log-delta partials (through smooth bounded effective log-delta).
    let rt_ld = ddelta_draw * a;
    let r1t_ld = ddelta_draw * a1;
    let r2t_ld = ddelta_draw * a2;
    let r3t_ld = ddelta_draw * a3;
    let u_ld = g1 * rt_ld;
    let u1_ld = g2 * rt_ld * r1 + g1 * r1t_ld;
    let u2_ld = g3 * rt_ld * r1 * r1 + 2.0 * g2 * r1 * r1t_ld + g2 * rt_ld * r2 + g1 * r2t_ld;
    let u3_ld = g4 * rt_ld * r1 * r1 * r1
        + 3.0 * g3 * r1 * r1 * r1t_ld
        + 3.0 * g3 * rt_ld * r1 * r2
        + 3.0 * g2 * (r1t_ld * r2 + r1 * r2t_ld)
        + g2 * rt_ld * r3
        + g1 * r3t_ld;
    let djet_dlog_delta = param_partials(u_ld, u1_ld, u2_ld, u3_ld);

    SasJetWithParamPartials {
        jet,
        djet_depsilon,
        djet_dlog_delta,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{InverseLink, LinkComponent, LinkFunction, MixtureLinkSpec, SasLinkState};

    #[test]
    fn mixture_jet_rho_partials_matchfd() {
        let spec = MixtureLinkSpec {
            components: vec![
                LinkComponent::Probit,
                LinkComponent::Logit,
                LinkComponent::CLogLog,
                LinkComponent::Cauchit,
            ],
            initial_rho: Array1::from_vec(vec![0.3, -0.6, 0.2]),
        };
        let state = state_fromspec(&spec).expect("state");
        let eta = 0.35;
        let out = mixture_inverse_link_jetwith_rho_partials(&state, eta);
        let h = 1e-6;
        for j in 0..state.rho.len() {
            let mut rp = state.rho.clone();
            rp[j] += h;
            let sp = MixtureLinkSpec {
                components: state.components.clone(),
                initial_rho: rp,
            };
            let jp = mixture_inverse_link_jet(&state_fromspec(&sp).expect("sp"), eta);
            let mut rm = state.rho.clone();
            rm[j] -= h;
            let sm = MixtureLinkSpec {
                components: state.components.clone(),
                initial_rho: rm,
            };
            let jm = mixture_inverse_link_jet(&state_fromspec(&sm).expect("sm"), eta);
            let fd = InverseLinkJet {
                mu: (jp.mu - jm.mu) / (2.0 * h),
                d1: (jp.d1 - jm.d1) / (2.0 * h),
                d2: (jp.d2 - jm.d2) / (2.0 * h),
                d3: (jp.d3 - jm.d3) / (2.0 * h),
            };
            let an = out.djet_drho[j];
            assert_eq!(an.mu.signum(), fd.mu.signum());
            assert_eq!(an.d1.signum(), fd.d1.signum());
            assert_eq!(an.d2.signum(), fd.d2.signum());
            assert_eq!(an.d3.signum(), fd.d3.signum());
            assert!((an.mu - fd.mu).abs() < 1e-6);
            assert!((an.d1 - fd.d1).abs() < 1e-6);
            assert!((an.d2 - fd.d2).abs() < 1e-6);
            assert!((an.d3 - fd.d3).abs() < 1e-6);
        }
    }

    #[test]
    fn sas_param_partials_matchfd() {
        let eta = 0.37;
        let epsilon = -0.12;
        let log_delta = 0.21;
        let out = sas_inverse_link_jetwith_param_partials(eta, epsilon, log_delta);
        let h = 1e-6;

        let ep_p = sas_inverse_link_jet(eta, epsilon + h, log_delta);
        let ep_m = sas_inverse_link_jet(eta, epsilon - h, log_delta);
        let fd_ep = InverseLinkJet {
            mu: (ep_p.mu - ep_m.mu) / (2.0 * h),
            d1: (ep_p.d1 - ep_m.d1) / (2.0 * h),
            d2: (ep_p.d2 - ep_m.d2) / (2.0 * h),
            d3: (ep_p.d3 - ep_m.d3) / (2.0 * h),
        };
        assert_eq!(out.djet_depsilon.mu.signum(), fd_ep.mu.signum());
        assert_eq!(out.djet_depsilon.d1.signum(), fd_ep.d1.signum());
        assert_eq!(out.djet_depsilon.d2.signum(), fd_ep.d2.signum());
        assert_eq!(out.djet_depsilon.d3.signum(), fd_ep.d3.signum());
        assert!((out.djet_depsilon.mu - fd_ep.mu).abs() < 5e-5);
        assert!((out.djet_depsilon.d1 - fd_ep.d1).abs() < 5e-5);
        assert!((out.djet_depsilon.d2 - fd_ep.d2).abs() < 5e-5);
        assert!((out.djet_depsilon.d3 - fd_ep.d3).abs() < 5e-4);

        let ld_p = sas_inverse_link_jet(eta, epsilon, log_delta + h);
        let ld_m = sas_inverse_link_jet(eta, epsilon, log_delta - h);
        let fd_ld = InverseLinkJet {
            mu: (ld_p.mu - ld_m.mu) / (2.0 * h),
            d1: (ld_p.d1 - ld_m.d1) / (2.0 * h),
            d2: (ld_p.d2 - ld_m.d2) / (2.0 * h),
            d3: (ld_p.d3 - ld_m.d3) / (2.0 * h),
        };
        assert_eq!(out.djet_dlog_delta.mu.signum(), fd_ld.mu.signum());
        assert_eq!(out.djet_dlog_delta.d1.signum(), fd_ld.d1.signum());
        assert_eq!(out.djet_dlog_delta.d2.signum(), fd_ld.d2.signum());
        assert_eq!(out.djet_dlog_delta.d3.signum(), fd_ld.d3.signum());
        assert!((out.djet_dlog_delta.mu - fd_ld.mu).abs() < 5e-5);
        assert!((out.djet_dlog_delta.d1 - fd_ld.d1).abs() < 5e-5);
        assert!((out.djet_dlog_delta.d2 - fd_ld.d2).abs() < 5e-5);
        assert!((out.djet_dlog_delta.d3 - fd_ld.d3).abs() < 5e-4);
    }

    #[test]
    fn sas_jet_extreme_inputs_stay_finite() {
        let cases = [
            (-1e6, 0.0, 0.0),
            (1e6, 0.0, 0.0),
            (3.0, 12.0, 12.0),
            (-3.0, -12.0, -12.0),
            (0.5, 40.0, 10.0),
            (0.5, -40.0, -10.0),
        ];
        for (eta, eps, log_delta) in cases {
            let j = sas_inverse_link_jet(eta, eps, log_delta);
            assert!(j.mu.is_finite());
            assert!(j.d1.is_finite());
            assert!(j.d2.is_finite());
            assert!(j.d3.is_finite());
            let p = sas_inverse_link_jetwith_param_partials(eta, eps, log_delta);
            assert!(p.djet_depsilon.mu.is_finite());
            assert!(p.djet_depsilon.d1.is_finite());
            assert!(p.djet_depsilon.d2.is_finite());
            assert!(p.djet_depsilon.d3.is_finite());
            assert!(p.djet_dlog_delta.mu.is_finite());
            assert!(p.djet_dlog_delta.d1.is_finite());
            assert!(p.djet_dlog_delta.d2.is_finite());
            assert!(p.djet_dlog_delta.d3.is_finite());
        }
    }

    #[test]
    fn sas_param_partials_remain_finite_in_extreme_region() {
        let eta = 10.0;
        let epsilon = -60.0;
        let log_delta = 40.0;
        let j = sas_inverse_link_jetwith_param_partials(eta, epsilon, log_delta);
        assert!(j.djet_depsilon.mu.is_finite());
        assert!(j.djet_depsilon.d1.is_finite());
        assert!(j.djet_depsilon.d2.is_finite());
        assert!(j.djet_depsilon.d3.is_finite());
        assert!(j.djet_dlog_delta.mu.is_finite());
        assert!(j.djet_dlog_delta.d1.is_finite());
        assert!(j.djet_dlog_delta.d2.is_finite());
        assert!(j.djet_dlog_delta.d3.is_finite());
    }

    #[test]
    fn sas_eta_jets_matchfd() {
        let eta = -0.43;
        let epsilon = 0.27;
        let log_delta = -0.31;
        let h = 1e-5;
        let j0 = sas_inverse_link_jet(eta, epsilon, log_delta);
        let jp = sas_inverse_link_jet(eta + h, epsilon, log_delta);
        let jm = sas_inverse_link_jet(eta - h, epsilon, log_delta);
        let d1fd = (jp.mu - jm.mu) / (2.0 * h);
        let d2fd = (jp.d1 - jm.d1) / (2.0 * h);
        let d3fd = (jp.d2 - jm.d2) / (2.0 * h);
        assert_eq!(j0.d1.signum(), d1fd.signum());
        assert_eq!(j0.d2.signum(), d2fd.signum());
        assert_eq!(j0.d3.signum(), d3fd.signum());
        assert!((j0.d1 - d1fd).abs() < 5e-5);
        assert!((j0.d2 - d2fd).abs() < 2e-4);
        assert!((j0.d3 - d3fd).abs() < 1e-3);
    }

    #[test]
    fn family_dispatch_requires_state_for_sas_and_mixture() {
        let sas_err = inverse_link_jet_for_family(
            crate::types::LikelihoodFamily::BinomialSas,
            0.1,
            None,
            None,
        )
        .expect_err("SAS without state should error");
        assert!(sas_err.to_string().contains("requires SAS link state"));

        let mix_err = inverse_link_jet_for_family(
            crate::types::LikelihoodFamily::BinomialMixture,
            0.1,
            None,
            None,
        )
        .expect_err("mixture without state should error");
        assert!(mix_err.to_string().contains("requires mixture link state"));
    }

    #[test]
    fn beta_logistic_reduces_to_logit_at_delta0_epsilon0() {
        let eta = 0.42;
        let j_bl = beta_logistic_inverse_link_jet(eta, 0.0, 0.0);
        let j_logit = component_inverse_link_jet(LinkComponent::Logit, eta);
        assert!((j_bl.mu - j_logit.mu).abs() < 1e-10);
        assert!((j_bl.d1 - j_logit.d1).abs() < 1e-10);
        assert!((j_bl.d2 - j_logit.d2).abs() < 1e-10);
        assert!((j_bl.d3 - j_logit.d3).abs() < 1e-10);
    }

    #[test]
    fn beta_logistic_eta_jets_matchfd() {
        let eta = -0.31;
        let delta = 0.27;
        let epsilon = -0.19;
        let h = 1e-5;
        let j0 = beta_logistic_inverse_link_jet(eta, delta, epsilon);
        let jp = beta_logistic_inverse_link_jet(eta + h, delta, epsilon);
        let jm = beta_logistic_inverse_link_jet(eta - h, delta, epsilon);
        let d1fd = (jp.mu - jm.mu) / (2.0 * h);
        let d2fd = (jp.d1 - jm.d1) / (2.0 * h);
        let d3fd = (jp.d2 - jm.d2) / (2.0 * h);
        assert_eq!(j0.d1.signum(), d1fd.signum());
        assert_eq!(j0.d2.signum(), d2fd.signum());
        assert_eq!(j0.d3.signum(), d3fd.signum());
        assert!((j0.d1 - d1fd).abs() < 5e-5);
        assert!((j0.d2 - d2fd).abs() < 5e-5);
        assert!((j0.d3 - d3fd).abs() < 2e-4);
    }

    #[test]
    fn standard_kernel_structs_match_component_jets() {
        let eta = 0.73;
        assert_eq!(
            ProbitLinkKernel.jet(eta).expect("probit"),
            component_inverse_link_jet(LinkComponent::Probit, eta)
        );
        assert_eq!(
            LogitLinkKernel.jet(eta).expect("logit"),
            component_inverse_link_jet(LinkComponent::Logit, eta)
        );
        assert_eq!(
            CLogLogLinkKernel.jet(eta).expect("cloglog"),
            component_inverse_link_jet(LinkComponent::CLogLog, eta)
        );
        assert_eq!(
            LogLogLinkKernel.jet(eta).expect("loglog"),
            component_inverse_link_jet(LinkComponent::LogLog, eta)
        );
        assert_eq!(
            CauchitLinkKernel.jet(eta).expect("cauchit"),
            component_inverse_link_jet(LinkComponent::Cauchit, eta)
        );
    }

    #[test]
    fn all_component_eta_jets_matchfd() {
        let components = [
            LinkComponent::Logit,
            LinkComponent::Probit,
            LinkComponent::CLogLog,
            LinkComponent::LogLog,
            LinkComponent::Cauchit,
        ];
        let points = [-3.0, -1.1, -0.2, 0.0, 0.7, 1.8, 3.2];
        let h = 1e-5;
        for c in components {
            for &eta in &points {
                let j0 = component_inverse_link_jet(c, eta);
                let jp = component_inverse_link_jet(c, eta + h);
                let jm = component_inverse_link_jet(c, eta - h);
                let d1fd = (jp.mu - jm.mu) / (2.0 * h);
                let d2fd = (jp.d1 - jm.d1) / (2.0 * h);
                let d3fd = (jp.d2 - jm.d2) / (2.0 * h);
                let d1_tol = if matches!(c, LinkComponent::CLogLog | LinkComponent::LogLog) {
                    1.2e-4
                } else {
                    5e-5
                };
                let d2_tol = if matches!(c, LinkComponent::CLogLog | LinkComponent::LogLog) {
                    4e-4
                } else {
                    1.2e-4
                };
                let d3_tol = if matches!(c, LinkComponent::CLogLog | LinkComponent::LogLog) {
                    1.2e-3
                } else {
                    4e-4
                };
                if j0.d1.abs().max(d1fd.abs()) > 1e-10 {
                    assert_eq!(
                        j0.d1.signum(),
                        d1fd.signum(),
                        "d1 sign mismatch for {c:?} eta={eta}"
                    );
                }
                if j0.d2.abs().max(d2fd.abs()) > 1e-10 {
                    assert_eq!(
                        j0.d2.signum(),
                        d2fd.signum(),
                        "d2 sign mismatch for {c:?} eta={eta}: analytic={} fd={}",
                        j0.d2,
                        d2fd
                    );
                }
                if j0.d3.abs().max(d3fd.abs()) > 1e-10 {
                    assert_eq!(
                        j0.d3.signum(),
                        d3fd.signum(),
                        "d3 sign mismatch for {c:?} eta={eta}"
                    );
                }
                assert!(
                    (j0.d1 - d1fd).abs() < d1_tol,
                    "d1 mismatch for {c:?} eta={eta}: analytic={} fd={}",
                    j0.d1,
                    d1fd
                );
                assert!(
                    (j0.d2 - d2fd).abs() < d2_tol,
                    "d2 mismatch for {c:?} eta={eta}: analytic={} fd={}",
                    j0.d2,
                    d2fd
                );
                assert!(
                    (j0.d3 - d3fd).abs() < d3_tol,
                    "d3 mismatch for {c:?} eta={eta}: analytic={} fd={}",
                    j0.d3,
                    d3fd
                );
            }
        }
    }

    #[test]
    fn sas_center_matches_probit_at_delta1_epsilon0() {
        let etas = [-3.0, -1.2, -0.3, 0.0, 0.4, 1.7, 3.0];
        for eta in etas {
            let sas = sas_inverse_link_jet(eta, 0.0, 0.0);
            let probit = ProbitLinkKernel.jet(eta).expect("probit");
            // SAS implementation uses a smooth bounded latent (`tanh_bound`) for
            // numerical robustness, so the probit center is approximate in practice.
            assert!(
                (sas.mu - probit.mu).abs() < 6e-4,
                "mu mismatch at eta={eta}"
            );
            assert!(
                (sas.d1 - probit.d1).abs() < 6e-4,
                "d1 mismatch at eta={eta}"
            );
            assert!(
                (sas.d2 - probit.d2).abs() < 2e-3,
                "d2 mismatch at eta={eta}"
            );
            assert!(
                (sas.d3 - probit.d3).abs() < 4e-3,
                "d3 mismatch at eta={eta}"
            );
        }
    }

    #[test]
    fn beta_logistic_param_partials_matchfd() {
        let eta = -0.41;
        let delta = 0.23;
        let epsilon = -0.17;
        let out = beta_logistic_inverse_link_jetwith_param_partials(eta, delta, epsilon);
        let h = 1e-6;

        let dp = beta_logistic_inverse_link_jet(eta, delta + h, epsilon);
        let dm = beta_logistic_inverse_link_jet(eta, delta - h, epsilon);
        let fd_delta = InverseLinkJet {
            mu: (dp.mu - dm.mu) / (2.0 * h),
            d1: (dp.d1 - dm.d1) / (2.0 * h),
            d2: (dp.d2 - dm.d2) / (2.0 * h),
            d3: (dp.d3 - dm.d3) / (2.0 * h),
        };
        assert_eq!(out.djet_dlog_delta.mu.signum(), fd_delta.mu.signum());
        assert_eq!(out.djet_dlog_delta.d1.signum(), fd_delta.d1.signum());
        assert_eq!(out.djet_dlog_delta.d2.signum(), fd_delta.d2.signum());
        assert_eq!(out.djet_dlog_delta.d3.signum(), fd_delta.d3.signum());
        assert!((out.djet_dlog_delta.mu - fd_delta.mu).abs() < 5e-5);
        assert!((out.djet_dlog_delta.d1 - fd_delta.d1).abs() < 5e-5);
        assert!((out.djet_dlog_delta.d2 - fd_delta.d2).abs() < 1.2e-4);
        assert!((out.djet_dlog_delta.d3 - fd_delta.d3).abs() < 4e-4);

        let ep = beta_logistic_inverse_link_jet(eta, delta, epsilon + h);
        let em = beta_logistic_inverse_link_jet(eta, delta, epsilon - h);
        let fd_epsilon = InverseLinkJet {
            mu: (ep.mu - em.mu) / (2.0 * h),
            d1: (ep.d1 - em.d1) / (2.0 * h),
            d2: (ep.d2 - em.d2) / (2.0 * h),
            d3: (ep.d3 - em.d3) / (2.0 * h),
        };
        assert_eq!(out.djet_depsilon.mu.signum(), fd_epsilon.mu.signum());
        assert_eq!(out.djet_depsilon.d1.signum(), fd_epsilon.d1.signum());
        assert_eq!(out.djet_depsilon.d2.signum(), fd_epsilon.d2.signum());
        assert_eq!(out.djet_depsilon.d3.signum(), fd_epsilon.d3.signum());
        assert!((out.djet_depsilon.mu - fd_epsilon.mu).abs() < 5e-5);
        assert!((out.djet_depsilon.d1 - fd_epsilon.d1).abs() < 5e-5);
        assert!((out.djet_depsilon.d2 - fd_epsilon.d2).abs() < 1.2e-4);
        assert!((out.djet_depsilon.d3 - fd_epsilon.d3).abs() < 4e-4);
    }

    #[test]
    fn inverse_link_pdfthird_derivative_matches_d3_finite_difference() {
        let sas = InverseLink::Sas(SasLinkState::new(-0.25, 0.35).expect("sas state"));
        let beta_logistic = InverseLink::BetaLogistic(SasLinkState {
            epsilon: 0.18,
            log_delta: -0.22,
            delta: (-0.22_f64).exp(),
        });
        let mixture = InverseLink::Mixture(
            state_fromspec(&MixtureLinkSpec {
                components: vec![
                    LinkComponent::Probit,
                    LinkComponent::Logit,
                    LinkComponent::CLogLog,
                    LinkComponent::Cauchit,
                ],
                initial_rho: Array1::from_vec(vec![0.35, -0.45, 0.2]),
            })
            .expect("mixture state"),
        );
        let links = [
            InverseLink::Standard(LinkFunction::Probit),
            InverseLink::Standard(LinkFunction::Logit),
            InverseLink::Standard(LinkFunction::CLogLog),
            sas,
            beta_logistic,
            mixture,
        ];
        let etas = [-1.1, -0.2, 0.6];
        let h = 1e-5;

        for link in &links {
            for &eta in &etas {
                let jp = inverse_link_jet_for_inverse_link(link, eta + h).expect("jet+");
                let jm = inverse_link_jet_for_inverse_link(link, eta - h).expect("jet-");
                let d4fd = (jp.d3 - jm.d3) / (2.0 * h);
                let d4 = inverse_link_pdfthird_derivative_for_inverse_link(link, eta)
                    .expect("analytic d4");
                assert_eq!(
                    d4.signum(),
                    d4fd.signum(),
                    "d4 sign mismatch for {:?} at eta={eta}: analytic={} fd={}",
                    link,
                    d4,
                    d4fd
                );
                assert!(
                    (d4 - d4fd).abs() < 5e-3,
                    "d4 mismatch for {:?} at eta={eta}: analytic={} fd={}",
                    link,
                    d4,
                    d4fd
                );
            }
        }
    }
}
