use crate::estimate::EstimationError;
use crate::probability::{normal_cdf_approx, normal_pdf};
use crate::types::{
    InverseLink, LikelihoodFamily, LinkComponent, LinkFunction, MixtureLinkSpec, MixtureLinkState,
    SasLinkSpec, SasLinkState,
};
use ndarray::Array1;
use statrs::function::beta::{beta_reg, ln_beta};
use statrs::function::gamma::digamma;

const ETA_CLAMP_GENERAL: f64 = 30.0;
const ETA_CLAMP_LOGIT: f64 = 700.0;
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

    fn param_partials(&self, _eta: f64) -> Result<Option<LinkParamPartials>, EstimationError> {
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

pub fn state_from_sas_spec(spec: SasLinkSpec) -> Result<SasLinkState, String> {
    SasLinkState::new(spec.initial_epsilon, spec.initial_log_delta)
}

pub fn state_from_beta_logistic_spec(spec: SasLinkSpec) -> Result<SasLinkState, String> {
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

pub fn validate_mixture_spec(spec: &MixtureLinkSpec) -> Result<(), String> {
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

pub fn softmax_last_fixed_zero(rho: &Array1<f64>) -> Array1<f64> {
    let k = rho.len() + 1;
    let mut logits = Vec::with_capacity(k);
    let mut max_v = 0.0_f64;
    for &v in rho {
        max_v = max_v.max(v);
        logits.push(v);
    }
    max_v = max_v.max(0.0);
    logits.push(0.0);

    let mut sum = 0.0_f64;
    let mut exps = vec![0.0_f64; k];
    for i in 0..k {
        let e = (logits[i] - max_v).exp();
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
pub fn softmax_with_jacobian_last_fixed_zero(
    rho: &Array1<f64>,
) -> (Array1<f64>, ndarray::Array2<f64>) {
    let pi = softmax_last_fixed_zero(rho);
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

pub fn state_from_spec(spec: &MixtureLinkSpec) -> Result<MixtureLinkState, String> {
    validate_mixture_spec(spec)?;
    let pi = softmax_last_fixed_zero(&spec.initial_rho);
    Ok(MixtureLinkState {
        components: spec.components.clone(),
        rho: spec.initial_rho.clone(),
        pi,
    })
}

#[inline]
pub fn component_inverse_link_jet(component: LinkComponent, eta: f64) -> InverseLinkJet {
    match component {
        LinkComponent::Logit => {
            let e = eta.clamp(-ETA_CLAMP_LOGIT, ETA_CLAMP_LOGIT);
            let mu = 1.0 / (1.0 + (-e).exp());
            let d1 = mu * (1.0 - mu);
            let d2 = d1 * (1.0 - 2.0 * mu);
            let d3 = d1 * (1.0 - 6.0 * d1);
            InverseLinkJet { mu, d1, d2, d3 }
        }
        LinkComponent::Probit => {
            let e = eta.clamp(-ETA_CLAMP_GENERAL, ETA_CLAMP_GENERAL);
            let d1 = normal_pdf(e);
            InverseLinkJet {
                mu: normal_cdf_approx(e),
                d1,
                d2: -e * d1,
                d3: (e * e - 1.0) * d1,
            }
        }
        LinkComponent::CLogLog => {
            let e = eta.clamp(-ETA_CLAMP_GENERAL, ETA_CLAMP_GENERAL);
            let t = e.exp();
            let s = (-t).exp();
            let d1 = t * s;
            InverseLinkJet {
                mu: 1.0 - s,
                d1,
                d2: d1 * (1.0 - t),
                d3: d1 * (1.0 - 3.0 * t + t * t),
            }
        }
        LinkComponent::LogLog => {
            let e = eta.clamp(-ETA_CLAMP_GENERAL, ETA_CLAMP_GENERAL);
            let r = (-e).exp();
            let mu = (-r).exp();
            let d1 = mu * r;
            let d2 = d1 * (r - 1.0);
            let d3 = d1 * (r * r - 3.0 * r + 1.0);
            InverseLinkJet { mu, d1, d2, d3 }
        }
        LinkComponent::Cauchit => {
            let e = eta;
            let den = 1.0 + e * e;
            let d1 = 1.0 / (std::f64::consts::PI * den);
            let d2 = -2.0 * e / (std::f64::consts::PI * den * den);
            let d3 = (6.0 * e * e - 2.0) / (std::f64::consts::PI * den * den * den);
            InverseLinkJet {
                mu: 0.5 + e.atan() / std::f64::consts::PI,
                d1,
                d2,
                d3,
            }
        }
    }
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
            sas_inverse_link_jet_with_param_partials(eta, self.epsilon, self.log_delta),
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
            beta_logistic_inverse_link_jet_with_param_partials(eta, self.delta, self.epsilon),
        )))
    }
}

impl InverseLinkKernel for MixtureLinkState {
    fn jet(&self, eta: f64) -> Result<InverseLinkJet, EstimationError> {
        Ok(mixture_inverse_link_jet(self, eta))
    }

    fn param_partials(&self, eta: f64) -> Result<Option<LinkParamPartials>, EstimationError> {
        Ok(Some(LinkParamPartials::Mixture(
            mixture_inverse_link_jet_with_rho_partials(self, eta),
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
pub fn mixture_inverse_link_jet_with_rho_partials(
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
    let jet = mixture_inverse_link_jet_with_rho_partials_into(state, eta, &mut djet_drho);
    MixtureJetWithRhoPartials { jet, djet_drho }
}

/// Computes mixture jet and writes exact rho partial jets into `out` (length >= K-1).
/// This avoids heap allocation in hot loops.
pub fn mixture_inverse_link_jet_with_rho_partials_into(
    state: &MixtureLinkState,
    eta: f64,
    out: &mut [InverseLinkJet],
) -> InverseLinkJet {
    let k = state.components.len().min(state.pi.len());
    let m = k.saturating_sub(1);
    debug_assert!(
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
fn logistic_u_with_derivatives(eta: f64) -> (f64, f64) {
    let e = eta.clamp(-ETA_CLAMP_LOGIT, ETA_CLAMP_LOGIT);
    let u = if e >= 0.0 {
        let z = (-e).exp();
        1.0 / (1.0 + z)
    } else {
        let z = e.exp();
        z / (1.0 + z)
    };
    let u = u.clamp(BETA_LOGISTIC_U_EPS, 1.0 - BETA_LOGISTIC_U_EPS);
    let du = u * (1.0 - u);
    (u, du)
}

/// Beta-Logistic inverse-link jet for:
///   u = logistic(eta)
///   a = exp(delta - epsilon), b = exp(delta + epsilon)
///   mu = I_u(a, b)
pub fn beta_logistic_inverse_link_jet(eta: f64, delta: f64, epsilon: f64) -> InverseLinkJet {
    let (u, du) = logistic_u_with_derivatives(eta);
    let a = (delta - epsilon).exp();
    let b = (delta + epsilon).exp();
    let mu = beta_reg(a, b, u).clamp(BETA_LOGISTIC_U_EPS, 1.0 - BETA_LOGISTIC_U_EPS);
    let log_d1 = a * u.ln() + b * (1.0 - u).ln() - ln_beta(a, b);
    let d1 = log_d1.exp();
    let t = a * (1.0 - u) - b * u;
    let d2 = d1 * t;
    let d3 = d1 * (t * t - (a + b) * du);
    InverseLinkJet { mu, d1, d2, d3 }
}

pub fn beta_logistic_inverse_link_jet_with_param_partials(
    eta: f64,
    delta: f64,
    epsilon: f64,
) -> SasJetWithParamPartials {
    let (u, du) = logistic_u_with_derivatives(eta);
    let a = (delta - epsilon).exp();
    let b = (delta + epsilon).exp();
    let mu = beta_reg(a, b, u).clamp(BETA_LOGISTIC_U_EPS, 1.0 - BETA_LOGISTIC_U_EPS);
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
    let e = if eta.is_finite() { eta } else { 0.0 };
    let a = e.asinh();
    let delta = sas_delta_from_raw_log_delta(log_delta);
    let u_raw = delta * a - epsilon;
    let u = tanh_bound(u_raw, SAS_U_CLAMP);
    let g1 = tanh_bound_d1(u_raw, SAS_U_CLAMP);
    let g2 = tanh_bound_d2(u_raw, SAS_U_CLAMP);
    let g3 = tanh_bound_d3(u_raw, SAS_U_CLAMP);
    let s = u.sinh();
    let c = u.cosh();
    let z = s;
    let phi = normal_pdf(z);
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
    let mu = normal_cdf_approx(z);
    let d1 = phi * z1;
    let d2 = phi * (z2 - z * z1 * z1);
    let d3 = phi * (z3 - 3.0 * z * z1 * z2 + (z * z - 1.0) * z1 * z1 * z1);
    InverseLinkJet { mu, d1, d2, d3 }
}

pub fn sas_inverse_link_jet_with_param_partials(
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
    let phi = normal_pdf(z);

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

    let base = InverseLinkJet {
        mu: normal_cdf_approx(z),
        d1: phi * z1,
        d2: phi * (z2 - z * z1 * z1),
        d3: phi * (z3 - 3.0 * z * z1 * z2 + (z * z - 1.0) * z1 * z1 * z1),
    };

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

        let phi_t = -z * phi * z_t;
        let mu_t = phi * z_t;
        let d1_t = phi_t * z1 + phi * z1_t;

        let k2 = z2 - z * z1 * z1;
        let k2_t = z2_t - z_t * z1 * z1 - z * 2.0 * z1 * z1_t;
        let d2_t = phi_t * k2 + phi * k2_t;

        let k3 = z3 - 3.0 * z * z1 * z2 + (z * z - 1.0) * z1 * z1 * z1;
        let k3_t = z3_t - 3.0 * (z_t * z1 * z2 + z * z1_t * z2 + z * z1 * z2_t)
            + 2.0 * z * z_t * z1 * z1 * z1
            + (z * z - 1.0) * 3.0 * z1 * z1 * z1_t;
        let d3_t = phi_t * k3 + phi * k3_t;

        InverseLinkJet {
            mu: mu_t,
            d1: d1_t,
            d2: d2_t,
            d3: d3_t,
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
        jet: base,
        djet_depsilon,
        djet_dlog_delta,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::LinkComponent;

    #[test]
    fn softmax_jacobian_matches_fd() {
        let rho = Array1::from_vec(vec![0.7, -1.2, 0.4]);
        let (pi, jac) = softmax_with_jacobian_last_fixed_zero(&rho);
        let h = 1e-6;
        for j in 0..rho.len() {
            let mut rp = rho.clone();
            rp[j] += h;
            let mut rm = rho.clone();
            rm[j] -= h;
            let pp = softmax_last_fixed_zero(&rp);
            let pm = softmax_last_fixed_zero(&rm);
            let fd = (&pp - &pm).mapv(|v| v / (2.0 * h));
            for k in 0..pi.len() {
                let err = (jac[[k, j]] - fd[k]).abs();
                assert_eq!(
                    jac[[k, j]].signum(),
                    fd[k].signum(),
                    "jac sign mismatch at ({k},{j}): analytic={} fd={}",
                    jac[[k, j]],
                    fd[k]
                );
                assert!(err < 5e-6, "jac mismatch at ({k},{j}): err={err:e}");
            }
        }
    }

    #[test]
    fn mixture_jet_rho_partials_match_fd() {
        let spec = MixtureLinkSpec {
            components: vec![
                LinkComponent::Probit,
                LinkComponent::Logit,
                LinkComponent::CLogLog,
                LinkComponent::Cauchit,
            ],
            initial_rho: Array1::from_vec(vec![0.3, -0.6, 0.2]),
        };
        let state = state_from_spec(&spec).expect("state");
        let eta = 0.35;
        let out = mixture_inverse_link_jet_with_rho_partials(&state, eta);
        let h = 1e-6;
        for j in 0..state.rho.len() {
            let mut rp = state.rho.clone();
            rp[j] += h;
            let sp = MixtureLinkSpec {
                components: state.components.clone(),
                initial_rho: rp,
            };
            let jp = mixture_inverse_link_jet(&state_from_spec(&sp).expect("sp"), eta);
            let mut rm = state.rho.clone();
            rm[j] -= h;
            let sm = MixtureLinkSpec {
                components: state.components.clone(),
                initial_rho: rm,
            };
            let jm = mixture_inverse_link_jet(&state_from_spec(&sm).expect("sm"), eta);
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
    fn sas_param_partials_match_fd() {
        let eta = 0.37;
        let epsilon = -0.12;
        let log_delta = 0.21;
        let out = sas_inverse_link_jet_with_param_partials(eta, epsilon, log_delta);
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
            let p = sas_inverse_link_jet_with_param_partials(eta, eps, log_delta);
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
        let j = sas_inverse_link_jet_with_param_partials(eta, epsilon, log_delta);
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
    fn sas_eta_jets_match_fd() {
        let eta = -0.43;
        let epsilon = 0.27;
        let log_delta = -0.31;
        let h = 1e-5;
        let j0 = sas_inverse_link_jet(eta, epsilon, log_delta);
        let jp = sas_inverse_link_jet(eta + h, epsilon, log_delta);
        let jm = sas_inverse_link_jet(eta - h, epsilon, log_delta);
        let d1_fd = (jp.mu - jm.mu) / (2.0 * h);
        let d2_fd = (jp.d1 - jm.d1) / (2.0 * h);
        let d3_fd = (jp.d2 - jm.d2) / (2.0 * h);
        assert_eq!(j0.d1.signum(), d1_fd.signum());
        assert_eq!(j0.d2.signum(), d2_fd.signum());
        assert_eq!(j0.d3.signum(), d3_fd.signum());
        assert!((j0.d1 - d1_fd).abs() < 5e-5);
        assert!((j0.d2 - d2_fd).abs() < 2e-4);
        assert!((j0.d3 - d3_fd).abs() < 1e-3);
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
    fn beta_logistic_eta_jets_match_fd() {
        let eta = -0.31;
        let delta = 0.27;
        let epsilon = -0.19;
        let h = 1e-5;
        let j0 = beta_logistic_inverse_link_jet(eta, delta, epsilon);
        let jp = beta_logistic_inverse_link_jet(eta + h, delta, epsilon);
        let jm = beta_logistic_inverse_link_jet(eta - h, delta, epsilon);
        let d1_fd = (jp.mu - jm.mu) / (2.0 * h);
        let d2_fd = (jp.d1 - jm.d1) / (2.0 * h);
        let d3_fd = (jp.d2 - jm.d2) / (2.0 * h);
        assert_eq!(j0.d1.signum(), d1_fd.signum());
        assert_eq!(j0.d2.signum(), d2_fd.signum());
        assert_eq!(j0.d3.signum(), d3_fd.signum());
        assert!((j0.d1 - d1_fd).abs() < 5e-5);
        assert!((j0.d2 - d2_fd).abs() < 5e-5);
        assert!((j0.d3 - d3_fd).abs() < 2e-4);
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
    fn all_component_eta_jets_match_fd() {
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
                let d1_fd = (jp.mu - jm.mu) / (2.0 * h);
                let d2_fd = (jp.d1 - jm.d1) / (2.0 * h);
                let d3_fd = (jp.d2 - jm.d2) / (2.0 * h);
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
                assert_eq!(
                    j0.d1.signum(),
                    d1_fd.signum(),
                    "d1 sign mismatch for {c:?} eta={eta}"
                );
                assert_eq!(
                    j0.d2.signum(),
                    d2_fd.signum(),
                    "d2 sign mismatch for {c:?} eta={eta}"
                );
                assert_eq!(
                    j0.d3.signum(),
                    d3_fd.signum(),
                    "d3 sign mismatch for {c:?} eta={eta}"
                );
                assert!(
                    (j0.d1 - d1_fd).abs() < d1_tol,
                    "d1 mismatch for {c:?} eta={eta}: analytic={} fd={}",
                    j0.d1,
                    d1_fd
                );
                assert!(
                    (j0.d2 - d2_fd).abs() < d2_tol,
                    "d2 mismatch for {c:?} eta={eta}: analytic={} fd={}",
                    j0.d2,
                    d2_fd
                );
                assert!(
                    (j0.d3 - d3_fd).abs() < d3_tol,
                    "d3 mismatch for {c:?} eta={eta}: analytic={} fd={}",
                    j0.d3,
                    d3_fd
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
    fn beta_logistic_param_partials_match_fd() {
        let eta = -0.41;
        let delta = 0.23;
        let epsilon = -0.17;
        let out = beta_logistic_inverse_link_jet_with_param_partials(eta, delta, epsilon);
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
}
