use crate::estimate::EstimationError;
use crate::quadrature::latent_cloglog_jet5;
use gam_math::{
    jet_tower::trigamma,
    probability::{normal_cdf, normal_pdf},
};
use gam_math::special::stable_polynomial_times_exp_neg as stable_nonnegative_poly_times_exp_neg;
use gam_problem::{
    InverseLink, LatentCLogLogState, LikelihoodSpec, LinkComponent, LinkFunction, MixtureLinkSpec,
    MixtureLinkState, ResponseFamily, SasLinkSpec, SasLinkState, StandardLink,
};
use ndarray::{Array1, Array2};
use statrs::function::beta::{beta_reg, ln_beta};
use statrs::function::gamma::digamma;
use std::ops::Neg;
use std::sync::OnceLock;

const SAS_U_CLAMP: f64 = 50.0;
/// Inclusive eta domain for the solver's standard log inverse-link derivative
/// seams. Within this conservative IEEE-754-safe interval, `exp(eta)` is finite,
/// positive, and normal, so the value and every analytic derivative are exactly
/// the same operation. Solver callers must reject steps outside this domain;
/// silently projecting eta would define a different, nonsmooth link.
pub const LOG_LINK_SOLVER_ETA_MIN: f64 = -700.0;
/// Inclusive upper endpoint of the standard log-link solver domain.
pub const LOG_LINK_SOLVER_ETA_MAX: f64 = 700.0;
/// Bound B used by the bounded sinh-arcsinh log-delta parameterisation:
/// `delta = exp(B * tanh(raw_log_delta / B))`. Exposed for the outer-strategy
/// edge-barrier helpers in `solver/estimate.rs` that previously had to
/// hard-code the same `12.0` with a "must match" comment.
pub(crate) const SAS_LOG_DELTA_BOUND: f64 = 12.0;

#[inline]
fn latent_cloglog_quadctx() -> &'static crate::quadrature::QuadratureContext {
    static QUADCTX: OnceLock<crate::quadrature::QuadratureContext> = OnceLock::new();
    QUADCTX.get_or_init(crate::quadrature::QuadratureContext::new)
}

#[inline]
fn latent_cloglog_point_jet(
    state: &LatentCLogLogState,
    eta: f64,
) -> Result<InverseLinkJet, EstimationError> {
    let jet = latent_cloglog_jet5(latent_cloglog_quadctx(), eta, state.latent_sd)?;
    Ok(InverseLinkJet {
        mu: jet.mean,
        d1: jet.d1,
        d2: jet.d2,
        d3: jet.d3,
    })
}

#[inline]
pub(crate) fn log_link_solver_exp(eta: f64) -> Result<f64, EstimationError> {
    if !(LOG_LINK_SOLVER_ETA_MIN..=LOG_LINK_SOLVER_ETA_MAX).contains(&eta) {
        return Err(EstimationError::InverseLinkDomainViolation {
            link: "standard log inverse link",
            eta,
            lower: LOG_LINK_SOLVER_ETA_MIN,
            upper: LOG_LINK_SOLVER_ETA_MAX,
        });
    }
    Ok(eta.exp())
}

#[inline]
fn finite_inverse_link_eta(link: &'static str, eta: f64) -> Result<f64, EstimationError> {
    if !eta.is_finite() {
        return Err(EstimationError::InverseLinkDomainViolation {
            link,
            eta,
            lower: -f64::MAX,
            upper: f64::MAX,
        });
    }
    Ok(eta)
}

#[derive(Clone, Copy)]
struct AsinhJet5 {
    value: f64,
    d1: f64,
    d2: f64,
    d3: f64,
    d4: f64,
    d5: f64,
}

/// Exact eta derivatives of `asinh(eta)`, factored through `hypot` so powers
/// of a large finite eta never form `inf * 0` in the derivative tails.
#[inline]
fn asinh_jet5(eta: f64) -> AsinhJet5 {
    let q = eta.hypot(1.0);
    let inv_q = q.recip();
    let inv_q2 = inv_q * inv_q;
    let inv_q3 = inv_q2 * inv_q;
    let inv_q4 = inv_q2 * inv_q2;
    let inv_q5 = inv_q4 * inv_q;
    let t = eta / q;
    let t2 = t * t;
    let t4 = t2 * t2;
    AsinhJet5 {
        value: eta.asinh(),
        d1: inv_q,
        d2: -t * inv_q2,
        d3: (2.0 * t2 - inv_q2) * inv_q3,
        d4: t * (9.0 * inv_q2 - 6.0 * t2) * inv_q4,
        d5: (9.0 * inv_q4 - 72.0 * t2 * inv_q2 + 24.0 * t4) * inv_q5,
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InverseLinkJet {
    pub mu: f64,
    pub d1: f64,
    pub d2: f64,
    pub d3: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogitJet5 {
    pub mu: f64,
    pub d1: f64,
    pub d2: f64,
    pub d3: f64,
    pub d4: f64,
    pub d5: f64,
}

#[inline]
fn canonicalzero(v: f64) -> f64 {
    // Normalize the two IEEE zero encodings for deterministic jets without
    // changing their mathematical support. A nonzero subnormal is still a
    // representable derivative and must survive: replacing it by zero creates
    // an artificial constant tail and a kink at MIN_POSITIVE.
    if v == 0.0 { 0.0 } else { v }
}

#[inline]
fn canonicalize_jet(mut jet: InverseLinkJet) -> InverseLinkJet {
    jet.d1 = canonicalzero(jet.d1);
    jet.d2 = canonicalzero(jet.d2);
    jet.d3 = canonicalzero(jet.d3);
    jet
}

#[inline]
pub fn logit_inverse_link_jet5(eta: f64) -> LogitJet5 {
    if eta.is_nan() {
        return LogitJet5 {
            mu: f64::NAN,
            d1: f64::NAN,
            d2: f64::NAN,
            d3: f64::NAN,
            d4: f64::NAN,
            d5: f64::NAN,
        };
    }
    if eta == f64::INFINITY {
        return LogitJet5 {
            mu: 1.0,
            d1: 0.0,
            d2: 0.0,
            d3: 0.0,
            d4: 0.0,
            d5: 0.0,
        };
    }
    if eta == f64::NEG_INFINITY {
        return LogitJet5 {
            mu: 0.0,
            d1: 0.0,
            d2: 0.0,
            d3: 0.0,
            d4: 0.0,
            d5: 0.0,
        };
    }

    let jet = if eta >= 0.0 {
        let z = (-eta).exp();
        let opz = 1.0 + z;
        let opz2 = opz * opz;
        let opz3 = opz2 * opz;
        let opz4 = opz3 * opz;
        let opz5 = opz4 * opz;
        let opz6 = opz5 * opz;
        let z2 = z * z;
        let z3 = z2 * z;
        let z4 = z3 * z;
        LogitJet5 {
            mu: 1.0 / opz,
            d1: z / opz2,
            d2: z * (z - 1.0) / opz3,
            d3: z * (z2 - 4.0 * z + 1.0) / opz4,
            d4: z * (z3 - 11.0 * z2 + 11.0 * z - 1.0) / opz5,
            d5: z * (z4 - 26.0 * z3 + 66.0 * z2 - 26.0 * z + 1.0) / opz6,
        }
    } else {
        let z = eta.exp();
        let opz = 1.0 + z;
        let opz2 = opz * opz;
        let opz3 = opz2 * opz;
        let opz4 = opz3 * opz;
        let opz5 = opz4 * opz;
        let opz6 = opz5 * opz;
        let z2 = z * z;
        let z3 = z2 * z;
        let z4 = z3 * z;
        LogitJet5 {
            mu: z / opz,
            d1: z / opz2,
            d2: z * (1.0 - z) / opz3,
            d3: z * (1.0 - 4.0 * z + z2) / opz4,
            d4: z * (1.0 - 11.0 * z + 11.0 * z2 - z3) / opz5,
            d5: z * (1.0 - 26.0 * z + 66.0 * z2 - 26.0 * z3 + z4) / opz6,
        }
    };
    LogitJet5 {
        mu: jet.mu,
        d1: canonicalzero(jet.d1),
        d2: canonicalzero(jet.d2),
        d3: canonicalzero(jet.d3),
        d4: canonicalzero(jet.d4),
        d5: canonicalzero(jet.d5),
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
    if phi == 0.0 {
        return InverseLinkJet {
            mu: normal_cdf(x),
            d1: 0.0,
            d2: 0.0,
            d3: 0.0,
        };
    }
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
    if phi == 0.0 {
        return 0.0;
    }
    canonicalzero(-(x * x * x - 3.0 * x) * phi)
}

#[inline]
fn probit_pdffourth_derivative(eta: f64) -> f64 {
    // mu''''' = Phi^{(5)}(eta) = (eta^4 - 6*eta^2 + 3) * phi(eta).
    if eta.is_nan() {
        return f64::NAN;
    }
    if !eta.is_finite() {
        return 0.0;
    }
    let x = eta;
    let phi = normal_pdf(x);
    if phi == 0.0 {
        return 0.0;
    }
    canonicalzero((x * x * x * x - 6.0 * x * x + 3.0) * phi)
}

/// Multiply two 5-term truncated Taylor series (coefficients `a_k = g^(k)/k!`,
/// `k = 0..=4`) and return the truncated product coefficients.
#[inline]
fn taylor5_mul(a: &[f64; 5], b: &[f64; 5]) -> [f64; 5] {
    let mut c = [0.0_f64; 5];
    for i in 0..5 {
        let ai = a[i];
        if ai == 0.0 {
            continue;
        }
        for j in 0..(5 - i) {
            c[i + j] += ai * b[j];
        }
    }
    c
}

/// Reciprocal of a 5-term truncated Taylor series with nonzero constant term.
#[inline]
fn taylor5_inv(a: &[f64; 5]) -> [f64; 5] {
    let mut b = [0.0_f64; 5];
    b[0] = 1.0 / a[0];
    for k in 1..5 {
        let mut s = 0.0_f64;
        for j in 1..=k {
            s += a[j] * b[k - j];
        }
        b[k] = -s * b[0];
    }
    b
}

/// 5-jet (value + four eta-derivatives) of the GLM Fisher working weight
/// `W(eta) = mu'(eta)^2 / V(mu(eta))` for the requested standard link, returned
/// as `(W, W', W'', W''', W'''')`.
///
/// For the canonical logit link this is exactly the binomial weight
/// `W = mu(1 - mu) = mu'`, whose eta-derivatives are the higher derivatives of
/// the inverse-link jet (`W^(k) = mu^(k+1)`); the dispatch returns
/// `logit_inverse_link_jet5`'s `d1..d5` byte-for-byte so the existing Firth
/// logit path is numerically unchanged.
///
/// Noncanonical Bernoulli links use the same truncated Taylor-series quotient:
/// assemble the inverse-link jet through `mu^(5)`, square the `mu'` series, and
/// divide by the Bernoulli variance series `mu(1-mu)`. As the variance
/// denominator saturates to zero in either tail, the weight and all derivatives
/// saturate to zero, matching the inverse-link jet convention.
pub(crate) fn fisher_weight_jet5(link: StandardLink, eta: f64) -> (f64, f64, f64, f64, f64) {
    match link {
        StandardLink::Logit => {
            let jet = logit_inverse_link_jet5(eta);
            (jet.d1, jet.d2, jet.d3, jet.d4, jet.d5)
        }
        StandardLink::Probit => probit_fisher_weight_jet5(eta),
        StandardLink::CLogLog => component_fisher_weight_jet5(LinkComponent::CLogLog, eta),
        StandardLink::LogLog => component_fisher_weight_jet5(LinkComponent::LogLog, eta),
        StandardLink::Cauchit => component_fisher_weight_jet5(LinkComponent::Cauchit, eta),
        StandardLink::Identity | StandardLink::Log => (0.0, 0.0, 0.0, 0.0, 0.0),
    }
}

pub(crate) fn fisher_weight_jet5_for_inverse_link(
    link: &InverseLink,
    eta: f64,
) -> Result<(f64, f64, f64, f64, f64), EstimationError> {
    match link {
        InverseLink::Standard(link) => Ok(fisher_weight_jet5(*link, eta)),
        InverseLink::LatentCLogLog(_)
        | InverseLink::Sas(_)
        | InverseLink::BetaLogistic(_)
        | InverseLink::Mixture(_) => {
            let jet = link.jet(eta)?;
            let d4 = inverse_link_pdfthird_derivative_for_inverse_link(link, eta)?;
            let d5 = inverse_link_pdffourth_derivative_for_inverse_link(link, eta)?;
            Ok(fisher_weight_jet5_from_inverse_link_derivatives(
                jet.mu, jet.d1, jet.d2, jet.d3, d4, d5,
            ))
        }
    }
}

#[inline]
fn component_fisher_weight_jet5(component: LinkComponent, eta: f64) -> (f64, f64, f64, f64, f64) {
    let jet = component_inverse_link_jet(component, eta);
    let d4 = component_inverse_link_pdfthird_derivative(component, eta);
    let d5 = component_inverse_link_pdffourth_derivative(component, eta);
    fisher_weight_jet5_from_inverse_link_derivatives(jet.mu, jet.d1, jet.d2, jet.d3, d4, d5)
}

#[inline]
fn fisher_weight_jet5_from_inverse_link_derivatives(
    mu: f64,
    d1: f64,
    d2: f64,
    d3: f64,
    d4: f64,
    d5: f64,
) -> (f64, f64, f64, f64, f64) {
    if [mu, d1, d2, d3, d4, d5].iter().any(|v| v.is_nan()) {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }
    let variance = mu * (1.0 - mu);
    if !(variance > 0.0) || !variance.is_finite() {
        return (0.0, 0.0, 0.0, 0.0, 0.0);
    }

    let factorial = [1.0_f64, 1.0, 2.0, 6.0, 24.0];
    let mu_d = [mu, d1, d2, d3, d4];
    let one_minus_mu_d = [1.0 - mu, -d1, -d2, -d3, -d4];
    let dmu_d = [d1, d2, d3, d4, d5];
    let mut mu_t = [0.0_f64; 5];
    let mut one_minus_mu_t = [0.0_f64; 5];
    let mut dmu_t = [0.0_f64; 5];
    for k in 0..5 {
        let inv_fact = 1.0 / factorial[k];
        mu_t[k] = mu_d[k] * inv_fact;
        one_minus_mu_t[k] = one_minus_mu_d[k] * inv_fact;
        dmu_t[k] = dmu_d[k] * inv_fact;
    }
    let num_t = taylor5_mul(&dmu_t, &dmu_t);
    let den_t = taylor5_mul(&mu_t, &one_minus_mu_t);
    if !(den_t[0] > 0.0) || !den_t[0].is_finite() {
        return (0.0, 0.0, 0.0, 0.0, 0.0);
    }
    let w_t = taylor5_mul(&num_t, &taylor5_inv(&den_t));
    (
        canonicalzero(w_t[0] * factorial[0]),
        canonicalzero(w_t[1] * factorial[1]),
        canonicalzero(w_t[2] * factorial[2]),
        canonicalzero(w_t[3] * factorial[3]),
        canonicalzero(w_t[4] * factorial[4]),
    )
}

/// Probit Bernoulli Fisher-weight 5-jet `W = phi^2 / (Phi (1 - Phi))` and its
/// first four eta-derivatives. See [`fisher_weight_jet5`].
#[inline]
fn probit_fisher_weight_jet5(eta: f64) -> (f64, f64, f64, f64, f64) {
    if eta.is_nan() {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }
    if !eta.is_finite() {
        return (0.0, 0.0, 0.0, 0.0, 0.0);
    }
    let x = eta;
    let p = normal_cdf(x);
    // Compute the complement directly via Phi(-x) rather than `1 - Phi(x)`:
    // in the positive tail `Phi(x)` rounds to 1.0 and `1 - Phi(x)` cancels to
    // zero, whereas `Phi(-x)` retains the accurate (tiny) tail mass.
    let q = normal_cdf(-x);
    let phi = normal_pdf(x);
    // Saturated tail: the denominator Phi(1-Phi) has underflowed to zero (or
    // would divide by zero); the working weight and all derivatives go to zero.
    if !(p > 0.0) || !(q > 0.0) || p * q <= 0.0 {
        return (0.0, 0.0, 0.0, 0.0, 0.0);
    }
    // Gaussian derivative ladder: phi^(k) for k = 0..=4 using phi' = -x phi.
    let phi1 = -x * phi;
    let phi2 = (x * x - 1.0) * phi;
    let phi3 = -(x * x * x - 3.0 * x) * phi;
    let phi4 = (x * x * x * x - 6.0 * x * x + 3.0) * phi;
    // Derivative arrays (d^k/deta^k) for f = phi, p = Phi, q = 1 - Phi.
    // p^(0) = Phi, p^(k>=1) = phi^(k-1); q is the negated complement.
    let f_d = [phi, phi1, phi2, phi3, phi4];
    let p_d = [p, phi, phi1, phi2, phi3];
    let q_d = [q, -phi, -phi1, -phi2, -phi3];
    // Convert derivative arrays to Taylor coefficients a_k = g^(k)/k!.
    let factorial = [1.0_f64, 1.0, 2.0, 6.0, 24.0];
    let mut f_t = [0.0_f64; 5];
    let mut p_t = [0.0_f64; 5];
    let mut q_t = [0.0_f64; 5];
    for k in 0..5 {
        let inv_fact = 1.0 / factorial[k];
        f_t[k] = f_d[k] * inv_fact;
        p_t[k] = p_d[k] * inv_fact;
        q_t[k] = q_d[k] * inv_fact;
    }
    let num_t = taylor5_mul(&f_t, &f_t);
    let den_t = taylor5_mul(&p_t, &q_t);
    let w_t = taylor5_mul(&num_t, &taylor5_inv(&den_t));
    // Back to derivatives W^(k) = w_t[k] * k!.
    (
        canonicalzero(w_t[0] * factorial[0]),
        canonicalzero(w_t[1] * factorial[1]),
        canonicalzero(w_t[2] * factorial[2]),
        canonicalzero(w_t[3] * factorial[3]),
        canonicalzero(w_t[4] * factorial[4]),
    )
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
        LinkComponent::Logit => logit_inverse_link_jet5(eta).d4,
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
            canonicalzero(stable_nonnegative_poly_times_exp_neg(
                t,
                &[0.0, 1.0, -7.0, 6.0, -1.0],
            ))
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
            canonicalzero(stable_nonnegative_poly_times_exp_neg(
                r,
                &[0.0, -1.0, 7.0, -6.0, 1.0],
            ))
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

/// Fifth derivative of a component inverse-link CDF (= fourth derivative of PDF).
/// Extends `component_inverse_link_pdfthird_derivative` by one derivative order.
#[inline]
fn component_inverse_link_pdffourth_derivative(component: LinkComponent, eta: f64) -> f64 {
    match component {
        LinkComponent::Probit => probit_pdffourth_derivative(eta),
        LinkComponent::Logit => logit_inverse_link_jet5(eta).d5,
        LinkComponent::CLogLog => {
            // Exact closed form:
            //   d5 = exp(-t) * (t - 15t^2 + 25t^3 - 10t^4 + t^5)
            //      = d1 * (1 - 15t + 25t^2 - 10t^3 + t^4),
            // where t = exp(eta).
            if eta.is_nan() {
                return f64::NAN;
            }
            if !eta.is_finite() {
                return 0.0;
            }
            let t = eta.exp();
            canonicalzero(stable_nonnegative_poly_times_exp_neg(
                t,
                &[0.0, 1.0, -15.0, 25.0, -10.0, 1.0],
            ))
        }
        LinkComponent::LogLog => {
            // Exact closed form:
            //   d5 = exp(-r) * (r - 15r^2 + 25r^3 - 10r^4 + r^5)
            //      = d1 * (1 - 15r + 25r^2 - 10r^3 + r^4),
            // where r = exp(-eta).
            if eta.is_nan() {
                return f64::NAN;
            }
            if !eta.is_finite() {
                return 0.0;
            }
            let r = (-eta).exp();
            canonicalzero(stable_nonnegative_poly_times_exp_neg(
                r,
                &[0.0, 1.0, -15.0, 25.0, -10.0, 1.0],
            ))
        }
        LinkComponent::Cauchit => {
            // d5 = 24(1 - 10eta^2 + 5eta^4) / [pi * (1+eta^2)^5]
            if eta.is_nan() {
                return f64::NAN;
            }
            if !eta.is_finite() {
                return 0.0;
            }
            let e2 = eta * eta;
            let denom = 1.0 + e2;
            24.0 * (1.0 - 10.0 * e2 + 5.0 * e2 * e2) / (std::f64::consts::PI * denom.powi(5))
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MixtureJetWithRhoPartials {
    pub jet: InverseLinkJet,
    /// Partial derivatives wrt free logits rho_j, j in [0, K-2].
    /// Each entry stores derivatives of (mu, d1, d2, d3) wrt one rho_j.
    pub djet_drho: Vec<InverseLinkJet>,
    /// Exact symmetric Hessian of `mu` in the free-logit coordinates.
    pub d2mu_drho2: Array2<f64>,
    /// Exact symmetric Hessian of `d1 = dmu/deta` in the free-logit coordinates.
    pub d2d1_drho2: Array2<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SasJetWithParamPartials {
    pub jet: InverseLinkJet,
    pub djet_depsilon: InverseLinkJet,
    pub djet_dlog_delta: InverseLinkJet,
    /// Exact symmetric Hessian of `mu` in `(epsilon, raw_log_delta)` order.
    /// For Beta-Logistic the second coordinate is its unbounded
    /// `log_shape_center`, matching the shared optimizer state field.
    pub d2mu_dparams2: Array2<f64>,
    /// Exact symmetric Hessian of `d1 = dmu/deta` in the same parameter order.
    pub d2d1_dparams2: Array2<f64>,
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

    fn param_partials(&self, eta: f64) -> Result<Option<LinkParamPartials>, EstimationError> {
        assert!(eta.is_finite(), "eta must be finite");
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

/// Construct SAS state from raw optimizer parameters using the same bounded
/// transform used everywhere in fitting/evaluation.
///
/// A free function rather than an inherent `SasLinkState::new` because the
/// bounded `delta` transform is solver-side math, so the constructor is hosted
/// here next to the transform rather than on the type. `SasLinkState`'s fields
/// are `pub`, so it builds directly.
pub fn sas_link_state_from_raw(
    raw_epsilon: f64,
    raw_log_delta: f64,
) -> Result<SasLinkState, String> {
    if !raw_epsilon.is_finite() || !raw_log_delta.is_finite() {
        return Err("SAS link parameters must be finite".to_string());
    }
    Ok(SasLinkState {
        epsilon: raw_epsilon,
        log_delta: raw_log_delta,
        delta: sas_delta_from_raw_log_delta(raw_log_delta),
    })
}

pub fn state_from_sasspec(spec: SasLinkSpec) -> Result<SasLinkState, String> {
    sas_link_state_from_raw(spec.initial_epsilon, spec.initial_log_delta)
}

pub fn state_from_beta_logisticspec(spec: SasLinkSpec) -> Result<SasLinkState, String> {
    if !spec.initial_epsilon.is_finite() || !spec.initial_log_delta.is_finite() {
        return Err("Beta-Logistic link parameters must be finite".to_string());
    }
    // For Beta-Logistic, `log_delta` is the unconstrained log geometric-mean beta
    // shape (the kernels' `log_shape_center`). Evaluation consumes `log_delta`,
    // never `delta`, but keep the shared `SasLinkState::delta` field on the same
    // bounded SAS parameterization used by `state_from_sasspec` so constructing a
    // state from a large finite raw log-delta cannot overflow this derived field.
    let log_shape_center = spec.initial_log_delta;
    Ok(SasLinkState {
        epsilon: spec.initial_epsilon,
        log_delta: log_shape_center,
        delta: sas_delta_from_raw_log_delta(log_shape_center),
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
fn tanh_bound_d5(value: f64, bound: f64) -> f64 {
    // 5th derivative of B * tanh(x/B):
    //   g5 = 8 * s * (2 - 15*t^2 + 15*t^4) / B^4
    // where t = tanh(x/B) and s = 1 - t^2.
    let b = bound.max(f64::EPSILON);
    let t = (value / b).tanh();
    let s = 1.0 - t * t;
    let t2 = t * t;
    let b4 = b * b * b * b;
    8.0 * s * (2.0 - 15.0 * t2 + 15.0 * t2 * t2) / b4
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
    // `LinkComponent` admits two variants (Cauchit, LogLog) that have no matching
    // `LinkFunction` entry. When two or more components are *blended*, the mixture-link
    // pipeline projects the blend back onto a single `LinkFunction` value for downstream
    // solver/IO bookkeeping (see `InverseLink::link_function`), so a multi-component blend
    // composed solely of components without a LinkFunction representative would silently
    // lie about its projected link. We therefore require any genuine *blend* (two or more
    // components) to contain at least one Logit/Probit/CLogLog "anchor" so the projection
    // is meaningful, and reject e.g. a blend of only {Cauchit, LogLog}.
    //
    // A *single-component* spec is not a blend at all: it is that one link, with weight
    // 1.0 and no free mixing logits. `LinkComponent::LogLog` / `LinkComponent::Cauchit`
    // implement their inverse link and derivative jets exactly, so a single-component
    // `{LogLog}` / `{Cauchit}` spec is a fully-defined standalone link and is accepted
    // here (this is how survival `--link loglog` / `--link cauchit` are represented).
    let has_anchor = spec.components.iter().any(|component| {
        matches!(
            component,
            LinkComponent::Logit | LinkComponent::Probit | LinkComponent::CLogLog
        )
    });
    if !has_anchor && spec.components.len() > 1 {
        let unsupported: Vec<&str> = spec
            .components
            .iter()
            .map(|component| component.name())
            .collect();
        return Err(format!(
            "mixture link components {{{}}} are unsupported: at least one component \
             must map to a LinkFunction variant (logit/probit/cloglog) so the mixture's \
             projected LinkFunction is well defined; cauchit and loglog have no \
             LinkFunction representative",
            unsupported.join(", ")
        ));
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
            let jet = logit_inverse_link_jet5(eta);
            InverseLinkJet {
                mu: jet.mu,
                d1: jet.d1,
                d2: jet.d2,
                d3: jet.d3,
            }
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
            let t = eta.exp();
            if !t.is_finite() {
                return InverseLinkJet {
                    mu: 1.0,
                    d1: 0.0,
                    d2: 0.0,
                    d3: 0.0,
                };
            }
            InverseLinkJet {
                mu: -(-t).exp_m1(),
                d1: stable_nonnegative_poly_times_exp_neg(t, &[0.0, 1.0]),
                d2: stable_nonnegative_poly_times_exp_neg(t, &[0.0, 1.0, -1.0]),
                d3: stable_nonnegative_poly_times_exp_neg(t, &[0.0, 1.0, -3.0, 1.0]),
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
            let r = (-eta).exp();
            if !r.is_finite() {
                return InverseLinkJet {
                    mu: 0.0,
                    d1: 0.0,
                    d2: 0.0,
                    d3: 0.0,
                };
            }
            InverseLinkJet {
                mu: (-r).exp(),
                d1: stable_nonnegative_poly_times_exp_neg(r, &[0.0, 1.0]),
                d2: stable_nonnegative_poly_times_exp_neg(r, &[0.0, -1.0, 1.0]),
                d3: stable_nonnegative_poly_times_exp_neg(r, &[0.0, 1.0, -3.0, 1.0]),
            }
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
            LinkFunction::LogLog => LogLogLinkKernel.jet(eta),
            LinkFunction::Cauchit => CauchitLinkKernel.jet(eta),
            LinkFunction::Identity => Ok(InverseLinkJet {
                mu: eta,
                d1: 1.0,
                d2: 0.0,
                d3: 0.0,
            }),
            LinkFunction::Log => {
                // A projected value with unprojected exp derivatives is not a jet:
                // outside the projection interval the value is constant but the
                // old implementation returned a nonzero derivative. Evaluate the
                // exact exponential on the declared solver domain and refuse every
                // other eta through the typed error channel. Public response-scale
                // transforms remain unrestricted and use their separate exact-exp
                // path below (issue #963).
                let e = log_link_solver_exp(eta)?;
                Ok(InverseLinkJet {
                    mu: e,
                    d1: e,
                    d2: e,
                    d3: e,
                })
            }
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
        sas_inverse_link_jet(eta, self.epsilon, self.log_delta)
    }

    fn param_partials(&self, eta: f64) -> Result<Option<LinkParamPartials>, EstimationError> {
        Ok(Some(LinkParamPartials::Sas(
            sas_inverse_link_jetwith_param_partials(eta, self.epsilon, self.log_delta)?,
        )))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BetaLogisticKernel {
    /// Unconstrained log of the geometric-mean beta shape — the raw optimization
    /// parameter `SasLinkState::log_delta`, NOT the derived `SasLinkState::delta`.
    pub log_shape_center: f64,
    pub epsilon: f64,
}

impl InverseLinkKernel for BetaLogisticKernel {
    fn jet(&self, eta: f64) -> Result<InverseLinkJet, EstimationError> {
        Ok(beta_logistic_inverse_link_jet(
            eta,
            self.log_shape_center,
            self.epsilon,
        ))
    }

    fn param_partials(&self, eta: f64) -> Result<Option<LinkParamPartials>, EstimationError> {
        Ok(Some(LinkParamPartials::Sas(
            beta_logistic_inverse_link_jetwith_param_partials(
                eta,
                self.log_shape_center,
                self.epsilon,
            ),
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
            InverseLink::Standard(StandardLink::Logit) => LogitLinkKernel.jet(eta),
            InverseLink::Standard(StandardLink::Probit) => ProbitLinkKernel.jet(eta),
            InverseLink::Standard(StandardLink::CLogLog) => CLogLogLinkKernel.jet(eta),
            InverseLink::Standard(StandardLink::LogLog) => LogLogLinkKernel.jet(eta),
            InverseLink::Standard(StandardLink::Cauchit) => CauchitLinkKernel.jet(eta),
            InverseLink::Standard(StandardLink::Identity) => LinkFunction::Identity.jet(eta),
            InverseLink::Standard(StandardLink::Log) => LinkFunction::Log.jet(eta),
            InverseLink::LatentCLogLog(state) => latent_cloglog_point_jet(state, eta),
            InverseLink::Sas(state) => state.jet(eta),
            InverseLink::BetaLogistic(state) => BetaLogisticKernel {
                log_shape_center: state.log_delta,
                epsilon: state.epsilon,
            }
            .jet(eta),
            InverseLink::Mixture(state) => state.jet(eta),
        }
    }

    fn param_partials(&self, eta: f64) -> Result<Option<LinkParamPartials>, EstimationError> {
        match self {
            InverseLink::Standard(_) => Ok(None),
            InverseLink::LatentCLogLog(_) => Ok(None),
            InverseLink::Sas(state) => state.param_partials(eta),
            InverseLink::BetaLogistic(state) => BetaLogisticKernel {
                log_shape_center: state.log_delta,
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
/// The standard log link is defined here only on the inclusive solver domain
/// [`LOG_LINK_SOLVER_ETA_MIN`] through [`LOG_LINK_SOLVER_ETA_MAX`]; inputs
/// outside it return [`EstimationError::InverseLinkDomainViolation`].
pub fn inverse_link_jet_for_inverse_link(
    link: &InverseLink,
    eta: f64,
) -> Result<InverseLinkJet, EstimationError> {
    link.jet(eta)
}

/// Specialized `(mu, d1)` inverse-link evaluation that skips the d2/d3
/// polynomial chain used by the full jet. Numerical semantics are preserved:
/// the returned `mu` and `d1` are bit-identical to the corresponding fields of
/// `inverse_link_jet_for_inverse_link(link, eta)?` for every supported link.
///
/// For latent cloglog the underlying lognormal-Laplace kernel produces all
/// orders together, so this falls back to the full jet for that branch — the
/// savings come from the parameterised polynomial links (SAS, beta-logistic,
/// mixture) and the simple analytic links where d2/d3 are pure waste.
/// Standard-log inputs obey the same solver domain as the full jet.
pub fn inverse_link_mu_d1_for_inverse_link(
    link: &InverseLink,
    eta: f64,
) -> Result<(f64, f64), EstimationError> {
    match link {
        InverseLink::Standard(link_fn) => Ok(link_function_mu_d1(link_fn.as_link_function(), eta)?),
        InverseLink::LatentCLogLog(state) => {
            let jet = latent_cloglog_point_jet(state, eta)?;
            Ok((jet.mu, jet.d1))
        }
        InverseLink::Sas(state) => sas_inverse_link_mu_d1(eta, state.epsilon, state.log_delta),
        InverseLink::BetaLogistic(state) => Ok(beta_logistic_inverse_link_mu_d1(
            eta,
            state.log_delta,
            state.epsilon,
        )),
        InverseLink::Mixture(state) => Ok(mixture_inverse_link_mu_d1(state, eta)),
    }
}

fn link_function_mu_d1(link: LinkFunction, eta: f64) -> Result<(f64, f64), EstimationError> {
    match link {
        LinkFunction::Identity => Ok((eta, 1.0)),
        LinkFunction::Log => {
            // Keep the fast seam mathematically identical to the full jet: exact
            // exp and exact exp derivative on the same declared solver domain.
            let e = log_link_solver_exp(eta)?;
            Ok((e, e))
        }
        LinkFunction::Logit => Ok(component_inverse_link_mu_d1(LinkComponent::Logit, eta)),
        LinkFunction::Probit => Ok(component_inverse_link_mu_d1(LinkComponent::Probit, eta)),
        LinkFunction::CLogLog => Ok(component_inverse_link_mu_d1(LinkComponent::CLogLog, eta)),
        LinkFunction::LogLog => Ok(component_inverse_link_mu_d1(LinkComponent::LogLog, eta)),
        LinkFunction::Cauchit => Ok(component_inverse_link_mu_d1(LinkComponent::Cauchit, eta)),
        LinkFunction::Sas => Err(EstimationError::InvalidInput(
            "LinkFunction::Sas inverse-link requires explicit SAS link state".to_string(),
        )),
        LinkFunction::BetaLogistic => Err(EstimationError::InvalidInput(
            "LinkFunction::BetaLogistic inverse-link requires explicit Beta-Logistic link state"
                .to_string(),
        )),
    }
}

#[inline]
fn component_inverse_link_mu_d1(component: LinkComponent, eta: f64) -> (f64, f64) {
    // The full per-component jet already factors `mu` and `d1` exactly the same
    // way the higher orders are derived, so we either reuse the cheap closed
    // forms directly (Logit/Probit/CLogLog/LogLog/Cauchit) or fall back to the
    // existing canonicalised jet for the few cases without a separate fast
    // path — bit-identical to `component_inverse_link_jet(...).{mu,d1}`.
    match component {
        LinkComponent::Logit => {
            let jet = logit_inverse_link_jet5(eta);
            (jet.mu, canonicalzero(jet.d1))
        }
        LinkComponent::Probit => {
            if eta.is_nan() {
                return (f64::NAN, f64::NAN);
            }
            if eta == f64::INFINITY {
                return (1.0, 0.0);
            }
            if eta == f64::NEG_INFINITY {
                return (0.0, 0.0);
            }
            let phi = normal_pdf(eta);
            (normal_cdf(eta), canonicalzero(phi))
        }
        LinkComponent::CLogLog => {
            if eta.is_nan() {
                return (f64::NAN, f64::NAN);
            }
            let t = eta.exp();
            if !t.is_finite() {
                return (1.0, 0.0);
            }
            (
                -(-t).exp_m1(),
                canonicalzero(stable_nonnegative_poly_times_exp_neg(t, &[0.0, 1.0])),
            )
        }
        LinkComponent::LogLog => {
            if eta.is_nan() {
                return (f64::NAN, f64::NAN);
            }
            let r = (-eta).exp();
            if !r.is_finite() {
                return (0.0, 0.0);
            }
            (
                (-r).exp(),
                canonicalzero(stable_nonnegative_poly_times_exp_neg(r, &[0.0, 1.0])),
            )
        }
        LinkComponent::Cauchit => {
            if eta.is_nan() {
                return (f64::NAN, f64::NAN);
            }
            let den = 1.0 + eta * eta;
            let d1 = if eta.is_finite() {
                1.0 / (std::f64::consts::PI * den)
            } else {
                0.0
            };
            (0.5 + eta.atan() / std::f64::consts::PI, canonicalzero(d1))
        }
    }
}

fn sas_inverse_link_mu_d1(
    eta: f64,
    epsilon: f64,
    log_delta: f64,
) -> Result<(f64, f64), EstimationError> {
    let eta = finite_inverse_link_eta("SAS inverse link", eta)?;
    let delta_id = sas_delta_from_raw_log_delta(log_delta);
    if epsilon.abs() < 1e-12 && (delta_id - 1.0).abs() < 1e-12 {
        return Ok(component_inverse_link_mu_d1(LinkComponent::Probit, eta));
    }
    let asinh = asinh_jet5(eta);
    let delta = delta_id;
    let u_raw = delta * asinh.value + epsilon;
    let u = tanh_bound(u_raw, SAS_U_CLAMP);
    let g1 = tanh_bound_d1(u_raw, SAS_U_CLAMP);
    let s = u.sinh();
    let c = u.cosh();
    let z = s;
    let r1 = delta * asinh.d1;
    let u1 = g1 * r1;
    let z1 = c * u1;
    // `mu = Phi(z)` and `d1 = phi(z) * z1`, the same closed forms used by the
    // full jet via `chain_inverse_link_jet(probit_jet(z), z1, _, _)`.
    let base = probit_jet(z);
    Ok((base.mu, canonicalzero(base.d1 * z1)))
}

fn beta_logistic_inverse_link_mu_d1(eta: f64, delta: f64, epsilon: f64) -> (f64, f64) {
    let logistic = logistic_uwith_derivatives(eta);
    let a = (delta - epsilon).exp();
    let b = (delta + epsilon).exp();
    let mu = beta_reg_logistic(a, b, logistic);
    let log_d1 = beta_logistic_log_d1(a, b, logistic);
    (mu, log_d1.exp())
}

fn mixture_inverse_link_mu_d1(state: &MixtureLinkState, eta: f64) -> (f64, f64) {
    let mut mu = 0.0_f64;
    let mut d1 = 0.0_f64;
    let k = state.components.len().min(state.pi.len());
    for i in 0..k {
        let (mu_i, d1_i) = component_inverse_link_mu_d1(state.components[i], eta);
        let w = state.pi[i];
        mu += w * mu_i;
        d1 += w * d1_i;
    }
    (mu, d1)
}

#[derive(Clone, Copy)]
enum PdfDerivativeOrder {
    Third,
    Fourth,
}

impl PdfDerivativeOrder {
    fn probit(self, eta: f64) -> f64 {
        match self {
            Self::Third => probit_pdfthird_derivative(eta),
            Self::Fourth => probit_pdffourth_derivative(eta),
        }
    }

    fn component(self, component: LinkComponent, eta: f64) -> f64 {
        match self {
            Self::Third => component_inverse_link_pdfthird_derivative(component, eta),
            Self::Fourth => component_inverse_link_pdffourth_derivative(component, eta),
        }
    }

    fn latent_cloglog(self, eta: f64, latent_sd: f64) -> Result<f64, EstimationError> {
        let jet = latent_cloglog_jet5(latent_cloglog_quadctx(), eta, latent_sd)?;
        Ok(match self {
            Self::Third => jet.d4,
            Self::Fourth => jet.d5,
        })
    }

    fn sas(self, eta: f64, epsilon: f64, log_delta: f64) -> Result<f64, EstimationError> {
        match self {
            Self::Third => sas_inverse_link_pdfthird_derivative(eta, epsilon, log_delta),
            Self::Fourth => sas_inverse_link_pdffourth_derivative(eta, epsilon, log_delta),
        }
    }

    fn beta_logistic(self, eta: f64, log_shape_center: f64, epsilon: f64) -> f64 {
        match self {
            Self::Third => {
                beta_logistic_inverse_link_pdfthird_derivative(eta, log_shape_center, epsilon)
            }
            Self::Fourth => {
                beta_logistic_inverse_link_pdffourth_derivative(eta, log_shape_center, epsilon)
            }
        }
    }
}

fn inverse_link_pdf_derivative_for_inverse_link(
    link: &InverseLink,
    eta: f64,
    order: PdfDerivativeOrder,
) -> Result<f64, EstimationError> {
    match link {
        InverseLink::Standard(StandardLink::Identity) => Ok(0.0),
        InverseLink::Standard(StandardLink::Log) => log_link_solver_exp(eta),
        InverseLink::Standard(StandardLink::Probit) => Ok(order.probit(eta)),
        InverseLink::Standard(StandardLink::Logit) => {
            Ok(order.component(LinkComponent::Logit, eta))
        }
        InverseLink::Standard(StandardLink::CLogLog) => {
            Ok(order.component(LinkComponent::CLogLog, eta))
        }
        InverseLink::Standard(StandardLink::LogLog) => {
            Ok(order.component(LinkComponent::LogLog, eta))
        }
        InverseLink::Standard(StandardLink::Cauchit) => {
            Ok(order.component(LinkComponent::Cauchit, eta))
        }
        InverseLink::LatentCLogLog(state) => order.latent_cloglog(eta, state.latent_sd),
        InverseLink::Sas(state) => order.sas(eta, state.epsilon, state.log_delta),
        InverseLink::BetaLogistic(state) => {
            Ok(order.beta_logistic(eta, state.log_delta, state.epsilon))
        }
        InverseLink::Mixture(state) => Ok(state
            .components
            .iter()
            .zip(state.pi.iter())
            .map(|(&component, &weight)| weight * order.component(component, eta))
            .sum()),
    }
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
    // Standard-log inputs outside the declared solver domain return the same
    // typed refusal as the lower-order jet seams.
    inverse_link_pdf_derivative_for_inverse_link(link, eta, PdfDerivativeOrder::Third)
}

/// Fifth derivative of the inverse-link CDF (= fourth derivative of the PDF).
///
/// Extends `inverse_link_pdfthird_derivative_for_inverse_link` by one order.
/// Used for the outer REML Hessian Q[v_k, v_l] term in survival models,
/// specifically the `m1 * u_{abcd}` Arbogast contribution.
/// Standard-log inputs obey the same solver domain as every lower-order seam.
pub fn inverse_link_pdffourth_derivative_for_inverse_link(
    link: &InverseLink,
    eta: f64,
) -> Result<f64, EstimationError> {
    inverse_link_pdf_derivative_for_inverse_link(link, eta, PdfDerivativeOrder::Fourth)
}

#[inline]
/// Exact Royston-Parmar survival jet `S(eta) = exp(-exp(eta))` for every finite
/// `f64` eta. Scaled polynomial tails preserve representable derivatives after
/// the survival value itself underflows; non-finite eta is a typed refusal.
fn royston_parmar_inverse_link_jet(eta: f64) -> Result<InverseLinkJet, EstimationError> {
    let eta = finite_inverse_link_eta("Royston-Parmar survival inverse link", eta)?;
    let hazard = eta.exp();
    let survival = (-hazard).exp();
    // For S(eta) = exp(-h), h = exp(eta), each derivative is a polynomial in
    // nonnegative h times exp(-h). Evaluate that product in its scaled form so
    // neither h^k nor h itself can create an inf*0 tail. If exp(eta) overflows,
    // the helper returns the exact asymptotic derivative limit 0.
    let d1 = -stable_nonnegative_poly_times_exp_neg(hazard, &[0.0, 1.0]);
    let d2 = stable_nonnegative_poly_times_exp_neg(hazard, &[0.0, -1.0, 1.0]);
    let d3 = stable_nonnegative_poly_times_exp_neg(hazard, &[0.0, -1.0, 3.0, -1.0]);
    Ok(InverseLinkJet {
        mu: survival,
        d1: canonicalzero(d1),
        d2: canonicalzero(d2),
        d3: canonicalzero(d3),
    })
}

pub fn inverse_link_jet_for_family(
    spec: &LikelihoodSpec,
    eta: f64,
) -> Result<InverseLinkJet, EstimationError> {
    // RoystonParmar uses its own analytic survival inverse link irrespective of
    // the (nominal `Identity`) link slot carried in the spec.
    if matches!(spec.response, ResponseFamily::RoystonParmar) {
        return royston_parmar_inverse_link_jet(eta);
    }
    spec.link.jet(eta)
}

/// Exact-public log inverse-link jet: `mu = d1 = d2 = d3 = exp(η)` with no
/// solver-domain restriction. The solver-internal sibling evaluates the same
/// exact expression only on [`LOG_LINK_SOLVER_ETA_MIN`] through
/// [`LOG_LINK_SOLVER_ETA_MAX`] and returns a typed refusal outside it; see issue
/// #963. Every derivative of `exp` is `exp`, so all four jet slots carry the
/// same value — finite wherever representable, `0.0` on underflow, and `+∞` on
/// overflow.
#[inline]
fn log_inverse_link_jet_exact(eta: f64) -> InverseLinkJet {
    let e = eta.exp();
    InverseLinkJet {
        mu: e,
        d1: e,
        d2: e,
        d3: e,
    }
}

/// EXACT public inverse-link jet for response-scale prediction outputs.
///
/// Identical to [`inverse_link_jet_for_family`] for every link EXCEPT the
/// standard `Log` link, where it accepts every IEEE input while the shared
/// solver derivative seam accepts only its declared domain. For example,
/// `eta = 705` remains a valid public prediction (`exp(705) ≈ 1.5e306`) but is
/// a typed solver-domain refusal. Public predictions
/// (`FamilyStrategy::inverse_link_jet`/`inverse_link_array`, the predict mean +
/// delta-method SE path) therefore route here. Within the inclusive solver
/// domain the two paths are byte-identical because both evaluate bare
/// `exp(eta)` (issue #963).
pub fn inverse_link_jet_for_family_public(
    spec: &LikelihoodSpec,
    eta: f64,
) -> Result<InverseLinkJet, EstimationError> {
    if matches!(spec.response, ResponseFamily::RoystonParmar) {
        return royston_parmar_inverse_link_jet(eta);
    }
    if let InverseLink::Standard(StandardLink::Log) = spec.link {
        return Ok(log_inverse_link_jet_exact(eta));
    }
    spec.link.jet(eta)
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
    // If `g_j = pi_j (f_j - f_mix)`, differentiating once more gives
    //
    //   H_jk = (1[j=k] - pi_k) g_j - pi_j g_k.
    //
    // This form reuses the first derivatives already in `djet_drho`, avoids
    // dividing by a possibly tiny mixture weight, and is algebraically
    // symmetric even though floating-point evaluation visits `(j,k)` in one
    // direction. Fill one triangle and mirror it bit-for-bit so downstream PSD
    // certification receives an exactly symmetric matrix.
    let mut d2mu_drho2 = Array2::<f64>::zeros((m, m));
    let mut d2d1_drho2 = Array2::<f64>::zeros((m, m));
    for j in 0..m {
        for k in j..m {
            let diagonal = if j == k { 1.0 } else { 0.0 };
            let mu = (diagonal - state.pi[k]) * djet_drho[j].mu
                - state.pi[j] * djet_drho[k].mu;
            let d1 = (diagonal - state.pi[k]) * djet_drho[j].d1
                - state.pi[j] * djet_drho[k].d1;
            d2mu_drho2[[j, k]] = mu;
            d2mu_drho2[[k, j]] = mu;
            d2d1_drho2[[j, k]] = d1;
            d2d1_drho2[[k, j]] = d1;
        }
    }
    MixtureJetWithRhoPartials {
        jet,
        djet_drho,
        d2mu_drho2,
        d2d1_drho2,
    }
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

#[derive(Clone, Copy)]
struct LogisticU {
    u: f64,
    one_minus_u: f64,
    ln_u: f64,
    ln_one_minus_u: f64,
    du: f64,
    use_upper_tail: bool,
}

#[inline]
fn logistic_uwith_derivatives(eta: f64) -> LogisticU {
    let ln_u = -gam_linalg::utils::stable_softplus(-eta);
    let ln_one_minus_u = -gam_linalg::utils::stable_softplus(eta);
    let u = ln_u.exp();
    let one_minus_u = ln_one_minus_u.exp();
    let du = (ln_u + ln_one_minus_u).exp();
    LogisticU {
        u,
        one_minus_u,
        ln_u,
        ln_one_minus_u,
        du,
        use_upper_tail: eta >= 0.0,
    }
}

#[inline]
fn beta_reg_logistic(a: f64, b: f64, logistic: LogisticU) -> f64 {
    if logistic.ln_u.is_nan() || logistic.ln_one_minus_u.is_nan() {
        return f64::NAN;
    }
    if logistic.ln_u == f64::NEG_INFINITY {
        return 0.0;
    }
    if logistic.ln_one_minus_u == f64::NEG_INFINITY {
        return 1.0;
    }
    if logistic.use_upper_tail {
        1.0 - beta_reg(b, a, logistic.one_minus_u)
    } else {
        beta_reg(a, b, logistic.u)
    }
}

#[derive(Clone, Copy)]
struct BetaShapePartials {
    value: f64,
    da: f64,
    db: f64,
    daa: f64,
    dab: f64,
    dbb: f64,
}

impl BetaShapePartials {
    #[inline]
    fn constant(value: f64) -> Self {
        Self {
            value,
            da: 0.0,
            db: 0.0,
            daa: 0.0,
            dab: 0.0,
            dbb: 0.0,
        }
    }
}

#[inline]
fn beta_reg_with_shape_partials_logistic(
    a: f64,
    b: f64,
    logistic: LogisticU,
) -> BetaShapePartials {
    if logistic.ln_u.is_nan() || logistic.ln_one_minus_u.is_nan() {
        return BetaShapePartials {
            value: f64::NAN,
            da: f64::NAN,
            db: f64::NAN,
            daa: f64::NAN,
            dab: f64::NAN,
            dbb: f64::NAN,
        };
    }
    if logistic.use_upper_tail {
        let tail = beta_reg_with_shape_partials(b, a, logistic.one_minus_u);
        BetaShapePartials {
            value: 1.0 - tail.value,
            da: -tail.db,
            db: -tail.da,
            daa: -tail.dbb,
            dab: -tail.dab,
            dbb: -tail.daa,
        }
    } else {
        beta_reg_with_shape_partials(a, b, logistic.u)
    }
}

#[inline]
fn beta_logistic_log_d1(a: f64, b: f64, logistic: LogisticU) -> f64 {
    a * logistic.ln_u + b * logistic.ln_one_minus_u - ln_beta(a, b)
}

#[derive(Clone, Copy)]
struct ShapeDual {
    v: f64,
    da: f64,
    db: f64,
    daa: f64,
    dab: f64,
    dbb: f64,
}

impl ShapeDual {
    #[inline]
    fn constant(v: f64) -> Self {
        Self {
            v,
            da: 0.0,
            db: 0.0,
            daa: 0.0,
            dab: 0.0,
            dbb: 0.0,
        }
    }

    #[inline]
    fn from_value_partials(v: f64, da: f64, db: f64) -> Self {
        Self {
            v,
            da,
            db,
            daa: 0.0,
            dab: 0.0,
            dbb: 0.0,
        }
    }

    #[inline]
    fn clamp_small(self, floor: f64) -> Self {
        if self.v.abs() < floor {
            Self::constant(floor)
        } else {
            self
        }
    }
}

impl std::ops::Add for ShapeDual {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            v: self.v + rhs.v,
            da: self.da + rhs.da,
            db: self.db + rhs.db,
            daa: self.daa + rhs.daa,
            dab: self.dab + rhs.dab,
            dbb: self.dbb + rhs.dbb,
        }
    }
}

impl std::ops::Sub for ShapeDual {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            v: self.v - rhs.v,
            da: self.da - rhs.da,
            db: self.db - rhs.db,
            daa: self.daa - rhs.daa,
            dab: self.dab - rhs.dab,
            dbb: self.dbb - rhs.dbb,
        }
    }
}

impl std::ops::Mul for ShapeDual {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            v: self.v * rhs.v,
            da: self.da * rhs.v + self.v * rhs.da,
            db: self.db * rhs.v + self.v * rhs.db,
            daa: self.daa * rhs.v + 2.0 * self.da * rhs.da + self.v * rhs.daa,
            dab: self.dab * rhs.v
                + self.da * rhs.db
                + self.db * rhs.da
                + self.v * rhs.dab,
            dbb: self.dbb * rhs.v + 2.0 * self.db * rhs.db + self.v * rhs.dbb,
        }
    }
}

impl std::ops::Div for ShapeDual {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self {
        let inv = 1.0 / rhs.v;
        let inv2 = inv * inv;
        let inv3 = inv2 * inv;
        let reciprocal = Self {
            v: inv,
            da: -rhs.da * inv2,
            db: -rhs.db * inv2,
            daa: 2.0 * rhs.da * rhs.da * inv3 - rhs.daa * inv2,
            dab: 2.0 * rhs.da * rhs.db * inv3 - rhs.dab * inv2,
            dbb: 2.0 * rhs.db * rhs.db * inv3 - rhs.dbb * inv2,
        };
        self * reciprocal
    }
}

impl std::ops::Neg for ShapeDual {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        ShapeDual {
            v: -self.v,
            da: -self.da,
            db: -self.db,
            daa: -self.daa,
            dab: -self.dab,
            dbb: -self.dbb,
        }
    }
}

#[inline]
fn shape_dual(v: f64) -> ShapeDual {
    ShapeDual::constant(v)
}

// Analytic shape partials for I_x(a,b), obtained by differentiating the same
// regularized-beta continued fraction used by statrs. The normalizing term uses
// d log B(a,b) / da = psi(a) - psi(a+b) and likewise for b.
fn beta_reg_with_shape_partials(a0: f64, b0: f64, x0: f64) -> BetaShapePartials {
    if x0 <= 0.0 {
        return BetaShapePartials::constant(0.0);
    }
    if x0 >= 1.0 {
        return BetaShapePartials::constant(1.0);
    }

    let symm_transform = x0 >= (a0 + 1.0) / (a0 + b0 + 2.0);
    let (a, b, x) = if symm_transform {
        (
            ShapeDual::from_value_partials(b0, 0.0, 1.0),
            ShapeDual::from_value_partials(a0, 1.0, 0.0),
            1.0 - x0,
        )
    } else {
        (
            ShapeDual::from_value_partials(a0, 1.0, 0.0),
            ShapeDual::from_value_partials(b0, 0.0, 1.0),
            x0,
        )
    };

    let ln_x = x.ln();
    let ln_1mx = (1.0 - x).ln();
    let psi_ab = digamma(a.v + b.v);
    let log_bt = statrs::function::gamma::ln_gamma(a.v + b.v)
        - statrs::function::gamma::ln_gamma(a.v)
        - statrs::function::gamma::ln_gamma(b.v)
        + a.v * ln_x
        + b.v * ln_1mx;
    let bt_v = log_bt.exp();
    let log_bt_a = psi_ab - digamma(a.v) + ln_x;
    let log_bt_b = psi_ab - digamma(b.v) + ln_1mx;
    let trigamma_ab = trigamma(a.v + b.v);
    let log_bt_aa = trigamma_ab - trigamma(a.v);
    let log_bt_ab = trigamma_ab;
    let log_bt_bb = trigamma_ab - trigamma(b.v);
    let log_bt_da = log_bt_a * a.da + log_bt_b * b.da;
    let log_bt_db = log_bt_a * a.db + log_bt_b * b.db;
    let log_bt_daa = log_bt_aa * a.da * a.da
        + 2.0 * log_bt_ab * a.da * b.da
        + log_bt_bb * b.da * b.da
        + log_bt_a * a.daa
        + log_bt_b * b.daa;
    let log_bt_dab = log_bt_aa * a.da * a.db
        + log_bt_ab * (a.da * b.db + b.da * a.db)
        + log_bt_bb * b.da * b.db
        + log_bt_a * a.dab
        + log_bt_b * b.dab;
    let log_bt_dbb = log_bt_aa * a.db * a.db
        + 2.0 * log_bt_ab * a.db * b.db
        + log_bt_bb * b.db * b.db
        + log_bt_a * a.dbb
        + log_bt_b * b.dbb;
    let bt = ShapeDual {
        v: bt_v,
        da: bt_v * log_bt_da,
        db: bt_v * log_bt_db,
        daa: bt_v * (log_bt_da * log_bt_da + log_bt_daa),
        dab: bt_v * (log_bt_da * log_bt_db + log_bt_dab),
        dbb: bt_v * (log_bt_db * log_bt_db + log_bt_dbb),
    };

    let eps = 0.00000000000000011102230246251565;
    let fpmin = f64::MIN_POSITIVE / eps;
    let one = shape_dual(1.0);
    let qab = a + b;
    let qap = a + one;
    let qam = a - one;
    let mut c = one;
    let mut d = (one - qab * shape_dual(x) / qap).clamp_small(fpmin);
    d = one / d;
    let mut h = d;

    for m in 1..141 {
        let mf = f64::from(m);
        let m2 = mf * 2.0;
        let md = shape_dual(mf);
        let m2d = shape_dual(m2);
        let mut aa = md * (b - md) * shape_dual(x) / ((qam + m2d) * (a + m2d));
        d = (one + aa * d).clamp_small(fpmin);
        c = (one + aa / c).clamp_small(fpmin);
        d = one / d;
        h = h * d * c;

        aa = (a + md).neg() * (qab + md) * shape_dual(x) / ((a + m2d) * (qap + m2d));
        d = (one + aa * d).clamp_small(fpmin);
        c = (one + aa / c).clamp_small(fpmin);
        d = one / d;
        let del = d * c;
        h = h * del;

        if (del.v - 1.0).abs() <= eps {
            let reg = bt * h / a;
            return if symm_transform {
                BetaShapePartials {
                    value: 1.0 - reg.v,
                    da: -reg.da,
                    db: -reg.db,
                    daa: -reg.daa,
                    dab: -reg.dab,
                    dbb: -reg.dbb,
                }
            } else {
                BetaShapePartials {
                    value: reg.v,
                    da: reg.da,
                    db: reg.db,
                    daa: reg.daa,
                    dab: reg.dab,
                    dbb: reg.dbb,
                }
            };
        }
    }
    let reg = bt * h / a;
    if symm_transform {
        BetaShapePartials {
            value: 1.0 - reg.v,
            da: -reg.da,
            db: -reg.db,
            daa: -reg.daa,
            dab: -reg.dab,
            dbb: -reg.dbb,
        }
    } else {
        BetaShapePartials {
            value: reg.v,
            da: reg.da,
            db: reg.db,
            daa: reg.daa,
            dab: reg.dab,
            dbb: reg.dbb,
        }
    }
}

/// Beta-Logistic inverse-link jet for:
///   u = logistic(eta)
///   a = exp(log_shape_center - epsilon), b = exp(log_shape_center + epsilon)
///   mu = I_u(a, b)
///
/// NOTE: `log_shape_center` is the *unconstrained* log of the geometric-mean
/// beta shape (so a·b = exp(2·log_shape_center)). Callers must pass the raw
/// optimization parameter `SasLinkState::log_delta`, NOT the derived positive
/// `SasLinkState::delta = exp(log_shape_center)`.
pub fn beta_logistic_inverse_link_jet(
    eta: f64,
    log_shape_center: f64,
    epsilon: f64,
) -> InverseLinkJet {
    let logistic = logistic_uwith_derivatives(eta);
    let a = (log_shape_center - epsilon).exp();
    let b = (log_shape_center + epsilon).exp();
    let mu = beta_reg_logistic(a, b, logistic);
    let log_d1 = beta_logistic_log_d1(a, b, logistic);
    let d1 = log_d1.exp();
    let t = a * logistic.one_minus_u - b * logistic.u;
    let d2 = d1 * t;
    let d3 = d1 * (t * t - (a + b) * logistic.du);
    InverseLinkJet { mu, d1, d2, d3 }
}

pub fn beta_logistic_inverse_link_pdfthird_derivative(
    eta: f64,
    log_shape_center: f64,
    epsilon: f64,
) -> f64 {
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
    let logistic = logistic_uwith_derivatives(eta);
    let a = (log_shape_center - epsilon).exp();
    let b = (log_shape_center + epsilon).exp();
    let log_d1 = beta_logistic_log_d1(a, b, logistic);
    let d1 = log_d1.exp();
    let c = a + b;
    let t = a * logistic.one_minus_u - b * logistic.u;
    let u2 = logistic.du * (logistic.one_minus_u - logistic.u);
    d1 * (t * t * t - 3.0 * c * t * logistic.du - c * u2)
}

/// Fifth derivative of the beta-logistic inverse-link CDF (= 4th deriv of PDF).
///
/// With `P_4 = t^3 - 3ct*u' - c*u''` giving `d4 = d1 * P_4`, the next order is:
///
///   d5 = d1 * [t^4 - 6c*t^2*u' - 4c*t*u'' + 3c^2*u'^2 - c*u''']
///
/// where u' = u(1-u), u'' = u'(1-2u), u''' = u''(1-2u) - 2*u'^2.
pub fn beta_logistic_inverse_link_pdffourth_derivative(
    eta: f64,
    log_shape_center: f64,
    epsilon: f64,
) -> f64 {
    let logistic = logistic_uwith_derivatives(eta);
    let a = (log_shape_center - epsilon).exp();
    let b = (log_shape_center + epsilon).exp();
    let log_d1 = beta_logistic_log_d1(a, b, logistic);
    let d1 = log_d1.exp();
    let c = a + b;
    let t = a * logistic.one_minus_u - b * logistic.u;
    let u2 = logistic.du * (logistic.one_minus_u - logistic.u);
    let u3 = u2 * (logistic.one_minus_u - logistic.u) - 2.0 * logistic.du * logistic.du;
    let t2 = t * t;
    d1 * (t2 * t2 - 6.0 * c * t2 * logistic.du - 4.0 * c * t * u2
        + 3.0 * c * c * logistic.du * logistic.du
        - c * u3)
}

pub fn beta_logistic_inverse_link_jetwith_param_partials(
    eta: f64,
    log_shape_center: f64,
    epsilon: f64,
) -> SasJetWithParamPartials {
    let logistic = logistic_uwith_derivatives(eta);
    let a = (log_shape_center - epsilon).exp();
    let b = (log_shape_center + epsilon).exp();
    let shape = beta_reg_with_shape_partials_logistic(a, b, logistic);
    let mu = shape.value;
    let dmu_dlog_shape_center = a * shape.da + b * shape.db;
    let dmu_depsilon = -a * shape.da + b * shape.db;
    let log_d1 = beta_logistic_log_d1(a, b, logistic);
    let d1 = log_d1.exp();
    let t = a * logistic.one_minus_u - b * logistic.u;
    let d2 = d1 * t;
    let k = t * t - (a + b) * logistic.du;
    let d3 = d1 * k;
    let jet = InverseLinkJet { mu, d1, d2, d3 };

    let psi_a = digamma(a);
    let psi_b = digamma(b);
    let psi_ab = digamma(a + b);
    let la = logistic.ln_u - psi_a + psi_ab;
    let lb = logistic.ln_one_minus_u - psi_b + psi_ab;

    let partials_for = |a_p: f64, b_p: f64, dmu: f64| -> InverseLinkJet {
        let logd1_p = a_p * la + b_p * lb;
        let d1_p = d1 * logd1_p;
        let t_p = a_p * logistic.one_minus_u - b_p * logistic.u;
        let d2_p = d1_p * t + d1 * t_p;
        let k_p = 2.0 * t * t_p - (a_p + b_p) * logistic.du;
        let d3_p = d1_p * k + d1 * k_p;
        InverseLinkJet {
            mu: dmu,
            d1: d1_p,
            d2: d2_p,
            d3: d3_p,
        }
    };
    let djet_dlog_shape_center = partials_for(a, b, dmu_dlog_shape_center);
    let djet_depsilon = partials_for(-a, b, dmu_depsilon);
    // Parameter order is `(epsilon, log_shape_center)`. The beta shapes obey
    // `a=exp(l-e)`, `b=exp(l+e)`, so their first and second parameter jets are
    // closed form. Contract those jets with the exact `(a,b)` Hessian of the
    // regularized beta CDF for `mu`, and with the exact log-density Hessian for
    // `d1`. No numerical differencing or profile replay enters this path.
    let a_first = [-a, a];
    let b_first = [b, b];
    let a_second = [[a, -a], [-a, a]];
    let b_second = [[b, b], [b, b]];
    let logd1_first = [
        a_first[0] * la + b_first[0] * lb,
        a_first[1] * la + b_first[1] * lb,
    ];
    let trigamma_ab = trigamma(a + b);
    let la_a = trigamma_ab - trigamma(a);
    let la_b = trigamma_ab;
    let lb_a = trigamma_ab;
    let lb_b = trigamma_ab - trigamma(b);
    let mut d2mu_dparams2 = Array2::<f64>::zeros((2, 2));
    let mut d2d1_dparams2 = Array2::<f64>::zeros((2, 2));
    for j in 0..2 {
        for k in j..2 {
            let mu_jk = shape.daa * a_first[j] * a_first[k]
                + shape.dab
                    * (a_first[j] * b_first[k] + b_first[j] * a_first[k])
                + shape.dbb * b_first[j] * b_first[k]
                + shape.da * a_second[j][k]
                + shape.db * b_second[j][k];
            let logd1_jk = la_a * a_first[j] * a_first[k]
                + la_b * a_first[j] * b_first[k]
                + lb_a * b_first[j] * a_first[k]
                + lb_b * b_first[j] * b_first[k]
                + la * a_second[j][k]
                + lb * b_second[j][k];
            let d1_jk = d1 * (logd1_first[j] * logd1_first[k] + logd1_jk);
            d2mu_dparams2[[j, k]] = mu_jk;
            d2mu_dparams2[[k, j]] = mu_jk;
            d2d1_dparams2[[j, k]] = d1_jk;
            d2d1_dparams2[[k, j]] = d1_jk;
        }
    }
    SasJetWithParamPartials {
        jet,
        djet_depsilon,
        djet_dlog_delta: djet_dlog_shape_center,
        d2mu_dparams2,
        d2d1_dparams2,
    }
}

/// SAS inverse-link jet for:
///   mu(eta) = Phi(sinh(delta * asinh(eta) + epsilon)),
///   delta = exp(B * tanh(log_delta / B)), B = SAS_LOG_DELTA_BOUND.
///
/// The mathematical solver domain is every finite `f64` eta. Non-finite eta
/// returns [`EstimationError::InverseLinkDomainViolation`]; no value is
/// substituted. The asinh derivatives are evaluated through a scaled jet so
/// both finite endpoints of the domain remain numerically well defined.
pub fn sas_inverse_link_jet(
    eta: f64,
    epsilon: f64,
    log_delta: f64,
) -> Result<InverseLinkJet, EstimationError> {
    let eta = finite_inverse_link_eta("SAS inverse link", eta)?;
    let delta_id = sas_delta_from_raw_log_delta(log_delta);
    if epsilon.abs() < 1e-12 && (delta_id - 1.0).abs() < 1e-12 {
        return Ok(component_inverse_link_jet(LinkComponent::Probit, eta));
    }
    let asinh = asinh_jet5(eta);
    let delta = delta_id;
    let u_raw = delta * asinh.value + epsilon;
    let u = tanh_bound(u_raw, SAS_U_CLAMP);
    let g1 = tanh_bound_d1(u_raw, SAS_U_CLAMP);
    let g2 = tanh_bound_d2(u_raw, SAS_U_CLAMP);
    let g3 = tanh_bound_d3(u_raw, SAS_U_CLAMP);
    let s = u.sinh();
    let c = u.cosh();
    let z = s;
    let r1 = delta * asinh.d1;
    let r2 = delta * asinh.d2;
    let r3 = delta * asinh.d3;
    let u1 = g1 * r1;
    let u2 = g2 * r1 * r1 + g1 * r2;
    let u3 = g3 * r1 * r1 * r1 + 3.0 * g2 * r1 * r2 + g1 * r3;
    let z1 = c * u1;
    let z2 = s * u1 * u1 + c * u2;
    let z3 = c * u1 * u1 * u1 + 3.0 * s * u1 * u2 + c * u3;
    let base = probit_jet(z);
    Ok(chain_inverse_link_jet(base, z1, z2, z3))
}

/// Fourth eta derivative of the SAS inverse-link CDF on the same finite domain
/// as [`sas_inverse_link_jet`].
pub fn sas_inverse_link_pdfthird_derivative(
    eta: f64,
    epsilon: f64,
    log_delta: f64,
) -> Result<f64, EstimationError> {
    // SAS link with bounded latent transform:
    //
    //   a  = asinh(eta),
    //   u  = tanh_bound(delta * a + epsilon),
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
    // which is the standard scalar Arbogast expansion for order four.
    let eta = finite_inverse_link_eta("SAS inverse link", eta)?;
    let asinh = asinh_jet5(eta);
    let delta = sas_delta_from_raw_log_delta(log_delta);
    let u_raw = delta * asinh.value + epsilon;
    let u = tanh_bound(u_raw, SAS_U_CLAMP);
    let g1 = tanh_bound_d1(u_raw, SAS_U_CLAMP);
    let g2 = tanh_bound_d2(u_raw, SAS_U_CLAMP);
    let g3 = tanh_bound_d3(u_raw, SAS_U_CLAMP);
    let g4 = tanh_bound_d4(u_raw, SAS_U_CLAMP);
    let s = u.sinh();
    let c = u.cosh();
    let z = s;
    let base = probit_jet(z);
    let r1 = delta * asinh.d1;
    let r2 = delta * asinh.d2;
    let r3 = delta * asinh.d3;
    let r4 = delta * asinh.d4;
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
    Ok(canonicalzero(out))
}

/// Fifth derivative of the SAS inverse-link CDF (= fourth derivative of the PDF).
///
/// Extends `sas_inverse_link_pdfthird_derivative` by one more derivative order,
/// using the same composition chain u(eta) = g(r(eta)), z = sinh(u), mu = Phi(z).
///
/// The Arbogast expansion at order 5 for u(eta) = g(r(eta)) is:
///   u5 = g5 r1^5 + 10 g4 r1^3 r2 + 15 g3 r1 r2^2 + 10 g3 r1^2 r3
///        + 10 g2 r2 r3 + 5 g2 r1 r4 + g1 r5
///
/// The z = sinh(u) expansion at order 5 is the standard Arbogast for sinh:
///   z5 = c*u1^5 + 10*s*u1^3*u2 + 15*c*u1*u2^2 + 10*c*u1^2*u3
///        + 10*s*u2*u3 + 5*s*u1*u4 + c*u5
///
/// The mu = Phi(z) expansion at order 5 uses probit derivatives:
///   mu^(5) = Phi5*z1^5 + 10*Phi4*z1^3*z2 + 15*Phi3*z1*z2^2 + 10*Phi3*z1^2*z3
///            + 10*Phi2*z2*z3 + 5*Phi2*z1*z4 + Phi1*z5
///
/// Non-finite eta is rejected by the shared SAS finite-domain contract.
pub fn sas_inverse_link_pdffourth_derivative(
    eta: f64,
    epsilon: f64,
    log_delta: f64,
) -> Result<f64, EstimationError> {
    let eta = finite_inverse_link_eta("SAS inverse link", eta)?;
    let asinh = asinh_jet5(eta);
    let delta = sas_delta_from_raw_log_delta(log_delta);
    let u_raw = delta * asinh.value + epsilon;
    let u = tanh_bound(u_raw, SAS_U_CLAMP);
    let g1 = tanh_bound_d1(u_raw, SAS_U_CLAMP);
    let g2 = tanh_bound_d2(u_raw, SAS_U_CLAMP);
    let g3 = tanh_bound_d3(u_raw, SAS_U_CLAMP);
    let g4 = tanh_bound_d4(u_raw, SAS_U_CLAMP);
    let g5 = tanh_bound_d5(u_raw, SAS_U_CLAMP);
    let s = u.sinh();
    let c = u.cosh();
    let z = s;

    // Probit derivatives at z.
    let base = probit_jet(z);
    let phi3 = probit_pdfthird_derivative(z); // Phi^{(4)}
    let phi4 = probit_pdffourth_derivative(z); // Phi^{(5)}

    let r1 = delta * asinh.d1;
    let r2 = delta * asinh.d2;
    let r3 = delta * asinh.d3;
    let r4 = delta * asinh.d4;
    let r5 = delta * asinh.d5;

    // u1..u5 via Arbogast for g(r(eta)).
    let u1 = g1 * r1;
    let u2 = g2 * r1 * r1 + g1 * r2;
    let u3 = g3 * r1 * r1 * r1 + 3.0 * g2 * r1 * r2 + g1 * r3;
    let u4 = g4 * r1.powi(4)
        + 6.0 * g3 * r1 * r1 * r2
        + 3.0 * g2 * r2 * r2
        + 4.0 * g2 * r1 * r3
        + g1 * r4;
    let u5 = g5 * r1.powi(5)
        + 10.0 * g4 * r1 * r1 * r1 * r2
        + 15.0 * g3 * r1 * r2 * r2
        + 10.0 * g3 * r1 * r1 * r3
        + 10.0 * g2 * r2 * r3
        + 5.0 * g2 * r1 * r4
        + g1 * r5;

    // z1..z5 via Arbogast for sinh(u(eta)).
    let z1 = c * u1;
    let z2 = s * u1 * u1 + c * u2;
    let z3 = c * u1 * u1 * u1 + 3.0 * s * u1 * u2 + c * u3;
    let z4 =
        s * u1.powi(4) + 6.0 * c * u1 * u1 * u2 + 3.0 * s * u2 * u2 + 4.0 * s * u1 * u3 + c * u4;
    let z5 = c * u1.powi(5)
        + 10.0 * s * u1 * u1 * u1 * u2
        + 15.0 * c * u1 * u2 * u2
        + 10.0 * c * u1 * u1 * u3
        + 10.0 * s * u2 * u3
        + 5.0 * s * u1 * u4
        + c * u5;

    // mu^(5) = Phi^(5)*z1^5 + 10*Phi^(4)*z1^3*z2 + 15*Phi^(3)*z1*z2^2
    //        + 10*Phi^(3)*z1^2*z3 + 10*Phi^(2)*z2*z3 + 5*Phi^(2)*z1*z4 + Phi^(1)*z5
    let out = phi4 * z1.powi(5)
        + 10.0 * phi3 * z1 * z1 * z1 * z2
        + 15.0 * base.d3 * z1 * z2 * z2
        + 10.0 * base.d3 * z1 * z1 * z3
        + 10.0 * base.d2 * z2 * z3
        + 5.0 * base.d2 * z1 * z4
        + base.d1 * z5;
    Ok(canonicalzero(out))
}

/// SAS eta jet plus epsilon/log-delta partial jets. This is fallible for the
/// same reason as the value jet: eta must be finite, and no non-finite eta is
/// silently replaced.
pub fn sas_inverse_link_jetwith_param_partials(
    eta: f64,
    epsilon: f64,
    log_delta: f64,
) -> Result<SasJetWithParamPartials, EstimationError> {
    let eta = finite_inverse_link_eta("SAS inverse link", eta)?;
    let asinh = asinh_jet5(eta);
    let (ld_eff, dld_eff_draw) = sas_effective_log_delta(log_delta);
    let d2ld_eff_draw2 = tanh_bound_d2(log_delta, SAS_LOG_DELTA_BOUND);
    let delta = ld_eff.exp();
    let ddelta_draw = delta * dld_eff_draw;
    let d2delta_draw2 = delta * (dld_eff_draw * dld_eff_draw + d2ld_eff_draw2);
    let u_raw = delta * asinh.value + epsilon;
    let u = tanh_bound(u_raw, SAS_U_CLAMP);
    let g1 = tanh_bound_d1(u_raw, SAS_U_CLAMP);
    let g2 = tanh_bound_d2(u_raw, SAS_U_CLAMP);
    let g3 = tanh_bound_d3(u_raw, SAS_U_CLAMP);
    let g4 = tanh_bound_d4(u_raw, SAS_U_CLAMP);
    let s = u.sinh();
    let c = u.cosh();
    let z = s;
    let a1 = asinh.d1;
    let a2 = asinh.d2;
    let a3 = asinh.d3;
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

    // epsilon partials (raw_u_t = +1).
    let rt_eps = 1.0;
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
    let rt_ld = ddelta_draw * asinh.value;
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

    // Exact parameter Hessians needed by the profiled likelihood. Only `mu`
    // and `d1` enter the scalar log-survival/log-density terms, so carry the
    // two-variable chain to second parameter order without constructing unused
    // Hessians for d2/d3. Parameter order is `(epsilon, raw_log_delta)`.
    let r_t = [rt_eps, rt_ld];
    let r1_t = [r1t_eps, r1t_ld];
    let r_tt = [[0.0, 0.0], [0.0, d2delta_draw2 * asinh.value]];
    let r1_tt = [[0.0, 0.0], [0.0, d2delta_draw2 * a1]];
    let u_t = [u_eps, u_ld];
    let u1_t = [u1_eps, u1_ld];
    let z_t = [c * u_t[0], c * u_t[1]];
    let z1_t = [
        s * u_t[0] * u1 + c * u1_t[0],
        s * u_t[1] * u1 + c * u1_t[1],
    ];
    let mut d2mu_dparams2 = Array2::<f64>::zeros((2, 2));
    let mut d2d1_dparams2 = Array2::<f64>::zeros((2, 2));
    for j in 0..2 {
        for k in j..2 {
            let u_jk = g2 * r_t[j] * r_t[k] + g1 * r_tt[j][k];
            let u1_jk = g3 * r_t[j] * r_t[k] * r1
                + g2 * r_tt[j][k] * r1
                + g2 * r_t[j] * r1_t[k]
                + g2 * r_t[k] * r1_t[j]
                + g1 * r1_tt[j][k];
            let z_jk = s * u_t[j] * u_t[k] + c * u_jk;
            let z1_jk = c * u_t[j] * u_t[k] * u1
                + s * u_jk * u1
                + s * u_t[j] * u1_t[k]
                + s * u_t[k] * u1_t[j]
                + c * u1_jk;
            let mu_jk = base.d2 * z_t[j] * z_t[k] + base.d1 * z_jk;
            let d1_jk = base.d3 * z_t[j] * z_t[k] * z1
                + base.d2 * z_jk * z1
                + base.d2 * z_t[j] * z1_t[k]
                + base.d2 * z_t[k] * z1_t[j]
                + base.d1 * z1_jk;
            d2mu_dparams2[[j, k]] = mu_jk;
            d2mu_dparams2[[k, j]] = mu_jk;
            d2d1_dparams2[[j, k]] = d1_jk;
            d2d1_dparams2[[k, j]] = d1_jk;
        }
    }

    Ok(SasJetWithParamPartials {
        jet,
        djet_depsilon,
        djet_dlog_delta,
        d2mu_dparams2,
        d2d1_dparams2,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_problem::{InverseLink, LikelihoodSpec, LinkComponent, MixtureLinkSpec, SasLinkState};

    fn assert_log_link_domain_error(error: EstimationError, eta: f64) {
        match error {
            EstimationError::InverseLinkDomainViolation {
                link,
                eta: rejected,
                lower,
                upper,
            } => {
                assert_eq!(link, "standard log inverse link");
                if eta.is_nan() {
                    assert!(rejected.is_nan());
                } else {
                    assert_eq!(rejected, eta);
                }
                assert_eq!(lower, LOG_LINK_SOLVER_ETA_MIN);
                assert_eq!(upper, LOG_LINK_SOLVER_ETA_MAX);
            }
            other => panic!("expected typed log-link domain refusal, got {other}"),
        }
    }

    fn assert_finite_eta_domain_error(
        error: EstimationError,
        expected_link: &'static str,
        eta: f64,
    ) {
        match error {
            EstimationError::InverseLinkDomainViolation {
                link,
                eta: rejected,
                lower,
                upper,
            } => {
                assert_eq!(link, expected_link);
                if eta.is_nan() {
                    assert!(rejected.is_nan());
                } else {
                    assert_eq!(rejected, eta);
                }
                assert_eq!(lower, -f64::MAX);
                assert_eq!(upper, f64::MAX);
            }
            other => panic!("expected typed finite-eta domain refusal, got {other}"),
        }
    }

    #[test]
    fn log_link_solver_boundaries_are_inclusive_exact_exp_jets() {
        let link = InverseLink::Standard(StandardLink::Log);
        let spec = LikelihoodSpec::poisson_log();
        for eta in [LOG_LINK_SOLVER_ETA_MIN, LOG_LINK_SOLVER_ETA_MAX] {
            let expected = eta.exp();
            assert!(expected.is_finite() && expected > 0.0);

            let jet = inverse_link_jet_for_inverse_link(&link, eta).expect("boundary jet");
            assert_eq!(
                LinkFunction::Log.jet(eta).expect("kernel boundary jet"),
                jet
            );
            assert_eq!(jet.mu, expected);
            assert_eq!(jet.d1, expected);
            assert_eq!(jet.d2, expected);
            assert_eq!(jet.d3, expected);

            assert_eq!(
                inverse_link_mu_d1_for_inverse_link(&link, eta).expect("boundary mu/d1"),
                (expected, expected)
            );
            assert_eq!(
                inverse_link_pdfthird_derivative_for_inverse_link(&link, eta)
                    .expect("boundary fourth derivative"),
                expected
            );
            assert_eq!(
                inverse_link_pdffourth_derivative_for_inverse_link(&link, eta)
                    .expect("boundary fifth derivative"),
                expected
            );
            assert_eq!(
                inverse_link_jet_for_family(&spec, eta).expect("boundary family jet"),
                jet
            );
        }
    }

    #[test]
    fn log_link_solver_seams_refuse_every_eta_outside_the_declared_domain() {
        let link = InverseLink::Standard(StandardLink::Log);
        let spec = LikelihoodSpec::poisson_log();
        let just_below = f64::from_bits(LOG_LINK_SOLVER_ETA_MIN.to_bits() + 1);
        let just_above = f64::from_bits(LOG_LINK_SOLVER_ETA_MAX.to_bits() + 1);

        for eta in [
            just_below,
            just_above,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NAN,
        ] {
            assert_log_link_domain_error(
                inverse_link_jet_for_inverse_link(&link, eta).expect_err("full jet must refuse"),
                eta,
            );
            assert_log_link_domain_error(
                LinkFunction::Log
                    .jet(eta)
                    .expect_err("kernel jet must refuse"),
                eta,
            );
            assert_log_link_domain_error(
                inverse_link_mu_d1_for_inverse_link(&link, eta)
                    .expect_err("mu/d1 seam must refuse"),
                eta,
            );
            assert_log_link_domain_error(
                inverse_link_pdfthird_derivative_for_inverse_link(&link, eta)
                    .expect_err("fourth derivative seam must refuse"),
                eta,
            );
            assert_log_link_domain_error(
                inverse_link_pdffourth_derivative_for_inverse_link(&link, eta)
                    .expect_err("fifth derivative seam must refuse"),
                eta,
            );
            assert_log_link_domain_error(
                inverse_link_jet_for_family(&spec, eta).expect_err("family jet seam must refuse"),
                eta,
            );
        }
    }

    #[test]
    fn log_link_solver_value_gradient_is_consistent_near_both_domain_edges() {
        let link = InverseLink::Standard(StandardLink::Log);
        let h = 1.0e-5;
        for eta in [
            LOG_LINK_SOLVER_ETA_MIN + 1.0,
            0.0,
            LOG_LINK_SOLVER_ETA_MAX - 1.0,
        ] {
            let jet = inverse_link_jet_for_inverse_link(&link, eta).expect("interior jet");
            let eta_plus = eta + h;
            let eta_minus = eta - h;
            let mu_plus = inverse_link_jet_for_inverse_link(&link, eta_plus)
                .expect("plus jet")
                .mu;
            let mu_minus = inverse_link_jet_for_inverse_link(&link, eta_minus)
                .expect("minus jet")
                .mu;
            let finite_difference = (mu_plus - mu_minus) / (eta_plus - eta_minus);
            let relative_error = ((finite_difference - jet.d1) / jet.d1).abs();
            assert!(
                relative_error < 5.0e-10,
                "log-link value/gradient mismatch at eta={eta}: analytic={}, finite_difference={}, relative_error={relative_error}",
                jet.d1,
                finite_difference
            );
        }
    }

    #[test]
    fn subnormal_inverse_link_derivatives_are_preserved_not_plateaued() {
        let left_eta = -743.0_f64;
        let left_scale = left_eta.exp();
        assert!(left_scale > 0.0 && left_scale < f64::MIN_POSITIVE);
        let left = logit_inverse_link_jet5(left_eta);
        for (order, derivative) in [left.d1, left.d2, left.d3, left.d4, left.d5]
            .into_iter()
            .enumerate()
        {
            assert!(
                derivative > 0.0 && derivative < f64::MIN_POSITIVE,
                "left-tail logit derivative order {} lost its represented subnormal: {derivative}",
                order + 1
            );
        }

        let right_eta = 743.0_f64;
        let right_scale = (-right_eta).exp();
        assert!(right_scale > 0.0 && right_scale < f64::MIN_POSITIVE);
        let right = logit_inverse_link_jet5(right_eta);
        for (order, derivative, sign) in [
            (1, right.d1, 1.0),
            (2, right.d2, -1.0),
            (3, right.d3, 1.0),
            (4, right.d4, -1.0),
            (5, right.d5, 1.0),
        ] {
            assert_eq!(derivative.signum(), sign, "wrong order-{order} tail sign");
            assert!(
                derivative.abs() > 0.0 && derivative.abs() < f64::MIN_POSITIVE,
                "right-tail logit derivative order {order} lost its represented subnormal: {derivative}"
            );
        }

        let royston_eta = 735.0_f64.ln();
        let royston = royston_parmar_inverse_link_jet(royston_eta)
            .expect("finite Royston-Parmar subnormal-tail eta");
        assert!(
            royston.d1 < 0.0 && royston.d1.abs() < f64::MIN_POSITIVE,
            "Royston-Parmar exact tail derivative must retain its subnormal: {}",
            royston.d1
        );
    }

    #[test]
    fn sas_all_derivative_seams_refuse_nonfinite_eta_with_one_typed_contract() {
        let state = sas_link_state_from_raw(0.25, -0.35).expect("SAS state");
        let link = InverseLink::Sas(state);
        for eta in [f64::NEG_INFINITY, f64::INFINITY, f64::NAN] {
            assert_finite_eta_domain_error(
                sas_inverse_link_jet(eta, state.epsilon, state.log_delta)
                    .expect_err("SAS full jet must refuse"),
                "SAS inverse link",
                eta,
            );
            assert_finite_eta_domain_error(
                sas_inverse_link_mu_d1(eta, state.epsilon, state.log_delta)
                    .expect_err("SAS mu/d1 must refuse"),
                "SAS inverse link",
                eta,
            );
            assert_finite_eta_domain_error(
                sas_inverse_link_pdfthird_derivative(eta, state.epsilon, state.log_delta)
                    .expect_err("SAS fourth derivative must refuse"),
                "SAS inverse link",
                eta,
            );
            assert_finite_eta_domain_error(
                sas_inverse_link_pdffourth_derivative(eta, state.epsilon, state.log_delta)
                    .expect_err("SAS fifth derivative must refuse"),
                "SAS inverse link",
                eta,
            );
            assert_finite_eta_domain_error(
                sas_inverse_link_jetwith_param_partials(eta, state.epsilon, state.log_delta)
                    .expect_err("SAS parameter partials must refuse"),
                "SAS inverse link",
                eta,
            );
            assert_finite_eta_domain_error(
                inverse_link_jet_for_inverse_link(&link, eta).expect_err("SAS kernel must refuse"),
                "SAS inverse link",
                eta,
            );
            assert_finite_eta_domain_error(
                inverse_link_mu_d1_for_inverse_link(&link, eta)
                    .expect_err("SAS fast dispatch must refuse"),
                "SAS inverse link",
                eta,
            );
            assert_finite_eta_domain_error(
                inverse_link_pdfthird_derivative_for_inverse_link(&link, eta)
                    .expect_err("SAS fourth-derivative dispatch must refuse"),
                "SAS inverse link",
                eta,
            );
            assert_finite_eta_domain_error(
                inverse_link_pdffourth_derivative_for_inverse_link(&link, eta)
                    .expect_err("SAS fifth-derivative dispatch must refuse"),
                "SAS inverse link",
                eta,
            );
        }
    }

    #[test]
    fn sas_jets_are_finite_at_both_finite_f64_domain_edges() {
        let state = sas_link_state_from_raw(0.25, -0.35).expect("SAS state");
        for eta in [-f64::MAX, f64::MAX] {
            let jet = sas_inverse_link_jet(eta, state.epsilon, state.log_delta)
                .expect("finite SAS boundary jet");
            let partials =
                sas_inverse_link_jetwith_param_partials(eta, state.epsilon, state.log_delta)
                    .expect("finite SAS boundary partials");
            let h4 = sas_inverse_link_pdfthird_derivative(eta, state.epsilon, state.log_delta)
                .expect("finite SAS boundary fourth derivative");
            let h5 = sas_inverse_link_pdffourth_derivative(eta, state.epsilon, state.log_delta)
                .expect("finite SAS boundary fifth derivative");
            for value in [
                jet.mu,
                jet.d1,
                jet.d2,
                jet.d3,
                partials.jet.mu,
                partials.jet.d1,
                partials.jet.d2,
                partials.jet.d3,
                partials.djet_depsilon.mu,
                partials.djet_depsilon.d1,
                partials.djet_depsilon.d2,
                partials.djet_depsilon.d3,
                partials.djet_dlog_delta.mu,
                partials.djet_dlog_delta.d1,
                partials.djet_dlog_delta.d2,
                partials.djet_dlog_delta.d3,
                h4,
                h5,
            ] {
                assert!(
                    value.is_finite(),
                    "non-finite SAS boundary jet at eta={eta}: {value}"
                );
            }
        }
    }

    #[test]
    fn royston_parmar_exact_jet_has_no_former_minus_thirty_plateau() {
        let left =
            royston_parmar_inverse_link_jet(-30.0 - 1.0e-6).expect("finite Royston-Parmar eta");
        let center = royston_parmar_inverse_link_jet(-30.0).expect("finite Royston-Parmar eta");
        let right =
            royston_parmar_inverse_link_jet(-30.0 + 1.0e-6).expect("finite Royston-Parmar eta");

        assert!(
            left.d1 < 0.0,
            "the exact left tail must not be a constant plateau"
        );
        assert!(center.d1 < 0.0 && right.d1 < 0.0);
        let left_relative = ((left.d1 - center.d1) / center.d1).abs();
        let right_relative = ((right.d1 - center.d1) / center.d1).abs();
        assert!(
            left_relative < 2.0e-6,
            "left derivative kink: {left_relative}"
        );
        assert!(
            right_relative < 2.0e-6,
            "right derivative kink: {right_relative}"
        );

        for eta in [-f64::MAX, -40.0, -30.0, 0.0, 7.0, 30.0, 40.0, f64::MAX] {
            let jet = royston_parmar_inverse_link_jet(eta).expect("finite Royston-Parmar eta");
            for value in [jet.mu, jet.d1, jet.d2, jet.d3] {
                assert!(
                    value.is_finite(),
                    "non-finite Royston-Parmar jet at eta={eta}: {value}"
                );
            }
        }
    }

    #[test]
    fn royston_parmar_seams_refuse_nonfinite_eta_instead_of_clamping() {
        let spec = LikelihoodSpec::new(
            ResponseFamily::RoystonParmar,
            InverseLink::Standard(StandardLink::Identity),
        );
        for eta in [f64::NEG_INFINITY, f64::INFINITY, f64::NAN] {
            assert_finite_eta_domain_error(
                royston_parmar_inverse_link_jet(eta)
                    .expect_err("direct Royston-Parmar jet must refuse"),
                "Royston-Parmar survival inverse link",
                eta,
            );
            assert_finite_eta_domain_error(
                inverse_link_jet_for_family(&spec, eta)
                    .expect_err("solver Royston-Parmar jet must refuse"),
                "Royston-Parmar survival inverse link",
                eta,
            );
            assert_finite_eta_domain_error(
                inverse_link_jet_for_family_public(&spec, eta)
                    .expect_err("public Royston-Parmar jet must refuse"),
                "Royston-Parmar survival inverse link",
                eta,
            );
        }
    }

    #[test]
    fn softmax_jacobian_matchesfd() {
        let rho = Array1::from_vec(vec![0.7, -1.2, 0.4]);
        let (pi, jac) = softmaxwith_jacobian_last_fixedzero(&rho);
        let h = 1e-6;
        for j in 0..rho.len() {
            let mut rp = rho.clone();
            rp[j] += h;
            let mut rm = rho.clone();
            rm[j] -= h;
            let pp = softmax_last_fixedzero(&rp);
            let pm = softmax_last_fixedzero(&rm);
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
    fn mixture_second_partials_obey_equal_weight_two_component_identity() {
        let state = state_fromspec(&MixtureLinkSpec {
            components: vec![LinkComponent::Probit, LinkComponent::Logit],
            initial_rho: Array1::from_vec(vec![0.0]),
        })
        .expect("valid two-component mixture");
        let out = mixture_inverse_link_jetwith_rho_partials(&state, 0.37);
        // For two components, f''(rho)=pi(1-pi)(1-2pi)(f0-f1), so both
        // response channels have exactly zero curvature at the equal-weight
        // coordinate rho=0. This checks the analytic softmax Hessian without
        // using a finite-difference oracle.
        assert_eq!(out.d2mu_drho2.dim(), (1, 1));
        assert_eq!(out.d2d1_drho2.dim(), (1, 1));
        assert_eq!(out.d2mu_drho2[[0, 0]], 0.0);
        assert_eq!(out.d2d1_drho2[[0, 0]], 0.0);
    }

    #[test]
    fn sas_param_partials_matchfd() {
        let eta = 0.37;
        let epsilon = -0.12;
        let log_delta = 0.21;
        let out = sas_inverse_link_jetwith_param_partials(eta, epsilon, log_delta)
            .expect("finite SAS eta");
        let h = 1e-6;

        let ep_p = sas_inverse_link_jet(eta, epsilon + h, log_delta).expect("finite SAS eta");
        let ep_m = sas_inverse_link_jet(eta, epsilon - h, log_delta).expect("finite SAS eta");
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

        let ld_p = sas_inverse_link_jet(eta, epsilon, log_delta + h).expect("finite SAS eta");
        let ld_m = sas_inverse_link_jet(eta, epsilon, log_delta - h).expect("finite SAS eta");
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
    fn sas_second_partials_have_exact_center_identities() {
        let out = sas_inverse_link_jetwith_param_partials(0.0, 0.0, 0.0)
            .expect("finite SAS center");
        let phi0 = normal_pdf(0.0);
        let expected_epsilon_d1 = -2.0 * phi0 / (SAS_U_CLAMP * SAS_U_CLAMP);
        assert_eq!(out.d2mu_dparams2, Array2::<f64>::zeros((2, 2)));
        assert_eq!(out.d2d1_dparams2[[0, 1]], out.d2d1_dparams2[[1, 0]]);
        assert!((out.d2d1_dparams2[[0, 0]] - expected_epsilon_d1).abs() < 1.0e-15);
        assert!((out.d2d1_dparams2[[1, 1]] - phi0).abs() < 1.0e-15);
        assert_eq!(out.d2d1_dparams2[[0, 1]], 0.0);
    }

    /// #1876 closability isolation gate. The SAS-link binomial FAMILY score at
    /// fixed η is `∂ℓ/∂ε = a1 · ∂μ/∂ε`, with the binomial score
    /// `a1 = w(y/μ − (1−y)/(1−μ))` and `∂μ/∂ε = djet_depsilon.mu`. This proves
    /// that single source is correct — in SIGN and MAGNITUDE — against an
    /// independent finite difference of the row log-likelihood over a whole
    /// (η, ε, log_δ) grid and both responses (`sas_param_partials_matchfd` only
    /// checks the pointwise link partials at one point; this composes them into
    /// the objective-level family score).
    ///
    /// Part 2 reproduces the issue's own symptom deterministically: plant
    /// ε*=0.38, δ*=1 and use expected fractional responses yᵢ=μ*(ηᵢ). By the
    /// score identity the summed data-fit ∂ℓ/∂ε then vanishes exactly at ε* and
    /// the negative-log-likelihood profile is minimized there — the summed
    /// ∂(NLL)/∂ε is strongly negative below ε* (pushes ε UP toward the truth),
    /// zero at ε*, strongly positive above, and STRICTLY increasing through it.
    /// That strict monotonicity is exactly what distinguishes +κ/skew from −κ,
    /// the sign-blindness #1876 reported. With the family derivative certified
    /// here, any wrong-sign ε recovery is provably the OUTER REML envelope path
    /// (the capped-β̂ / KKT-residual clobber fixed in 574129459), not this
    /// derivative.
    #[test]
    fn sas_family_score_depsilon_matches_fd_and_reproduces_profile_1876() {
        let h = 1e-6;
        let mu_at = |eta: f64, eps: f64, ld: f64| {
            sas_inverse_link_jet(eta, eps, ld)
                .expect("finite SAS eta")
                .mu
        };
        let dmu_deps = |eta: f64, eps: f64, ld: f64| {
            sas_inverse_link_jetwith_param_partials(eta, eps, ld)
                .expect("finite SAS eta")
                .djet_depsilon
                .mu
        };
        // Binomial row log-likelihood and score dℓ/dμ (== link_binomial_aux.a1).
        let row_ll = |y: f64, w: f64, mu: f64| w * (y * mu.ln() + (1.0 - y) * (1.0 - mu).ln());
        let a1 = |y: f64, w: f64, mu: f64| w * (y / mu - (1.0 - y) / (1.0 - mu));

        // ── Part 1: pointwise family-score correctness across a grid. ──
        let etas = [-1.0, -0.6, -0.2, 0.15, 0.5, 0.9];
        let epsilons = [-0.5, -0.2, 0.0, 0.3, 0.6];
        let log_deltas = [-0.3, 0.0, 0.4];
        for &eta in &etas {
            for &eps in &epsilons {
                for &ld in &log_deltas {
                    let mu0 = mu_at(eta, eps, ld);
                    // Stay in the numerically comfortable interior; the far tail
                    // is covered by `sas_jet_extreme_inputs_stay_finite`.
                    if !(0.02..=0.98).contains(&mu0) {
                        continue;
                    }
                    let dmu = dmu_deps(eta, eps, ld);
                    for &y in &[0.0_f64, 1.0] {
                        let analytic = a1(y, 1.0, mu0) * dmu; // dℓ/dε
                        let fd = (row_ll(y, 1.0, mu_at(eta, eps + h, ld))
                            - row_ll(y, 1.0, mu_at(eta, eps - h, ld)))
                            / (2.0 * h);
                        let scale = analytic.abs().max(fd.abs()).max(1.0);
                        assert_eq!(
                            analytic.signum(),
                            fd.signum(),
                            "∂ℓ/∂ε sign mismatch η={eta} ε={eps} log_δ={ld} y={y}: \
                             analytic={analytic:e} fd={fd:e}"
                        );
                        assert!(
                            (analytic - fd).abs() < 1e-5 * scale,
                            "∂ℓ/∂ε magnitude mismatch η={eta} ε={eps} log_δ={ld} y={y}: \
                             analytic={analytic:e} fd={fd:e}"
                        );
                    }
                }
            }
        }

        // ── Part 2: summed-profile symptom reproduction (deterministic). ──
        let eps_true = 0.38;
        let ld_true = 0.0;
        let etas_ds: Vec<f64> = (0..21).map(|i| -1.0 + 0.1 * i as f64).collect();
        let y: Vec<f64> = etas_ds
            .iter()
            .map(|&e| mu_at(e, eps_true, ld_true))
            .collect();

        // Summed ∂(NLL)/∂ε = −Σ a1·∂μ/∂ε, analytic and by FD of the summed NLL.
        let grad_nll = |eps: f64| -> (f64, f64) {
            let mut analytic = 0.0;
            let mut nll_p = 0.0;
            let mut nll_m = 0.0;
            for (i, &eta) in etas_ds.iter().enumerate() {
                let mu0 = mu_at(eta, eps, ld_true);
                analytic += -a1(y[i], 1.0, mu0) * dmu_deps(eta, eps, ld_true);
                nll_p += -row_ll(y[i], 1.0, mu_at(eta, eps + h, ld_true));
                nll_m += -row_ll(y[i], 1.0, mu_at(eta, eps - h, ld_true));
            }
            (analytic, (nll_p - nll_m) / (2.0 * h))
        };

        // (a) analytic == FD at every probe ε.
        for &eps in &[0.0, eps_true, 0.6] {
            let (analytic, fd) = grad_nll(eps);
            let scale = analytic.abs().max(fd.abs()).max(1.0);
            assert!(
                (analytic - fd).abs() < 1e-4 * scale,
                "summed ∂NLL/∂ε analytic≠fd at ε={eps}: {analytic:e} vs {fd:e}"
            );
        }
        // (b) minimum exactly at the planted ε*: strictly increasing through it.
        let (g_below, _) = grad_nll(0.0);
        let (g_at, _) = grad_nll(eps_true);
        let (g_above, _) = grad_nll(0.6);
        assert!(
            g_below < -1.0,
            "expected strongly negative ∂NLL/∂ε below ε* (pushes ε up toward truth), got {g_below:e}"
        );
        assert!(
            g_at.abs() < 1e-6,
            "expected ≈0 ∂NLL/∂ε at the planted ε* (score identity), got {g_at:e}"
        );
        assert!(
            g_above > 1.0,
            "expected strongly positive ∂NLL/∂ε above ε*, got {g_above:e}"
        );
        assert!(
            g_below < g_at && g_at < g_above,
            "∂NLL/∂ε must strictly increase through ε* (distinguishes ±ε): \
             {g_below:e} < {g_at:e} < {g_above:e}"
        );
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
            let j = sas_inverse_link_jet(eta, eps, log_delta).expect("finite SAS eta");
            assert!(j.mu.is_finite());
            assert!(j.d1.is_finite());
            assert!(j.d2.is_finite());
            assert!(j.d3.is_finite());
            let p = sas_inverse_link_jetwith_param_partials(eta, eps, log_delta)
                .expect("finite SAS eta");
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
        let j = sas_inverse_link_jetwith_param_partials(eta, epsilon, log_delta)
            .expect("finite SAS eta");
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
        let j0 = sas_inverse_link_jet(eta, epsilon, log_delta).expect("finite SAS eta");
        let jp = sas_inverse_link_jet(eta + h, epsilon, log_delta).expect("finite SAS eta");
        let jm = sas_inverse_link_jet(eta - h, epsilon, log_delta).expect("finite SAS eta");
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
    fn family_dispatch_resolves_parameterized_links_from_spec() {
        // After the LikelihoodSpec migration, the dispatch no longer needs
        // out-of-band state arguments — the parameterized link state lives on
        // `spec.link`. Pin the dispatch against the direct stateful kernels.
        let sas_state = sas_link_state_from_raw(0.0, 0.0).expect("sas state");
        let expected_sas =
            sas_inverse_link_jet(0.1, sas_state.epsilon, sas_state.log_delta).expect("direct SAS");
        let sas_spec = gam_problem::LikelihoodSpec {
            response: gam_problem::ResponseFamily::Binomial,
            link: InverseLink::Sas(sas_state),
        };
        let sas_jet = inverse_link_jet_for_family(&sas_spec, 0.1).expect("sas jet");
        assert_eq!(sas_jet.mu.to_bits(), expected_sas.mu.to_bits());
        assert_eq!(sas_jet.d1.to_bits(), expected_sas.d1.to_bits());

        let mix_state = MixtureLinkState {
            components: vec![LinkComponent::Logit, LinkComponent::Probit],
            rho: ndarray::array![0.0],
            pi: ndarray::array![0.5, 0.5],
        };
        let expected_mix = mixture_inverse_link_jet(&mix_state, 0.1);
        let mix_spec = gam_problem::LikelihoodSpec {
            response: gam_problem::ResponseFamily::Binomial,
            link: InverseLink::Mixture(mix_state),
        };
        let mix_jet = inverse_link_jet_for_family(&mix_spec, 0.1).expect("mix jet");
        assert_eq!(mix_jet.mu.to_bits(), expected_mix.mu.to_bits());
        assert_eq!(mix_jet.d1.to_bits(), expected_mix.d1.to_bits());
    }

    #[test]
    fn beta_logistic_reduces_to_logit_at_delta0_epsilon0() {
        let etas = [-40.0, -30.0, -5.0, 0.42, 5.0, 30.0, 40.0];
        for eta in etas {
            let j_bl = beta_logistic_inverse_link_jet(eta, 0.0, 0.0);
            let expected_mu = gam_linalg::utils::stable_logistic(eta);
            let expected_d1 = (-gam_linalg::utils::stable_softplus(-eta)
                - gam_linalg::utils::stable_softplus(eta))
            .exp();
            assert!(
                (j_bl.mu - expected_mu).abs() <= 1e-15 * expected_mu.abs().max(1.0),
                "mu mismatch at eta={eta}: got {}, expected {}",
                j_bl.mu,
                expected_mu
            );
            assert!(
                (j_bl.d1 - expected_d1).abs() <= 1e-12 * expected_d1.abs().max(f64::MIN_POSITIVE),
                "d1 mismatch at eta={eta}: got {}, expected {}",
                j_bl.d1,
                expected_d1
            );
            assert!(j_bl.d1 > 0.0, "d1 should stay positive at eta={eta}");
        }

        let eta = 0.42;
        let j_bl = beta_logistic_inverse_link_jet(eta, 0.0, 0.0);
        let j_logit = component_inverse_link_jet(LinkComponent::Logit, eta);
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
            let sas = sas_inverse_link_jet(eta, 0.0, 0.0).expect("finite SAS eta");
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
    fn beta_logistic_second_partials_obey_center_symmetry() {
        let out = beta_logistic_inverse_link_jetwith_param_partials(0.0, 0.37, 0.0);
        // At eta=0 and epsilon=0, a=b for every log-shape center, hence
        // I_{1/2}(a,a)=1/2 identically. Its pure epsilon and pure log-shape
        // second derivatives vanish by complement symmetry, while the density
        // is even in epsilon and therefore has zero mixed derivative there.
        assert_eq!(out.d2mu_dparams2[[0, 1]], out.d2mu_dparams2[[1, 0]]);
        assert_eq!(out.d2d1_dparams2[[0, 1]], out.d2d1_dparams2[[1, 0]]);
        assert!(out.d2mu_dparams2[[0, 0]].abs() < 1.0e-12);
        assert!(out.d2mu_dparams2[[1, 1]].abs() < 1.0e-12);
        assert!(out.d2d1_dparams2[[0, 1]].abs() < 1.0e-12);
    }

    #[test]
    fn beta_logistic_left_tail_uses_unclamped_log_space() {
        let eta = -40.0_f64;
        let delta = 0.2_f64;
        let epsilon = -0.1_f64;
        let a = (delta - epsilon).exp();
        let b = (delta + epsilon).exp();
        let expected_mu = beta_reg(a, b, eta.exp());
        let out = beta_logistic_inverse_link_jet(eta, delta, epsilon);

        assert!(
            (out.mu - expected_mu).abs() <= 1e-12 * expected_mu.abs().max(f64::MIN_POSITIVE),
            "left-tail mu mismatch: got {}, expected {}",
            out.mu,
            expected_mu
        );
        assert!(out.d1 > 0.0);
        assert!(out.d2 > 0.0);
        assert!(out.d3 > 0.0);
        assert!(out.d1 < 1e-20);

        let partials = beta_logistic_inverse_link_jetwith_param_partials(eta, delta, epsilon);
        assert!(partials.jet.d1 > 0.0);
        assert!(partials.jet.d2 > 0.0);
        assert!(partials.jet.d3 > 0.0);
        assert!(partials.djet_dlog_delta.d1.is_finite());
        assert!(partials.djet_depsilon.d1.is_finite());
    }

    #[test]
    fn beta_logistic_mu_is_symmetric_in_logistic_tails() {
        let delta = 0.2;
        let epsilon = -0.35;
        let etas = [-40.0, -30.0, -5.0, -0.42, 0.0, 0.42, 5.0, 30.0, 40.0];
        for eta in etas {
            let left = beta_logistic_inverse_link_jet(eta, delta, epsilon).mu;
            let right = 1.0 - beta_logistic_inverse_link_jet(-eta, delta, -epsilon).mu;
            assert!(
                (left - right).abs() <= 1e-14,
                "symmetry mismatch at eta={eta}: left={left}, right={right}"
            );
        }
    }

    #[test]
    fn inverse_link_pdfthird_derivative_matches_d3_finite_difference() {
        let sas = InverseLink::Sas(sas_link_state_from_raw(-0.25, 0.35).expect("sas state"));
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
            InverseLink::Standard(StandardLink::Probit),
            InverseLink::Standard(StandardLink::Logit),
            InverseLink::Standard(StandardLink::CLogLog),
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

    #[test]
    fn cloglog_large_finite_eta_should_saturate_without_nan_derivatives() {
        let eta = 800.0;
        let jet = component_inverse_link_jet(LinkComponent::CLogLog, eta);
        assert_eq!(jet.mu, 1.0);
        assert!(
            jet.d1 == 0.0,
            "for mu(eta)=1-exp(-exp(eta)), dmu/deta = exp(eta-exp(eta)) and should underflow to 0 at eta={eta}; got d1={}",
            jet.d1
        );
        assert!(
            jet.d2 == 0.0,
            "the saturated cloglog second derivative should also be 0 at eta={eta}; got d2={}",
            jet.d2
        );
        assert!(
            jet.d3 == 0.0,
            "the saturated cloglog third derivative should also be 0 at eta={eta}; got d3={}",
            jet.d3
        );

        let d4 = inverse_link_pdfthird_derivative_for_inverse_link(
            &InverseLink::Standard(StandardLink::CLogLog),
            eta,
        )
        .expect("cloglog d4");
        assert!(
            d4 == 0.0,
            "the saturated cloglog fourth derivative should also be 0 at eta={eta}; got d4={d4}"
        );
    }

    #[test]
    fn loglog_large_negative_finite_eta_should_saturate_without_nan_derivatives() {
        let eta = -800.0;
        let jet = component_inverse_link_jet(LinkComponent::LogLog, eta);
        assert_eq!(jet.mu, 0.0);
        assert!(
            jet.d1 == 0.0,
            "for mu(eta)=exp(-exp(-eta)), dmu/deta = exp(-eta-exp(-eta)) and should underflow to 0 at eta={eta}; got d1={}",
            jet.d1
        );
        assert!(
            jet.d2 == 0.0,
            "the saturated loglog second derivative should also be 0 at eta={eta}; got d2={}",
            jet.d2
        );
        assert!(
            jet.d3 == 0.0,
            "the saturated loglog third derivative should also be 0 at eta={eta}; got d3={}",
            jet.d3
        );

        let d4 = inverse_link_pdfthird_derivative_for_inverse_link(
            &InverseLink::Mixture(
                state_fromspec(&MixtureLinkSpec {
                    components: vec![LinkComponent::LogLog, LinkComponent::Probit],
                    initial_rho: Array1::from_vec(vec![12.0]),
                })
                .expect("mixture state"),
            ),
            eta,
        )
        .expect("loglog mixture d4");
        assert!(
            d4.is_finite(),
            "even a nearly pure loglog mixture should not produce NaN fourth derivatives at eta={eta}; got d4={d4}"
        );
    }

    #[test]
    fn logit_tail_derivatives_should_match_stable_closed_forms() {
        let eta = 50.0_f64;
        let z = (-eta).exp();
        let denom = 1.0_f64 + z;
        let stable_d1 = z / denom.powi(2);
        let stable_d2 = z * (z - 1.0) / denom.powi(3);
        let stable_d3 = z * (z * z - 4.0 * z + 1.0) / denom.powi(4);
        let stable_d4 = z * (z * z * z - 11.0 * z * z + 11.0 * z - 1.0) / denom.powi(5);
        let stable_d5 =
            z * (z * z * z * z - 26.0 * z * z * z + 66.0 * z * z - 26.0 * z + 1.0) / denom.powi(6);

        assert!(stable_d1 > 0.0);
        assert!(stable_d2 < 0.0);
        assert!(stable_d3 > 0.0);
        assert!(stable_d4 < 0.0);
        assert!(stable_d5 > 0.0);

        let jet = component_inverse_link_jet(LinkComponent::Logit, eta);
        assert!(
            (jet.d1 - stable_d1).abs() < 1e-30,
            "logit d1 should equal the stable tail formula z/(1+z)^2 at eta={eta}; got {} vs {}",
            jet.d1,
            stable_d1
        );
        assert!(
            (jet.d2 - stable_d2).abs() < 1e-30,
            "logit d2 should equal the stable tail formula z(z-1)/(1+z)^3 at eta={eta}; got {} vs {}",
            jet.d2,
            stable_d2
        );
        assert!(
            (jet.d3 - stable_d3).abs() < 1e-30,
            "logit d3 should equal the stable tail formula z(z^2-4z+1)/(1+z)^4 at eta={eta}; got {} vs {}",
            jet.d3,
            stable_d3
        );

        let d4 = inverse_link_pdfthird_derivative_for_inverse_link(
            &InverseLink::Standard(StandardLink::Logit),
            eta,
        )
        .expect("logit d4");
        assert!(
            (d4 - stable_d4).abs() < 1e-30,
            "logit d4 should equal the stable tail formula z(z^3-11z^2+11z-1)/(1+z)^5 at eta={eta}; got {} vs {}",
            d4,
            stable_d4
        );

        let d5 = inverse_link_pdffourth_derivative_for_inverse_link(
            &InverseLink::Standard(StandardLink::Logit),
            eta,
        )
        .expect("logit d5");
        assert!(
            (d5 - stable_d5).abs() < 1e-30,
            "logit d5 should equal the stable tail formula z(z^4-26z^3+66z^2-26z+1)/(1+z)^6 at eta={eta}; got {} vs {}",
            d5,
            stable_d5
        );
    }

    #[test]
    fn cloglog_negative_tail_value_should_match_expm1_form() {
        let eta = -50.0_f64;
        let t = eta.exp();
        let stable_mu = -(-t).exp_m1();
        assert!(stable_mu > 0.0);

        let jet = component_inverse_link_jet(LinkComponent::CLogLog, eta);
        assert!(
            (jet.mu - stable_mu).abs() < 1e-30,
            "cloglog mu should equal -expm1(-exp(eta)) in the negative tail at eta={eta}; got {} vs {}",
            jet.mu,
            stable_mu
        );
    }

    #[test]
    fn non_logit_probit_fisher_weight_jets_match_finite_differences() {
        fn rel_err(a: f64, b: f64) -> f64 {
            (a - b).abs() / a.abs().max(b.abs()).max(1.0e-8)
        }

        let cases = [
            (LinkComponent::CLogLog, [-3.0_f64, -0.5, 0.4, 1.5]),
            (LinkComponent::LogLog, [-1.5_f64, -0.4, 0.5, 3.0]),
            (LinkComponent::Cauchit, [-3.0_f64, -0.7, 0.6, 3.0]),
        ];
        for (component, etas) in cases {
            for eta in etas {
                let (w, w1, w2, w3, w4) = component_fisher_weight_jet5(component, eta);
                let jet = component_inverse_link_jet(component, eta);
                let expected = jet.d1 * jet.d1 / (jet.mu * (1.0 - jet.mu));
                assert!(
                    rel_err(w, expected) < 1.0e-12,
                    "{component:?} Fisher weight mismatch at eta={eta}: got {w}, expected {expected}"
                );

                let h = 1.0e-4;
                let fd1 = (component_fisher_weight_jet5(component, eta + h).0
                    - component_fisher_weight_jet5(component, eta - h).0)
                    / (2.0 * h);
                let fd2 = (component_fisher_weight_jet5(component, eta + h).1
                    - component_fisher_weight_jet5(component, eta - h).1)
                    / (2.0 * h);
                let fd3 = (component_fisher_weight_jet5(component, eta + h).2
                    - component_fisher_weight_jet5(component, eta - h).2)
                    / (2.0 * h);
                let fd4 = (component_fisher_weight_jet5(component, eta + h).3
                    - component_fisher_weight_jet5(component, eta - h).3)
                    / (2.0 * h);

                assert!(
                    rel_err(w1, fd1) < 1.0e-5,
                    "{component:?} W' mismatch at eta={eta}: {w1} vs {fd1}"
                );
                assert!(
                    rel_err(w2, fd2) < 1.0e-5,
                    "{component:?} W'' mismatch at eta={eta}: {w2} vs {fd2}"
                );
                assert!(
                    rel_err(w3, fd3) < 5.0e-5,
                    "{component:?} W''' mismatch at eta={eta}: {w3} vs {fd3}"
                );
                assert!(
                    rel_err(w4, fd4) < 5.0e-4,
                    "{component:?} W'''' mismatch at eta={eta}: {w4} vs {fd4}"
                );
            }
        }
    }

    #[test]
    fn mixture_fisher_weight_jet_covers_loglog_and_cauchit_components() {
        let state = state_fromspec(&MixtureLinkSpec {
            components: vec![
                LinkComponent::CLogLog,
                LinkComponent::LogLog,
                LinkComponent::Cauchit,
            ],
            initial_rho: Array1::from_vec(vec![0.3, -0.2]),
        })
        .expect("mixture state");
        let link = InverseLink::Mixture(state);
        assert!(
            link.has_fisher_weight_jet(),
            "anchored mixtures with loglog/cauchit components must remain eligible for Firth"
        );
        assert!(
            LikelihoodSpec::new(ResponseFamily::Binomial, link.clone()).supports_firth(),
            "Firth support should use the mixture inverse-link Fisher jet, not standalone LinkFunction coverage"
        );

        for eta in [-2.0_f64, -0.25, 0.75, 2.5] {
            let (w, w1, w2, w3, w4) =
                fisher_weight_jet5_for_inverse_link(&link, eta).expect("mixture Fisher jet");
            for value in [w, w1, w2, w3, w4] {
                assert!(
                    value.is_finite(),
                    "mixture Fisher weight jet should be finite at eta={eta}; got {value}"
                );
            }
            assert!(
                w > 0.0,
                "mixture Fisher working weight should be positive away from saturated tails at eta={eta}; got {w}"
            );
        }
    }

    #[test]
    fn loglog_fifth_derivative_should_match_closed_form_sign() {
        let eta = 0.0_f64;
        let r = (-eta).exp();
        let expected =
            (-r).exp() * (r - 15.0 * r * r + 25.0 * r.powi(3) - 10.0 * r.powi(4) + r.powi(5));
        let d5 = component_inverse_link_pdffourth_derivative(LinkComponent::LogLog, eta);
        assert!(
            (d5 - expected).abs() < 1e-15,
            "loglog d5 should equal exp(-r) * (r - 15r^2 + 25r^3 - 10r^4 + r^5) at eta={eta}; got {d5} vs {expected}"
        );
        assert!(d5 > 0.0, "loglog d5 should be positive at eta=0; got {d5}");
    }
}
