use crate::estimate::EstimationError;
use crate::quadrature::{latent_cloglog_d6, latent_cloglog_jet5};
use gam_math::{
    probability::{normal_cdf, normal_pdf},
    special::{digamma, trigamma},
};
use gam_math::special::stable_polynomial_times_exp_neg as stable_nonnegative_poly_times_exp_neg;
use gam_problem::{
    InverseLink, LatentCLogLogState, LikelihoodSpec, LinkComponent, LinkFunction, MixtureLinkSpec,
    MixtureLinkState, ResponseFamily, SasLinkSpec, SasLinkState, StandardLink,
};
use ndarray::{Array1, Array2};
use statrs::function::beta::{beta_reg, ln_beta};
use std::ops::Neg;
use std::sync::OnceLock;

/// The half-width of the sinh-arcsinh latent's own domain: the largest `|u|`
/// at which this link's published arithmetic exists (#2902).
///
/// `z = sinh(u)` and `dz/du = cosh(u)` are each bounded below by `e^{|u|}/2 − 1`,
/// and the jet this link publishes runs to `μ⁽⁶⁾`, whose leading factors are
/// their sixth powers (`cosh⁶` from the chain and `sinh⁶` from the probit
/// polynomial). The composition is therefore evaluable exactly while
/// `e^{6|u|} ≤ f64::MAX`, and not at all beyond it — this is a property of the
/// parameterization and binary64, not a box chosen for a search. Past it
/// `smooth_bound_jet`'s compact support makes the saturated branch exact, with
/// every derivative identically zero, so the `0·∞` an overflowing `sinh` would
/// inject is annihilated rather than clamped away.
#[inline]
pub fn sas_latent_domain_bound() -> f64 {
    f64::MAX.ln() / 6.0
}

/// Inclusive eta domain for the solver's standard log inverse-link derivative
/// seams. Within this conservative IEEE-754-safe interval, `exp(eta)` is finite,
/// positive, and normal, so the value and every analytic derivative are exactly
/// the same operation. Solver callers must reject steps outside this domain;
/// silently projecting eta would define a different, nonsmooth link.
pub(crate) const LOG_LINK_SOLVER_ETA_MIN: f64 = -700.0;
/// Inclusive upper endpoint of the standard log-link solver domain.
pub(crate) const LOG_LINK_SOLVER_ETA_MAX: f64 = 700.0;
/// Bound B used by the bounded sinh-arcsinh log-delta parameterisation:
/// `delta = exp(g(raw_log_delta))` with `g = smooth_bound_jet(·, B)`. Exposed
/// so the outer optimizer can search raw log δ over the support of this map
/// (`smooth_bound_support`).
///
/// `log δ` is a log-scale shape coordinate with no penalty spectrum, so its
/// domain is the one `gam_problem::precision_box` gives every such coordinate:
/// `ln(1/√ε)` e-folds either side of unit scale, the point past which a
/// criterion gradient read through a scale ratio holds no digits (#2812). The
/// standardized beta-logistic link's `[ε, log δ]` pair already takes exactly
/// this domain. It is a bound on the chart's resolution, not a box chosen for
/// the search: `δ = exp(g)` is representable far past it, and unresolvable far
/// inside it.
#[inline]
pub fn sas_log_delta_domain_bound() -> f64 {
    -gam_problem::log_gradient_resolution()
}

/// The raw interval on which `smooth_bound_jet(·, bound)` still depends on its
/// argument (#2902 row 8). At `|x| = a + 2·(B − a)`, `a = SPLICE_INTERIOR_FRAC·B`,
/// the map reaches `±B` with every derivative exactly zero and stays there, so a
/// point outside the interval moves nothing a point on its edge does not. The
/// edge is formed with the same operations `smooth_bound_jet` compares against,
/// so the saturated branch holds exactly at it.
pub(crate) fn smooth_bound_support(bound: f64) -> (f64, f64) {
    let interior = SPLICE_INTERIOR_FRAC * bound;
    let edge = interior + 2.0 * (bound - interior);
    (-edge, edge)
}

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

/// Exact 6-jet `(mu, mu', …, mu^(5))` of the reciprocal-power inverse links
/// `mu = eta^(-a)`: `a = 1` is the inverse link `g(mu) = 1/mu` (canonical for
/// Gamma), `a = 1/2` the inverse-squared link `g(mu) = 1/mu²` (canonical for
/// the inverse Gaussian).
///
/// Both links map only `eta > 0` onto the mean space `mu > 0`; `eta <= 0` has
/// no mean at all (and `eta → 0⁺` sends `mu → ∞`). Such an `eta` is refused
/// through the same typed [`EstimationError::InverseLinkDomainViolation`] the
/// log link uses, which the PIRLS LM step search treats as an infeasible trial
/// step and damps: the inner solver stays in the domain by step-halving to
/// feasibility, never by projecting `eta`.
///
/// `d_k = c_k · eta^(-a-k)` with `c_0 = 1`, `c_{k+1} = c_k · (-a - k)`.
pub(crate) fn reciprocal_power_link_jet6(
    link: &'static str,
    exponent: f64,
    eta: f64,
) -> Result<[f64; 6], EstimationError> {
    if !(eta > 0.0 && eta.is_finite()) {
        return Err(EstimationError::InverseLinkDomainViolation {
            link,
            eta,
            lower: 0.0,
            upper: f64::MAX,
        });
    }
    let mut out = [0.0; 6];
    let mut coef = 1.0;
    for (k, slot) in out.iter_mut().enumerate() {
        *slot = coef * eta.powf(-exponent - k as f64);
        coef *= -exponent - k as f64;
    }
    Ok(out)
}

/// Canonical names of the reciprocal links, from the one link vocabulary.
pub(crate) const INVERSE_LINK_NAME: &str = StandardLink::Inverse.name();
pub(crate) const INVERSE_SQUARED_LINK_NAME: &str = StandardLink::InverseSquared.name();
pub(crate) const SQRT_LINK_NAME: &str = StandardLink::Sqrt.name();

/// The inverse-link stack `[h, h′, …, h⁽⁵⁾]` of the square-root link
/// `g(μ) = √μ`, whose inverse `μ = η²` is a bijection only on the branch
/// `η > 0`; every other `η` is refused through the typed domain error (the
/// recoverable step-rejection channel), never folded back onto the branch.
pub(crate) fn sqrt_link_jet6(eta: f64) -> Result<[f64; 6], EstimationError> {
    if !(eta > 0.0 && eta.is_finite()) {
        return Err(EstimationError::InverseLinkDomainViolation {
            link: SQRT_LINK_NAME,
            eta,
            lower: 0.0,
            upper: f64::MAX,
        });
    }
    Ok([eta * eta, 2.0 * eta, 2.0, 0.0, 0.0, 0.0])
}

/// The inverse-link stack `[h, h′, …, h⁽⁵⁾]` at `η` for the links of the
/// power/log ladder (identity, log, sqrt, `1/μ`, `1/μ²`), `None` for the
/// probability links. Each link's own domain is enforced through
/// [`EstimationError::InverseLinkDomainViolation`].
pub(crate) fn standard_ladder_link_jet6(
    link: StandardLink,
    eta: f64,
) -> Option<Result<[f64; 6], EstimationError>> {
    match link {
        StandardLink::Identity => Some(
            finite_inverse_link_eta(StandardLink::Identity.name(), eta)
                .map(|eta| [eta, 1.0, 0.0, 0.0, 0.0, 0.0]),
        ),
        StandardLink::Log => Some(log_link_solver_exp(eta).map(|e| [e; 6])),
        StandardLink::Sqrt => Some(sqrt_link_jet6(eta)),
        StandardLink::Inverse | StandardLink::InverseSquared => {
            standard_reciprocal_power_jet6(link, eta)
        }
        StandardLink::Logit
        | StandardLink::Probit
        | StandardLink::CLogLog
        | StandardLink::LogLog
        | StandardLink::Cauchit => None,
    }
}

/// [`reciprocal_power_link_jet6`] for the standard link, `None` for every link
/// that is not a reciprocal power.
pub(crate) fn standard_reciprocal_power_jet6(
    link: StandardLink,
    eta: f64,
) -> Option<Result<[f64; 6], EstimationError>> {
    match link {
        StandardLink::Inverse => Some(reciprocal_power_link_jet6(INVERSE_LINK_NAME, 1.0, eta)),
        StandardLink::InverseSquared => Some(reciprocal_power_link_jet6(
            INVERSE_SQUARED_LINK_NAME,
            0.5,
            eta,
        )),
        _ => None,
    }
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
struct AsinhJet6 {
    value: f64,
    d1: f64,
    d2: f64,
    d3: f64,
    d4: f64,
    d5: f64,
    d6: f64,
}

/// Exact eta derivatives of `asinh(eta)`, factored through `hypot` so powers
/// of a large finite eta never form `inf * 0` in the derivative tails.
#[inline]
fn asinh_jet6(eta: f64) -> AsinhJet6 {
    let q = eta.hypot(1.0);
    let inv_q = q.recip();
    let inv_q2 = inv_q * inv_q;
    let inv_q3 = inv_q2 * inv_q;
    let inv_q4 = inv_q2 * inv_q2;
    let inv_q5 = inv_q4 * inv_q;
    let inv_q6 = inv_q3 * inv_q3;
    let t = eta / q;
    let t2 = t * t;
    let t4 = t2 * t2;
    // `f64::asinh` computes `ln(x + sqrt(x*x + 1))`, whose `x*x` overflows to
    // `inf` near `±f64::MAX` even though `asinh(±f64::MAX) ≈ ±710.48` is finite
    // and well inside range. An `inf` value here would poison the far-tail SAS
    // jet with `0·∞` once the bounded map saturates (`g^(k)=0 · inf`), so fall
    // back to the overflow-free asymptotic `sign(x)·(ln|x| + ln 2)` — exact to
    // full f64 precision wherever `x*x` overflows — when the library result is
    // non-finite. In the finite region this is bit-identical to `asinh`.
    let value = {
        let v = eta.asinh();
        if v.is_finite() {
            v
        } else {
            eta.signum() * (eta.abs().ln() + std::f64::consts::LN_2)
        }
    };
    AsinhJet6 {
        value,
        d1: inv_q,
        d2: -t * inv_q2,
        d3: (2.0 * t2 - inv_q2) * inv_q3,
        d4: t * (9.0 * inv_q2 - 6.0 * t2) * inv_q4,
        d5: (9.0 * inv_q4 - 72.0 * t2 * inv_q2 + 24.0 * t4) * inv_q5,
        // -15 eta (8 eta^4 - 40 eta^2 + 15) / (1 + eta^2)^(11/2).
        d6: -15.0 * t * (8.0 * t4 - 40.0 * t2 * inv_q2 + 15.0 * inv_q4) * inv_q6,
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
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
fn cauchit_mean(eta: f64) -> f64 {
    // Keep the same arithmetic as the public vector inverse link. Reciprocal
    // reflection preserves the small CDF that 1/2 + atan(eta)/pi cancels away.
    if eta < -1.0 {
        (-eta.recip()).atan() / std::f64::consts::PI
    } else if eta > 1.0 {
        1.0 - eta.recip().atan() / std::f64::consts::PI
    } else {
        0.5 + eta.atan() / std::f64::consts::PI
    }
}

#[inline]
fn cauchit_rational_factors(eta: f64) -> (f64, f64) {
    // q = 1/(1+eta^2), r = eta/(1+eta^2). Both are bounded, including
    // at infinite eta, so derivatives need no overflowing denominator powers.
    if eta.abs() > 1.0 {
        let inv = eta.recip();
        let inv2 = inv * inv;
        let den = 1.0 + inv2;
        (inv2 / den, inv / den)
    } else {
        let q = (1.0 + eta * eta).recip();
        (q, eta * q)
    }
}

#[inline]
fn cauchit_inverse_link_jet(eta: f64) -> InverseLinkJet {
    let (q, r) = cauchit_rational_factors(eta);
    let d1 = q / std::f64::consts::PI;
    InverseLinkJet {
        mu: cauchit_mean(eta),
        d1,
        d2: (-2.0 * r) * d1,
        d3: (6.0 * r * r - 2.0 * q * q) * d1,
    }
}

#[inline]
fn cauchit_inverse_link_d4(eta: f64) -> f64 {
    let (q, r) = cauchit_rational_factors(eta);
    // 24 eta (1-eta^2) / [pi (1+eta^2)^4]. Apply the coefficient
    // before the final small product to preserve representable subnormals.
    canonicalzero((24.0 * r * (q * q - r * r)) * (q / std::f64::consts::PI))
}

#[inline]
fn cauchit_inverse_link_d5(eta: f64) -> f64 {
    let (q, r) = cauchit_rational_factors(eta);
    let q2 = q * q;
    let r2 = r * r;
    // 24(1-10eta^2+5eta^4) / [pi (1+eta^2)^5].
    canonicalzero((24.0 * q2 * q2 - 240.0 * q2 * r2 + 120.0 * r2 * r2)
        * (q / std::f64::consts::PI))
}

#[inline]
fn cauchit_inverse_link_d6(eta: f64) -> f64 {
    let (q, r) = cauchit_rational_factors(eta);
    let q2 = q * q;
    let r2 = r * r;
    // -240 eta (3 - 10eta^2 + 3eta^4) / [pi (1+eta^2)^6].
    canonicalzero((-240.0 * r * (3.0 * q2 * q2 - 10.0 * q2 * r2 + 3.0 * r2 * r2))
        * (q / std::f64::consts::PI))
}

/// Sixth derivative of the logistic CDF, the order after
/// [`logit_inverse_link_jet5`]'s `d5`, in the same overflow-free `z = e^{-|eta|}`
/// form: `z(1 - 57z + 302z² - 302z³ + 57z⁴ - z⁵)/(1 + z)⁷` for `eta < 0` and its
/// odd reflection for `eta ≥ 0`.
#[inline]
pub(crate) fn logit_inverse_link_d6(eta: f64) -> f64 {
    if eta.is_nan() {
        return f64::NAN;
    }
    if !eta.is_finite() {
        return 0.0;
    }
    let z = (-eta.abs()).exp();
    let opz = 1.0 + z;
    let opz2 = opz * opz;
    let opz7 = opz2 * opz2 * opz2 * opz;
    let polynomial = 1.0 + z * (-57.0 + z * (302.0 + z * (-302.0 + z * (57.0 - z))));
    let value = z * polynomial / opz7;
    canonicalzero(if eta >= 0.0 { -value } else { value })
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

/// Multiply a degree-at-most-five Hermite factor by the normal density.
/// A subnormal density can regain representable digits after multiplication;
/// include the factor before exponentiation in that tail.
#[inline]
fn probit_density_product(x: f64, density: f64, polynomial: f64) -> f64 {
    if density >= f64::MIN_POSITIVE {
        return polynomial * density;
    }
    if x.abs() > 40.0 {
        // Even the fifth-degree factor times phi(40), about 1.5e-340, is below
        // half the smallest subnormal. Larger finite arguments also round to zero.
        return 0.0;
    }
    if polynomial == 0.0 {
        return 0.0;
    }
    polynomial.signum()
        * (polynomial.abs().ln() - 0.5 * x * x - 0.918_938_533_204_672_7).exp()
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
        d2: probit_density_product(x, phi, -x),
        d3: probit_density_product(x, phi, x * x - 1.0),
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
    canonicalzero(probit_density_product(x, phi, -(x * x * x - 3.0 * x)))
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
    canonicalzero(probit_density_product(x, phi, x * x * x * x - 6.0 * x * x + 3.0))
}

#[inline]
fn probit_pdffifth_derivative(eta: f64) -> f64 {
    // mu'''''' = Phi^{(6)}(eta) = -(eta^5 - 10*eta^3 + 15*eta) * phi(eta).
    if eta.is_nan() {
        return f64::NAN;
    }
    if !eta.is_finite() {
        return 0.0;
    }
    let x = eta;
    let x2 = x * x;
    let phi = normal_pdf(x);
    canonicalzero(probit_density_product(x, phi, -x * (x2 * x2 - 10.0 * x2 + 15.0)))
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
/// Noncanonical Bernoulli links use normalized Taylor series with unit constant
/// terms and a separately evaluated log weight. Direct complementary tails
/// retain variance information after the reported mean rounds to an endpoint;
/// each derivative is rescaled separately, including when W itself underflows.
pub fn fisher_weight_jet5(link: StandardLink, eta: f64) -> (f64, f64, f64, f64, f64) {
    match link {
        StandardLink::Logit => {
            let jet = logit_inverse_link_jet5(eta);
            (jet.d1, jet.d2, jet.d3, jet.d4, jet.d5)
        }
        StandardLink::Probit => probit_fisher_weight_jet5(eta),
        StandardLink::CLogLog => component_fisher_weight_jet5(LinkComponent::CLogLog, eta),
        StandardLink::LogLog => component_fisher_weight_jet5(LinkComponent::LogLog, eta),
        StandardLink::Cauchit => component_fisher_weight_jet5(LinkComponent::Cauchit, eta),
        StandardLink::Identity
        | StandardLink::Log
        | StandardLink::Sqrt
        | StandardLink::Inverse
        | StandardLink::InverseSquared => (0.0, 0.0, 0.0, 0.0, 0.0),
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
    match component {
        LinkComponent::Logit => {
            let jet = logit_inverse_link_jet5(eta);
            (jet.d1, jet.d2, jet.d3, jet.d4, jet.d5)
        }
        LinkComponent::Probit => probit_fisher_weight_jet5(eta),
        LinkComponent::CLogLog => cloglog_fisher_weight_jet5(eta),
        LinkComponent::LogLog => {
            let (w, d1, d2, d3, d4) = cloglog_fisher_weight_jet5(-eta);
            (w, -d1, d2, -d3, d4)
        }
        LinkComponent::Cauchit => {
            if eta.is_nan() {
                return (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN);
            }
            if !eta.is_finite() {
                return (0.0, 0.0, 0.0, 0.0, 0.0);
            }
            let p = cauchit_mean(eta);
            let q = cauchit_mean(-eta);
            let (a, b) = cauchit_rational_factors(eta);
            let a2 = a * a;
            let b2 = b * b;
            let density_ratios = [
                1.0,
                -2.0 * b,
                6.0 * b2 - 2.0 * a2,
                24.0 * b * (a2 - b2),
                24.0 * a2 * a2 - 240.0 * a2 * b2 + 120.0 * b2 * b2,
            ];
            let log_density = -std::f64::consts::PI.ln() - 2.0 * eta.hypot(1.0).ln();
            normalized_fisher_weight_jet5(
                2.0 * log_density - p.ln() - q.ln(),
                density_ratios,
                (log_density - p.ln()).exp(),
                (log_density - q.ln()).exp(),
            )
        }
    }
}

/// Form the Taylor quotient after dividing each of f=mu', p=mu and q=1-mu
/// by its own value. The ratios stay representable when f² or p*q do not.
fn normalized_fisher_weight_jet5(
    log_weight: f64,
    density_ratios: [f64; 5],
    density_over_p: f64,
    density_over_q: f64,
) -> (f64, f64, f64, f64, f64) {
    let factorial = [1.0_f64, 1.0, 2.0, 6.0, 24.0];
    let mut f = [1.0; 5];
    let mut p = [1.0; 5];
    let mut q = [1.0; 5];
    for k in 1..5 {
        f[k] = density_ratios[k] / factorial[k];
        p[k] = density_over_p * density_ratios[k - 1] / factorial[k];
        q[k] = -density_over_q * density_ratios[k - 1] / factorial[k];
    }
    let normalized = taylor5_mul(
        &taylor5_mul(&f, &f),
        &taylor5_inv(&taylor5_mul(&p, &q)),
    );
    let mut derivatives = [0.0; 5];
    for k in 0..5 {
        let multiplier = normalized[k] * factorial[k];
        if multiplier != 0.0 {
            derivatives[k] = multiplier.signum() * (log_weight + multiplier.abs().ln()).exp();
        }
    }
    (derivatives[0], derivatives[1], derivatives[2], derivatives[3], derivatives[4])
}

fn cloglog_fisher_weight_jet5(eta: f64) -> (f64, f64, f64, f64, f64) {
    if eta.is_nan() {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }
    let a = eta.exp();
    if (-a).exp() == 0.0 {
        // W=a² sum_{m>=1} exp(-m*a). Once exp(-a) underflows, every
        // m>=2 contribution to orders 0..4 is also below representability.
        // Evaluate the m=1 polynomials in log scale: a derivative can still
        // be representable when both exp(-a) and W have rounded to zero.
        let eval = |coefficients: &[f64]| stable_nonnegative_poly_times_exp_neg(a, coefficients);
        return (
            eval(&[0.0, 0.0, 1.0]),
            eval(&[0.0, 0.0, 2.0, -1.0]),
            eval(&[0.0, 0.0, 4.0, -5.0, 1.0]),
            eval(&[0.0, 0.0, 8.0, -19.0, 9.0, -1.0]),
            eval(&[0.0, 0.0, 16.0, -65.0, 55.0, -14.0, 1.0]),
        );
    }
    let a2 = a * a;
    let density_ratios = [
        1.0,
        1.0 - a,
        1.0 - 3.0 * a + a2,
        1.0 - 7.0 * a + 6.0 * a2 - a2 * a,
        1.0 - 15.0 * a + 25.0 * a2 - 10.0 * a2 * a + a2 * a2,
    ];
    // W=a/exprel(a), f/p=1/exprel(a), f/q=a. These expressions also
    // retain the exact small-a limit without dividing two subnormal tails.
    let log_exprel = gam_math::special::log_exprel(a);
    normalized_fisher_weight_jet5(
        eta - log_exprel,
        density_ratios,
        (-log_exprel).exp(),
        a,
    )
}

#[cfg(test)]
mod fisher_tail_tests {
    use super::*;

    fn entries(jet: (f64, f64, f64, f64, f64)) -> [f64; 5] {
        [jet.0, jet.1, jet.2, jet.3, jet.4]
    }

    #[test]
    fn cauchit_fisher_derivatives_survive_rounded_endpoint_means() {
        // W(x)~1/(pi*x³). At these x the relative correction is below one ulp,
        // so derivatives have coefficients [1,-3,12,-60,360]/(pi*x^(3+k)).
        let coefficients = [1.0, -3.0, 12.0, -60.0, 360.0];
        for x in [1e20_f64, 1e100] {
            let positive = entries(component_fisher_weight_jet5(LinkComponent::Cauchit, x));
            let negative = entries(component_fisher_weight_jet5(LinkComponent::Cauchit, -x));
            for k in 0..5 {
                let expected = coefficients[k] / std::f64::consts::PI * x.recip().powi(k as i32 + 3);
                if expected == 0.0 {
                    assert_eq!(positive[k], 0.0);
                    assert_eq!(negative[k], 0.0);
                } else {
                    assert!((positive[k] / expected - 1.0).abs() < 3e-12);
                    let reflected = if k % 2 == 0 { expected } else { -expected };
                    assert!((negative[k] / reflected - 1.0).abs() < 3e-12);
                }
            }
        }
    }

    #[test]
    fn cloglog_fisher_keeps_derivatives_after_the_weight_underflows() {
        for a in [750.0_f64, 780.0] {
            let eta = a.ln();
            let a = eta.exp();
            let jet = entries(component_fisher_weight_jet5(LinkComponent::CLogLog, eta));
            let reflected = entries(component_fisher_weight_jet5(LinkComponent::LogLog, -eta));
            let expected_weight = (2.0 * a.ln() - a).exp();
            let expected_fourth = (6.0 * a.ln() - a
                + (-14.0 / a + 55.0 / a.powi(2) - 65.0 / a.powi(3)
                    + 16.0 / a.powi(4)).ln_1p()).exp();
            let ulp = f64::from_bits(1);
            assert!((jet[0] - expected_weight).abs() <= 2.0 * ulp);
            assert!((jet[4] - expected_fourth).abs() <= expected_fourth * 1e-12 + 2.0 * ulp);
            assert!(jet[4] > 0.0);
            if a > 770.0 {
                assert_eq!(jet[0], 0.0);
            }
            for k in 0..5 {
                assert_eq!(reflected[k], if k % 2 == 0 { jet[k] } else { -jet[k] });
            }
        }
    }

    #[test]
    fn probit_fisher_does_not_square_a_tiny_density_before_division() {
        let x = 30.0;
        let density = normal_pdf(x);
        let expected = density * (density / normal_cdf(-x)) / normal_cdf(x);
        let actual = probit_fisher_weight_jet5(x).0;
        assert!(actual > 0.0);
        assert!((actual / expected - 1.0).abs() < 1e-12);
        let jet = probit_fisher_weight_jet5(39.0);
        assert_eq!(jet.0, 0.0);
        assert!(jet.4 > 0.0, "the fourth derivative remains representable at eta=39");
    }

    #[test]
    fn probit_fisher_tail_jets_match_high_precision_reference_values() {
        // Independently differentiated at 100 decimal digits on MSI, using
        // the exact binary64 arguments and direct erfc probabilities.
        let cases: [(f64, [f64; 5]); 3] = [
            (30.0, [
                4.425_839_702_671_741e-195,
                -1.326_279_891_235_266e-193,
                3.969_997_786_076_700_5e-192,
                -1.187_023_436_924_559e-190,
                3.545_199_141_911_902_5e-189,
            ]),
            (38.6, [
                4.436_957_158_657_313e-323,
                -1.711_517_530_278_145_3e-321,
                6.597_589_692_438_285e-320,
                -2.541_537_363_017_449_5e-318,
                9.783_952_718_564_348e-317,
            ]),
            (39.0, [0.0, 0.0, 0.0, 0.0, 1.873_718_168_939_258_4e-323]),
        ];
        for (eta, expected) in cases {
            let actual = entries(probit_fisher_weight_jet5(eta));
            for k in 0..5 {
                let tolerance = expected[k].abs() * 3e-12 + 2.0 * f64::from_bits(1);
                assert!(
                    (actual[k] - expected[k]).abs() <= tolerance,
                    "eta={eta}, derivative={k}: {} vs {}",
                    actual[k], expected[k],
                );
            }
        }
    }

    #[test]
    fn noncanonical_fisher_jets_have_finite_limits_at_huge_finite_eta() {
        for component in [LinkComponent::Cauchit, LinkComponent::CLogLog, LinkComponent::LogLog, LinkComponent::Probit] {
            for eta in [-f64::MAX, f64::MAX] {
                assert!(entries(component_fisher_weight_jet5(component, eta)).iter().all(|&v| v == 0.0));
            }
        }
    }
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
    let x2 = x * x;
    let x4 = x2 * x2;
    // Protect the normalized Hermite arithmetic only at |x|>1e76, where
    // exp(-x²/2) times every polynomial through the required order rounds zero.
    if x4 > f64::MAX / 256.0 {
        return (0.0, 0.0, 0.0, 0.0, 0.0);
    }
    let (log_p, density_over_p) =
        gam_math::probability::signed_probit_logcdf_and_mills_ratio(x);
    let (log_q, density_over_q) =
        gam_math::probability::signed_probit_logcdf_and_mills_ratio(-x);
    let log_density = -0.5 * x2 - 0.5 * (2.0 * std::f64::consts::PI).ln();
    // Pair the density with the SMALL-tail Mills ratio, avoiding cancellation
    // between two nearly equal negative log-density terms.
    let log_weight = if x >= 0.0 {
        log_density + density_over_q.ln() - log_p
    } else {
        log_density + density_over_p.ln() - log_q
    };
    normalized_fisher_weight_jet5(
        log_weight,
        [1.0, -x, x2 - 1.0, -x * (x2 - 3.0), x4 - 6.0 * x2 + 3.0],
        density_over_p,
        density_over_q,
    )
}

/// η-derivatives of the two Bernoulli log-probabilities (#3317):
/// `log_mu[k] = ∂^{k+1} log μ(η)/∂η^{k+1}` and `log_complement[k]` the same for
/// `log(1 − μ(η))`, for `k = 0..5`.
///
/// A Bernoulli row's log-likelihood is `y·log μ + (1−y)·log(1−μ)`, so its
/// observed information and every η-derivative of it are linear in these two
/// jets. Nothing here divides by the variance `μ(1−μ)`, which the ratio tower
/// `μ'/(φμ(1−μ))` must: each side is normalized by its OWN probability. The
/// series `μ(η+t)/μ(η)` has coefficients `(μ'/μ)·(μ^(m)/μ')/m!` and
/// `(1−μ)(η+t)/(1−μ)(η)` has `−(μ'/(1−μ))·(μ^(m)/μ')/m!`, the reverse hazard
/// and the hazard times the density ratios; the jet is the Taylor logarithm of
/// each. Both stay representable where `μ'` and `1−μ` underflow together and
/// the tower is `0/0` — cloglog at `η = 6.65`, whose `y = 1` row has the exact,
/// representable observed information `≈ 0` that the tower reported as NaN.
///
/// Accuracy is relative wherever a side's probability is not small, and on
/// the closed forms (logit, and the cloglog/loglog side `−e^{±η}`). Where a
/// side's own probability is small and it has no closed form, its higher
/// derivatives are small differences of terms of size `ρ^m`, `ρ` the side's
/// first derivative (`|η|` for probit, `1` for the cloglog probability at
/// `η ≪ 0`), so the error there is absolute, `~ε·ρ^m`: `5e-8` on the fifth
/// derivative of `log Φ(−30)`, whose second is `−0.9989`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BernoulliLogJet {
    pub(crate) log_mu: [f64; 6],
    pub(crate) log_complement: [f64; 6],
}

pub(crate) fn bernoulli_log_jet6_for_inverse_link(
    link: &InverseLink,
    eta: f64,
) -> Result<BernoulliLogJet, EstimationError> {
    if eta.is_nan() {
        return Ok(BernoulliLogJet {
            log_mu: [f64::NAN; 6],
            log_complement: [f64::NAN; 6],
        });
    }
    match link {
        InverseLink::Standard(StandardLink::Logit) => {
            Ok(symmetric_bernoulli_log_jet6(SymmetricBernoulliCdf::Logistic, eta))
        }
        InverseLink::Standard(StandardLink::Probit) => {
            Ok(symmetric_bernoulli_log_jet6(SymmetricBernoulliCdf::Normal, eta))
        }
        InverseLink::Standard(StandardLink::Cauchit) => {
            Ok(symmetric_bernoulli_log_jet6(SymmetricBernoulliCdf::Cauchy, eta))
        }
        InverseLink::Standard(StandardLink::CLogLog) => Ok(BernoulliLogJet {
            log_mu: cloglog_log_mean_jet6(eta),
            // log(1 − μ) = −e^η: every derivative is −e^η.
            log_complement: [-eta.exp(); 6],
        }),
        // loglog μ(η) = 1 − cloglog μ(−η): the two sides trade places.
        InverseLink::Standard(StandardLink::LogLog) => Ok(BernoulliLogJet {
            log_mu: reflect_log_jet6([-(-eta).exp(); 6]),
            log_complement: reflect_log_jet6(cloglog_log_mean_jet6(-eta)),
        }),
        _ => bernoulli_log_jet6_from_inverse_link_jet(link, eta),
    }
}

/// The link-generic construction behind [`bernoulli_log_jet6_for_inverse_link`],
/// from the inverse-link jet and its fourth through sixth derivatives,
/// normalizing each side by its own probability.
pub(crate) fn bernoulli_log_jet6_from_inverse_link_jet(
    link: &InverseLink,
    eta: f64,
) -> Result<BernoulliLogJet, EstimationError> {
    let jet = link.jet(eta)?;
    let derivatives = [
        jet.d1,
        jet.d2,
        jet.d3,
        inverse_link_pdfthird_derivative_for_inverse_link(link, eta)?,
        inverse_link_pdffourth_derivative_for_inverse_link(link, eta)?,
        inverse_link_pdffifth_derivative_for_inverse_link(link, eta)?,
    ];
    let complement = inverse_link_complement_for_inverse_link(link, eta, jet.mu);
    let factorial = SERIES_FACTORIALS6;
    let mut mean_series = [0.0; 6];
    let mut complement_series = [0.0; 6];
    for m in 0..6 {
        mean_series[m] = derivatives[m] / (factorial[m] * jet.mu);
        complement_series[m] = -derivatives[m] / (factorial[m] * complement);
    }
    Ok(BernoulliLogJet {
        log_mu: log_of_normalized_series6(mean_series),
        log_complement: log_of_normalized_series6(complement_series),
    })
}

/// `f(−η)`'s derivative jet from `f`'s jet evaluated at `−η`:
/// order `m` picks up `(−1)^m`.
#[inline]
fn reflect_log_jet6(jet: [f64; 6]) -> [f64; 6] {
    [-jet[0], jet[1], -jet[2], jet[3], -jet[4], jet[5]]
}

/// `m!` for `m = 1..=6`, the Taylor normalizers of the six-term log-jet series.
const SERIES_FACTORIALS6: [f64; 6] = [1.0, 2.0, 6.0, 24.0, 120.0, 720.0];

/// Derivatives of `log P(t)` at `t = 0` for `P(t) = 1 + Σ_{m=1}^{6} p_m t^m`,
/// given `p_m` as `series[m-1]`. From `P' = (log P)'·P`, the Taylor coefficients
/// `l_m` of `log P` obey `l_m = p_m − (1/m) Σ_{j=1}^{m-1} j l_j p_{m-j}`.
#[inline]
fn log_of_normalized_series6(series: [f64; 6]) -> [f64; 6] {
    let factorial = SERIES_FACTORIALS6;
    let mut log_coefficients = [0.0_f64; 6];
    for m in 1..=6 {
        let mut convolution = 0.0;
        for j in 1..m {
            let p = series[m - j - 1];
            if p != 0.0 {
                convolution += j as f64 * log_coefficients[j - 1] * p;
            }
        }
        log_coefficients[m - 1] = series[m - 1] - convolution / m as f64;
    }
    let mut derivatives = [0.0; 6];
    for m in 0..6 {
        derivatives[m] = canonicalzero(log_coefficients[m] * factorial[m]);
    }
    derivatives
}

/// `p_m = ratio·density_ratios[m-1]/m!`; a zero ratio is an exactly zero series
/// whatever the polynomials' size.
#[inline]
fn density_ratio_series6(ratio: f64, density_ratios: [f64; 6]) -> [f64; 6] {
    let factorial = SERIES_FACTORIALS6;
    let mut series = [0.0; 6];
    if ratio != 0.0 {
        for m in 0..6 {
            series[m] = ratio * density_ratios[m] / factorial[m];
        }
    }
    series
}

/// The inverse-link CDFs symmetric about zero, `1 − F(x) = F(−x)`.
#[derive(Clone, Copy)]
enum SymmetricBernoulliCdf {
    Logistic,
    Normal,
    Cauchy,
}

/// Both sides of a symmetric link: `1 − F(η) = F(−η)`, so the complement's
/// jet is the mean's evaluated at `−η` and reflected.
fn symmetric_bernoulli_log_jet6(cdf: SymmetricBernoulliCdf, eta: f64) -> BernoulliLogJet {
    BernoulliLogJet {
        log_mu: symmetric_log_cdf_jet6(cdf, eta),
        log_complement: reflect_log_jet6(symmetric_log_cdf_jet6(cdf, -eta)),
    }
}

/// Derivatives of `log F(x)` for a symmetric Bernoulli CDF `F`.
fn symmetric_log_cdf_jet6(cdf: SymmetricBernoulliCdf, x: f64) -> [f64; 6] {
    match cdf {
        SymmetricBernoulliCdf::Logistic => {
            // (log F)' = 1 − F and (1 − F)' = −F', so order m ≥ 2 is −F^(m−1).
            let jet = logit_inverse_link_jet5(x);
            [
                logit_inverse_link_jet5(-x).mu,
                -jet.d1,
                -jet.d2,
                -jet.d3,
                -jet.d4,
                -jet.d5,
            ]
        }
        SymmetricBernoulliCdf::Normal => {
            let x2 = x * x;
            let density_ratios = [
                1.0,
                -x,
                x2 - 1.0,
                -x * (x2 - 3.0),
                x2 * x2 - 6.0 * x2 + 3.0,
                -x * (x2 * x2 - 10.0 * x2 + 15.0),
            ];
            let (log_p, ratio) = gam_math::probability::signed_probit_logcdf_and_mills_ratio(x);
            if ratio >= f64::MIN_POSITIVE {
                return log_of_normalized_series6(density_ratio_series6(ratio, density_ratios));
            }
            // φ(x) is subnormal or zero, which happens only at x ≫ 0 where
            // log Φ(x) is an exact −Φ(−x): form each coefficient φ·dr/(Φ·m!)
            // in log scale so it survives the underflow of φ alone. A Hermite
            // polynomial overflows only where e^{−x²/2} times it is zero.
            let log_ratio = -0.5 * x2 - 0.5 * (2.0 * std::f64::consts::PI).ln() - log_p;
            if log_ratio == f64::NEG_INFINITY || density_ratios.iter().any(|v| !v.is_finite()) {
                return [0.0; 6];
            }
            let factorial = SERIES_FACTORIALS6;
            let mut series = [0.0; 6];
            for m in 0..6 {
                let ratio_poly = density_ratios[m];
                if ratio_poly != 0.0 {
                    series[m] =
                        ratio_poly.signum() * (log_ratio + ratio_poly.abs().ln()).exp() / factorial[m];
                }
            }
            log_of_normalized_series6(series)
        }
        SymmetricBernoulliCdf::Cauchy => {
            let (a, b) = cauchit_rational_factors(x);
            let a2 = a * a;
            let b2 = b * b;
            let density_ratios = [
                1.0,
                -2.0 * b,
                6.0 * b2 - 2.0 * a2,
                24.0 * b * (a2 - b2),
                24.0 * a2 * a2 - 240.0 * a2 * b2 + 120.0 * b2 * b2,
                -240.0 * b * (3.0 * a2 * a2 - 10.0 * a2 * b2 + 3.0 * b2 * b2),
            ];
            let log_density = -std::f64::consts::PI.ln() - 2.0 * x.hypot(1.0).ln();
            let ratio = (log_density - cauchit_mean(x).ln()).exp();
            log_of_normalized_series6(density_ratio_series6(ratio, density_ratios))
        }
    }
}

/// Derivatives of `log μ(η)` for `μ = 1 − exp(−e^η)`.
fn cloglog_log_mean_jet6(eta: f64) -> [f64; 6] {
    let u = eta.exp();
    let survival = (-u).exp();
    let series = if 1.0 - survival == 1.0 {
        // μ rounds to 1, so dividing by it is exact to rounding and each
        // coefficient is μ^(m)/m! = u·dr_{m−1}(u)·e^{−u}/m!, evaluated without
        // underflowing e^{−u} or overflowing the polynomial.
        let eval = |coefficients: &[f64]| stable_nonnegative_poly_times_exp_neg(u, coefficients);
        [
            eval(&[0.0, 1.0]),
            eval(&[0.0, 1.0, -1.0]) / 2.0,
            eval(&[0.0, 1.0, -3.0, 1.0]) / 6.0,
            eval(&[0.0, 1.0, -7.0, 6.0, -1.0]) / 24.0,
            eval(&[0.0, 1.0, -15.0, 25.0, -10.0, 1.0]) / 120.0,
            eval(&[0.0, 1.0, -31.0, 90.0, -65.0, 15.0, -1.0]) / 720.0,
        ]
    } else {
        let u2 = u * u;
        // μ'/μ = 1/exprel(u), with the exact u → 0 limit.
        density_ratio_series6(
            (-gam_math::special::log_exprel(u)).exp(),
            [
                1.0,
                1.0 - u,
                1.0 - 3.0 * u + u2,
                1.0 - 7.0 * u + 6.0 * u2 - u2 * u,
                1.0 - 15.0 * u + 25.0 * u2 - 10.0 * u2 * u + u2 * u2,
                1.0 - 31.0 * u + 90.0 * u2 - 65.0 * u2 * u + 15.0 * u2 * u2 - u2 * u2 * u,
            ],
        )
    };
    log_of_normalized_series6(series)
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
        LinkComponent::Cauchit => cauchit_inverse_link_d4(eta),
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
        LinkComponent::Cauchit => cauchit_inverse_link_d5(eta),
    }
}

/// Sixth derivative of a component inverse-link CDF (= fifth derivative of PDF).
/// Extends `component_inverse_link_pdffourth_derivative` by one derivative order.
#[inline]
fn component_inverse_link_pdffifth_derivative(component: LinkComponent, eta: f64) -> f64 {
    match component {
        LinkComponent::Probit => probit_pdffifth_derivative(eta),
        LinkComponent::Logit => logit_inverse_link_d6(eta),
        LinkComponent::CLogLog => {
            // d6 = exp(-t) * (t - 31t^2 + 90t^3 - 65t^4 + 15t^5 - t^6), t = exp(eta).
            if eta.is_nan() {
                return f64::NAN;
            }
            if !eta.is_finite() {
                return 0.0;
            }
            let t = eta.exp();
            canonicalzero(stable_nonnegative_poly_times_exp_neg(
                t,
                &[0.0, 1.0, -31.0, 90.0, -65.0, 15.0, -1.0],
            ))
        }
        LinkComponent::LogLog => {
            // mu(eta) = 1 - cloglog mu(-eta), so the even order flips sign:
            // d6 = -exp(-r) * (r - 31r^2 + 90r^3 - 65r^4 + 15r^5 - r^6),
            // r = exp(-eta).
            if eta.is_nan() {
                return f64::NAN;
            }
            if !eta.is_finite() {
                return 0.0;
            }
            let r = (-eta).exp();
            canonicalzero(stable_nonnegative_poly_times_exp_neg(
                r,
                &[0.0, -1.0, 31.0, -90.0, 65.0, -15.0, 1.0],
            ))
        }
        LinkComponent::Cauchit => cauchit_inverse_link_d6(eta),
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
    /// Exact symmetric Hessian of `d2 = d2mu/deta2` in the free-logit
    /// coordinates. Required by the outer link-parameter Hessian: the observed
    /// working weight `W_obs` depends on `d2`, so its second parameter
    /// derivative cannot be formed without this block (#2665).
    pub d2d2_drho2: Array2<f64>,
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
    /// Exact symmetric Hessian of `d2 = d2mu/deta2` in the same parameter
    /// order. Required by the outer link-parameter Hessian: the observed
    /// working weight `W_obs` depends on `d2`, so its second parameter
    /// derivative cannot be formed without this block (#2665).
    pub d2d2_dparams2: Array2<f64>,
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
pub(crate) struct ProbitLinkKernel;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct LogitLinkKernel;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CLogLogLinkKernel;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct LogLogLinkKernel;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CauchitLinkKernel;

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

/// Interior half-width fraction of the bounded latent map. The map is the EXACT
/// identity on `|x| <= SPLICE_INTERIOR_FRAC * B`; the compact-support saturation
/// splice then runs from there to `±B` at `(2 - SPLICE_INTERIOR_FRAC) * B`.
const SPLICE_INTERIOR_FRAC: f64 = 0.8;

/// Value and first six derivatives of the bounded latent map `g` at one point.
#[derive(Clone, Copy, Debug)]
struct SmoothBoundJet {
    g: f64,
    d1: f64,
    d2: f64,
    d3: f64,
    d4: f64,
    d5: f64,
    d6: f64,
}

/// Interior-exact bounded latent map for the sinh-arcsinh link, replacing the
/// everywhere-soft `B·tanh(x/B)`.
///
/// `tanh` distorts *every* interior point by a relative `(x/B)²/3` — at the mild
/// SAS point `η=−1, ε=0, δ=1` that is a `~1e-4` perturbation of the latent, which
/// (a) breaks the `SAS(ε=0, δ=1) ≡ probit` reduction identity by `~2e-4` in `μ`
/// and (b) leaves a spurious optimizer-visible kink where the fast probit path
/// (`|ε|<1e-12 ∧ |δ−1|<1e-12`) meets the full composition. This map removes both:
/// it is the exact identity on the whole interior, so the reduction is exact and
/// the surfaces agree bitwise across `ε=0`.
///
/// Three regions (odd in `x`; `B = bound`, `a = 0.8B`, `c = 1.2B`):
///
///   |x| ≤ a:      g(x) = x            (exact identity — every g^(k≥2) is 0)
///   a < |x| < c:  C⁶ splice of x → ±B
///   |x| ≥ c:      g(x) = ±B           (compact support — every g^(k≥1) is 0)
///
/// On the splice `g'(x) = 1 − S(w)`, `w = (|x|−a)/(c−a)`, with `S` the order-5
/// smoothstep `462w⁶−1980w⁷+3465w⁸−3080w⁹+1386w¹⁰−252w¹¹` (its first five
/// derivatives vanish at `w = 0, 1`), so `g` is C⁶ at both seams — the order the
/// SAS jet tower needs: the Bernoulli observed information's third η-derivative,
/// which the exact outer ρ-Hessian reads, is `μ⁽⁶⁾`.
/// `S` is symmetric with `∫₀¹S = ½`, so matching `g(c)=B` fixes `c = 2B − a` with
/// no free constant and keeps `g` non-expansive (`0 ≤ g′ ≤ 1`) and monotone.
///
/// Compact support (rather than an asymptotic `tanh` tail) makes the fully
/// saturated regime *exact*: `g ≡ ±B` with every derivative identically zero, so
/// a saturated row contributes exactly zero Fisher weight and, crucially, the
/// `g^(k)=0` factors annihilate the `0·∞` that an overflowing `asinh(±f64::MAX)`
/// would otherwise inject into the far-tail jet.
#[inline]
fn smooth_bound_jet(value: f64, bound: f64) -> SmoothBoundJet {
    // Every caller passes a named positive bound constant.
    let b = bound;
    let a = SPLICE_INTERIOR_FRAC * b; // interior half-width
    let l = 2.0 * (b - a); // splice width; c = a + l = (2 - frac) * b
    let ax = value.abs();
    if ax <= a {
        // Interior: exact identity. The value carries the sign of `value`.
        return SmoothBoundJet {
            g: value,
            d1: 1.0,
            d2: 0.0,
            d3: 0.0,
            d4: 0.0,
            d5: 0.0,
            d6: 0.0,
        };
    }
    let sign = if value < 0.0 { -1.0 } else { 1.0 };
    if ax >= a + l {
        // Compact-support saturation: g ≡ ±B, every derivative exactly zero.
        return SmoothBoundJet {
            g: sign * b,
            d1: 0.0,
            d2: 0.0,
            d3: 0.0,
            d4: 0.0,
            d5: 0.0,
            d6: 0.0,
        };
    }
    // Splice seam. `w ∈ (0, 1)`; `S` is the order-5 smoothstep and `s1..s5`
    // its w-derivatives (all vanish at the endpoints, giving C⁶ seams).
    let w = (ax - a) / l;
    let w2 = w * w;
    let w3 = w2 * w;
    let w4 = w3 * w;
    let w5 = w4 * w;
    let w6 = w5 * w;
    let w7 = w6 * w;
    let w8 = w7 * w;
    let w9 = w8 * w;
    let w10 = w9 * w;
    let w11 = w10 * w;
    let w12 = w11 * w;
    let s = 462.0 * w6 - 1980.0 * w7 + 3465.0 * w8 - 3080.0 * w9 + 1386.0 * w10 - 252.0 * w11;
    let s1 = 2772.0 * w5 - 13860.0 * w6 + 27720.0 * w7 - 27720.0 * w8 + 13860.0 * w9
        - 2772.0 * w10;
    let s2 = 13860.0 * w4 - 83160.0 * w5 + 194040.0 * w6 - 221760.0 * w7 + 124740.0 * w8
        - 27720.0 * w9;
    let s3 = 55440.0 * w3 - 415800.0 * w4 + 1164240.0 * w5 - 1552320.0 * w6 + 997920.0 * w7
        - 249480.0 * w8;
    let s4 = 166320.0 * w2 - 1663200.0 * w3 + 5821200.0 * w4 - 9313920.0 * w5
        + 6985440.0 * w6
        - 1995840.0 * w7;
    let s5 = 332640.0 * w - 4989600.0 * w2 + 23284800.0 * w3 - 46569600.0 * w4
        + 41912640.0 * w5
        - 13970880.0 * w6;
    // `I(w) = ∫₀ʷ (1 − S)` is the nonneg-branch value offset above `a`.
    let iw = w - 66.0 * w7 + 247.5 * w8 - 385.0 * w9 + 308.0 * w10 - 126.0 * w11 + 21.0 * w12;
    let g0 = a + l * iw;
    // `g'(x) = 1 − S(w)`; higher x-orders differentiate `−S(w)` through the `1/l`
    // chain, then odd symmetry sets the parities (value/d2/d4/d6 odd; d1/d3/d5 even).
    let l2 = l * l;
    let l4 = l2 * l2;
    SmoothBoundJet {
        g: sign * g0,
        d1: 1.0 - s,
        d2: sign * (-s1 / l),
        d3: -s2 / l2,
        d4: sign * (-s3 / (l2 * l)),
        d5: -s4 / l4,
        d6: sign * (-s5 / (l4 * l)),
    }
}

#[inline]
fn sas_effective_log_delta(raw_log_delta: f64) -> (f64, f64) {
    let sb = smooth_bound_jet(raw_log_delta, sas_log_delta_domain_bound());
    (sb.g, sb.d1)
}

#[inline]
fn sas_delta_from_raw_log_delta(raw_log_delta: f64) -> f64 {
    let (ld_eff, _) = sas_effective_log_delta(raw_log_delta);
    ld_eff.exp()
}

pub(crate) fn validate_mixturespec(spec: &MixtureLinkSpec) -> Result<(), String> {
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

pub(crate) fn softmax_last_fixedzero(rho: &Array1<f64>) -> Array1<f64> {
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
        LinkComponent::Cauchit => cauchit_inverse_link_jet(eta),
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
            LinkFunction::Sqrt => {
                let [mu, d1, d2, d3, _, _] = sqrt_link_jet6(eta)?;
                Ok(InverseLinkJet { mu, d1, d2, d3 })
            }
            LinkFunction::Inverse => {
                let [mu, d1, d2, d3, _, _] =
                    reciprocal_power_link_jet6(INVERSE_LINK_NAME, 1.0, eta)?;
                Ok(InverseLinkJet { mu, d1, d2, d3 })
            }
            LinkFunction::InverseSquared => {
                let [mu, d1, d2, d3, _, _] =
                    reciprocal_power_link_jet6(INVERSE_SQUARED_LINK_NAME, 0.5, eta)?;
                Ok(InverseLinkJet { mu, d1, d2, d3 })
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
pub(crate) struct BetaLogisticKernel {
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
            InverseLink::Standard(StandardLink::Sqrt) => LinkFunction::Sqrt.jet(eta),
            InverseLink::Standard(StandardLink::Inverse) => LinkFunction::Inverse.jet(eta),
            InverseLink::Standard(StandardLink::InverseSquared) => {
                LinkFunction::InverseSquared.jet(eta)
            }
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
/// `LOG_LINK_SOLVER_ETA_MIN` through `LOG_LINK_SOLVER_ETA_MAX`; inputs
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

/// Stable complement `1 - mu(eta)` for a Bernoulli inverse link, evaluated
/// directly from `eta` rather than as `1.0 - mu`.
///
/// The forward `mu` rounds to exactly `1.0` in f64 far inside the tail — cloglog
/// at `eta ≈ 3.62`, probit at `eta ≈ 8.29` — after which the naive `1.0 - mu` is
/// a hard zero even though the true complement is a representable quantity down
/// to `~1e-300`. Both the Bernoulli variance `mu(1-mu)` and the working residual
/// `y - mu` depend on that complement, so recovering it exactly is what lets a
/// saturating cloglog/probit row proceed instead of being refused. This mirrors
/// the tail-complement already carried on the canonical logit path.
///
/// Each link with a cancellation-free closed form for `1 - mu` uses it; links
/// without one fall back to `1.0 - mu` (unchanged behaviour). The complement is
/// clamped into `[0, 1]` only against round-off just past the boundary.
pub fn inverse_link_complement_for_inverse_link(
    link: &InverseLink,
    eta: f64,
    mu: f64,
) -> f64 {
    let raw = match link {
        InverseLink::Standard(link_fn) => standard_link_complement(*link_fn, eta, mu),
        InverseLink::Sas(state) => sas_link_complement(eta, state.epsilon, state.log_delta, mu),
        InverseLink::BetaLogistic(state) => {
            beta_logistic_link_complement(eta, state.log_delta, state.epsilon, mu)
        }
        InverseLink::Mixture(state) => mixture_link_complement(state, eta, mu),
        // The latent-cloglog mean is `1 − S(eta, σ_L)` with the lognormal-Laplace
        // survival `S(m, σ) = E[exp(−exp η)]`, `η ~ N(m, σ²)`
        // (`latent_cloglog_jet5` forms it as `−expm1(ln S)`). Its complement is
        // `S` itself, read from the same log-space survival surface, so it keeps
        // its digits where the mean rounds to one.
        InverseLink::LatentCLogLog(state) => {
            if eta.is_nan() {
                f64::NAN
            } else {
                crate::quadrature::survival_posterior_mean(
                    latent_cloglog_quadctx(),
                    eta,
                    state.latent_sd,
                )
            }
        }
    };
    if raw.is_nan() {
        raw
    } else {
        raw.clamp(0.0, 1.0)
    }
}

/// Cancellation-free `1 - mu(eta)` for the standard Bernoulli links.
#[inline]
fn standard_link_complement(link: StandardLink, eta: f64, mu: f64) -> f64 {
    match link {
        StandardLink::Probit => {
            // 1 - Phi(eta) = Phi(-eta); the reflected CDF keeps the tiny upper-tail
            // mass that `1 - Phi(eta)` cancels away.
            if eta.is_nan() {
                f64::NAN
            } else if eta == f64::INFINITY {
                0.0
            } else if eta == f64::NEG_INFINITY {
                1.0
            } else {
                normal_cdf(-eta)
            }
        }
        StandardLink::CLogLog => {
            // mu = 1 - exp(-exp(eta))  =>  1 - mu = exp(-exp(eta)).
            if eta.is_nan() {
                f64::NAN
            } else {
                let t = eta.exp();
                if !t.is_finite() { 0.0 } else { (-t).exp() }
            }
        }
        StandardLink::LogLog => {
            // mu = exp(-exp(-eta))  =>  1 - mu = -expm1(-exp(-eta)).
            if eta.is_nan() {
                f64::NAN
            } else {
                let r = (-eta).exp();
                if !r.is_finite() { 1.0 } else { -(-r).exp_m1() }
            }
        }
        StandardLink::Cauchit => cauchit_mean(-eta),
        // Relative-risk Bernoulli link: mu = exp(eta)  =>  1 - mu = -expm1(eta),
        // which keeps the complement's leading digits as eta -> 0⁻.
        StandardLink::Log => -eta.exp_m1(),
        // Logit carries its own tail complement on the canonical path; the
        // remaining links are not Bernoulli-variance links. The naive
        // complement is exact enough for these here.
        StandardLink::Logit
        | StandardLink::Identity
        | StandardLink::Sqrt
        | StandardLink::Inverse
        | StandardLink::InverseSquared => 1.0 - mu,
    }
}

/// Cancellation-free `1 - mu(eta)` for the beta-logistic inverse link.
///
/// `mu = I_u(a, b)` with `u = logistic(x)` at the standardized latent argument `x`
/// (see [`beta_logistic_standardization`]). So the exact complement is the
/// regularized incomplete beta's own reflection identity
/// `1 - I_u(a, b) = I_{1-u}(b, a)`, and `1 - u = logistic(-x)` is already carried
/// alongside `u` by [`logistic_uwith_derivatives`]. On the saturated side
/// (`use_upper_tail`) the forward map computes `mu` AS `1 - beta_reg(b, a, 1-u)`,
/// so the complement is that `beta_reg` call with no subtraction at all — the
/// tail mass the forward `1 - ...` throws away. On the other side `mu` is small
/// and `1.0 - mu` loses nothing.
#[inline]
pub(crate) fn beta_logistic_link_complement(eta: f64, log_delta: f64, epsilon: f64, mu: f64) -> f64 {
    let (a, b) = beta_logistic_shapes(log_delta, epsilon);
    let logistic = logistic_uwith_derivatives(beta_logistic_latent_argument(eta, a, b).0);
    if logistic.ln_u.is_nan() || logistic.ln_one_minus_u.is_nan() {
        return f64::NAN;
    }
    if logistic.ln_u == f64::NEG_INFINITY {
        return 1.0;
    }
    if logistic.ln_one_minus_u == f64::NEG_INFINITY {
        return 0.0;
    }
    if logistic.use_upper_tail {
        beta_reg(b, a, logistic.one_minus_u)
    } else {
        1.0 - mu
    }
}

/// Cancellation-free `1 - mu(eta)` for the mixture inverse link.
///
/// `mu = sum_i pi_i mu_i`, so
/// `1 - mu = (1 - sum_i pi_i) + sum_i pi_i (1 - mu_i)` — exact for any weight
/// vector, and each component is one of the standard bounded links whose own
/// complement is already cancellation-free. Summing the component TAILS keeps a
/// mixture whose components all saturate from returning a hard zero: every
/// `mu_i` rounds to `1.0` while every `1 - mu_i` is still representable.
#[inline]
fn mixture_link_complement(state: &MixtureLinkState, eta: f64, mu: f64) -> f64 {
    let k = state.components.len().min(state.pi.len());
    let mut weight_total = 0.0_f64;
    let mut complement = 0.0_f64;
    for i in 0..k {
        let (mu_i, _) = component_inverse_link_mu_d1(state.components[i], eta);
        if mu_i.is_nan() {
            return f64::NAN;
        }
        let component_complement =
            standard_link_complement(state.components[i].as_standard_link(), eta, mu_i);
        if component_complement.is_nan() {
            return f64::NAN;
        }
        weight_total += state.pi[i];
        complement += state.pi[i] * component_complement;
    }
    if k == 0 {
        return 1.0 - mu;
    }
    (1.0 - weight_total) + complement
}

/// Cancellation-free `1 - mu` for the SAS inverse link. `mu = Phi(z)` with
/// `z = sinh(smooth_bound(delta*asinh(eta) + epsilon, the latent domain))`, so the exact
/// complement is `Phi(-z)`, mirroring the `sas_inverse_link_mu_d1` forward map.
/// `Phi(-z)` keeps the tiny upper-tail mass that `1 - Phi(z)` cancels; it
/// underflows to `0` only once the row is genuinely fully saturated (`z` at the
/// latent domain's sinh scale), which is the correct value there. `Phi(-z)` is
/// always in `[0, 1]`, so no clamp is needed.
///
/// The latent `asinh(eta)` is taken through the overflow-free [`asinh_jet6`]
/// value — exactly as `sas_inverse_link_mu_d1` does — NOT the raw `f64::asinh`.
/// The library `asinh` forms `x·x` internally, which overflows to `±∞` for
/// `|eta| > 1.34e154` even though `asinh(±f64::MAX) ≈ ±710` is finite and well
/// inside range. With a compressing `delta < 1` the true latent `delta·asinh`
/// then stays in the map's identity interior (finite, unsaturated `mu`), so an
/// overflowing complement would saturate to `±B` and return `Phi(∓sinh B) ∈
/// {0,1}` — disagreeing with the forward `1 - mu` by up to ~0.15 and poisoning
/// any `log(1 - mu)` tail term at extreme eta (#2389).
#[inline]
pub(crate) fn sas_link_complement(eta: f64, epsilon: f64, log_delta: f64, mu: f64) -> f64 {
    let eta = match finite_inverse_link_eta("SAS inverse link complement", eta) {
        Ok(value) => value,
        Err(_) => return 1.0 - mu,
    };
    let delta = sas_delta_from_raw_log_delta(log_delta);
    // The identity parameters are exact: the bound map fixes 0 and exp(0) = 1 (#2469).
    if epsilon == 0.0 && delta == 1.0 {
        return standard_link_complement(StandardLink::Probit, eta, mu);
    }
    let u_raw = delta * asinh_jet6(eta).value + epsilon;
    let u = smooth_bound_jet(u_raw, sas_latent_domain_bound()).g;
    normal_cdf(-u.sinh())
}

/// The SAS link's latent probit argument `z` and its first `eta` derivative.
///
/// `mu = Phi(z)` with `z = sinh(smooth_bound(delta·asinh(eta) + epsilon,
/// the latent domain)`, exactly as [`sas_inverse_link_mu_d1`] evaluates it — this
/// returns the *pre-probit* pair instead of applying `Phi`.
///
/// A consumer that needs `ln mu` or `ln(1 - mu)` must have this pair, not `mu`:
/// past `z ≈ -38` the mean underflows to exactly `0.0` and past `z ≈ +8.3` the
/// complement does, so `mu.ln()` / `(1 - mu).ln()` are `-inf` on a row whose
/// true log-probability is a perfectly ordinary finite number (`ln Phi(-59.45)
/// = -1774.6`). Evaluating `ln Phi(±z)` from `z` keeps the whole saturating
/// band representable, which is what the standard probit link already does
/// through `signed_probit_logcdf_and_mills_ratio` and what SAS — the *same*
/// probit CDF, reparameterized — had no route to.
///
/// The derivative is the chain factor `dz/deta`; a consumer converts a
/// `d/dz` score into a `d/deta` score by multiplying by it. Inside the
/// `smooth_bound` saturation band `dz/deta` is exactly `0`, which is the
/// correct value: `mu` is genuinely constant in `eta` there.
///
/// At `epsilon = 0, delta = 1` this returns `(eta, 1.0)` bitwise, so a SAS link
/// at its identity parameters and the standard probit link produce the *same*
/// geometry rather than two numerically different ones.
pub(crate) fn sas_latent_probit_argument(
    eta: f64,
    epsilon: f64,
    log_delta: f64,
) -> Result<(f64, f64), EstimationError> {
    let eta = finite_inverse_link_eta("SAS inverse link", eta)?;
    let delta = sas_delta_from_raw_log_delta(log_delta);
    // The identity parameters are exact: the bound map fixes 0 and exp(0) = 1 (#2469).
    if epsilon == 0.0 && delta == 1.0 {
        return Ok((eta, 1.0));
    }
    let asinh = asinh_jet6(eta);
    let u_raw = delta * asinh.value + epsilon;
    let sb = smooth_bound_jet(u_raw, sas_latent_domain_bound());
    let u = sb.g;
    let c = u.cosh();
    let r1 = delta * asinh.d1;
    let u1 = sb.d1 * r1;
    Ok((u.sinh(), c * u1))
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
        LinkFunction::Sqrt => {
            let jet = sqrt_link_jet6(eta)?;
            Ok((jet[0], jet[1]))
        }
        LinkFunction::Inverse => {
            let jet = reciprocal_power_link_jet6(INVERSE_LINK_NAME, 1.0, eta)?;
            Ok((jet[0], jet[1]))
        }
        LinkFunction::InverseSquared => {
            let jet = reciprocal_power_link_jet6(INVERSE_SQUARED_LINK_NAME, 0.5, eta)?;
            Ok((jet[0], jet[1]))
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
            let (q, _) = cauchit_rational_factors(eta);
            (cauchit_mean(eta), q / std::f64::consts::PI)
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
    // The identity parameters are exact: the bound map fixes 0 and exp(0) = 1 (#2469).
    if epsilon == 0.0 && delta_id == 1.0 {
        return Ok(component_inverse_link_mu_d1(LinkComponent::Probit, eta));
    }
    let asinh = asinh_jet6(eta);
    let delta = delta_id;
    let u_raw = delta * asinh.value + epsilon;
    let sb = smooth_bound_jet(u_raw, sas_latent_domain_bound());
    let u = sb.g;
    let g1 = sb.d1;
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
    let (a, b) = beta_logistic_shapes(delta, epsilon);
    let (x, s) = beta_logistic_latent_argument(eta, a, b);
    let logistic = logistic_uwith_derivatives(x);
    let mu = beta_reg_logistic(a, b, logistic);
    let log_d1 = beta_logistic_log_d1(a, b, logistic);
    (mu, s * log_d1.exp())
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
    Fifth,
}

impl PdfDerivativeOrder {
    fn probit(self, eta: f64) -> f64 {
        match self {
            Self::Third => probit_pdfthird_derivative(eta),
            Self::Fourth => probit_pdffourth_derivative(eta),
            Self::Fifth => probit_pdffifth_derivative(eta),
        }
    }

    fn component(self, component: LinkComponent, eta: f64) -> f64 {
        match self {
            Self::Third => component_inverse_link_pdfthird_derivative(component, eta),
            Self::Fourth => component_inverse_link_pdffourth_derivative(component, eta),
            Self::Fifth => component_inverse_link_pdffifth_derivative(component, eta),
        }
    }

    fn latent_cloglog(self, eta: f64, latent_sd: f64) -> Result<f64, EstimationError> {
        let ctx = latent_cloglog_quadctx();
        match self {
            Self::Third => Ok(latent_cloglog_jet5(ctx, eta, latent_sd)?.d4),
            Self::Fourth => Ok(latent_cloglog_jet5(ctx, eta, latent_sd)?.d5),
            Self::Fifth => latent_cloglog_d6(ctx, eta, latent_sd),
        }
    }

    fn sas(self, eta: f64, epsilon: f64, log_delta: f64) -> Result<f64, EstimationError> {
        match self {
            Self::Third => sas_inverse_link_pdfthird_derivative(eta, epsilon, log_delta),
            Self::Fourth => sas_inverse_link_pdffourth_derivative(eta, epsilon, log_delta),
            Self::Fifth => sas_inverse_link_pdffifth_derivative(eta, epsilon, log_delta),
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
            Self::Fifth => {
                beta_logistic_inverse_link_pdffifth_derivative(eta, log_shape_center, epsilon)
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
        InverseLink::Standard(StandardLink::Sqrt) => sqrt_link_jet6(eta).map(|_| 0.0),
        InverseLink::Standard(StandardLink::Log) => log_link_solver_exp(eta),
        InverseLink::Standard(link @ (StandardLink::Inverse | StandardLink::InverseSquared)) => {
            let jet = standard_reciprocal_power_jet6(*link, eta)
                .expect("reciprocal-power arm matched a reciprocal-power link")?;
            Ok(match order {
                PdfDerivativeOrder::Third => jet[4],
                PdfDerivativeOrder::Fourth => jet[5],
                PdfDerivativeOrder::Fifth => {
                    // d6 = d5·(−a − 5)/η: the next step of the jet's
                    // `c_{k+1} = c_k·(−a − k)` recursion.
                    let exponent = if *link == StandardLink::Inverse { 1.0 } else { 0.5 };
                    jet[5] * (-exponent - 5.0) / eta
                }
            })
        }
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

/// Sixth derivative of the inverse-link CDF (= fifth derivative of the PDF).
///
/// Extends [`inverse_link_pdffourth_derivative_for_inverse_link`] by one order.
/// It is the top order of the Bernoulli log-probability jet the Firth
/// observed-information tower differentiates, so the TK outer ρ-Hessian's
/// `d⁴W/dη⁴` term is analytic for every Firth link.
pub fn inverse_link_pdffifth_derivative_for_inverse_link(
    link: &InverseLink,
    eta: f64,
) -> Result<f64, EstimationError> {
    inverse_link_pdf_derivative_for_inverse_link(link, eta, PdfDerivativeOrder::Fifth)
}

/// θ-partials of the inverse-link density's third derivative `f‴ = ∂⁴μ/∂η⁴`,
/// the quantity [`inverse_link_pdfthird_derivative_for_inverse_link`] returns,
/// one per free shape parameter in [`LinkParamPartials`] order: `(ε, log δ)` for
/// SAS and Beta-Logistic, the free logits for a mixture. `None` for a link with
/// no shape parameters. The jet partials stop at `f″`; a survival row program
/// needs this one order further for the mixed information drift of a shape axis
/// (#2904).
pub fn inverse_link_pdfthird_derivative_param_partials(
    link: &InverseLink,
    eta: f64,
) -> Result<Option<Vec<f64>>, EstimationError> {
    match link {
        InverseLink::Standard(_) | InverseLink::LatentCLogLog(_) => Ok(None),
        InverseLink::Sas(state) => Ok(Some(
            sas_inverse_link_pdfthird_derivative_param_partials(eta, state.epsilon, state.log_delta)?
                .to_vec(),
        )),
        InverseLink::BetaLogistic(state) => Ok(Some(
            beta_logistic_inverse_link_pdfthird_derivative_param_partials(
                eta,
                state.log_delta,
                state.epsilon,
            )
            .to_vec(),
        )),
        InverseLink::Mixture(state) => {
            // `f‴ = Σ_j π_j f_j‴` with softmax weights, so a free logit moves it as
            // `∂f‴/∂ρ_j = π_j (f_j‴ − f‴)`, the rule the jet partials use.
            let k = state.components.len().min(state.pi.len());
            let component_third: Vec<f64> = state.components[..k]
                .iter()
                .map(|&component| component_inverse_link_pdfthird_derivative(component, eta))
                .collect();
            let mixed_third: f64 = component_third
                .iter()
                .zip(state.pi.iter())
                .map(|(&third, &weight)| weight * third)
                .sum();
            Ok(Some(
                (0..k.saturating_sub(1))
                    .map(|j| state.pi[j] * (component_third[j] - mixed_third))
                    .collect(),
            ))
        }
    }
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

/// Exact-public log inverse-link jet: `mu = d1 = d2 = d3 = exp(η)` with no
/// solver-domain restriction. The solver-internal sibling evaluates the same
/// exact expression only on `LOG_LINK_SOLVER_ETA_MIN` through
/// `LOG_LINK_SOLVER_ETA_MAX` and returns a typed refusal outside it; see issue
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
/// Identical to the family's standard inverse-link jet for every link EXCEPT the
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
pub(crate) fn mixture_inverse_link_jetwith_rho_partials(
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
    let mut d2d2_drho2 = Array2::<f64>::zeros((m, m));
    for j in 0..m {
        for k in j..m {
            let diagonal = if j == k { 1.0 } else { 0.0 };
            let mu = (diagonal - state.pi[k]) * djet_drho[j].mu
                - state.pi[j] * djet_drho[k].mu;
            let d1 = (diagonal - state.pi[k]) * djet_drho[j].d1
                - state.pi[j] * djet_drho[k].d1;
            let d2 = (diagonal - state.pi[k]) * djet_drho[j].d2
                - state.pi[j] * djet_drho[k].d2;
            d2mu_drho2[[j, k]] = mu;
            d2mu_drho2[[k, j]] = mu;
            d2d1_drho2[[j, k]] = d1;
            d2d1_drho2[[k, j]] = d1;
            d2d2_drho2[[j, k]] = d2;
            d2d2_drho2[[k, j]] = d2;
        }
    }
    MixtureJetWithRhoPartials {
        jet,
        djet_drho,
        d2mu_drho2,
        d2d1_drho2,
        d2d2_drho2,
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
    let ln_u = -gam_math::special::softplus(-eta);
    let ln_one_minus_u = -gam_math::special::softplus(eta);
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

/// `(ln K, ln(1 − K), ln K′)` of the latent kernel `K(x) = I_u(a, b)`,
/// `u = logistic(x)`, without forming `K` (#2902 row 34).
///
/// By reflection `1 − K = I_{1−u}(b,a)`. Both are evaluated in log space from
/// `ln u` and `ln(1 − u)`, which [`logistic_uwith_derivatives`] carries exactly, so
/// neither saturates when `u` rounds to `1.0` (x > 36.7) or when a tail probability
/// leaves `f64`. The smaller of the two is the accurate one, and the larger is
/// taken as its log-complement, so the pair always describes one probability.
/// `ln K′` is [`beta_logistic_log_d1`].
pub(crate) fn beta_logistic_latent_log_probabilities(x: f64, a: f64, b: f64) -> (f64, f64, f64) {
    let logistic = logistic_uwith_derivatives(x);
    let direct_mu =
        gam_math::probability::ln_regularized_beta_lower_from_log_x(logistic.ln_u, a, b);
    let direct_complement =
        gam_math::probability::ln_regularized_beta_lower_from_log_x(logistic.ln_one_minus_u, b, a);
    let (log_mu, log_one_minus_mu) = if direct_mu <= direct_complement {
        (direct_mu, gam_math::special::log_abs_one_minus_exp(direct_mu))
    } else {
        (gam_math::special::log_abs_one_minus_exp(direct_complement), direct_complement)
    };
    (log_mu, log_one_minus_mu, beta_logistic_log_d1(a, b, logistic))
}

/// `(ln μ, ln(1 − μ), ln μ′)` of the beta-logistic inverse link at a link state:
/// [`beta_logistic_latent_log_probabilities`] at the standardized latent argument,
/// with `ln μ′ = ln s + ln K′`.
pub(crate) fn beta_logistic_binomial_log_probabilities(
    eta: f64,
    log_shape_center: f64,
    epsilon: f64,
) -> (f64, f64, f64) {
    let (a, b) = beta_logistic_shapes(log_shape_center, epsilon);
    let (x, s) = beta_logistic_latent_argument(eta, a, b);
    let (log_mu, log_one_minus_mu, log_latent_d1) = beta_logistic_latent_log_probabilities(x, a, b);
    (log_mu, log_one_minus_mu, s.ln() + log_latent_d1)
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
/// The beta shapes `a = exp(log_shape_center − epsilon)` and
/// `b = exp(log_shape_center + epsilon)`, the exact exponential.
///
/// No bound: the standardization below removes the scale gauge a bound on the
/// shapes used to hold in place (#2902 row 34).
#[inline]
fn beta_logistic_shapes(log_shape_center: f64, epsilon: f64) -> (f64, f64) {
    ((log_shape_center - epsilon).exp(), (log_shape_center + epsilon).exp())
}

/// The standardization that holds the beta-logistic link at logit's location and
/// scale (#2902 row 34).
///
/// The link is the CDF of `Z = logit(U)`, `U ~ Beta(a, b)`, read through an affine
/// map of `η`: `μ(η) = K(x)` with `K(x) = I_{logistic(x)}(a, b)`, `x = E Z + s·η`,
/// `E Z = ψ(a) − ψ(b)` and `s = √((ψ₁(a) + ψ₁(b))/(2ψ₁(1)))`. `2ψ₁(1) = π²/3` is
/// logit's own variance, so `s = sd Z·√3/π` and `x` keeps logit's location and
/// scale whatever the shapes.
///
/// Unstandardized, the shapes and `β` both carry the scale of `η`. A flatter link
/// lets `β` grow while the observed information collapses, and LAML's `½ log|H|`
/// lowers the criterion along that shared direction although the fit worsens. On
/// the #2685 fixture with the old shape bound lifted (job 1144648), the criterion
/// fell by 3.04 while the deviance rose by 0.82. Max η went from 12.8 to 438.4, and
/// Δ log|H| = −14.35 against 4·ln(438.4/12.8) = 14.13. Standardized, `(ε, log δ)`
/// move only skew and tails.
///
/// At `a = b = 1`, both polygamma stacks are evaluated at one argument, so `E Z`
/// is exactly `0.0` and `s` exactly `1.0`. The standardization is inert there, and
/// the link is the latent kernel itself.
///
/// `latent_first[j]`, `latent_second[j][k]` are `∂x/∂θ_j`, `∂²x/∂θ_j∂θ_k` at this
/// `η`, and `scale_first`, `scale_second` the same for `s`. The parameter order is
/// `(epsilon, log_shape_center)`, as in
/// [`beta_logistic_inverse_link_jetwith_param_partials`].
struct BetaLogisticStandardization {
    latent: f64,
    scale: f64,
    latent_first: [f64; 2],
    scale_first: [f64; 2],
    latent_second: [[f64; 2]; 2],
    scale_second: [[f64; 2]; 2],
}

fn beta_logistic_standardization(eta: f64, a: f64, b: f64) -> BetaLogisticStandardization {
    let pa = [
        gam_math::special::digamma(a),
        gam_math::special::trigamma(a),
        gam_math::special::tetragamma(a),
        gam_math::special::pentagamma(a),
    ];
    let pb = [
        gam_math::special::digamma(b),
        gam_math::special::trigamma(b),
        gam_math::special::tetragamma(b),
        gam_math::special::pentagamma(b),
    ];
    let logit_variance = 2.0 * gam_math::special::trigamma(1.0);
    let scale = ((pa[1] + pb[1]) / logit_variance).sqrt();
    let location = pa[0] - pb[0];
    // `a = exp(l − e)`, `b = exp(l + e)`, in the order `(e, l)`.
    let a_first = [-a, a];
    let b_first = [b, b];
    let a_second = [[a, -a], [-a, a]];
    let b_second = [[b, b], [b, b]];
    // In the shapes: `∂E Z/∂a = ψ₁(a)`, `∂E Z/∂b = −ψ₁(b)`, `∂s/∂a = ψ₂(a)/(2·v₀·s)`,
    // `∂²s/∂a² = ψ₃(a)/(2·v₀·s) − ψ₂(a)²/(4·v₀²·s³)`, and
    // `∂²s/∂a∂b = −ψ₂(a)·ψ₂(b)/(4·v₀²·s³)`, with `v₀ = 2ψ₁(1)`.
    let half_over = 1.0 / (2.0 * logit_variance * scale);
    let quarter_over = 1.0 / (4.0 * logit_variance * logit_variance * scale * scale * scale);
    let s_a = pa[2] * half_over;
    let s_b = pb[2] * half_over;
    let s_aa = pa[3] * half_over - pa[2] * pa[2] * quarter_over;
    let s_bb = pb[3] * half_over - pb[2] * pb[2] * quarter_over;
    let s_ab = -pa[2] * pb[2] * quarter_over;
    let mut latent_first = [0.0; 2];
    let mut scale_first = [0.0; 2];
    let mut latent_second = [[0.0; 2]; 2];
    let mut scale_second = [[0.0; 2]; 2];
    for j in 0..2 {
        scale_first[j] = s_a * a_first[j] + s_b * b_first[j];
        latent_first[j] = pa[1] * a_first[j] - pb[1] * b_first[j] + scale_first[j] * eta;
        for k in 0..2 {
            let location_jk = pa[2] * a_first[j] * a_first[k] - pb[2] * b_first[j] * b_first[k]
                + pa[1] * a_second[j][k]
                - pb[1] * b_second[j][k];
            scale_second[j][k] = s_aa * a_first[j] * a_first[k]
                + s_ab * (a_first[j] * b_first[k] + b_first[j] * a_first[k])
                + s_bb * b_first[j] * b_first[k]
                + s_a * a_second[j][k]
                + s_b * b_second[j][k];
            latent_second[j][k] = location_jk + scale_second[j][k] * eta;
        }
    }
    BetaLogisticStandardization {
        latent: location + scale * eta,
        scale,
        latent_first,
        scale_first,
        latent_second,
        scale_second,
    }
}

/// `(x, s)` of [`beta_logistic_standardization`] without its partials, for the per-row
/// callers that need only the latent argument and the scale. It evaluates the same
/// scalars in the same order, so `x` and `s` are the same bits the full
/// standardization produces.
#[inline]
fn beta_logistic_latent_argument(eta: f64, a: f64, b: f64) -> (f64, f64) {
    let pa = [gam_math::special::digamma(a), gam_math::special::trigamma(a)];
    let pb = [gam_math::special::digamma(b), gam_math::special::trigamma(b)];
    let logit_variance = 2.0 * gam_math::special::trigamma(1.0);
    let scale = ((pa[1] + pb[1]) / logit_variance).sqrt();
    (pa[0] - pb[0] + scale * eta, scale)
}

/// Jet of the latent kernel `K(x) = I_{logistic(x)}(a, b)` in `x`.
fn beta_logistic_latent_jet(x: f64, a: f64, b: f64) -> InverseLinkJet {
    let logistic = logistic_uwith_derivatives(x);
    let mu = beta_reg_logistic(a, b, logistic);
    let log_d1 = beta_logistic_log_d1(a, b, logistic);
    let d1 = log_d1.exp();
    let t = a * logistic.one_minus_u - b * logistic.u;
    let d2 = d1 * t;
    let d3 = d1 * (t * t - (a + b) * logistic.du);
    InverseLinkJet { mu, d1, d2, d3 }
}

/// Beta-logistic inverse-link jet: `μ = K(x)` and `dᵏμ/dηᵏ = sᵏ·K⁽ᵏ⁾(x)` (see
/// [`beta_logistic_standardization`]).
pub fn beta_logistic_inverse_link_jet(
    eta: f64,
    log_shape_center: f64,
    epsilon: f64,
) -> InverseLinkJet {
    let (a, b) = beta_logistic_shapes(log_shape_center, epsilon);
    let standardization = beta_logistic_standardization(eta, a, b);
    let s = standardization.scale;
    let latent = beta_logistic_latent_jet(standardization.latent, a, b);
    InverseLinkJet {
        mu: latent.mu,
        d1: s * latent.d1,
        d2: s * s * latent.d2,
        d3: s * s * s * latent.d3,
    }
}

/// Fourth derivative of the beta-logistic link, `s⁴·K⁗(x)` with `x = E Z + s·η`
/// (see [`beta_logistic_standardization`]).
pub(crate) fn beta_logistic_inverse_link_pdfthird_derivative(
    eta: f64,
    log_shape_center: f64,
    epsilon: f64,
) -> f64 {
    let (a, b) = beta_logistic_shapes(log_shape_center, epsilon);
    let standardization = beta_logistic_standardization(eta, a, b);
    standardization.scale.powi(4)
        * beta_logistic_latent_pdfthird_derivative(standardization.latent, a, b)
}

/// Fifth derivative of the beta-logistic link, `s⁵·K⁽⁵⁾(x)`.
pub(crate) fn beta_logistic_inverse_link_pdffourth_derivative(
    eta: f64,
    log_shape_center: f64,
    epsilon: f64,
) -> f64 {
    let (a, b) = beta_logistic_shapes(log_shape_center, epsilon);
    let standardization = beta_logistic_standardization(eta, a, b);
    standardization.scale.powi(5)
        * beta_logistic_latent_pdffourth_derivative(standardization.latent, a, b)
}

/// Sixth derivative of the beta-logistic link, `s⁶·K⁽⁶⁾(x)`.
pub(crate) fn beta_logistic_inverse_link_pdffifth_derivative(
    eta: f64,
    log_shape_center: f64,
    epsilon: f64,
) -> f64 {
    let (a, b) = beta_logistic_shapes(log_shape_center, epsilon);
    let standardization = beta_logistic_standardization(eta, a, b);
    standardization.scale.powi(6)
        * beta_logistic_latent_pdffifth_derivative(standardization.latent, a, b)
}

/// Fourth derivative of the latent kernel `K(x) = I_{logistic(x)}(a, b)`, the
/// CDF of `Z = logit(U)`, `U ~ Beta(a, b)` (= third derivative of its density).
fn beta_logistic_latent_pdfthird_derivative(x: f64, a: f64, b: f64) -> f64 {
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
    let logistic = logistic_uwith_derivatives(x);
    let log_d1 = beta_logistic_log_d1(a, b, logistic);
    let d1 = log_d1.exp();
    let c = a + b;
    let t = a * logistic.one_minus_u - b * logistic.u;
    let u2 = logistic.du * (logistic.one_minus_u - logistic.u);
    d1 * (t * t * t - 3.0 * c * t * logistic.du - c * u2)
}

/// Fifth derivative of the latent kernel `K(x)` (= 4th derivative of its density).
///
/// With `P_4 = t^3 - 3ct*u' - c*u''` giving `d4 = d1 * P_4`, the next order is:
///
///   d5 = d1 * [t^4 - 6c*t^2*u' - 4c*t*u'' + 3c^2*u'^2 - c*u''']
///
/// where u' = u(1-u), u'' = u'(1-2u), u''' = u''(1-2u) - 2*u'^2.
fn beta_logistic_latent_pdffourth_derivative(x: f64, a: f64, b: f64) -> f64 {
    let logistic = logistic_uwith_derivatives(x);
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

/// Sixth derivative of the latent kernel `K(x)` (= 5th derivative of its density).
///
/// `log d1 = a·log u + b·log(1-u) + const` has x-derivatives `L1 = t` and
/// `L_{k+1} = -c·u⁽ᵏ⁾`, so `K⁽ⁿ⁺¹⁾ = d1·Y_n(L1..Ln)` with the complete Bell
/// polynomial `Y_n` (`Y4` is the bracket of the fifth derivative above). Order
/// five:
///
///   d6 = d1 * [L1^5 + 10 L1^3 L2 + 15 L1 L2^2 + 10 L1^2 L3 + 10 L2 L3 + 5 L1 L4 + L5]
///
/// with u'''' = u'''(1-2u) - 6u'u''.
fn beta_logistic_latent_pdffifth_derivative(x: f64, a: f64, b: f64) -> f64 {
    let logistic = logistic_uwith_derivatives(x);
    let log_d1 = beta_logistic_log_d1(a, b, logistic);
    let d1 = log_d1.exp();
    let c = a + b;
    let one_minus_2u = logistic.one_minus_u - logistic.u;
    let u1 = logistic.du;
    let u2 = u1 * one_minus_2u;
    let u3 = u2 * one_minus_2u - 2.0 * u1 * u1;
    let u4 = u3 * one_minus_2u - 6.0 * u1 * u2;
    let l1 = a * logistic.one_minus_u - b * logistic.u;
    let (l2, l3, l4, l5) = (-c * u1, -c * u2, -c * u3, -c * u4);
    let l1_2 = l1 * l1;
    d1 * (l1_2 * l1_2 * l1
        + 10.0 * l1_2 * l1 * l2
        + 15.0 * l1 * l2 * l2
        + 10.0 * l1_2 * l3
        + 10.0 * l2 * l3
        + 5.0 * l1 * l4
        + l5)
}

/// Parameter partials of the latent kernel `K(x; a, b)` at a fixed latent `x`, in
/// the order `(epsilon, log_shape_center)` with `a = exp(l − e)`, `b = exp(l + e)`.
fn beta_logistic_latent_jet_with_param_partials(x: f64, a: f64, b: f64) -> SasJetWithParamPartials {
    let logistic = logistic_uwith_derivatives(x);
    // `da`/`db` are `∂a/∂s`, `∂b/∂t` for `s = l − e`, `t = l + e`, and `daa`/`dbb`
    // the second derivatives; the exact exponential makes each equal its shape.
    let (da, db, daa, dbb) = (a, b, a, b);
    let shape = beta_reg_with_shape_partials_logistic(a, b, logistic);
    let mu = shape.value;
    let dmu_dlog_shape_center = da * shape.da + db * shape.db;
    let dmu_depsilon = -da * shape.da + db * shape.db;
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
    let djet_dlog_shape_center = partials_for(da, db, dmu_dlog_shape_center);
    let djet_depsilon = partials_for(-da, db, dmu_depsilon);
    // Parameter order is `(epsilon, log_shape_center)`. The beta shapes obey
    // `a=exp(l-e)`, `b=exp(l+e)`, so their first and second parameter jets are
    // closed form. Contract those jets with the exact `(a,b)` Hessian of the
    // regularized beta CDF for `mu`, and with the exact log-density Hessian for
    // `d1`. No numerical differencing or profile replay enters this path.
    let a_first = [-da, da];
    let b_first = [db, db];
    let a_second = [[daa, -daa], [-daa, daa]];
    let b_second = [[dbb, dbb], [dbb, dbb]];
    let logd1_first = [
        a_first[0] * la + b_first[0] * lb,
        a_first[1] * la + b_first[1] * lb,
    ];
    let trigamma_ab = trigamma(a + b);
    let la_a = trigamma_ab - trigamma(a);
    let la_b = trigamma_ab;
    let lb_a = trigamma_ab;
    let lb_b = trigamma_ab - trigamma(b);
    // `t = a(1−u) − b u` is linear in the shapes, so its parameter jets follow
    // the same first/second shape derivatives with no new special functions.
    let t_first = [
        a_first[0] * logistic.one_minus_u - b_first[0] * logistic.u,
        a_first[1] * logistic.one_minus_u - b_first[1] * logistic.u,
    ];
    let mut d2mu_dparams2 = Array2::<f64>::zeros((2, 2));
    let mut d2d1_dparams2 = Array2::<f64>::zeros((2, 2));
    let mut d2d2_dparams2 = Array2::<f64>::zeros((2, 2));
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
            // d2 = d1·t  ⇒  d2_jk = d1_jk·t + d1_j·t_k + d1_k·t_j + d1·t_jk,
            // with d1_j = d1·logd1_first[j] (#2665).
            let t_jk = a_second[j][k] * logistic.one_minus_u - b_second[j][k] * logistic.u;
            let d2_jk = d1_jk * t
                + d1 * logd1_first[j] * t_first[k]
                + d1 * logd1_first[k] * t_first[j]
                + d1 * t_jk;
            d2mu_dparams2[[j, k]] = mu_jk;
            d2mu_dparams2[[k, j]] = mu_jk;
            d2d1_dparams2[[j, k]] = d1_jk;
            d2d1_dparams2[[k, j]] = d1_jk;
            d2d2_dparams2[[j, k]] = d2_jk;
            d2d2_dparams2[[k, j]] = d2_jk;
        }
    }
    SasJetWithParamPartials {
        jet,
        djet_depsilon,
        djet_dlog_delta: djet_dlog_shape_center,
        d2mu_dparams2,
        d2d1_dparams2,
        d2d2_dparams2,
    }
}

/// Beta-logistic inverse-link jet with its exact parameter partials, through the
/// standardization `x = E Z + s·η` (see [`beta_logistic_standardization`]).
///
/// Notation: `K` is the latent kernel, `K_j` a partial at fixed `x`, and `x_j`, `s_j`
/// the standardization's partials. The link's `μ = K(x)`, `d1 = s·K′`, `d2 = s²·K″`
/// and `d3 = s³·K‴` then differentiate by the chain rule:
/// - `μ_j = K_j + K′·x_j`
/// - `d1_j = s_j·K′ + s·(K′_j + K″·x_j)`
/// - `d2_j = 2s·s_j·K″ + s²·(K″_j + K‴·x_j)`
/// - `d3_j = 3s²·s_j·K‴ + s³·(K‴_j + K⁗·x_j)`
///
/// The second partials apply the same rule once more.
pub fn beta_logistic_inverse_link_jetwith_param_partials(
    eta: f64,
    log_shape_center: f64,
    epsilon: f64,
) -> SasJetWithParamPartials {
    let (a, b) = beta_logistic_shapes(log_shape_center, epsilon);
    let standardization = beta_logistic_standardization(eta, a, b);
    let latent = beta_logistic_latent_jet_with_param_partials(standardization.latent, a, b);
    let k = latent.jet;
    let k4 = beta_logistic_latent_pdfthird_derivative(standardization.latent, a, b);
    let kj = [latent.djet_depsilon, latent.djet_dlog_delta];
    let s = standardization.scale;
    let x1 = standardization.latent_first;
    let s1 = standardization.scale_first;
    let x2 = standardization.latent_second;
    let s2 = standardization.scale_second;
    let first = |j: usize| InverseLinkJet {
        mu: kj[j].mu + k.d1 * x1[j],
        d1: s1[j] * k.d1 + s * (kj[j].d1 + k.d2 * x1[j]),
        d2: 2.0 * s * s1[j] * k.d2 + s * s * (kj[j].d2 + k.d3 * x1[j]),
        d3: 3.0 * s * s * s1[j] * k.d3 + s * s * s * (kj[j].d3 + k4 * x1[j]),
    };
    let mut d2mu_dparams2 = Array2::<f64>::zeros((2, 2));
    let mut d2d1_dparams2 = Array2::<f64>::zeros((2, 2));
    let mut d2d2_dparams2 = Array2::<f64>::zeros((2, 2));
    for j in 0..2 {
        for i in j..2 {
            let mu_ji = latent.d2mu_dparams2[[j, i]]
                + kj[j].d1 * x1[i]
                + kj[i].d1 * x1[j]
                + k.d2 * x1[j] * x1[i]
                + k.d1 * x2[j][i];
            let d1_ji = s2[j][i] * k.d1
                + s1[j] * (kj[i].d1 + k.d2 * x1[i])
                + s1[i] * (kj[j].d1 + k.d2 * x1[j])
                + s * (latent.d2d1_dparams2[[j, i]]
                    + kj[j].d2 * x1[i]
                    + kj[i].d2 * x1[j]
                    + k.d3 * x1[j] * x1[i]
                    + k.d2 * x2[j][i]);
            // `d2 = q·K″` with `q = s²`, `q_j = 2s·s_j`, `q_ji = 2(s_j·s_i + s·s_ji)`.
            let q = s * s;
            let q_j = 2.0 * s * s1[j];
            let q_i = 2.0 * s * s1[i];
            let q_ji = 2.0 * (s1[j] * s1[i] + s * s2[j][i]);
            let d2_ji = q_ji * k.d2
                + q_j * (kj[i].d2 + k.d3 * x1[i])
                + q_i * (kj[j].d2 + k.d3 * x1[j])
                + q * (latent.d2d2_dparams2[[j, i]]
                    + kj[j].d3 * x1[i]
                    + kj[i].d3 * x1[j]
                    + k4 * x1[j] * x1[i]
                    + k.d3 * x2[j][i]);
            d2mu_dparams2[[j, i]] = mu_ji;
            d2mu_dparams2[[i, j]] = mu_ji;
            d2d1_dparams2[[j, i]] = d1_ji;
            d2d1_dparams2[[i, j]] = d1_ji;
            d2d2_dparams2[[j, i]] = d2_ji;
            d2d2_dparams2[[i, j]] = d2_ji;
        }
    }
    SasJetWithParamPartials {
        jet: InverseLinkJet {
            mu: k.mu,
            d1: s * k.d1,
            d2: s * s * k.d2,
            d3: s * s * s * k.d3,
        },
        djet_depsilon: first(0),
        djet_dlog_delta: first(1),
        d2mu_dparams2,
        d2d1_dparams2,
        d2d2_dparams2,
    }
}

/// `(∂f‴/∂ε, ∂f‴/∂log_shape_center)` of the Beta-Logistic density's third
/// derivative `f‴ = d1·P₄`, `P₄ = t³ − 3c·t·u′ − c·u″` (see
/// `beta_logistic_inverse_link_pdfthird_derivative`), in the parameter order of
/// [`beta_logistic_inverse_link_jetwith_param_partials`] (#2904). `u′` and `u″`
/// do not depend on the shapes; a shape motion `(a_θ, b_θ)` moves
/// `log d1` by `a_θ·l_a + b_θ·l_b`, `t` by `a_θ(1−u) − b_θ·u` and `c` by `a_θ + b_θ`.
pub(crate) fn beta_logistic_inverse_link_pdfthird_derivative_param_partials(
    eta: f64,
    log_shape_center: f64,
    epsilon: f64,
) -> [f64; 2] {
    // `f‴ = s⁴·K⁗(x)`, so `∂f‴/∂θ_j = 4s³·s_j·K⁗ + s⁴·(K⁗_j + K⁽⁵⁾·x_j)`.
    let (a, b) = beta_logistic_shapes(log_shape_center, epsilon);
    let standardization = beta_logistic_standardization(eta, a, b);
    let x = standardization.latent;
    let s = standardization.scale;
    let k4 = beta_logistic_latent_pdfthird_derivative(x, a, b);
    let k5 = beta_logistic_latent_pdffourth_derivative(x, a, b);
    let k4_j = beta_logistic_latent_pdfthird_derivative_param_partials(x, a, b);
    let s3 = s * s * s;
    let partial = |j: usize| -> f64 {
        4.0 * s3 * standardization.scale_first[j] * k4
            + s3 * s * (k4_j[j] + k5 * standardization.latent_first[j])
    };
    [partial(0), partial(1)]
}

/// `(∂K⁗/∂ε, ∂K⁗/∂log_shape_center)` of the latent kernel at a fixed latent `x`,
/// with the exact exponential shapes (`∂a/∂s = a`, `∂b/∂t = b`).
fn beta_logistic_latent_pdfthird_derivative_param_partials(x: f64, a: f64, b: f64) -> [f64; 2] {
    let logistic = logistic_uwith_derivatives(x);
    let (da, db) = (a, b);
    let d1 = beta_logistic_log_d1(a, b, logistic).exp();
    let c = a + b;
    let t = a * logistic.one_minus_u - b * logistic.u;
    let u2 = logistic.du * (logistic.one_minus_u - logistic.u);
    let p4 = t * t * t - 3.0 * c * t * logistic.du - c * u2;
    let psi_ab = digamma(a + b);
    let la = logistic.ln_u - digamma(a) + psi_ab;
    let lb = logistic.ln_one_minus_u - digamma(b) + psi_ab;
    let partial = |a_t: f64, b_t: f64| -> f64 {
        let d1_t = d1 * (a_t * la + b_t * lb);
        let t_t = a_t * logistic.one_minus_u - b_t * logistic.u;
        let c_t = a_t + b_t;
        let p4_t = 3.0 * t * t * t_t - 3.0 * (c_t * t + c * t_t) * logistic.du - c_t * u2;
        d1_t * p4 + d1 * p4_t
    };
    [partial(-da, db), partial(da, db)]
}

/// SAS inverse-link jet for:
///   mu(eta) = Phi(sinh(smooth_bound(delta * asinh(eta) + epsilon, latent domain))),
///   delta = exp(smooth_bound(log_delta, log-delta domain)).
/// `smooth_bound` is the interior-exact bounded latent map (see
/// `smooth_bound_jet`); on the interior it is the identity, so this reduces to
/// the pure probit jet exactly at `epsilon=0, delta=1`.
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
    // The identity parameters are exact: the bound map fixes 0 and exp(0) = 1 (#2469).
    if epsilon == 0.0 && delta_id == 1.0 {
        return Ok(component_inverse_link_jet(LinkComponent::Probit, eta));
    }
    let asinh = asinh_jet6(eta);
    let delta = delta_id;
    let u_raw = delta * asinh.value + epsilon;
    let sb = smooth_bound_jet(u_raw, sas_latent_domain_bound());
    let u = sb.g;
    let g1 = sb.d1;
    let g2 = sb.d2;
    let g3 = sb.d3;
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
pub(crate) fn sas_inverse_link_pdfthird_derivative(
    eta: f64,
    epsilon: f64,
    log_delta: f64,
) -> Result<f64, EstimationError> {
    // SAS link with bounded latent transform:
    //
    //   a  = asinh(eta),
    //   u  = smooth_bound(delta * a + epsilon),
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
    //   g = smooth_bound, r = delta * asinh(eta) + epsilon:
    //
    //   u4 = g'''' r1^4 + 6 g''' r1² r2 + 3 g'' r2² + 4 g'' r1 r3 + g' r4,
    //
    // which is the standard scalar Arbogast expansion for order four.
    let eta = finite_inverse_link_eta("SAS inverse link", eta)?;
    let asinh = asinh_jet6(eta);
    let delta = sas_delta_from_raw_log_delta(log_delta);
    let u_raw = delta * asinh.value + epsilon;
    let sb = smooth_bound_jet(u_raw, sas_latent_domain_bound());
    let u = sb.g;
    let g1 = sb.d1;
    let g2 = sb.d2;
    let g3 = sb.d3;
    let g4 = sb.d4;
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

/// Fifth derivative of the SAS inverse-link CDF (= fourth derivative of the PDF),
/// on the same finite domain as [`sas_inverse_link_jet`]: order five of the
/// composed chain [`sas_inverse_link_derivatives6`], so the fifth and sixth
/// orders share one evaluation of the bounded latent map.
pub(crate) fn sas_inverse_link_pdffourth_derivative(
    eta: f64,
    epsilon: f64,
    log_delta: f64,
) -> Result<f64, EstimationError> {
    Ok(sas_inverse_link_derivatives6(eta, epsilon, log_delta)?[4])
}

/// Derivatives `1..=6` of a composition `f(h(η))` from the outer derivatives
/// `outer[k-1] = f⁽ᵏ⁾(h(η))` and the inner derivatives `inner[j-1] = h⁽ʲ⁾(η)`
/// (Faà di Bruno), by composing truncated Taylor series: with
/// `a(t) = Σ_j h⁽ʲ⁾ tʲ/j!`, `(f∘h)⁽ⁿ⁾ = n!·Σ_k f⁽ᵏ⁾/k!·[tⁿ] a(t)ᵏ`.
fn compose_derivatives6(outer: [f64; 6], inner: [f64; 6]) -> [f64; 6] {
    const FACTORIAL: [f64; 7] = [1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0];
    let mut a = [0.0_f64; 7];
    for j in 1..=6 {
        a[j] = inner[j - 1] / FACTORIAL[j];
    }
    let mut power = a;
    let mut out = [0.0_f64; 6];
    for k in 1..=6 {
        let weight = outer[k - 1] / FACTORIAL[k];
        for n in k..=6 {
            out[n - 1] += weight * power[n];
        }
        let mut next = [0.0_f64; 7];
        for i in (k)..=6 {
            for j in 1..=(6 - i) {
                next[i + j] += power[i] * a[j];
            }
        }
        power = next;
    }
    for n in 1..=6 {
        out[n - 1] *= FACTORIAL[n];
    }
    out
}

/// Bounded SAS core `u = smooth_bound(δ·asinh(η) + ε, latent domain)` and its
/// derivatives in the raw core, shared by the chains that differentiate it.
#[inline]
fn sas_bounded_core_jet(delta: f64, asinh_value: f64, epsilon: f64) -> SmoothBoundJet {
    smooth_bound_jet(delta * asinh_value + epsilon, sas_latent_domain_bound())
}

/// Sixth derivative of the SAS inverse-link CDF (= fifth derivative of the PDF),
/// on the same finite domain as [`sas_inverse_link_jet`]. The chain
/// `r = δ·asinh(η) + ε`, `u = smooth_bound(r)`, `z = sinh(u)`, `μ = Φ(z)` is
/// composed link by link with [`compose_derivatives6`].
pub(crate) fn sas_inverse_link_pdffifth_derivative(
    eta: f64,
    epsilon: f64,
    log_delta: f64,
) -> Result<f64, EstimationError> {
    Ok(sas_inverse_link_derivatives6(eta, epsilon, log_delta)?[5])
}

/// `μ⁽¹⁾..μ⁽⁶⁾` of the SAS inverse link through the composed chain.
fn sas_inverse_link_derivatives6(
    eta: f64,
    epsilon: f64,
    log_delta: f64,
) -> Result<[f64; 6], EstimationError> {
    let eta = finite_inverse_link_eta("SAS inverse link", eta)?;
    let asinh = asinh_jet6(eta);
    let delta = sas_delta_from_raw_log_delta(log_delta);
    let sb = sas_bounded_core_jet(delta, asinh.value, epsilon);
    let r = [asinh.d1, asinh.d2, asinh.d3, asinh.d4, asinh.d5, asinh.d6].map(|d| delta * d);
    let u = compose_derivatives6([sb.d1, sb.d2, sb.d3, sb.d4, sb.d5, sb.d6], r);
    let (s, c) = (sb.g.sinh(), sb.g.cosh());
    let z = compose_derivatives6([c, s, c, s, c, s], u);
    let base = probit_jet(s);
    let phi = [
        base.d1,
        base.d2,
        base.d3,
        probit_pdfthird_derivative(s),
        probit_pdffourth_derivative(s),
        probit_pdffifth_derivative(s),
    ];
    Ok(compose_derivatives6(phi, z).map(canonicalzero))
}

/// `(∂f‴/∂ε, ∂f‴/∂log δ)` of the SAS density's third derivative `f‴ = μ⁗`
/// (#2904): the chain `sas_inverse_link_pdfthird_derivative` evaluates,
/// differentiated term by term in one parameter. The parameter enters the raw
/// latent `u_raw = δ·asinh(η) + ε` with partial `rt` and its η-derivatives
/// `r_k = δ·asinh⁽ᵏ⁾` with partials `r_k,t`; `δ` moves through the bounded
/// effective log-delta exactly as in [`sas_inverse_link_jetwith_param_partials`].
pub(crate) fn sas_inverse_link_pdfthird_derivative_param_partials(
    eta: f64,
    epsilon: f64,
    log_delta: f64,
) -> Result<[f64; 2], EstimationError> {
    let eta = finite_inverse_link_eta("SAS inverse link", eta)?;
    let asinh = asinh_jet6(eta);
    let (ld_eff, dld_eff_draw) = sas_effective_log_delta(log_delta);
    let delta = ld_eff.exp();
    let ddelta_draw = delta * dld_eff_draw;
    let sb = sas_bounded_core_jet(delta, asinh.value, epsilon);
    let (g1, g2, g3, g4, g5) = (sb.d1, sb.d2, sb.d3, sb.d4, sb.d5);
    let s = sb.g.sinh();
    let c = sb.g.cosh();
    let base = probit_jet(s);
    let phi4 = probit_pdfthird_derivative(s);
    let phi5 = probit_pdffourth_derivative(s);
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
    let partial = |rt: f64, r1t: f64, r2t: f64, r3t: f64, r4t: f64| -> f64 {
        let u_t = g1 * rt;
        let u1_t = g2 * rt * r1 + g1 * r1t;
        let u2_t = g3 * rt * r1 * r1 + 2.0 * g2 * r1 * r1t + g2 * rt * r2 + g1 * r2t;
        let u3_t = g4 * rt * r1 * r1 * r1
            + 3.0 * g3 * r1 * r1 * r1t
            + 3.0 * g3 * rt * r1 * r2
            + 3.0 * g2 * (r1t * r2 + r1 * r2t)
            + g2 * rt * r3
            + g1 * r3t;
        let u4_t = g5 * rt * r1.powi(4)
            + 4.0 * g4 * r1 * r1 * r1 * r1t
            + 6.0 * (g4 * rt * r1 * r1 * r2 + g3 * (2.0 * r1 * r1t * r2 + r1 * r1 * r2t))
            + 3.0 * (g3 * rt * r2 * r2 + 2.0 * g2 * r2 * r2t)
            + 4.0 * (g3 * rt * r1 * r3 + g2 * (r1t * r3 + r1 * r3t))
            + g2 * rt * r4
            + g1 * r4t;
        let z_t = c * u_t;
        let z1_t = s * u_t * u1 + c * u1_t;
        let z2_t = c * u_t * u1 * u1 + 2.0 * s * u1 * u1_t + s * u_t * u2 + c * u2_t;
        let z3_t = s * u_t * u1 * u1 * u1
            + 3.0 * c * u1 * u1 * u1_t
            + 3.0 * c * u_t * u1 * u2
            + 3.0 * s * (u1_t * u2 + u1 * u2_t)
            + s * u_t * u3
            + c * u3_t;
        let z4_t = c * u_t * u1.powi(4)
            + 4.0 * s * u1 * u1 * u1 * u1_t
            + 6.0 * (s * u_t * u1 * u1 * u2 + c * (2.0 * u1 * u1_t * u2 + u1 * u1 * u2_t))
            + 3.0 * (c * u_t * u2 * u2 + 2.0 * s * u2 * u2_t)
            + 4.0 * (c * u_t * u1 * u3 + s * (u1_t * u3 + u1 * u3_t))
            + s * u_t * u4
            + c * u4_t;
        phi5 * z_t * z1.powi(4)
            + 4.0 * phi4 * z1 * z1 * z1 * z1_t
            + 6.0 * (phi4 * z_t * z1 * z1 * z2 + base.d3 * (2.0 * z1 * z1_t * z2 + z1 * z1 * z2_t))
            + 3.0 * (base.d3 * z_t * z2 * z2 + 2.0 * base.d2 * z2 * z2_t)
            + 4.0 * (base.d3 * z_t * z1 * z3 + base.d2 * (z1_t * z3 + z1 * z3_t))
            + base.d2 * z_t * z4
            + base.d1 * z4_t
    };
    Ok([
        partial(1.0, 0.0, 0.0, 0.0, 0.0),
        partial(
            ddelta_draw * asinh.value,
            ddelta_draw * asinh.d1,
            ddelta_draw * asinh.d2,
            ddelta_draw * asinh.d3,
            ddelta_draw * asinh.d4,
        ),
    ])
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
    let asinh = asinh_jet6(eta);
    let ld_sb = smooth_bound_jet(log_delta, sas_log_delta_domain_bound());
    let (ld_eff, dld_eff_draw) = (ld_sb.g, ld_sb.d1);
    let d2ld_eff_draw2 = ld_sb.d2;
    let delta = ld_eff.exp();
    let ddelta_draw = delta * dld_eff_draw;
    let d2delta_draw2 = delta * (dld_eff_draw * dld_eff_draw + d2ld_eff_draw2);
    let u_raw = delta * asinh.value + epsilon;
    let sb = smooth_bound_jet(u_raw, sas_latent_domain_bound());
    let u = sb.g;
    let g1 = sb.d1;
    let g2 = sb.d2;
    let g3 = sb.d3;
    let g4 = sb.d4;
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

    // Exact parameter Hessians. `mu` and `d1` enter the scalar
    // log-survival/log-density terms; `d2` is carried as well because the
    // observed working weight `W_obs` of the outer link-parameter Hessian
    // depends on it (#2665, see `build_link_ext_pair_callback`). Parameter
    // order is `(epsilon, raw_log_delta)`.
    let r_t = [rt_eps, rt_ld];
    let r1_t = [r1t_eps, r1t_ld];
    let r2_t = [r2t_eps, r2t_ld];
    let r_tt = [[0.0, 0.0], [0.0, d2delta_draw2 * asinh.value]];
    let r1_tt = [[0.0, 0.0], [0.0, d2delta_draw2 * a1]];
    let r2_tt = [[0.0, 0.0], [0.0, d2delta_draw2 * a2]];
    let u_t = [u_eps, u_ld];
    let u1_t = [u1_eps, u1_ld];
    let u2_t = [u2_eps, u2_ld];
    let z_t = [c * u_t[0], c * u_t[1]];
    let z1_t = [
        s * u_t[0] * u1 + c * u1_t[0],
        s * u_t[1] * u1 + c * u1_t[1],
    ];
    let z2_t = [
        c * u_t[0] * u1 * u1 + 2.0 * s * u1 * u1_t[0] + s * u_t[0] * u2 + c * u2_t[0],
        c * u_t[1] * u1 * u1 + 2.0 * s * u1 * u1_t[1] + s * u_t[1] * u2 + c * u2_t[1],
    ];
    let phi3 = probit_pdfthird_derivative(z);
    let mut d2mu_dparams2 = Array2::<f64>::zeros((2, 2));
    let mut d2d1_dparams2 = Array2::<f64>::zeros((2, 2));
    let mut d2d2_dparams2 = Array2::<f64>::zeros((2, 2));
    for j in 0..2 {
        for k in j..2 {
            let u_jk = g2 * r_t[j] * r_t[k] + g1 * r_tt[j][k];
            let u1_jk = g3 * r_t[j] * r_t[k] * r1
                + g2 * r_tt[j][k] * r1
                + g2 * r_t[j] * r1_t[k]
                + g2 * r_t[k] * r1_t[j]
                + g1 * r1_tt[j][k];
            // u2 = g2 r1² + g1 r2, differentiated twice in the parameters.
            let u2_jk = g4 * r_t[j] * r_t[k] * r1 * r1
                + g3 * r_tt[j][k] * r1 * r1
                + 2.0 * g3 * r_t[j] * r1 * r1_t[k]
                + 2.0 * g3 * r_t[k] * r1 * r1_t[j]
                + 2.0 * g2 * (r1_t[j] * r1_t[k] + r1 * r1_tt[j][k])
                + g3 * r_t[j] * r_t[k] * r2
                + g2 * r_tt[j][k] * r2
                + g2 * r_t[j] * r2_t[k]
                + g2 * r_t[k] * r2_t[j]
                + g1 * r2_tt[j][k];
            let z_jk = s * u_t[j] * u_t[k] + c * u_jk;
            let z1_jk = c * u_t[j] * u_t[k] * u1
                + s * u_jk * u1
                + s * u_t[j] * u1_t[k]
                + s * u_t[k] * u1_t[j]
                + c * u1_jk;
            // z2 = sinh(u) u1² + cosh(u) u2, differentiated twice.
            let z2_jk = s * u_t[j] * u_t[k] * u1 * u1
                + c * u_jk * u1 * u1
                + 2.0 * c * u_t[j] * u1 * u1_t[k]
                + 2.0 * c * u_t[k] * u1 * u1_t[j]
                + 2.0 * s * (u1_t[j] * u1_t[k] + u1 * u1_jk)
                + c * u_t[j] * u_t[k] * u2
                + s * u_jk * u2
                + s * u_t[j] * u2_t[k]
                + s * u_t[k] * u2_t[j]
                + c * u2_jk;
            let mu_jk = base.d2 * z_t[j] * z_t[k] + base.d1 * z_jk;
            let d1_jk = base.d3 * z_t[j] * z_t[k] * z1
                + base.d2 * z_jk * z1
                + base.d2 * z_t[j] * z1_t[k]
                + base.d2 * z_t[k] * z1_t[j]
                + base.d1 * z1_jk;
            // d2 = Phi''(z) z1² + Phi'(z) z2, differentiated twice; `phi3` is
            // the next probit derivative in the same ladder.
            let d2_jk = phi3 * z_t[j] * z_t[k] * z1 * z1
                + base.d3 * z_jk * z1 * z1
                + 2.0 * base.d3 * z_t[j] * z1 * z1_t[k]
                + 2.0 * base.d3 * z_t[k] * z1 * z1_t[j]
                + 2.0 * base.d2 * (z1_t[j] * z1_t[k] + z1 * z1_jk)
                + base.d3 * z_t[j] * z_t[k] * z2
                + base.d2 * z_jk * z2
                + base.d2 * z_t[j] * z2_t[k]
                + base.d2 * z_t[k] * z2_t[j]
                + base.d1 * z2_jk;
            d2mu_dparams2[[j, k]] = mu_jk;
            d2mu_dparams2[[k, j]] = mu_jk;
            d2d1_dparams2[[j, k]] = d1_jk;
            d2d1_dparams2[[k, j]] = d1_jk;
            d2d2_dparams2[[j, k]] = d2_jk;
            d2d2_dparams2[[k, j]] = d2_jk;
        }
    }

    Ok(SasJetWithParamPartials {
        jet,
        djet_depsilon,
        djet_dlog_delta,
        d2mu_dparams2,
        d2d1_dparams2,
        d2d2_dparams2,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_problem::{InverseLink, LikelihoodSpec, LinkComponent, MixtureLinkSpec, SasLinkState};

    #[test]
    fn cauchit_derivatives_preserve_representable_extreme_tails() {
        // At these eta values the denominator powers in the unscaled formulas
        // overflow, although every listed derivative remains representable.
        let cases = [
            (1_u32, 1.0e150_f64, 1.0),
            (1, 1.0e155, 1.0),
            (2, 1.0e100, -2.0),
            (3, 1.0e80, 6.0),
            (4, 1.0e64, -24.0),
            (5, 1.0e52, 120.0),
        ];
        for (order, eta, coefficient) in cases {
            // The leading tail derivative is (-1)^(order-1) order! /
            // (pi eta^(order+1)); corrections here are far below one ulp.
            let expected = (0..=order)
                .fold(coefficient / std::f64::consts::PI, |value, _| value / eta);
            for sign in [-1.0, 1.0] {
                let x = sign * eta;
                let jet = component_inverse_link_jet(LinkComponent::Cauchit, x);
                let actual = match order {
                    1 => jet.d1,
                    2 => jet.d2,
                    3 => jet.d3,
                    4 => component_inverse_link_pdfthird_derivative(LinkComponent::Cauchit, x),
                    5 => component_inverse_link_pdffourth_derivative(LinkComponent::Cauchit, x),
                    _ => unreachable!(),
                };
                let expected = if order % 2 == 0 { sign * expected } else { expected };
                assert!(actual.is_finite() && actual != 0.0, "order={order}, eta={x}: {actual}");
                let tolerance = 4.0 * f64::from_bits(1) + 3.0e-14 * expected.abs();
                assert!((actual - expected).abs() <= tolerance,
                    "order={order}, eta={x}: actual={actual}, expected={expected}");
            }
        }
    }

    #[test]
    fn cauchit_mean_fast_path_and_complement_share_exact_tail_arithmetic() {
        for eta in [
            -f64::MAX, -1.0e150, -5.0, -1.0, -0.5, 0.0, 0.5, 1.0, 5.0,
            1.0e150, f64::MAX,
        ] {
            let jet = component_inverse_link_jet(LinkComponent::Cauchit, eta);
            let fast = component_inverse_link_mu_d1(LinkComponent::Cauchit, eta);
            assert_eq!(fast, (jet.mu, jet.d1));
            let complement = standard_link_complement(StandardLink::Cauchit, eta, jet.mu);
            assert_eq!(complement, component_inverse_link_jet(LinkComponent::Cauchit, -eta).mu);
            assert!(jet.mu.is_finite() && complement.is_finite());
            if eta < 0.0 {
                assert!(jet.mu > 0.0, "representable lower tail at {eta}");
            } else {
                assert!(complement > 0.0, "representable upper tail at {eta}");
            }
            assert!(jet.d1.is_finite() && jet.d2.is_finite() && jet.d3.is_finite());
            assert!(component_inverse_link_pdfthird_derivative(LinkComponent::Cauchit, eta).is_finite());
            assert!(component_inverse_link_pdffourth_derivative(LinkComponent::Cauchit, eta).is_finite());
        }
        for eta in [f64::NEG_INFINITY, f64::INFINITY] {
            let jet = component_inverse_link_jet(LinkComponent::Cauchit, eta);
            assert_eq!(jet.mu, if eta > 0.0 { 1.0 } else { 0.0 });
            assert_eq!((jet.d1, jet.d2, jet.d3), (0.0, 0.0, 0.0));
            assert_eq!(component_inverse_link_pdfthird_derivative(LinkComponent::Cauchit, eta), 0.0);
            assert_eq!(component_inverse_link_pdffourth_derivative(LinkComponent::Cauchit, eta), 0.0);
        }
        let nan = component_inverse_link_jet(LinkComponent::Cauchit, f64::NAN);
        assert!(nan.mu.is_nan() && nan.d1.is_nan() && nan.d2.is_nan() && nan.d3.is_nan());
        assert!(component_inverse_link_pdfthird_derivative(LinkComponent::Cauchit, f64::NAN).is_nan());
        assert!(component_inverse_link_pdffourth_derivative(LinkComponent::Cauchit, f64::NAN).is_nan());
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
            let h6 = sas_inverse_link_pdffifth_derivative(eta, state.epsilon, state.log_delta)
                .expect("finite SAS boundary sixth derivative");
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
                h6,
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

    /// #2665: `d2d2_*` is the parameter Hessian of `d2 = d²mu/deta²`, the
    /// ingredient the observed working weight `W_obs` depends on. Each entry is
    /// differenced against the *analytic first* partial `djet_*.d2` (itself
    /// FD-gated by `sas_param_partials_matchfd` above), so a sign or chain
    /// error in the new second-order ladder cannot hide behind the truncation
    /// error of a second difference of the raw value.
    #[test]
    fn sas_d2d2_param_hessian_matchesfd_of_the_first_partial() {
        let eta = 0.37;
        let epsilon = -0.12;
        let log_delta = 0.21;
        let h = 1e-6;
        let out = sas_inverse_link_jetwith_param_partials(eta, epsilon, log_delta)
            .expect("finite SAS eta");
        let at = |e: f64, l: f64| {
            sas_inverse_link_jetwith_param_partials(eta, e, l).expect("finite SAS eta")
        };
        // Column 0 = d/d(epsilon), column 1 = d/d(raw log delta).
        let d_eps = [
            (at(epsilon + h, log_delta).djet_depsilon.d2
                - at(epsilon - h, log_delta).djet_depsilon.d2)
                / (2.0 * h),
            (at(epsilon, log_delta + h).djet_depsilon.d2
                - at(epsilon, log_delta - h).djet_depsilon.d2)
                / (2.0 * h),
        ];
        let d_ld = [
            (at(epsilon + h, log_delta).djet_dlog_delta.d2
                - at(epsilon - h, log_delta).djet_dlog_delta.d2)
                / (2.0 * h),
            (at(epsilon, log_delta + h).djet_dlog_delta.d2
                - at(epsilon, log_delta - h).djet_dlog_delta.d2)
                / (2.0 * h),
        ];
        for (idx, reference) in [((0, 0), d_eps[0]), ((0, 1), d_eps[1])] {
            let got = out.d2d2_dparams2[[idx.0, idx.1]];
            assert!(
                (got - reference).abs() < 1e-5 * (1.0 + reference.abs()),
                "d2d2_dparams2{idx:?} = {got:e} against FD {reference:e}"
            );
        }
        for (idx, reference) in [((1, 0), d_ld[0]), ((1, 1), d_ld[1])] {
            let got = out.d2d2_dparams2[[idx.0, idx.1]];
            assert!(
                (got - reference).abs() < 1e-5 * (1.0 + reference.abs()),
                "d2d2_dparams2{idx:?} = {got:e} against FD {reference:e}"
            );
        }
        assert_eq!(out.d2d2_dparams2[[0, 1]], out.d2d2_dparams2[[1, 0]]);
    }

    /// Same gate for the beta-logistic flexible link, whose `d2 = d1·t` chain
    /// is a different derivation from the SAS probit ladder.
    #[test]
    fn beta_logistic_d2d2_param_hessian_matchesfd_of_the_first_partial() {
        let eta = -0.29;
        let epsilon = 0.18;
        let log_shape_center = 0.24;
        let h = 1e-6;
        let out =
            beta_logistic_inverse_link_jetwith_param_partials(eta, log_shape_center, epsilon);
        let at =
            |e: f64, l: f64| beta_logistic_inverse_link_jetwith_param_partials(eta, l, e);
        let reference = [
            [
                (at(epsilon + h, log_shape_center).djet_depsilon.d2
                    - at(epsilon - h, log_shape_center).djet_depsilon.d2)
                    / (2.0 * h),
                (at(epsilon, log_shape_center + h).djet_depsilon.d2
                    - at(epsilon, log_shape_center - h).djet_depsilon.d2)
                    / (2.0 * h),
            ],
            [
                (at(epsilon + h, log_shape_center).djet_dlog_delta.d2
                    - at(epsilon - h, log_shape_center).djet_dlog_delta.d2)
                    / (2.0 * h),
                (at(epsilon, log_shape_center + h).djet_dlog_delta.d2
                    - at(epsilon, log_shape_center - h).djet_dlog_delta.d2)
                    / (2.0 * h),
            ],
        ];
        for j in 0..2 {
            for k in 0..2 {
                let got = out.d2d2_dparams2[[j, k]];
                let want = reference[j][k];
                assert!(
                    (got - want).abs() < 1e-4 * (1.0 + want.abs()),
                    "beta-logistic d2d2_dparams2[{j},{k}] = {got:e} against FD {want:e}"
                );
            }
        }
    }

    /// Same gate for the mixture link's free-logit coordinates.
    #[test]
    fn mixture_d2d2_rho_hessian_matchesfd_of_the_first_partial() {
        let state = state_fromspec(&MixtureLinkSpec {
            components: vec![
                LinkComponent::Probit,
                LinkComponent::Logit,
                LinkComponent::CLogLog,
            ],
            initial_rho: Array1::from_vec(vec![0.31, -0.17]),
        })
        .expect("valid three-component mixture");
        let eta = 0.42;
        let h = 1e-6;
        let out = mixture_inverse_link_jetwith_rho_partials(&state, eta);
        let m = state.rho.len();
        let shifted = |j: usize, delta: f64| {
            let mut rho = state.rho.clone();
            rho[j] += delta;
            state_fromspec(&MixtureLinkSpec {
                components: state.components.clone(),
                initial_rho: rho,
            })
            .expect("valid shifted mixture")
        };
        for j in 0..m {
            for k in 0..m {
                let plus = mixture_inverse_link_jetwith_rho_partials(&shifted(k, h), eta);
                let minus = mixture_inverse_link_jetwith_rho_partials(&shifted(k, -h), eta);
                let want = (plus.djet_drho[j].d2 - minus.djet_drho[j].d2) / (2.0 * h);
                let got = out.d2d2_drho2[[j, k]];
                assert!(
                    (got - want).abs() < 1e-5 * (1.0 + want.abs()),
                    "mixture d2d2_drho2[{j},{k}] = {got:e} against FD {want:e}"
                );
            }
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
        // At the reduction center (η=0, ε=0, log_δ=0) the bounded latent map is on
        // its exact-identity interior, so `u = ε + asinh(η)/… ` composes with no
        // bounded-map curvature. The ε–ε second partial of `d1 = μ'` is therefore
        // the TRUE sinh-arcsinh value 0 — not the old `-2·φ(0)/B²`, which
        // was precisely the spurious `tanh` third-derivative `g'''(0) = -2/B²` that
        // this fix removes. Expanding `d1(ε) = φ(sinh ε)·cosh ε = φ(0)(1 + O(ε⁴))`
        // at η=0 confirms `∂²d1/∂ε² = 0` exactly.
        assert_eq!(out.d2mu_dparams2, Array2::<f64>::zeros((2, 2)));
        assert_eq!(out.d2d1_dparams2[[0, 1]], out.d2d1_dparams2[[1, 0]]);
        assert_eq!(
            out.d2d1_dparams2[[0, 0]], 0.0,
            "ε–ε ∂²(μ') must be exactly zero at the identity-interior center"
        );
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
    fn beta_logistic_reduces_to_logit_at_delta0_epsilon0() {
        let etas = [-40.0, -30.0, -5.0, 0.42, 5.0, 30.0, 40.0];
        for eta in etas {
            let j_bl = beta_logistic_inverse_link_jet(eta, 0.0, 0.0);
            let expected_mu = gam_math::special::logistic(eta);
            let expected_d1 = (-gam_math::special::softplus(-eta)
                - gam_math::special::softplus(eta))
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
        // `(ε=0, δ=1)` takes the fast probit reduction path, which returns the
        // probit jet bitwise. This pins that contract exactly (no tolerance).
        let etas = [-3.0, -1.2, -0.3, 0.0, 0.4, 1.7, 3.0];
        for eta in etas {
            let sas = sas_inverse_link_jet(eta, 0.0, 0.0).expect("finite SAS eta");
            let probit = ProbitLinkKernel.jet(eta).expect("probit");
            assert_eq!(sas.mu.to_bits(), probit.mu.to_bits(), "mu at eta={eta}");
            assert_eq!(sas.d1.to_bits(), probit.d1.to_bits(), "d1 at eta={eta}");
            assert_eq!(sas.d2.to_bits(), probit.d2.to_bits(), "d2 at eta={eta}");
            assert_eq!(sas.d3.to_bits(), probit.d3.to_bits(), "d3 at eta={eta}");
        }
    }

    /// #2389 regression: the interior-exact bounded latent map makes the
    /// `SAS(ε=0, δ=1) ≡ probit` reduction hold on the FULL composition path (not
    /// just the fast short-circuit), and removes the `μ(ε)` cliff at `ε=0`.
    ///
    /// The old `B·tanh(x/B)` distorted the latent by `~1e-4` at every interior
    /// point, so (1) the full path returned `Φ(sinh(tanh_bound(asinh η))) ≈
    /// Φ(0.99987·η)` — off probit by `~2e-4` in μ — and (2) crossing the
    /// `|ε|<1e-12` fast-path threshold jumped μ between the two surfaces by that
    /// same `2e-4`. Both are gone: the full path is machine-exact probit, and
    /// μ is smooth through `ε=0`.
    #[test]
    fn sas_probit_reduction_is_exact_on_full_path_and_smooth_across_epsilon_zero() {
        let etas = [-3.0, -1.2, -0.3, 0.0, 0.4, 1.7, 3.0];
        for &eta in &etas {
            let probit = ProbitLinkKernel.jet(eta).expect("probit");
            // Full composition path, just outside the fast-path window in ε. With
            // the identity interior, `sinh(smooth_bound(asinh η)) = η` exactly, so
            // the whole jet collapses to probit up to sinh∘asinh round-off.
            for &eps in &[1e-11_f64, -1e-11] {
                let sas = sas_inverse_link_jet(eta, eps, 0.0).expect("finite SAS eta");
                // ε=1e-11 shifts the latent by 1e-11, a first-order μ move of
                // φ(η)·1e-11 ≲ 4e-12; everything beyond that must be < 1e-11.
                assert!(
                    (sas.mu - probit.mu).abs() < 1e-11,
                    "full-path μ off probit at eta={eta} eps={eps}: {} vs {}",
                    sas.mu,
                    probit.mu
                );
                assert!(
                    (sas.d1 - probit.d1).abs() < 1e-10,
                    "full-path d1 off probit at eta={eta} eps={eps}"
                );
            }
            // No cliff at ε=0: the fast-path value (ε exactly 0) and the full-path
            // values on either side agree to first order — the jump is O(ε), not
            // the old O(2e-4) surface gap.
            let center = sas_inverse_link_jet(eta, 0.0, 0.0).expect("fast path").mu;
            let lo = sas_inverse_link_jet(eta, -1e-11, 0.0).expect("full path").mu;
            let hi = sas_inverse_link_jet(eta, 1e-11, 0.0).expect("full path").mu;
            assert!(
                (hi - center).abs() < 1e-11 && (center - lo).abs() < 1e-11,
                "μ cliff across ε=0 at eta={eta}: lo={lo} center={center} hi={hi}"
            );
        }
    }

    /// #2389 design point (b): all-order FD gate on the interior-exact bounded
    /// latent map's jet tower, at interior / splice / saturation points. The map
    /// underpins every SAS evaluation, yet the SAS-level tests only reach its
    /// interior (the splice needs `|δ·asinh(η)+ε| ∈ (0.8B, 1.2B)`, i.e. η≈1e17).
    /// This pins `smooth_bound_jet` directly: exact identities in the two flat
    /// regimes and at the seams, odd symmetry, non-expansiveness, and a
    /// derivative ladder (`d_{k} = d/dx d_{k-1}`) through sixth order in the
    /// splice — where a wrong smoothstep coefficient would otherwise hide.
    #[test]
    fn smooth_bound_jet_tower_is_c6_and_fd_exact() {
        let b = sas_latent_domain_bound();
        let a = SPLICE_INTERIOR_FRAC * b;
        let c = (2.0 - SPLICE_INTERIOR_FRAC) * b;
        let jet = |x: f64| smooth_bound_jet(x, b);

        // Interior |x| ≤ a: exact identity, every higher derivative exactly 0.
        for &x in &[0.0, 0.5, 17.3, a - 1e-9, a] {
            let j = jet(x);
            assert_eq!(j.g, x, "interior identity value at x={x}");
            assert_eq!(j.d1, 1.0, "interior d1 at x={x}");
            assert_eq!((j.d2, j.d3, j.d4, j.d5, j.d6), (0.0, 0.0, 0.0, 0.0, 0.0));
        }
        // Saturation |x| ≥ c: exact ±B plateau, every derivative exactly 0.
        for &x in &[c, c + 1e-9, 75.0, 1e12, f64::MAX] {
            let j = jet(x);
            assert_eq!(j.g, b, "saturation value at x={x}");
            assert_eq!(
                (j.d1, j.d2, j.d3, j.d4, j.d5, j.d6),
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            );
        }
        // Seam value + compactness: g(c) = B exactly (the c = 2B − a closure).
        assert_eq!(jet(c).g, b);

        // Odd symmetry: g,d2,d4,d6 flip sign; d1,d3,d5 are even.
        for &x in &[3.0, a - 2.0, 44.0, 50.0, 56.0, 100.0] {
            let p = jet(x);
            let m = jet(-x);
            assert_eq!(m.g, -p.g, "g odd at x={x}");
            assert_eq!(m.d1, p.d1, "d1 even at x={x}");
            assert_eq!(m.d2, -p.d2, "d2 odd at x={x}");
            assert_eq!(m.d3, p.d3, "d3 even at x={x}");
            assert_eq!(m.d4, -p.d4, "d4 odd at x={x}");
            assert_eq!(m.d5, p.d5, "d5 even at x={x}");
            assert_eq!(m.d6, -p.d6, "d6 odd at x={x}");
        }

        // Non-expansive + monotone + bounded across the whole range.
        for i in 0..=240 {
            let x = -80.0 + 160.0 * (i as f64) / 240.0;
            let j = jet(x);
            assert!((0.0..=1.0).contains(&j.d1), "d1 out of [0,1] at x={x}: {}", j.d1);
            assert!(j.g.abs() <= b + 1e-12, "|g| exceeds B at x={x}: {}", j.g);
        }

        // FD derivative ladder through sixth order, at splice-INTERIOR points
        // (kept ≥ 5 away from both seams so the O(h²) truncation stays small and
        // the h-stencil never straddles a regime change). d_k must be the
        // x-derivative of d_{k-1}; per-order tolerances track the realistic
        // central-difference error, still orders of magnitude below the O(1)
        // shift any wrong smoothstep coefficient would produce.
        let h = 0.02;
        for &x0 in &[45.0, 47.5, 50.0, 52.5, 55.0, -47.5, -52.5] {
            let jp = jet(x0 + h);
            let jm = jet(x0 - h);
            let j0 = jet(x0);
            let fd = |hi: f64, lo: f64| (hi - lo) / (2.0 * h);
            let checks = [
                ("d1", j0.d1, fd(jp.g, jm.g), 2e-6),
                ("d2", j0.d2, fd(jp.d1, jm.d1), 2e-5),
                ("d3", j0.d3, fd(jp.d2, jm.d2), 2e-4),
                ("d4", j0.d4, fd(jp.d3, jm.d3), 2e-3),
                ("d5", j0.d5, fd(jp.d4, jm.d4), 2e-2),
                ("d6", j0.d6, fd(jp.d5, jm.d5), 2e-1),
            ];
            for (name, analytic, numeric, tol) in checks {
                assert!(
                    (analytic - numeric).abs() < tol * (1.0 + analytic.abs()),
                    "smooth_bound {name} FD mismatch at x={x0}: analytic={analytic:e} fd={numeric:e}"
                );
            }
        }
    }

    /// #2389 general-case follow-up: the cancellation-free SAS complement must
    /// live on the SAME latent surface as the forward inverse-link, so that
    /// `μ + (1−μ) = 1` holds to full precision across the ENTIRE finite-`f64`
    /// eta domain — including `|η| > 1.34e154`, where `η·η` overflows.
    ///
    /// The forward map routes `asinh` through the overflow-free [`asinh_jet6`]
    /// (`hypot`-based value with an asymptotic `ln|η|+ln2` fallback), but
    /// `sas_link_complement` used the raw `f64::asinh`, whose internal `x·x`
    /// overflows to `+∞` near the domain edge. With a compressing `δ<1` the true
    /// latent `δ·asinh(η)` stays deep in the map's identity interior (finite,
    /// unsaturated μ), yet the overflowing complement drove `u_raw→∞`, saturated
    /// to `±B`, and returned `Φ(∓sinh B)=0` — a full ~0.15 disagreement with the
    /// forward `1−μ`, silently poisoning any `log(1−μ)` tail term at extreme η.
    #[test]
    fn sas_link_complement_mirrors_forward_map_across_finite_domain() {
        let params = [
            (0.0, 0.0),     // fast probit reduction
            (0.25, -0.35),  // generic interior skew/scale
            (0.0, -6.0),    // strong compression δ≈2.5e-3
            (-0.5, -6.0),
            (0.3, 6.0),     // strong dilation δ≈4e2
        ];
        let etas = [
            -f64::MAX,
            -1e160, // η·η overflows here; asinh(η)≈-369 is finite and in range
            -1e6,
            -3.0,
            -0.2,
            0.0,
            0.4,
            3.0,
            1e6,
            1e160,
            f64::MAX,
        ];
        for &(epsilon, log_delta) in &params {
            for &eta in &etas {
                let mu = sas_inverse_link_jet(eta, epsilon, log_delta)
                    .expect("finite SAS eta")
                    .mu;
                let complement = sas_link_complement(eta, epsilon, log_delta, mu);
                assert!(
                    complement.is_finite() && (0.0..=1.0).contains(&complement),
                    "SAS complement out of [0,1] at η={eta} ε={epsilon} log_δ={log_delta}: {complement}"
                );
                // μ + (1−μ) = 1 on one consistent surface. Both endpoints are
                // Φ(±z) for the same z, so equality holds to erfc round-off.
                let sum = mu + complement;
                assert!(
                    (sum - 1.0).abs() < 1e-12,
                    "SAS complement off the forward surface at η={eta} ε={epsilon} \
                     log_δ={log_delta}: μ={mu} complement={complement} μ+comp={sum}"
                );
            }
        }
    }

    /// #2902 row 8: the outer box on a coordinate charted by `smooth_bound_jet` is
    /// the map's own support. Just inside its edge the map still moves; at the
    /// edge and beyond it is exactly `±B` with a zero slope, so every point outside
    /// repeats a value the box already holds.
    #[test]
    fn a_bounded_map_support_ends_where_the_map_stops_moving_2902() {
        for bound in [sas_log_delta_domain_bound(), sas_latent_domain_bound()] {
            let (lower, upper) = smooth_bound_support(bound);
            assert_eq!(lower, -upper);
            let inside = smooth_bound_jet(0.99 * upper, bound);
            assert!(
                inside.d1 > 0.0,
                "bound={bound}: the map must still move just inside its support, d1={}",
                inside.d1
            );
            for beyond in [upper, 2.0 * upper] {
                let saturated = smooth_bound_jet(beyond, bound);
                assert_eq!(saturated.g, bound, "bound={bound} x={beyond}");
                assert_eq!(saturated.d1, 0.0, "bound={bound} x={beyond}");
                assert_eq!(
                    smooth_bound_jet(-beyond, bound).g,
                    -bound,
                    "bound={bound} x=-{beyond}"
                );
            }
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

    /// Central differences of `f‴` over each shape parameter of `link`, at a
    /// spread of η, against [`inverse_link_pdfthird_derivative_param_partials`].
    fn assert_pdfthird_param_partials_match_fd_2904(
        label: &str,
        link: &InverseLink,
        perturbed: impl Fn(usize, f64) -> InverseLink,
    ) {
        let h = 1e-6;
        for eta in [-1.3, -0.2, 0.45, 2.1] {
            let analytic = inverse_link_pdfthird_derivative_param_partials(link, eta)
                .expect("finite eta")
                .expect("a shape-parameterized link has partials");
            assert!(!analytic.is_empty(), "{label}: no shape axes");
            for (axis, &value) in analytic.iter().enumerate() {
                let plus = inverse_link_pdfthird_derivative_for_inverse_link(&perturbed(axis, h), eta)
                    .expect("finite eta");
                let minus =
                    inverse_link_pdfthird_derivative_for_inverse_link(&perturbed(axis, -h), eta)
                        .expect("finite eta");
                let finite_difference = (plus - minus) / (2.0 * h);
                assert!(
                    (value - finite_difference).abs() <= 1e-5 * finite_difference.abs().max(1.0),
                    "{label} eta={eta} axis {axis}: analytic={value}, finite difference={finite_difference}"
                );
            }
        }
    }

    /// #2902 row 34 (i): at the logit shapes `a = b = 1` the standardization is
    /// exactly inert. `E Z` is exactly `0.0` and `s` exactly `1.0`, so the link's
    /// jet and higher derivatives are the latent kernel's, bit for bit.
    #[test]
    fn beta_logistic_standardization_is_inert_at_the_logit_shapes_2902() {
        let standardization = beta_logistic_standardization(0.7, 1.0, 1.0);
        assert_eq!(standardization.scale.to_bits(), 1.0_f64.to_bits());
        assert_eq!(standardization.latent.to_bits(), 0.7_f64.to_bits());
        let (latent, scale) = beta_logistic_latent_argument(0.7, 1.0, 1.0);
        assert_eq!(latent.to_bits(), standardization.latent.to_bits());
        assert_eq!(scale.to_bits(), standardization.scale.to_bits());
        for eta in [-40.0_f64, -5.0, -0.3, 0.42, 5.0, 40.0] {
            let link = beta_logistic_inverse_link_jet(eta, 0.0, 0.0);
            let kernel = beta_logistic_latent_jet(eta, 1.0, 1.0);
            assert_eq!(link.mu.to_bits(), kernel.mu.to_bits(), "eta={eta}");
            assert_eq!(link.d1.to_bits(), kernel.d1.to_bits(), "eta={eta}");
            assert_eq!(link.d2.to_bits(), kernel.d2.to_bits(), "eta={eta}");
            assert_eq!(link.d3.to_bits(), kernel.d3.to_bits(), "eta={eta}");
            assert_eq!(
                beta_logistic_inverse_link_pdfthird_derivative(eta, 0.0, 0.0).to_bits(),
                beta_logistic_latent_pdfthird_derivative(eta, 1.0, 1.0).to_bits(),
                "eta={eta}"
            );
            assert_eq!(
                beta_logistic_inverse_link_pdffourth_derivative(eta, 0.0, 0.0).to_bits(),
                beta_logistic_latent_pdffourth_derivative(eta, 1.0, 1.0).to_bits(),
                "eta={eta}"
            );
        }
    }

    /// #2902 row 34 (iii): every derivative the outer search and its certificate read
    /// matches central differences at shapes far from `a = b = 1`, where `E Z` and
    /// `s` move with the parameters. That covers the η-jet through the density's
    /// fourth derivative, the first and second parameter partials of `μ`, `d1`, `d2`
    /// (and the first of `d3`), and the parameter partials of the density's third
    /// derivative.
    ///
    /// At `h = 1e-5` a central difference carries `O(h²·f‴)` truncation and
    /// `ε·|f|/h ≈ 2e-11·|f|` rounding, both below the `1e-6` relative (plus `1e-9`
    /// absolute) band.
    #[test]
    fn standardized_beta_logistic_derivatives_match_central_differences_2902() {
        let h = 1.0e-5;
        let close = |analytic: f64, numeric: f64, what: &str| {
            let band = 1.0e-9 + 1.0e-6 * analytic.abs().max(numeric.abs());
            assert!(
                (analytic - numeric).abs() <= band,
                "{what}: analytic={analytic:e} central difference={numeric:e} band={band:e}"
            );
        };
        for (log_shape_center, epsilon) in [(-2.5_f64, 0.8_f64), (3.0, -1.0), (0.6, 0.55)] {
            for eta in [-3.0_f64, -0.4, 1.7] {
                let label = format!("(l={log_shape_center}, e={epsilon}, eta={eta})");
                let jet = |e: f64| beta_logistic_inverse_link_jet(e, log_shape_center, epsilon);
                let (centre, plus, minus) = (jet(eta), jet(eta + h), jet(eta - h));
                close(centre.d1, (plus.mu - minus.mu) / (2.0 * h), &format!("d1 {label}"));
                close(centre.d2, (plus.d1 - minus.d1) / (2.0 * h), &format!("d2 {label}"));
                close(centre.d3, (plus.d2 - minus.d2) / (2.0 * h), &format!("d3 {label}"));
                let f4 = |e: f64| {
                    beta_logistic_inverse_link_pdfthird_derivative(e, log_shape_center, epsilon)
                };
                close(f4(eta), (plus.d3 - minus.d3) / (2.0 * h), &format!("d4 {label}"));
                close(
                    beta_logistic_inverse_link_pdffourth_derivative(eta, log_shape_center, epsilon),
                    (f4(eta + h) - f4(eta - h)) / (2.0 * h),
                    &format!("d5 {label}"),
                );

                let out = beta_logistic_inverse_link_jetwith_param_partials(
                    eta,
                    log_shape_center,
                    epsilon,
                );
                // Axis 0 is `epsilon`, axis 1 is `log_shape_center`.
                let at = |axis: usize, step: f64| -> (f64, f64) {
                    if axis == 0 {
                        (log_shape_center, epsilon + step)
                    } else {
                        (log_shape_center + step, epsilon)
                    }
                };
                let first = [out.djet_depsilon, out.djet_dlog_delta];
                let f4_partials = beta_logistic_inverse_link_pdfthird_derivative_param_partials(
                    eta,
                    log_shape_center,
                    epsilon,
                );
                for axis in 0..2 {
                    let (lp, ep) = at(axis, h);
                    let (lm, em) = at(axis, -h);
                    let jp = beta_logistic_inverse_link_jet(eta, lp, ep);
                    let jm = beta_logistic_inverse_link_jet(eta, lm, em);
                    let what = |q: &str| format!("d{q}/dtheta{axis} {label}");
                    close(first[axis].mu, (jp.mu - jm.mu) / (2.0 * h), &what("mu"));
                    close(first[axis].d1, (jp.d1 - jm.d1) / (2.0 * h), &what("d1"));
                    close(first[axis].d2, (jp.d2 - jm.d2) / (2.0 * h), &what("d2"));
                    close(first[axis].d3, (jp.d3 - jm.d3) / (2.0 * h), &what("d3"));
                    close(
                        f4_partials[axis],
                        (beta_logistic_inverse_link_pdfthird_derivative(eta, lp, ep)
                            - beta_logistic_inverse_link_pdfthird_derivative(eta, lm, em))
                            / (2.0 * h),
                        &what("f4"),
                    );
                    let op = beta_logistic_inverse_link_jetwith_param_partials(eta, lp, ep);
                    let om = beta_logistic_inverse_link_jetwith_param_partials(eta, lm, em);
                    let (fp, fm) = (
                        [op.djet_depsilon, op.djet_dlog_delta],
                        [om.djet_depsilon, om.djet_dlog_delta],
                    );
                    for other in 0..2 {
                        let pair = |q: &str| format!("d2{q}/dtheta{axis}dtheta{other} {label}");
                        close(
                            out.d2mu_dparams2[[other, axis]],
                            (fp[other].mu - fm[other].mu) / (2.0 * h),
                            &pair("mu"),
                        );
                        close(
                            out.d2d1_dparams2[[other, axis]],
                            (fp[other].d1 - fm[other].d1) / (2.0 * h),
                            &pair("d1"),
                        );
                        close(
                            out.d2d2_dparams2[[other, axis]],
                            (fp[other].d2 - fm[other].d2) / (2.0 * h),
                            &pair("d2"),
                        );
                    }
                }
            }
        }
    }

    /// #2904: the θ-partials of the density's third derivative match central
    /// differences for SAS, Beta-Logistic and a mixture, and a link without shape
    /// parameters has none.
    #[test]
    fn pdfthird_derivative_param_partials_match_fd_2904() {
        let sas = |epsilon: f64, log_delta: f64| {
            InverseLink::Sas(
                state_from_sasspec(gam_problem::SasLinkSpec {
                    initial_epsilon: epsilon,
                    initial_log_delta: log_delta,
                })
                .expect("sas state"),
            )
        };
        let (epsilon, log_delta) = (-0.12, 0.21);
        assert_pdfthird_param_partials_match_fd_2904("sas", &sas(epsilon, log_delta), |axis, step| {
            if axis == 0 {
                sas(epsilon + step, log_delta)
            } else {
                sas(epsilon, log_delta + step)
            }
        });

        let beta_logistic = |epsilon: f64, log_delta: f64| {
            InverseLink::BetaLogistic(
                state_from_beta_logisticspec(gam_problem::SasLinkSpec {
                    initial_epsilon: epsilon,
                    initial_log_delta: log_delta,
                })
                .expect("beta-logistic state"),
            )
        };
        let (b_epsilon, b_log_delta) = (-0.17, 0.23);
        assert_pdfthird_param_partials_match_fd_2904(
            "beta-logistic",
            &beta_logistic(b_epsilon, b_log_delta),
            |axis, step| {
                if axis == 0 {
                    beta_logistic(b_epsilon + step, b_log_delta)
                } else {
                    beta_logistic(b_epsilon, b_log_delta + step)
                }
            },
        );

        let components = vec![LinkComponent::Probit, LinkComponent::Logit, LinkComponent::CLogLog];
        let rho0 = Array1::from_vec(vec![0.3, -0.6]);
        let mixture = |rho: Array1<f64>| {
            InverseLink::Mixture(
                state_fromspec(&MixtureLinkSpec {
                    components: components.clone(),
                    initial_rho: rho,
                })
                .expect("mixture state"),
            )
        };
        assert_pdfthird_param_partials_match_fd_2904("mixture", &mixture(rho0.clone()), |axis, step| {
            let mut rho = rho0.clone();
            rho[axis] += step;
            mixture(rho)
        });

        assert!(
            inverse_link_pdfthird_derivative_param_partials(
                &InverseLink::Standard(gam_problem::StandardLink::Probit),
                0.3,
            )
            .expect("finite eta")
            .is_none()
        );
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
        // The link reads the latent kernel at the standardized `x = E Z + s·η`
        // (#2902 row 34); at `x ≈ −34.9`, `exp(x)` is `logistic(x)` to one part in
        // `e^{−34.9}`.
        let (x, s) = beta_logistic_latent_argument(eta, a, b);
        assert!(x < -30.0 && s > 0.0, "the fixture must sit deep in the left tail: x={x}");
        let expected_mu = beta_reg(a, b, x.exp());
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
    fn composed_sas_derivatives_reproduce_the_hand_arbogast_orders_3203() {
        // `compose_derivatives6` is the Faà di Bruno sum the SAS fifth and sixth
        // orders are built from; its orders 1..4 must be the hand-expanded
        // Arbogast chains of the SAS jet and `f‴` to rounding.
        for (epsilon, log_delta) in [(-0.25, 0.35), (0.4, -0.3), (0.0, 0.0)] {
            let state = sas_link_state_from_raw(epsilon, log_delta).expect("sas state");
            for eta in [-3.1, -0.8, 0.0, 0.45, 2.2] {
                let composed = sas_inverse_link_derivatives6(eta, state.epsilon, state.log_delta)
                    .expect("composed SAS derivatives");
                let jet = sas_inverse_link_jet(eta, state.epsilon, state.log_delta)
                    .expect("SAS jet");
                let fourth = |x: f64| {
                    sas_inverse_link_pdfthird_derivative(x, state.epsilon, state.log_delta)
                        .expect("SAS fourth derivative")
                };
                let hand = [jet.d1, jet.d2, jet.d3, fourth(eta)];
                for (order, (&got, &want)) in composed.iter().zip(hand.iter()).enumerate() {
                    assert!(
                        (got - want).abs() <= 1e-12 * (1.0 + want.abs()),
                        "SAS order {} at eta={eta}, (eps, log_delta)=({epsilon}, {log_delta}): \
                         composed {got:e}, hand {want:e}",
                        order + 1
                    );
                }
                // Order five has no hand chain of its own: it is the eta-slope of
                // the hand `f⁗`. The central difference errs by `h²|μ⁽⁷⁾|/6` plus
                // `ε|μ⁽⁴⁾|/h` of rounding, both far inside the band at `h = 1e-4`.
                let h = 1e-4;
                let fd = (fourth(eta + h) - fourth(eta - h)) / (2.0 * h);
                assert!(
                    (composed[4] - fd).abs() <= 1e-6 * (1.0 + composed[4].abs()),
                    "SAS order 5 at eta={eta}, (eps, log_delta)=({epsilon}, {log_delta}): \
                     composed {:e}, fd of the hand fourth {fd:e}",
                    composed[4]
                );
            }
        }
    }

    #[test]
    fn inverse_link_pdffifth_derivative_matches_fd_of_the_fourth_3203() {
        // `μ⁽⁶⁾` is the top order of the Bernoulli log jet behind the Firth TK
        // outer ρ-Hessian (#3203). Each link's closed form must be the η-slope
        // of its own `μ⁽⁵⁾`. The central difference errs by `h²|μ⁽⁷⁾|/6`
        // plus `ε|μ⁽⁵⁾|/h` of rounding, both far inside the band at `h = 1e-4`.
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
                    LinkComponent::LogLog,
                    LinkComponent::Cauchit,
                ],
                initial_rho: Array1::from_vec(vec![0.35, -0.45, 0.2, -0.1]),
            })
            .expect("mixture state"),
        );
        let latent = InverseLink::LatentCLogLog(
            gam_problem::types::LatentCLogLogState::new(0.4).expect("valid latent SD"),
        );
        let links = [
            InverseLink::Standard(StandardLink::Logit),
            InverseLink::Standard(StandardLink::Probit),
            InverseLink::Standard(StandardLink::Cauchit),
            InverseLink::Standard(StandardLink::CLogLog),
            InverseLink::Standard(StandardLink::LogLog),
            sas,
            beta_logistic,
            mixture,
            latent,
        ];
        let h = 1e-4;
        let mut failures = Vec::new();
        for link in &links {
            for eta in [-2.3, -1.1, -0.2, 0.35, 0.6, 1.7] {
                let fifth = |x: f64| {
                    inverse_link_pdffourth_derivative_for_inverse_link(link, x).expect("mu^(5)")
                };
                let fd = (fifth(eta + h) - fifth(eta - h)) / (2.0 * h);
                let sixth = inverse_link_pdffifth_derivative_for_inverse_link(link, eta)
                    .expect("mu^(6)");
                if !((sixth - fd).abs() <= 1e-6 * (1.0 + sixth.abs())) {
                    failures.push(format!("{link:?} eta={eta}: analytic {sixth:e}, fd {fd:e}"));
                }
            }
        }
        assert!(failures.is_empty(), "#3203:\n  {}", failures.join("\n  "));
    }

    /// The latent-cloglog complement is the survival `S(eta, σ_L)`, not
    /// `1 − mu`: at `(eta, σ_L) = (8, 0.5)` the mean rounds to one while
    /// `ln S = −70.97988851759840` (the #2714 high-precision reference row, which
    /// `ln S` meets to `1e-13` relative).
    #[test]
    fn latent_cloglog_complement_keeps_the_survival_tail() {
        let link = InverseLink::LatentCLogLog(
            gam_problem::types::LatentCLogLogState::new(0.5).expect("valid latent SD"),
        );
        let (mu, _) = inverse_link_mu_d1_for_inverse_link(&link, 8.0).expect("latent jet");
        assert_eq!(mu, 1.0, "the mean saturates, so 1 - mu carries nothing");
        let complement = inverse_link_complement_for_inverse_link(&link, 8.0, mu);
        assert!(
            complement > 0.0,
            "the complement must keep the representable survival tail, got {complement:e}"
        );
        let reference_log_survival = -7.097_988_851_759_84e1;
        let relative =
            (complement.ln() - reference_log_survival).abs() / reference_log_survival.abs();
        assert!(
            relative <= 1.0e-12,
            "ln complement = {:.17e}, reference {reference_log_survival:.17e} \
             (relative {relative:.3e})",
            complement.ln()
        );

        // Where the mean does not saturate, the complement and the mean share
        // one `ln S`, so they add to one up to the rounding of each.
        let (mu, _) = inverse_link_mu_d1_for_inverse_link(&link, 0.35).expect("latent jet");
        let complement = inverse_link_complement_for_inverse_link(&link, 0.35, mu);
        assert!(
            (mu + complement - 1.0).abs() <= 4.0 * f64::EPSILON,
            "mu {mu:.17e} + complement {complement:.17e} must be one"
        );
        assert!(inverse_link_complement_for_inverse_link(&link, f64::NAN, f64::NAN).is_nan());
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

#[cfg(test)]
mod probit_density_tail_tests {
    use super::*;

    #[test]
    fn probit_derivatives_survive_underflow_of_the_unweighted_density() {
        for x in [-38.6_f64, 38.6] {
            assert_eq!(normal_pdf(x), 0.0);
            let jet = probit_jet(x);
            let actual = [jet.d2, jet.d3, probit_pdfthird_derivative(x), probit_pdffourth_derivative(x)];
            let factors = [-x, x * x - 1.0, -x * (x * x - 3.0), x.powi(4) - 6.0 * x * x + 3.0];
            for (&derivative, &factor) in actual.iter().zip(&factors) {
                let expected = factor.signum()
                    * ((factor.abs() / std::f64::consts::TAU.sqrt()).ln() - x * (0.5 * x)).exp();
                assert!(derivative != 0.0 && derivative.is_finite());
                assert!((derivative - expected).abs() <= f64::from_bits(2));
            }
        }
        for x in [-f64::MAX, f64::MAX] {
            let jet = probit_jet(x);
            assert_eq!([jet.d1, jet.d2, jet.d3], [0.0; 3]);
            assert_eq!(probit_pdfthird_derivative(x), 0.0);
            assert_eq!(probit_pdffourth_derivative(x), 0.0);
        }
    }
}
