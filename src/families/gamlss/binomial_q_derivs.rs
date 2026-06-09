//! Closed-form and jet-based derivatives of the binomial negative log-likelihood
//! in the latent-position coordinate `q`.
//!
//! For the binomial location-scale family the per-row loss is
//!   F_i(q) = -w_i[y_i log G(q) + (1 - y_i) log(1 - G(q))],
//! where `G` is the inverse link (probit / logit / cloglog / generic) and `q`
//! is the latent position. The exact joint Newton calculus consumes the first
//! through fourth derivatives `m1..m4 = d^kF/dq^k`.
//!
//! Probit, logit, and cloglog have stable closed forms (the fast path); any
//! other link falls back to the generic inverse-link jet plus the analytic
//! fourth derivative of the inverse-link pdf. All functions here are pure.

use crate::mixture_link::inverse_link_pdfthird_derivative_for_inverse_link;
use crate::probability::signed_probit_logcdf_and_mills_ratio;
use crate::types::{InverseLink, StandardLink};

use super::MIN_PROB;

#[inline]
pub(super) fn binomial_score_curvaturethird_from_jet(
    y: f64,
    weight: f64,
    mu: f64,
    d1: f64,
    d2: f64,
    d3: f64,
) -> (f64, f64, f64) {
    // Binomial derivatives wrt q via mu:
    // Per-row log-likelihood is represented in weighted-proportion form:
    //   ell_i = m_i * [ y_i log(mu_i) + (1-y_i) log(1-mu_i) ],
    // where `weight = m_i` and `y` is the observed proportion in [0,1].
    //
    // mu-space derivatives:
    //   ellmu    = y/mu - (1-y)/(1-mu)
    //   ellmumu  = -y/mu^2 - (1-y)/(1-mu)^2
    //   ellmumum = 2y/mu^3 - 2(1-y)/(1-mu)^3
    //
    // q-jet using mu(q) derivatives d1=mu', d2=mu'', d3=mu''':
    //   s = dell/dq   = ellmu * mu'
    //   c = d2ell/dq2 = ellmumu*(mu')^2 + ellmu*mu''
    //   t = d3ell/dq3 = ellmumum*(mu')^3 + 3*ellmumu*mu'*mu'' + ellmu*mu'''
    //
    // Returns (score_q, curvature_q, third_q) with curvature_q = -d2ell/dq2.
    let m = mu;
    let one_minus = 1.0 - m;
    let ellmu = y / m - (1.0 - y) / one_minus;
    let ellmumu = -y / (m * m) - (1.0 - y) / (one_minus * one_minus);
    let ellmumum = 2.0 * y / (m * m * m) - 2.0 * (1.0 - y) / (one_minus * one_minus * one_minus);

    let score_q = weight * ellmu * d1;
    let d2ell_dq2 = weight * (ellmumu * d1 * d1 + ellmu * d2);
    let curvature_q = -d2ell_dq2;
    let third_q = weight * (ellmumum * d1 * d1 * d1 + 3.0 * ellmumu * d1 * d2 + ellmu * d3);
    (score_q, curvature_q, third_q)
}

#[inline]
pub(super) fn binomial_neglog_q_derivatives_from_jet(
    y: f64,
    weight: f64,
    mu: f64,
    d1: f64,
    d2: f64,
    d3: f64,
) -> (f64, f64, f64) {
    // Returns (m1,m2,m3) for F_i(q) = -ell_i(q):
    //   m1 = dF/dq, m2 = d²F/dq², m3 = d³F/dq³.
    let (score_q, curvature_q, third_q) =
        binomial_score_curvaturethird_from_jet(y, weight, mu, d1, d2, d3);
    (-score_q, curvature_q, -third_q)
}

#[inline]
pub(super) fn binomial_neglog_q_derivatives_probit_closed_form(
    y: f64,
    weight: f64,
    q: f64,
) -> (f64, f64, f64) {
    // Closed-form derivatives for F_i(q) = -w_i[y log Phi(q) + (1-y) log(1-Phi(q))].
    // Uses stable Mills ratios instead of `phi / mu` divisions. In the
    // incompatible separated tail (for example y=0, q>>0), `phi(q)` underflows
    // to zero while `phi(q)/Phi(-q) ≈ q`; computing the ratio in log-CDF space
    // preserves the true score/curvature signal instead of manufacturing a
    // flat optimum.
    if weight == 0.0 || !q.is_finite() {
        return (0.0, 0.0, 0.0);
    }
    let (_, left) = signed_probit_logcdf_and_mills_ratio(q);
    let (_, right) = signed_probit_logcdf_and_mills_ratio(-q);

    let left_prime = -left * (q + left);
    let left_m2 = -left_prime;
    let left_m3 = left + left_prime * (q + 2.0 * left);

    let right_prime = right * (right - q);
    let right_m2 = right_prime;
    let right_m3 = right_prime * (2.0 * right - q) - right;

    let y0 = 1.0 - y;
    let m1 = weight * (y0 * right - y * left);
    let m2 = weight * (y0 * right_m2 + y * left_m2);
    let m3 = weight * (y0 * right_m3 + y * left_m3);
    (m1, m2, m3)
}

#[inline]
pub(super) fn binomial_neglog_q_fourth_derivative_probit_closed_form(
    y: f64,
    weight: f64,
    q: f64,
) -> f64 {
    // Closed-form m4 for F_i(q) = -w_i[y log Phi(q) + (1-y) log(1-Phi(q))].
    // Stability (Issue 5): see binomial_neglog_q_derivatives_probit_closed_form.
    if weight == 0.0 || !q.is_finite() {
        return 0.0;
    }
    let (_, left) = signed_probit_logcdf_and_mills_ratio(q);
    let (_, right) = signed_probit_logcdf_and_mills_ratio(-q);

    let left_prime = -left * (q + left);
    let left_m3 = left + left_prime * (q + 2.0 * left);
    let left_m4 = 2.0 * left_prime - left_m3 * (q + 2.0 * left) + 2.0 * left_prime * left_prime;

    let right_prime = right * (right - q);
    let right_m3 = right_prime * (2.0 * right - q) - right;
    let right_m4 =
        right_m3 * (2.0 * right - q) + 2.0 * right_prime * right_prime - 2.0 * right_prime;

    weight * ((1.0 - y) * right_m4 + y * left_m4)
}

// ---------------------------------------------------------------------------
// Logit closed-form m1–m4
// ---------------------------------------------------------------------------
//
// For the logit (sigmoid) inverse link, F(q) = -w[y log G(q) + (1-y) log(1-G(q))]
// where G(q) = 1/(1 + e^{-q}) is the standard logistic CDF.
//
// Because logit is the canonical link for Bernoulli, the derivatives of F
// collapse to especially simple closed forms in terms of p = G(q) and
// s = p(1-p) = Var(Bernoulli(p)):
//
//   m1 = w(p - y)
//   m2 = ws                           (always non-negative)
//   m3 = ws(1 - 2p) = -ws tanh(q/2)
//   m4 = ws(1 - 6s) = ws(1 - 6p + 6p^2)
//
// Derivation: since log G(q) = -log(1 + e^{-q}) and log(1 - G(q)) = -log(1 + e^q),
// F(q) = w[-y log G + (1-y)(-log(1-G))]
//       = w[y log(1+e^{-q}) + (1-y) log(1+e^q)]
//       = w[(1-y)q + log(1+e^{-q})]     (the standard softplus form).
//
// Differentiating: F' = w(G(q) - y) = w(p - y), which is m1.
// F'' = wG'(q) = wp(1-p) = ws, which is m2.
// F''' = w[p(1-p)(1-2p)] = ws(1-2p), which is m3.
// F'''' = w[s(1-6s)], which is m4. The identity 1-6s = 1-6p+6p^2 follows directly.
//
// Numerical stability (see response.md Section 1a):
// - p is computed with a branched expit to avoid overflow:
//     p = (1+e^{-q})^{-1} for q >= 0,  p = e^q/(1+e^q) for q < 0.
// - s = p(1-p). For extreme tails, s = t/(1+t)^2 with t = e^{-|q|}, which
//   decays as O(e^{-|q|}). Once |q| > ~36, e^{-|q|} < machine epsilon and
//   all derivatives are genuinely below precision, so saturation to 0 is safe.
// - The identity 1-2p = -tanh(q/2) provides a stable alternative for m3.
//
// Reference: response.md Section 1a.
// ---------------------------------------------------------------------------

#[inline]
pub(super) fn binomial_neglog_q_derivatives_logit_closed_form(
    y: f64,
    weight: f64,
    q: f64,
) -> (f64, f64, f64) {
    // Returns (m1, m2, m3) for F(q) = -w[y log G(q) + (1-y) log(1-G(q))]
    // with G = logistic CDF.
    if weight == 0.0 || !q.is_finite() {
        return (0.0, 0.0, 0.0);
    }
    // Branched expit for numerical stability:
    //   q >= 0: p = 1/(1+e^{-q}), avoids overflow in e^q
    //   q < 0:  p = e^q/(1+e^q),  avoids overflow in e^{-q}
    let p = if q >= 0.0 {
        1.0 / (1.0 + (-q).exp())
    } else {
        let eq = q.exp();
        eq / (1.0 + eq)
    };
    // Clamp `p` AND its complement `1 - p` separately so that the
    // saturated-boundary product `p_var * one_minus_p_var` equals the
    // mathematical `MIN_PROB · (1 − MIN_PROB)` exactly. Recomputing
    // `1 - p_var` after clamping `p` would catastrophically cancel near the
    // boundary (e.g. `1 - (1 − 1e-10)` yields `1.0000000827e-10`, not the
    // intended `1e-10`), inflating the variance by ~8e-18 — small in
    // absolute terms but enough to corrupt the Fisher information used by
    // the GAMLSS exact-Newton step in the deep tail.
    let p_var = p.clamp(MIN_PROB, 1.0 - MIN_PROB);
    let one_minus_p_var = (1.0 - p).clamp(MIN_PROB, 1.0 - MIN_PROB);
    let s = p_var * one_minus_p_var;
    // For extreme |q|, s settles at the clamped floor `MIN_PROB·(1−MIN_PROB)`
    // — never below — so the second-order Newton block stays bounded away
    // from zero curvature on saturated rows.

    let m1 = weight * (p - y);
    let m2 = weight * s;
    // m3 = ws(1 - 2p). Using the identity 1-2p = -tanh(q/2) for stability:
    let m3 = weight * s * (1.0 - 2.0 * p);
    (m1, m2, m3)
}

#[inline]
pub(super) fn binomial_neglog_q_fourth_derivative_logit_closed_form(
    x: f64,
    weight: f64,
    q: f64,
) -> f64 {
    assert!(!x.is_nan());
    // Returns m4 = d^4F/dq^4 for logit link.
    // m4 = ws(1 - 6s) = ws(1 - 6p(1-p)).
    //
    // Note: m4 does not depend on y at all (same as m2), because all
    // even-order derivatives of the canonical Bernoulli NLL are functions
    // of p alone. The y-dependence cancels out in the chain rule because
    // the logit is the canonical link.
    if weight == 0.0 || !q.is_finite() {
        return 0.0;
    }
    let p = if q >= 0.0 {
        1.0 / (1.0 + (-q).exp())
    } else {
        let eq = q.exp();
        eq / (1.0 + eq)
    };
    // Same cancellation-free `p · (1 − p)` form as
    // `binomial_neglog_q_derivatives_logit_closed_form` above — see the
    // note there.
    let p_var = p.clamp(MIN_PROB, 1.0 - MIN_PROB);
    let one_minus_p_var = (1.0 - p).clamp(MIN_PROB, 1.0 - MIN_PROB);
    let s = p_var * one_minus_p_var;
    weight * s * (1.0 - 6.0 * s)
}

// ---------------------------------------------------------------------------
// CLogLog / Gumbel closed-form m1–m4
// ---------------------------------------------------------------------------
//
// For the complementary log-log link, G(q) = 1 - exp(-exp(q)), so
// F(q) = -w[y log G(q) + (1-y) log(1-G(q))].
//
// Define:
//   z = e^q            (the "inner exponential")
//   r = e^{-z}         (survival probability 1 - G(q))
//   p = 1 - r = G(q) = -expm1(-z)
//   h = z / expm1(z) = z*r / p
//
// The ratio h is the key stable building block. It arises because the
// y=1 branch of the loss is F_{y=1} = -w log(1 - e^{-z}), and differentiating
// log(-expm1(-z)) w.r.t. q produces factors of z*e^{-z}/(1 - e^{-z}) = z*r/p = h.
// The function h = z/(e^z - 1) is smooth on all of R, with h -> 1 as z -> 0
// (removable singularity), and h -> z*e^{-z} -> 0 as z -> +inf.
//
// For y=0, the loss is simply F_{y=0} = w*e^q = w*z, so all derivatives are w*z.
//
// For y=1, the derivatives in the "h-form" (from response.md Section 1b) are:
//   F'_{y=1}    = -wh
//   F''_{y=1}   = wh(h + z - 1)
//   F'''_{y=1}  = -wh(2h^2 + 3(z-1)h + z^2 - 3z + 1)
//   F''''_{y=1} = wh(6h^3 + 12(z-1)h^2 + (7z^2 - 18z + 7)h + z^3 - 6z^2 + 7z - 1)
//
// For general y in [0,1], combining linearly:
//   m1 = w[(1-y)z - yh]
//   m2 = w[(1-y)z + yh(h + z - 1)]
//   m3 = w[(1-y)z - yh(2h^2 + 3(z-1)h + z^2 - 3z + 1)]
//   m4 = w[(1-y)z + yh(6h^3 + 12(z-1)h^2 + (7z^2 - 18z + 7)h + z^3 - 6z^2 + 7z - 1)]
//
// Numerical stability (see response.md Section 1b):
//
// Left tail (q << 0, z small):
//   p = -expm1(-z) avoids cancellation in 1 - e^{-z} when z is tiny.
//   h = z/expm1(z) is computed directly via expm1; no separate Taylor branch
//   is strictly necessary because expm1 is accurate for small arguments.
//   As z -> 0, h -> 1 - z/2 + z^2/12 - z^4/720 + O(z^6).
//
// Right tail (q >> 0, z > 36.7):
//   r = e^{-z} underflows to 0, so p rounds to 1. In this regime,
//   h = z*r/(1-r) ≈ z*r, which gracefully underflows to 0.
//   For y=1, all four derivatives -> 0. For y=0, they equal w*z.
//   The overflow boundary for e^z is z ≈ 709, i.e. q ≈ 6.56. Beyond that,
//   we must not compute e^z directly; instead h = z*r/p with r = e^{-z}.
//
// Reference: response.md Section 1b.
// ---------------------------------------------------------------------------

#[inline]
pub(super) fn cloglog_stable_h(z: f64) -> f64 {
    // Compute h = z / expm1(z) = z / (e^z - 1) = z * e^{-z} / (1 - e^{-z}).
    //
    // This is the fundamental stable building block for cloglog derivatives.
    // It has a removable singularity at z=0 where h -> 1.
    //
    // For large z (z > 36.7), expm1(z) overflows to infinity, but z*e^{-z}
    // is tiny and the derivatives are negligible. We use the identity
    // h = z * r / p where r = e^{-z} and p = -expm1(-z) for all z, which
    // is stable across the full range because:
    //   - For z near 0: expm1(z) is accurate, so z/expm1(z) is fine.
    //   - For large z: r = e^{-z} -> 0, making h -> 0 as well.
    if z.abs() < 1e-12 {
        // Taylor: h = 1 - z/2 + z^2/12 - z^4/720 + ...
        return 1.0 - z * 0.5 + z * z / 12.0;
    }
    let expm1_z = z.exp_m1();
    if expm1_z.is_infinite() {
        // z is very large (> ~709), e^z overflows. h = z*e^{-z}/(1 - e^{-z}).
        // Since z > 709, e^{-z} is essentially 0, so h ≈ 0.
        let r = (-z).exp();
        if r == 0.0 {
            return 0.0;
        }
        return z * r / (1.0 - r);
    }
    z / expm1_z
}

#[inline]
pub(super) fn binomial_neglog_q_derivatives_cloglog_closed_form(
    y: f64,
    weight: f64,
    q: f64,
) -> (f64, f64, f64) {
    // Returns (m1, m2, m3) for F(q) = -w[y log G(q) + (1-y) log(1-G(q))]
    // with G = cloglog CDF: G(q) = 1 - exp(-exp(q)).
    if weight == 0.0 || !q.is_finite() {
        return (0.0, 0.0, 0.0);
    }
    let z = q.exp(); // z = e^q; may be large but that's handled below
    let h = cloglog_stable_h(z);
    let y0 = 1.0 - y;
    let y0_term = if y0 == 0.0 { 0.0 } else { y0 * z };

    // y=0 branch: all derivatives equal w*z (since F_{y=0} = w*e^q).
    // y=1 branch: uses the h-polynomial forms.
    // General y: linear combination.
    //
    // Once h rounds to 0, the y=1 contribution has already underflowed to 0
    // in f64. Returning the remaining y=0 branch here avoids 0 * inf products
    // when q is deep in the right tail.
    if y == 0.0 || h == 0.0 {
        let base = weight * y0_term;
        return (base, base, base);
    }

    let m1 = weight * (y0_term - y * h);
    let m2 = weight * (y0_term + y * h * (h + z - 1.0));
    let m3 =
        weight * (y0_term - y * h * (2.0 * h * h + 3.0 * (z - 1.0) * h + z * z - 3.0 * z + 1.0));
    (m1, m2, m3)
}

#[inline]
pub(super) fn binomial_neglog_q_fourth_derivative_cloglog_closed_form(
    y: f64,
    weight: f64,
    q: f64,
) -> f64 {
    // Returns m4 = d^4F/dq^4 for cloglog link.
    // m4 = w[(1-y)z + yh(6h^3 + 12(z-1)h^2 + (7z^2-18z+7)h + z^3-6z^2+7z-1)]
    if weight == 0.0 || !q.is_finite() {
        return 0.0;
    }
    let z = q.exp();
    let h = cloglog_stable_h(z);
    let y0 = 1.0 - y;
    let y0_term = if y0 == 0.0 { 0.0 } else { y0 * z };
    if y == 0.0 || h == 0.0 {
        return weight * y0_term;
    }
    let h2 = h * h;
    let h3 = h2 * h;
    let z2 = z * z;
    let z3 = z2 * z;
    let y1_poly = 6.0 * h3 + 12.0 * (z - 1.0) * h2 + (7.0 * z2 - 18.0 * z + 7.0) * h + z3
        - 6.0 * z2
        + 7.0 * z
        - 1.0;
    weight * (y0_term + y * h * y1_poly)
}

#[inline]
pub(super) fn binomial_neglog_q_fourth_derivative_from_jet(
    y: f64,
    weight: f64,
    mu: f64,
    d1: f64,
    d2: f64,
    d3: f64,
    d4: f64,
) -> f64 {
    // Stability (Issue 5): floor μ inside divisions but allow the chain
    // rule to propagate; non-finite inputs still short-circuit (the LM
    // gain-ratio guard rejects non-finite candidate gradients).
    if weight == 0.0
        || !mu.is_finite()
        || !d1.is_finite()
        || !d2.is_finite()
        || !d3.is_finite()
        || !d4.is_finite()
    {
        return 0.0;
    }
    let m = mu.clamp(MIN_PROB, 1.0 - MIN_PROB);
    let one_minus = 1.0 - m;
    let ellmu = y / m - (1.0 - y) / one_minus;
    let ellmumu = -y / (m * m) - (1.0 - y) / (one_minus * one_minus);
    let ellmumum = 2.0 * y / (m * m * m) - 2.0 * (1.0 - y) / (one_minus * one_minus * one_minus);
    let ellmumumum = -6.0 * y / m.powi(4) - 6.0 * (1.0 - y) / one_minus.powi(4);
    let fourth_q = weight
        * (ellmumumum * d1.powi(4)
            + 6.0 * ellmumum * d1 * d1 * d2
            + ellmumu * (3.0 * d2 * d2 + 4.0 * d1 * d3)
            + ellmu * d4);
    -fourth_q
}

// ---------------------------------------------------------------------------
// Unified exact dispatch for binomial m1–m4
// ---------------------------------------------------------------------------
//
// Closed forms remain the fast path for Probit, Logit, and CLogLog, but the
// exact joint Newton calculus is not restricted to those links. When no
// closed form is available, we use the generic inverse-link jet plus the
// analytic third derivative of the inverse-link pdf (f''' = μ'''', the fourth
// derivative of the inverse-link CDF).
// ---------------------------------------------------------------------------

#[inline]
pub(super) fn binomial_neglog_q_derivatives_dispatch(
    y: f64,
    weight: f64,
    q: f64,
    mu: f64,
    d1: f64,
    d2: f64,
    d3: f64,
    link_kind: &InverseLink,
) -> (f64, f64, f64) {
    if binomial_link_has_closed_form(link_kind) {
        return binomial_neglog_q_derivatives_closed_form_dispatch(y, weight, q, link_kind);
    }
    binomial_neglog_q_derivatives_from_jet(y, weight, mu, d1, d2, d3)
}

#[inline]
pub(super) fn binomial_neglog_q_fourth_derivative_dispatch(
    y: f64,
    weight: f64,
    q: f64,
    mu: f64,
    d1: f64,
    d2: f64,
    d3: f64,
    link_kind: &InverseLink,
) -> Result<f64, String> {
    if binomial_link_has_closed_form(link_kind) {
        return Ok(binomial_neglog_q_fourth_derivative_closed_form_dispatch(
            y, weight, q, link_kind,
        ));
    }
    // `binomial_neglog_q_fourth_derivative_from_jet` consumes `d4 = μ''''(q)`,
    // the fourth derivative of the inverse link μ = G. Since the pdf f = G' = μ',
    // this equals f''' — the THIRD derivative of the inverse-link pdf (= the
    // fourth derivative of the inverse-link CDF). The `pdffourth` helper would
    // return f'''' = μ''''', one order too high (issue #947).
    let d4 = inverse_link_pdfthird_derivative_for_inverse_link(link_kind, q)
        .map_err(|e| format!("binomial inverse-link third derivative evaluation failed: {e}"))?;
    Ok(binomial_neglog_q_fourth_derivative_from_jet(
        y, weight, mu, d1, d2, d3, d4,
    ))
}

#[inline]
pub(super) fn binomial_neglog_q_derivatives_closed_form_dispatch(
    y: f64,
    weight: f64,
    q: f64,
    link_kind: &InverseLink,
) -> (f64, f64, f64) {
    match link_kind {
        InverseLink::Standard(StandardLink::Probit) => {
            binomial_neglog_q_derivatives_probit_closed_form(y, weight, q)
        }
        InverseLink::Standard(StandardLink::Logit) => {
            binomial_neglog_q_derivatives_logit_closed_form(y, weight, q)
        }
        InverseLink::Standard(StandardLink::CLogLog) => {
            binomial_neglog_q_derivatives_cloglog_closed_form(y, weight, q)
        }
        _ => {
            // Should not be called for unsupported links; caller should use jet path.
            // This is a safety fallback.
            (0.0, 0.0, 0.0)
        }
    }
}

#[inline]
pub(super) fn binomial_neglog_q_fourth_derivative_closed_form_dispatch(
    y: f64,
    weight: f64,
    q: f64,
    link_kind: &InverseLink,
) -> f64 {
    match link_kind {
        InverseLink::Standard(StandardLink::Probit) => {
            binomial_neglog_q_fourth_derivative_probit_closed_form(y, weight, q)
        }
        InverseLink::Standard(StandardLink::Logit) => {
            binomial_neglog_q_fourth_derivative_logit_closed_form(y, weight, q)
        }
        InverseLink::Standard(StandardLink::CLogLog) => {
            binomial_neglog_q_fourth_derivative_cloglog_closed_form(y, weight, q)
        }
        _ => 0.0,
    }
}

/// Returns true if the given link supports closed-form m1–m4 derivatives for
/// the binomial location-scale family, enabling the exact joint Newton path.
#[inline]
pub(super) fn binomial_link_has_closed_form(link_kind: &InverseLink) -> bool {
    matches!(
        link_kind,
        InverseLink::Standard(StandardLink::Probit)
            | InverseLink::Standard(StandardLink::Logit)
            | InverseLink::Standard(StandardLink::CLogLog)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // Cauchit inverse link μ(q) = ½ + atan(q)/π and its eta-derivatives, derived
    // independently of the production link machinery. With u = 1 + q²:
    //   μ'    = 1 / (π u)
    //   μ''   = −2 q / (π u²)
    //   μ'''  = 2 (3 q² − 1) / (π u³)
    //   μ'''' = 24 q (1 − q²) / (π u⁴)
    fn cauchit_jet(q: f64) -> (f64, f64, f64, f64, f64) {
        let u = 1.0 + q * q;
        let mu = 0.5 + q.atan() / std::f64::consts::PI;
        let d1 = 1.0 / (std::f64::consts::PI * u);
        let d2 = -2.0 * q / (std::f64::consts::PI * u.powi(2));
        let d3 = 2.0 * (3.0 * q * q - 1.0) / (std::f64::consts::PI * u.powi(3));
        let d4 = 24.0 * q * (1.0 - q * q) / (std::f64::consts::PI * u.powi(4));
        (mu, d1, d2, d3, d4)
    }

    // Analytic generic m4 for the cauchit link via the jet path, matching the
    // dispatch's `from_jet` consumer exactly.
    fn cauchit_m4(y: f64, weight: f64, q: f64) -> f64 {
        let (mu, d1, d2, d3, d4) = cauchit_jet(q);
        binomial_neglog_q_fourth_derivative_from_jet(y, weight, mu, d1, d2, d3, d4)
    }

    // Analytic generic m3 for the cauchit link via the jet path (third entry of
    // the returned (m1, m2, m3) tuple).
    fn cauchit_m3(y: f64, weight: f64, q: f64) -> f64 {
        let (mu, d1, d2, d3, _d4) = cauchit_jet(q);
        binomial_neglog_q_derivatives_from_jet(y, weight, mu, d1, d2, d3).2
    }

    #[test]
    fn generic_binomial_m4_matches_finite_difference_of_m3_cauchit() {
        // High-order (5-point) central finite difference of m3 = dF³/dq³ should
        // equal the analytic m4 = dF⁴/dq⁴. This independently pins the receiving
        // fourth-derivative formula and is blind to the dispatch helper naming
        // (regression for issue #947: the wrong helper injected a spurious μ'''''
        // term, flipping both sign and magnitude).
        let h = 1e-4;
        for &(y, weight, q) in &[
            (0.3_f64, 2.0_f64, 0.7_f64),
            (0.8_f64, 1.0_f64, -0.4_f64),
            (0.1_f64, 3.0_f64, 1.3_f64),
            (0.6_f64, 0.5_f64, -1.1_f64),
        ] {
            let fd = (-cauchit_m3(y, weight, q + 2.0 * h)
                + 8.0 * cauchit_m3(y, weight, q + h)
                - 8.0 * cauchit_m3(y, weight, q - h)
                + cauchit_m3(y, weight, q - 2.0 * h))
                / (12.0 * h);
            let analytic = cauchit_m4(y, weight, q);
            let tol = 1e-5 * (1.0 + analytic.abs());
            assert!(
                (analytic - fd).abs() < tol,
                "cauchit m4 (y={y}, w={weight}, q={q}): analytic={analytic}, fd={fd}, diff={}",
                (analytic - fd).abs()
            );
        }
    }

    #[test]
    fn generic_binomial_m4_matches_analytic_cauchit_ground_truth() {
        // Issue #947 concrete check: at q=0.7, y=0.3, w=2 the correct generic m4
        // is +2.1168155916; the off-by-one bug produced −10.3779706944.
        let analytic = cauchit_m4(0.3, 2.0, 0.7);
        assert!(
            (analytic - 2.1168155916).abs() < 1e-7,
            "expected +2.1168155916, got {analytic}"
        );
        // Guard against regression to the buggy fifth-derivative value.
        assert!(
            (analytic - (-10.3779706944)).abs() > 1.0,
            "matched the buggy m4 value {analytic}"
        );
    }
}
