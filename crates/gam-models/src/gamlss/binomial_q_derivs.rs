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

use gam_math::jet_tower::{Tower3, Tower4};
use gam_math::probability::normal_logcdf_derivatives;
use gam_problem::{InverseLink, StandardLink};
use gam_solve::mixture_link::inverse_link_pdfthird_derivative_for_inverse_link;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BinomialClosedFormLink {
    Probit,
    Logit,
    CLogLog,
}

#[inline]
fn binomial_closed_form_link(link_kind: &InverseLink) -> Option<BinomialClosedFormLink> {
    match link_kind {
        InverseLink::Standard(StandardLink::Probit) => Some(BinomialClosedFormLink::Probit),
        InverseLink::Standard(StandardLink::Logit) => Some(BinomialClosedFormLink::Logit),
        InverseLink::Standard(StandardLink::CLogLog) => Some(BinomialClosedFormLink::CLogLog),
        _ => None,
    }
}

/// The generic-link binomial NLL `F(q) = −ℓ(μ(q))` as ONE `Tower4<1>` jet in
/// the latent coordinate `q` (issue #932 migration of the hand-written
/// composition tower).
///
/// The two halves of the calculus enter through their single sources:
/// * the **inner** map `μ(q)` arrives as its stored jet `(μ, d1, d2, d3, d4)`,
///   seeded as the value/first/second/third/fourth channels of a `Tower4<1>`;
/// * the **outer** map `ℓ(μ)` enters as the hand-certified μ-space derivative
///   stack `[·, ℓ′, ℓ″, ℓ‴, ℓ⁗]` from [`binomial_loglik_mu_derivatives`]
///   through [`Tower4::compose_unary`] (exact multivariate Faà di Bruno).
///
/// The composition's Leibniz/Faà-di-Bruno coefficients — the very cross terms
/// the old hand path (`ellmumu·μ′² + ellmu·μ″`, the 6/3/4-weighted fourth-order
/// sum, …) wrote out by hand and that the #736/#947/#948 bugs lived in — are
/// now produced mechanically. The returned tower's `g`/`h`/`t3`/`t4` channels
/// are `ℓ(μ(q))`'s exact `dⁿℓ/dqⁿ`; callers negate for the NLL `F`.
///
/// `ℓ(μ)`'s value channel is irrelevant (only derivative channels are
/// consumed), so the stack's slot-0 entry is left at `0.0`. Saturation /
/// non-finite-jet short-circuits stay with the callers, exactly as before.
#[inline]
fn binomial_loglik_q_tower(y: f64, mu: f64, d1: f64, d2: f64, d3: f64, d4: f64) -> Tower4<1> {
    let (ellmu, ellmumu, ellmumum, ellmumumum) = binomial_loglik_mu_derivatives(y, mu);
    // μ(q) as the inner jet: value mu, derivatives d1..d4 in the single slot.
    let mut mu_tower = Tower4::<1>::constant(mu);
    mu_tower.g[0] = d1;
    mu_tower.h[0][0] = d2;
    mu_tower.t3[0][0][0] = d3;
    mu_tower.t4[0][0][0][0] = d4;
    // ℓ(μ) via Faà di Bruno; slot-0 (the value f(u)) is unused downstream.
    mu_tower.compose_unary([0.0, ellmu, ellmumu, ellmumum, ellmumumum])
}

/// The third-order truncation of [`binomial_loglik_q_tower`] as a `Tower3<1>`.
///
/// The score/curvature/third consumer
/// ([`binomial_score_curvaturethird_from_jet`]) reads only the `g`/`h`/`t3`
/// channels — never `t4`. The order-≤3 Faà-di-Bruno partitions never reach the
/// `f⁗` stack slot (`ellmumumum` and beyond) nor the inner `t4` tensor, so a
/// `Tower3<1>` produces those three channels BIT-IDENTICALLY to the full
/// `Tower4<1>` while skipping the discarded fourth-order seeding, composition,
/// and tensor (proven by the standalone `to_bits` oracle for #1591). The
/// fourth-derivative consumer keeps using [`binomial_loglik_q_tower`].
#[inline]
fn binomial_loglik_q_tower3(y: f64, mu: f64, d1: f64, d2: f64, d3: f64) -> Tower3<1> {
    let (ellmu, ellmumu, ellmumum, _ellmumumum) = binomial_loglik_mu_derivatives(y, mu);
    // μ(q) as the inner jet: value mu, derivatives d1..d3 in the single slot.
    let mut mu_tower = Tower3::<1>::constant(mu);
    mu_tower.g[0] = d1;
    mu_tower.h[0][0] = d2;
    mu_tower.t3[0][0][0] = d3;
    // ℓ(μ) via the order-≤3 Faà di Bruno; slot-0 (the value f(u)) is unused.
    mu_tower.compose_unary([0.0, ellmu, ellmumu, ellmumum])
}

/// Exact derivatives of the per-row binomial log-likelihood in μ-space,
///   ℓ(μ) = y·ln μ + (1−y)·ln(1−μ),
/// through fourth order, returned as `(ℓ', ℓ'', ℓ''', ℓ'''')`.
///
/// This is a pure function of the **raw, unclamped** probability `mu` and the
/// observed proportion `y ∈ [0, 1]`. There is no flooring: the values are the
/// exact derivatives of the loss the family actually evaluates. The closed
/// forms are
///   ℓ'    =  y/μ − (1−y)/(1−μ)
///   ℓ''   = −y/μ² − (1−y)/(1−μ)²
///   ℓ'''  =  2y/μ³ − 2(1−y)/(1−μ)³
///   ℓ'''' = −6y/μ⁴ − 6(1−y)/(1−μ)⁴
///
/// Each half is split by its numerator (`y` and `1−y`) so that a saturated
/// **compatible** observation — `y = 0` at any μ, or `y = 1` at any μ — cannot
/// manufacture a `0/0 = NaN`: the dead half is forced to exactly zero, which is
/// its true value (that branch of the likelihood is constant). Callers must
/// have already screened `μ ∈ (0, 1)`; an incompatible saturated boundary
/// (`y > 0` with `μ = 0`, or `y < 1` with `μ = 1`) is a genuine ±∞ and is the
/// caller's responsibility (see the saturation guard in the `*_from_jet`
/// consumers).
#[inline]
pub(super) fn binomial_loglik_mu_derivatives(y: f64, mu: f64) -> (f64, f64, f64, f64) {
    // y-branch (numerator y): y/μ, −y/μ², 2y/μ³, −6y/μ⁴.
    let (a1, a2, a3, a4) = if y == 0.0 {
        (0.0, 0.0, 0.0, 0.0)
    } else {
        let im = 1.0 / mu;
        let y_im = y * im;
        (
            y_im,
            -y_im * im,
            2.0 * y_im * im * im,
            -6.0 * y_im * im * im * im,
        )
    };
    // (1−y)-branch (numerator z = 1−y): −z/(1−μ), −z/(1−μ)², −2z/(1−μ)³, −6z/(1−μ)⁴.
    let z = 1.0 - y;
    let (b1, b2, b3, b4) = if z == 0.0 {
        (0.0, 0.0, 0.0, 0.0)
    } else {
        let io = 1.0 / (1.0 - mu);
        let z_io = z * io;
        (
            -z_io,
            -z_io * io,
            -2.0 * z_io * io * io,
            -6.0 * z_io * io * io * io,
        )
    };
    (a1 + b1, a2 + b2, a3 + b3, a4 + b4)
}

/// True iff `mu` is strictly interior, i.e. the inverse link has NOT saturated
/// past the representable range. When this is false the μ-space tower above has
/// no finite f64 representation, but the q-space derivatives built on top of it
/// have collapsed below precision (the inverse-link density `d1 = μ'` and its
/// successors underflow at least as fast as μ reaches the boundary), so the
/// honest q-space limit is zero — NOT a clipped-μ surrogate (issue #948).
#[inline]
pub(crate) fn binomial_mu_is_interior(mu: f64) -> bool {
    mu > 0.0 && mu < 1.0
}

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
    // q-jet using the EXACT mu-space derivatives (binomial_loglik_mu_derivatives)
    // and the inverse-link mu(q) derivatives d1=mu', d2=mu'', d3=mu''':
    //   s = dell/dq   = ellmu * mu'
    //   c = d2ell/dq2 = ellmumu*(mu')^2 + ellmu*mu''
    //   t = d3ell/dq3 = ellmumum*(mu')^3 + 3*ellmumu*mu'*mu'' + ellmu*mu'''
    //
    // Returns (score_q, curvature_q, third_q) with curvature_q = -d2ell/dq2.
    //
    // `mu` is the RAW inverse-link value (no flooring): the result is the exact
    // derivative of the evaluated loss for every representable mu in (0,1). A
    // saturated boundary mu collapses the q-space tower below precision, so we
    // return zero there rather than a clipped surrogate (issue #948).
    if weight == 0.0 || !binomial_mu_is_interior(mu) {
        return (0.0, 0.0, 0.0);
    }
    // The first three q-derivatives of ℓ(μ(q)) come from ONE jet composition
    // (issue #932): the inner μ-jet (d1,d2,d3) composed with the μ-space
    // ℓ-derivative stack. The hand-summed chain rule `ellmumu·d1² + ellmu·d2`
    // etc. is now the tower's `g`/`h`/`t3` channels. A `Tower3<1>` is used (not
    // `Tower4<1>`): nothing below reads the fourth channel, so the fourth-order
    // seeding/compose/tensor would be computed and discarded (#1591 prune,
    // proven `to_bits`-identical to the `Tower4<1>` read channels).
    let tower = binomial_loglik_q_tower3(y, mu, d1, d2, d3);
    let score_q = weight * tower.g[0];
    let curvature_q = -weight * tower.h[0][0];
    let third_q = weight * tower.t3[0][0][0];
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
    // The shared log-Phi stack carries both separated tails without Mills-ratio
    // cancellation. Zero-coefficient branches are skipped explicitly so a
    // compatible saturated observation cannot manufacture `0 * infinity`.
    if weight == 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let derivatives = binomial_neglog_q_derivatives_probit_stack(y, weight, q);
    (derivatives[0], derivatives[1], derivatives[2])
}

#[inline]
pub(super) fn binomial_neglog_q_fourth_derivative_probit_closed_form(
    y: f64,
    weight: f64,
    q: f64,
) -> f64 {
    // Closed-form m4 for F_i(q) = -w_i[y log Phi(q) + (1-y) log(1-Phi(q))].
    if weight == 0.0 {
        return 0.0;
    }
    binomial_neglog_q_derivatives_probit_stack(y, weight, q)[3]
}

#[inline]
fn binomial_neglog_q_derivatives_probit_stack(y: f64, weight: f64, q: f64) -> [f64; 4] {
    let left = normal_logcdf_derivatives(q);
    let right = normal_logcdf_derivatives(-q);
    let y0 = 1.0 - y;
    let mut out = [0.0; 4];
    for order in 1..=4 {
        let left_term = coefficient_product(y, left[order]);
        let reflected = if order % 2 == 0 {
            right[order]
        } else {
            -right[order]
        };
        let right_term = coefficient_product(y0, reflected);
        let derivative = -weight * (left_term + right_term);
        out[order - 1] = if derivative == 0.0 { 0.0 } else { derivative };
    }
    out
}

#[inline]
fn coefficient_product(coefficient: f64, value: f64) -> f64 {
    if coefficient == 0.0 {
        0.0
    } else {
        coefficient * value
    }
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

/// Stable logistic probability `p = σ(q)` and its variance `s = p(1−p)`, with
/// NO flooring.
///
/// `p` uses the branched expit (avoids overflow of `e^{±q}`), and `s` is formed
/// directly from the tail exponential as `s = t/(1+t)²` with `t = e^{−|q|}`,
/// the cancellation-free spelling of `p(1−p)`:
///   q ≥ 0: p = 1/(1+t),  1−p = t/(1+t)  ⇒  p(1−p) = t/(1+t)²
///   q < 0: p = t/(1+t),  1−p = 1/(1+t)  ⇒  p(1−p) = t/(1+t)²
/// This is exact across the whole range and underflows *gracefully to the true
/// value 0* in the saturated tail — the Bernoulli curvature genuinely vanishes
/// there. It is never recovered as `1 − p` after `p` has rounded to 1 (which
/// would catastrophically cancel) nor floored to a surrogate (issue #948). At
/// `q = 40`, `s ≈ e^{−40} ≈ 4.25e−18`, the true variance — not `1e−10`.
#[inline]
pub(crate) fn logit_probability_and_variance(q: f64) -> (f64, f64) {
    if q >= 0.0 {
        let t = (-q).exp();
        let denom = 1.0 + t;
        (1.0 / denom, t / (denom * denom))
    } else {
        let t = q.exp();
        let denom = 1.0 + t;
        (t / denom, t / (denom * denom))
    }
}

#[inline]
pub(super) fn binomial_neglog_q_derivatives_logit_closed_form(
    y: f64,
    weight: f64,
    q: f64,
) -> (f64, f64, f64) {
    // Returns (m1, m2, m3) for F(q) = -w[y log G(q) + (1-y) log(1-G(q))]
    // with G = logistic CDF. All three are exact derivatives of the evaluated
    // softplus loss F(q) = w[(1-y)q + softplus(-q)]:
    //   m1 = w(p - y),  m2 = ws,  m3 = ws(1 - 2p),  with s = p(1-p).
    if weight == 0.0 || !q.is_finite() {
        return (0.0, 0.0, 0.0);
    }
    let (p, s) = logit_probability_and_variance(q);

    let m1 = weight * (p - y);
    let m2 = weight * s;
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
    // Exact `s = p(1-p)` via the cancellation-free tail form — see
    // `logit_probability_and_variance`. m4 depends only on s.
    let (_p, s) = logit_probability_and_variance(q);
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
    // Exact m4 from the RAW inverse-link value (no flooring). A saturated
    // boundary mu, or any non-finite jet input, collapses the q-space tower
    // below precision, so we short-circuit to zero rather than a clipped
    // surrogate (issue #948). Non-finite inputs short-circuiting also matches
    // the LM gain-ratio guard, which rejects non-finite candidate gradients.
    if weight == 0.0
        || !binomial_mu_is_interior(mu)
        || !d1.is_finite()
        || !d2.is_finite()
        || !d3.is_finite()
        || !d4.is_finite()
    {
        return 0.0;
    }
    // m4 = −d⁴ℓ/dq⁴ from the SAME single jet composition (issue #932): the
    // tower's fourth channel is the exact Faà-di-Bruno fourth derivative
    // `ℓ⁗·d1⁴ + 6ℓ‴·d1²d2 + ℓ″·(3d2² + 4d1·d3) + ℓ′·d4`, mechanized rather than
    // hand-summed (the term whose off-by-one was issue #947).
    let tower = binomial_loglik_q_tower(y, mu, d1, d2, d3, d4);
    -weight * tower.t4[0][0][0][0]
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
    if let Some(closed_form) = binomial_closed_form_link(link_kind) {
        return binomial_neglog_q_derivatives_closed_form_dispatch(y, weight, q, closed_form);
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
    if let Some(closed_form) = binomial_closed_form_link(link_kind) {
        return Ok(binomial_neglog_q_fourth_derivative_closed_form_dispatch(
            y,
            weight,
            q,
            closed_form,
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
fn binomial_neglog_q_derivatives_closed_form_dispatch(
    y: f64,
    weight: f64,
    q: f64,
    link_kind: BinomialClosedFormLink,
) -> (f64, f64, f64) {
    match link_kind {
        BinomialClosedFormLink::Probit => {
            binomial_neglog_q_derivatives_probit_closed_form(y, weight, q)
        }
        BinomialClosedFormLink::Logit => {
            binomial_neglog_q_derivatives_logit_closed_form(y, weight, q)
        }
        BinomialClosedFormLink::CLogLog => {
            binomial_neglog_q_derivatives_cloglog_closed_form(y, weight, q)
        }
    }
}

#[inline]
fn binomial_neglog_q_fourth_derivative_closed_form_dispatch(
    y: f64,
    weight: f64,
    q: f64,
    link_kind: BinomialClosedFormLink,
) -> f64 {
    match link_kind {
        BinomialClosedFormLink::Probit => {
            binomial_neglog_q_fourth_derivative_probit_closed_form(y, weight, q)
        }
        BinomialClosedFormLink::Logit => {
            binomial_neglog_q_fourth_derivative_logit_closed_form(y, weight, q)
        }
        BinomialClosedFormLink::CLogLog => {
            binomial_neglog_q_fourth_derivative_cloglog_closed_form(y, weight, q)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probit_closed_form_preserves_separated_and_compatible_tail_limits() {
        assert_eq!(
            binomial_neglog_q_derivatives_probit_closed_form(1.0, 2.0, f64::INFINITY),
            (0.0, 0.0, 0.0)
        );
        assert_eq!(
            binomial_neglog_q_derivatives_probit_closed_form(0.0, 2.0, f64::INFINITY),
            (f64::INFINITY, 2.0, 0.0)
        );
        assert_eq!(
            binomial_neglog_q_derivatives_probit_closed_form(0.0, 2.0, f64::NEG_INFINITY),
            (0.0, 0.0, 0.0)
        );
        assert_eq!(
            binomial_neglog_q_derivatives_probit_closed_form(1.0, 2.0, f64::NEG_INFINITY),
            (f64::NEG_INFINITY, 2.0, 0.0)
        );
        let nan_derivatives = binomial_neglog_q_derivatives_probit_closed_form(0.5, 2.0, f64::NAN);
        assert!(
            [nan_derivatives.0, nan_derivatives.1, nan_derivatives.2]
                .into_iter()
                .all(f64::is_nan)
        );
    }

    #[test]
    fn probit_closed_form_keeps_deep_tail_curvature_and_subnormal_channels() {
        let (m1, m2, m3) = binomial_neglog_q_derivatives_probit_closed_form(0.0, 1.0, 1.0e100);
        assert_eq!(m1, 1.0e100);
        assert_eq!(m2, 1.0);
        assert!(m3 > 0.0 && m3.is_finite());
        assert_eq!(
            binomial_neglog_q_fourth_derivative_probit_closed_form(0.0, 1.0, 1.0e100),
            0.0
        );

        let (m1, m2, m3) = binomial_neglog_q_derivatives_probit_closed_form(1.0, 1.0, 38.6);
        assert_eq!(m1, 0.0);
        assert!(m2 > 0.0 && m2.is_subnormal());
        assert!(m3 < 0.0 && m3.is_subnormal());
        let m4 = binomial_neglog_q_fourth_derivative_probit_closed_form(1.0, 1.0, 38.6);
        assert!(m4 > 0.0 && m4.is_subnormal());
    }

    // Cauchit inverse link μ(q) = ½ + atan(q)/π and its eta-derivatives, derived
    // independently of the production link machinery. With u = 1 + q²:
    //   μ'    = 1 / (π u)
    //   μ''   = −2 q / (π u²)
    //   μ'''  = 2 (3 q² − 1) / (π u³)
    //   μ'''' = 24 q (1 − q²) / (π u⁴)
    pub(crate) fn cauchit_jet(q: f64) -> (f64, f64, f64, f64, f64) {
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
    pub(crate) fn cauchit_m4(y: f64, weight: f64, q: f64) -> f64 {
        let (mu, d1, d2, d3, d4) = cauchit_jet(q);
        binomial_neglog_q_fourth_derivative_from_jet(y, weight, mu, d1, d2, d3, d4)
    }

    // Analytic generic m3 for the cauchit link via the jet path (third entry of
    // the returned (m1, m2, m3) tuple).
    pub(crate) fn cauchit_m3(y: f64, weight: f64, q: f64) -> f64 {
        let (mu, d1, d2, d3, _d4) = cauchit_jet(q);
        binomial_neglog_q_derivatives_from_jet(y, weight, mu, d1, d2, d3).2
    }

    /// The PRE-MIGRATION hand-summed chain rule for the generic-link binomial
    /// q-derivative tower, kept verbatim here as the bit-identity witness for the
    /// #932 `Tower4`-composition migration. If the tower path ever drifts from
    /// the hand calculus these four channels disagree.
    fn legacy_hand_q_derivatives(
        y: f64,
        weight: f64,
        mu: f64,
        d1: f64,
        d2: f64,
        d3: f64,
        d4: f64,
    ) -> (f64, f64, f64, f64) {
        let (ellmu, ellmumu, ellmumum, ellmumumum) = binomial_loglik_mu_derivatives(y, mu);
        let score_q = weight * ellmu * d1;
        let curvature_q = -weight * (ellmumu * d1 * d1 + ellmu * d2);
        let third_q = weight * (ellmumum * d1 * d1 * d1 + 3.0 * ellmumu * d1 * d2 + ellmu * d3);
        let fourth_q = weight
            * (ellmumumum * d1.powi(4)
                + 6.0 * ellmumum * d1 * d1 * d2
                + ellmumu * (3.0 * d2 * d2 + 4.0 * d1 * d3)
                + ellmu * d4);
        (score_q, curvature_q, third_q, -fourth_q)
    }

    /// #932 migration bit-identity: the `Tower4<1>` composition path
    /// (`binomial_score_curvaturethird_from_jet` /
    /// `binomial_neglog_q_fourth_derivative_from_jet`) reproduces the legacy
    /// hand-summed chain rule to machine precision on a dense (y, w, q)×link grid
    /// of interior points. Exercises cauchit and logit inner jets so every
    /// Faà-di-Bruno term participates with a nonzero coefficient.
    #[test]
    pub(crate) fn generic_jet_tower_matches_legacy_hand_chain_rule() {
        for &(y, w) in &[
            (0.3_f64, 2.0_f64),
            (0.7, 1.0),
            (0.0, 1.5),
            (1.0, 0.5),
            (0.42, 3.0),
        ] {
            for &q in &[-1.3_f64, -0.4, 0.0, 0.5, 0.7, 1.3, 2.0] {
                for &(mu, d1, d2, d3, d4) in &[cauchit_jet(q), logit_jet(q)] {
                    let (s, c, t) = binomial_score_curvaturethird_from_jet(y, w, mu, d1, d2, d3);
                    let m4 = binomial_neglog_q_fourth_derivative_from_jet(y, w, mu, d1, d2, d3, d4);
                    let (ls, lc, lt, lm4) = legacy_hand_q_derivatives(y, w, mu, d1, d2, d3, d4);
                    let tol = 1e-12;
                    let close = |label: &str, got: f64, want: f64| {
                        assert!(
                            (got - want).abs() <= tol * want.abs().max(1.0),
                            "{label} (y={y}, w={w}, q={q}, mu={mu}): tower {got:+.17e} vs legacy {want:+.17e}"
                        );
                    };
                    close("score_q", s, ls);
                    close("curvature_q", c, lc);
                    close("third_q", t, lt);
                    close("m4", m4, lm4);
                }
            }
        }
    }

    #[test]
    pub(crate) fn generic_binomial_m4_matches_finite_difference_of_m3_cauchit() {
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
            let fd = (-cauchit_m3(y, weight, q + 2.0 * h) + 8.0 * cauchit_m3(y, weight, q + h)
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
    pub(crate) fn generic_binomial_m4_matches_analytic_cauchit_ground_truth() {
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

    // Logit inverse-link jet (μ and its first four q-derivatives), derived
    // independently of production code: with p = σ(q) and s = p(1-p),
    //   μ = p,  μ' = s,  μ'' = s(1-2p),  μ''' = s(1-6s),  μ'''' = s(1-2p)(1-12s).
    pub(crate) fn logit_jet(q: f64) -> (f64, f64, f64, f64, f64) {
        let p = 1.0 / (1.0 + (-q).exp());
        let s = p * (1.0 - p);
        (
            p,
            s,
            s * (1.0 - 2.0 * p),
            s * (1.0 - 6.0 * s),
            s * (1.0 - 2.0 * p) * (1.0 - 12.0 * s),
        )
    }

    #[test]
    pub(crate) fn logit_closed_form_agrees_with_generic_jet_path() {
        // The canonical-logit closed form and the generic μ-jet path are two
        // independent computations of the same derivative tower. Where both are
        // numerically valid (μ comfortably interior) they must agree to float
        // precision — a cross-check that pins the sign and coefficient of every
        // term in BOTH paths (issue #948).
        for &(y, w, q) in &[
            (0.3_f64, 2.0_f64, 0.5_f64),
            (0.7, 1.0, -1.3),
            (0.0, 1.5, 2.0),
            (1.0, 0.5, -0.8),
            (0.42, 3.0, 0.0),
        ] {
            let (m1, m2, m3) = binomial_neglog_q_derivatives_logit_closed_form(y, w, q);
            let m4 = binomial_neglog_q_fourth_derivative_logit_closed_form(y, w, q);

            let (mu, d1, d2, d3, d4) = logit_jet(q);
            let (s1, c2, t3) = binomial_neglog_q_derivatives_from_jet(y, w, mu, d1, d2, d3);
            let j4 = binomial_neglog_q_fourth_derivative_from_jet(y, w, mu, d1, d2, d3, d4);

            let tol = 1e-9 * (1.0 + m1.abs() + m2.abs() + m3.abs() + m4.abs());
            assert!(
                (m1 - s1).abs() < tol,
                "m1 mismatch q={q}: closed={m1} jet={s1}"
            );
            assert!(
                (m2 - c2).abs() < tol,
                "m2 mismatch q={q}: closed={m2} jet={c2}"
            );
            assert!(
                (m3 - t3).abs() < tol,
                "m3 mismatch q={q}: closed={m3} jet={t3}"
            );
            assert!(
                (m4 - j4).abs() < tol,
                "m4 mismatch q={q}: closed={m4} jet={j4}"
            );
        }
    }

    #[test]
    pub(crate) fn logit_curvature_exact_through_the_old_clamp_boundary() {
        // Issue #948 (2b): once 1-p < MIN_PROB (q ≳ 23) the old code floored the
        // variance at MIN_PROB·(1-MIN_PROB) ≈ 1e-10. The exact Bernoulli variance
        // is s = p(1-p) = e^{-q}/(1+e^{-q})², which must be reported verbatim.
        // Walk several q past the boundary and require exactness — and that the
        // result is emphatically NOT the ~1e-10 floor.
        for &q in &[24.0_f64, 30.0, 40.0, 50.0] {
            let t = (-q).exp();
            let denom = 1.0 + t;
            let s_exact = t / (denom * denom);
            let (_, m2, _) = binomial_neglog_q_derivatives_logit_closed_form(1.0, 1.0, q);
            let m4 = binomial_neglog_q_fourth_derivative_logit_closed_form(1.0, 1.0, q);
            assert!(
                (m2 - s_exact).abs() <= 1e-12 * s_exact,
                "q={q}: m2={m2} != exact s={s_exact}"
            );
            assert!(
                m2 < 1e-10,
                "q={q}: m2={m2} looks floored at MIN_PROB·(1-MIN_PROB)"
            );
            assert!(
                (m4 - s_exact * (1.0 - 6.0 * s_exact)).abs() <= 1e-12 * s_exact,
                "q={q}: m4={m4} not exact ws(1-6s)"
            );
        }
    }

    #[test]
    pub(crate) fn generic_jet_uses_raw_sub_min_prob_mu_not_floored() {
        // Issue #948 (2a): the generic μ-jet path must divide by the RAW μ, not a
        // value floored at MIN_PROB=1e-10. Feed μ = 1e-12 (a representable
        // probability two orders below the old floor) with a unit jet to isolate
        // the ℓ''''(μ)·(μ')⁴ term: m4 = -w·(-6y/μ⁴) = 6wy/μ⁴, ~1e8× the value the
        // old clamp-to-1e-10 produced.
        let (y, w) = (1.0_f64, 1.0_f64);
        let mu = 1e-12_f64;
        let m4 = binomial_neglog_q_fourth_derivative_from_jet(y, w, mu, 1.0, 0.0, 0.0, 0.0);
        let exact = 6.0 * w * y / mu.powi(4);
        assert!(
            (m4 - exact).abs() <= 1e-6 * exact,
            "raw-μ m4 should be 6yw/μ⁴={exact}, got {m4}"
        );
        let floored = 6.0 * w * y / 1e-10_f64.powi(4);
        assert!(
            m4 > 100.0 * floored,
            "m4={m4} is near the floored value {floored}; μ was clamped"
        );

        // score_q = w·ℓ'(μ)·μ' = w·y/μ at y=1 — also raw.
        let (score, _curv, _third) =
            binomial_score_curvaturethird_from_jet(y, w, mu, 1.0, 0.0, 0.0);
        let exact_score = w * y / mu;
        assert!(
            (score - exact_score).abs() <= 1e-6 * exact_score,
            "raw-μ score should be wy/μ={exact_score}, got {score}"
        );
    }

    #[test]
    pub(crate) fn generic_jet_saturated_boundary_collapses_to_zero() {
        // Issue #948 (2a): a μ that has saturated past the representable range
        // (μ ≤ 0 or μ ≥ 1, or non-finite) has no finite μ-space tower, but the
        // q-space derivatives have collapsed below precision. The honest limit is
        // exactly zero — never NaN, ∞, or a clipped surrogate.
        for &mu in &[0.0_f64, 1.0, -0.0, f64::NAN, f64::INFINITY] {
            let (s, c, t) = binomial_score_curvaturethird_from_jet(0.7, 2.0, mu, 1.0, 0.5, 0.1);
            let m4 = binomial_neglog_q_fourth_derivative_from_jet(0.7, 2.0, mu, 1.0, 0.5, 0.1, 0.2);
            assert_eq!(
                (s, c, t),
                (0.0, 0.0, 0.0),
                "boundary μ={mu} must give zero score/curv/third"
            );
            assert_eq!(m4, 0.0, "boundary μ={mu} must give zero m4");
        }
    }

    #[test]
    pub(crate) fn loglik_mu_derivatives_no_nan_at_compatible_boundary() {
        // The per-branch split must keep a compatible saturated observation
        // finite: y=0 kills the y/μ half (no 0/0 at μ=0), y=1 kills the
        // (1-y)/(1-μ) half (no 0/0 at μ=1).
        let (e1, e2, e3, e4) = binomial_loglik_mu_derivatives(0.0, 0.0);
        for v in [e1, e2, e3, e4] {
            assert!(v.is_finite(), "y=0,μ=0 produced non-finite {v}");
        }
        assert_eq!(e1, -1.0, "ℓ'(0)=-(1-y)/(1-μ)=-1 at y=0,μ=0");

        let (f1, f2, f3, f4) = binomial_loglik_mu_derivatives(1.0, 1.0);
        for v in [f1, f2, f3, f4] {
            assert!(v.is_finite(), "y=1,μ=1 produced non-finite {v}");
        }
        assert_eq!(f1, 1.0, "ℓ'(1)=y/μ=1 at y=1,μ=1");
    }
}
