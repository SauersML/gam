use crate::estimate::EstimationError;
use crate::mixture_link::inverse_link_jet_for_family_public;
use crate::types::LikelihoodSpec;
use ndarray::{Array1, ArrayView1};
use statrs::function::erf::erfc;

/// Standard normal PDF φ(x).
#[inline]
pub fn normal_pdf(x: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

/// Standard normal CDF Φ(x) evaluated via the exact special-function identity
///
///   Phi(x) = 0.5 * erfc(-x / sqrt(2)).
///
/// This is the exact Gaussian CDF semantics used throughout the codebase. The
/// numerical `erfc` implementation may use internal approximations, but the
/// returned function is the standard normal CDF itself rather than a separate
/// polynomial surrogate surface.
#[inline]
pub fn normal_cdf(x: f64) -> f64 {
    0.5 * erfc(-x / std::f64::consts::SQRT_2)
}

/// Scaled complementary error function `erfcx(x) = exp(x²) · erfc(x)`,
/// specialized to `x ≥ 0`.  Returns `1.0` for `x ≤ 0` and `0.0` for
/// `x = +∞`.  For `0 < x < 26` uses the direct `exp(x²)·erfc(x)` form;
/// beyond that the (otherwise overflowing) `exp(x²)` is replaced by a
/// 4-term asymptotic expansion `(1/(x√π))·(1 − 1/(2x²) + 3/(4x⁴) − …)`,
/// keeping relative accuracy near machine epsilon. The non-negative
/// restriction lets the caller skip the reflection identity.
#[inline]
pub fn erfcx_nonnegative(x: f64) -> f64 {
    if !x.is_finite() {
        return if x.is_sign_positive() {
            0.0
        } else {
            f64::INFINITY
        };
    }
    if x <= 0.0 {
        return 1.0;
    }
    if x < 26.0 {
        ((x * x).min(700.0)).exp() * erfc(x)
    } else {
        let inv = 1.0 / x;
        let inv2 = inv * inv;
        let poly = 1.0 - 0.5 * inv2 + 0.75 * inv2 * inv2 - 1.875 * inv2 * inv2 * inv2
            + 6.5625 * inv2 * inv2 * inv2 * inv2;
        inv * poly / std::f64::consts::PI.sqrt()
    }
}

/// Computes `log(1 - exp(-a))` for `a >= 0` without cancellation.
#[inline]
pub fn log1mexp_positive(a: f64) -> f64 {
    assert!(a >= 0.0, "log1mexp_positive requires a >= 0: a={a}");
    if a > core::f64::consts::LN_2 {
        (-(-a).exp()).ln_1p()
    } else if a > 0.0 {
        (-(-a).exp_m1()).ln()
    } else {
        f64::NEG_INFINITY
    }
}

/// Numerically stable signed log-sum-exp.  Given pairs
/// `(log|aⱼ|, sign(aⱼ))` (with `signs[j] ∈ {−1, 0, +1}`), returns
/// `(log|S|, sign(S))` for `S = Σⱼ signs[j]·exp(log_mags[j])`.  Positive
/// and negative magnitudes are reduced separately with the standard
/// log-sum-exp trick (subtract the max, sum, log, add back); the two
/// partial sums are then combined via `log(|p − n|) =
/// max(log p, log n) + log1mexp(|log p − log n|)`, preserving accuracy
/// even when `p ≈ n` (catastrophic cancellation regime).  When all
/// signs are zero or all magnitudes are `−∞`, returns
/// `(NEG_INFINITY, 0.0)`.
///
/// A `+∞` log-magnitude denotes an infinite-magnitude term (`exp(+∞) = +∞`)
/// and dominates the sum: if it appears only with positive sign the result
/// is `(+∞, +1)`; only with negative sign, `(+∞, −1)` (a log-magnitude of
/// `+∞` with sign `−1` encodes the value `−∞`); with both signs the sum is
/// the indeterminate `+∞ − ∞`, returned as `(NaN, 0.0)`.  A `−∞`
/// log-magnitude is `exp(−∞) = 0` and is correctly dropped.
pub fn signed_log_sum_exp(log_mags: &[f64], signs: &[f64]) -> (f64, f64) {
    // Infinite-magnitude terms dominate any finite contribution, so resolve
    // them before the finite log-sum-exp reduction below. `−∞` log-magnitudes
    // are `exp(−∞) = 0` and need no special handling.
    let mut has_pos_inf = false;
    let mut has_neg_inf = false;
    for (idx, &lm) in log_mags.iter().enumerate() {
        if lm == f64::INFINITY {
            if signs[idx] > 0.0 {
                has_pos_inf = true;
            } else if signs[idx] < 0.0 {
                has_neg_inf = true;
            }
        }
    }
    match (has_pos_inf, has_neg_inf) {
        // P = +∞, N = +∞ ⇒ indeterminate +∞ − ∞.
        (true, true) => return (f64::NAN, 0.0),
        // P = +∞, N < ∞ ⇒ S = +∞.
        (true, false) => return (f64::INFINITY, 1.0),
        // N = +∞, P < ∞ ⇒ S = −∞, encoded as log-magnitude +∞ with sign −1.
        (false, true) => return (f64::INFINITY, -1.0),
        (false, false) => {}
    }

    let mut pos_max = f64::NEG_INFINITY;
    let mut neg_max = f64::NEG_INFINITY;
    for (idx, &lm) in log_mags.iter().enumerate() {
        if signs[idx] > 0.0 {
            pos_max = pos_max.max(lm);
        } else if signs[idx] < 0.0 {
            neg_max = neg_max.max(lm);
        }
    }

    let mut pos_sum = 0.0_f64;
    let mut neg_sum = 0.0_f64;
    for (idx, &lm) in log_mags.iter().enumerate() {
        if !lm.is_finite() {
            continue;
        }
        if signs[idx] > 0.0 {
            pos_sum += (lm - pos_max).exp();
        } else if signs[idx] < 0.0 {
            neg_sum += (lm - neg_max).exp();
        }
    }

    let log_pos = if pos_sum > 0.0 {
        pos_max + pos_sum.ln()
    } else {
        f64::NEG_INFINITY
    };
    let log_neg = if neg_sum > 0.0 {
        neg_max + neg_sum.ln()
    } else {
        f64::NEG_INFINITY
    };

    if log_neg == f64::NEG_INFINITY {
        return (log_pos, 1.0);
    }
    if log_pos == f64::NEG_INFINITY {
        return (log_neg, -1.0);
    }
    if log_pos > log_neg {
        let gap = log_pos - log_neg;
        (log_pos + log1mexp_positive(gap), 1.0)
    } else if log_neg > log_pos {
        let gap = log_neg - log_pos;
        (log_neg + log1mexp_positive(gap), -1.0)
    } else {
        (f64::NEG_INFINITY, 0.0)
    }
}

#[inline]
fn horner_polynomial(x: f64, coeffs: &[f64]) -> f64 {
    coeffs.iter().rev().fold(0.0, |acc, &c| acc * x + c)
}

/// Evaluate `(Σ_k coeffs[k]·x^k) · exp(−x)` without overflow.  For moderate
/// `x ≤ 600` uses Horner + `exp(−x)` directly; for very large `x` rewrites
/// `xᵈ · exp(−x) = exp(d·ln x − x)` and runs Horner in `1/x`, which keeps
/// both the polynomial sum and its multiplier inside double range.  Returns
/// `0.0` for non-finite `x` or empty `coeffs`.
#[inline]
pub fn stable_polynomial_times_exp_neg(x: f64, coeffs: &[f64]) -> f64 {
    if coeffs.is_empty() || !x.is_finite() {
        return 0.0;
    }
    // Below this argument `(-x).exp()` is still well-resolved, so the direct
    // Horner-times-exp form is both accurate and cheapest. Above it the factor
    // underflows toward zero and we switch to the convergent asymptotic tail
    // series to retain the leading significant digits.
    const DIRECT_EXP_SWITCH: f64 = 600.0;
    if x <= DIRECT_EXP_SWITCH {
        return horner_polynomial(x, coeffs) * (-x).exp();
    }

    let inv_x = x.recip();
    let mut tail = 0.0;
    for &c in coeffs {
        tail = tail * inv_x + c;
    }
    let degree = (coeffs.len() - 1) as f64;
    let scale = (degree * x.ln() - x).exp();
    scale * tail
}

/// Numerically stable `C(n,k) = n! / (k!·(n−k)!)` as `f64`.  Uses the
/// symmetry `C(n,k) = C(n, n−k)` to keep the loop count `min(k, n−k)`
/// and the multiplicative recurrence `C(n,j+1) = C(n,j)·(n−j)/(j+1)`,
/// avoiding the overflow of separate factorial evaluations.  Returns
/// `0.0` for `k > n` and exact integer results within `2^53`.
#[inline]
pub fn binomial_coefficient_f64(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    if k == 0 || k == n {
        return 1.0;
    }
    let k_eff = k.min(n - k);
    let mut out = 1.0;
    for j in 0..k_eff {
        out *= (n - j) as f64 / (j + 1) as f64;
    }
    out
}

/// Numerically stable `ln Φ(x)` for the standard normal CDF.  For `x ≥ 0`
/// computes `ln(Φ(x))` directly with a small floor against underflow; for
/// `x < 0` rewrites
/// `ln Φ(x) = −u² + ln(½·erfcx(u))`, `u = −x/√2`,
/// which preserves digits all the way into the deep left tail (no
/// `ln(0)`).  Returns `±∞` and `NaN` at the corresponding inputs.
#[inline]
pub fn normal_logcdf(x: f64) -> f64 {
    if x == f64::INFINITY {
        return 0.0;
    }
    if x == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    if x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 {
        let u = -x / std::f64::consts::SQRT_2;
        -u * u + (0.5 * erfcx_nonnegative(u).max(1e-300)).ln()
    } else {
        normal_cdf(x).clamp(1e-300, 1.0).ln()
    }
}

/// Numerically stable `ln(1 − Φ(x)) = ln Φ(−x)` for the standard normal
/// survival function.  Delegates to `normal_logcdf(-x)` so the deep-right
/// tail benefits from the same `erfcx`-based representation.
#[inline]
pub fn normal_logsf(x: f64) -> f64 {
    normal_logcdf(-x)
}

/// Joint evaluation of `ln Φ(x)` and the Mills-ratio analogue
/// `φ(x) / Φ(x)`, signed for the symmetric branch.  Used by the latent
/// probit families where the inverse-link gradient needs the ratio and
/// the likelihood needs the log-CDF on the same `x`; computing both in
/// one call shares the `erfcx` evaluation that dominates the cost in the
/// deep tail.
#[inline]
pub fn signed_probit_logcdf_and_mills_ratio(x: f64) -> (f64, f64) {
    if x == f64::INFINITY {
        return (0.0, 0.0);
    }
    if x == f64::NEG_INFINITY {
        return (f64::NEG_INFINITY, f64::INFINITY);
    }
    if x.is_nan() {
        return (f64::NAN, f64::NAN);
    }
    if x < 0.0 {
        let u = -x / std::f64::consts::SQRT_2;
        let ex = erfcx_nonnegative(u).max(1e-300);
        let log_cdf = -u * u + (0.5 * ex).ln();
        let lambda = (2.0 / std::f64::consts::PI).sqrt() / ex;
        (log_cdf, lambda)
    } else {
        let cdf = normal_cdf(x).clamp(1e-300, 1.0);
        let lambda = normal_pdf(x) / cdf;
        (cdf.ln(), lambda)
    }
}

/// Standard normal quantile Φ⁻¹(p) using Acklam's rational approximation.
#[inline]
pub fn standard_normal_quantile(p: f64) -> Result<f64, String> {
    if !(p.is_finite() && p > 0.0 && p < 1.0) {
        return Err(format!("normal quantile requires p in (0,1), got {p}"));
    }

    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    let mut x = if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };
    for _ in 0..2 {
        let density = normal_pdf(x);
        if !(density.is_finite() && density > 0.0) {
            break;
        }
        // Residual F(x) − p, formed without catastrophic cancellation in
        // either tail. For an upper-tail iterate `x > 0`, `normal_cdf(x)`
        // saturates to ~1, so the direct `normal_cdf(x) − p` annihilates the
        // tiny residual the polish must act on; instead use the upper-tail
        // complement `F(x) − p = (1 − p) − 0.5·erfc(x/√2)`, where both terms
        // are the small upper-tail quantities (`1 − p` is exact by Sterbenz
        // for `p ∈ [½,1)`). For `x ≤ 0`, `normal_cdf(x) = 0.5·erfc(|x|/√2)` is
        // itself the faithfully carried small lower-tail value, so the direct
        // form is already cancellation-free.
        let residual = if x > 0.0 {
            (1.0 - p) - 0.5 * erfc(x / std::f64::consts::SQRT_2)
        } else {
            normal_cdf(x) - p
        };
        let correction = residual / density;
        let denominator = 1.0 + 0.5 * x * correction;
        if !(correction.is_finite() && denominator.is_finite() && denominator != 0.0) {
            break;
        }
        let step = correction / denominator;
        if !step.is_finite() {
            break;
        }
        x -= step;
        if step.abs() <= 2.0 * f64::EPSILON * x.abs().max(1.0) {
            break;
        }
    }
    Ok(x)
}

/// Quantile (inverse CDF) of a Gamma distribution parameterized by shape
/// `k > 0` and scale `θ > 0` at probability `p ∈ (0, 1)`: the value `x` with
/// `P(X ≤ x) = p` for `X ~ Gamma(shape = k, scale = θ)` (mean `kθ`, variance
/// `kθ²`).
///
/// Equals `θ · Q(p; k)`, where `Q(p; k)` inverts the regularized lower
/// incomplete gamma `P(k, x)` (the unit-scale Gamma CDF). `p ≤ 0` maps to the
/// `0` support floor and `p ≥ 1` to `+∞`; a non-finite or non-positive shape or
/// scale yields `NaN`.
///
/// This is the building block for *skew-aware* response-scale predictive
/// (observation) intervals: a Gamma response is strongly right-skewed, so the
/// symmetric `μ ± z·σ` band mis-covers each tail even when its width (variance)
/// is correct. Equal-tailed Gamma quantiles place the right mass in each tail.
pub fn gamma_quantile(p: f64, shape: f64, scale: f64) -> f64 {
    if !(shape.is_finite() && shape > 0.0 && scale.is_finite() && scale > 0.0) {
        return f64::NAN;
    }
    scale * inverse_regularized_lower_gamma(p, shape)
}

/// Equal-tailed predictive interval for a strictly-positive, right-skewed
/// response modelled as a Gamma whose first two moments match a point
/// prediction: mean `mu` and total predictive variance `total_var`
/// (estimation + observation noise). Returns the pair of Gamma quantiles at
/// lower-tail probabilities `p_lo < p_hi` — the skew-correct replacement for a
/// symmetric `mu ± z·σ` band, which for a Gamma pins the lower edge near the
/// support floor and mis-covers each tail (#817).
///
/// Moment matching fixes `shape k = mu²/V` and `scale θ = V/mu`, so the
/// predictive carries exactly the requested mean and variance. When estimation
/// uncertainty vanishes (`total_var → φμ²`) this is *exact*: `k → 1/φ`,
/// `θ → φμ`, recovering the conditional Gamma `Gamma(shape = 1/φ, scale = φμ)`.
/// With nonzero estimation variance it is the moment-matched Gamma predictive —
/// the minimal skew-correct widening.
///
/// Returns `None` when the inputs are degenerate (non-positive mean or
/// variance, non-finite), or when the incomplete-gamma inverse yields a
/// non-finite / mis-ordered pair — which happens for an enormous shape, where
/// the Gamma is essentially Gaussian and the caller should fall back to the
/// then-accurate symmetric edges.
pub fn gamma_moment_matched_interval(
    mu: f64,
    total_var: f64,
    p_lo: f64,
    p_hi: f64,
) -> Option<(f64, f64)> {
    if !(mu.is_finite() && mu > 0.0 && total_var.is_finite() && total_var > 0.0) {
        return None;
    }
    let shape = mu * mu / total_var;
    let scale = total_var / mu;
    let q_lo = gamma_quantile(p_lo, shape, scale);
    let q_hi = gamma_quantile(p_hi, shape, scale);
    if q_lo.is_finite() && q_hi.is_finite() && q_hi >= q_lo {
        Some((q_lo, q_hi))
    } else {
        None
    }
}

/// Regularized lower incomplete gamma `P(a, x) = γ(a, x) / Γ(a)` — the CDF of a
/// unit-scale `Gamma(shape = a)` variate — accurate down to the smallest
/// representable `x`.
///
/// This is the exact function [`inverse_regularized_lower_gamma`] inverts, so we
/// own it rather than borrowing `statrs::gamma_lr`. That routine hard-clamps to
/// `0.0` for every `x ≤ 1.11e-15` (its `almost_eq(x, 0)` guard, with accuracy
/// `DEFAULT_F64_ACC`), which silently zeroes the residual `P(a, x) − p` in the
/// small-shape lower tail: the Halley iterate is then driven *up* — away from a
/// good sub-`1e-15` seed — until `x` crosses that clamp around `~1.6e-15`, where
/// the returned point carries far more mass than `p` (#1018). The Numerical
/// Recipes split — a power series for `x < a + 1`, the modified-Lentz continued
/// fraction for the complement `Q = 1 − P` otherwise — keeps the leading
/// `exp(a·ln x − x − ln Γ(a))` factor in logs, so the value stays finite and
/// nonzero for arguments far below that clamp, and always evaluates the *smaller*
/// tail directly (no catastrophic cancellation near either edge).
fn regularized_lower_gamma(a: f64, x: f64) -> f64 {
    use statrs::function::gamma::ln_gamma;
    // Callers (`inverse_regularized_lower_gamma`) validate `a > 0` upstream; a
    // non-positive `a` would only mis-feed `ln_gamma`, never UB.
    if x <= 0.0 {
        return 0.0;
    }
    let gln = ln_gamma(a);
    if x < a + 1.0 {
        // Power series: P(a,x) = exp(a·ln x − x − ln Γ(a)) · Σ_{n≥0} xⁿ / Π_{k=0}^{n}(a+k).
        // The running term `del` is the ratio form, so no factorial overflows.
        let mut ap = a;
        let mut del = 1.0 / a;
        let mut sum = del;
        for _ in 0..1000 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() <= sum.abs() * f64::EPSILON {
                break;
            }
        }
        (sum.ln() + a * x.ln() - x - gln).exp()
    } else {
        // Modified-Lentz continued fraction for Q(a,x) = 1 − P(a,x); P = 1 − Q.
        // Evaluating the *upper* tail here keeps the directly-computed quantity
        // small wherever P is near 1, so `1 − Q` loses no significant digits.
        const FPMIN: f64 = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / FPMIN;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..1000 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < FPMIN {
                d = FPMIN;
            }
            c = b + an / c;
            if c.abs() < FPMIN {
                c = FPMIN;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() <= f64::EPSILON {
                break;
            }
        }
        let q = (a * x.ln() - x - gln + h.ln()).exp();
        1.0 - q
    }
}

/// Inverse of the regularized lower incomplete gamma function: the `x ≥ 0` with
/// `P(a, x) = p`, where `P(a, x) = γ(a, x) / Γ(a)` is the CDF of a unit-scale
/// `Gamma(shape = a)` variate, `a > 0`, `p ∈ (0, 1)`.
///
/// Uses the standard rational/Wilson–Hilferty initial estimate (a series form
/// for `a ≤ 1`) refined by Halley's method on `P(a, x) − p` — third order, a
/// Newton step scaled by the local curvature of `P`. The ratio `P(a, x)` is the
/// crate's own [`regularized_lower_gamma`] (NOT `statrs::gamma_lr`, which clamps
/// the residual to `−p` for tiny `x`; see that fn's note); the density
/// `f(x) = x^{a−1} e^{−x} / Γ(a)` is evaluated through the same overflow-safe
/// log factorization Numerical Recipes uses (`invgammp`), so the iteration stays
/// finite across a wide range of `a`. A positivity step-halving guard keeps the
/// iterate inside the support.
fn inverse_regularized_lower_gamma(p: f64, a: f64) -> f64 {
    use statrs::function::gamma::ln_gamma;

    if !(a.is_finite() && a > 0.0) {
        return f64::NAN;
    }
    if !p.is_finite() || p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    let gln = ln_gamma(a);
    let a1 = a - 1.0;

    // Initial estimate. For `a > 1` a Wilson–Hilferty transform of a normal
    // quantile (the rational tail approximation avoids depending on the caller's
    // own quantile); for `a ≤ 1` the small-`x` series / large-`x` log form.
    let mut x = if a > 1.0 {
        let pp = if p < 0.5 { p } else { 1.0 - p };
        let t = (-2.0 * pp.ln()).sqrt();
        let mut z = (2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t;
        if p < 0.5 {
            z = -z;
        }
        (a * (1.0 - 1.0 / (9.0 * a) - z / (3.0 * a.sqrt())).powi(3)).max(1.0e-3)
    } else {
        let t = 1.0 - a * (0.253 + a * 0.12);
        if p < t {
            (p / t).powf(1.0 / a)
        } else {
            1.0 - (1.0 - (p - t) / (1.0 - t)).ln()
        }
    };

    // Density factorization constants for `a > 1` (kept overflow-safe in logs).
    let (lna1, afac) = if a > 1.0 {
        let lna1 = a1.ln();
        (lna1, (a1 * (lna1 - 1.0) - gln).exp())
    } else {
        (0.0, 0.0)
    };

    // Halley refinement of the seeded quantile. Halley's cubic convergence
    // reaches `f64` accuracy from the standard Wilson-Hilferty / asymptotic seed
    // in only a few steps; this cap is a generous safety bound, not the expected
    // iteration count, and the loop also exits early via the in-loop tolerance.
    const MAX_HALLEY_STEPS: usize = 16;
    for _ in 0..MAX_HALLEY_STEPS {
        if x <= 0.0 {
            return 0.0;
        }
        let err = regularized_lower_gamma(a, x) - p;
        let dens = if a > 1.0 {
            afac * (-(x - a1) + a1 * (x.ln() - lna1)).exp()
        } else {
            (-x + a1 * x.ln() - gln).exp()
        };
        if !(dens.is_finite() && dens > 0.0) {
            break;
        }
        // Newton step `u = (P(a,x) − p) / f(x)`, then the Halley scaling by the
        // local curvature `f'/f = (a−1)/x − 1`, capped (per NR) so the
        // denominator never collapses below ½.
        let u = err / dens;
        let step = u / (1.0 - 0.5 * (u * (a1 / x - 1.0)).min(1.0));
        x -= step;
        if x <= 0.0 {
            // Overshot the support floor: step back to half the prior iterate.
            x = 0.5 * (x + step);
        }
        if step.abs() < 1.0e-12 * x.max(1.0e-300) {
            break;
        }
    }
    x
}

/// Inverse-link transform per likelihood specification (response scale).
///
/// Uses the EXACT public inverse-link jet, so the log link reports `exp(η)`
/// (finite wherever representable) rather than the solver's clamped value
/// (issue #963).
#[inline]
pub fn try_inverse_link_array(
    likelihood: &LikelihoodSpec,
    eta: ArrayView1<'_, f64>,
) -> Result<Array1<f64>, EstimationError> {
    let mut out = Array1::<f64>::zeros(eta.len());
    for i in 0..eta.len() {
        out[i] = inverse_link_jet_for_family_public(likelihood, eta[i])?.mu;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mixture_link::{state_from_sasspec, state_fromspec};
    use crate::types::{
        InverseLink, LinkComponent, MixtureLinkSpec, ResponseFamily, SasLinkSpec, StandardLink,
    };
    use ndarray::array;

    #[test]
    fn signed_log_sum_exp_propagates_positive_infinities() {
        // A single +∞ positive-sign term dominates ⇒ S = +∞ ⇒ (+∞, +1).
        let (lm, s) = signed_log_sum_exp(&[f64::INFINITY], &[1.0]);
        assert_eq!(lm, f64::INFINITY);
        assert_eq!(s, 1.0);

        // A single +∞ negative-sign term ⇒ S = −∞, encoded as (+∞, −1).
        let (lm, s) = signed_log_sum_exp(&[f64::INFINITY], &[-1.0]);
        assert_eq!(lm, f64::INFINITY);
        assert_eq!(s, -1.0);

        // +∞ on both signs ⇒ indeterminate +∞ − ∞ ⇒ (NaN, 0).
        let (lm, s) = signed_log_sum_exp(&[f64::INFINITY, f64::INFINITY], &[1.0, -1.0]);
        assert!(lm.is_nan());
        assert_eq!(s, 0.0);

        // A finite positive term alongside a +∞ positive term still gives +∞.
        let (lm, s) = signed_log_sum_exp(&[0.0, f64::INFINITY], &[1.0, 1.0]);
        assert_eq!(lm, f64::INFINITY);
        assert_eq!(s, 1.0);

        // −∞ log-magnitudes are exp(−∞)=0 and must be dropped: mixing a finite
        // term with a −∞ term reproduces the lone finite term unchanged.
        let (lm, s) = signed_log_sum_exp(&[2.0, f64::NEG_INFINITY], &[1.0, -1.0]);
        assert!((lm - 2.0).abs() < 1e-12);
        assert_eq!(s, 1.0);

        // Finite sanity check: exp(ln 3) − exp(ln 1) = 2 ⇒ (ln 2, +1).
        let (lm, s) = signed_log_sum_exp(&[3.0_f64.ln(), 1.0_f64.ln()], &[1.0, -1.0]);
        assert!((lm - 2.0_f64.ln()).abs() < 1e-12);
        assert_eq!(s, 1.0);
    }

    #[test]
    fn standard_inverse_link_specs_evaluate() {
        let eta = array![0.1, -0.2, 0.3];
        let likelihood = LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::Logit),
        );
        let mu = try_inverse_link_array(&likelihood, eta.view()).expect("standard logit spec");
        assert_eq!(mu.len(), eta.len());
        assert!(mu.iter().all(|p| p.is_finite() && *p > 0.0 && *p < 1.0));
    }

    #[test]
    fn sas_and_mixture_stateful_inverse_link_evaluates() {
        let eta = array![0.1, -0.2, 0.3];
        let sas_likelihood = LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Sas(
                state_from_sasspec(SasLinkSpec {
                    initial_epsilon: 0.2,
                    initial_log_delta: -0.1,
                })
                .expect("sas state"),
            ),
        );
        let sas = try_inverse_link_array(&sas_likelihood, eta.view()).expect("SAS with params");
        assert_eq!(sas.len(), eta.len());
        assert!(sas.iter().all(|p| p.is_finite() && *p > 0.0 && *p < 1.0));

        let spec = MixtureLinkSpec {
            components: vec![LinkComponent::Probit, LinkComponent::CLogLog],
            initial_rho: array![0.3],
        };
        let state = state_fromspec(&spec).expect("mixture state");
        let mix_likelihood =
            LikelihoodSpec::new(ResponseFamily::Binomial, InverseLink::Mixture(state));
        let mix = try_inverse_link_array(&mix_likelihood, eta.view()).expect("mixture with state");
        assert_eq!(mix.len(), eta.len());
        assert!(mix.iter().all(|p| p.is_finite() && *p > 0.0 && *p < 1.0));
    }

    #[test]
    fn gamma_quantile_matches_known_reference_values() {
        // Reference quantiles for unit-scale Gamma(shape=a) from the regularized
        // lower incomplete gamma inverse (cross-checked against scipy
        // `gamma.ppf(p, a)` to ~1e-6). Spanning a < 1, a = 1 (exponential), and
        // a ≫ 1 exercises every initial-estimate / density branch.
        let cases: [(f64, f64, f64); 9] = [
            // (p, shape a, expected unit-scale quantile)
            (0.025, 4.0, 1.089_865_4),
            (0.5, 4.0, 3.672_060_4),
            (0.975, 4.0, 8.767_273_4),
            (0.025, 1.0, 0.025_317_8), // Exp(1): -ln(1-p)
            (0.975, 1.0, 3.688_879_4),
            (0.5, 0.5, 0.227_468_2),
            (0.99, 0.5, 3.317_448_3),
            (0.025, 50.0, 37.110_963_7),
            (0.975, 50.0, 64.780_598_6),
        ];
        for (p, a, expected) in cases {
            let got = gamma_quantile(p, a, 1.0);
            let rel = (got - expected).abs() / expected.max(1e-12);
            assert!(
                rel < 1e-4,
                "gamma_quantile(p={p}, a={a}) = {got}, expected ≈ {expected} (rel err {rel})"
            );
        }
    }

    #[test]
    fn gamma_quantile_is_consistent_with_the_cdf_round_trip() {
        // The inverse must invert the CDF: P(a, Q(p; a)) = p. Verify across a
        // grid of shapes and probabilities using statrs `gamma_lr` as the CDF.
        use statrs::function::gamma::gamma_lr;
        for &a in &[0.3_f64, 0.75, 1.0, 2.5, 10.0, 80.0] {
            for &p in &[0.001_f64, 0.01, 0.025, 0.25, 0.5, 0.75, 0.975, 0.99, 0.999] {
                let x = gamma_quantile(p, a, 1.0);
                assert!(
                    x.is_finite() && x > 0.0,
                    "non-finite quantile a={a} p={p}: {x}"
                );
                let recovered = gamma_lr(a, x);
                assert!(
                    (recovered - p).abs() < 1e-6,
                    "CDF round-trip failed a={a} p={p}: P(a, {x}) = {recovered}"
                );
            }
        }
    }

    #[test]
    fn gamma_quantile_scale_and_monotonicity() {
        // Scale is a pure multiplier, and the quantile is strictly increasing
        // in p (an equal-tailed interval must order correctly).
        let q_unit = gamma_quantile(0.9, 3.0, 1.0);
        let q_scaled = gamma_quantile(0.9, 3.0, 7.5);
        assert!((q_scaled - 7.5 * q_unit).abs() < 1e-9 * q_scaled.max(1.0));

        let mut prev = 0.0;
        for i in 1..100 {
            let p = i as f64 / 100.0;
            let q = gamma_quantile(p, 2.0, 1.0);
            assert!(q > prev, "quantile not increasing at p={p}: {q} <= {prev}");
            prev = q;
        }
    }

    #[test]
    fn gamma_quantile_rejects_degenerate_parameters() {
        assert!(gamma_quantile(0.5, -1.0, 1.0).is_nan());
        assert!(gamma_quantile(0.5, 1.0, 0.0).is_nan());
        assert!(gamma_quantile(0.5, f64::NAN, 1.0).is_nan());
        assert_eq!(gamma_quantile(0.0, 2.0, 1.0), 0.0);
        assert_eq!(gamma_quantile(-0.1, 2.0, 1.0), 0.0);
        assert!(gamma_quantile(1.0, 2.0, 1.0).is_infinite());
    }

    #[test]
    fn gamma_moment_matched_interval_is_the_exact_conditional_gamma_when_se_vanishes() {
        // With no estimation uncertainty the total predictive variance is the
        // pure observation noise `Var(Y|μ) = φμ²`, and the moment-matched Gamma
        // must coincide *exactly* with the conditional `Gamma(shape = 1/φ,
        // scale = φμ)` (#817). Check against the analytic Gamma quantiles for a
        // shape-4 (φ = 0.25) Gamma at the equal-tailed 2.5%/97.5% levels.
        let phi = 0.25_f64; // shape k = 1/φ = 4
        let mu = 7.5_f64;
        let total_var = phi * mu * mu; // SE(μ̂) = 0
        let (lo, hi) = gamma_moment_matched_interval(mu, total_var, 0.025, 0.975)
            .expect("non-degenerate moment-matched Gamma interval");

        let analytic_lo = gamma_quantile(0.025, 1.0 / phi, phi * mu);
        let analytic_hi = gamma_quantile(0.975, 1.0 / phi, phi * mu);
        assert!(
            (lo - analytic_lo).abs() < 1e-9 * analytic_lo.max(1.0)
                && (hi - analytic_hi).abs() < 1e-9 * analytic_hi.max(1.0),
            "moment-matched interval [{lo}, {hi}] != conditional Gamma \
             [{analytic_lo}, {analytic_hi}]"
        );
    }

    #[test]
    fn gamma_moment_matched_interval_is_right_skewed_not_symmetric() {
        // The whole point of #817: for a right-skewed Gamma the equal-tailed
        // band is *asymmetric* about the mean — the upper gap exceeds the lower
        // gap — and the lower edge sits FAR above the symmetric-band edge
        // `μ·(1 − z/√k)`, which for shape 4 hugs the support floor at ≈ 0.02·μ.
        let phi = 0.25_f64; // shape 4, CV = 0.5
        let mu = 10.0_f64;
        let total_var = phi * mu * mu;
        let z = 1.959_963_984_540_054_f64; // 97.5% standard-normal quantile
        let (lo, hi) =
            gamma_moment_matched_interval(mu, total_var, normal_cdf(-z), normal_cdf(z)).unwrap();

        // Ordered, strictly positive, brackets the mean.
        assert!(
            0.0 < lo && lo < mu && mu < hi,
            "interval [{lo}, {hi}] ∌ μ={mu}"
        );
        // Right skew: the upper gap is the larger one.
        let lower_gap = mu - lo;
        let upper_gap = hi - mu;
        assert!(
            upper_gap > 1.3 * lower_gap,
            "expected a right-skewed band (upper gap ≫ lower gap), got \
             lower_gap={lower_gap}, upper_gap={upper_gap}"
        );
        // The symmetric lower edge would be μ·(1 − z·√φ) = 10·(1 − 1.96·0.5) ≈
        // 0.20 — essentially the support floor. The skew-correct lower edge sits
        // well above it (true Gamma 2.5% quantile ≈ 0.27·μ for shape 4).
        let symmetric_lower = mu * (1.0 - z * phi.sqrt());
        assert!(
            lo > 2.0 * symmetric_lower.max(0.0) + 1.0,
            "skew-correct lower edge {lo} should sit well above the symmetric \
             edge {symmetric_lower}"
        );
    }

    #[test]
    fn gamma_moment_matched_interval_widens_with_estimation_uncertainty() {
        // Adding estimation variance SE(μ̂)² to the observation noise must widen
        // the predictive band (lower edge down, upper edge up) — it is the
        // moment-matched predictive, not just the conditional law.
        let phi = 0.25_f64;
        let mu = 5.0_f64;
        let obs_var = phi * mu * mu;
        let (lo0, hi0) = gamma_moment_matched_interval(mu, obs_var, 0.025, 0.975).unwrap();
        let (lo1, hi1) = gamma_moment_matched_interval(mu, obs_var + 4.0, 0.025, 0.975).unwrap();
        assert!(
            lo1 < lo0 && hi1 > hi0,
            "estimation uncertainty must widen the band: [{lo0},{hi0}] -> [{lo1},{hi1}]"
        );
    }

    #[test]
    fn gamma_moment_matched_interval_rejects_degenerate_and_near_gaussian_inputs() {
        // Non-positive mean / variance, or non-finite inputs => None (caller
        // falls back to the symmetric Gaussian edges).
        assert!(gamma_moment_matched_interval(0.0, 1.0, 0.025, 0.975).is_none());
        assert!(gamma_moment_matched_interval(-1.0, 1.0, 0.025, 0.975).is_none());
        assert!(gamma_moment_matched_interval(1.0, 0.0, 0.025, 0.975).is_none());
        assert!(gamma_moment_matched_interval(1.0, -1.0, 0.025, 0.975).is_none());
        assert!(gamma_moment_matched_interval(f64::NAN, 1.0, 0.025, 0.975).is_none());
        assert!(gamma_moment_matched_interval(1.0, f64::INFINITY, 0.025, 0.975).is_none());
        // A finite, well-conditioned case still returns Some.
        assert!(gamma_moment_matched_interval(3.0, 2.0, 0.025, 0.975).is_some());
    }
}
