use crate::estimate::EstimationError;
use crate::mixture_link::inverse_link_jet_for_family_public;
use crate::types::LikelihoodSpec;
use ndarray::{Array1, ArrayView1};
use statrs::function::beta::{beta_reg, inv_beta_reg};
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

/// Quantile (inverse CDF) of a Beta distribution with shape parameters `a > 0`
/// and `b > 0` at probability `p ∈ (0, 1)`: the value `x ∈ [0, 1]` with
/// `I_x(a, b) = p`, where `I` is the regularized incomplete beta (the Beta CDF).
///
/// `p ≤ 0` maps to the `0` support floor and `p ≥ 1` to the `1` support ceiling;
/// a non-finite or non-positive shape yields `NaN`. Built on the AS 64/109
/// inverse-incomplete-beta routine.
///
/// This is the bounded-support analogue of [`gamma_quantile`]: a Beta response
/// (a proportion modelled by the Beta family) is skewed toward whichever edge
/// its mean is near, so a symmetric `μ ± z·σ` predictive band mis-covers *both*
/// tails even when its width is correct. Equal-tailed Beta quantiles place the
/// right mass in each tail (#1194).
pub fn beta_quantile(p: f64, a: f64, b: f64) -> f64 {
    if !(a.is_finite() && a > 0.0 && b.is_finite() && b > 0.0) {
        return f64::NAN;
    }
    if !p.is_finite() || p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }
    inv_beta_reg(a, b, p)
}

/// Equal-tailed predictive interval for a `(0, 1)`-bounded response modelled as a
/// Beta whose first two moments match a point prediction: mean `mu ∈ (0, 1)` and
/// total predictive variance `total_var` (estimation + observation noise).
/// Returns the pair of Beta quantiles at lower-tail probabilities `p_lo < p_hi` —
/// the skew-correct replacement for a symmetric `mu ± z·σ` band, which for a
/// skewed Beta lands *both* edges below the corresponding true quantile and so
/// mis-covers each tail (#1194).
///
/// Moment matching fixes the precision `φ = a + b = μ(1−μ)/V − 1`, then
/// `a = μφ`, `b = (1−μ)φ`, so the predictive carries exactly the requested mean
/// and variance. When estimation uncertainty vanishes
/// (`total_var → μ(1−μ)/(1+φ₀)`) this is *exact*: `φ → φ₀`, recovering the
/// conditional `Beta(μφ₀, (1−μ)φ₀)`. With nonzero estimation variance it is the
/// moment-matched Beta predictive — the minimal skew-correct widening.
///
/// Returns `None` when the inputs are degenerate (mean outside `(0, 1)`,
/// non-positive variance, non-finite), or when the requested variance reaches
/// the Bernoulli ceiling `μ(1−μ)` (no Beta has that much spread for the given
/// mean) — in which case the caller falls back to the symmetric edges.
pub fn beta_moment_matched_interval(
    mu: f64,
    total_var: f64,
    p_lo: f64,
    p_hi: f64,
) -> Option<(f64, f64)> {
    if !(mu.is_finite() && mu > 0.0 && mu < 1.0 && total_var.is_finite() && total_var > 0.0) {
        return None;
    }
    // A Beta on (0,1) with mean μ can carry variance only up to the Bernoulli
    // limit μ(1−μ); at or beyond it no Beta exists, so the moment match fails.
    let max_var = mu * (1.0 - mu);
    if total_var >= max_var {
        return None;
    }
    let precision = max_var / total_var - 1.0; // = a + b > 0
    let a = mu * precision;
    let b = (1.0 - mu) * precision;
    let q_lo = beta_quantile(p_lo, a, b);
    let q_hi = beta_quantile(p_hi, a, b);
    if q_lo.is_finite() && q_hi.is_finite() && q_hi >= q_lo {
        Some((q_lo, q_hi))
    } else {
        None
    }
}

/// CDF of a Negative-Binomial with mean `μ ≥ 0` and dispersion `θ > 0`
/// (`Var = μ + μ²/θ`) at the integer count `k ≥ 0`:
/// `P(Y ≤ k) = I_{θ/(θ+μ)}(θ, k+1)`, the regularized incomplete beta. Increasing
/// in `k`; `P(Y ≤ 0) = (θ/(θ+μ))^θ` is the zero mass.
#[inline]
fn negative_binomial_cdf_at(k: f64, theta: f64, prob: f64) -> f64 {
    // `prob ∈ (0, 1)`; `beta_reg` requires its last argument in [0, 1].
    beta_reg(theta, k + 1.0, prob.clamp(0.0, 1.0))
}

/// Quantile (inverse CDF) of a Negative-Binomial with mean `μ ≥ 0` and
/// dispersion `θ > 0` at probability `p ∈ (0, 1)`: the smallest integer count
/// `k ≥ 0` with `P(Y ≤ k) ≥ p`, returned as an `f64`.
///
/// `p ≤ 0` maps to the `0` support floor and `p ≥ 1` to `+∞`; a non-finite or
/// non-positive dispersion, or a non-finite / negative mean, yields `NaN`; a
/// zero mean is the degenerate point mass at `0`.
///
/// Unlike the continuous Gamma/Beta quantiles, the NB is *discrete* with a real
/// atom at zero, so its skew-correct predictive band must come from the genuine
/// integer quantiles — a moment-matched *continuous* surrogate (e.g. a Gamma)
/// has no zero atom and grossly over-covers the lower tail on low-mean counts
/// (#1193). A normal-approximation seed brackets the root, then an exact
/// bisection on the incomplete-beta CDF finds the smallest qualifying integer.
pub fn negative_binomial_quantile(p: f64, mu: f64, theta: f64) -> f64 {
    if !(mu.is_finite() && mu >= 0.0 && theta.is_finite() && theta > 0.0) {
        return f64::NAN;
    }
    if !p.is_finite() || p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if mu == 0.0 {
        return 0.0;
    }
    let prob = theta / (theta + mu); // P(success) ∈ (0, 1); mean = θ(1−prob)/prob = μ
    let cdf = |k: f64| negative_binomial_cdf_at(k, theta, prob);

    // The zero atom already covers the requested lower-tail mass on low-mean
    // counts (the common right-skewed case), so short-circuit before bracketing.
    if cdf(0.0) >= p {
        return 0.0;
    }

    // Normal-approximation seed on the NB moments, floored into the support.
    let var = mu + mu * mu / theta;
    let z = standard_normal_quantile(p).unwrap_or(0.0);
    let seed = (mu + z * var.sqrt()).floor().max(1.0);

    // Bracket the smallest integer with CDF ≥ p: `lo` always satisfies
    // CDF(lo) < p (starts at 0, which failed the short-circuit) and `hi`
    // satisfies CDF(hi) ≥ p. Grow geometrically from the seed in whichever
    // direction is needed.
    let mut lo: f64;
    let mut hi: f64;
    if cdf(seed) >= p {
        hi = seed;
        lo = 0.0;
        // Tighten `lo` upward toward `hi` so the bisection starts narrow.
        let mut step = 1.0;
        let mut cand = seed - 1.0;
        while cand > 0.0 && cdf(cand) >= p {
            hi = cand;
            step *= 2.0;
            cand = seed - step;
        }
        if cand > 0.0 {
            lo = cand; // CDF(cand) < p
        }
    } else {
        lo = seed; // CDF(seed) < p
        let mut step = 1.0;
        let mut cand = seed + 1.0;
        // CDF → 1 as k → ∞ and p < 1, so this terminates; the cap is a
        // finite-arithmetic backstop (returns an effectively infinite edge).
        while cdf(cand) < p {
            lo = cand;
            step *= 2.0;
            cand = seed + step;
            if cand > 1.0e18 {
                return f64::INFINITY;
            }
        }
        hi = cand;
    }

    // Bisection for the smallest integer k with CDF(k) ≥ p, maintaining the
    // invariant CDF(lo) < p ≤ CDF(hi).
    while hi - lo > 1.0 {
        let mid = (lo + (hi - lo) / 2.0).floor();
        if cdf(mid) >= p {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    hi
}

/// Equal-tailed predictive interval for a Negative-Binomial count response whose
/// conditional law has mean `mu > 0` and dispersion `theta > 0`, widened for
/// estimation uncertainty to a total predictive variance `total_var`
/// (estimation + observation noise). Returns the pair of integer NB quantiles at
/// lower-tail probabilities `p_lo < p_hi` — the skew-correct, zero-atom-aware
/// replacement for a symmetric `mu ± z·σ` band, which on right-skewed counts
/// sits below the true upper quantile and under-covers the upper tail (#1193).
///
/// Estimation uncertainty is folded in through an *effective dispersion*: an NB
/// with mean `μ` has variance `μ + μ²/θ`, so the `θ_eff` matching the inflated
/// total variance solves `μ + μ²/θ_eff = total_var`, i.e.
/// `θ_eff = μ² / (total_var − μ)`. When estimation uncertainty vanishes
/// (`total_var → μ + μ²/θ`) this is *exact*: `θ_eff → θ`, recovering the
/// conditional `NB(μ, θ)`. With nonzero estimation variance `θ_eff < θ` widens
/// the band — the minimal skew-correct widening that stays inside the NB family.
///
/// Returns `None` for degenerate inputs (non-positive mean / variance,
/// non-finite), or a numerically mis-ordered pair, in which case the caller
/// falls back to the symmetric edges.
pub fn negative_binomial_moment_matched_interval(
    mu: f64,
    theta: f64,
    total_var: f64,
    p_lo: f64,
    p_hi: f64,
) -> Option<(f64, f64)> {
    if !(mu.is_finite()
        && mu > 0.0
        && theta.is_finite()
        && theta > 0.0
        && total_var.is_finite()
        && total_var > 0.0)
    {
        return None;
    }
    // `total_var = SE(μ̂)² + (μ + μ²/θ) > μ` always, so the excess is positive;
    // fall back to the nominal dispersion only if a degenerate caller breaks it.
    let excess = total_var - mu;
    let theta_eff = if excess > 0.0 { mu * mu / excess } else { theta };
    let q_lo = negative_binomial_quantile(p_lo, mu, theta_eff);
    let q_hi = negative_binomial_quantile(p_hi, mu, theta_eff);
    if q_lo.is_finite() && q_hi.is_finite() && q_hi >= q_lo {
        Some((q_lo, q_hi))
    } else {
        None
    }
}

/// CDF of a Poisson with mean `mu ≥ 0` at the integer count `k ≥ 0`:
/// `P(Y ≤ k) = Q(k+1, μ)`, the regularized *upper* incomplete gamma (the standard
/// Poisson↔gamma identity). Increasing in `k`; `P(Y ≤ 0) = e^{−μ}` is the zero mass.
#[inline]
fn poisson_cdf_at(k: f64, mu: f64) -> f64 {
    // P(Y ≤ k) = Q(k+1, μ) = 1 − P(k+1, μ); `regularized_lower_gamma` is `P`.
    (1.0 - regularized_lower_gamma(k + 1.0, mu)).clamp(0.0, 1.0)
}

/// Quantile (inverse CDF) of a Poisson with mean `mu ≥ 0` at probability
/// `p ∈ (0, 1)`: the smallest integer count `k ≥ 0` with `P(Y ≤ k) ≥ p`,
/// returned as an `f64`.
///
/// `p ≤ 0` maps to the `0` support floor and `p ≥ 1` to `+∞`; a non-finite or
/// negative mean yields `NaN`; a zero mean is the degenerate point mass at `0`.
///
/// Like the Negative-Binomial, the Poisson is *discrete* with a real atom at
/// zero, so its skew-correct predictive band must come from the genuine integer
/// quantiles — a symmetric `μ ± z·σ` band sits below the true upper quantile on
/// low-rate counts and under-covers the upper tail (the #817 defect, Poisson
/// sibling of #1193). A normal-approximation seed brackets the root, then an
/// exact bisection on the gamma-tail CDF finds the smallest qualifying integer.
pub fn poisson_quantile(p: f64, mu: f64) -> f64 {
    if !(mu.is_finite() && mu >= 0.0) {
        return f64::NAN;
    }
    if !p.is_finite() || p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if mu == 0.0 {
        return 0.0;
    }
    let cdf = |k: f64| poisson_cdf_at(k, mu);

    // The zero atom already covers the requested lower-tail mass on low-rate
    // counts (the common right-skewed case), so short-circuit before bracketing.
    if cdf(0.0) >= p {
        return 0.0;
    }

    // Normal-approximation seed on the Poisson moments (Var = μ), floored into
    // the support.
    let z = standard_normal_quantile(p).unwrap_or(0.0);
    let seed = (mu + z * mu.sqrt()).floor().max(1.0);

    // Bracket the smallest integer with CDF ≥ p: `lo` always satisfies
    // CDF(lo) < p (starts at 0, which failed the short-circuit) and `hi`
    // satisfies CDF(hi) ≥ p. Grow geometrically from the seed in whichever
    // direction is needed.
    let mut lo: f64;
    let mut hi: f64;
    if cdf(seed) >= p {
        hi = seed;
        lo = 0.0;
        let mut step = 1.0;
        let mut cand = seed - 1.0;
        while cand > 0.0 && cdf(cand) >= p {
            hi = cand;
            step *= 2.0;
            cand = seed - step;
        }
        if cand > 0.0 {
            lo = cand; // CDF(cand) < p
        }
    } else {
        lo = seed; // CDF(seed) < p
        let mut step = 1.0;
        let mut cand = seed + 1.0;
        // CDF → 1 as k → ∞ and p < 1, so this terminates; the cap is a
        // finite-arithmetic backstop (returns an effectively infinite edge).
        while cdf(cand) < p {
            lo = cand;
            step *= 2.0;
            cand = seed + step;
            if cand > 1.0e18 {
                return f64::INFINITY;
            }
        }
        hi = cand;
    }

    // Bisection for the smallest integer k with CDF(k) ≥ p, maintaining the
    // invariant CDF(lo) < p ≤ CDF(hi).
    while hi - lo > 1.0 {
        let mid = (lo + (hi - lo) / 2.0).floor();
        if cdf(mid) >= p {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    hi
}

/// Equal-tailed predictive interval for a Poisson count response whose
/// conditional law has mean `mu > 0` (so `Var(Y|μ) = μ`), widened for estimation
/// uncertainty to a total predictive variance `total_var ≥ μ` (estimation +
/// observation noise). Returns the pair of integer quantiles at lower-tail
/// probabilities `p_lo < p_hi` — the skew-correct, zero-atom-aware replacement
/// for a symmetric `mu ± z·σ` band, which on low-rate counts sits below the true
/// upper quantile and under-covers the upper tail (the #817 defect, Poisson
/// sibling of #1193).
///
/// A pure Poisson has no free dispersion parameter to absorb estimation
/// uncertainty, so the widening is carried by the *conjugate over-dispersed count
/// law*: if the point estimate `μ̂` carries (approximately) a Gamma sampling
/// uncertainty with mean `μ` and variance `SE(μ̂)² = total_var − μ`, the posterior
/// predictive for a *new* Poisson draw is exactly a Negative-Binomial — the
/// Gamma–Poisson mixture — with mean `μ` and dispersion `θ_eff = μ² / (total_var − μ)`
/// (matching the inflated variance `μ + μ²/θ_eff = total_var`). As estimation
/// uncertainty vanishes (`total_var → μ`, `θ_eff → ∞`) the NB collapses to the
/// *exact* conditional Poisson, which is then used directly — both because it is
/// the correct limit and because an NB with `θ → ∞` is numerically degenerate.
/// The two regimes agree (both are integer quantiles that coincide once `θ_eff`
/// is large), so the switch introduces no discontinuity in the emitted edge.
///
/// Returns `None` for degenerate inputs (non-positive mean, non-finite, or a
/// total variance below the Poisson floor `μ`), or a numerically mis-ordered
/// pair, in which case the caller falls back to the symmetric edges.
pub fn poisson_moment_matched_interval(
    mu: f64,
    total_var: f64,
    p_lo: f64,
    p_hi: f64,
) -> Option<(f64, f64)> {
    if !(mu.is_finite() && mu > 0.0 && total_var.is_finite() && total_var > 0.0) {
        return None;
    }
    // Estimation uncertainty inflates the count variance beyond the Poisson
    // floor `Var(Y|μ) = μ`; the excess is the (approximate) sampling variance of
    // `μ̂`. A `total_var` below `μ` is degenerate (a caller broke the contract).
    let excess = total_var - mu;
    if excess < 0.0 {
        return None;
    }
    // Above this effective dispersion the NB surrogate and the conditional
    // Poisson agree to far more than the integer resolution of the quantile, and
    // `negative_binomial_quantile`'s `I_{θ/(θ+μ)}(θ, k+1)` is better conditioned
    // as the exact Poisson; below it the NB widening is genuine.
    const THETA_EFF_MAX: f64 = 1.0e9;
    let theta_eff = if excess > 0.0 {
        mu * mu / excess
    } else {
        f64::INFINITY
    };
    let (q_lo, q_hi) = if theta_eff > THETA_EFF_MAX {
        (poisson_quantile(p_lo, mu), poisson_quantile(p_hi, mu))
    } else {
        (
            negative_binomial_quantile(p_lo, mu, theta_eff),
            negative_binomial_quantile(p_hi, mu, theta_eff),
        )
    };
    if q_lo.is_finite() && q_hi.is_finite() && q_hi >= q_lo {
        Some((q_lo, q_hi))
    } else {
        None
    }
}

/// CDF of a Tweedie compound Poisson–Gamma response (power `1 < p < 2`) with
/// mean `mu > 0` and dispersion `phi > 0` at `y ≥ 0`:
/// `P(Y ≤ y) = e^{−λ} + Σ_{k≥1} Poisson(k; λ)·GammaCDF(y; kα, γ)`, the mixture of
/// a point mass at zero (no jumps) and `k` i.i.d. Gamma jumps. The Tweedie
/// parameters map to `λ = μ^{2−p} / (φ(2−p))` (Poisson mean number of jumps),
/// Gamma jump shape `α = (2−p)/(p−1)` and scale `γ = φ(p−1)μ^{p−1}`, which
/// reproduce `E[Y] = μ` and `Var(Y) = φμ^p`.
///
/// The zero atom `e^{−λ}` is returned directly at `y = 0`. For `y > 0` the
/// Poisson weights are accumulated in log-space and the series is truncated once
/// the remaining Poisson mass beyond the current term is negligible — the Gamma
/// CDF factor is ≤ 1, so the unsummed tail is bounded by the Poisson survival.
#[inline]
fn tweedie_cdf_at(y: f64, mu: f64, phi: f64, power: f64) -> f64 {
    if !(y.is_finite() && y >= 0.0) {
        return f64::NAN;
    }
    let lambda = mu.powf(2.0 - power) / (phi * (2.0 - power));
    let alpha = (2.0 - power) / (power - 1.0);
    let scale = phi * (power - 1.0) * mu.powf(power - 1.0);
    let zero_mass = (-lambda).exp();
    if y <= 0.0 {
        return zero_mass;
    }
    let x = y / scale; // unit-scale Gamma argument
    // Poisson(k; λ) weights via a log-space recurrence: w_k = w_{k-1}·λ/k.
    // Sum k ≥ 1 only; the k = 0 term contributes the zero atom (GammaCDF = 1 at
    // any y > 0 for shape 0 is the degenerate point mass already in `zero_mass`).
    let mut acc = zero_mass; // P(Y ≤ y) includes the no-jump mass (Y = 0 ≤ y)
    let mut ln_w = -lambda; // ln Poisson(0; λ)
    // Centre the truncation window on the Poisson mode so very large λ stays cheap.
    let k_max = (lambda + 10.0 * lambda.sqrt()).ceil() as usize + 50;
    let mut remaining = 1.0 - zero_mass; // Poisson mass still unaccounted for (k ≥ 1)
    for k in 1..=k_max {
        ln_w += lambda.ln() - (k as f64).ln();
        let w = ln_w.exp();
        remaining -= w;
        // GammaCDF(y; kα, γ) = P(kα, y/γ) on the unit scale.
        acc += w * regularized_lower_gamma(alpha * k as f64, x);
        if remaining <= 1e-15 && k as f64 > lambda {
            break;
        }
    }
    acc.clamp(0.0, 1.0)
}

/// Quantile (inverse CDF) of a Tweedie compound Poisson–Gamma response
/// (power `1 < p < 2`) with mean `mu > 0` and dispersion `phi > 0` at
/// probability `q ∈ (0, 1)`: the value `y ≥ 0` with `P(Y ≤ y) = q`.
///
/// `q ≤ 0` maps to the `0` support floor and `q ≥ 1` to `+∞`. If the requested
/// lower-tail probability is at or below the zero atom `e^{−λ}` the quantile is
/// exactly `0` (the common right-skewed lower-tail case). Otherwise a normal seed
/// on the Tweedie moments brackets the root, which is then refined by bisection
/// on [`tweedie_cdf_at`] — the continuous part above the atom is strictly
/// increasing, so the bracket converges.
pub fn tweedie_quantile(q: f64, mu: f64, phi: f64, power: f64) -> f64 {
    if !(mu.is_finite()
        && mu > 0.0
        && phi.is_finite()
        && phi > 0.0
        && power.is_finite()
        && power > 1.0
        && power < 2.0)
    {
        return f64::NAN;
    }
    if !q.is_finite() || q <= 0.0 {
        return 0.0;
    }
    if q >= 1.0 {
        return f64::INFINITY;
    }
    let lambda = mu.powf(2.0 - power) / (phi * (2.0 - power));
    let zero_mass = (-lambda).exp();
    // The zero atom carries the lower-tail mass: q at or below it ⇒ quantile 0.
    if q <= zero_mass {
        return 0.0;
    }

    // Normal-approximation seed on the Tweedie moments, then geometric bracketing.
    let var = phi * mu.powf(power);
    let z = standard_normal_quantile(q).unwrap_or(0.0);
    let mut hi = (mu + z * var.sqrt()).max(scale_floor(mu));
    let cdf = |y: f64| tweedie_cdf_at(y, mu, phi, power);

    // Grow `hi` until it covers `q`; `lo` stays below it. CDF → 1 as y → ∞.
    let mut lo = 0.0_f64;
    let mut guard = 0;
    while cdf(hi) < q {
        lo = hi;
        hi *= 2.0;
        guard += 1;
        if guard > 200 || hi > 1.0e18 {
            return f64::INFINITY;
        }
    }

    // Bisection on the strictly-increasing continuous part above the atom.
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if cdf(mid) < q {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo <= (hi.abs() + 1.0) * 1e-12 {
            break;
        }
    }
    0.5 * (lo + hi)
}

/// A strictly-positive starting scale for the Tweedie bracket: a small fraction
/// of the mean keeps the initial `hi` inside the support when the normal seed
/// underflows to or below zero on a heavily right-skewed row.
#[inline]
fn scale_floor(mu: f64) -> f64 {
    (mu * 1e-3).max(f64::MIN_POSITIVE)
}

/// Equal-tailed predictive interval for a Tweedie compound Poisson–Gamma
/// response (power `1 < p < 2`) whose conditional law has mean `mu > 0` and
/// dispersion `phi > 0`, widened for estimation uncertainty to a total
/// predictive variance `total_var` (estimation + observation noise). Returns the
/// pair of Tweedie quantiles at lower-tail probabilities `p_lo < p_hi` — the
/// skew-correct, zero-atom-aware replacement for a symmetric `mu ± z·σ` band,
/// which on a right-skewed Tweedie sits below the true upper quantile and
/// under-covers the upper tail (the #817 defect, Tweedie sibling of #1193).
///
/// Estimation uncertainty is folded in through an *effective dispersion*: a
/// Tweedie with mean `μ` has variance `φμ^p`, so the `φ_eff` matching the
/// inflated total variance solves `φ_eff·μ^p = total_var`, i.e.
/// `φ_eff = total_var / μ^p`. When estimation uncertainty vanishes
/// (`total_var → φμ^p`) this is *exact*: `φ_eff → φ`, recovering the conditional
/// Tweedie. With nonzero estimation variance `φ_eff > φ` widens the band inside
/// the Tweedie family — the minimal skew-correct widening. Unlike a moment-
/// matched Gamma surrogate, this keeps the genuine zero atom, so it does not
/// over-cover the lower tail on low-mean rows (#1193).
///
/// Returns `None` for degenerate inputs (non-positive mean / variance,
/// non-finite, power outside `(1, 2)`) or a mis-ordered pair, in which case the
/// caller falls back to the symmetric edges.
pub fn tweedie_moment_matched_interval(
    mu: f64,
    phi: f64,
    power: f64,
    total_var: f64,
    p_lo: f64,
    p_hi: f64,
) -> Option<(f64, f64)> {
    if !(mu.is_finite()
        && mu > 0.0
        && phi.is_finite()
        && phi > 0.0
        && power.is_finite()
        && power > 1.0
        && power < 2.0
        && total_var.is_finite()
        && total_var > 0.0)
    {
        return None;
    }
    let phi_eff = total_var / mu.powf(power);
    if !(phi_eff.is_finite() && phi_eff > 0.0) {
        return None;
    }
    let q_lo = tweedie_quantile(p_lo, mu, phi_eff, power);
    let q_hi = tweedie_quantile(p_hi, mu, phi_eff, power);
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
/// Uses the standard rational/Wilson–Hilferty initial estimate, except in the
/// extreme lower tail where the exact small-`x` seed
/// `exp((ln p + ln Γ(a + 1)) / a)` follows from `P(a, x) ~ x^a / Γ(a + 1)`.
/// For `a ≤ 1` it keeps the Numerical Recipes series/log initial estimate. The
/// seed is refined by Halley's method on `P(a, x) − p` — third order, a Newton
/// step scaled by the local curvature of `P`. The ratio `P(a, x)` is the crate's
/// own [`regularized_lower_gamma`] (NOT `statrs::gamma_lr`, which clamps the
/// residual to `−p` for tiny `x`; see that fn's note); the density
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
    // quantile works away from the extreme lower tail; there, the small-`x`
    // analytic seed is essentially exact. Both seeds feed the same Halley polish,
    // so the crossover is continuous at the converged quantile.
    let mut x = if a > 1.0 {
        let pp = if p < 0.5 { p } else { 1.0 - p };
        let t = (-2.0 * pp.ln()).sqrt();
        let mut z = (2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t;
        if p < 0.5 {
            z = -z;
        }
        let wh_inner = 1.0 - 1.0 / (9.0 * a) - z / (3.0 * a.sqrt());
        let wh_seed = if wh_inner > 0.0 {
            a * wh_inner.powi(3)
        } else {
            f64::NAN
        };
        let analytic_seed = ((p.ln() + ln_gamma(a + 1.0)) / a).exp();
        if analytic_seed == 0.0 {
            return 0.0;
        }
        if !wh_seed.is_finite() || wh_seed <= 0.0 || wh_seed < 1.0e-2 || analytic_seed < 1.0e-2 {
            analytic_seed
        } else {
            wh_seed
        }
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
    fn gamma_quantile_handles_extreme_lower_tail_for_shape_two() {
        let got = gamma_quantile(1.0e-300, 2.0, 1.0);
        let expected = 1.414_213_562_373_095_1e-150;
        let rel = (got - expected).abs() / expected;
        assert!(
            rel < 1.0e-6,
            "gamma_quantile(1e-300, 2, 1) = {got}, expected {expected} (rel err {rel})"
        );
    }

    #[test]
    fn gamma_quantile_round_trips_extreme_lower_tail_for_shape_above_one() {
        for &a in &[1.5_f64, 2.0, 5.0, 20.0] {
            for &p in &[1.0e-300_f64, 1.0e-100, 1.0e-12, 1.0e-3, 0.5, 0.999] {
                let x = gamma_quantile(p, a, 1.0);
                assert!(
                    x.is_finite() && x >= 0.0,
                    "non-finite quantile a={a} p={p}: {x}"
                );
                let recovered = regularized_lower_gamma(a, x);
                let rel = (recovered - p).abs() / p;
                assert!(
                    rel < 1.0e-6,
                    "round-trip failed a={a} p={p}: q={x}, P(a,q)={recovered}, rel err {rel}"
                );
            }
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
    fn regularized_lower_gamma_is_accurate_and_unclamped_below_statrs_floor() {
        use statrs::function::gamma::{gamma_lr, ln_gamma};

        // (1) Agrees with statrs `gamma_lr` everywhere statrs is itself valid
        // (arguments well above its `x ≤ 1.11e-15` clamp), across both the
        // series (x < a+1) and continued-fraction (x ≥ a+1) branches.
        for &a in &[0.05_f64, 0.3, 1.0, 2.5, 50.0] {
            for &x in &[1e-6_f64, 0.01, 0.5, 1.0, 3.0, 25.0, 120.0] {
                let ours = regularized_lower_gamma(a, x);
                let theirs = gamma_lr(a, x);
                assert!(
                    (ours - theirs).abs() < 1e-12,
                    "P({a},{x}): ours={ours} statrs={theirs}"
                );
                assert!(
                    (0.0..=1.0).contains(&ours),
                    "P({a},{x})={ours} out of [0,1]"
                );
            }
        }

        // (2) Exp(1) closed form P(1, x) = 1 − e^{−x}.
        for &x in &[1e-3_f64, 0.25, 2.0, 9.0] {
            assert!((regularized_lower_gamma(1.0, x) - (1.0 - (-x).exp())).abs() < 1e-13);
        }

        // (3) The regression heart of #1018: for x far below statrs's clamp the
        // CDF must remain a faithful, nonzero value, not snap to 0. Compare to
        // the small-x leading order P(a, x) ≈ x^a / Γ(a+1).
        for &(a, x) in &[(0.05_f64, 1e-20_f64), (0.1, 1e-25), (0.02, 1e-40)] {
            assert_eq!(
                gamma_lr(a, x),
                0.0,
                "precondition: statrs clamps P({a},{x}) to 0"
            );
            let ours = regularized_lower_gamma(a, x);
            let leading = (a * x.ln() - ln_gamma(a + 1.0)).exp();
            assert!(ours > 0.0, "P({a},{x})={ours} clamped to 0 like statrs");
            assert!(
                (ours - leading).abs() < 1e-9 * leading,
                "P({a},{x})={ours}, leading order {leading}"
            );
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

    #[test]
    fn beta_quantile_matches_known_reference_values() {
        // Reference Beta quantiles cross-checked against scipy `beta.ppf(p,a,b)`.
        // Spans symmetric (a=b), left-skewed (a<b), right-skewed (a>b), and a
        // high-precision case to exercise the inverse-incomplete-beta branches.
        let cases: [(f64, f64, f64, f64); 8] = [
            // (p, a, b, expected) — scipy `beta.ppf`.
            (0.025, 2.0, 2.0, 0.094_299_3),
            (0.975, 2.0, 2.0, 0.905_700_7),
            (0.5, 2.0, 2.0, 0.5),
            (0.025, 0.8, 4.0, 0.002_339_1),
            (0.975, 0.8, 4.0, 0.564_717_3),
            (0.025, 5.0, 1.5, 0.408_549_1),
            (0.5, 20.0, 80.0, 0.197_994_8),
            (0.975, 20.0, 80.0, 0.283_367_6),
        ];
        for (p, a, b, expected) in cases {
            let got = beta_quantile(p, a, b);
            let abs = (got - expected).abs();
            assert!(
                abs < 1e-5,
                "beta_quantile(p={p}, a={a}, b={b}) = {got}, expected ≈ {expected} (abs err {abs})"
            );
        }
    }

    #[test]
    fn beta_quantile_boundaries_and_degeneracy() {
        // p at/over the support boundaries map to 0 / 1; bad shapes => NaN; and
        // the quantile is strictly increasing in p.
        assert_eq!(beta_quantile(0.0, 2.0, 3.0), 0.0);
        assert_eq!(beta_quantile(-0.5, 2.0, 3.0), 0.0);
        assert_eq!(beta_quantile(1.0, 2.0, 3.0), 1.0);
        assert_eq!(beta_quantile(1.5, 2.0, 3.0), 1.0);
        assert!(beta_quantile(0.5, -1.0, 3.0).is_nan());
        assert!(beta_quantile(0.5, 2.0, 0.0).is_nan());
        assert!(beta_quantile(0.5, f64::NAN, 3.0).is_nan());
        let mut prev = 0.0;
        for i in 1..100 {
            let p = i as f64 / 100.0;
            let q = beta_quantile(p, 3.0, 5.0);
            assert!(q > prev, "beta quantile not increasing at p={p}: {q} <= {prev}");
            prev = q;
        }
    }

    #[test]
    fn beta_moment_matched_interval_is_the_exact_conditional_beta_when_se_vanishes() {
        // With no estimation uncertainty the total predictive variance is the
        // pure observation noise `μ(1−μ)/(1+φ)`, and the moment-matched Beta must
        // coincide *exactly* with the conditional `Beta(μφ, (1−μ)φ)` (#1194).
        let phi = 8.0_f64;
        let mu = 0.2_f64;
        let total_var = mu * (1.0 - mu) / (1.0 + phi); // SE(μ̂) = 0
        let (lo, hi) = beta_moment_matched_interval(mu, total_var, 0.025, 0.975)
            .expect("non-degenerate moment-matched Beta interval");
        let analytic_lo = beta_quantile(0.025, mu * phi, (1.0 - mu) * phi);
        let analytic_hi = beta_quantile(0.975, mu * phi, (1.0 - mu) * phi);
        assert!(
            (lo - analytic_lo).abs() < 1e-9 && (hi - analytic_hi).abs() < 1e-9,
            "moment-matched interval [{lo}, {hi}] != conditional Beta [{analytic_lo}, {analytic_hi}]"
        );
    }

    #[test]
    fn beta_moment_matched_interval_is_skewed_not_symmetric() {
        // For a small-mean Beta the equal-tailed band is asymmetric about μ (the
        // upper gap exceeds the lower gap) and the lower edge sits well above the
        // symmetric edge `μ − z·σ`, which on this data dives below 0.
        let phi = 8.0_f64;
        let mu = 0.15_f64;
        let total_var = mu * (1.0 - mu) / (1.0 + phi);
        let z = 1.959_963_984_540_054_f64;
        let (lo, hi) =
            beta_moment_matched_interval(mu, total_var, normal_cdf(-z), normal_cdf(z)).unwrap();
        assert!(0.0 < lo && lo < mu && mu < hi && hi < 1.0, "interval [{lo},{hi}] ∌ μ={mu}");
        let lower_gap = mu - lo;
        let upper_gap = hi - mu;
        assert!(
            upper_gap > 1.2 * lower_gap,
            "expected a right-skewed band (upper gap > lower gap): lower={lower_gap}, upper={upper_gap}"
        );
        let symmetric_lower = mu - z * total_var.sqrt();
        assert!(
            symmetric_lower < 0.0 && lo > 0.0,
            "skew-correct lower edge {lo} should stay positive where the symmetric edge {symmetric_lower} goes negative"
        );
    }

    #[test]
    fn beta_moment_matched_interval_rejects_degenerate_and_over_dispersed_inputs() {
        // Mean outside (0,1), non-positive variance, non-finite => None.
        assert!(beta_moment_matched_interval(0.0, 0.01, 0.025, 0.975).is_none());
        assert!(beta_moment_matched_interval(1.0, 0.01, 0.025, 0.975).is_none());
        assert!(beta_moment_matched_interval(-0.1, 0.01, 0.025, 0.975).is_none());
        assert!(beta_moment_matched_interval(0.3, 0.0, 0.025, 0.975).is_none());
        assert!(beta_moment_matched_interval(f64::NAN, 0.01, 0.025, 0.975).is_none());
        // Variance at/over the Bernoulli ceiling μ(1−μ): no Beta matches => None.
        assert!(beta_moment_matched_interval(0.5, 0.25, 0.025, 0.975).is_none());
        assert!(beta_moment_matched_interval(0.5, 0.30, 0.025, 0.975).is_none());
        // A well-conditioned case still returns Some.
        assert!(beta_moment_matched_interval(0.4, 0.02, 0.025, 0.975).is_some());
    }

    #[test]
    fn beta_moment_matched_interval_widens_with_estimation_uncertainty() {
        let phi = 8.0_f64;
        let mu = 0.3_f64;
        let obs_var = mu * (1.0 - mu) / (1.0 + phi);
        let (lo0, hi0) = beta_moment_matched_interval(mu, obs_var, 0.025, 0.975).unwrap();
        let (lo1, hi1) =
            beta_moment_matched_interval(mu, obs_var + 0.01, 0.025, 0.975).unwrap();
        assert!(
            lo1 < lo0 && hi1 > hi0,
            "estimation uncertainty must widen the band: [{lo0},{hi0}] -> [{lo1},{hi1}]"
        );
    }

    #[test]
    fn negative_binomial_quantile_matches_known_reference_values() {
        // Reference NB quantiles cross-checked against scipy
        // `nbinom.ppf(p, n=θ, prob=θ/(θ+μ))` — the integer count k with the
        // smallest CDF ≥ p. Spans the zero-atom lower tail, the right-skewed
        // upper tail, and a larger-mean near-Gaussian case.
        let cases: [(f64, f64, f64, f64); 8] = [
            // (p, μ, θ, expected integer quantile)
            (0.025, 1.6, 1.5, 0.0), // zero mass ≈ 0.34 > 0.025 ⇒ lower edge 0
            (0.5, 1.6, 1.5, 1.0),
            (0.975, 1.6, 1.5, 6.0),
            (0.99, 1.6, 1.5, 8.0),
            (0.025, 20.0, 5.0, 5.0),
            (0.975, 20.0, 5.0, 43.0),
            (0.5, 20.0, 5.0, 19.0),
            (0.975, 0.5, 2.0, 3.0),
        ];
        for (p, mu, theta, expected) in cases {
            let got = negative_binomial_quantile(p, mu, theta);
            assert_eq!(
                got, expected,
                "negative_binomial_quantile(p={p}, μ={mu}, θ={theta}) = {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn negative_binomial_quantile_is_a_valid_cdf_inverse() {
        // The returned integer k must be the *smallest* with CDF(k) ≥ p:
        // CDF(k) ≥ p and (for k ≥ 1) CDF(k−1) < p, across a grid of (μ, θ, p).
        use statrs::function::beta::beta_reg;
        for &mu in &[0.3_f64, 1.6, 5.0, 25.0, 120.0] {
            for &theta in &[0.5_f64, 1.5, 5.0, 40.0] {
                let prob = theta / (theta + mu);
                for &p in &[0.01_f64, 0.025, 0.1, 0.5, 0.9, 0.975, 0.99] {
                    let k = negative_binomial_quantile(p, mu, theta);
                    assert!(k.is_finite() && k >= 0.0 && k.fract() == 0.0, "non-integer k={k}");
                    let cdf_k = beta_reg(theta, k + 1.0, prob);
                    assert!(
                        cdf_k + 1e-12 >= p,
                        "CDF({k}) = {cdf_k} < p = {p} (μ={mu}, θ={theta})"
                    );
                    if k >= 1.0 {
                        let cdf_below = beta_reg(theta, k, prob);
                        assert!(
                            cdf_below < p,
                            "k={k} not minimal: CDF({}) = {cdf_below} ≥ p = {p} (μ={mu}, θ={theta})",
                            k - 1.0
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn negative_binomial_quantile_boundaries_and_degeneracy() {
        assert_eq!(negative_binomial_quantile(0.0, 2.0, 1.5), 0.0);
        assert_eq!(negative_binomial_quantile(-0.1, 2.0, 1.5), 0.0);
        assert!(negative_binomial_quantile(1.0, 2.0, 1.5).is_infinite());
        assert_eq!(negative_binomial_quantile(0.5, 0.0, 1.5), 0.0); // point mass at 0
        assert!(negative_binomial_quantile(0.5, -1.0, 1.5).is_nan());
        assert!(negative_binomial_quantile(0.5, 2.0, 0.0).is_nan());
        assert!(negative_binomial_quantile(0.5, 2.0, f64::NAN).is_nan());
        // Monotone non-decreasing in p (discrete ⇒ plateaus allowed).
        let mut prev = 0.0;
        for i in 1..100 {
            let p = i as f64 / 100.0;
            let q = negative_binomial_quantile(p, 4.0, 2.0);
            assert!(q >= prev, "NB quantile decreased at p={p}: {q} < {prev}");
            prev = q;
        }
    }

    #[test]
    fn negative_binomial_moment_matched_interval_is_exact_conditional_when_se_vanishes() {
        // SE(μ̂) = 0 ⇒ total_var = μ + μ²/θ ⇒ θ_eff = θ, recovering the exact
        // conditional NB quantiles.
        let mu = 1.6_f64;
        let theta = 1.5_f64;
        let total_var = mu + mu * mu / theta;
        let (lo, hi) =
            negative_binomial_moment_matched_interval(mu, theta, total_var, 0.025, 0.975).unwrap();
        assert_eq!(lo, negative_binomial_quantile(0.025, mu, theta));
        assert_eq!(hi, negative_binomial_quantile(0.975, mu, theta));
    }

    #[test]
    fn negative_binomial_moment_matched_interval_widens_with_estimation_uncertainty() {
        // Adding estimation variance lowers θ_eff (more overdispersion) and must
        // not shrink the band; with enough added variance the upper edge grows.
        let mu = 8.0_f64;
        let theta = 4.0_f64;
        let obs_var = mu + mu * mu / theta;
        let (lo0, hi0) =
            negative_binomial_moment_matched_interval(mu, theta, obs_var, 0.025, 0.975).unwrap();
        let (lo1, hi1) =
            negative_binomial_moment_matched_interval(mu, theta, obs_var + 40.0, 0.025, 0.975)
                .unwrap();
        assert!(lo1 <= lo0 && hi1 > hi0, "band did not widen: [{lo0},{hi0}] -> [{lo1},{hi1}]");
    }

    #[test]
    fn negative_binomial_moment_matched_interval_rejects_degenerate_inputs() {
        assert!(negative_binomial_moment_matched_interval(0.0, 1.5, 1.0, 0.025, 0.975).is_none());
        assert!(negative_binomial_moment_matched_interval(-1.0, 1.5, 1.0, 0.025, 0.975).is_none());
        assert!(negative_binomial_moment_matched_interval(2.0, 0.0, 1.0, 0.025, 0.975).is_none());
        assert!(negative_binomial_moment_matched_interval(2.0, 1.5, 0.0, 0.025, 0.975).is_none());
        assert!(
            negative_binomial_moment_matched_interval(f64::NAN, 1.5, 1.0, 0.025, 0.975).is_none()
        );
        assert!(negative_binomial_moment_matched_interval(2.0, 1.5, 6.0, 0.025, 0.975).is_some());
    }

    #[test]
    fn poisson_quantile_matches_known_reference_values() {
        // Reference integer quantiles from scipy.stats.poisson.ppf.
        let cases: [(f64, f64, f64); 9] = [
            // (p, μ, expected integer quantile)
            (0.025, 1.6, 0.0), // zero mass e^{−1.6} ≈ 0.20 < 0.025? no: 0.20 > 0.025 ⇒ 0
            (0.5, 1.6, 1.0),
            (0.975, 1.6, 4.0),
            (0.99, 1.6, 5.0),
            (0.025, 20.0, 12.0),
            (0.975, 20.0, 29.0),
            (0.5, 20.0, 20.0),
            (0.975, 0.5, 2.0),
            (0.025, 0.5, 0.0),
        ];
        for (p, mu, expected) in cases {
            let got = poisson_quantile(p, mu);
            assert_eq!(
                got, expected,
                "poisson_quantile(p={p}, μ={mu}) = {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn poisson_quantile_is_a_valid_cdf_inverse() {
        // The returned integer k must be the *smallest* with CDF(k) ≥ p:
        // CDF(k) ≥ p and (for k ≥ 1) CDF(k−1) < p, across a grid of (μ, p).
        for &mu in &[0.3_f64, 1.6, 5.0, 25.0, 120.0] {
            for &p in &[0.01_f64, 0.025, 0.1, 0.5, 0.9, 0.975, 0.99] {
                let k = poisson_quantile(p, mu);
                assert!(k.is_finite() && k >= 0.0 && k.fract() == 0.0, "non-integer k={k}");
                let cdf_k = poisson_cdf_at(k, mu);
                assert!(cdf_k + 1e-12 >= p, "CDF({k}) = {cdf_k} < p = {p} (μ={mu})");
                if k >= 1.0 {
                    let cdf_below = poisson_cdf_at(k - 1.0, mu);
                    assert!(
                        cdf_below < p,
                        "k={k} not minimal: CDF({}) = {cdf_below} ≥ p = {p} (μ={mu})",
                        k - 1.0
                    );
                }
            }
        }
    }

    #[test]
    fn poisson_quantile_boundaries_and_degeneracy() {
        assert_eq!(poisson_quantile(0.0, 2.0), 0.0);
        assert_eq!(poisson_quantile(-0.1, 2.0), 0.0);
        assert!(poisson_quantile(1.0, 2.0).is_infinite());
        assert_eq!(poisson_quantile(0.5, 0.0), 0.0); // point mass at 0
        assert!(poisson_quantile(0.5, -1.0).is_nan());
        assert!(poisson_quantile(0.5, f64::NAN).is_nan());
        // Monotone non-decreasing in p (discrete ⇒ plateaus allowed).
        let mut prev = 0.0;
        for i in 1..100 {
            let p = i as f64 / 100.0;
            let q = poisson_quantile(p, 4.0);
            assert!(q >= prev, "Poisson quantile decreased at p={p}: {q} < {prev}");
            prev = q;
        }
    }

    #[test]
    fn poisson_moment_matched_interval_is_exact_conditional_when_se_vanishes() {
        // SE(μ̂) = 0 ⇒ total_var = μ ⇒ θ_eff = ∞, recovering the exact conditional
        // Poisson quantiles directly (no NB widening).
        for &mu in &[0.5_f64, 1.6, 20.0] {
            let (lo, hi) = poisson_moment_matched_interval(mu, mu, 0.025, 0.975).unwrap();
            assert_eq!(lo, poisson_quantile(0.025, mu));
            assert_eq!(hi, poisson_quantile(0.975, mu));
        }
    }

    #[test]
    fn poisson_moment_matched_interval_widens_with_estimation_uncertainty() {
        // Adding estimation variance lowers θ_eff (genuine overdispersion) and
        // must not shrink the band; with enough added variance the upper edge
        // grows beyond the conditional Poisson quantile.
        let mu = 20.0_f64;
        let (lo0, hi0) = poisson_moment_matched_interval(mu, mu, 0.025, 0.975).unwrap();
        let (lo1, hi1) = poisson_moment_matched_interval(mu, mu + 40.0, 0.025, 0.975).unwrap();
        assert!(lo1 <= lo0 && hi1 > hi0, "band did not widen: [{lo0},{hi0}] -> [{lo1},{hi1}]");
        // A negligible excess (θ_eff above the switch threshold) must coincide
        // with the exact conditional Poisson — no discontinuity at the boundary.
        let (lo2, hi2) =
            poisson_moment_matched_interval(mu, mu + mu * mu * 1.0e-12, 0.025, 0.975).unwrap();
        assert_eq!((lo2, hi2), (lo0, hi0));
    }

    #[test]
    fn poisson_moment_matched_interval_is_skewed_not_symmetric() {
        // The whole point of #1193/#817: on a low-rate count the equal-tailed
        // upper edge sits ABOVE the symmetric `μ + z·√μ` band that under-covers
        // the upper tail, and the band is asymmetric about μ.
        let mu = 2.0_f64;
        let z = standard_normal_quantile(0.975).unwrap();
        let (lo, hi) = poisson_moment_matched_interval(mu, mu, 0.025, 0.975).unwrap();
        let sym_hi = mu + z * mu.sqrt();
        assert!(
            hi > sym_hi,
            "equal-tailed upper {hi} should exceed symmetric upper {sym_hi}"
        );
        // Upper tail reaches further from μ than the lower tail (right skew).
        assert!((hi - mu) > (mu - lo), "band not right-skewed: lo={lo}, hi={hi}, μ={mu}");
    }

    #[test]
    fn poisson_moment_matched_interval_rejects_degenerate_inputs() {
        assert!(poisson_moment_matched_interval(0.0, 1.0, 0.025, 0.975).is_none());
        assert!(poisson_moment_matched_interval(-1.0, 1.0, 0.025, 0.975).is_none());
        assert!(poisson_moment_matched_interval(2.0, 0.0, 0.025, 0.975).is_none());
        assert!(poisson_moment_matched_interval(2.0, 1.0, 0.025, 0.975).is_none()); // total_var < μ
        assert!(poisson_moment_matched_interval(f64::NAN, 5.0, 0.025, 0.975).is_none());
        assert!(poisson_moment_matched_interval(2.0, 5.0, 0.025, 0.975).is_some());
    }

    #[test]
    fn tweedie_quantile_is_a_valid_cdf_inverse() {
        // For a probability strictly above the zero atom the quantile `y` must
        // satisfy `CDF(y) ≈ q`: the bisection inverts `tweedie_cdf_at` exactly.
        let mu = 3.0_f64;
        let phi = 1.2_f64;
        let power = 1.5_f64;
        let lambda = mu.powf(2.0 - power) / (phi * (2.0 - power));
        let zero_mass = (-lambda).exp();
        for &q in &[0.30_f64, 0.5, 0.75, 0.9, 0.975, 0.99] {
            assert!(q > zero_mass, "test q must exceed the zero atom {zero_mass}");
            let y = tweedie_quantile(q, mu, phi, power);
            assert!(y.is_finite() && y > 0.0, "quantile out of support: {y}");
            let cdf = tweedie_cdf_at(y, mu, phi, power);
            assert!((cdf - q).abs() < 1e-6, "CDF(Q(q)) != q: q={q}, cdf={cdf}");
        }
    }

    #[test]
    fn tweedie_quantile_returns_zero_atom_for_low_tail() {
        // When the requested lower-tail probability is at or below the point
        // mass at zero `e^{−λ}`, the quantile is exactly 0 (right-skewed low
        // means) — the zero-atom behaviour a continuous surrogate cannot mimic.
        let mu = 0.4_f64; // small mean ⇒ large zero atom
        let phi = 1.0_f64;
        let power = 1.5_f64;
        let lambda = mu.powf(2.0 - power) / (phi * (2.0 - power));
        let zero_mass = (-lambda).exp();
        assert!(zero_mass > 0.025, "fixture must have a fat zero atom: {zero_mass}");
        assert_eq!(tweedie_quantile(0.025, mu, phi, power), 0.0);
        assert_eq!(tweedie_quantile(0.5 * zero_mass, mu, phi, power), 0.0);
    }

    #[test]
    fn tweedie_quantile_boundaries_and_degeneracy() {
        let (mu, phi, power) = (2.0_f64, 1.0_f64, 1.6_f64);
        assert_eq!(tweedie_quantile(0.0, mu, phi, power), 0.0);
        assert_eq!(tweedie_quantile(-0.1, mu, phi, power), 0.0);
        assert_eq!(tweedie_quantile(1.0, mu, phi, power), f64::INFINITY);
        // Power outside (1, 2) or non-positive params are NaN.
        assert!(tweedie_quantile(0.5, mu, phi, 2.0).is_nan());
        assert!(tweedie_quantile(0.5, mu, phi, 1.0).is_nan());
        assert!(tweedie_quantile(0.5, 0.0, phi, power).is_nan());
        assert!(tweedie_quantile(0.5, mu, 0.0, power).is_nan());
    }

    #[test]
    fn tweedie_moment_matched_interval_is_exact_conditional_when_se_vanishes() {
        // total_var = φμ^p ⇒ φ_eff = φ, recovering the exact conditional Tweedie
        // quantiles.
        let mu = 3.0_f64;
        let phi = 1.2_f64;
        let power = 1.5_f64;
        let total_var = phi * mu.powf(power);
        let (lo, hi) =
            tweedie_moment_matched_interval(mu, phi, power, total_var, 0.025, 0.975).unwrap();
        assert_eq!(lo, tweedie_quantile(0.025, mu, phi, power));
        assert_eq!(hi, tweedie_quantile(0.975, mu, phi, power));
    }

    #[test]
    fn tweedie_moment_matched_interval_is_skewed_not_symmetric() {
        // A right-skewed Tweedie has the upper edge farther from the mean than
        // the lower edge — the symmetric `mu ± z·σ` band cannot reproduce this.
        let mu = 2.0_f64;
        let phi = 1.5_f64;
        let power = 1.5_f64;
        let total_var = phi * mu.powf(power);
        let (lo, hi) =
            tweedie_moment_matched_interval(mu, phi, power, total_var, 0.025, 0.975).unwrap();
        assert!(lo >= 0.0 && hi > mu && lo < mu);
        assert!(hi - mu > mu - lo, "interval is not right-skewed: lo={lo}, hi={hi}");
    }

    #[test]
    fn tweedie_moment_matched_interval_widens_with_estimation_uncertainty() {
        // Adding estimation variance raises φ_eff and must not shrink the band;
        // the upper edge grows.
        let mu = 4.0_f64;
        let phi = 1.0_f64;
        let power = 1.5_f64;
        let obs_var = phi * mu.powf(power);
        let (lo0, hi0) =
            tweedie_moment_matched_interval(mu, phi, power, obs_var, 0.025, 0.975).unwrap();
        let (lo1, hi1) =
            tweedie_moment_matched_interval(mu, phi, power, obs_var + 30.0, 0.025, 0.975).unwrap();
        assert!(lo1 <= lo0 && hi1 > hi0, "band did not widen: [{lo0},{hi0}] -> [{lo1},{hi1}]");
    }

    #[test]
    fn tweedie_moment_matched_interval_rejects_degenerate_inputs() {
        assert!(tweedie_moment_matched_interval(0.0, 1.0, 1.5, 1.0, 0.025, 0.975).is_none());
        assert!(tweedie_moment_matched_interval(-1.0, 1.0, 1.5, 1.0, 0.025, 0.975).is_none());
        assert!(tweedie_moment_matched_interval(2.0, 0.0, 1.5, 1.0, 0.025, 0.975).is_none());
        assert!(tweedie_moment_matched_interval(2.0, 1.0, 2.0, 1.0, 0.025, 0.975).is_none());
        assert!(tweedie_moment_matched_interval(2.0, 1.0, 1.5, 0.0, 0.025, 0.975).is_none());
        assert!(tweedie_moment_matched_interval(f64::NAN, 1.0, 1.5, 1.0, 0.025, 0.975).is_none());
        assert!(tweedie_moment_matched_interval(2.0, 1.0, 1.5, 6.0, 0.025, 0.975).is_some());
    }
}
