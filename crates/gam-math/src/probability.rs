use statrs::function::erf::erfc;

/// Standard normal PDF phi(x).
#[inline]
pub fn normal_pdf(x: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

/// Standard normal CDF Phi(x) evaluated via the exact special-function identity
///
///   Phi(x) = 0.5 * erfc(-x / sqrt(2)).
///
/// This is the exact Gaussian CDF semantics used throughout the codebase. The
/// numerical `erfc` implementation may use internal approximations, but the
/// returned function is the standard normal CDF itself rather than a separate
/// polynomial surrogate surface.
#[inline]
pub fn normal_cdf(x: f64) -> f64 {
    0.5 * statrs::function::erf::erfc(-x / std::f64::consts::SQRT_2)
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

    if log_pos == f64::NEG_INFINITY && log_neg == f64::NEG_INFINITY {
        // Both partial sums are empty: no terms at all, all signs zero, or every
        // magnitude `−∞` (each `exp(−∞) = 0`). The signed sum is exactly `0`, so
        // the contract requires `(−∞, 0.0)` — NOT the positive-sum convention,
        // which would mislabel a zero as `+1` and corrupt any downstream cascade
        // that reads back the sign.
        return (f64::NEG_INFINITY, 0.0);
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    fn rel_err(got: f64, expected: f64) -> f64 {
        (got - expected).abs() / expected.abs().max(1e-300)
    }

    // ── normal_pdf ────────────────────────────────────────────────────────────

    #[test]
    fn normal_pdf_at_zero() {
        let expected = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
        assert!((normal_pdf(0.0) - expected).abs() < TOL);
    }

    #[test]
    fn normal_pdf_symmetry() {
        for &x in &[0.5, 1.0, 2.0, 3.0, 5.0] {
            assert_eq!(normal_pdf(x), normal_pdf(-x), "symmetry failed at x={x}");
        }
    }

    #[test]
    fn normal_pdf_positive() {
        for &x in &[-5.0, -1.0, 0.0, 1.0, 5.0] {
            assert!(normal_pdf(x) > 0.0, "pdf should be positive at x={x}");
        }
    }

    // ── normal_cdf ────────────────────────────────────────────────────────────

    #[test]
    fn normal_cdf_at_zero_is_half() {
        assert!((normal_cdf(0.0) - 0.5).abs() < TOL);
    }

    #[test]
    fn normal_cdf_symmetry() {
        for &x in &[0.5, 1.0, 2.0, 3.0] {
            let sum = normal_cdf(x) + normal_cdf(-x);
            assert!(
                (sum - 1.0).abs() < TOL,
                "cdf symmetry failed at x={x}: sum={sum}"
            );
        }
    }

    #[test]
    fn normal_cdf_bounds() {
        assert!(normal_cdf(10.0) > 0.9999);
        assert!(normal_cdf(-10.0) < 1e-22);
        assert!(normal_cdf(0.0) > 0.0);
        assert!(normal_cdf(0.0) < 1.0);
    }

    #[test]
    fn normal_cdf_at_1_96_near_0975() {
        // Phi(1.96) ≈ 0.975 — canonical two-sided 5% critical value.
        let p = normal_cdf(1.959_963_985);
        assert!((p - 0.975).abs() < 1e-8, "p={p}");
    }

    // ── erfcx_nonnegative ─────────────────────────────────────────────────────

    #[test]
    fn erfcx_at_nonpositive_returns_one() {
        assert_eq!(erfcx_nonnegative(0.0), 1.0);
        assert_eq!(erfcx_nonnegative(-1.0), 1.0);
        assert_eq!(erfcx_nonnegative(-100.0), 1.0);
    }

    #[test]
    fn erfcx_positive_inf_returns_zero() {
        assert_eq!(erfcx_nonnegative(f64::INFINITY), 0.0);
    }

    #[test]
    fn erfcx_negative_inf_returns_inf() {
        assert_eq!(erfcx_nonnegative(f64::NEG_INFINITY), f64::INFINITY);
    }

    #[test]
    fn erfcx_small_positive_matches_direct() {
        use statrs::function::erf::erfc;
        for &x in &[0.1_f64, 0.5, 1.0, 5.0, 10.0, 25.0] {
            let got = erfcx_nonnegative(x);
            let expected = (x * x).exp() * erfc(x);
            let err = rel_err(got, expected);
            assert!(
                err < 1e-10,
                "x={x}: got={got} expected={expected} rel={err}"
            );
        }
    }

    #[test]
    fn erfcx_large_x_positive_and_finite() {
        // For x >= 26 the asymptotic branch must remain positive and finite.
        let got = erfcx_nonnegative(50.0);
        assert!(got.is_finite() && got > 0.0, "erfcx(50)={got}");
        // Leading asymptotic term: 1/(x*sqrt(pi)).
        let asymptotic = 1.0 / (50.0 * std::f64::consts::PI.sqrt());
        assert!(
            rel_err(got, asymptotic) < 1e-3,
            "got={got} asymptotic={asymptotic}"
        );
    }

    // ── log1mexp_positive ─────────────────────────────────────────────────────

    #[test]
    fn log1mexp_at_zero_is_neg_inf() {
        assert_eq!(log1mexp_positive(0.0), f64::NEG_INFINITY);
    }

    #[test]
    fn log1mexp_recovers_log_one_minus_exp() {
        // Verify exp(log1mexp(a)) + exp(-a) ≈ 1 for several a > 0. This
        // roundtrip avoids computing `(1 - exp(-a)).ln()` directly, which
        // suffers catastrophic cancellation for large a (e.g. a=20 where
        // `1.0 - exp(-20)` loses 9 decimal digits from the subtraction).
        for &a in &[0.001_f64, 0.5, std::f64::consts::LN_2, 1.0, 5.0, 20.0] {
            let lm = log1mexp_positive(a);
            let roundtrip = lm.exp() + (-a).exp();
            assert!(
                (roundtrip - 1.0).abs() < 1e-14,
                "a={a}: exp(log1mexp(a)) + exp(-a) = {roundtrip}, expected 1.0"
            );
        }
    }

    #[test]
    fn log1mexp_at_ln2_is_neg_ln2() {
        let ln2 = std::f64::consts::LN_2;
        let got = log1mexp_positive(ln2);
        assert!((got - (-ln2)).abs() < TOL, "got={got}");
    }

    // ── signed_log_sum_exp ────────────────────────────────────────────────────

    #[test]
    fn slse_all_positive_single() {
        let (lm, sg) = signed_log_sum_exp(&[2.0], &[1.0]);
        assert!((lm - 2.0).abs() < TOL);
        assert!((sg - 1.0).abs() < TOL);
    }

    #[test]
    fn slse_difference_recovers_log2() {
        // 3 - 1 = 2 → log|2| = ln(2), sign = +1.
        let log3 = 3.0_f64.ln();
        let log1 = 0.0_f64; // ln(1)
        let (lm, sg) = signed_log_sum_exp(&[log3, log1], &[1.0, -1.0]);
        assert!((lm - 2.0_f64.ln()).abs() < TOL, "lm={lm}");
        assert!((sg - 1.0).abs() < TOL, "sg={sg}");
    }

    #[test]
    fn slse_cancellation_gives_neg_inf() {
        // a - a = 0 → log|0| = -∞.
        let ln2 = 2.0_f64.ln();
        let (lm, sg) = signed_log_sum_exp(&[ln2, ln2], &[1.0, -1.0]);
        assert_eq!(lm, f64::NEG_INFINITY);
        assert_eq!(sg, 0.0);
    }

    #[test]
    fn slse_empty_returns_neg_inf_with_zero_sign() {
        // With no terms the sum is exactly 0, so the docstring contract is
        // `(−∞, 0.0)`. (This test previously encoded the buggy `+1.0` positive-sum
        // convention, which contradicted both the docstring and the cancellation
        // test below; rewritten to the correct zero sign.)
        let (lm, sg) = signed_log_sum_exp(&[], &[]);
        assert_eq!(lm, f64::NEG_INFINITY);
        assert_eq!(sg, 0.0);
    }

    #[test]
    fn slse_all_zero_signs_return_zero_sign() {
        // A single term whose sign is 0 contributes nothing; S = 0 ⇒ (−∞, 0.0).
        let (lm, sg) = signed_log_sum_exp(&[0.0], &[0.0]);
        assert_eq!(lm, f64::NEG_INFINITY);
        assert_eq!(sg, 0.0);
    }

    #[test]
    fn slse_all_neg_inf_magnitudes_return_zero_sign() {
        // Every magnitude is exp(−∞) = 0 regardless of sign, so the sum is 0 and
        // the reported sign must be 0.0, not +1.0.
        let (lm, sg) = signed_log_sum_exp(&[f64::NEG_INFINITY, f64::NEG_INFINITY], &[1.0, -1.0]);
        assert_eq!(lm, f64::NEG_INFINITY);
        assert_eq!(sg, 0.0);
    }

    #[test]
    fn slse_pos_inf_dominates() {
        let (lm, sg) = signed_log_sum_exp(&[f64::INFINITY, 1.0], &[1.0, -1.0]);
        assert_eq!(lm, f64::INFINITY);
        assert_eq!(sg, 1.0);
    }

    #[test]
    fn slse_neg_inf_dominates() {
        let (lm, sg) = signed_log_sum_exp(&[f64::INFINITY, 1.0], &[-1.0, 1.0]);
        assert_eq!(lm, f64::INFINITY);
        assert_eq!(sg, -1.0);
    }

    #[test]
    fn slse_both_inf_signs_gives_nan() {
        let (lm, sg) = signed_log_sum_exp(&[f64::INFINITY, f64::INFINITY], &[1.0, -1.0]);
        assert!(lm.is_nan());
        assert_eq!(sg, 0.0);
    }

    // ── normal_logcdf ─────────────────────────────────────────────────────────

    #[test]
    fn logcdf_at_zero_is_log_half() {
        let got = normal_logcdf(0.0);
        let expected = 0.5_f64.ln();
        assert!((got - expected).abs() < TOL, "got={got}");
    }

    #[test]
    fn logcdf_pos_inf_is_zero() {
        assert_eq!(normal_logcdf(f64::INFINITY), 0.0);
    }

    #[test]
    fn logcdf_neg_inf_is_neg_inf() {
        assert_eq!(normal_logcdf(f64::NEG_INFINITY), f64::NEG_INFINITY);
    }

    #[test]
    fn logcdf_nan_is_nan() {
        assert!(normal_logcdf(f64::NAN).is_nan());
    }

    #[test]
    fn logcdf_matches_log_cdf_for_moderate_x() {
        for &x in &[-2.0_f64, -1.0, 0.0, 1.0, 2.0, 3.0] {
            let got = normal_logcdf(x);
            let expected = normal_cdf(x).ln();
            assert!(
                (got - expected).abs() < 1e-10,
                "x={x}: got={got} expected={expected}"
            );
        }
    }

    #[test]
    fn logcdf_deep_left_tail_stays_finite() {
        // For very negative x, normal_cdf(x) underflows to 0, but logcdf should
        // remain finite and large-negative.
        let got = normal_logcdf(-20.0);
        assert!(got.is_finite() && got < -100.0, "logcdf(-20)={got}");
    }

    // ── normal_logsf ─────────────────────────────────────────────────────────

    #[test]
    fn logsf_at_zero_is_log_half() {
        let got = normal_logsf(0.0);
        let expected = 0.5_f64.ln();
        assert!((got - expected).abs() < TOL, "got={got}");
    }

    #[test]
    fn logsf_mirrors_logcdf() {
        // logsf(x) = logcdf(-x) by definition.
        for &x in &[-3.0_f64, -1.0, 0.0, 1.0, 3.0] {
            assert_eq!(normal_logsf(x), normal_logcdf(-x));
        }
    }

    // ── signed_probit_logcdf_and_mills_ratio ──────────────────────────────────

    #[test]
    fn probit_at_pos_inf() {
        let (lc, mr) = signed_probit_logcdf_and_mills_ratio(f64::INFINITY);
        assert_eq!(lc, 0.0);
        assert_eq!(mr, 0.0);
    }

    #[test]
    fn probit_at_neg_inf() {
        let (lc, mr) = signed_probit_logcdf_and_mills_ratio(f64::NEG_INFINITY);
        assert_eq!(lc, f64::NEG_INFINITY);
        assert_eq!(mr, f64::INFINITY);
    }

    #[test]
    fn probit_nan_propagates() {
        let (lc, mr) = signed_probit_logcdf_and_mills_ratio(f64::NAN);
        assert!(lc.is_nan() && mr.is_nan());
    }

    #[test]
    fn probit_at_zero_logcdf_and_mills() {
        let (lc, mr) = signed_probit_logcdf_and_mills_ratio(0.0);
        assert!((lc - 0.5_f64.ln()).abs() < TOL, "lc={lc}");
        // phi(0)/Phi(0) = 0.3989.../0.5 ≈ 0.7979.
        assert!((mr - 0.797_884_560_802_865).abs() < 1e-10, "mr={mr}");
    }

    #[test]
    fn probit_positive_branch_matches_logcdf() {
        for &x in &[0.5_f64, 1.0, 2.0, 3.0] {
            let (lc, mr) = signed_probit_logcdf_and_mills_ratio(x);
            let lc_ref = normal_logcdf(x);
            let mr_ref = normal_pdf(x) / normal_cdf(x);
            assert!(
                (lc - lc_ref).abs() < 1e-10,
                "x={x}: lc={lc} lc_ref={lc_ref}"
            );
            assert!(
                (mr - mr_ref).abs() < 1e-10,
                "x={x}: mr={mr} mr_ref={mr_ref}"
            );
        }
    }

    #[test]
    fn probit_negative_branch_matches_logcdf() {
        for &x in &[-0.5_f64, -1.0, -2.0, -5.0] {
            let (lc, mr) = signed_probit_logcdf_and_mills_ratio(x);
            let lc_ref = normal_logcdf(x);
            assert!(
                (lc - lc_ref).abs() < 1e-10,
                "x={x}: lc={lc} lc_ref={lc_ref}"
            );
            assert!(mr.is_finite() && mr > 0.0, "x={x}: mr={mr}");
        }
    }

    // ── standard_normal_quantile ──────────────────────────────────────────────

    #[test]
    fn quantile_rejects_out_of_range() {
        assert!(standard_normal_quantile(0.0).is_err());
        assert!(standard_normal_quantile(1.0).is_err());
        assert!(standard_normal_quantile(-0.1).is_err());
        assert!(standard_normal_quantile(1.1).is_err());
        assert!(standard_normal_quantile(f64::NAN).is_err());
    }

    #[test]
    fn quantile_at_half_is_near_zero() {
        let q = standard_normal_quantile(0.5).unwrap();
        assert!(q.abs() < 1e-10, "quantile(0.5)={q}");
    }

    #[test]
    fn quantile_at_0975_is_near_196() {
        let q = standard_normal_quantile(0.975).unwrap();
        assert!((q - 1.959_963_985).abs() < 1e-7, "q={q}");
    }

    #[test]
    fn quantile_antisymmetry() {
        let q_lo = standard_normal_quantile(0.1).unwrap();
        let q_hi = standard_normal_quantile(0.9).unwrap();
        assert!((q_lo + q_hi).abs() < 1e-10, "q_lo={q_lo} q_hi={q_hi}");
    }

    #[test]
    fn quantile_roundtrip_cdf() {
        for &p in &[
            0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999,
        ] {
            let q = standard_normal_quantile(p).unwrap();
            let p_back = normal_cdf(q);
            assert!(
                (p_back - p).abs() < 1e-10,
                "roundtrip failed at p={p}: q={q} p_back={p_back}"
            );
        }
    }
}
