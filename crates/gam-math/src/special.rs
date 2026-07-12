//! Scalar special-function primitives shared across the workspace.
//!
//! These are pure (`std`/`libm`-only) numeric kernels with no upward crate
//! dependencies, so they live in the lowest crate (`gam-math`) and can be
//! consumed by any term/basis/inference code without inducing an SCC edge.

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
    // Carry the recurrence in u128, not f64. At step `j` the running product
    // equals the integer `C(n, j)`, which is always divisible by the next
    // denominator `(j + 1)` (the partial product of `(j+1)` consecutive
    // integers `(n−j)…(n)` is divisible by `(j+1)!`), so each integer division
    // is exact and no rounding accumulates. The earlier all-`f64` recurrence
    // divided in floating point, where `(n−j)/(j+1)` is generally inexact, and
    // the drift pushed results off the true integer well below `2^53`
    // (e.g. `C(54,24)` came back one short). Converting the exact `u128` at the
    // end is bit-exact for every value at or below `2^53`.
    let mut num: u128 = 1;
    for j in 0..k_eff {
        match num.checked_mul((n - j) as u128) {
            Some(scaled) => num = scaled / (j as u128 + 1),
            None => {
                // The true coefficient overflows u128 — astronomically above
                // `2^53`, where the exactness contract no longer applies.
                // Finish the (now necessarily inexact) recurrence in f64.
                let mut out = num as f64;
                for jj in j..k_eff {
                    out = out * (n - jj) as f64 / (jj + 1) as f64;
                }
                return out;
            }
        }
    }
    num as f64
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

/// Large-argument (`|x| >= 3.75`) Abramowitz & Stegun 9.8.2 polynomial for the
/// exponentially scaled modified Bessel function `I0`:
/// `sqrt(x) exp(-x) I0(x)`. Factoring out the common `exp(x) / sqrt(x)`
/// envelope lets the log partition and `I1 / I0` ratio be evaluated without
/// overflow.
#[inline]
fn bessel_i0_scaled_polynomial_and_centered_log_derivative(ax: f64) -> (f64, f64) {
    let y = 3.75 / ax;
    const COEFFICIENTS: [f64; 9] = [
        0.39894228,
        0.01328592,
        0.00225319,
        -0.00157565,
        0.00916281,
        -0.02057706,
        0.02635537,
        -0.01647633,
        0.00392377,
    ];
    let mut polynomial = COEFFICIENTS[COEFFICIENTS.len() - 1];
    let mut derivative = 0.0_f64;
    for &coefficient in COEFFICIENTS[..COEFFICIENTS.len() - 1].iter().rev() {
        derivative = derivative * y + polynomial;
        polynomial = polynomial * y + coefficient;
    }
    // For L(x) = log I0(x) - x = -½ log x + log P0(3.75/x),
    // dL/d(log x) = -½ - y P0'(y)/P0(y). Differentiating the same
    // approximation used for the value keeps the ARD objective and gradient
    // consistent and tends to the exact -½ limit without subtracting two
    // rounded numbers near one.
    let scaled_centered_log_derivative = -0.5 - y * derivative / polynomial;
    (polynomial, scaled_centered_log_derivative)
}

/// Large-argument (`|x| >= 3.75`) Abramowitz & Stegun 9.8.4 polynomial for the
/// exponentially scaled modified Bessel function `I1`. Its envelope is the
/// same as [`bessel_i0_scaled_polynomial_and_centered_log_derivative`], so it
/// cancels exactly in `I1 / I0`.
#[inline]
fn bessel_i1_scaled_polynomial(ax: f64) -> f64 {
    let y = 3.75 / ax;
    0.39894228
        + y * (-0.03988024
            + y * (-0.00362018
                + y * (0.00163801
                    + y * (-0.01031555
                        + y * (0.02282967
                            + y * (-0.02895312 + y * (0.01787654 - y * 0.00420059)))))))
}

#[inline]
fn bessel_i0_small(ax: f64) -> f64 {
    let t = ax / 3.75;
    let t2 = t * t;
    1.0 + t2
        * (3.5156229
            + t2 * (3.0899424
                + t2 * (1.2067492 + t2 * (0.2659732 + t2 * (0.0360768 + t2 * 0.0045813)))))
}

#[inline]
fn bessel_i1_small(ax: f64) -> f64 {
    let t = ax / 3.75;
    let t2 = t * t;
    ax * (0.5
        + t2 * (0.87890594
            + t2 * (0.51498869
                + t2 * (0.15084934 + t2 * (0.02658733 + t2 * (0.00301532 + t2 * 0.00032411))))))
}

/// Overflow-free centered Bessel value, ratio, and log-scale derivative.
///
/// For `x = |eta|`, returns
/// `(log I0(x) - x, I1(x) / I0(x), x d/dx[log I0(x) - x])`. The third term is
/// the stable form of `x·(I1/I0 - 1)`: it approaches `-½` instead of becoming
/// `x·0` after the ordinary ratio rounds to one. Centering the logarithm by its
/// leading `x` term likewise prevents catastrophic cancellation.
pub fn bessel_i0_centered_terms(eta: f64) -> (f64, f64, f64) {
    let ax = eta.abs();
    if ax < 3.75 {
        let i0 = bessel_i0_small(ax);
        let i1 = bessel_i1_small(ax);
        let ratio = i1 / i0;
        (i0.ln() - ax, ratio, ax * (ratio - 1.0))
    } else {
        let (polynomial_0, scaled_centered_log_derivative) =
            bessel_i0_scaled_polynomial_and_centered_log_derivative(ax);
        let polynomial_1 = bessel_i1_scaled_polynomial(ax);
        (
            -0.5 * ax.ln() + polynomial_0.ln(),
            polynomial_1 / polynomial_0,
            scaled_centered_log_derivative,
        )
    }
}

/// Stable centered Bessel terms when only `log(|eta|)` is representable.
///
/// For a finite representable `|eta|`, this is exactly
/// [`bessel_i0_centered_terms`]. Beyond the float range, inverse-`eta`
/// corrections are themselves below float resolution, so the limiting terms
/// `log I0(eta)-eta = -½ log(2 pi eta)` and
/// `eta d/deta[log I0(eta)-eta] = -½` are the correctly rounded result.
pub fn bessel_i0_centered_terms_from_log_abs(log_abs_eta: f64) -> (f64, f64, f64) {
    if log_abs_eta.is_nan() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    if log_abs_eta == f64::NEG_INFINITY {
        return (0.0, 0.0, 0.0);
    }
    if log_abs_eta <= f64::MAX.ln() {
        return bessel_i0_centered_terms(log_abs_eta.exp());
    }
    (
        -0.5 * (std::f64::consts::TAU.ln() + log_abs_eta),
        1.0,
        -0.5,
    )
}

/// Overflow-free `(log I0(eta) - |eta|, I1(|eta|) / I0(|eta|))`.
///
/// Centering the logarithm by its leading `|eta|` term is essential whenever a
/// likelihood cancels the Bessel growth against an equally large quadratic,
/// as in a Gaussian-blurred circle. The large-argument branch never forms
/// `exp(|eta|)`, and therefore remains finite beyond the ordinary exponential
/// overflow threshold and up to the largest finite `f64`.
pub fn bessel_i0_log_minus_abs_and_ratio(eta: f64) -> (f64, f64) {
    let (centered_log_i0, ratio, _) = bessel_i0_centered_terms(eta);
    (centered_log_i0, ratio)
}

/// Overflow-free `(log I0(eta), I1(|eta|) / I0(|eta|))`.
///
/// Consumers whose formulas cancel the leading `|eta|` term should use
/// [`bessel_i0_log_minus_abs_and_ratio`] directly, rather than forming that
/// cancellation after this function returns.
pub fn bessel_i0_log_and_ratio(eta: f64) -> (f64, f64) {
    let (centered_log_i0, ratio) = bessel_i0_log_minus_abs_and_ratio(eta);
    (eta.abs() + centered_log_i0, ratio)
}

/// Gauss-Legendre nodes and weights on `[-1, 1]` for `n` points, computed via
/// Newton iteration on the Legendre-polynomial roots (Bonnet's three-term
/// recurrence, cosine initial guess). Returns `(nodes, weights)` with nodes
/// ascending; for odd `n` the central node is exactly `0.0`.
///
/// Canonical home for the routine previously triplicated in
/// `gam-terms/basis/closed_form_penalty.rs`, `gam-model-kernels/
/// cubic_cell_kernel.rs`, and `gam-models/survival/base.rs`; this copy keeps
/// the tightest of their Newton settings (200-iteration cap, `1e-15`
/// convergence).
pub fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut tmp: Vec<(f64, f64)> = Vec::with_capacity(n);
    let half = n.div_ceil(2);
    for i in 0..half {
        let mut z = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut pp = 0.0_f64;
        for _ in 0..200 {
            let mut p1 = 1.0_f64;
            let mut p2 = 0.0_f64;
            for j in 0..n {
                let p3 = p2;
                p2 = p1;
                p1 = ((2.0 * j as f64 + 1.0) * z * p2 - j as f64 * p3) / (j as f64 + 1.0);
            }
            pp = n as f64 * (z * p1 - p2) / (z * z - 1.0);
            let z_prev = z;
            z = z_prev - p1 / pp;
            if (z - z_prev).abs() < 1e-15 {
                break;
            }
        }
        let w = 2.0 / ((1.0 - z * z) * pp * pp);
        // For odd n the central node is at z = 0; record once.
        if !n.is_multiple_of(2) && i == half - 1 {
            tmp.push((0.0, w));
        } else {
            tmp.push((-z.abs(), w));
            tmp.push((z.abs(), w));
        }
    }
    tmp.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut nodes = Vec::with_capacity(n);
    let mut weights = Vec::with_capacity(n);
    for (z, w) in tmp.into_iter().take(n) {
        nodes.push(z);
        weights.push(w);
    }
    (nodes, weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centered_bessel_log_is_finite_and_derivative_consistent() {
        for eta in [0.25_f64, 1.0, 3.74, 3.76, 12.0, 900.0] {
            let (centered, ratio, scaled_derivative) = bessel_i0_centered_terms(eta);
            assert!(centered.is_finite());
            assert!((0.0..=1.0).contains(&ratio));

            let h = 1.0e-4 * eta.max(1.0);
            let (plus, _) = bessel_i0_log_and_ratio(eta + h);
            let (minus, _) = bessel_i0_log_and_ratio(eta - h);
            let derivative = (plus - minus) / (2.0 * h);
            assert!((derivative - ratio).abs() <= 1.0e-6 + 1.0e-5 * ratio.abs());

            let log_step = 1.0e-5;
            let (centered_plus, _, _) =
                bessel_i0_centered_terms(eta * log_step.exp());
            let (centered_minus, _, _) =
                bessel_i0_centered_terms(eta * (-log_step).exp());
            let finite_difference = (centered_plus - centered_minus) / (2.0 * log_step);
            assert!(
                (finite_difference - scaled_derivative).abs() < 2.0e-5,
                "centered Bessel value/gradient mismatch at eta={eta}: analytic={scaled_derivative}, finite_difference={finite_difference}"
            );
        }
        for eta in [1.0e20_f64, 1.0e100, 1.0e300] {
            let (centered, ratio, scaled_derivative) = bessel_i0_centered_terms(eta);
            let asymptotic = -0.5 * (std::f64::consts::TAU * eta).ln();
            assert!(centered.is_finite() && ratio.is_finite());
            assert!((centered - asymptotic).abs() < 2.0e-8);
            assert!(
                (scaled_derivative + 0.5).abs() < 1.0e-12,
                "large-eta centered derivative must retain its -1/2 limit; eta={eta:e}, derivative={scaled_derivative}"
            );
        }

        assert_eq!(bessel_i0_centered_terms(0.0), (0.0, 0.0, 0.0));

        let log_eta = 1_200.0;
        let (centered, ratio, scaled_derivative) =
            bessel_i0_centered_terms_from_log_abs(log_eta);
        assert!(centered.is_finite());
        assert_eq!(ratio, 1.0);
        assert_eq!(scaled_derivative, -0.5);
        assert_eq!(
            centered,
            -0.5 * (std::f64::consts::TAU.ln() + log_eta)
        );
    }

    #[test]
    fn gauss_legendre_integrates_polynomials_exactly() {
        // An n-point rule is exact for polynomials of degree ≤ 2n−1.
        for n in [1usize, 2, 3, 5, 8, 40, 64] {
            let (nodes, weights) = gauss_legendre(n);
            assert_eq!(nodes.len(), n);
            assert_eq!(weights.len(), n);
            assert!(nodes.windows(2).all(|w| w[0] < w[1]), "nodes ascending");
            if !n.is_multiple_of(2) {
                assert_eq!(nodes[n / 2], 0.0, "odd-n central node is exact zero");
            }
            let total: f64 = weights.iter().sum();
            assert!((total - 2.0).abs() < 1e-13, "∫1 dx = 2, got {total}");
            if n >= 2 {
                let x2: f64 = nodes.iter().zip(&weights).map(|(x, w)| w * x * x).sum();
                assert!((x2 - 2.0 / 3.0).abs() < 1e-13, "∫x² dx = 2/3, got {x2}");
            }
        }
    }

    #[test]
    fn binom_k_exceeds_n_returns_zero() {
        assert_eq!(binomial_coefficient_f64(3, 5), 0.0);
        assert_eq!(binomial_coefficient_f64(0, 1), 0.0);
        assert_eq!(binomial_coefficient_f64(10, 11), 0.0);
    }

    #[test]
    fn binom_k_zero_returns_one() {
        assert_eq!(binomial_coefficient_f64(0, 0), 1.0);
        assert_eq!(binomial_coefficient_f64(5, 0), 1.0);
        assert_eq!(binomial_coefficient_f64(100, 0), 1.0);
    }

    #[test]
    fn binom_k_equals_n_returns_one() {
        assert_eq!(binomial_coefficient_f64(1, 1), 1.0);
        assert_eq!(binomial_coefficient_f64(5, 5), 1.0);
        assert_eq!(binomial_coefficient_f64(20, 20), 1.0);
    }

    #[test]
    fn binom_small_exact_values() {
        assert_eq!(binomial_coefficient_f64(5, 2), 10.0);
        assert_eq!(binomial_coefficient_f64(10, 3), 120.0);
        assert_eq!(binomial_coefficient_f64(20, 10), 184_756.0);
        assert_eq!(binomial_coefficient_f64(6, 3), 20.0);
    }

    #[test]
    fn binom_symmetry() {
        assert_eq!(
            binomial_coefficient_f64(10, 3),
            binomial_coefficient_f64(10, 7)
        );
        assert_eq!(
            binomial_coefficient_f64(20, 5),
            binomial_coefficient_f64(20, 15)
        );
        assert_eq!(
            binomial_coefficient_f64(54, 24),
            binomial_coefficient_f64(54, 30)
        );
    }

    #[test]
    fn binom_c54_24_is_exact() {
        // The u128-recurrence fix restored this value (old f64 recurrence
        // returned 1_402_659_561_581_459, one short of the true integer).
        assert_eq!(binomial_coefficient_f64(54, 24), 1_402_659_561_581_460.0);
    }

    #[test]
    fn poly_exp_empty_coeffs_returns_zero() {
        assert_eq!(stable_polynomial_times_exp_neg(1.0, &[]), 0.0);
        assert_eq!(stable_polynomial_times_exp_neg(0.0, &[]), 0.0);
        assert_eq!(stable_polynomial_times_exp_neg(700.0, &[]), 0.0);
    }

    #[test]
    fn poly_exp_nonfinite_x_returns_zero() {
        assert_eq!(
            stable_polynomial_times_exp_neg(f64::INFINITY, &[1.0, 2.0]),
            0.0
        );
        assert_eq!(
            stable_polynomial_times_exp_neg(f64::NEG_INFINITY, &[1.0, 2.0]),
            0.0
        );
        assert_eq!(stable_polynomial_times_exp_neg(f64::NAN, &[1.0]), 0.0);
    }

    #[test]
    fn poly_exp_constant_at_zero() {
        // At x=0: poly(0) = coeffs[0], exp(0)=1 → result = coeffs[0].
        assert_eq!(stable_polynomial_times_exp_neg(0.0, &[5.0]), 5.0);
        assert_eq!(stable_polynomial_times_exp_neg(0.0, &[3.0, 1.0, 2.0]), 3.0);
    }

    #[test]
    fn poly_exp_constant_poly_direct_path() {
        // x=2.0 < 600: direct Horner * exp(-x).
        let x = 2.0;
        let got = stable_polynomial_times_exp_neg(x, &[3.0]);
        let expected = 3.0 * (-x).exp();
        assert!(
            (got - expected).abs() < 1e-14,
            "got={got} expected={expected}"
        );
    }

    #[test]
    fn poly_exp_linear_poly_direct_path() {
        // coeffs = [a, b] → poly = a + b*x.
        let x = 1.5;
        let (a, b) = (2.0, 3.0);
        let got = stable_polynomial_times_exp_neg(x, &[a, b]);
        let expected = (a + b * x) * (-x).exp();
        assert!(
            (got - expected).abs() < 1e-14,
            "got={got} expected={expected}"
        );
    }

    #[test]
    fn poly_exp_constant_poly_asymptotic_path() {
        // x=700 > 600: asymptotic path. For poly = [1.0], result = exp(-700).
        let x = 700.0_f64;
        let got = stable_polynomial_times_exp_neg(x, &[1.0]);
        let expected = (-x).exp();
        let rel = (got - expected).abs() / expected;
        assert!(rel < 1e-12, "got={got} expected={expected} rel={rel}");
    }

    #[test]
    fn poly_exp_quadratic_asymptotic_path() {
        // x=620 > 600: poly = x^2 (coeffs=[0,0,1]). Result = x^2 * exp(-x).
        // x=800 would underflow to 0.0 in both the asymptotic path and the
        // reference, making the relative-error check degenerate; x=620 keeps
        // the result in the normal f64 range (~10^-264) while still exercising
        // the asymptotic branch (threshold is x=600).
        let x = 620.0_f64;
        let got = stable_polynomial_times_exp_neg(x, &[0.0, 0.0, 1.0]);
        let expected = (2.0 * x.ln() - x).exp();
        let rel = (got - expected).abs() / expected.abs();
        assert!(rel < 1e-12, "got={got} expected={expected} rel={rel}");
    }
}
