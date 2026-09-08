//! Stable scalar divided differences for the separation log-softplus barrier.
//!
//! Close eigenvalues use an analytic Taylor series, not subtraction of nearly
//! equal scalar values. The series is evaluated with symmetric polynomials in
//! the nodes, so coincident modes need no eigenvector-dependent convention.

const TAYLOR_TERMS: usize = 24;

/// log(softplus(x)) and its first three derivatives. Negative x uses a scaled
/// log1p expression, including the representable limit exp(x)=0. Positive x
/// uses exp(-x), so no intermediate overflows and no artificial tail joins.
fn log_softplus_derivatives(x: f64) -> [f64; 4] {
    if x >= 0.0 {
        let t = (-x).exp();
        let p = 1.0 / (1.0 + t);
        let one_minus_p = t / (1.0 + t);
        let l = x + t.ln_1p();
        let q = p / l;
        let first = q * (one_minus_p - q);
        let second = q * (one_minus_p * (1.0 - 2.0 * p) - 3.0 * q * one_minus_p + 2.0 * q * q);
        [l.ln(), q, first, second]
    } else {
        let t = x.exp();
        let scaled_log = if t == 0.0 { 1.0 } else { t.ln_1p() / t };
        // (log(1+t)-t)/t² has a regular limit. The omitted term is bounded
        // by t^6/8 here, below f64 roundoff of this order-one quantity.
        let scaled_difference = if t <= 1.0e-3 {
            -0.5 + t * (1.0 / 3.0 + t * (-0.25 + t * (0.2 + t * (-1.0 / 6.0 + t / 7.0))))
        } else {
            (t.ln_1p() - t) / (t * t)
        };
        let denominator = (1.0 + t) * scaled_log;
        let first = t * scaled_difference / (denominator * denominator);
        let second = t
            * (-1.0 - (1.0 + 2.0 * t) * scaled_difference
                + t * (1.0 - t) * scaled_difference * scaled_difference)
            / (denominator * denominator * denominator);
        [x + scaled_log.ln(), 1.0 / denominator, first, second]
    }
}

pub(super) fn log_derivatives(lambda: f64, eps: f64) -> (f64, f64, f64) {
    let f = log_softplus_derivatives((lambda + eps) / eps);
    (eps.ln() + f[0], f[1] / eps, f[2] / (eps * eps))
}

/// Coefficients of q(x+d)=sigmoid(x+d)/softplus(x+d), in powers of d.
/// The two recurrences are analytic power-series identities: p'=p-p²,
/// L'=p, and L*q=p. They are not an automatic-differentiation evaluation of
/// an arbitrary expression. The first two coefficients use cancellation-free
/// closed forms before they enter the higher coefficient recurrence.
fn h_taylor(x: f64) -> [f64; TAYLOR_TERMS] {
    if x < -2.0 {
        // q(x)=1/[(1+t) log(1+t)/t], t=exp(x). The denominator coefficients
        // are b_0=1, b_n=(-1)^(n+1)/(n(n+1)). Its reciprocal has coefficients
        // c_n=-sum_{j=1}^n b_j*c_(n-j). Expanding each t^n exp(n*d) avoids
        // subtracting order-one quantities to recover exponentially small
        // derivatives. With |d|<=1/3 and t<exp(-2), 48 terms put the omitted
        // scalar tail below 1e-34; the differentiated Taylor evaluation uses
        // the same convergent series.
        let t = x.exp();
        let mut coefficients = [0.0; 48];
        coefficients[0] = 1.0;
        let mut out = [0.0; TAYLOR_TERMS];
        out[0] = 1.0;
        let mut t_power = 1.0;
        for n in 1..coefficients.len() {
            coefficients[n] = -(1..=n)
                .map(|j| {
                    let sign = if j % 2 == 1 { 1.0 } else { -1.0 };
                    sign * coefficients[n - j] / (j * (j + 1)) as f64
                })
                .sum::<f64>();
            t_power *= t;
            let mut contribution = coefficients[n] * t_power;
            for (k, value) in out.iter_mut().enumerate() {
                if k > 0 {
                    contribution *= n as f64 / k as f64;
                }
                *value += contribution;
            }
        }
        return out;
    }
    let t = (-x).exp();
    let p0 = 1.0 / (1.0 + t);
    let l0 = if x >= 0.0 {
        x + t.ln_1p()
    } else {
        x.exp().ln_1p()
    };
    let mut p = [0.0; TAYLOR_TERMS];
    p[0] = p0;
    p[1] = p0 * t / (1.0 + t);
    for n in 2..TAYLOR_TERMS {
        p[n] = (p[n - 1] * (1.0 - 2.0 * p0) - (1..n - 1).map(|j| p[j] * p[n - 1 - j]).sum::<f64>())
            / n as f64;
    }
    let f = log_softplus_derivatives(x);
    let mut out = [0.0; TAYLOR_TERMS];
    out[0] = f[1];
    out[1] = f[2];
    out[2] = 0.5 * f[3];
    for n in 3..TAYLOR_TERMS {
        out[n] = (p[n]
            - (1..=n)
                .map(|j| p[j - 1] * out[n - j] / j as f64)
                .sum::<f64>())
            / l0;
    }
    out
}

pub(super) fn h_first_difference(a: f64, b: f64, eps: f64) -> f64 {
    let (a, b) = if a <= b { (a, b) } else { (b, a) };
    if a == b {
        log_derivatives(a, eps).2
    } else if (b - a) / eps <= 0.5 {
        // Hermite identity h[a,b] = h'(a) + (b-a) h[a,a,b]. Sorting the
        // endpoints preserves bitwise symmetry without a subtractive quotient.
        log_derivatives(a, eps).2 + (b - a) * h_second_difference(a, a, b, eps)
    } else {
        (log_derivatives(a, eps).1 - log_derivatives(b, eps).1) / (a - b)
    }
}

pub(super) fn h_second_difference(a: f64, b: f64, c: f64, eps: f64) -> f64 {
    let mut nodes = [a, b, c];
    nodes.sort_by(f64::total_cmp);
    let [a, b, c] = nodes;
    if (c - a) / eps <= 0.5 {
        // At these distances the nearest complex singularity is at least pi
        // away in x. The largest centered node is at most 1/3 away. Twenty-four
        // coefficients put the geometric tail below f64 rounding, including
        // after removing the first two Taylor powers in this divided difference.
        let center = a + (b - a) / 3.0 + (c - a) / 3.0;
        let coefficients = h_taylor((center + eps) / eps);
        let nodes = [(a - center) / eps, (b - center) / eps, (c - center) / eps];
        let elementary1 = nodes[0] + nodes[1] + nodes[2];
        let elementary2 = nodes[0] * nodes[1] + nodes[0] * nodes[2] + nodes[1] * nodes[2];
        let elementary3 = nodes[0] * nodes[1] * nodes[2];
        // The divided difference of x^(n+2) is the complete homogeneous
        // polynomial H_n in the three nodes. Its symmetric recurrence remains
        // valid for coincident nodes; no spectral direction or gap is omitted.
        let mut homogeneous = [0.0; TAYLOR_TERMS - 2];
        homogeneous[0] = 1.0;
        let mut out = coefficients[2];
        for n in 1..homogeneous.len() {
            homogeneous[n] = elementary1 * homogeneous[n - 1]
                - if n >= 2 {
                    elementary2 * homogeneous[n - 2]
                } else {
                    0.0
                }
                + if n >= 3 {
                    elementary3 * homogeneous[n - 3]
                } else {
                    0.0
                };
            out += coefficients[n + 2] * homogeneous[n];
        }
        out / (eps * eps * eps)
    } else {
        (h_first_difference(b, c, eps) - h_first_difference(a, b, eps)) / (c - a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spectral_first_difference_resolves_close_modes_and_both_tails_2820() {
        let eps = 0.17;
        for x in [-1000.0, -30.0, -5.0, -2.0, 0.0, 0.5, 2.0, 30.0, 1000.0] {
            let center = eps * (x - 1.0);
            let expected = log_derivatives(center, eps).2;
            for gap in [0.0, 1.0e-12, 1.0e-8] {
                let a = center - eps * gap;
                let b = center + eps * gap;
                let actual = h_first_difference(a, b, eps);
                assert!(
                    (actual - expected).abs() < 2.0e-13 * (1.0 + expected.abs()),
                    "x={x}, gap={gap}: divided={actual:e}, derivative={expected:e}"
                );
                assert_eq!(actual, h_first_difference(b, a, eps));
            }
            let step = eps * 2.0e-4;
            let fd = (log_derivatives(center + step, eps).1
                - log_derivatives(center - step, eps).1)
                / (2.0 * step);
            assert!((fd - expected).abs() < 2.0e-8 * (1.0 + expected.abs()));
        }
    }

    #[test]
    fn spectral_second_difference_preserves_repeated_modes_and_node_symmetry_2820() {
        let eps = 0.17;
        for x in [-1000.0, -30.0, -5.0, -2.0, 0.0, 0.5, 2.0, 30.0, 1000.0] {
            let center = eps * (x - 1.0);
            let step = eps * 2.0e-4;
            let fd = (log_derivatives(center + step, eps).2
                - log_derivatives(center - step, eps).2)
                / (2.0 * step);
            let actual = 2.0 * h_second_difference(center, center, center, eps);
            assert!(
                (actual - fd).abs() < 5.0e-8 * (1.0 + fd.abs()),
                "x={x}: second derivative={actual:e}, FD={fd:e}"
            );
            for span in [1.0e-12, 1.0e-6, 0.4, 2.0] {
                let a = center - eps * span;
                let b = center + eps * span / 3.0;
                let c = center + eps * span;
                let value = h_second_difference(a, b, c, eps);
                assert!(value.is_finite());
                assert_eq!(value, h_second_difference(c, a, b, eps));
                assert_eq!(value, h_second_difference(b, c, a, eps));
            }
        }
    }
}
