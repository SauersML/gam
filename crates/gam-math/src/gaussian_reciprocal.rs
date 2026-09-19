//! Gaussian expectations of the reciprocal-power inverse links.
//!
//! For `η ~ N(m, s²)` neither `1/η` nor `η^{-1/2}` has a Lebesgue expectation
//! over the whole line: `1/η` is not integrable at `0`, and `η^{-1/2}` is not
//! real on `η < 0`. Both are defined here as the real part of the boundary
//! value `E[g⁻¹(η + i0)]`:
//!
//! - `1/(η + i0) = PV(1/η) − iπδ(η)`, so the mean of `1/η` is the Cauchy
//!   principal value `PV E[1/η] = (√2/s)·F(m/(√2 s))`, with `F` Dawson's
//!   integral;
//! - `(η + i0)^{-1/2}` is purely imaginary on `η < 0`, so the mean of
//!   `η^{-1/2}` is `∫₀^∞ x^{-1/2} φ_s(x − m) dx`.
//!
//! Each reduces to the plug-in `g⁻¹(m)` at `s = 0` and agrees, to every order
//! in `s/m`, with the moment expansion of `g⁻¹` about `m`
//! (`1/m + s²/m³ + 3s⁴/m⁵ + …` for the inverse link). Every routine returns
//! the value and its first three derivatives in `m`.

use std::f64::consts::{FRAC_2_SQRT_PI, FRAC_PI_2, SQRT_2};
use std::sync::OnceLock;

use crate::special::gauss_legendre;

/// `1/√π`.
const FRAC_1_SQRT_PI: f64 = 0.5 * FRAC_2_SQRT_PI;
/// `1/√(2π)`.
const FRAC_1_SQRT_2PI: f64 = 0.5 * FRAC_2_SQRT_PI / SQRT_2;

/// Below this `|x|` Dawson's Maclaurin series converges in a handful of terms
/// and avoids the cancellation Rybicki's symmetric sum suffers near `x = 0`.
const DAWSON_SERIES_MAX: f64 = 0.2;

/// From this `|x|` the asymptotic series `F(x) ~ Σ (2k−1)!!/(2^{k+1} x^{2k+1})`
/// is used. Its optimally truncated error is of order `e^{-x²}`, which at
/// `x = 7` is `5·10⁻²²` relative to `F`, below double precision even after
/// three derivatives; the Rybicki branch below it would lose digits to the
/// cancellation in `F' = 1 − 2xF` as `x` grows.
const DAWSON_ASYMPTOTIC_MIN: f64 = 7.0;

/// Dawson's integral `F(x) = e^{-x²} ∫₀ˣ e^{t²} dt` and its derivatives
/// `[F, F', F'', F''']`.
pub fn dawson_jet(x: f64) -> [f64; 4] {
    if x.is_nan() {
        return [f64::NAN; 4];
    }
    let a = x.abs();
    let mut jet = if a.is_infinite() {
        [0.0; 4]
    } else if a >= DAWSON_ASYMPTOTIC_MIN {
        // `F(x) = ½·M(x)` for the principal-value mean `M` at `s² = ½`.
        let series = reciprocal_moment_series(a, 0.5);
        [0.5 * series[0], 0.5 * series[1], 0.5 * series[2], 0.5 * series[3]]
    } else {
        let value = if a < DAWSON_SERIES_MAX {
            dawson_maclaurin(a)
        } else {
            dawson_rybicki(a)
        };
        let d1 = 1.0 - 2.0 * a * value;
        let d2 = -2.0 * value - 2.0 * a * d1;
        let d3 = -4.0 * d1 - 2.0 * a * d2;
        [value, d1, d2, d3]
    };
    if x < 0.0 {
        // `F` is odd, so its even derivatives are odd too.
        jet[0] = -jet[0];
        jet[2] = -jet[2];
    }
    jet
}

/// `F(x) = Σ_k (−2x²)^k x/(2k+1)!!`, summed until a term no longer changes
/// the sum.
fn dawson_maclaurin(x: f64) -> f64 {
    let ratio = -2.0 * x * x;
    let mut term = x;
    let mut sum = x;
    let mut k = 0.0_f64;
    while term.abs() > f64::EPSILON * sum.abs() {
        term *= ratio / (2.0 * k + 3.0);
        sum += term;
        k += 1.0;
    }
    sum
}

/// Rybicki's sampling sum `F(x) ≈ π^{-1/2} Σ_{n odd} e^{-(x − nh)²}/n`,
/// recentred on the even multiple `n₀h` of `h` nearest `x`.
///
/// The discretisation error is of order `e^{-(π/(2h))²}`; `h` is chosen so
/// that exponent is `ln ε²`, far below double precision. The sum runs outward
/// from the peak until a new pair of terms no longer changes it.
fn dawson_rybicki(x: f64) -> f64 {
    let h = FRAC_PI_2 / (-2.0 * f64::EPSILON.ln()).sqrt();
    let n0 = 2.0 * (x / (2.0 * h)).round();
    let xp = x - n0 * h;
    let mut sum = 0.0_f64;
    let mut n = 1.0_f64;
    loop {
        let above = (-(xp - n * h).powi(2)).exp() / (n0 + n);
        let below = (-(xp + n * h).powi(2)).exp() / (n0 - n);
        let added = above + below;
        sum += added;
        if n * h > xp.abs() && added.abs() <= f64::EPSILON * sum.abs() {
            break;
        }
        n += 2.0;
    }
    FRAC_1_SQRT_PI * sum
}

/// The moment expansion `M(m) = Σ_k (2k−1)!! s^{2k} m^{-(2k+1)}` of
/// `E[1/η]`, `η ~ N(m, s²)`, with its first three `m`-derivatives, for
/// `s²/m² ≤ 1/(2·DAWSON_ASYMPTOTIC_MIN²)`.
///
/// The series is asymptotic: its terms fall until `k ≈ m²/(2s²)` and then
/// grow. It is summed until the terms stop changing the sums, or truncated at
/// the smallest term, whose size is of order `e^{-m²/(2s²)}`.
fn reciprocal_moment_series(m: f64, s2: f64) -> [f64; 4] {
    let r = s2 / (m * m);
    let mut sums = [0.0_f64; 4];
    // `t_k = (2k−1)!!·r^k`; the `j`-th derivative carries the rising
    // factorial `(2k+1)(2k+2)…(2k+j)`.
    let mut t = 1.0_f64;
    let mut k = 0.0_f64;
    let mut previous_top = f64::INFINITY;
    loop {
        let p = 2.0 * k + 1.0;
        let terms = [t, t * p, t * p * (p + 1.0), t * p * (p + 1.0) * (p + 2.0)];
        if terms[3] > previous_top {
            break;
        }
        for (sum, term) in sums.iter_mut().zip(terms) {
            *sum += term;
        }
        let converged = sums
            .iter()
            .zip(terms)
            .all(|(sum, term)| term <= f64::EPSILON * sum.abs());
        if converged {
            break;
        }
        previous_top = terms[3];
        t *= p * r;
        k += 1.0;
    }
    let inv = 1.0 / m;
    let inv2 = inv * inv;
    [
        sums[0] * inv,
        -sums[1] * inv2,
        sums[2] * inv2 * inv,
        -sums[3] * inv2 * inv2,
    ]
}

/// The principal-value mean `PV E[1/η]`, `η ~ N(m, s²)`, and its first three
/// `m`-derivatives. At `s = 0` this is the plug-in `1/m` and its derivatives.
pub fn principal_value_inverse_normal_jet(m: f64, s: f64) -> [f64; 4] {
    if s == 0.0 {
        let inv = 1.0 / m;
        return [inv, -inv * inv, 2.0 * inv * inv * inv, -6.0 * inv * inv * inv * inv];
    }
    if m.abs() >= DAWSON_ASYMPTOTIC_MIN * SQRT_2 * s {
        return reciprocal_moment_series(m, s * s);
    }
    // `M(m) = 2c·F(cm)` with `c = 1/(√2 s)`, so `M^{(j)} = 2c^{j+1} F^{(j)}(cm)`.
    let c = 1.0 / (SQRT_2 * s);
    let f = dawson_jet(c * m);
    let scale = 2.0 * c;
    [
        scale * f[0],
        scale * c * f[1],
        scale * c * c * f[2],
        scale * c * c * c * f[3],
    ]
}

/// Half-width, in standard deviations, of the window outside which the
/// Gaussian kernel `e^{-v²/2}` is dropped: `e^{-L²/2} = ε²`, so even after the
/// cubic Hermite weight the discarded tail is far below double precision.
fn gaussian_window_half_width() -> f64 {
    (-4.0 * f64::EPSILON.ln()).sqrt()
}

/// Gauss–Legendre rule for the window `v ∈ [−L, L]` when the singularity of
/// `(m + s v)^{-1/2}` lies at least `L` beyond it.
fn window_rule() -> &'static (Vec<f64>, Vec<f64>) {
    static RULE: OnceLock<(Vec<f64>, Vec<f64>)> = OnceLock::new();
    RULE.get_or_init(|| gauss_legendre(128))
}

/// Gauss–Legendre rule for each of the two `t = √x` panels.
fn panel_rule() -> &'static (Vec<f64>, Vec<f64>) {
    static RULE: OnceLock<(Vec<f64>, Vec<f64>)> = OnceLock::new();
    RULE.get_or_init(|| gauss_legendre(64))
}

/// `J(m) = ∫₀^∞ x^{-1/2} φ_s(x − m) dx`, the mean of `η^{-1/2}` over
/// `η ~ N(m, s²)` restricted to `η > 0`, and its first three `m`-derivatives.
/// At `s = 0` this is the plug-in `m^{-1/2}` (zero for `m < 0`).
///
/// With `u = m/s`:
///
/// - for `u ≥ 2L` the positive half-line holds the whole window
///   `η = m + s v`, `|v| ≤ L`, and `(m + s v)^{-1/2-j}` is analytic there, so
///   the derivatives are taken under the integral and the window is
///   integrated in `v`;
/// - otherwise the window reaches the singularity at `η = 0`. Substituting
///   `x = s t²` gives `J = s^{-1/2} G(u)`, `G(u) = √(2/π) ∫₀^∞ e^{-(t²−u)²/2} dt`,
///   with a smooth integrand, and `G^{(j)}(u) = √(2/π) ∫₀^∞ He_j(v) e^{-v²/2} dt`
///   for `v = t² − u`. The `t` range is split at the peak `t = √u`.
pub fn positive_part_inverse_sqrt_normal_jet(m: f64, s: f64) -> [f64; 4] {
    if s == 0.0 {
        if m < 0.0 {
            return [0.0; 4];
        }
        let root = m.sqrt();
        let base = 1.0 / root;
        let inv = 1.0 / m;
        return [
            base,
            -0.5 * base * inv,
            0.75 * base * inv * inv,
            -1.875 * base * inv * inv * inv,
        ];
    }
    let l = gaussian_window_half_width();
    let u = m / s;
    if u >= 2.0 * l {
        let (nodes, weights) = window_rule();
        let mut sums = [0.0_f64; 4];
        for (&node, &weight) in nodes.iter().zip(weights) {
            let v = l * node;
            let eta = m + s * v;
            let inv = 1.0 / eta;
            let base = weight * (-0.5 * v * v).exp() / eta.sqrt();
            sums[0] += base;
            sums[1] += base * inv;
            sums[2] += base * inv * inv;
            sums[3] += base * inv * inv * inv;
        }
        let scale = l * FRAC_1_SQRT_2PI;
        return [
            scale * sums[0],
            -0.5 * scale * sums[1],
            0.75 * scale * sums[2],
            -1.875 * scale * sums[3],
        ];
    }
    // `t` bounds: the kernel is negligible once `v² − v_min² ≥ L²`, where
    // `v_min = max(−u, 0)` is the smallest `|v|` on `t ≥ 0`.
    let peak_sq = u.max(0.0);
    let low_sq = (u - l).max(0.0);
    let high_sq = if u >= 0.0 {
        u + l
    } else {
        // `u + √(u² + L²)`, rationalised to avoid cancellation for `u < 0`.
        l * l / ((u * u + l * l).sqrt() - u)
    };
    let (nodes, weights) = panel_rule();
    let mut sums = [0.0_f64; 4];
    for (lo, hi) in [(low_sq.sqrt(), peak_sq.sqrt()), (peak_sq.sqrt(), high_sq.sqrt())] {
        let half = 0.5 * (hi - lo);
        if half <= 0.0 {
            continue;
        }
        let mid = 0.5 * (hi + lo);
        for (&node, &weight) in nodes.iter().zip(weights) {
            let t = mid + half * node;
            let v = t * t - u;
            let kernel = half * weight * (-0.5 * v * v).exp();
            let v2 = v * v;
            sums[0] += kernel;
            sums[1] += kernel * v;
            sums[2] += kernel * (v2 - 1.0);
            sums[3] += kernel * v * (v2 - 3.0);
        }
    }
    let inv_s = 1.0 / s;
    let scale = 2.0 * FRAC_1_SQRT_2PI / s.sqrt();
    [
        scale * sums[0],
        scale * inv_s * sums[1],
        scale * inv_s * inv_s * sums[2],
        scale * inv_s * inv_s * inv_s * sums[3],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(label: &str, got: f64, want: f64, rel: f64) {
        let err = (got - want).abs();
        assert!(
            err <= rel * want.abs().max(f64::MIN_POSITIVE),
            "{label}: got {got:e}, want {want:e}, rel err {:e}",
            err / want.abs()
        );
    }

    /// `scipy.special.dawsn` reference values.
    #[test]
    fn dawson_matches_scipy_reference() {
        let reference = [
            (1e-3, 0.0009999993333336),
            (0.1, 0.0993359923978529),
            (0.5, 0.4244363835020223),
            (1.0, 0.5380795069127684),
            (2.0, 0.301340388923792),
            (5.0, 0.10213407442427686),
            (7.0, 0.0721809746582363),
            (10.0, 0.05025384718759854),
            (50.0, 0.010002001201201684),
        ];
        for (x, want) in reference {
            assert_close(&format!("F({x})"), dawson_jet(x)[0], want, 4e-15);
            assert_close(&format!("F(-{x})"), -dawson_jet(-x)[0], want, 4e-15);
        }
        assert_eq!(dawson_jet(0.0)[0], 0.0);
        assert_eq!(dawson_jet(0.0)[1], 1.0);
    }

    /// The branches meet continuously, derivatives included.
    #[test]
    fn dawson_branches_agree_at_their_boundaries() {
        for boundary in [DAWSON_SERIES_MAX, DAWSON_ASYMPTOTIC_MIN] {
            let below = dawson_jet(boundary * (1.0 - 1e-15));
            let above = dawson_jet(boundary);
            for j in 0..4 {
                assert_close(&format!("F^({j}) at {boundary}"), below[j], above[j], 1e-10);
            }
        }
    }

    /// Principal-value means `2c^{j+1} F^{(j)}(cm)` evaluated in 40-digit
    /// arithmetic (mpmath `erfi` and `diff`); the `(0.5, 2)` mean also matches
    /// `scipy.integrate.quad(..., weight='cauchy')` to `1e-15`.
    #[test]
    fn principal_value_inverse_matches_high_precision_reference() {
        let reference = [
            (1.0, 0.3, [1.136183556761769872, -1.5131506306863320228, 4.1885230436062464308, -12.913575358150916346]),
            (2.0, 1.0, [0.63998807456540892568, -0.27997614913081785136, -0.080035776303773222954, 0.72002385086918214864]),
            (0.5, 2.0, [0.12242809678001077814, 0.23469648790249865273, -0.059944085182815026125, -0.1098552333033974481]),
            (1.0, 0.1, [1.0103161564918598884, -1.0316156491859887243, 2.1299492694128833509, -6.6717971040905894996]),
            (1.0, 0.01, [1.0001000300150105095, -1.0003001501050946041, 2.0012009008409462493, -6.0060063075704112449]),
        ];
        for (m, s, want) in reference {
            let got = principal_value_inverse_normal_jet(m, s);
            for j in 0..4 {
                assert_close(&format!("PV^({j})(m={m}, s={s})"), got[j], want[j], 1e-11);
            }
        }
    }

    /// `∫₀^∞ x^{-1/2} φ_s(x − m) dx` and its Hermite-weighted `m`-derivatives,
    /// evaluated by mpmath `quad` at 40 digits.
    #[test]
    fn positive_part_inverse_sqrt_matches_high_precision_reference() {
        let reference = [
            (1.0, 0.3, [1.0453698928919120859, -0.64419572261692659933, 1.350119735233006282, -4.264735014529071238]),
            (2.0, 1.0, [0.79602337881767523887, -0.2354061594959698236, 0.072800629583102027772, 0.20750798007775067986]),
            (0.5, 2.0, [0.67030836581842647808, 0.1016192208441600066, -0.096490948332823310585, -0.026045839274957088651]),
            (1.0, 1e-3, [1.0000003750008203159, -0.50000093750369142824, 0.75000328127030289934, -1.8750147657569691756]),
            (3.0, 0.05, [0.57741044642841008325, -0.096275216982430608404, 0.048171093234713430299, -0.040181692197751311987]),
        ];
        for (m, s, want) in reference {
            let got = positive_part_inverse_sqrt_normal_jet(m, s);
            for j in 0..4 {
                assert_close(&format!("J^({j})(m={m}, s={s})"), got[j], want[j], 1e-11);
            }
        }
    }

    /// The window and panel quadratures agree where they hand over.
    #[test]
    fn positive_part_inverse_sqrt_branches_agree_at_handover() {
        let l = gaussian_window_half_width();
        let s = 0.25;
        let panel = positive_part_inverse_sqrt_normal_jet(2.0 * l * s * (1.0 - 1e-12), s);
        let window = positive_part_inverse_sqrt_normal_jet(2.0 * l * s, s);
        for j in 0..4 {
            assert_close(&format!("J^({j}) handover"), panel[j], window[j], 1e-10);
        }
    }

    /// Both means reduce to the plug-in inverse link as `s → 0`.
    #[test]
    fn reciprocal_means_reduce_to_plug_in() {
        let m = 1.7;
        let pv = principal_value_inverse_normal_jet(m, 0.0);
        let pv_small = principal_value_inverse_normal_jet(m, 1e-9);
        let root = positive_part_inverse_sqrt_normal_jet(m, 0.0);
        let root_small = positive_part_inverse_sqrt_normal_jet(m, 1e-9);
        for j in 0..4 {
            assert_close("PV plug-in", pv_small[j], pv[j], 1e-14);
            assert_close("root plug-in", root_small[j], root[j], 1e-14);
        }
        assert_close("1/m", pv[0], 1.0 / m, 1e-15);
        assert_close("m^-1/2", root[0], 1.0 / m.sqrt(), 1e-15);
    }
}
