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
    dawson_jet_n::<4>(x)
}

/// Dawson's integral and its first `N − 1` derivatives, `N ≤ 5`. Below
/// [`DAWSON_ASYMPTOTIC_MIN`] the derivatives follow from `F' = 1 − 2xF` and
/// `F^{(j+1)} = −2j·F^{(j−1)} − 2x·F^{(j)}`.
fn dawson_jet_n<const N: usize>(x: f64) -> [f64; N] {
    if x.is_nan() {
        return [f64::NAN; N];
    }
    let a = x.abs();
    // At an infinite argument `F` and every derivative vanish, so the jet stays zero.
    let mut jet = [0.0_f64; N];
    if a < DAWSON_ASYMPTOTIC_MIN {
        let value = if a < DAWSON_SERIES_MAX {
            dawson_maclaurin(a)
        } else {
            dawson_rybicki(a)
        };
        jet[0] = value;
        if N > 1 {
            jet[1] = 1.0 - 2.0 * a * value;
        }
        for j in 1..N.saturating_sub(1) {
            jet[j + 1] = -2.0 * (j as f64) * jet[j - 1] - 2.0 * a * jet[j];
        }
    } else if a.is_finite() {
        // `F(x) = ½·M(x)` for the principal-value mean `M` at `s² = ½`.
        let series = reciprocal_moment_series::<N>(a, 0.5 / (a * a));
        for (value, term) in jet.iter_mut().zip(series) {
            *value = 0.5 * term;
        }
    }
    if x < 0.0 {
        // `F` is odd, so its even derivatives are odd too.
        for value in jet.iter_mut().step_by(2) {
            *value = -*value;
        }
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

/// The inverse-moment series `Σ_k (2k−1)!!·r^k·m^{-(2k+1)}` and its first
/// `N − 1` `m`-derivatives (`N ≤ 5`), for a signed ratio `r` with
/// `|r| ≤ 1/(2·DAWSON_ASYMPTOTIC_MIN²)`.
///
/// `r = s²/m²` is the moment expansion of `E[1/η]`, `η ~ N(m, s²)`. `r = −σ/m²`
/// is the expansion of the half-line integral `∫₀^∞ e^{−mt − σt²/2} dt`
/// ([`half_line_gaussian_log_jet`]), the same function continued to `σ = −s²`.
/// The series is asymptotic: its terms fall until `k ≈ 1/(2|r|)` and then
/// grow. It is summed until the terms stop changing the sums, or truncated at
/// the smallest term, whose size is of order `e^{-1/(2|r|)}`.
fn reciprocal_moment_series<const N: usize>(m: f64, r: f64) -> [f64; N] {
    let mut sums = [0.0_f64; N];
    // `t_k = (2k−1)!!·r^k`; the `j`-th derivative carries the rising
    // factorial `(2k+1)(2k+2)…(2k+j)`.
    let mut t = 1.0_f64;
    let mut k = 0.0_f64;
    let mut previous_top = f64::INFINITY;
    loop {
        let p = 2.0 * k + 1.0;
        let mut terms = [0.0_f64; N];
        let mut factor = t;
        for (j, term) in terms.iter_mut().enumerate() {
            *term = factor;
            factor *= p + j as f64;
        }
        if terms[N - 1].abs() > previous_top {
            break;
        }
        for (sum, term) in sums.iter_mut().zip(terms) {
            *sum += term;
        }
        let converged = sums
            .iter()
            .zip(terms)
            .all(|(sum, term)| term.abs() <= f64::EPSILON * sum.abs());
        if converged {
            break;
        }
        previous_top = terms[N - 1].abs();
        t *= p * r;
        k += 1.0;
    }
    let inv = 1.0 / m;
    let inv2 = inv * inv;
    std::array::from_fn(|j| match j {
        0 => sums[j] * inv,
        1 => -sums[j] * inv2,
        2 => sums[j] * inv2 * inv,
        3 => -sums[j] * inv2 * inv2,
        _ => sums[j] * inv2 * inv2 * inv,
    })
}

/// The principal-value mean `PV E[1/η]`, `η ~ N(m, s²)`, and its first three
/// `m`-derivatives. At `s = 0` this is the plug-in `1/m` and its derivatives.
pub fn principal_value_inverse_normal_jet(m: f64, s: f64) -> [f64; 4] {
    principal_value_inverse_normal_jet_n::<4>(m, s)
}

/// [`principal_value_inverse_normal_jet`] with its first `N − 1`
/// `m`-derivatives, `N ≤ 5`.
fn principal_value_inverse_normal_jet_n<const N: usize>(m: f64, s: f64) -> [f64; N] {
    if s == 0.0 {
        let inv = 1.0 / m;
        return std::array::from_fn(|j| match j {
            0 => inv,
            1 => -inv * inv,
            2 => 2.0 * inv * inv * inv,
            3 => -6.0 * inv * inv * inv * inv,
            _ => 24.0 * inv * inv * inv * inv * inv,
        });
    }
    if m.abs() >= DAWSON_ASYMPTOTIC_MIN * SQRT_2 * s {
        return reciprocal_moment_series::<N>(m, s * s / (m * m));
    }
    // `M(m) = 2c·F(cm)` with `c = 1/(√2 s)`, so `M^{(j)} = 2c^{j+1} F^{(j)}(cm)`.
    let c = 1.0 / (SQRT_2 * s);
    let f = dawson_jet_n::<N>(c * m);
    let mut scale = 2.0 * c;
    let mut jet = [0.0_f64; N];
    for (value, derivative) in jet.iter_mut().zip(f) {
        *value = scale * derivative;
        scale *= c;
    }
    jet
}

/// `J(μ, σ) = ∫₀^∞ e^{−μt − σt²/2} dt`, the mass a half-line leaves an
/// exponential-quadratic weight, as `[ln J, κ₁, κ₂, κ₃, κ₄]`.
///
/// `κ_k` are the cumulants of `t` under the normalized weight,
/// `κ_k = (−1)^k ∂^k_μ ln J`, so `∂_μκ_k = −κ_{k+1}`. Since `∂_σ J = −½∂²_μ J`,
/// the `σ`-rates follow from the same jet: `∂_σ ln J = −½(κ₂ + κ₁²)`.
///
/// For `σ < 0` the integral diverges, and `J` is continued by the principal
/// value `J(μ, σ) = PV E[1/η]` with `η ~ N(μ, −σ)`. Every branch below
/// satisfies `σ·∂_μJ = μJ − 1`, and the continuation is the solution of it that
/// keeps the expansion `J ~ μ⁻¹ Σ_k (2k−1)!!·(−σ/μ²)^k` on both sides of
/// `σ = 0`. So at `μ > 0`, `J` is smooth through `σ = 0`, where it is the
/// exponential mass `1/μ`: the Laplace factor of a boundary with multiplier
/// `μ`, whose normal curvature `σ` of either sign is a correction of relative
/// order `σ/μ²`.
///
/// The branches all evaluate this one function:
/// - `σ ≤ 0`: the principal value, through the series and Dawson branches of
///   [`principal_value_inverse_normal_jet`] at `s = √−σ`;
/// - `σ > 0` with `μ ≥ DAWSON_ASYMPTOTIC_MIN·√(2σ)`: the same series with the
///   alternating ratio `−σ/μ²`, where the log-normal-CDF form would cancel;
/// - `σ > 0` otherwise: `J = √(2π/σ)·e^{z²/2}·Φ(z)` with `z = −μ/√σ`.
///
/// Returns `None` where neither the integral nor its continuation is a
/// positive mass (`μ ≤ 0` with `σ ≤ 0`), or for a non-finite argument.
pub fn half_line_gaussian_log_jet(mu: f64, sigma: f64) -> Option<[f64; 5]> {
    if !(mu.is_finite() && sigma.is_finite()) {
        return None;
    }
    if sigma > 0.0 && !(mu > 0.0 && mu >= DAWSON_ASYMPTOTIC_MIN * SQRT_2 * sigma.sqrt()) {
        let root = sigma.sqrt();
        let z = -mu / root;
        let f = crate::probability::normal_logcdf_derivatives(z);
        let log_mass = 0.5 * (2.0 * std::f64::consts::PI / sigma).ln() + 0.5 * z * z + f[0];
        let jet = [
            log_mass,
            (z + f[1]) / root,
            (1.0 + f[2]) / sigma,
            f[3] / (sigma * root),
            f[4] / (sigma * sigma),
        ];
        return jet.iter().all(|value| value.is_finite()).then_some(jet);
    }
    if !(mu > 0.0) {
        return None;
    }
    let derivatives = if sigma > 0.0 {
        reciprocal_moment_series::<5>(mu, -sigma / (mu * mu))
    } else {
        principal_value_inverse_normal_jet_n::<5>(mu, (-sigma).sqrt())
    };
    let mass = derivatives[0];
    if !(mass > 0.0 && mass.is_finite()) {
        return None;
    }
    // Raw moments `E[tʲ] = (−1)ʲ J^{(j)}/J`, then cumulants.
    let m1 = -derivatives[1] / mass;
    let m2 = derivatives[2] / mass;
    let m3 = -derivatives[3] / mass;
    let m4 = derivatives[4] / mass;
    let jet = [
        mass.ln(),
        m1,
        m2 - m1 * m1,
        m3 - 3.0 * m2 * m1 + 2.0 * m1 * m1 * m1,
        m4 - 4.0 * m3 * m1 - 3.0 * m2 * m2 + 12.0 * m2 * m1 * m1 - 6.0 * m1 * m1 * m1 * m1,
    ];
    jet.iter().all(|value| value.is_finite()).then_some(jet)
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

    /// `[ln J, E[t], …, E[t⁴]]` of `∫₀^∞ e^{−μt − σt²/2} dt`, `σ > 0`, by 20-point
    /// Gauss–Legendre on 400 panels of `[0, T]`. Past `T = t* + √(2L/σ)`, with
    /// `t* = max(0, −μ/σ)` the weight's peak and `L = −ln ε²`, the exponent has
    /// fallen by more than `L` below its peak.
    fn half_line_moments_by_quadrature(mu: f64, sigma: f64) -> [f64; 5] {
        let (nodes, weights) = crate::special::gauss_legendre(20);
        let peak = (-mu / sigma).max(0.0);
        let exponent = |t: f64| -mu * t - 0.5 * sigma * t * t;
        let top = exponent(peak);
        let end = peak + (-4.0 * f64::EPSILON.ln() / sigma).sqrt();
        let panels = 400;
        let width = end / panels as f64;
        let mut sums = [0.0_f64; 5];
        for panel in 0..panels {
            let centre = (panel as f64 + 0.5) * width;
            for (node, weight) in nodes.iter().zip(&weights) {
                let t = centre + 0.5 * width * node;
                let mass = 0.5 * width * weight * (exponent(t) - top).exp();
                let mut power = 1.0;
                for sum in &mut sums {
                    *sum += mass * power;
                    power *= t;
                }
            }
        }
        [sums[0].ln() + top, sums[1] / sums[0], sums[2] / sums[0], sums[3] / sums[0], sums[4] / sums[0]]
    }

    fn raw_moments_from_cumulants(jet: &[f64; 5]) -> [f64; 4] {
        let [_, k1, k2, k3, k4] = *jet;
        [
            k1,
            k2 + k1 * k1,
            k3 + 3.0 * k2 * k1 + k1 * k1 * k1,
            k4 + 4.0 * k3 * k1 + 3.0 * k2 * k2 + 6.0 * k2 * k1 * k1 + k1 * k1 * k1 * k1,
        ]
    }

    /// Against quadrature on every `σ > 0` branch: the alternating series
    /// (`μ ≥ 7√(2σ)`), the log-normal-CDF form on both sides of its seam with
    /// the series, and an interior peak (`μ < 0`). The bar: the log-CDF form's
    /// brackets `z + f′` and `1 + f″` cancel by at most `1 + z²` with `z² = μ²/σ ≤ 98`
    /// on that branch, and the moment-to-cumulant map amplifies relative error
    /// by at most `Σ|terms|/κ₄ = 15` on the exponential law, the worst of these
    /// weights. So `3·15·(1 + 98)·ε` bounds every entry, with a factor 3 for the
    /// quadrature's own rounding.
    #[test]
    fn half_line_log_jet_matches_quadrature_2765() {
        let seam = 3.0 * 3.0 / (2.0 * DAWSON_ASYMPTOTIC_MIN * DAWSON_ASYMPTOTIC_MIN);
        let bar = 3.0 * 15.0 * 99.0 * f64::EPSILON;
        for (mu, sigma) in [
            (3.0, 0.05),
            (3.0, seam),
            (3.0, seam * (1.0 + 4.0 * f64::EPSILON)),
            (3.0, 0.5),
            (0.5, 2.0),
            (0.0, 1.0),
            (-2.0, 1.0),
        ] {
            let jet = half_line_gaussian_log_jet(mu, sigma).expect("a positive-σ half-line mass");
            let reference = half_line_moments_by_quadrature(mu, sigma);
            assert_close(&format!("ln J({mu}, {sigma})"), jet[0], reference[0], bar);
            let moments = raw_moments_from_cumulants(&jet);
            for j in 0..4 {
                assert_close(&format!("E[t^{}]({mu}, {sigma})", j + 1), moments[j], reference[j + 1], bar);
            }
        }
    }

    /// `σ·∂_μJ = μJ − 1` and its `μ`-derivatives hold on every branch, the
    /// principal-value continuation at `σ < 0` included. In raw moments:
    /// `σm₁ + μ = 1/J`, `σm₂ + μm₁ = 1`, `σm₃ + μm₂ = 2m₁`, `σm₄ + μm₃ = 3m₂`.
    /// Each identity is judged relative to the magnitudes it sums, at the same
    /// bar as the quadrature comparison.
    #[test]
    fn half_line_log_jet_satisfies_its_differential_equation_2765() {
        let bar = 3.0 * 15.0 * 99.0 * f64::EPSILON;
        for (mu, sigma) in [
            (3.0, 0.5),
            (3.0, 0.05),
            (3.0, 1e-4),
            (3.0, -1e-4),
            (3.0, -0.05),
            (3.0, -0.5),
            (3.0, -4.0),
            (0.7, -0.2),
            (0.5, 2.0),
            (-2.0, 1.0),
        ] {
            let jet = half_line_gaussian_log_jet(mu, sigma).expect("a half-line mass or its continuation");
            let [m1, m2, m3, m4] = raw_moments_from_cumulants(&jet);
            let inverse_mass = (-jet[0]).exp();
            let checks = [
                (sigma * m1 + mu, inverse_mass, (sigma * m1).abs() + mu.abs()),
                (sigma * m2 + mu * m1, 1.0, (sigma * m2).abs() + (mu * m1).abs()),
                (sigma * m3 + mu * m2, 2.0 * m1, (sigma * m3).abs() + (mu * m2).abs()),
                (sigma * m4 + mu * m3, 3.0 * m2, (sigma * m4).abs() + (mu * m3).abs()),
            ];
            for (order, (lhs, rhs, magnitude)) in checks.into_iter().enumerate() {
                assert!(
                    (lhs - rhs).abs() <= bar * magnitude.max(rhs.abs()),
                    "order {order} at (μ={mu}, σ={sigma}): {lhs:e} against {rhs:e}"
                );
            }
        }
    }

    /// At `σ = 0` the mass is the exponential law's, and on either side it
    /// follows the one expansion `ln J = −ln μ − s + (5/2)s² − (37/3)s³ + …`,
    /// `μκ₁ = 1 − 2s + 10s² − 74s³ + …`, `s = σ/μ²`: the continuation is smooth
    /// through `σ = 0`. Each bar is the first omitted coefficient rounded up
    /// (`37/3 → 13`, `10 → 11`); at `|s| ≤ 3·10⁻³` the next terms (`353/4·s⁴`,
    /// `74|s|³`) stay inside that rounding.
    #[test]
    fn half_line_log_jet_is_smooth_through_zero_curvature_2765() {
        let mu = 2.5;
        let at_zero = half_line_gaussian_log_jet(mu, 0.0).expect("the exponential mass");
        let exponential = [-mu.ln(), 1.0 / mu, 1.0 / (mu * mu), 2.0 / mu.powi(3), 6.0 / mu.powi(4)];
        for j in 0..5 {
            assert_close(&format!("exponential jet {j}"), at_zero[j], exponential[j], 64.0 * f64::EPSILON);
        }
        for s in [3e-3, 1e-3, 1e-5, -1e-5, -1e-3, -3e-3] {
            let jet = half_line_gaussian_log_jet(mu, s * mu * mu).expect("a mass near zero curvature");
            let log_deviation = jet[0] + mu.ln() - (-s + 2.5 * s * s);
            assert!(log_deviation.abs() <= 13.0 * s.abs().powi(3), "ln J at s={s}: {log_deviation:e}");
            let mean_deviation = mu * jet[1] - (1.0 - 2.0 * s);
            assert!(mean_deviation.abs() <= 11.0 * s * s, "μκ₁ at s={s}: {mean_deviation:e}");
        }
    }

    /// Neither the integral nor its continuation is a positive mass where the
    /// weight has no decay along the half-line.
    #[test]
    fn half_line_log_jet_refuses_a_weight_without_decay_2765() {
        for (mu, sigma) in [(-1.0, -1.0), (0.0, 0.0), (0.0, -1.0), (-1.0, 0.0), (1.0, f64::NAN)] {
            assert!(half_line_gaussian_log_jet(mu, sigma).is_none(), "(μ={mu}, σ={sigma})");
        }
    }
}
