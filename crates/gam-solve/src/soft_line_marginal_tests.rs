//! Pins for the soft-line marginal (#3173).
//!
//! - A Gaussian slice with moving ends has a closed form. It pins the value, the Leibniz end terms and the covariance
//!   term of the Hessian.
//! - A quartic slice with a cone term has an inner-mode fold. It pins that the marginal stays finite and smooth where
//!   the single-mode Laplace gradient diverges, against an independent quadrature and against finite differences.
//! - The refusals, and the stationary-slicing reduction on synthetic marginals.

use super::*;
use gam_math::probability::normal_cdf;
use gam_math::special::gauss_legendre;
use ndarray::array;

/// A cost's own rounding: a handful of rounded operations on its magnitude.
fn rounding(value: f64, operations: usize) -> f64 {
    accumulation_growth(operations) * value.abs()
}

/// A fourth-order central difference of `f` at `x` with step `h`: `[−f(x+2h) + 8f(x+h) − 8f(x−h) + f(x−2h)]/(12h)`.
///
/// Its truncation error is `h⁴f⁽⁵⁾/30`. Its rounding is at most `(1 + 8 + 8 + 1)/(12h) = 1.5/h` times the error of
/// one evaluation of `f`.
fn central_difference(f: &mut dyn FnMut(f64) -> f64, x: f64, h: f64) -> f64 {
    (-f(x + 2.0 * h) + 8.0 * f(x + h) - 8.0 * f(x - h) + f(x - 2.0 * h)) / (12.0 * h)
}

/// The fourth-order difference at `h` with a tolerance read from the data.
///
/// For a fourth-order rule the difference at `2h` errs by `16×` that at `h`, so `|D(h) − D(2h)|/15` estimates the
/// truncation error at `h`. The tolerance charges twice that estimate, plus the rounding of both differences:
/// `1.5·band/h` for `D(h)` and `0.75·band/h` for `D(2h)`, which also enter the estimate.
fn difference_with_tolerance(f: &mut dyn FnMut(f64) -> f64, x: f64, h: f64, band: f64) -> (f64, f64) {
    let fine = central_difference(f, x, h);
    let coarse = central_difference(f, x, 2.0 * h);
    let rounding = 1.5 * band / h + 0.75 * band / h;
    (fine, 2.0 * (fine - coarse).abs() / 15.0 + 2.0 * rounding)
}

// ---------------------------------------------------------------------------------------------------------------------
// Gaussian slice with moving ends.
//
// q(s; μ, λ) = κ(s − μ)²/2 with κ = e^λ. Then q_μ = −κ(s − μ), q_λ = κ(s − μ)²/2, q_μμ = κ, q_μλ = −κ(s − μ),
// q_λλ = κ(s − μ)²/2 and q_s = κ(s − μ).
//
// The ends are a = −0.7 + 0.2μ² and b = 1.1 + 0.3μλ, so a_μ = 0.4μ, a_μμ = 0.4, b_μ = 0.3λ, b_λ = 0.3μ and b_μλ = 0.3.
//
// ∫_a^b e^{−q} ds = √(2π/κ)·[Φ(√κ(b − μ)) − Φ(√κ(a − μ))], and T is minus its logarithm.
// ---------------------------------------------------------------------------------------------------------------------

fn gaussian_slice(s: f64, mu: f64, lambda: f64) -> SoftLineSlice {
    let kappa = lambda.exp();
    let offset = s - mu;
    let cost = 0.5 * kappa * offset * offset;
    SoftLineSlice {
        cost,
        cost_band: rounding(cost, 6),
        cost_s: kappa * offset,
        cost_gradient: array![-kappa * offset, cost],
        cost_hessian: array![[kappa, -kappa * offset], [-kappa * offset, cost]],
        derivative_band: accumulation_growth(6) * (kappa * (1.0 + offset.abs() + offset * offset)),
    }
}

fn gaussian_ends(mu: f64, lambda: f64) -> (SoftLineEnd, SoftLineEnd) {
    let lower = SoftLineEnd {
        at: -0.7 + 0.2 * mu * mu,
        gradient: array![0.4 * mu, 0.0],
        hessian: array![[0.4, 0.0], [0.0, 0.0]],
    };
    let upper = SoftLineEnd {
        at: 1.1 + 0.3 * mu * lambda,
        gradient: array![0.3 * lambda, 0.3 * mu],
        hessian: array![[0.0, 0.3], [0.3, 0.0]],
    };
    (lower, upper)
}

fn gaussian_closed_form(mu: f64, lambda: f64) -> f64 {
    let kappa = lambda.exp();
    let (lower, upper) = gaussian_ends(mu, lambda);
    let root = kappa.sqrt();
    let mass = (2.0 * std::f64::consts::PI / kappa).sqrt()
        * (normal_cdf(root * (upper.at - mu)) - normal_cdf(root * (lower.at - mu)));
    -mass.ln()
}

fn gaussian_marginal(mu: f64, lambda: f64) -> SoftLineMarginal {
    let (lower, upper) = gaussian_ends(mu, lambda);
    soft_line_marginal::<(), _>(&SoftLineInterval::Bounded { lower, upper }, 2, |s| {
        Ok(gaussian_slice(s, mu, lambda))
    })
    .expect("the Gaussian soft line resolves")
}

/// The closed form's own error.
///
/// `libm`'s `erfc` is faithful to about one ulp. The mass is a difference of two `Φ` values near `0.03` and `0.99`, so
/// it keeps their relative error. The prefactor, the difference and the logarithm add a few more roundings, and each
/// relative error `r` in the mass moves `T` by `r` absolutely. Sixteen ulps of `1 + |T|` covers them with room.
fn closed_form_band(value: f64) -> f64 {
    16.0 * f64::EPSILON * (1.0 + value.abs())
}

#[test]
fn gaussian_bounded_value_matches_closed_form() {
    let (mu, lambda) = (0.2, 1.1);
    let marginal = gaussian_marginal(mu, lambda);
    let exact = gaussian_closed_form(mu, lambda);
    let gap = (marginal.value - exact).abs();
    assert!(
        gap <= marginal.value_band + closed_form_band(exact),
        "T = {} against the closed form {exact}: gap {gap}, band {}",
        marginal.value,
        marginal.value_band
    );
    // The band is a rounding band, not a slack: it sits within a few hundred ulps of |T|.
    assert!(marginal.value_band < 1e3 * f64::EPSILON * (1.0 + exact.abs()), "band {}", marginal.value_band);
}

#[test]
fn gaussian_bounded_gradient_matches_closed_form_differences() {
    let (mu, lambda) = (0.2, 1.1);
    let marginal = gaussian_marginal(mu, lambda);
    let band = closed_form_band(gaussian_closed_form(mu, lambda));
    let h = 1e-3;
    let (d_mu, tol_mu) = difference_with_tolerance(&mut |x| gaussian_closed_form(x, lambda), mu, h, band);
    let (d_lambda, tol_lambda) = difference_with_tolerance(&mut |x| gaussian_closed_form(mu, x), lambda, h, band);
    for (name, analytic, analytic_band, difference, tolerance) in [
        ("T_μ", marginal.gradient[0], marginal.gradient_band[0], d_mu, tol_mu),
        ("T_λ", marginal.gradient[1], marginal.gradient_band[1], d_lambda, tol_lambda),
    ] {
        assert!(
            (analytic - difference).abs() <= analytic_band + tolerance,
            "{name} = {analytic} against the difference {difference} (tolerance {tolerance})"
        );
    }
}

#[test]
fn gaussian_bounded_hessian_matches_differences_of_the_gradient() {
    // The Hessian's covariance and end terms against fourth-order differences of the analytic gradient, which the
    // previous pin ties to the closed form.
    let (mu, lambda) = (0.2, 1.1);
    let marginal = gaussian_marginal(mu, lambda);
    let h = 1e-3;
    for (i, name) in ["μ", "λ"].iter().enumerate() {
        let band = marginal.gradient_band[i];
        let (by_mu, tol_mu) =
            difference_with_tolerance(&mut |x| gaussian_marginal(x, lambda).gradient[i], mu, h, band);
        let (by_lambda, tol_lambda) =
            difference_with_tolerance(&mut |x| gaussian_marginal(mu, x).gradient[i], lambda, h, band);
        for (j, difference, tolerance) in [(0, by_mu, tol_mu), (1, by_lambda, tol_lambda)] {
            let analytic = marginal.hessian[[i, j]];
            assert!(
                (analytic - difference).abs() <= marginal.hessian_band[[i, j]] + tolerance,
                "T_{name}{j} = {analytic} against the difference {difference} (tolerance {tolerance})"
            );
        }
    }
}

#[test]
fn gaussian_end_terms_are_load_bearing() {
    // Holding the ends fixed drops the Leibniz terms. The gradient must then differ from the moving-end one by the
    // end densities times the end velocities, which are resolved here: the check fails for a kernel without them.
    let (mu, lambda) = (0.2, 1.1);
    let moving = gaussian_marginal(mu, lambda);
    let (mut lower, mut upper) = gaussian_ends(mu, lambda);
    for end in [&mut lower, &mut upper] {
        end.gradient.fill(0.0);
        end.hessian.fill(0.0);
    }
    let fixed = soft_line_marginal::<(), _>(&SoftLineInterval::Bounded { lower, upper }, 2, |s| {
        Ok(gaussian_slice(s, mu, lambda))
    })
    .expect("the fixed-end soft line resolves");
    let (lower, upper) = gaussian_ends(mu, lambda);
    let density = |s: f64| (-(gaussian_slice(s, mu, lambda).cost - moving.value)).exp();
    for i in 0..2 {
        let expected = density(upper.at) * upper.gradient[i] - density(lower.at) * lower.gradient[i];
        let shift = fixed.gradient[i] - moving.gradient[i];
        assert!(
            (shift - expected).abs() <= moving.gradient_band[i] + fixed.gradient_band[i] + 1e-12 * expected.abs(),
            "coordinate {i}: fixed − moving = {shift}, expected p(b)b_i − p(a)a_i = {expected}"
        );
        assert!(expected.abs() > 1e3 * (moving.gradient_band[i] + fixed.gradient_band[i]));
    }
}

#[test]
fn gaussian_unbounded_and_half_line_match_closed_forms() {
    let (mu, lambda): (f64, f64) = (0.3, 0.4);
    let kappa = lambda.exp();
    let slice = |s: f64| -> Result<SoftLineSlice, ()> { Ok(gaussian_slice(s, mu, lambda)) };

    // (−∞, ∞): T = −½ln(2π/κ), T_μ = 0, T_λ = ½, and T_λλ = 0 because only the normalizer depends on λ.
    let whole = soft_line_marginal(&SoftLineInterval::Unbounded { centre: 0.0, scale: 0.5 }, 2, slice)
        .expect("the Gaussian line resolves");
    let exact = -0.5 * (2.0 * std::f64::consts::PI / kappa).ln();
    assert!((whole.value - exact).abs() <= whole.value_band + closed_form_band(exact), "T = {}", whole.value);
    assert!(whole.gradient[0].abs() <= whole.gradient_band[0], "T_μ = {}", whole.gradient[0]);
    assert!((whole.gradient[1] - 0.5).abs() <= whole.gradient_band[1], "T_λ = {}", whole.gradient[1]);
    // T_μμ = E[q_μμ] − Var(q_μ) = κ − κ = 0, T_μλ = 0 by symmetry, and T_λλ = E[q_λλ] − Var(q_λ) = ½ − ½ = 0.
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                whole.hessian[[i, j]].abs() <= whole.hessian_band[[i, j]],
                "T_{i}{j} = {} (band {})",
                whole.hessian[[i, j]],
                whole.hessian_band[[i, j]]
            );
        }
    }

    // [μ, ∞) with a fixed end: half the mass, T = −½ln(π/(2κ)).
    let lower = SoftLineEnd { at: mu, gradient: array![0.0, 0.0], hessian: Array2::zeros((2, 2)) };
    let half = soft_line_marginal(&SoftLineInterval::LowerBounded { lower, scale: 0.5 }, 2, slice)
        .expect("the Gaussian half-line resolves");
    let exact = -0.5 * (std::f64::consts::PI / (2.0 * kappa)).ln();
    assert!((half.value - exact).abs() <= half.value_band + closed_form_band(exact), "T = {}", half.value);
    // T_μ = E[q_μ] = −κE[s − μ] = −√(2κ/π) on the half-line.
    let expected = -(2.0 * kappa / std::f64::consts::PI).sqrt();
    assert!(
        (half.gradient[0] - expected).abs() <= half.gradient_band[0] + closed_form_band(expected),
        "T_μ = {} against {expected}",
        half.gradient[0]
    );
}

#[test]
fn the_ray_scale_moves_the_work_not_the_value() {
    let (mu, lambda) = (0.3, 0.4);
    let slice = |s: f64| -> Result<SoftLineSlice, ()> { Ok(gaussian_slice(s, mu, lambda)) };
    let narrow = soft_line_marginal(&SoftLineInterval::Unbounded { centre: 0.0, scale: 0.05 }, 2, slice).unwrap();
    let wide = soft_line_marginal(&SoftLineInterval::Unbounded { centre: 0.0, scale: 20.0 }, 2, slice).unwrap();
    assert!((narrow.value - wide.value).abs() <= narrow.value_band + wide.value_band);
    for i in 0..2 {
        assert!((narrow.gradient[i] - wide.gradient[i]).abs() <= narrow.gradient_band[i] + wide.gradient_band[i]);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// An inner-mode fold.
//
// q(s; ρ) = n(s⁴/4 − s²/2 + ρs) + c(s), with the cone term c(s) = ½ln(n(1 + 0.3s²)/2π). Then q_ρ = ns and q_ρρ = 0, so
// T_ρ = nE[s] and T_ρρ = −n²Var(s).
//
// The right-hand minimum of q meets the saddle at the fold (s_f, ρ_f), where q_s = q_ss = 0:
// - q_s = n(s³ − s + ρ) + c'(s), with c' = 0.3s/(1 + 0.3s²);
// - q_ss = n(3s² − 1) + c''(s), with c'' = 0.3(1 − 0.3s²)/(1 + 0.3s²)²;
// - q_sss = 6ns + c'''(s), with c''' = −0.18s(3 − 0.3s²)/(1 + 0.3s²)³.
//
// Along that minimum, the single-mode Laplace criterion V_L = q(s*) + ½ln(q_ss(s*)/2π) has
// dV_L/dρ = ns* − ½q_sss·n/q_ss², which diverges as q_ss → 0 at the fold.
// ---------------------------------------------------------------------------------------------------------------------

const FOLD_N: f64 = 50.0;

fn fold_q(s: f64, rho: f64) -> f64 {
    FOLD_N * (s.powi(4) / 4.0 - s * s / 2.0 + rho * s)
        + 0.5 * (FOLD_N * (1.0 + 0.3 * s * s) / (2.0 * std::f64::consts::PI)).ln()
}

fn fold_q_s(s: f64, rho: f64) -> f64 {
    FOLD_N * (s.powi(3) - s + rho) + 0.3 * s / (1.0 + 0.3 * s * s)
}

fn fold_q_ss(s: f64) -> f64 {
    let cone = 1.0 + 0.3 * s * s;
    FOLD_N * (3.0 * s * s - 1.0) + 0.3 * (1.0 - 0.3 * s * s) / (cone * cone)
}

fn fold_q_sss(s: f64) -> f64 {
    let cone = 1.0 + 0.3 * s * s;
    6.0 * FOLD_N * s - 0.18 * s * (3.0 - 0.3 * s * s) / (cone * cone * cone)
}

fn fold_slice(s: f64, rho: f64) -> SoftLineSlice {
    let cost = fold_q(s, rho);
    let magnitude = FOLD_N * (s.powi(4) / 4.0 + s * s / 2.0 + (rho * s).abs()) + cost.abs();
    SoftLineSlice {
        cost,
        cost_band: accumulation_growth(12) * magnitude,
        cost_s: fold_q_s(s, rho),
        cost_gradient: array![FOLD_N * s],
        cost_hessian: array![[0.0]],
        derivative_band: accumulation_growth(12) * (FOLD_N * (s.abs().powi(3) + s.abs() + rho.abs()) + 1.0),
    }
}

fn fold_marginal(rho: f64) -> SoftLineMarginal {
    soft_line_marginal::<(), _>(&SoftLineInterval::Unbounded { centre: 0.0, scale: 0.5 }, 1, |s| {
        Ok(fold_slice(s, rho))
    })
    .expect("the fold's soft line resolves")
}

/// Bisection to adjacent `f64`s for a sign change of `f` on `[lower, upper]`.
fn bisect(f: impl Fn(f64) -> f64, mut lower: f64, mut upper: f64) -> f64 {
    let lower_sign = f(lower).signum();
    assert!(lower_sign != f(upper).signum(), "no sign change on [{lower}, {upper}]");
    loop {
        let middle = 0.5 * (lower + upper);
        if !(lower < middle && middle < upper) {
            return middle;
        }
        if f(middle).signum() == lower_sign {
            lower = middle;
        } else {
            upper = middle;
        }
    }
}

/// The fold `(s_f, ρ_f)` of the right-hand minimum.
fn fold_point() -> (f64, f64) {
    let s_f = bisect(fold_q_ss, 0.3, 1.0);
    let rho_f = -(s_f.powi(3) - s_f) - (0.3 * s_f / (1.0 + 0.3 * s_f * s_f)) / FOLD_N;
    (s_f, rho_f)
}

/// An independent reference: composite 16-point Gauss–Legendre on `[−3, 3]` in 400 panels.
///
/// Outside `[−3, 3]`, `q − min q ≥ n(81/4 − 9/2 − 3|ρ|) − 1 > 700` for `|ρ| < 1`, so the truncated mass is below
/// `e^{−700}` of the total. The panels are 0.015 wide against a curvature scale of `1/√(q_ss) ≥ 0.02` away from the
/// fold, so the rule's own error is far below rounding. The returned band is the rounding of the three sums,
/// `γ_m·Σ|terms|` over `m = 6400` terms, carried through `T`, `T_ρ` and `T_ρρ`.
struct FoldReference {
    value: f64,
    gradient: f64,
    hessian: f64,
    value_band: f64,
    gradient_band: f64,
    hessian_band: f64,
}

fn fold_reference(rho: f64) -> FoldReference {
    let (nodes, weights) = gauss_legendre(16);
    let panels = 400;
    let width = 6.0 / panels as f64;
    let mut points = Vec::with_capacity(panels * nodes.len());
    for panel in 0..panels {
        let centre = -3.0 + (panel as f64 + 0.5) * width;
        for (node, weight) in nodes.iter().zip(&weights) {
            points.push((centre + 0.5 * width * node, 0.5 * width * weight));
        }
    }
    let reference = points.iter().map(|&(s, _)| fold_q(s, rho)).fold(f64::INFINITY, f64::min);
    let (mut mass, mut first, mut second) = (0.0, 0.0, 0.0);
    for &(s, weight) in &points {
        let g = weight * (-(fold_q(s, rho) - reference)).exp();
        mass += g;
        first += g * s;
        second += g * s * s;
    }
    let growth = accumulation_growth(points.len() + 8);
    let mean = first / mass;
    let variance = second / mass - mean * mean;
    FoldReference {
        value: reference - mass.ln(),
        gradient: FOLD_N * mean,
        hessian: -FOLD_N * FOLD_N * variance,
        // Every term is positive in `mass` and `second`, and `|s| ≤ 3` bounds `first`'s magnitude by `3·mass`.
        value_band: growth * (1.0 + reference.abs()),
        gradient_band: FOLD_N * growth * (mean.abs() + 3.0),
        hessian_band: FOLD_N * FOLD_N * growth * (2.0 * second / mass + 6.0 * mean.abs() + mean * mean),
    }
}

#[test]
fn fold_marginal_matches_the_reference_through_the_fold() {
    let (_, rho_f) = fold_point();
    for rho in [0.2, rho_f - 1e-2, rho_f - 1e-6, rho_f, rho_f + 1e-6, rho_f + 1e-2, 0.6] {
        let marginal = fold_marginal(rho);
        let reference = fold_reference(rho);
        assert!(marginal.value.is_finite() && marginal.gradient[0].is_finite() && marginal.hessian[[0, 0]].is_finite());
        for (name, value, band, exact, exact_band) in [
            ("T", marginal.value, marginal.value_band, reference.value, reference.value_band),
            ("T_ρ", marginal.gradient[0], marginal.gradient_band[0], reference.gradient, reference.gradient_band),
            ("T_ρρ", marginal.hessian[[0, 0]], marginal.hessian_band[[0, 0]], reference.hessian, reference.hessian_band),
        ] {
            assert!(
                (value - exact).abs() <= band + exact_band,
                "ρ = {rho}: {name} = {value} against the reference {exact} (bands {band}, {exact_band})"
            );
        }
    }
}

#[test]
fn fold_gradient_matches_differences_across_the_fold() {
    let (_, rho_f) = fold_point();
    let h = 1e-4;
    for rho in [rho_f - 1e-2, rho_f, rho_f + 1e-2] {
        let marginal = fold_marginal(rho);
        let band = [-2.0, -1.0, 1.0, 2.0].iter().map(|k| fold_marginal(rho + k * h).value_band).fold(0.0, f64::max);
        let (difference, tolerance) = difference_with_tolerance(&mut |x| fold_marginal(x).value, rho, h, band);
        assert!(
            (marginal.gradient[0] - difference).abs() <= marginal.gradient_band[0] + tolerance,
            "ρ = {rho}: T_ρ = {} against the difference {difference} (tolerance {tolerance})",
            marginal.gradient[0]
        );
        let band = [-2.0, -1.0, 1.0, 2.0]
            .iter()
            .map(|k| fold_marginal(rho + k * h).gradient_band[0])
            .fold(0.0, f64::max);
        let (difference, tolerance) = difference_with_tolerance(&mut |x| fold_marginal(x).gradient[0], rho, h, band);
        assert!(
            (marginal.hessian[[0, 0]] - difference).abs() <= marginal.hessian_band[[0, 0]] + tolerance,
            "ρ = {rho}: T_ρρ = {} against the difference {difference} (tolerance {tolerance})",
            marginal.hessian[[0, 0]]
        );
    }
}

#[test]
fn laplace_gradient_diverges_where_the_marginal_does_not() {
    let (s_f, rho_f) = fold_point();
    let rho = rho_f - 1e-8;
    // The right-hand minimum just before the fold: q_s < 0 at s_f and q_s > 0 at 2, with q_ss > 0 between.
    let minimum = bisect(|s| fold_q_s(s, rho), s_f, 2.0);
    let curvature = fold_q_ss(minimum);
    assert!(curvature > 0.0, "q_ss = {curvature} at the tracked minimum");
    let laplace = FOLD_N * minimum - 0.5 * fold_q_sss(minimum) * FOLD_N / (curvature * curvature);
    let largest = [0.2, rho_f - 1e-2, rho_f - 1e-8, rho_f, rho_f + 1e-2, 0.6]
        .iter()
        .map(|&rho| fold_marginal(rho).gradient[0].abs())
        .fold(0.0, f64::max);
    assert!(
        laplace.abs() > 1e3 * largest,
        "dV_L/dρ = {laplace} at ρ_f − 1e-8 against max |T_ρ| = {largest}"
    );
}

#[test]
fn fold_hessian_is_continuous_at_the_fold() {
    // Across ρ_f ± 1e-6 the Hessian moves by at most 2e-6·max|T_ρρρ| plus its bands. T_ρρρ = −n³κ_3(s), and
    // |κ_3(s)| ≤ E|s − E s|³ ≤ 6³ on s's effective support [−3, 3].
    let (_, rho_f) = fold_point();
    let below = fold_marginal(rho_f - 1e-6);
    let above = fold_marginal(rho_f + 1e-6);
    let jump = (above.hessian[[0, 0]] - below.hessian[[0, 0]]).abs();
    let bound = 2e-6 * FOLD_N.powi(3) * 216.0 + above.hessian_band[[0, 0]] + below.hessian_band[[0, 0]];
    assert!(jump <= bound, "T_ρρ jumps by {jump} across the fold (bound {bound})");
}

// ---------------------------------------------------------------------------------------------------------------------
// Refusals.
// ---------------------------------------------------------------------------------------------------------------------

fn flat_slice(rho: f64) -> Result<SoftLineSlice, ()> {
    assert!(rho.is_finite(), "a flat slice is only probed at finite rho, not {rho}");
    Ok(SoftLineSlice {
        cost: 0.0,
        cost_band: 0.0,
        cost_s: 0.0,
        cost_gradient: array![0.0],
        cost_hessian: array![[0.0]],
        derivative_band: 0.0,
    })
}

fn fixed_end(at: f64) -> SoftLineEnd {
    SoftLineEnd { at, gradient: array![0.0], hessian: array![[0.0]] }
}

#[test]
fn a_flat_half_line_is_improper() {
    let error = soft_line_marginal(&SoftLineInterval::LowerBounded { lower: fixed_end(0.0), scale: 1.0 }, 1, flat_slice)
        .unwrap_err();
    assert!(matches!(error, SoftLineError::ImproperSoftLine { .. }), "{error:?}");
}

#[test]
fn a_non_finite_slice_is_refused() {
    let error = soft_line_marginal::<(), _>(
        &SoftLineInterval::Bounded { lower: fixed_end(0.0), upper: fixed_end(1.0) },
        1,
        |s| {
            let mut slice = flat_slice(s)?;
            if s > 0.5 {
                slice.cost = f64::NAN;
            }
            Ok(slice)
        },
    )
    .unwrap_err();
    assert!(matches!(error, SoftLineError::NonFiniteSlice { at } if at > 0.5), "{error:?}");
}

#[test]
fn malformed_intervals_and_slices_are_refused() {
    let empty = soft_line_marginal(
        &SoftLineInterval::Bounded { lower: fixed_end(1.0), upper: fixed_end(1.0) },
        1,
        flat_slice,
    );
    assert!(matches!(empty, Err(SoftLineError::EmptyInterval { .. })), "{empty:?}");
    let scale = soft_line_marginal(&SoftLineInterval::Unbounded { centre: 0.0, scale: 0.0 }, 1, flat_slice);
    assert!(matches!(scale, Err(SoftLineError::NonPositiveScale { .. })), "{scale:?}");
    let dimension = soft_line_marginal(
        &SoftLineInterval::Bounded { lower: fixed_end(0.0), upper: fixed_end(1.0) },
        2,
        flat_slice,
    );
    assert!(matches!(dimension, Err(SoftLineError::DimensionMismatch { expected: 2, found: 1 })), "{dimension:?}");
    let failed = soft_line_marginal::<&str, _>(
        &SoftLineInterval::Bounded { lower: fixed_end(0.0), upper: fixed_end(1.0) },
        1,
        |_| Err("no slice"),
    );
    assert!(matches!(failed, Err(SoftLineError::Slice("no slice"))), "{failed:?}");
}

// ---------------------------------------------------------------------------------------------------------------------
// Stationary slicing.
// ---------------------------------------------------------------------------------------------------------------------

fn synthetic(gradient: Array1<f64>, hessian: Array2<f64>) -> SoftLineMarginal {
    let d = gradient.len();
    SoftLineMarginal {
        value: 1.5,
        value_band: 1e-15,
        gradient,
        gradient_band: Array1::from_elem(d, 1e-14),
        hessian,
        hessian_band: Array2::from_elem((d, d), 1e-14),
        slice_evaluations: 0,
    }
}

#[test]
fn stationary_slicing_forms_the_schur_complement() {
    let marginal = synthetic(
        array![0.7, 0.0, 0.0],
        array![[2.0, 0.3, 0.1], [0.3, 4.0, 0.5], [0.1, 0.5, 3.0]],
    );
    let reduced = reduce_to_stationary_slicing(&marginal, 1).expect("the slicing is stationary and resolved");
    // c = (0.3, 0.1), M = [[4, 0.5], [0.5, 3]], det M = 11.75 and M⁻¹ = [[3, −0.5], [−0.5, 4]]/11.75, so
    // cM⁻¹cᵀ = (0.27 − 0.03 + 0.04)/11.75 = 0.28/11.75.
    let expected = 2.0 - 0.28 / 11.75;
    assert!(
        (reduced.hessian[[0, 0]] - expected).abs() <= reduced.hessian_band[[0, 0]],
        "Schur complement {} against {expected} (band {})",
        reduced.hessian[[0, 0]],
        reduced.hessian_band[[0, 0]]
    );
    assert!(reduced.hessian_band[[0, 0]] < 1e-12, "band {}", reduced.hessian_band[[0, 0]]);
    assert_eq!(reduced.gradient, array![0.7]);
    assert!(reduced.gradient_band[0] >= marginal.gradient_band[0]);
    assert_eq!(reduced.value, marginal.value);
}

#[test]
fn an_indefinite_resolved_slicing_curvature_is_accepted() {
    let marginal = synthetic(array![0.7, 0.0, 0.0], array![[2.0, 0.3, 0.1], [0.3, -4.0, 0.5], [0.1, 0.5, 3.0]]);
    let reduced = reduce_to_stationary_slicing(&marginal, 1).expect("a resolved indefinite T_vv is accepted");
    // M = [[−4, 0.5], [0.5, 3]], det M = −12.25 and M⁻¹ = [[3, −0.5], [−0.5, −4]]/(−12.25), so
    // cM⁻¹cᵀ = (0.27 − 0.03 − 0.04)/(−12.25) = −0.2/12.25.
    let expected = 2.0 + 0.2 / 12.25;
    assert!(
        (reduced.hessian[[0, 0]] - expected).abs() <= reduced.hessian_band[[0, 0]],
        "Schur complement {} against {expected}",
        reduced.hessian[[0, 0]]
    );
}

#[test]
fn a_slicing_that_is_not_stationary_is_refused() {
    let marginal = synthetic(array![0.7, 1e-3, 0.0], array![[2.0, 0.3, 0.1], [0.3, 4.0, 0.5], [0.1, 0.5, 3.0]]);
    let error = reduce_to_stationary_slicing(&marginal, 1).unwrap_err();
    assert!(matches!(error, SlicingError::NotStationary { coordinate: 1, .. }), "{error:?}");
}

#[test]
fn an_unresolved_slicing_curvature_is_refused() {
    let marginal = synthetic(array![0.7, 0.0, 0.0], array![[2.0, 0.3, 0.1], [0.3, 1e-15, 0.0], [0.1, 0.0, 3.0]]);
    let error = reduce_to_stationary_slicing(&marginal, 1).unwrap_err();
    match error {
        SlicingError::UnresolvedSlicingCurvature { eigenvalue, band, coupling } => {
            assert!(eigenvalue.abs() <= band, "eigenvalue {eigenvalue}, band {band}");
            // The flat direction is the first slicing coordinate, which couples to θ through 0.3.
            assert!((coupling - 0.3).abs() < 1e-12, "coupling {coupling}");
        }
        other => panic!("expected UnresolvedSlicingCurvature, got {other:?}"),
    }
}

#[test]
fn too_many_outer_coordinates_are_refused() {
    let marginal = synthetic(array![0.7], array![[2.0]]);
    assert!(matches!(
        reduce_to_stationary_slicing(&marginal, 2),
        Err(SlicingError::DimensionMismatch { outer_dimension: 2, dimension: 1 })
    ));
    let whole = reduce_to_stationary_slicing(&marginal, 1).unwrap();
    assert_eq!(whole.hessian, marginal.hessian);
}
