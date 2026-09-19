//! `Φ₂` with a relative a-priori error bound at plain-evaluation cost, through one route per quantity (#3158).
//!
//! # Tree
//!
//! With `n₁ = h − ρk` and `n₂ = k − ρh`, every orthant is a positive term or a difference led by `Φ` of a both-active
//! orthant, one whose constraints both have `nᵢ ≤ 0`:
//! - both `nᵢ ≤ 0`: `Φ₂(h, k; ρ)` itself;
//! - only `n₁ ≤ 0`: `Φ(h) − Φ₂(h, −k; −ρ)`, and symmetrically for `n₂`;
//! - neither: `P(−k ≤ Z ≤ h) + Φ₂(−h, −k; ρ)`, where `n₁ + n₂ = (1 − ρ)(h + k) > 0` makes the interval positive.
//!
//! A both-active orthant with `ρ > 0` splits as `Φ₂(u*, k; −ĉ) + Φ₂(−u*, h; −ĉ)`, with `ĉ = √((1 − ρ)/2)` and
//! `u* = (h − k)/(2ĉ)`. Both pieces are both-active again: `u* + ĉk = n₁/(2ĉ)`, `k + ĉu* = (h + k)/2`, and `n₁ + n₂ ≤ 0`
//! gives `h + k ≤ 0`. So every leaf is a both-active orthant with `ρ ≤ 0`, where nothing cancels.
//!
//! # Leaf
//!
//! For `ρ = −τ ≤ 0` and `c = 1 − ρ²`, integrate `Y` in closed form along the ray from the apex:
//! `Φ₂ = c·φ₂(h, k; ρ)·K`, with `K = ∫₀^∞ e^{−bw − w²/2} R(a + τw) dw`, `a = −n₁/√c`, `b = −n₂/√c` (ordered `a ≤ b`,
//! which leaves `K` unchanged) and `R` Mills' ratio. `R` is entire, and `|R(x + iy)| ≤ R(x)`, so the integrand is
//! bounded on a Bernstein ellipse by its real part times `e^{y²/2}`.
//! - **Convexity.** `(ln R)'' = V = 1 − λμ ∈ (0, 1)`, with `λ = 1/R` and `μ = λ − t`, the truncated-normal variance,
//!   which decreases. So for `w ≥ 0`, `ln g(w) − ln R(a) ≤ −βw − c₊w²/2`, with `β = b + τμ(a)` and `c₊ = 1 − τ²V(a)`,
//!   and for `w < 0` the same holds with `1 − τ²` in place of `c₊`. In particular `K ≥ R(a)R(β)`.
//! - **Tail.** `∫_T^∞ g ≤ R(a)e^{−βT − c₊T²/2}/(β + c₊T)`. `T` solves `βT + c₊T²/2 = L`, with the level
//!   `L = ln(2/u) − ln R(β)`, so the tail is `u/2` of the lower bound on `K`.
//! - **Rule.** On `[0, T]` the `n`-point Gauss-Legendre rule misses by at most `(32T/15)·M·e^{−2nS₀}/(e^{2S₀} − 1)`
//!   (Trefethen 2008, Thm 4.5), with `M` the bound of the integrand on the ellipse of log-radius `S₀`. The order is the
//!   smallest that makes it `u/2` of the lower bound on `K`. `S₀` minimizes `(1 + cosh s)/s`, the growth of the
//!   ellipse's quadratic exponent against its decay.
//!
//! Neither the length nor the order is a switch: both come from the leaf's own bounds, and the charged truncation is
//! the bound evaluated at the chosen `T` and `n`, not the target.
//!
//! # Rounding
//!
//! The bound is first order in `u`, the core's model: the cited `libm::exp` contract (libm 0.2.16,
//! `src/math/exp.rs:58-60`, pinned by `the_cited_libm_is_the_locked_one`), [`NORMAL_SCALED_TAIL_RELATIVE_ERROR`] and
//! [`NORMAL_CDF_RELATIVE_ERROR`] for `R` and `Φ`, and the certified rule's node and weight errors. A count of `k`
//! rounded operations is carried as Wilkinson's `γ_k`. Only a straddling interval `P(lo ≤ Z ≤ hi)` with `lo < 0 < hi`
//! calls `erf`, charged the one ulp the core charges its `erfc`, a measurement of the platform library.
//!
//! A leaf's bound, and a sum's, scales with its value. A difference's bound scales with its minuend `Φ(h)`. Its
//! subtrahend is `P(X ≤ h, Y > k)` with `n₂ = k − ρh > 0`.
//! - For `ρ > 0` every `x ≤ h` has its conditional mean `ρx ≤ ρh < k`, so the subtrahend is below `Φ(h)/2` and the
//!   minuend is below twice the result.
//! - For `ρ < 0` the conditional mean `ρx` grows as `x` falls. With `h = −H`, the branch is `|ρ|H < k ≤ H/|ρ|`, and as
//!   `ρ → −1` with `k` near `H` the result `≈ P(−k ≤ X ≤ −H)` is a small part of `Φ(h)`. The bound then carries the
//!   ratio `Φ(h)/Φ₂`: it stays a valid bound, not a relative one.

use super::{BoundedProbability, Correlation, density};
use crate::double_double::SMALLEST_SUBNORMAL;
use crate::probability::{
    NORMAL_CDF_RELATIVE_ERROR, NORMAL_CDF_UNDERFLOW_FLOOR, NORMAL_SCALED_TAIL_RELATIVE_ERROR, normal_cdf_and_pdf,
    normal_pdf_bounded, normal_scaled_tail,
};
use crate::roundoff::{UNIT_ROUNDOFF, accumulation_growth as growth, inflated};
use crate::special::{CertifiedGaussLegendreRule, gauss_legendre_certified};
use std::f64::consts::{FRAC_1_SQRT_2, LN_2};
use std::sync::{Arc, RwLock};

/// The ellipse log-radius `S₀`, the root of `s·tanh(s/2) = 1`, where `(1 + cosh s)/s` is least. The truncation's
/// exponent grows as `T²(1 + cosh s)²/16` against the decay `2ns`, so `S₀` needs the fewest nodes at large `T`.
const LOG_RADIUS: f64 = 1.543_404_638_418_208_5;

/// `√(2π)`, the double nearest.
const SQRT_2PI: f64 = 2.506_628_274_631_000_2;

/// `√(2/π) = μ(0)`, the mean excess at zero, the double nearest.
const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;

/// The certified rules computed so far, indexed by order. A rule is computed once per order and shared.
static CERTIFIED_RULES: RwLock<Vec<Option<Arc<CertifiedGaussLegendreRule>>>> = RwLock::new(Vec::new());

pub(super) fn rule_of_order(order: usize) -> Arc<CertifiedGaussLegendreRule> {
    let cached = CERTIFIED_RULES
        .read()
        .ok()
        .and_then(|rules| rules.get(order).cloned().flatten());
    if let Some(rule) = cached {
        return rule;
    }
    let rule = Arc::new(gauss_legendre_certified(order));
    if let Ok(mut rules) = CERTIFIED_RULES.write() {
        if rules.len() <= order {
            rules.resize(order + 1, None);
        }
        rules[order] = Some(Arc::clone(&rule));
    }
    rule
}

/// Mills' ratio `R(t) = Φ(−t)/φ(t)` and a bound on its relative error.
/// - `t ≥ 0`: `√(2π)·Q(t)` from the table, within [`NORMAL_SCALED_TAIL_RELATIVE_ERROR`], plus the constant and the product.
/// - `t < 0` (a rounding-level negative of a both-active leaf): `Φ(−t) ≥ ½` over `φ(t)`, each within its own bound.
fn mills_ratio(t: f64) -> (f64, f64) {
    if t >= 0.0 {
        return (
            SQRT_2PI * normal_scaled_tail(t),
            NORMAL_SCALED_TAIL_RELATIVE_ERROR + growth(3),
        );
    }
    let upper = normal_cdf_and_pdf(-t).0;
    let (weight, weight_error) = normal_pdf_bounded(t);
    let relative = NORMAL_CDF_RELATIVE_ERROR + 2.0 * NORMAL_CDF_UNDERFLOW_FLOOR + weight_error / weight + growth(2);
    (upper / weight, relative)
}

/// An upper bound on the mean excess `μ(t) = λ(t) − t`, which decreases from `√(2/π)` at zero and is below `1/t`.
fn mean_excess_bound(t: f64) -> f64 {
    if t > 0.0 {
        SQRT_2_OVER_PI.min(t.recip())
    } else {
        SQRT_2_OVER_PI - t
    }
}

/// The length `T` with `βT + c₊T²/2 = L`, raised to `L − ln(β + c₊T)` when the tail's rate is below one.
fn tail_length(slope: f64, curvature: f64, level: f64) -> f64 {
    let root = |target: f64| 2.0 * target / (slope + (slope * slope + 2.0 * curvature * target).sqrt());
    let log_rate = 0.5 * (slope * slope + 2.0 * curvature * level).ln();
    if log_rate >= 0.0 {
        root(level)
    } else {
        root(level - log_rate)
    }
}

/// The largest value of `q₀u² + q₁u + q₂` on `[lo, hi]`, with `q₀ < 0`.
fn quadratic_max(q: [f64; 3], lo: f64, hi: f64) -> f64 {
    let at = |u: f64| (q[0] * u + q[1]) * u + q[2];
    let vertex = -q[1] / (2.0 * q[0]);
    let ends = at(lo).max(at(hi));
    if lo < vertex && vertex < hi {
        ends.max(at(vertex))
    } else {
        ends
    }
}

/// An upper bound on `ln(|g|/R(a))` over the Bernstein ellipse of `[0, T]` at log-radius [`LOG_RADIUS`].
///
/// A point `w = C + Au + iB√(1 − u²)`, with `C = T/2`, `A = C cosh S₀` and `B = C sinh S₀`, has
/// `ln|g| − ln R(a) ≤ −β·Re w − c·(Re w)²/2 + (Im w)²/2`: a quadratic in `u`. `Re w < 0` for `u < −C/A`, where the
/// larger slope and the curvature `1 − τ²` bound it, and the smaller slope and `c₊` bound the rest. The terms are at
/// most `mag` in size, so their rounding is within `γ₁₆·mag`.
fn ellipse_log_bound(length: f64, slopes: (f64, f64), curvatures: (f64, f64)) -> f64 {
    let (slope_lo, slope_hi) = slopes;
    let (curvature_left, curvature_right) = curvatures;
    let centre = 0.5 * length;
    let major = centre * LOG_RADIUS.cosh();
    let minor = centre * LOG_RADIUS.sinh();
    let quadratic = |slope: f64, curvature: f64| {
        [
            -0.5 * (curvature * major * major + minor * minor),
            -major * (slope + curvature * centre),
            0.5 * minor * minor - centre * (slope + 0.5 * curvature * centre),
        ]
    };
    let split = -centre / major;
    let left = quadratic_max(quadratic(slope_hi, curvature_left), -1.0, split);
    let right = quadratic_max(quadratic(slope_lo, curvature_right), split, 1.0);
    let magnitude =
        minor * minor + (major + centre) * (slope_lo.abs().max(slope_hi.abs()) + centre) + major * major;
    left.max(right) + growth(16) * magnitude
}

/// Absolute errors of a leaf's inputs against the orthant it stands for: `h` and `τ` absolute, the complement relative
/// to `1 − τ²` of the computed `τ`. A caller's leaf has none. A split leaf's `u*`, `ĉ` and `(1 + ĉ)(1 − ĉ)` round.
#[derive(Clone, Copy, Default)]
struct Perturbation {
    h: f64,
    tau: f64,
    complement: f64,
}

/// A both-active orthant with `ρ ≤ 0`: `c·φ₂·K` (module docs), and its bound.
///
/// The relative error of `K` sums the tail and rule charges, the rule's weights and sum, each node's value and the
/// rounding of `a` and `b`, which move `ln K` by at most `μ(a)` and `μ(b)` per unit. The perturbations move `ln Φ₂` by
/// `|n₁|/c + (μ(a) + τμ(b))/√c` per unit of `h`, by `1 + E + (|a|μ(a) + |b|μ(b))/2` per relative unit of `c`, with
/// `E = n₁²/(2c) + k²/2` the density's exponent, and `Φ₂` by `φ₂` per unit of `τ` (Plackett).
fn leaf(h: f64, k: f64, correlation: Correlation, perturbation: Perturbation) -> BoundedProbability {
    let (density_value, density_error) = density(h, k, correlation);
    if density_value == 0.0 {
        // `K ≤ R(a)R(b) ≤ π/2` and `c ≤ 1`, so `Φ₂ = cφ₂K` is within twice the density's bound.
        return BoundedProbability {
            value: 0.0,
            rounding: inflated(2.0 * density_error * (1.0 + perturbation.tau), 8),
        };
    }
    let tau = -correlation.rho;
    let complement = correlation.complement;
    let root = complement.sqrt();
    // Fused, `n₁ = h + τk` rounds once, on the result, so its rounding stays relative as `h` and `τk` cancel near
    // `ρ → −1`. With `√c` and the quotient, `a` and `b` err by `γ₃` of themselves.
    let (first, second) = (tau.mul_add(k, h), tau.mul_add(h, k));
    let exponent = 0.5 * (first * first / complement + k * k);
    let (mut a, mut b) = (-first / root, -second / root);
    let mut a_error = growth(3) * a.abs();
    let mut b_error = growth(3) * b.abs();
    let swapped = a > b;
    if swapped {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut a_error, &mut b_error);
    }

    let (ratio, ratio_error) = mills_ratio(a);
    let inverse = ratio.recip();
    let inverse_error = ratio_error + UNIT_ROUNDOFF;
    let excess = inverse - a;
    let excess_error = (inverse_error + growth(1)) * inverse + growth(1) * a.abs();
    let slope = b + tau * excess;
    let slope_error = growth(4) * slope.abs() + inflated(tau * excess_error, 2);
    let (slope_lo, slope_hi) = (slope - slope_error, slope + slope_error);
    let variance = 1.0 + a * inverse - inverse * inverse;
    let variance_hi =
        variance + 3.0 * (inverse_error + growth(3)) * (1.0 + a.abs() * inverse + inverse * inverse);
    // `V < 1` makes `1 − τ²` a lower bound on `c₊` too, so the larger of the two bounds it.
    let curvature_left = (1.0 - tau) * (1.0 + tau) * (1.0 - growth(4));
    let curvature_right = ((1.0 - tau * tau * variance_hi) - growth(4)).max(curvature_left);

    // `K ≥ R(a)R(β_hi)`, and each truncation charge is relative to it.
    let (floor, floor_error) = mills_ratio(slope_hi);
    let floor_lo = floor * (1.0 - floor_error);
    let level = 54.0 * LN_2 - floor.ln() + floor_error;
    let length = tail_length(slope_lo, curvature_right, level);
    let tail_exponent = slope_lo * length + 0.5 * curvature_right * length * length;
    let tail_scale = slope_lo.abs() * length + 0.5 * curvature_right * length * length;
    let tail_rate = slope_lo + curvature_right * length;
    let tail_relative = inflated(
        libm::exp(-tail_exponent + growth(4) * tail_scale) / (tail_rate * (1.0 - growth(2))) / floor_lo,
        6,
    );

    let log_max = ellipse_log_bound(length, (slope_lo, slope_hi), (curvature_left, curvature_right));
    let decay = libm::expm1(2.0 * LOG_RADIUS);
    let span = 32.0 * length / 15.0;
    let order = ((log_max + span.ln() - decay.ln() + level) / (2.0 * LOG_RADIUS))
        .ceil()
        .max(1.0) as usize;
    let quadrature_relative = inflated(
        libm::exp(log_max - 2.0 * order as f64 * LOG_RADIUS) * span / (decay * (1.0 - growth(3))) / floor_lo,
        8,
    );

    let rule = rule_of_order(order);
    let centre = 0.5 * length;
    let (mu_a, mu_b) = (mean_excess_bound(a), mean_excess_bound(b));
    let (mut sum, mut sum_error) = (0.0, 0.0);
    for (&node, &weight) in rule.nodes.iter().zip(&rule.weights) {
        let w = centre * (1.0 + node);
        let damping = libm::exp(-(w * (b + 0.5 * w)));
        let (ratio, ratio_error) = mills_ratio(a + tau * w);
        let term = weight * (damping * ratio);
        // The node's position, the exponent, `exp`, the ratio's argument and value, and the product.
        let relative = (b.abs() + tau * mu_a + w) * (centre * rule.node_error + growth(2) * w)
            + 2.0 * UNIT_ROUNDOFF
            + growth(2) * w * (b.abs() + 0.5 * w)
            + mu_a * growth(2) * (a.abs() + tau * w)
            + ratio_error
            + UNIT_ROUNDOFF;
        sum += term;
        sum_error += term * relative;
    }
    let integral = centre * sum;

    let (mu_first, mu_second) = if swapped { (mu_b, mu_a) } else { (mu_a, mu_b) };
    let h_gain = first.abs() / complement + (mu_first + tau * mu_second) / root;
    let complement_gain = 1.0 + exponent + 0.5 * (mu_a * a.abs() + mu_b * b.abs());
    let relative = tail_relative
        + quadrature_relative
        + rule.weight_relative_error
        + growth(order + 1)
        + sum_error / sum
        + growth(1)
        + mu_a * a_error
        + mu_b * b_error
        + complement_gain * perturbation.complement
        + h_gain * perturbation.h
        + growth(2);
    let value = complement * density_value * integral;
    let rounding = value * relative
        + complement * integral * density_error
        + (density_value + density_error) * perturbation.tau
        + SMALLEST_SUBNORMAL * (1.0 + integral);
    BoundedProbability {
        value,
        rounding: inflated(rounding, 8),
    }
}

/// A both-active orthant: a leaf when `ρ ≤ 0`, the two split leaves otherwise.
/// - `ĉ = √½·√(1 − ρ)` errs by at most `γ₃` of itself: the factor, both products and `√`.
/// - `u* = (h − k)/(2ĉ)` then errs by `γ₅|u*|`, and the piece's `(1 + ĉ)(1 − ĉ)` by `γ₃` of `1 − ĉ²`.
fn both_active(h: f64, k: f64, correlation: Correlation) -> BoundedProbability {
    if correlation.rho <= 0.0 {
        return leaf(h, k, correlation, Perturbation::default());
    }
    let half_gap = FRAC_1_SQRT_2 * correlation.one_minus.sqrt();
    let split = (h - k) / (2.0 * half_gap);
    let piece = Correlation::from_rho(-half_gap);
    let perturbation = Perturbation {
        h: growth(5) * split.abs(),
        tau: growth(3) * half_gap,
        complement: growth(3),
    };
    sum(leaf(split, k, piece, perturbation), leaf(-split, h, piece, perturbation))
}

/// `Φ(x)` within [`NORMAL_CDF_RELATIVE_ERROR`] of itself, plus [`NORMAL_CDF_UNDERFLOW_FLOOR`].
fn normal_cdf_bounded(x: f64) -> BoundedProbability {
    let value = normal_cdf_and_pdf(x).0;
    BoundedProbability {
        value,
        rounding: NORMAL_CDF_RELATIVE_ERROR * value + NORMAL_CDF_UNDERFLOW_FLOOR,
    }
}

fn sum(left: BoundedProbability, right: BoundedProbability) -> BoundedProbability {
    let value = left.value + right.value;
    BoundedProbability {
        value,
        rounding: left.rounding + right.rounding + UNIT_ROUNDOFF * value.abs(),
    }
}

fn difference(left: BoundedProbability, right: BoundedProbability) -> BoundedProbability {
    sum(
        left,
        BoundedProbability {
            value: -right.value,
            rounding: right.rounding,
        },
    )
}

/// `Φ(hi) − Φ(lo)`, signed, as a difference of same-side tails, or of `erf` halves when the interval straddles zero.
/// The halves have opposite signs, so nothing cancels. Each `½erf(x/√2)` errs by one ulp, `2u` of itself, and by
/// `φ(x)` per unit of its argument's `γ₂|x|` rounding.
fn signed_interval(lo: f64, hi: f64) -> BoundedProbability {
    if lo > hi {
        let swapped = signed_interval(hi, lo);
        return BoundedProbability {
            value: -swapped.value,
            rounding: swapped.rounding,
        };
    }
    if lo >= 0.0 {
        return difference(normal_cdf_bounded(-lo), normal_cdf_bounded(-hi));
    }
    if hi <= 0.0 {
        return difference(normal_cdf_bounded(hi), normal_cdf_bounded(lo));
    }
    let upper = 0.5 * libm::erf(hi * FRAC_1_SQRT_2);
    let lower = 0.5 * libm::erf(lo * FRAC_1_SQRT_2);
    let value = upper - lower;
    let argument = growth(2) * (lo.abs() * normal_cdf_and_pdf(lo).1 + hi.abs() * normal_cdf_and_pdf(hi).1);
    BoundedProbability {
        value,
        rounding: inflated(
            2.0 * UNIT_ROUNDOFF * (upper.abs() + lower.abs()) + argument + UNIT_ROUNDOFF * value.abs(),
            4,
        ),
    }
}

/// `Φ₂(h, k; ρ)` through the tree of the module docs, with its bound at the computed arguments, projected onto
/// `[0, 1]`. Infinite bounds and `ρ ∈ {−1, 0, 1}` are exact special cases.
pub(super) fn orthant(h: f64, k: f64, correlation: Correlation) -> BoundedProbability {
    let exact_zero = BoundedProbability {
        value: 0.0,
        rounding: 0.0,
    };
    let bounded = if h == f64::NEG_INFINITY || k == f64::NEG_INFINITY {
        exact_zero
    } else if h == f64::INFINITY {
        normal_cdf_bounded(k)
    } else if k == f64::INFINITY {
        normal_cdf_bounded(h)
    } else if correlation.one_minus == 0.0 {
        normal_cdf_bounded(h.min(k))
    } else if correlation.one_plus == 0.0 {
        if -k < h { signed_interval(-k, h) } else { exact_zero }
    } else if correlation.rho == 0.0 {
        let (first, second) = (normal_cdf_bounded(h), normal_cdf_bounded(k));
        let value = first.value * second.value;
        BoundedProbability {
            value,
            rounding: first.rounding * second.value + second.rounding * first.value + UNIT_ROUNDOFF * value,
        }
    } else {
        let rho = correlation.rho;
        let (first, second) = (h - rho * k, k - rho * h);
        match (first <= 0.0, second <= 0.0) {
            (true, true) => both_active(h, k, correlation),
            (true, false) => difference(normal_cdf_bounded(h), both_active(h, -k, correlation.negated())),
            (false, true) => difference(normal_cdf_bounded(k), both_active(-h, k, correlation.negated())),
            (false, false) => sum(signed_interval(-k, h), both_active(-h, -k, correlation)),
        }
    };
    BoundedProbability {
        value: bounded.value.clamp(0.0, 1.0),
        rounding: bounded.rounding,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bivariate_normal::{
        BIVARIATE_NORMAL_CDF_ERROR_BOUND, bivariate_normal_cdf_with_complement, bivariate_normal_interval_probability,
    };

    /// `(h, k, ρ, Φ₂)` from #3158's table: rare-event binomial cells whose `Φ₂` the core's absolute bound leaves with no
    /// digit. Truths by 60-digit piecewise Gauss-Legendre of `∫_{−∞}^h φ(x)Φ((k − ρx)/√(1 − ρ²)) dx` over 400 panels.
    const ISSUE_TABLE: &[(f64, f64, f64, f64)] = &[
        (-11.464024688443615, 0.5, -0.2873478855663454, 1.6302187740911292421e-33),
        (-11.464024688443615, 0.5, 0.2873478855663454, 9.9996637035816518385e-31),
        (-21.273453560965326, 0.5, -0.2873478855663454, 2.1332696457695513455e-109),
        (-21.273453560965326, 0.5, 0.2873478855663454, 9.999999999976710228e-101),
        (-30.20559417957964, 0.5, -0.2873478855663454, 6.1985068716753879522e-218),
        (-30.20559417957964, 0.5, 0.2873478855663454, 1.0000000000000492453e-200),
        (-37.0470962993612, 0.5, 0.2873478855663454, 9.9999999999995236547e-301),
        (-7.034483825301132, 0.5, -0.7071067811865475, 6.547499082969384869e-23),
        (-11.464024688443615, 0.5, -0.7071067811865475, 1.4173458841174327982e-57),
        (-21.273453560965326, 0.5, -0.7071067811865475, 1.3969090574103750934e-194),
    ];

    /// The bound covers the truth and is within `10⁻¹²` of the value: the density's `γ₈(1 + E)` at `E ≈ 226` is
    /// `2·10⁻¹³` of it, and `E ≤ 745` wherever the density is normal.
    fn covers(route: &str, result: BoundedProbability, truth: f64) {
        let error = (result.value - truth).abs();
        assert!(
            error <= result.rounding && result.rounding <= 1.0e-12 * result.value,
            "{route}: {:e} ± {:e} against {truth:e} (error {error:e})",
            result.value,
            result.rounding
        );
    }

    #[test]
    fn the_issue_table_keeps_its_relative_digits_through_both_entries() {
        for &(h, k, rho, truth) in ISSUE_TABLE {
            let complement = (1.0 - rho) * (1.0 + rho);
            covers(&format!("orthant ({h}, {k}, {rho})"), orthant(h, k, Correlation::from_complement(rho, complement)), truth);
            // `Φ₂(h, k; ρ) = P(X ≤ h, −k ≤ Y')` with `Y' = −Y` at correlation `−ρ`.
            covers(
                &format!("lower interval ({h}, {k}, {rho})"),
                bivariate_normal_interval_probability(h, f64::NEG_INFINITY, k, rho).unwrap(),
                truth,
            );
            covers(
                &format!("upper interval ({h}, {k}, {rho})"),
                bivariate_normal_interval_probability(h, -k, f64::INFINITY, -rho).unwrap(),
                truth,
            );
        }
    }

    #[test]
    fn the_log_radius_minimizes_the_ellipse_growth() {
        // `d/ds (1 + cosh s)/s = 0` is `s sinh s = 1 + cosh s`, that is `s tanh(s/2) = 1`.
        let residual = LOG_RADIUS * (0.5 * LOG_RADIUS).tanh() - 1.0;
        assert!(residual.abs() <= 4.0 * f64::EPSILON, "residual {residual:e}");
        let growth = |s: f64| (1.0 + s.cosh()) / s;
        assert!(growth(LOG_RADIUS) < growth(LOG_RADIUS * (1.0 + 1.0e-6)));
        assert!(growth(LOG_RADIUS) < growth(LOG_RADIUS * (1.0 - 1.0e-6)));
    }

    /// Every branch of the tree against the core: both constraints active, either one, neither, `ρ` of either sign,
    /// `ρ → ±1` and the exact special cases.
    #[test]
    fn every_branch_meets_the_core_within_both_bounds() {
        let arguments = [-9.0, -4.0, -1.5, -0.3, 0.0, 0.3, 1.5, 4.0, 9.0];
        let correlations = [-0.999_999, -0.95, -0.6, -0.2, 0.0, 0.2, 0.6, 0.95, 0.999_999, -1.0, 1.0];
        for &h in &arguments {
            for &k in &arguments {
                for &rho in &correlations {
                    let complement = (1.0 - rho) * (1.0 + rho);
                    let apex = orthant(h, k, Correlation::from_complement(rho, complement));
                    let core = bivariate_normal_cdf_with_complement(h, k, rho, complement).unwrap();
                    assert!(
                        (apex.value - core).abs() <= apex.rounding + BIVARIATE_NORMAL_CDF_ERROR_BOUND,
                        "({h}, {k}, {rho}): apex {:e} ± {:e}, core {core:e}",
                        apex.value,
                        apex.rounding
                    );
                    // The bound scales with the value, except in a difference at `ρ < 0`, where it scales with the
                    // minuend `Φ(min(h, k))` (module docs). A vanished value keeps a subnormal bound.
                    let scale = if rho < 0.0 {
                        apex.value.max(normal_cdf_and_pdf(h.min(k)).0)
                    } else {
                        apex.value
                    };
                    assert!(
                        apex.rounding <= 1.0e-12 * scale + f64::MIN_POSITIVE,
                        "({h}, {k}, {rho}): {:e} ± {:e}",
                        apex.value,
                        apex.rounding
                    );
                }
            }
        }
    }

    #[test]
    fn extreme_arguments_stay_finite_and_in_the_unit_interval() {
        let arguments = [
            f64::NEG_INFINITY,
            -f64::MAX,
            -1.0e300,
            -1.0e10,
            -40.0,
            -1.0e-300,
            0.0,
            1.0e-300,
            40.0,
            1.0e10,
            1.0e300,
            f64::MAX,
            f64::INFINITY,
        ];
        let correlations = [-1.0, -(1.0 - f64::EPSILON), -0.5, -1.0e-300, 1.0e-300, 0.5, 1.0 - f64::EPSILON, 1.0];
        for &h in &arguments {
            for &k in &arguments {
                for &rho in &correlations {
                    let complement = (1.0 - rho) * (1.0 + rho);
                    let result = orthant(h, k, Correlation::from_complement(rho, complement));
                    assert!(
                        (0.0..=1.0).contains(&result.value) && result.rounding.is_finite() && result.rounding >= 0.0,
                        "({h}, {k}, {rho}): {:e} ± {:e}",
                        result.value,
                        result.rounding
                    );
                }
            }
        }
    }
}
