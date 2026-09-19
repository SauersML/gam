//! The positive form of `Φ₂` in the negative-correlation lower tail, with a certificate evaluated per call (#2946;
//! derivation in comments 5721025036, 5721676253, 5721704551 and 5725339297 there).
//!
//! # Identity
//!
//! Shift the origin to the apex, `x = h − s` and `y = k − t`, and integrate the radial direction in closed form:
//!
//! `Φ₂(h, k; ρ) = φ₂(h, k; ρ)·I`, with `I = ∫₀^{π/2} g(θ) dθ` and `g = N(A/√B)/B`, where
//! - `A = α₁cos θ + α₂sin θ`, with `α₁ = −(h − ρk)/c` and `α₂ = −(k − ρh)/c`;
//! - `B = (1 + τ sin 2θ)/c`, with `τ = −ρ` and `c = 1 − ρ²`;
//! - `N(z) = ∫₀^∞ w e^{−zw − w²/2} dw = 1 − z·R(z)`, with `R` Mills' ratio.
//!
//! When `ρ ≤ 0` and `α₁, α₂ ≥ 0` every factor is positive. So nothing cancels, and the value keeps its relative digits
//! however small it is. That is the certified region. The core's `Φ(h)Φ(k) + T` cancels there instead.
//!
//! # Map
//!
//! Order `α₁ ≤ α₂`: `g(π/2 − θ)` with the two swapped is `g(θ)`, so the swap leaves `I` unchanged. Then
//! `φ₀ = atan(α₁/α₂) ∈ [0, π/4]`. The near-double pole of `g` at `θ = −φ₀` is resolved by the map `θ = δ·expm1(v)`
//! for the float `δ = fl(φ_hi + 1/(σ|α|))`, with `σ = √c`, on `v ∈ [0, L]`. Here `L = log1p(π/(2δ))` is the exact real
//! length, so the rule integrates exactly `[0, π/2]`, and `I = ∫₀^L G(v) dv` with `G = (θ + δ)·g(θ(v))`.
//!
//! # Truncation
//!
//! If `G` is analytic in the Bernstein ellipse `E_r` of `[0, L]` with `|G| ≤ M` there, the `n`-point Gauss-Legendre
//! rule misses `I` by at most `T = (64/15)·(L/2)·M·r^{−2n}/(r² − 1)` (Trefethen 2008, Thm 4.5).
//! [`PositiveForm::ellipse_bound`] bounds `M` by region.
//! - Its lever is the radial form: for `θ = p + iq` with `Re B > 0`,
//!   `|g(θ)| ≤ ∫₀^∞ w e^{−Re(A)w − Re(B)w²/2} dw = N(Re A/√Re B)/Re B`,
//!   with `Re A = cosh q·A(p)` and `Re B = (1 + τ sin 2p cosh 2q)/c`.
//! - Where `Re A ≥ 0`, `N(x) ≤ 1/(x² + 1)` (from Birnbaum's bound on `R`) gives `|g| ≤ 1/((Re A)² + Re B)`.
//! - Where `Re A/√Re B ≥ −X`, `N` decreasing gives `|g| ≤ N(−X)/Re B`.
//!
//! # Arithmetic
//!
//! Every quantity is an exact-real enclosure, a [`ClosedInterval`] with outward-rounded operations. So the published
//! bound needs no count of rounded operations.
//! - `libm::exp` is called explicitly and widened by two ulps on each side of fdlibm's one-ulp error analysis.
//! - No other libm function enters. Sines, cosines, `expm1` and `atan` are enclosed by their series, and `log1p` by
//!   [`certified_ln_1p`], which uses IEEE operations only.
//! - `N` comes from [`normal_left_tail_ratios`], whose bound covers its argument's interval.
//! - The rule's node and weight errors come from [`gauss_legendre_certified`].
//!
//! The result encloses `Φ₂`. The published value is the enclosure's midpoint, and the rounding its larger distance to
//! an end.

use super::{BoundedProbability, RoundingContract};
use crate::probability::normal_left_tail_ratios;
use crate::roundoff::UNIT_ROUNDOFF;
use crate::score_opt::{ClosedInterval, certified_ln_1p};
use crate::special::{CertifiedGaussLegendreRule, gauss_legendre_certified};
use std::f64::consts::PI;
use std::sync::{Arc, RwLock};

/// The truncation target relative to the lower bound on `I`: `2ε`, the rounding scale the core order targets.
const TRUNCATION_TARGET: f64 = 2.0 * f64::EPSILON;

/// A log-radius admissible for every law in the certified region, so every call can certify at it.
/// - `φ₂` normal gives `E ≤ 708.4 − ln 2π + ½ln(1/c) ≤ 1079`. Then `E = ½αᵀΣα ≥ ½(1 − τ)|α|²` gives
///   `σ|α| ≤ √(2E(1 + τ)) < 66`, so `δ > 1/66`, `L ≤ log1p(33π) < 4.66` and `H < 2.33`.
/// - Also `δH ≤ π/4`, since `log1p(x) ≤ x`, and `(π/2 + δ)H ≤ (π/4)(1 + x)ln(1 + x)/x ≤ 3.7` with `x = π/(2δ)`.
/// - At `ln r = 0.2`: `y_m < 0.47`, `y_c < 0.093` and `e < 0.047`. The middle's `|p| < 0.038` and `q < 0.18` give
///   `β₁ > 0.92`. The left cap's `|p| < 0.018` and `q < 0.032` give `β_L > 0.96`. The right cap's `p_R < 0.078`,
///   `π/2 − p_lo < 0.007` and `q < 0.16` give `β_R > 0.83`, with `u ∈ (π/2 − 0.007, π/2 + 0.078 + π/4] ⊂ (0, π)`.
const ANCHOR_LOG_RADIUS: f64 = 0.2;

/// A series stops once its next term, or its tail, is at most `u²` (of the sum, for positive series). The enclosure
/// already carries at least `u` of its value, so a smaller term is not resolved. The term or tail is added to the
/// enclosure either way.
const SERIES_FLOOR: f64 = UNIT_ROUNDOFF * UNIT_ROUNDOFF;

/// The certified rules computed so far, indexed by order. A rule is computed once per order and shared.
static CERTIFIED_RULES: RwLock<Vec<Option<Arc<CertifiedGaussLegendreRule>>>> = RwLock::new(Vec::new());

fn rule_of_order(order: usize) -> Arc<CertifiedGaussLegendreRule> {
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

fn point(value: f64) -> ClosedInterval {
    ClosedInterval::point(value)
}

/// A float inside `interval`, halfway when both ends are finite.
fn midpoint(interval: ClosedInterval) -> f64 {
    (interval.lo + 0.5 * (interval.hi - interval.lo))
        .max(interval.lo)
        .min(interval.hi)
}

/// `numerator/denominator` for a denominator enclosure bounded away from zero, or `None`.
fn quotient(numerator: ClosedInterval, denominator: ClosedInterval) -> Option<ClosedInterval> {
    (denominator.lo > 0.0).then(|| numerator.div_positive(denominator))
}

/// `π`, between the neighbours of the nearest double.
fn pi() -> ClosedInterval {
    ClosedInterval::new(PI.next_down(), PI.next_up())
}

/// `π/2`, halving `π`'s enclosure exactly.
fn half_pi() -> ClosedInterval {
    ClosedInterval::new(0.5 * PI.next_down(), 0.5 * PI.next_up())
}

/// `exp` over an interval, from `libm::exp` at its ends. fdlibm's error analysis (libm 0.2.16 `exp.rs`) keeps each
/// result within one ulp. Two steps outward cover that on either side of a binade boundary, where the spacing halves.
fn exp_enclosure(argument: ClosedInterval) -> ClosedInterval {
    ClosedInterval::new(
        libm::exp(argument.lo).next_down().next_down().max(0.0),
        libm::exp(argument.hi).next_up().next_up(),
    )
}

/// `cosh q` from above.
fn cosh_upper(argument: f64) -> f64 {
    0.5 * exp_enclosure(point(argument))
        .add(exp_enclosure(point(-argument)))
        .hi
}

/// `e^v − 1` over an interval of `v ≥ 0`.
/// - While `e^v < 2`: its Taylor series at each end, whose terms are positive and increase with `v` (see
///   [`expm1_series`]).
/// - Otherwise `exp(v) − 1`. The subtraction costs at most the factor `e^v/(e^v − 1) ≤ 2` of the exponential's width.
fn expm1_enclosure(argument: ClosedInterval) -> ClosedInterval {
    if libm::exp(argument.hi) < 2.0 {
        ClosedInterval::new(
            expm1_series(argument.lo.max(0.0)).lo,
            expm1_series(argument.hi).hi,
        )
    } else {
        let shifted = exp_enclosure(argument).sub(point(1.0));
        ClosedInterval::new(shifted.lo.max(0.0), shifted.hi)
    }
}

/// `Σ_{j≥1} v^j/j!` at a point `0 ≤ v < 1`. After the terms through `t_j`, the ratios of the rest are at most
/// `v/(j + 2)`, so the tail is at most `t_{j+1}/(1 − v/(j + 2))`. The sum stops once that bound is at most `u²` of it,
/// or once the terms stop shrinking at the underflow floor, and the bound joins the upper end.
fn expm1_series(value: f64) -> ClosedInterval {
    if value == 0.0 {
        return point(0.0);
    }
    let argument = point(value);
    let mut term = argument;
    let mut sum = argument;
    let mut index = 1.0_f64;
    loop {
        let next = term.mul(argument).div_positive(point(index + 1.0));
        let ratio = argument.div_positive(point(index + 2.0)).hi;
        let tail = next.div_positive(point(1.0).sub(point(ratio))).hi;
        if tail <= SERIES_FLOOR * sum.lo || !(next.hi < term.hi) {
            return ClosedInterval::new(sum.lo, sum.add(point(tail)).hi);
        }
        sum = sum.add(next);
        term = next;
        index += 1.0;
    }
}

/// `(cos θ, sin θ)` over an interval of `θ ≥ 0` with `θ² < 6`, or `None` beyond. Both Taylor series alternate, and their
/// terms decrease from the second on because `θ² < 6`. So each remainder is at most its first omitted term (see
/// [`alternating_series`]).
fn cos_sin(angle: ClosedInterval) -> Option<(ClosedInterval, ClosedInterval)> {
    let angle = ClosedInterval::new(angle.lo.max(0.0), angle.hi);
    let square = angle.mul(angle);
    if !(square.hi < 6.0) {
        return None;
    }
    Some((
        alternating_series(point(1.0), square, 0.0),
        alternating_series(angle, square, 1.0),
    ))
}

/// `Σ_j (−1)^j t_j` for `t_0 = first` and `t_{j+1} = t_j·square/((m + 1)(m + 2))`, where `m = power + 2j` is the degree of
/// `t_j`. The sum stops once the next term's magnitude is at most `u²`, and widens by it.
fn alternating_series(first: ClosedInterval, square: ClosedInterval, power: f64) -> ClosedInterval {
    let mut term = first;
    let mut sum = first;
    let mut degree = power;
    let mut subtract = true;
    loop {
        let next = term
            .mul(square)
            .div_positive(point((degree + 1.0) * (degree + 2.0)));
        let magnitude = next.hi.abs().max(next.lo.abs());
        if magnitude <= SERIES_FLOOR {
            return sum.add(ClosedInterval::new(-magnitude, magnitude));
        }
        sum = if subtract { sum.sub(next) } else { sum.add(next) };
        term = next;
        degree += 2.0;
        subtract = !subtract;
    }
}

/// The partial sum `Σ_{j=0}^{5} (−1)^j t_j` of the series in [`alternating_series`], from below. Its last term is
/// subtracted, and the terms decrease from the second on, so it is below the series' value. The region bounds need
/// only these short partial sums: through `y^10/10!` the cosine's falls short by at most `(π/2)^12/12! < 5e-7`.
fn alternating_lower(first: ClosedInterval, square: ClosedInterval, power: f64) -> f64 {
    let mut term = first;
    let mut sum = first;
    let mut degree = power;
    for index in 1..=5 {
        term = term
            .mul(square)
            .div_positive(point((degree + 1.0) * (degree + 2.0)));
        sum = if index % 2 == 1 { sum.sub(term) } else { sum.add(term) };
        degree += 2.0;
    }
    sum.lo
}

/// A lower bound on `cos y` at a point `0 ≤ y ≤ π/2`.
fn cos_lower(angle: f64) -> f64 {
    alternating_lower(point(1.0), point(angle).mul(point(angle)), 0.0)
}

/// A lower bound on `sin u` at a point `0 ≤ u < π`: the partial sum up to `π/2`, and `cos(u − π/2)` above it.
fn sin_lower(angle: f64) -> f64 {
    if angle <= half_pi().lo {
        let argument = point(angle.max(0.0));
        return alternating_lower(argument, argument.mul(argument), 1.0);
    }
    let shifted = point(angle).sub(half_pi());
    // `cos` is even and decreasing on `[0, π/2]`, so it is at least its value at the largest `|w|`.
    cos_lower(shifted.hi.max(-shifted.lo))
}

/// `atan x` over an interval of `x ≥ 0`. It is increasing, so each end is enclosed at its own point.
fn atan_enclosure(argument: ClosedInterval) -> Option<ClosedInterval> {
    Some(ClosedInterval::new(
        atan_point(argument.lo.max(0.0))?.lo,
        atan_point(argument.hi)?.hi,
    ))
}

/// `atan x` at a point `x ≥ 0`: [`atan_small`] up to 1, and `π/2 − atan(1/x)` above it.
fn atan_point(value: f64) -> Option<ClosedInterval> {
    if value > 1.0 {
        let reciprocal = quotient(point(1.0), point(value))?;
        return Some(half_pi().sub(atan_small(reciprocal)?));
    }
    atan_small(point(value))
}

/// `atan x` over an interval inside `[0, 1]`. Three half-angle steps `x ← x/(1 + √(1 + x²))`, each increasing in `x`,
/// give `atan x = 8·atan x₃` with `x₃ ≤ tan(π/32)`. The series `Σ_j (−1)^j x₃^{2j+1}/(2j + 1)` then alternates with
/// decreasing terms, so its remainder is at most its first omitted term. It stops once that term is at most `u²` of
/// the sum, or once the terms stop shrinking at the underflow floor.
fn atan_small(argument: ClosedInterval) -> Option<ClosedInterval> {
    let halve = |value: ClosedInterval| quotient(value, point(1.0).add(point(1.0).add(value.mul(value)).sqrt()));
    let reduced = halve(halve(halve(argument)?)?)?;
    let square = reduced.mul(reduced);
    let mut power = reduced;
    let mut sum = reduced;
    let mut last = reduced.hi;
    let mut degree = 1.0_f64;
    let mut subtract = true;
    loop {
        power = power.mul(square);
        degree += 2.0;
        let next = power.div_positive(point(degree));
        if next.hi <= SERIES_FLOOR * sum.lo || !(next.hi < last) {
            return Some(sum.add(ClosedInterval::new(-next.hi, next.hi)).scale(8.0));
        }
        sum = if subtract { sum.sub(next) } else { sum.add(next) };
        last = next.hi;
        subtract = !subtract;
    }
}

/// `N(−X) = 1 + X·Φ(X)/φ(X) ≤ 1 + X·√(2π)·e^{X²/2}` for `X ≥ 0`, from above, since `Φ(X) ≤ 1`.
fn laplace_growth(reach: f64) -> f64 {
    if reach == 0.0 {
        return 1.0;
    }
    let argument = point(reach);
    let root_two_pi = pi().scale(2.0).sqrt();
    point(1.0)
        .add(
            argument
                .mul(root_two_pi)
                .mul(exp_enclosure(argument.mul(argument).scale(0.5))),
        )
        .hi
}

/// `N(z) = q/λ` at `x = −z`, over an interval of `z ≥ 0`, from [`normal_left_tail_ratios`] at the midpoint. Its bound
/// covers every argument within the radius passed.
fn mills_enclosure(argument: ClosedInterval) -> Option<ClosedInterval> {
    if !(argument.lo >= 0.0 && argument.hi.is_finite()) {
        return None;
    }
    let centre = midpoint(argument);
    let radius = point(argument.hi)
        .sub(point(centre))
        .hi
        .max(point(centre).sub(point(argument.lo)).hi);
    let ratios = normal_left_tail_ratios(-centre, radius).ok()?;
    let value = point(ratios.positive_part_over_density);
    let rounding = point(ratios.positive_part_over_density_rounding);
    Some(ClosedInterval::new(
        value.sub(rounding).lo.max(0.0),
        value.add(rounding).hi,
    ))
}

/// `φ₂ = exp(−E)/(2π√c)`, with `E = a/c + b/f` in non-negative terms: `a = (|h| − |k|)²/2` and `b = |hk|`. The factor
/// `f` is `1 + ρ = c/(1 + |ρ|)` when `h` and `k` share a sign, and `1 − ρ = 1 + |ρ|` otherwise, as the complement route
/// forms them.
fn density_enclosure(
    h: f64,
    k: f64,
    complement: f64,
    factors: (ClosedInterval, ClosedInterval),
    deviation: ClosedInterval,
) -> Option<ClosedInterval> {
    let (far, near) = factors;
    let gap = point(h.abs()).sub(point(k.abs()));
    let squared = gap.mul(gap).scale(0.5);
    let offset = ClosedInterval::new(squared.lo.max(0.0), squared.hi);
    let cross = point(h.abs()).mul(point(k.abs()));
    let opposite = h != 0.0 && k != 0.0 && (h < 0.0) != (k < 0.0);
    let factor = if opposite { far } else { near };
    let exponent = quotient(offset, point(complement))?.add(quotient(cross, factor)?);
    quotient(exp_enclosure(exponent.neg()), pi().scale(2.0).mul(deviation))
}

/// The law and map of one call inside the certified region.
struct PositiveForm {
    /// `τ = −ρ ≥ 0`.
    tau: f64,
    /// The caller's `c = 1 − ρ²`.
    complement: f64,
    /// `1/c`.
    inverse_complement: ClosedInterval,
    /// `(α₁, α₂)` with `α₁ ≤ α₂`.
    alpha: [ClosedInterval; 2],
    /// `|α|`.
    magnitude: ClosedInterval,
    /// `σ = √c`.
    deviation: ClosedInterval,
    /// `φ₀ = atan(α₁/α₂)`.
    angle: ClosedInterval,
    /// The map scale `δ`, a float.
    scale: f64,
    /// `κ_m = δ − φ₀`, the distance from the map's origin to the zero of `A`.
    offset: ClosedInterval,
    /// `L/2`.
    half_length: ClosedInterval,
    /// A lower bound on the chord slope `μ = sin(u_max)/u_max`, `u_max = π/2 + φ_hi`. `sin` is concave on `[0, π]`, so
    /// `sin u ≥ μu` on `[0, u_max]`.
    chord: f64,
    /// `φ₂(h, k; ρ)`.
    density: ClosedInterval,
}

impl PositiveForm {
    /// The law at `(h, k, ρ, c)` when it lies in the certified region, else `None`. The region needs finite `h` and `k`,
    /// `ρ ≤ 0`, `c > 0`, both `α` enclosures in `[0, ∞)` with `α₂ > 0`, a normal `φ₂`, and `κ_m > 0`.
    ///
    /// `1 + ρ` is `c/(1 + |ρ|)`, as the complement route forms it, so `h − ρk = (h + k) − k(1 + ρ)` keeps its digits as
    /// `ρ → −1`.
    fn new(h: f64, k: f64, rho: f64, complement: f64) -> Option<Self> {
        if !(h.is_finite() && k.is_finite() && rho <= 0.0 && complement > 0.0) {
            return None;
        }
        let tau = -rho;
        let complement_interval = point(complement);
        let far = point(1.0).add(point(tau));
        // Exact factors stay points, so an apex exactly on a constraint (`h = ρk`) keeps `α₁ = 0` resolved.
        let near = if far.lo == far.hi {
            ClosedInterval::quotient(complement, far.lo)
        } else {
            quotient(complement_interval, far)?
        };
        let scaled = |value: f64| {
            if near.lo == near.hi {
                ClosedInterval::product(value, near.lo)
            } else {
                point(value).mul(near)
            }
        };
        let total = point(h).add(point(k));
        let first = quotient(total.sub(scaled(k)).neg(), complement_interval)?;
        let second = quotient(total.sub(scaled(h)).neg(), complement_interval)?;
        if !(first.lo >= 0.0 && second.lo >= 0.0) {
            return None;
        }
        let alpha = if midpoint(first) <= midpoint(second) {
            [first, second]
        } else {
            [second, first]
        };
        if !(alpha[1].lo > 0.0) {
            return None;
        }
        let magnitude = alpha[0].mul(alpha[0]).add(alpha[1].mul(alpha[1])).sqrt();
        let deviation = complement_interval.sqrt();
        let density = density_enclosure(h, k, complement, (far, near), deviation)?;
        if !(density.lo >= f64::MIN_POSITIVE && density.hi.is_finite()) {
            return None;
        }
        let angle = atan_enclosure(quotient(alpha[0], alpha[1])?)?;
        let scale = angle.hi + 1.0 / (midpoint(deviation) * midpoint(magnitude));
        if !(scale.is_finite() && scale > 0.0) {
            return None;
        }
        let offset = point(scale).sub(angle);
        if !(offset.lo > 0.0) {
            return None;
        }
        let reach = quotient(half_pi(), point(scale))?;
        let length = ClosedInterval::new(
            certified_ln_1p(reach.lo)?.lo,
            certified_ln_1p(reach.hi)?.hi,
        );
        let top = half_pi().add(point(angle.hi)).hi;
        if !(top < pi().lo) {
            return None;
        }
        let chord = quotient(point(sin_lower(top)), point(top))?.lo;
        if !(chord > 0.0) {
            return None;
        }
        Some(Self {
            tau,
            complement,
            inverse_complement: quotient(point(1.0), complement_interval)?,
            alpha,
            magnitude,
            deviation,
            angle,
            scale,
            offset,
            half_length: length.scale(0.5),
            chord,
            density,
        })
    }

    /// An upper bound on `|G|` over the filled ellipse `E_r` of `[0, L]`, or `None` if `r` is inadmissible.
    ///
    /// With `a = (r + 1/r)/2` and `b = (r − 1/r)/2`, the ellipse reaches `e = H(a − 1)` past each end, its height is
    /// `y_m = H·b` over `[0, L]`, and it is at most `y_c = H·b²/a` high past the ends. All three are taken at the upper
    /// end of `H = L/2`. A point `v = x + iy` maps to `θ + δ = s·e^{iy}` with `s = δe^x`, so `p = s cos y − δ` and
    /// `q = s sin y`. Admissible needs `y_m < π/2` and positive cosine lower bounds. The regions:
    /// - the middle, `x ∈ [0, L]`, split at `p = 0` ([`Self::middle_upper`] and [`Self::flank_bound`]);
    /// - the left cap, `x ∈ [−e, 0)`, where `s ∈ [δe^{−e}, δ)`, so `|p| ≤ δ(1 − e^{−e}cos y_c)` and `|q| ≤ δ·y_c`;
    /// - the right cap, `x ∈ (L, L + e]` ([`Self::right_cap_bound`]).
    fn ellipse_bound(&self, radius: f64) -> Option<f64> {
        let r = point(radius);
        let inverse = quotient(point(1.0), r)?;
        let major = r.add(inverse).scale(0.5);
        let minor = r.sub(inverse).scale(0.5);
        let step = r.sub(point(1.0));
        let excess = quotient(step.mul(step), r.scale(2.0))?;
        let half = point(self.half_length.hi);
        let extension = half.mul(excess).hi;
        let height = half.mul(minor).hi;
        let cap_height = quotient(half.mul(minor).mul(minor), major)?.hi;
        if !(height < half_pi().lo) {
            return None;
        }
        let cos_height = cos_lower(height);
        let cos_cap = cos_lower(cap_height);
        if !(cos_height > 0.0 && cos_cap > 0.0) {
            return None;
        }
        let scale = point(self.scale);
        let middle = self.middle_upper(cos_height)?;
        // Below the axis over `[0, L]`: `s ≥ δ` and `s cos y < δ`, so `|p| ≤ δ(1 − cos y_m)` and `q < δ tan y_m`, with
        // `tan y ≤ y/cos y`.
        let dip = scale.mul(point(1.0).sub(point(cos_height))).hi;
        let lift = quotient(scale.mul(point(height)), point(cos_height))?.hi;
        let below = self.flank_bound(dip, lift, cos_height)?;
        let shrink = exp_enclosure(point(-extension)).mul(point(cos_cap)).lo;
        let left_dip = scale.mul(point(1.0).sub(point(shrink))).hi;
        let left_lift = scale.mul(point(cap_height)).hi;
        let left = self.flank_bound(left_dip, left_lift, 1.0)?;
        let right = self.right_cap_bound(extension, cap_height, cos_cap)?;
        let bound = middle.max(below).max(left).max(right);
        bound.is_finite().then_some(bound)
    }

    /// The middle above the axis, `x ∈ [0, L]` with `p ≥ 0`. There `p ≤ δe^L − δ = π/2`, so `Re B ≥ 1/c`, and
    /// `u = p + φ₀ ∈ [φ₀, π/2 + φ₀]`, where `sin u ≥ μu`. With `s = (u + κ_m)/cos y`,
    /// `|G| ≤ s/(|α|²sin²u + 1/c) ≤ F(u)/cos y_m`, where `F(u) = (u + κ_m)/(A₂u² + B₀)`, `A₂ = μ²|α|²` and `B₀ = 1/c`.
    /// - `F` rises to its maximum `(κ_m + √(κ_m² + B₀/A₂))/(2B₀)` at `u* = (B₀/A₂)/(κ_m + √(κ_m² + B₀/A₂))`, and falls
    ///   after it.
    /// - `u ≥ max(φ₀, δ cos y_m − κ_m)`.
    /// - A lower `A₂` raises `F` and `u*`, so it is taken from below.
    fn middle_upper(&self, cos_height: f64) -> Option<f64> {
        let steepness = point(self.chord).mul(point(self.magnitude.lo));
        let curvature = point(steepness.mul(steepness).lo);
        let ratio = quotient(self.inverse_complement, curvature)?;
        let root = self.offset.mul(self.offset).add(ratio).sqrt();
        let turning = quotient(ratio, self.offset.add(root))?;
        let start = point(self.scale)
            .mul(point(cos_height))
            .sub(point(self.offset.hi))
            .lo
            .max(self.angle.lo);
        let peak = if start >= turning.hi {
            let lowest = point(start);
            quotient(
                lowest.add(self.offset),
                curvature.mul(lowest.mul(lowest)).add(self.inverse_complement),
            )?
        } else {
            quotient(self.offset.add(root), self.inverse_complement.scale(2.0))?
        };
        Some(quotient(peak, point(cos_height))?.hi)
    }

    /// A region with `p < 0`, where `|p| ≤ dip`, `|q| ≤ lift` and `s ≤ δ/cos` (`cos = cos y_m` in the middle, 1 on the
    /// left cap). There `|sin 2p| ≤ min(2·dip, 1)`, so `Re B ≥ β/c` ([`Self::curvature_floor`]). And `sin w ≥ w` for
    /// `w ≤ 0` gives `Re A/√Re B ≥ −X`, with `X = cosh(lift)·|α|σ·max(0, dip − φ₀)/√β`. So
    /// `|G| ≤ s·N(−X)/Re B ≤ δ·c·N(−X)/(β·cos)`.
    fn flank_bound(&self, dip: f64, lift: f64, cosine: f64) -> Option<f64> {
        let damping = self.curvature_floor(dip, lift)?;
        let depth = point(dip).sub(point(self.angle.lo)).hi.max(0.0);
        let reach = quotient(
            point(cosh_upper(lift))
                .mul(point(self.magnitude.hi))
                .mul(point(self.deviation.hi))
                .mul(point(depth)),
            point(damping).sqrt(),
        )?
        .hi;
        let numerator = point(self.scale)
            .mul(point(self.complement))
            .mul(point(laplace_growth(reach)));
        Some(quotient(numerator, point(damping).mul(point(cosine)))?.hi)
    }

    /// `β = 1 − τ·min(2·dip, 1)·cosh(2·lift)` from below, or `None` unless it is positive. It bounds
    /// `1 + τ sin 2p cosh 2q` from below wherever `|sin 2p| ≤ min(2·dip, 1)` and `|q| ≤ lift`.
    fn curvature_floor(&self, dip: f64, lift: f64) -> Option<f64> {
        let loss = if self.tau == 0.0 || dip == 0.0 {
            0.0
        } else {
            point(self.tau)
                .mul(point((2.0 * dip).min(1.0)))
                .mul(point(cosh_upper(2.0 * lift)))
                .hi
        };
        let floor = point(1.0).sub(point(loss)).lo;
        (floor > 0.0).then_some(floor)
    }

    /// The right cap, `x ∈ (L, L + e]`. Since `δe^L = π/2 + δ` exactly, `s ∈ (π/2 + δ, (π/2 + δ)e^e]`. So
    /// `p − π/2 ≤ p_R = (π/2 + δ)·expm1(e)`, `p ≥ p_lo = (π/2 + δ)cos y_c − δ`, and `|q| ≤ (π/2 + δ)e^e·y_c`.
    /// - `u = p + φ₀` lies in `[p_lo + φ_lo, π/2 + p_R + φ_hi]`, which must sit inside `(0, π)`. `sin` is concave there,
    ///   so `sin u ≥ σ_R`, the smaller end value.
    /// - `sin 2p < 0` only for `p > π/2`, where `|sin 2p| ≤ min(2p_R, 1)`, and for `p < 0`, where
    ///   `|sin 2p| ≤ min(2|p_lo|, 1)`.
    /// - So `|G| ≤ (π/2 + δ)e^e/(|α|²σ_R² + β_R/c)`.
    fn right_cap_bound(&self, extension: f64, cap_height: f64, cos_cap: f64) -> Option<f64> {
        let base = half_pi().add(point(self.scale));
        let growth = exp_enclosure(point(extension));
        let top = base.mul(growth).hi;
        let overshoot = base.mul(growth.sub(point(1.0))).hi.max(0.0);
        let lowest = point(base.lo)
            .mul(point(cos_cap))
            .sub(point(self.scale))
            .lo;
        let lift = point(top).mul(point(cap_height)).hi;
        let start = point(lowest).add(point(self.angle.lo)).lo;
        let end = half_pi()
            .add(point(overshoot))
            .add(point(self.angle.hi))
            .hi;
        if !(start > 0.0 && end < pi().lo) {
            return None;
        }
        let sine = sin_lower(start).min(sin_lower(end));
        if !(sine > 0.0) {
            return None;
        }
        let damping = self.curvature_floor(overshoot.max(-lowest), lift)?;
        let slope = point(self.magnitude.lo).mul(point(sine));
        let denominator = slope
            .mul(slope)
            .add(point(damping).mul(self.inverse_complement));
        Some(quotient(point(top), denominator)?.hi)
    }

    /// Trefethen's `(64/15)·H·M·r^{−2n}/(r² − 1)` from above, or infinity when `r ≤ 1`.
    fn truncation(&self, radius: f64, bound: f64, order: usize) -> f64 {
        let r = point(radius);
        let squared = r.mul(r);
        let mut power = point(1.0);
        let mut factor = squared;
        let mut remaining = order;
        while remaining > 0 {
            if remaining % 2 == 1 {
                power = power.mul(factor);
            }
            factor = factor.mul(factor);
            remaining /= 2;
        }
        let numerator = quotient(point(64.0), point(15.0))
            .map(|constant| constant.mul(point(self.half_length.hi)).mul(point(bound)));
        numerator
            .and_then(|numerator| quotient(numerator, power.mul(squared.sub(point(1.0)))))
            .map_or(f64::INFINITY, |value| value.hi)
    }

    /// A lower bound on `I`, used only to choose the order:
    /// `I ≥ ∫₀^{π/2} dθ/(|α|²(θ + φ₀)² + β)`, with `β = 3(1 + τ)/c`, from `N(x) ≥ 1/(x² + 3)` (Sampford's bound on `R`),
    /// `sin u ≤ u` and `B ≤ (1 + τ)/c`. In closed form it is `(w/β)·atan(πw/(2w² + φ₀(π + 2φ₀)))` with `w = √β/|α|`.
    /// It decreases in `φ₀`, so it is taken at `φ_hi`.
    fn integral_floor(&self) -> Option<f64> {
        let beta = point(3.0)
            .mul(point(1.0).add(point(self.tau)))
            .mul(self.inverse_complement);
        let width = quotient(beta.sqrt(), self.magnitude)?;
        let angle = point(self.angle.hi);
        let spread = width
            .mul(width)
            .scale(2.0)
            .add(angle.mul(pi().add(angle.scale(2.0))));
        let argument = quotient(pi().mul(width), spread)?;
        let floor = quotient(width, beta)?.mul(atan_enclosure(argument)?).lo;
        (floor > 0.0).then_some(floor)
    }

    /// The radius `r`, the bound `M` on its ellipse, and the certified order `n`.
    /// - `r` minimizes the order proposed by logarithms, `ln((64/15)·H·M/((r² − 1)·η·I_low))/(2 ln r)`, by golden
    ///   section on `ln r` (see [`golden_section_minimum`]). `y_m = H·b < π/2` caps `b` at `π/(2H)`, so the bracket is
    ///   `(0, ln(b_cap + √(b_cap² + 1))]`. An inadmissible `r` scores `+∞`.
    /// - The anchor [`ANCHOR_LOG_RADIUS`] is always a candidate. A finite proposal is at most `ln(f64::MAX)/(2 ln r)`, so
    ///   the proposed order is at most `⌈ln(f64::MAX)/(2·0.2)⌉ = 1775`, and a larger proposal declines.
    /// - `n` is the smallest order from the proposal on whose truncation, in intervals, is at most `η·I_low`.
    ///
    /// Any admissible `r` and any `n` give a valid enclosure with their own truncation, so neither choice bears on
    /// soundness.
    fn certified_order(&self) -> Option<(f64, f64, usize)> {
        let floor = self.integral_floor()?;
        let proposal = |log_radius: f64| -> f64 {
            let radius = libm::exp(log_radius);
            match self.ellipse_bound(radius) {
                Some(bound) => {
                    let ratio = 64.0 / 15.0 * self.half_length.hi * bound
                        / ((radius * radius - 1.0) * floor * TRUNCATION_TARGET);
                    let order = libm::log(ratio) / (2.0 * log_radius);
                    if order.is_nan() { f64::INFINITY } else { order }
                }
                None => f64::INFINITY,
            }
        };
        let cap = half_pi().hi / self.half_length.lo;
        let upper = libm::log(cap + (cap * cap + 1.0).sqrt());
        if !(upper.is_finite() && upper > ANCHOR_LOG_RADIUS) {
            return None;
        }
        let anchor = (ANCHOR_LOG_RADIUS, proposal(ANCHOR_LOG_RADIUS));
        let (log_radius, estimate) = match golden_section_minimum(&proposal, upper) {
            Some(searched) if searched.1 < anchor.1 => searched,
            _ => anchor,
        };
        let ceiling = (libm::log(f64::MAX) / (2.0 * ANCHOR_LOG_RADIUS)).ceil();
        if !(estimate <= ceiling) {
            return None;
        }
        let radius = libm::exp(log_radius);
        let bound = self.ellipse_bound(radius)?;
        if !(radius > 1.0) {
            return None;
        }
        let target = point(floor).mul(point(TRUNCATION_TARGET)).lo;
        let mut order = estimate.ceil().max(1.0) as usize;
        let mut truncation = self.truncation(radius, bound, order);
        while !(truncation <= target) {
            let next = self.truncation(radius, bound, order + 1);
            if !(next < truncation) {
                return None;
            }
            order += 1;
            truncation = next;
        }
        Some((radius, bound, order))
    }

    /// `Φ₂` enclosed by the `order`-point certified rule plus `truncation`, or `None` when an enclosure fails. Each
    /// true node `v_j = H(1 + x_j)` lies in the interval from `H`'s enclosure and `|x̂_j − x_j| ≤ e_x`, and each true
    /// weight in `ŵ_j(1 ± e_w)`. So the interval sum encloses the exact rule, and adding `±truncation` encloses `I`.
    fn orthant_enclosure(&self, order: usize, truncation: f64) -> Option<ClosedInterval> {
        let rule = rule_of_order(order);
        let (node_error, weight_error) = (rule.node_error, rule.weight_relative_error);
        if !(node_error.is_finite() && weight_error.is_finite() && truncation.is_finite()) {
            return None;
        }
        let node_spread = ClosedInterval::new(-node_error, node_error);
        let weight_spread = ClosedInterval::new(
            point(1.0).sub(point(weight_error)).lo,
            point(1.0).add(point(weight_error)).hi,
        );
        let mut sum = point(0.0);
        for (&node, &weight) in rule.nodes.iter().zip(&rule.weights) {
            let position = self
                .half_length
                .mul(point(1.0).add(point(node)).add(node_spread));
            let term = self.integrand(ClosedInterval::new(position.lo.max(0.0), position.hi))?;
            sum = sum.add(term.mul(point(weight).mul(weight_spread)));
        }
        let integral = self
            .half_length
            .mul(sum)
            .add(ClosedInterval::new(-truncation, truncation));
        let orthant = self
            .density
            .mul(ClosedInterval::new(integral.lo.max(0.0), integral.hi));
        Some(ClosedInterval::new(orthant.lo, orthant.hi.min(1.0)))
    }

    /// `G(v) = (θ + δ)·g(θ)` over an interval of `v ≥ 0`, with `θ = δ·expm1(v)`, or `None` when an enclosure fails.
    fn integrand(&self, position: ClosedInterval) -> Option<ClosedInterval> {
        let growth = expm1_enclosure(position);
        let scale = point(self.scale);
        let (cosine, sine) = cos_sin(scale.mul(growth))?;
        let jacobian = scale.mul(point(1.0).add(growth));
        let slope = self.alpha[0].mul(cosine).add(self.alpha[1].mul(sine));
        let spread = point(1.0)
            .add(point(2.0 * self.tau).mul(sine.mul(cosine)))
            .mul(self.inverse_complement);
        let argument = quotient(slope, spread.sqrt())?;
        let tail = mills_enclosure(ClosedInterval::new(argument.lo.max(0.0), argument.hi))?;
        Some(jacobian.mul(quotient(tail, spread)?))
    }
}

/// The golden-section minimum of `objective` over `(0, upper]`: the best point evaluated and its value, or `None` if
/// every value was infinite. It stops once the bracket is within `√ε·upper`. Near a minimum the objective is quadratic,
/// so values resolved to `ε` place the minimizer only to `√ε` of the scale. The comparison keeps the left part on a
/// tie, so a bracket whose probes are both inadmissible moves toward the smaller radii, where admissibility holds.
fn golden_section_minimum(objective: impl Fn(f64) -> f64, upper: f64) -> Option<(f64, f64)> {
    let shrink = 0.5 * (5.0_f64.sqrt() - 1.0);
    let resolution = f64::EPSILON.sqrt() * upper;
    let (mut lower, mut higher) = (0.0_f64, upper);
    let mut left = higher - shrink * (higher - lower);
    let mut right = lower + shrink * (higher - lower);
    let (mut left_value, mut right_value) = (objective(left), objective(right));
    let mut best = if left_value <= right_value {
        (left, left_value)
    } else {
        (right, right_value)
    };
    while higher - lower > resolution {
        if left_value <= right_value {
            higher = right;
            right = left;
            right_value = left_value;
            left = higher - shrink * (higher - lower);
            left_value = objective(left);
            if left_value < best.1 {
                best = (left, left_value);
            }
        } else {
            lower = left;
            left = right;
            left_value = right_value;
            right = lower + shrink * (higher - lower);
            right_value = objective(right);
            if right_value < best.1 {
                best = (right, right_value);
            }
        }
    }
    best.1.is_finite().then_some(best)
}

/// The positive form's value with its certified bound at `(h, k, ρ, c)`, or `None` outside the certified region or when
/// an enclosure fails.
pub(super) fn relative_orthant(h: f64, k: f64, rho: f64, complement: f64) -> Option<BoundedProbability> {
    let form = PositiveForm::new(h, k, rho, complement)?;
    let (radius, bound, order) = form.certified_order()?;
    let enclosure = form.orthant_enclosure(order, form.truncation(radius, bound, order))?;
    bounded(enclosure)
}

/// The midpoint of a positive, finite enclosure, with the larger distance to an end as its rounding.
fn bounded(enclosure: ClosedInterval) -> Option<BoundedProbability> {
    if !(enclosure.lo > 0.0 && enclosure.hi.is_finite()) {
        return None;
    }
    let value = midpoint(enclosure);
    let rounding = point(enclosure.hi)
        .sub(point(value))
        .hi
        .max(point(value).sub(point(enclosure.lo)).hi);
    Some(BoundedProbability {
        value,
        rounding,
        contract: RoundingContract::Relative,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bivariate_normal::{
        BIVARIATE_NORMAL_CDF_ERROR_BOUND, bivariate_normal_cdf_with_complement,
        bivariate_normal_cdf_with_complement_bounded,
    };

    /// `(h, k, ρ, c, Φ₂)` at exact binary64 laws inside the certified region: fr-kernel's 14 cells with their
    /// `c = (1 − ρ)(1 + ρ)` as formed in binary64, a sweep over `φ₀ ∈ {π/4, 0.2, 0.02}`, `|α| ∈ {0.1, 1, 10, 40}` and
    /// `ρ ∈ {0, −½, −⅞, −15/16, −(1 − 2^−20)}` with `c = 1 − ρ²` exact, the `α₁ = 0` boundary, and `ρ → −1`.
    /// The values are the positive form at 60 digits on the log map (MSI job 1246513, fr-bvn-refs.py). At every cell
    /// they agree with the same form at 40 digits within 8e−39 and with plain `θ` on graded panels within 1e−54.
    const REFERENCES: &[(f64, f64, f64, f64, f64)] = &[
        (-3.0, -3.0, -0.9, 0.18999999999999995, 3.269436016883910729583269e-43),
        (-3.0, -3.0, -0.5, 0.75, 7.147502181270789972727562e-11),
        (-5.0, -5.0, -0.5, 0.75, 3.432573480035108395745147e-25),
        (-5.0, -5.0, -0.9, 0.18999999999999995, 3.874806403645786240613379e-113),
        (-8.0, -8.0, -0.9, 0.18999999999999995, 6.40858386024772904391029e-283),
        (-8.0, -8.0, -0.5, 0.75, 1.822994799115843598842091e-59),
        (-6.0, 1.0, -0.8, 0.35999999999999993, 4.915404365687094306588655e-20),
        (-2.1213203435596424, -2.1213203435596424, -0.4499999999999999, 0.7975, 0.000002824828911581415154655137),
        (-2.1213203435596424, -2.1213203435596424, -0.24999999999999994, 0.9375, 0.00004038751556712353431887849),
        (-3.5355339059327373, -3.5355339059327373, -0.4499999999999999, 0.7975, 5.427585456102467196119415e-13),
        (-3.5355339059327373, -3.5355339059327373, -0.24999999999999994, 0.9375, 3.880210803103823873157458e-10),
        (-5.65685424949238, -5.65685424949238, -0.4499999999999999, 0.7975, 8.836551294977451106681253e-29),
        (-5.65685424949238, -5.65685424949238, -0.24999999999999994, 0.9375, 8.192900472692605917746837e-22),
        (-4.242640687119285, 0.7071067811865475, -0.3999999999999999, 0.8400000000000001, 0.00000133819440443039165197172),
        (-0.07071067811865475, -0.07071067811865477, 0.0, 1.0, 0.2226084610716785195505632),
        (-0.7071067811865475, -0.7071067811865476, 0.0, 1.0, 0.05748009179432582607233821),
        (-7.071067811865475, -7.0710678118654755, 0.0, 1.0, 5.90945654870675352221836e-25),
        (-0.019866933079506124, -0.09800665778412417, 0.0, 1.0, 0.2268285060840146675217905),
        (-0.19866933079506122, -0.9800665778412416, 0.0, 1.0, 0.06888734303308200621694888),
        (-1.9866933079506122, -9.800665778412416, 0.0, 1.0, 1.312971034525271658290848e-24),
        (-0.0019998666693333083, -0.09998000066665778, 0.0, 1.0, 0.2297229048827058579763947),
        (-0.01999866669333308, -0.9998000066665778, 0.0, 1.0, 0.07808572253262010208341956),
        (-0.1999866669333308, -9.998000066665778, 0.0, 1.0, 3.27141784560058206997374e-24),
        (-0.03535533905932737, -0.03535533905932739, -0.5, 0.75, 0.1529092303189264771823167),
        (-0.3535533905932737, -0.35355339059327384, -0.5, 0.75, 0.06091110402609455890653886),
        (-3.535533905932737, -3.535533905932738, -0.5, 0.75, 4.79732813743937939265979e-14),
        (-14.142135623730947, -14.142135623730953, -0.5, 0.75, 4.381330415995519399874518e-178),
        (0.02913639581255596, -0.0880731912443711, -0.5, 0.75, 0.1548560552134127555335868),
        (0.2913639581255596, -0.880731912443711, -0.5, 0.75, 0.06036490361454385725362012),
        (2.9136395812555955, -8.80731912443711, -0.5, 0.75, 2.399897393813549822009433e-20),
        (11.654558325022382, -35.22927649774844, -0.5, 0.75, 9.42101222435047345595731e-284),
        (0.047990133663995585, -0.09898006733199113, -0.5, 0.75, 0.1562075060690998935110428),
        (0.4799013366399558, -0.9898006733199112, -0.5, 0.75, 0.06103588079274596242648163),
        (4.799013366399558, -9.898006733199113, -0.5, 0.75, 8.685425776607361207765002e-24),
        (-0.008838834764831834, -0.008838834764831854, -0.875, 0.234375, 0.07695263522584738693237734),
        (-0.08838834764831838, -0.0883883476483186, -0.875, 0.234375, 0.04997453892520629328487762),
        (-0.8838834764831835, -0.8838834764831853, -0.875, 0.234375, 0.0000104166417332957010934838),
        (-3.535533905932734, -3.5355339059327413, -0.875, 0.234375, 1.505852693080382092701864e-47),
        (0.06588889248160251, -0.08062309133955631, -0.875, 0.234375, 0.07731262180522052625428407),
        (0.6588889248160252, -0.8062309133955631, -0.875, 0.234375, 0.04126335651070971027015134),
        (6.588889248160251, -8.06230913395563, -0.875, 0.234375, 4.644503247137468248306677e-17),
        (26.355556992641002, -32.24923653582252, -0.875, 0.234375, 8.750618674217942718184707e-233),
        (0.08548263391399225, -0.09823011733099114, -0.875, 0.234375, 0.07757754644192469746442788),
        (0.8548263391399225, -0.9823011733099113, -0.875, 0.234375, 0.03727044648979818222493337),
        (8.548263391399225, -9.823011733099113, -0.875, 0.234375, 1.761220941861926798807014e-23),
        (-0.004419417382415913, -0.004419417382415941, -0.9375, 0.12109375, 0.05482126082916888175565239),
        (-0.04419417382415913, -0.044194173824159355, -0.9375, 0.12109375, 0.04066307181674537693561862),
        (-0.44194173824159133, -0.4419417382415931, -0.9375, 0.12109375, 0.000286428351546224434968987),
        (-1.7677669529663653, -1.7677669529663724, -0.9375, 0.12109375, 1.070813419334029402694757e-25),
        (0.07201430859311028, -0.07938140802208718, -0.9375, 0.12109375, 0.05495015455897372657171033),
        (0.7201430859311028, -0.7938140802208717, -0.9375, 0.12109375, 0.03222019995793202794013565),
        (7.201430859311027, -7.938140802208717, -0.9375, 0.12109375, 1.695481475701880661439378e-16),
        (28.805723437244108, -31.752563208834868, -0.9375, 0.12109375, 3.289205327406240826238164e-224),
        (0.09173138395565836, -0.09810512566415781, -0.9375, 0.12109375, 0.05505338932474612821028315),
        (0.9173138395565836, -0.981051256641578, -0.9375, 0.12109375, 0.02835687868719278843577421),
        (9.173138395565836, -9.810512566415781, -0.9375, 0.12109375, 1.8927969827920183217224e-23),
        (-6.743495760408447e-08, -6.743495763184004e-08, -0.9999990463256836, 1.9073477233177982e-06, 0.0002197769039850678399824443),
        (-6.743495760686002e-07, -6.743495762906448e-07, -0.9999990463256836, 1.9073477233177982e-06, 0.000219534883845604394422168),
        (-6.743495760908047e-06, -6.743495762684404e-06, -0.9999990463256836, 1.9073477233177982e-06, 0.0002171240209912400198497796),
        (-2.6973983043632188e-05, -2.6973983050737615e-05, -0.9999990463256836, 1.9073477233177982e-06, 0.0002092104189070752842026982),
        (0.07813963123818567, -0.07813974365120187, -0.9999990463256836, 1.9073477233177982e-06, 0.0002191114344607616549259628),
        (0.7813963123818568, -0.7813974365120187, -0.9999990463256836, 1.9073477233177982e-06, 0.0001618102635827044333501329),
        (7.813963123818566, -7.8139743651201865, -0.9999990463256836, 1.9073477233177982e-06, 1.199485054094102574371211e-17),
        (31.255852495274265, -31.255897460480746, -0.9999990463256836, 1.9073477233177982e-06, 1.536572807519442832567512e-216),
        (0.09798003864896568, -0.09798013590454595, -0.9999990463256836, 1.9073477233177982e-06, 0.000218731958345036925494093),
        (0.9798003864896568, -0.9798013590454595, -0.9999990463256836, 1.9073477233177982e-06, 0.0001358902632114275035899176),
        (9.798003864896568, -9.798013590454595, -0.9999990463256836, 1.9073477233177982e-06, 3.103457587841605118742749e-25),
        (2.0, -4.0, -0.5, 0.75, 0.00001421363540820410034036554),
        (10.0, -20.0, -0.5, 0.75, 1.345282423548078274910184e-89),
        (1.5, -4.0, -0.375, 0.859375, 0.00001469085271207348992920217),
        (0.0, -3.0, 0.0, 1.0, 0.0006749490158150472633259074),
        (-0.001, -0.001, -0.9999990463256836, 1.9073477233177982e-06, 0.00001815382887394531657418327),
        (-0.01, -0.01, -0.9999990463256836, 1.9073477233177982e-06, 2.986869434678768811981115e-52),
    ];

    /// Cells the sweep reaches whose density `φ₂` is below the normal range, so the certificate declines.
    const SUBNORMAL_DENSITY: &[(f64, f64, f64, f64)] = &[
        (-28.2842712474619, -28.284271247461902, 0.0, 1.0),
        (-7.946773231802449, -39.20266311364966, 0.0, 1.0),
        (-0.7999466677333232, -39.99200026666311, 0.0, 1.0),
        (19.196053465598233, -39.59202693279645, -0.5, 0.75),
        (34.1930535655969, -39.29204693239645, -0.875, 0.234375),
        (36.69255358226334, -39.242050265663124, -0.9375, 0.12109375),
        (39.19201545958627, -39.19205436181838, -0.9999990463256836, 1.9073477233177982e-06),
        (-0.03, -0.03, -0.9999990463256836, 1.9073477233177982e-06),
    ];

    fn bounded_at(h: f64, k: f64, rho: f64, complement: f64) -> BoundedProbability {
        bivariate_normal_cdf_with_complement_bounded(h, k, rho, complement).unwrap()
    }

    #[test]
    fn certified_bound_covers_the_references() {
        let mut absolute_certifies_nothing = false;
        for &(h, k, rho, complement, reference) in REFERENCES {
            let result = bounded_at(h, k, rho, complement);
            let error = (result.value - reference).abs();
            assert!(
                error <= result.rounding,
                "({h}, {k}, {rho}, {complement}) value={:e} reference={reference:e} error={error:e} rounding={:e}",
                result.value,
                result.rounding
            );
            // Where the absolute contract certifies no digit, the entry must deliver the relative one.
            if reference <= BIVARIATE_NORMAL_CDF_ERROR_BOUND {
                absolute_certifies_nothing = true;
                assert_eq!(
                    result.contract,
                    RoundingContract::Relative,
                    "({h}, {k}, {rho}, {complement}) reference={reference:e}"
                );
            }
            if result.contract == RoundingContract::Relative {
                assert!(result.rounding < BIVARIATE_NORMAL_CDF_ERROR_BOUND);
                assert!(result.rounding < result.value, "({h}, {k}, {rho}) certifies no digit");
            }
        }
        assert!(
            absolute_certifies_nothing,
            "some reference must lie below the absolute bound, where only the relative contract resolves it"
        );
    }

    #[test]
    fn doubled_order_agrees_within_both_bounds() {
        for &(h, k, rho, complement, reference) in REFERENCES {
            let form = PositiveForm::new(h, k, rho, complement).unwrap();
            let (radius, bound, order) = form.certified_order().unwrap();
            let certified = bounded(form.orthant_enclosure(order, form.truncation(radius, bound, order)).unwrap()).unwrap();
            let doubled_order = 2 * order;
            let doubled = bounded(
                form.orthant_enclosure(doubled_order, form.truncation(radius, bound, doubled_order))
                    .unwrap(),
            )
            .unwrap();
            let gap = (certified.value - doubled.value).abs();
            assert!(
                gap <= certified.rounding + doubled.rounding,
                "({h}, {k}, {rho}) n={order} gap={gap:e} bounds={:e}+{:e}",
                certified.rounding,
                doubled.rounding
            );
            assert!((doubled.value - reference).abs() <= doubled.rounding, "({h}, {k}, {rho}) at 2n");
        }
    }

    #[test]
    fn every_reference_certifies_at_the_anchor_radius() {
        // ANCHOR_LOG_RADIUS is derived admissible throughout the certified region; the references span it.
        let radius = libm::exp(ANCHOR_LOG_RADIUS);
        for &(h, k, rho, complement, reference) in REFERENCES {
            let form = PositiveForm::new(h, k, rho, complement).unwrap();
            assert!(
                form.ellipse_bound(radius).is_some(),
                "({h}, {k}, {rho}) with value {reference:e} is inadmissible at the anchor"
            );
            let order = form.certified_order().unwrap().2;
            assert!((1..=1775).contains(&order), "({h}, {k}, {rho}) order {order}");
        }
    }

    #[test]
    fn a_rule_below_the_certified_order_misses_the_certified_bound() {
        // The rule alone at half the certified order, with no truncation term: if it stayed within the certified
        // bound everywhere, the order derivation would not be tested by the reference comparison.
        let mut missed = false;
        for &(h, k, rho, complement, reference) in REFERENCES {
            let form = PositiveForm::new(h, k, rho, complement).unwrap();
            let (radius, bound, order) = form.certified_order().unwrap();
            let certified = bounded(form.orthant_enclosure(order, form.truncation(radius, bound, order)).unwrap()).unwrap();
            let halved = form.orthant_enclosure((order / 2).max(1), 0.0).unwrap();
            missed |= (midpoint(halved) - reference).abs() > certified.rounding;
        }
        assert!(missed, "a rule at half the certified order must miss the certified bound somewhere");
    }

    #[test]
    fn outside_the_certified_region_the_value_is_the_core_bit_for_bit() {
        let mut cells: Vec<(f64, f64, f64, f64)> = SUBNORMAL_DENSITY.to_vec();
        cells.extend([
            // ρ > 0.
            (-3.0, -3.0, 0.5, 0.75),
            (-8.0, -8.0, 0.5, 0.75),
            // An inactive constraint: α₁ < 0.
            (2.0, 1.0, -0.5, 0.75),
            (-6.0, 2.0, -0.25, 0.9375),
            // The apex at the origin, α = 0.
            (0.0, 0.0, -0.5, 0.75),
            // A singular correlation and infinite bounds.
            (-3.0, -3.0, -1.0, 0.0),
            (f64::NEG_INFINITY, -1.0, -0.5, 0.75),
            (-1.0, f64::INFINITY, -0.5, 0.75),
        ]);
        for &(h, k, rho, complement) in &cells {
            let result = bounded_at(h, k, rho, complement);
            let plain = bivariate_normal_cdf_with_complement(h, k, rho, complement).unwrap();
            assert_eq!(result.contract, RoundingContract::Absolute, "({h}, {k}, {rho}, {complement})");
            assert_eq!(result.rounding, BIVARIATE_NORMAL_CDF_ERROR_BOUND);
            assert_eq!(result.value.to_bits(), plain.to_bits(), "({h}, {k}, {rho}, {complement})");
        }
        assert!(bivariate_normal_cdf_with_complement_bounded(0.0, 0.0, -0.5, -1.0e-3).is_err());
        assert!(bivariate_normal_cdf_with_complement_bounded(f64::NAN, 0.0, -0.5, 0.75).is_err());
    }

    #[test]
    fn the_cited_libm_is_the_locked_one() {
        // `exp_enclosure` rests on the error analysis of this libm release.
        let lock = include_str!("../../../../Cargo.lock");
        assert!(lock.contains("name = \"libm\"\nversion = \"0.2.16\"\n"), "Cargo.lock no longer pins libm 0.2.16");
    }

    #[test]
    fn series_enclosures_contain_the_library_values() {
        // libm's own results are within an ulp of the truth, so each enclosure widened by one ulp must contain them.
        let near = |enclosure: ClosedInterval, value: f64| enclosure.lo.next_down() <= value && value <= enclosure.hi.next_up();
        let mut resolves_offsets = false;
        for step in 0..=64 {
            let angle = 0.5 * PI * f64::from(step) / 64.0;
            let (cosine, sine) = cos_sin(point(angle)).unwrap();
            assert!(near(cosine, libm::cos(angle)) && near(sine, libm::sin(angle)), "angle={angle}");
            assert!(cosine.hi - cosine.lo <= 32.0 * f64::EPSILON && sine.hi - sine.lo <= 32.0 * f64::EPSILON);
            resolves_offsets |= !near(cos_sin(point(angle + 1.0e-12)).unwrap().0, libm::cos(angle));
            // The region bounds' short partial sums: below, and short by at most their first omitted term.
            let (cos_below, sin_below) = (cos_lower(angle), sin_lower(angle));
            assert!(cos_below <= libm::cos(angle).next_up() && libm::cos(angle) - cos_below <= 5.0e-7, "angle={angle}");
            assert!(sin_below <= libm::sin(angle).next_up() && libm::sin(angle) - sin_below <= 6.0e-8, "angle={angle}");
            let beyond = angle + 0.5 * PI;
            assert!(sin_lower(beyond) <= libm::sin(beyond).next_up() && libm::sin(beyond) - sin_lower(beyond) <= 5.0e-7);
        }
        assert!(resolves_offsets, "the enclosures must resolve a 1e-12 offset in the angle");
        for value in [0.0, 1.0e-300, 1.0e-12, 0.1, 0.5, 1.0, 1.5, 10.0, 1.0e8] {
            let enclosure = atan_point(value).unwrap();
            assert!(near(enclosure, libm::atan(value)), "atan({value})");
            assert!(enclosure.hi - enclosure.lo <= 64.0 * f64::EPSILON * libm::atan(value).max(f64::MIN_POSITIVE));
        }
        for value in [0.0, 1.0e-300, 1.0e-12, 0.3, 0.69, 0.7, 2.0, 4.4] {
            let enclosure = expm1_enclosure(point(value));
            assert!(near(enclosure, libm::expm1(value)), "expm1({value})");
            assert!(enclosure.hi - enclosure.lo <= 32.0 * f64::EPSILON * libm::expm1(value).max(f64::MIN_POSITIVE));
        }
        let root = point(2.0).sqrt();
        assert!(root.lo < std::f64::consts::SQRT_2 && std::f64::consts::SQRT_2 < root.hi);
        assert_eq!(point(2.25).sqrt(), point(1.5));
        // Exact products and quotients stay points; inexact ones are one ulp wide on the residual's side.
        assert_eq!(ClosedInterval::product(0.5, -20.0), point(-10.0));
        assert_eq!(ClosedInterval::quotient(0.75, 1.5), point(0.5));
        let third = ClosedInterval::quotient(1.0, 3.0);
        assert!(third.hi == third.lo.next_up() && third.lo.mul_add(3.0, -1.0) < 0.0 && third.hi.mul_add(3.0, -1.0) > 0.0);
        let tenth = ClosedInterval::product(0.1, 3.0);
        assert!(tenth.hi == tenth.lo.next_up() && 0.1_f64.mul_add(3.0, -tenth.lo) > 0.0 && 0.1_f64.mul_add(3.0, -tenth.hi) < 0.0);
    }
}
