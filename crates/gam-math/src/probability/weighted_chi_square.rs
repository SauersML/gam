//! Upper tail of a signed weighted sum of independent central chi-squares, with a relative
//! error bound on the value it returns.

use crate::double_double::SMALLEST_SUBNORMAL;
use crate::roundoff::{UNIT_ROUNDOFF, accumulation_growth};
use crate::special::{logaddexp, softplus};

/// One term `λ_j·χ²_{h_j}` of a weighted sum of independent central chi-squares, with the
/// weight's SIGN and the term's degrees of freedom both carried explicitly.
///
/// Two things separate this from a plain list of non-negative one-degree-of-freedom
/// weights, and each of them is a distribution that form cannot express:
///
/// * **A negative weight makes a RATIO a tail.** `P(A/B > t)` for independent
///   non-negative `A`, `B` is `P(A − tB > 0)`, so every F-shaped reference —
///   any statistic whose scale was estimated from the same data — is a
///   *signed* combination evaluated at zero. The classical `F_{a,b}` is the
///   two-term case `λ = (1, −t·a/b)`, `h = (a, b)`.
/// * **A multiplicity is not `h` copies of a weight.** It is, mathematically,
///   but the integrand costs one complex logarithm per TERM, and a residual sum
///   of squares carries `n − p` unit weights. Folding them into one term with
///   `h = n − p` is what makes an `n`-sized reference cost the same as a
///   `p`-sized one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeightedChiSquareTerm {
    /// `λ_j`, of either sign. A zero weight contributes nothing and is dropped.
    pub weight: f64,
    /// `h_j > 0`. Real rather than integral: a two-moment summary of a spectrum
    /// is a chi-square with a fractional shape, and this type is what carries it.
    pub degrees_of_freedom: f64,
}

/// A probability together with a bound on its RELATIVE error: `|probability − P| ≤
/// relative_error·P` for the exact `P`.
///
/// Relative rather than absolute because the values this carries reach `1e-300`, where an
/// absolute bound says nothing. A `relative_error` of `1` or more is a bound that carries no
/// information about the value beyond `0 ≤ P ≤ 1` — the typed form of "not resolved", which a
/// caller reads instead of a number that looks exact. `NaN` in both fields marks invalid input.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TailProbability {
    pub probability: f64,
    pub relative_error: f64,
}

impl TailProbability {
    const fn exact(probability: f64) -> Self {
        Self { probability, relative_error: 0.0 }
    }

    /// The same bound stated absolutely: `|probability − P| ≤ absolute_error()`. From
    /// `P ≥ probability/(1 + relative_error)` and `P ≤ probability/(1 − relative_error)`, the
    /// second being the binding one; a relative error of `1` or more leaves only `P ∈ [0, 1]`.
    pub fn absolute_error(&self) -> f64 {
        if self.probability.is_nan() || self.relative_error.is_nan() {
            return f64::NAN;
        }
        if self.relative_error >= 1.0 {
            return 1.0;
        }
        (self.relative_error * self.probability / (1.0 - self.relative_error)).min(1.0)
    }
}

/// Survival probability `P(Σ_j λ_j χ²_{h_j} > statistic)` for independent central chi-squares,
/// with weights of EITHER SIGN, with a relative error bound on the value returned. See
/// [`WeightedChiSquareTerm`] for why the sign and the multiplicity are worth carrying.
///
/// # Method
///
/// The upper tail is the Bromwich inversion of the moment generating function,
///
/// ```text
/// P(Q > x) = (1/2πi) ∫_{c−i∞}^{c+i∞} exp(K(s) − s·x) ds/s,
/// K(s) = −½ Σ_j h_j ln(1 − 2λ_j s),
/// ```
///
/// valid for any `c` in `(0, s₊)`, `s₊ = 1/(2·max λ_j⁺)`. It is the tail itself rather than
/// `1/2` plus a signed correction, which is where Imhof's form stops: the two halves of
/// `½ + (1/π)∫…` cancel, and no quadrature recovers a tail below about `1e-16` from their
/// difference. Here `exp(K(c) − c·x)` is factored out exactly, and the integral that remains
/// is of order one at every `x`.
///
/// `c` is the saddle point of `K(s) − s·x − ln s`, found by bisection on the sign of its
/// derivative. Any `c` in the strip gives the same integral, so the saddle is a choice that
/// makes the integrand concentrated, not a quantity whose error enters the answer, and the
/// bisection stops once it is inside the saddle's own width. The contour through `c` is bent
/// into a hyperbola with 45° asymptotes, opening toward the side where `exp(−s·x)` decays and
/// away from the nearest singularity, so the integrand decays exponentially in `|Im s|`
/// wherever `x ≠ 0` and algebraically at `x = 0`. The trapezoidal rule on `t = sinh v` is
/// refined by halving until successive levels agree to their own rounding.
///
/// # The bound
///
/// `relative_error` adds four derived pieces, none a tolerance: the difference between the
/// last two trapezoid levels, a truncation bound on the integrand beyond the last node (from
/// `|1 + w·d| ≥ max(b, |w|·|Im d|)` on the hyperbola), the rounding of every node from the
/// condition of its complex logarithms, and the rounding of the exponent `K(c) − c·x`. A
/// probability below the subnormal range is `0` with relative error `1`, found either from the
/// Chernoff bound `P ≤ exp(K(c) − c·x)` before any quadrature or from the exponent after it.
///
/// # Exact special cases
///
/// * no nonzero weight — `Q ≡ 0`;
/// * all weights positive and `x ≤ 0`, or all negative and `x ≥ 0` — the inequality is
///   decided by the support;
/// * `x = ±∞`.
///
/// Returns `NaN` in both fields if any weight is non-finite, if any degrees-of-freedom is not
/// finite and positive, or if `statistic` is `NaN`.
pub fn signed_weighted_chi_square_sf(terms: &[WeightedChiSquareTerm], statistic: f64) -> TailProbability {
    let invalid = TailProbability { probability: f64::NAN, relative_error: f64::NAN };
    if statistic.is_nan() {
        return invalid;
    }
    let mut active = Vec::with_capacity(terms.len());
    for term in terms {
        if !term.weight.is_finite() || !(term.degrees_of_freedom.is_finite() && term.degrees_of_freedom > 0.0) {
            return invalid;
        }
        if term.weight != 0.0 {
            active.push(*term);
        }
    }
    if active.is_empty() {
        // `Q` is identically zero: it exceeds a negative threshold with certainty and a
        // non-negative one never.
        return TailProbability::exact(if statistic < 0.0 { 1.0 } else { 0.0 });
    }
    let any_positive = active.iter().any(|term| term.weight > 0.0);
    let any_negative = active.iter().any(|term| term.weight < 0.0);
    if !any_negative && statistic <= 0.0 {
        // `Q > 0` almost surely once every weight is positive.
        return TailProbability::exact(1.0);
    }
    if !any_positive && statistic >= 0.0 {
        return TailProbability::exact(0.0);
    }
    if statistic == f64::INFINITY {
        return TailProbability::exact(0.0);
    }
    if statistic == f64::NEG_INFINITY {
        return TailProbability::exact(1.0);
    }
    match Problem::new(&active, statistic) {
        Ok(problem) => problem.tail(),
        Err(tail) => tail,
    }
}

/// Roundings between the saddle parameter `y` and one factor's coefficient `w_k`, apart from the
/// size of the logarithms on the way, which enter separately: `θ` or `η` (a softplus: exp,
/// log1p, sum), `ln m_k` (a log-add: exp, log1p, sum, or a softplus), `ln|ρ_k|` (two sums or
/// a softplus), `ln τ` (a scaled sum, a log, a softplus, a halving), and the final exp.
const COEFFICIENT_OPS: usize = 16;

/// Roundings in one node outside the sum over factors: `t = sinh v` and `cosh v`, the
/// hyperbola's `g` and `g′` (a product, a hypot, a sum and a quotient each), `z = w·d` (two
/// products), the complex `log1p` (its squared modulus in three operations, then `ln_1p` or a
/// hypot and a log, and an atan2), the exp, cos and sin of `Φ`, and the final product and sum.
const NODE_OPS: usize = 20;

/// Roundings in the exponent `−Σ e_k ln m_k − θξ + ln τ − ln π + ln J` beyond its terms'
/// own errors and the sum over `k` (the sum of its five parts and `ln π`), and the final exp.
const PREFIX_OPS: usize = 8;

/// `Σ_{i ∈ range} term(i)` by recursive halving.
///
/// Every summand then passes through at most [`pairwise_depth`]`(n)` additions, so the sum errs
/// by at most `γ_{⌈log₂ n⌉}·Σ|term(i)|` (Higham, *Accuracy and Stability of Numerical
/// Algorithms*, 2nd ed., §4.2) where a running sum errs by `γ_{n−1}·Σ|term(i)|`. The sums over
/// terms are the only place `n` enters the rounding, and at a thousand components the running
/// sum's `γ_{n−1}` alone was `1e-13` times an exponent of order `n`.
fn pairwise<T: Copy + std::ops::Add<Output = T>>(range: std::ops::Range<usize>, term: &impl Fn(usize) -> T) -> T {
    if range.len() == 1 {
        return term(range.start);
    }
    let middle = range.start + range.len() / 2;
    pairwise(range.start..middle, term) + pairwise(middle..range.end, term)
}

/// `⌈log₂ n⌉`, the most additions a summand passes through in [`pairwise`] over `n ≥ 1` terms:
/// the larger half of `n` is `⌈n/2⌉`.
fn pairwise_depth(n: usize) -> usize {
    n.next_power_of_two().trailing_zeros() as usize
}

/// One factor's contribution at a node: `e·ln(1 + w·d)`, its modulus, and its propagated error.
#[derive(Clone, Copy)]
struct NodeTerm {
    re: f64,
    im: f64,
    size: f64,
    propagated: f64,
}

impl std::ops::Add for NodeTerm {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
            size: self.size + other.size,
            propagated: self.propagated + other.propagated,
        }
    }
}

/// A term after scaling by the reference weight: `r = λ/λ_ref`, kept in logarithms so that
/// dynamic range beyond `f64` in `r` costs nothing.
#[derive(Clone, Copy)]
struct Scaled {
    /// `e = h/2`, the exponent of this term's factor.
    exponent: f64,
    kind: Kind,
    /// `ln|r|`.
    ln_ratio: f64,
    /// Magnitude of the logarithms `ln|r|` was formed from, which set its rounding.
    ln_ratio_scale: f64,
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    /// `λ = λ_ref`: `m = η` exactly.
    Reference,
    /// `0 < λ < λ_ref`: `m = q + r·η` with `q = 1 − r`, carried as `ln q`.
    Inner { ln_gap: f64 },
    /// `λ < 0`: `m = 1 + |r|·θ`.
    Negative,
}

/// How the saddle `c ∈ (0, s₊)` is parameterized. `θ = 2·λ_ref·c`.
#[derive(Clone, Copy, PartialEq)]
enum Chart {
    /// Some weight is positive, so `θ ∈ (0, 1)`: `y = logit θ`, which resolves both ends —
    /// `θ` near `0` (a saddle near the pole, as in a lower tail) and `η = 1 − θ` near `0` (the
    /// saddle pressed against `s₊`, as in a deep upper tail).
    Logit,
    /// No weight is positive (hence `x < 0`) and `c` is unbounded above: `y = ln θ`.
    Log,
}

/// The saddle-point quantities at one value of the parameter `y`.
struct Point {
    y: f64,
    ln_theta: f64,
    /// `θ·ξ = c·x`.
    theta_xi: f64,
    /// `ln m_k` with `m_k = 1 − r_k·θ`, the factor `K` scales by.
    ln_m: Vec<f64>,
    /// `ln|ρ_k|` with `ρ_k = θ·r_k/m_k = 2λ_k c/(1 − 2λ_k c)`.
    ln_rho: Vec<f64>,
    /// `c·ψ′(c) = Σ e_k ρ_k − θξ − 1`, whose sign is that of the saddle equation.
    slope: f64,
    /// `ln τ = −½ ln(1 + ½ Σ h_k ρ_k²) = −½ ln(c² ψ″(c))`, the saddle's width relative to `c`.
    ln_tau: f64,
}

/// One factor `(1 + w·d)^{−e}` of the integrand, the pole `1/s` among them.
struct Factor {
    weight: f64,
    ln_abs_weight: f64,
    exponent: f64,
    /// Relative error bound on the computed `weight`.
    weight_error: f64,
}

struct Problem {
    factors: Vec<Factor>,
    /// `a = θξτ`, the coefficient of the linear term `−a·d` of `Φ`.
    linear: f64,
    linear_error: f64,
    /// `+1` when the contour opens right (`x ≥ 0`), `−1` when it opens left.
    side: f64,
    /// Half the vertex curvature of the hyperbola `d = σ·g(t) + i·t`.
    alpha: f64,
    /// `ln(exp(K(c) − c·x)·τ/π)` less `ln J`, and the bound on its rounding before `ln J`.
    ln_prefix: f64,
    prefix_error_scale: f64,
    /// [`pairwise_depth`] of the sum over terms in the exponent, and of the sum over factors at
    /// a node.
    prefix_depth: usize,
    node_depth: usize,
    /// `Σ e_k` over all factors, which scales the rounding of `Φ`.
    total_exponent: f64,
}

impl Problem {
    fn new(active: &[WeightedChiSquareTerm], statistic: f64) -> Result<Self, TailProbability> {
        let largest_positive = active.iter().map(|term| term.weight).fold(0.0, f64::max);
        let chart = if largest_positive > 0.0 { Chart::Logit } else { Chart::Log };
        let reference = match chart {
            Chart::Logit => largest_positive,
            Chart::Log => active.iter().map(|term| -term.weight).fold(0.0, f64::max),
        };
        // `ξ = x/(2λ_ref)`, halved after the division so that a reference weight near
        // `f64::MAX` does not overflow `2λ_ref`.
        let xi = statistic / reference / 2.0;
        if !xi.is_finite() {
            // The statistic is beyond every weight's scale by more than `f64` spans: above them it
            // is an upper tail that underflows, below them nothing is resolved.
            return Err(if statistic > 0.0 {
                TailProbability { probability: 0.0, relative_error: 1.0 }
            } else {
                TailProbability { probability: 1.0, relative_error: f64::INFINITY }
            });
        }
        let ln_reference = reference.ln();
        let scaled: Vec<Scaled> = active
            .iter()
            .map(|term| {
                let ln_weight = term.weight.abs().ln();
                let kind = if term.weight == reference {
                    Kind::Reference
                } else if term.weight > 0.0 {
                    // `λ_ref − λ` is exact for `λ ≥ λ_ref/2` and one rounding otherwise.
                    Kind::Inner { ln_gap: ((reference - term.weight) / reference).ln() }
                } else {
                    Kind::Negative
                };
                Scaled {
                    exponent: 0.5 * term.degrees_of_freedom,
                    kind,
                    ln_ratio: ln_weight - ln_reference,
                    ln_ratio_scale: ln_weight.abs() + ln_reference.abs(),
                }
            })
            .collect();
        let total: f64 = scaled.iter().map(|term| term.exponent).sum();
        let point = saddle(chart, &scaled, xi, total);

        // Chernoff: `P ≤ E[e^{cQ}]·e^{−cx} = exp(K(c) − c·x)` for every `c` in the strip.
        let ln_mgf = -pairwise(0..scaled.len(), &|k| scaled[k].exponent * point.ln_m[k]);
        let ln_chernoff = ln_mgf - point.theta_xi;
        if ln_chernoff < (0.5 * SMALLEST_SUBNORMAL).ln() {
            return Err(TailProbability { probability: 0.0, relative_error: 1.0 });
        }

        let ln_tau = point.ln_tau;
        let tau = ln_tau.exp();
        let log_error = |scale: f64| accumulation_growth(COEFFICIENT_OPS) * (1.0 + scale);
        let mut factors = Vec::with_capacity(scaled.len() + 1);
        factors.push(Factor { weight: tau, ln_abs_weight: ln_tau, exponent: 1.0, weight_error: log_error(ln_tau.abs()) });
        for ((term, &ln_m), &ln_rho) in scaled.iter().zip(&point.ln_m).zip(&point.ln_rho) {
            let ln_abs_weight = ln_rho + ln_tau;
            // `w = −ρτ`: negative for a positive weight's branch point, positive otherwise.
            let sign = if term.kind == Kind::Negative { 1.0 } else { -1.0 };
            let scale = point.y.abs()
                + point.ln_theta.abs()
                + term.ln_ratio_scale
                + ln_m.abs()
                + ln_rho.abs()
                + ln_tau.abs()
                + match term.kind {
                    Kind::Inner { ln_gap } => ln_gap.abs(),
                    _ => 0.0,
                };
            factors.push(Factor {
                weight: sign * ln_abs_weight.exp(),
                ln_abs_weight,
                exponent: term.exponent,
                weight_error: log_error(scale),
            });
        }
        let linear = point.theta_xi * tau;
        let linear_error = if point.theta_xi == 0.0 {
            0.0
        } else {
            log_error(point.y.abs() + point.ln_theta.abs() + point.theta_xi.abs().ln().abs() + ln_tau.abs())
        };
        // The contour opens toward the side where `exp(−s·x)` decays, and its vertex curvature is
        // set by the nearest singularity on that side, which is the reference branch point for
        // `x ≥ 0` and the pole `s = 0` for `x < 0`; see `truncation` for why that is the one
        // whose distance fixes the bound.
        let side = if statistic >= 0.0 { 1.0 } else { -1.0 };
        let alpha = 0.5
            * factors
                .iter()
                .filter(|factor| side * factor.weight < 0.0)
                .map(|factor| factor.weight.abs())
                .fold(0.0, f64::max);
        let prefix_error_scale = 1.0
            + scaled.iter().zip(&point.ln_m).map(|(term, ln_m)| term.exponent * (1.0 + ln_m.abs())).sum::<f64>()
            + if point.theta_xi == 0.0 { 0.0 } else { point.theta_xi.abs() * (1.0 + point.theta_xi.abs().ln().abs()) }
            + ln_tau.abs();
        // One more addition joins the linear term `−a·d`.
        let node_depth = pairwise_depth(factors.len()) + 1;
        Ok(Self {
            factors,
            linear,
            linear_error,
            side,
            alpha,
            ln_prefix: ln_chernoff + ln_tau - std::f64::consts::PI.ln(),
            prefix_error_scale,
            prefix_depth: pairwise_depth(scaled.len()),
            node_depth,
            total_exponent: 1.0 + total,
        })
    }

    /// `g(t)` and `g′(t)` of the hyperbola `(g + 1/2α)² − t² = 1/(4α²)` through the vertex
    /// `d = 0`, written without cancellation near the vertex.
    fn hyperbola(&self, t: f64) -> (f64, f64) {
        let u = 2.0 * self.alpha * t;
        let root = u.hypot(1.0);
        (t * (u / (root + 1.0)), u / root)
    }

    /// The integrand `f(t) = Im(e^{Φ(d)}·d′(t))` at one node, the magnitude `|e^Φ|·|d′|` it is
    /// bounded by, and a bound on the relative error of `f` against that magnitude.
    fn node(&self, t: f64) -> (f64, f64, f64) {
        let (g, slope) = self.hyperbola(t);
        let (d_re, d_im) = (self.side * g, t);
        let d_abs = d_re.hypot(d_im);
        let linear_size = self.linear.abs() * d_abs;
        let logs = pairwise(0..self.factors.len(), &|k| {
            let factor = &self.factors[k];
            let (log_re, log_im, condition) = complex_log1p(factor.weight * d_re, factor.weight * d_im);
            NodeTerm {
                re: factor.exponent * log_re,
                im: factor.exponent * log_im,
                size: factor.exponent * log_re.hypot(log_im),
                propagated: factor.exponent * condition * (factor.weight_error + accumulation_growth(NODE_OPS)),
            }
        });
        let phi_re = -self.linear * d_re - logs.re;
        let phi_im = -self.linear * d_im - logs.im;
        let size = linear_size + logs.size;
        let propagated = linear_size * (self.linear_error + accumulation_growth(2)) + logs.propagated;
        let magnitude = phi_re.exp();
        let bound = magnitude * slope.hypot(1.0);
        if magnitude == 0.0 {
            return (0.0, 0.0, 0.0);
        }
        let value = magnitude * phi_im.cos() + magnitude * phi_im.sin() * self.side * slope;
        let phi_error =
            accumulation_growth(self.node_depth + NODE_OPS) * (self.total_exponent + size) + propagated;
        let error = phi_error.exp_m1() + accumulation_growth(NODE_OPS);
        (value, bound, error)
    }

    /// An upper bound on `∫_T^∞ |f(t)| dt` that also bounds the trapezoid sum `h·Σ F(v_k)` over
    /// the nodes past `V = asinh T`, or `∞` when neither holds.
    ///
    /// On the hyperbola `|1 + w·d| ≥ |w|·t` (its imaginary part) and `|1 + w·d| ≥ b`: `b = 1` for
    /// a singularity behind the vertex (`σw > 0`, where `Re(1 + w·d) ≥ 1`) and for the nearest one
    /// in front (`|w| = 2α`: the squared distance from the hyperbola to it is `1 + 2(X − 1)²` in
    /// units of its own distance), and `b = 1/√2` for any farther one in front (the same distance
    /// at `β = 2α/|w| ≥ 1` is at least `½ + 1/β − 1/(2β²) ≥ ½`). With `|d′| ≤ √2`, `|e^{−a·d}| =
    /// e^{−γ·g}` and every factor past `T` on whichever bound is larger at `T`,
    /// `|f| ≤ A·t^{−E}·e^{−γg(t)}`, which integrates to `A·T^{1−E}/(E − 1)` and, since `g` is
    /// convex, to at most `A·T^{−E}·e^{−γg(T)}/(γ·g′(T))`. The bound times `cosh v` decreases in
    /// `v` once `E ≥ 1` or `γ·g′(T)·cosh V ≥ 1`, which is what lets it stand for the sum.
    fn truncation(&self, t: f64, v: f64) -> f64 {
        let ln_t = t.ln();
        let mut ln_amplitude = 0.5 * std::f64::consts::LN_2;
        let mut power = 0.0;
        for factor in &self.factors {
            let floor = if self.side * factor.weight > 0.0 || factor.weight.abs() >= 2.0 * self.alpha {
                1.0
            } else {
                std::f64::consts::FRAC_1_SQRT_2
            };
            if factor.weight.abs() * t >= floor {
                ln_amplitude -= factor.exponent * factor.ln_abs_weight;
                power += factor.exponent;
            } else {
                ln_amplitude -= factor.exponent * floor.ln();
            }
        }
        let decay = self.linear.abs();
        let (g, slope) = self.hyperbola(t);
        if !(power >= 1.0 || decay * slope * v.cosh() >= 1.0) {
            return f64::INFINITY;
        }
        let algebraic = if power > 1.0 {
            ln_amplitude + (1.0 - power) * ln_t - (power - 1.0).ln()
        } else {
            f64::INFINITY
        };
        let exponential = if decay > 0.0 && slope > 0.0 {
            ln_amplitude - power * ln_t - decay * g - (decay * slope).ln()
        } else {
            f64::INFINITY
        };
        algebraic.min(exponential).exp()
    }

    fn tail(&self) -> TailProbability {
        // Past `t_max` the products `w·d` and `a·d` would leave `f64`.
        let reach = self.factors.iter().map(|factor| factor.weight.abs()).fold(1.0f64, f64::max);
        let t_max = f64::MAX / (4.0 * reach.max(self.alpha).max(self.linear.abs()));
        let v_max = t_max.asinh();

        // Level 0: step `1/2` in `v` — a starting point, since the halving below decides the
        // accuracy — extended until the tail beyond the last node is below the rounding of the sum.
        let mut step = 0.5;
        let (origin, origin_bound, origin_error) = self.node(0.0);
        let mut sum = 0.5 * origin;
        let mut noise = 0.5 * origin_bound * origin_error;
        let mut magnitude = 0.5 * origin_bound;
        let mut nodes = 1usize;
        let mut last_v = 0.0;
        let mut truncation = f64::INFINITY;
        for k in 1u32.. {
            let v = f64::from(k) * step;
            if v > v_max {
                break;
            }
            let (value, bound, error) = self.node(v.sinh());
            let weight = v.cosh();
            sum += value * weight;
            noise += bound * weight * error;
            magnitude += bound * weight;
            nodes += 1;
            last_v = v;
            truncation = self.truncation(v.sinh(), v);
            if truncation <= UNIT_ROUNDOFF * (step * sum).abs() {
                break;
            }
        }
        if !truncation.is_finite() {
            return self.assemble(step * sum, f64::INFINITY);
        }
        let floor = |noise: f64, magnitude: f64, nodes: usize, step: f64| {
            step * (noise + accumulation_growth(nodes) * magnitude)
        };
        let mut integral = step * sum;
        let mut previous_floor = floor(noise, magnitude, nodes, step);
        let mut previous_difference = f64::INFINITY;
        let mut difference = f64::INFINITY;
        let mut current_floor = previous_floor;
        for level in 1.. {
            step *= 0.5;
            if step < last_v * UNIT_ROUNDOFF {
                break;
            }
            let mut v = step;
            while v <= last_v {
                let (value, bound, error) = self.node(v.sinh());
                let weight = v.cosh();
                sum += value * weight;
                noise += bound * weight * error;
                magnitude += bound * weight;
                nodes += 1;
                v += 2.0 * step;
            }
            let refined = step * sum;
            difference = (refined - integral).abs();
            current_floor = floor(noise, magnitude, nodes, step);
            integral = refined;
            if difference <= current_floor + previous_floor {
                break;
            }
            if level >= 2 && difference >= previous_difference {
                break;
            }
            previous_difference = difference;
            previous_floor = current_floor;
        }
        self.assemble(integral, difference + truncation + current_floor)
    }

    fn assemble(&self, integral: f64, integral_error: f64) -> TailProbability {
        if !(integral > 0.0) {
            return TailProbability { probability: 0.0, relative_error: 1.0 };
        }
        let relative = integral_error / integral;
        let ln_integral = integral.ln();
        let exponent_error = accumulation_growth(self.prefix_depth + PREFIX_OPS)
            * (self.prefix_error_scale + (ln_integral - std::f64::consts::PI.ln()).abs());
        let log_error = if relative < 1.0 {
            exponent_error + UNIT_ROUNDOFF - (-relative).ln_1p()
        } else {
            f64::INFINITY
        };
        let probability = (self.ln_prefix + ln_integral).exp().min(1.0);
        if probability == 0.0 {
            return TailProbability { probability: 0.0, relative_error: 1.0 };
        }
        let mut relative_error = log_error.exp_m1();
        if probability < f64::MIN_POSITIVE {
            // A subnormal result is rounded to an absolute half-spacing, not a relative one.
            let half = 0.5 * SMALLEST_SUBNORMAL;
            relative_error += half * log_error.exp() / (probability - half);
        }
        TailProbability { probability, relative_error }
    }
}

/// `ln(1 + z)` for complex `z`, and the factor `|z|(2 + |z|)/|1 + z|²` by which a relative
/// perturbation of `z`, or the rounding of `|1 + z|² − 1 = z_re(2 + z_re) + z_im²`, reaches it.
fn complex_log1p(re: f64, im: f64) -> (f64, f64, f64) {
    let shift = re * (2.0 + re) + im * im;
    let real = if shift.abs() <= 1.0 { 0.5 * shift.ln_1p() } else { (1.0 + re).hypot(im).ln() };
    let abs = re.hypot(im);
    let one_plus = (1.0 + re).hypot(im);
    (real, im.atan2(1.0 + re), abs * (2.0 + abs) / (one_plus * one_plus))
}

/// Bisection for the saddle in the chart's parameter, stopped once the bracket is inside the
/// saddle's own width `τ` (in relative units of `c`), where moving `c` further changes only how
/// concentrated the integrand is, not its integral.
fn saddle(chart: Chart, scaled: &[Scaled], xi: f64, total: f64) -> Point {
    let (mut low, mut high) = match chart {
        Chart::Logit => {
            let (mut positive, mut negative, mut reference) = (0.0, 0.0, 0.0);
            for term in scaled {
                match term.kind {
                    Kind::Reference => {
                        positive += term.exponent;
                        reference += term.exponent;
                    }
                    Kind::Inner { .. } => positive += term.exponent,
                    Kind::Negative => negative += term.exponent,
                }
            }
            // At `low` every positive `ρ ≤ e^y` and `−θξ ≤ θ·max(0, −ξ)`, so the slope is `≤ 0`; at
            // `high` the reference terms alone give `1 + H₋/2 + max(ξ, 0)`, which outweighs every
            // negative `ρ > −1` and `θξ < max(ξ, 0)`, so it is `≥ 0`.
            let low = -if xi < 0.0 { logaddexp(positive.ln(), (-xi).ln()) } else { positive.ln() };
            let base = (1.0 + negative).ln();
            let high = if xi > 0.0 { logaddexp(base, xi.ln()) } else { base } - reference.ln();
            (low, high)
        }
        Chart::Log => {
            // `θ|ξ| = 1` leaves the negative `Σ e ρ` to make the slope negative, and `θ|ξ| = 1 + H/2`
            // outweighs it since every `ρ > −1`.
            let ln_xi = (-xi).ln();
            (-ln_xi, (1.0 + total).ln() - ln_xi)
        }
    };
    loop {
        let mid = 0.5 * (low + high);
        let point = evaluate(chart, scaled, xi, mid);
        let width = match chart {
            Chart::Logit => (-softplus(mid)).exp() * (high - low),
            Chart::Log => high - low,
        };
        if mid == low || mid == high || width <= point.ln_tau.exp() {
            return point;
        }
        if point.slope < 0.0 {
            low = mid;
        } else {
            high = mid;
        }
    }
}

fn evaluate(chart: Chart, scaled: &[Scaled], xi: f64, y: f64) -> Point {
    let (ln_theta, ln_eta) = match chart {
        Chart::Logit => (-softplus(-y), -softplus(y)),
        Chart::Log => (y, f64::NAN),
    };
    let theta_xi = match chart {
        Chart::Logit => ln_theta.exp() * xi,
        Chart::Log => -(y + (-xi).ln()).exp(),
    };
    let mut ln_m = Vec::with_capacity(scaled.len());
    let mut ln_rho = Vec::with_capacity(scaled.len());
    let mut slope = -theta_xi - 1.0;
    for term in scaled {
        let (m, rho) = match term.kind {
            Kind::Reference => (ln_eta, y),
            Kind::Inner { ln_gap } => {
                let m = logaddexp(ln_gap, term.ln_ratio + ln_eta);
                (m, ln_theta + term.ln_ratio - m)
            }
            Kind::Negative => {
                let z = term.ln_ratio + ln_theta;
                (softplus(z), -softplus(-z))
            }
        };
        let sign = if term.kind == Kind::Negative { -1.0 } else { 1.0 };
        slope += sign * term.exponent * rho.exp();
        ln_m.push(m);
        ln_rho.push(rho);
    }
    if chart == Chart::Logit && ln_eta.exp() == 0.0 {
        // The saddle is closer to `s₊` than `f64` resolves; the slope there is positive.
        slope = f64::INFINITY;
    }
    if slope.is_nan() {
        slope = f64::INFINITY;
    }
    // `½ Σ h ρ² = Σ e ρ²`, scaled by its largest term so that `ρ` beyond `f64` costs nothing.
    let largest = ln_rho.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let ln_tau = if largest == f64::NEG_INFINITY {
        0.0
    } else {
        let scaled_sum: f64 =
            scaled.iter().zip(&ln_rho).map(|(term, rho)| term.exponent * (2.0 * (rho - largest)).exp()).sum();
        -0.5 * softplus(scaled_sum.ln() + 2.0 * largest)
    };
    Point { y, ln_theta, theta_xi, ln_m, ln_rho, slope, ln_tau }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probability::{chi_square_sf, fisher_snedecor_sf};

    fn term(weight: f64, degrees_of_freedom: f64) -> WeightedChiSquareTerm {
        WeightedChiSquareTerm { weight, degrees_of_freedom }
    }

    /// The identity the signed form exists for, against the regularized incomplete beta:
    /// `P(F_{a,b} > f) = P(χ²_a − (f·a/b)·χ²_b > 0)`. The returned bound must hold against the
    /// beta evaluation, whose own error is a few ulps, and must be tight.
    #[test]
    fn the_f_tail_is_the_two_term_signed_combination_at_zero() {
        for &(a, b) in &[(1.0_f64, 5.0_f64), (2.0, 17.0), (3.0, 26.0), (0.7, 24.0), (5.4, 191.0), (11.0, 4.0)] {
            for &f in &[0.05_f64, 0.5, 1.0, 2.5, 9.0, 40.0] {
                let tail = signed_weighted_chi_square_sf(&[term(1.0, a), term(-f * a / b, b)], 0.0);
                let want = fisher_snedecor_sf(f, a, b);
                assert!(tail.relative_error <= 1e-10, "F({a},{b}) at {f}: {tail:?}");
                assert!(
                    (tail.probability - want).abs() <= (tail.relative_error + 1e-12) * want,
                    "F({a},{b}) at {f}: {tail:?} against {want}",
                );
            }
        }
    }

    /// One positive term is a scaled chi-square; the bound must hold against `chi_square_sf`.
    #[test]
    fn a_single_positive_term_is_a_scaled_chi_square() {
        for &(weight, h) in &[(1.0_f64, 1.0_f64), (0.3, 4.0), (7.5, 0.4), (1e-12, 30.0)] {
            for &x in &[0.1_f64, 1.0, 10.0, 200.0] {
                let statistic = weight * x;
                let tail = signed_weighted_chi_square_sf(&[term(weight, h)], statistic);
                let want = chi_square_sf(x, h);
                assert!(tail.relative_error <= 1e-10, "{weight}·χ²_{h} at {statistic}: {tail:?}");
                assert!(
                    (tail.probability - want).abs() <= (tail.relative_error + 1e-12) * want,
                    "{weight}·χ²_{h} at {statistic}: {tail:?} against {want}",
                );
            }
        }
    }

    #[test]
    fn the_support_decides_the_degenerate_cases_exactly() {
        let positive = [term(1.0, 2.0), term(0.5, 1.0)];
        let negative = [term(-1.0, 2.0), term(-0.5, 1.0)];
        assert_eq!(signed_weighted_chi_square_sf(&positive, 0.0), TailProbability::exact(1.0));
        assert_eq!(signed_weighted_chi_square_sf(&positive, -3.0), TailProbability::exact(1.0));
        assert_eq!(signed_weighted_chi_square_sf(&negative, 0.0), TailProbability::exact(0.0));
        assert_eq!(signed_weighted_chi_square_sf(&[term(0.0, 3.0)], -1.0), TailProbability::exact(1.0));
        assert_eq!(signed_weighted_chi_square_sf(&[], 0.0), TailProbability::exact(0.0));
        let mixed = [term(1.0, 2.0), term(-0.5, 1.0)];
        assert_eq!(signed_weighted_chi_square_sf(&mixed, f64::INFINITY), TailProbability::exact(0.0));
        assert_eq!(signed_weighted_chi_square_sf(&mixed, f64::NEG_INFINITY), TailProbability::exact(1.0));
        for invalid in [
            signed_weighted_chi_square_sf(&mixed, f64::NAN),
            signed_weighted_chi_square_sf(&[term(f64::INFINITY, 1.0)], 1.0),
            signed_weighted_chi_square_sf(&[term(1.0, 0.0)], 1.0),
        ] {
            assert!(invalid.probability.is_nan() && invalid.relative_error.is_nan(), "{invalid:?}");
        }
    }

    /// `absolute_error` is the binding side of `P ∈ [p/(1 + r), p/(1 − r)]`, and an unresolved
    /// bound leaves only `P ∈ [0, 1]`.
    #[test]
    fn the_absolute_error_is_the_binding_side_of_the_relative_bound() {
        let tail = TailProbability { probability: 0.25, relative_error: 0.2 };
        assert_eq!(tail.absolute_error(), 0.2 * 0.25 / 0.8);
        assert_eq!(TailProbability { probability: 0.0, relative_error: 1.0 }.absolute_error(), 1.0);
        assert_eq!(TailProbability::exact(0.5).absolute_error(), 0.0);
        assert!(TailProbability { probability: f64::NAN, relative_error: f64::NAN }.absolute_error().is_nan());
    }
}
