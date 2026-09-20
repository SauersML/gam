//! #2946 acceptance A1 for the exact GELU `σ(t) = t Φ(t)`: the smoothing
//! `T_v σ` with its t-derivatives, the pair kernel `K = E[σ(X) σ(Y)]` and
//! Price's derivative `∂_r K = E[σ'(X) σ'(Y)]` of
//! `gam_math::gaussian_activation`, with and without means. Each is checked
//! against an oracle whose error is bounded a priori, and against the closed
//! forms of a separate derivation (#2946 comment 5716000828). The laws reach
//! the tails, `v → 0`, a zero variance, and `|r| → √(vw)`, where
//! `ρ = r/√((1 + v)(1 + w))` approaches `±1`.
//!
//! # The oracle
//!
//! Every expectation is `∫∫ φ(z₁) φ(z₂) f(X, Y) dz₁ dz₂` with the standardized
//! pair `X = b + p₁₁ z₁`, `Y = c + p₂₁ z₁ + p₂₂ z₂` and an entire `f`. The oracle
//! is the tensor trapezoidal rule truncated to `|zᵢ| ≤ Lᵢ`, and its error has
//! four parts, each bounded in closed form:
//!
//! - **Aliasing.** For `w` analytic in the strip `|Im z| < a`, decaying there,
//!   with `∫ |w(x + iy)| dx ≤ M` for every `|y| < a`, the untruncated rule of
//!   step `h` errs by at most `2M/(e^{2πa/h} − 1)` (Trefethen and Weideman,
//!   SIAM Review 56 (2014), Theorem 5.1). The tensor error splits as
//!   `(I₁ − T₁) I₂ + T₁ (I₂ − T₂)`, one strip per axis.
//! - **Truncation.** `xᵐ φ(x)` decreases beyond `x = √m`, so the dropped nodes
//!   of a polynomial majorant sum to at most `2 ∫_L^∞ xᵐ φ(x) dx`.
//! - **The integrated law.** The rounded Cholesky factors integrate a law whose
//!   variances and covariance differ from the requested ones by fused residuals
//!   the oracle measures. The heat equation `∂_v E f(X) = ½ E f''(X)` and
//!   Price's theorem turn those shifts into bounds.
//! - **Rounding.** Each node's first-order rounding is bounded from its own
//!   values, and compensated summation bounds the sum.
//!
//! On the strip, `|φ(ξ + iη)| = φ(ξ) e^{η²/2}`,
//! `|Φ(ξ + iη)| ≤ 1 + φ(0) |η| e^{η²/2}` and `|He_m(ζ)| ≤ |He|_m(|ζ|)`, the
//! Hermite polynomial with its coefficients' magnitudes. These give polynomial
//! majorants in `|Re ζ| + |Im ζ|` times one factor `e^{η²/2}` per pre-activation.
//! A majorant growing like `e^{a² S/2}` balances against `e^{2πa/h}` at
//! `a = √(2D/S)`, `h = π √(2/(S D))`, where the aliasing bound is `2 e^{−D}` times
//! polynomial moments. `D` sizes the rule, not the tolerance: the tolerance is
//! the bound the rule then carries.
//!
//! # The closed forms' contract
//!
//! The API is held to the first-order rounding of a stable evaluation of its
//! closed form: an operation count times `ε` on the summed magnitudes of the
//! form's terms, plus the conditioning of each special function on its rounded
//! standardized argument, plus `Φ₂`'s owner per-call bound times its coefficient. A
//! form that amplifies rounding through an ill-conditioned intermediate, such
//! as `arccos(−ρ)` from a rounded `ρ` as `ρ → −1`, is outside that contract.

use gam_math::bivariate_normal::bivariate_normal_cdf_with_complement;
use gam_math::gaussian_activation::{
    GaussianActivation, PairKernel, PreactivationPair, gaussian_smoothing_derivatives, pair_kernel,
};
use gam_math::probability::{normal_cdf, normal_pdf};
use std::f64::consts::{PI, TAU};

const EPSILON: f64 = f64::EPSILON;

/// The aliasing balance `a² S/2 = D`, `2πa/h = 2D` that sizes each rule.
const ALIASING_NEPERS: f64 = 60.0;

/// Operations along the longest path of an elementary closed form, each adding
/// at most one ulp of relative error to first order. The zero-mean pair kernel
/// chains `1 + v` and `1 + w` (2), the residual `vw − r²` with its two
/// exact-product corrections (6), the discriminant (3), its square root (1),
/// `atan2` within one ulp (1) and its scale (1), the density (2), two
/// reciprocals and their sum (3), the coupling (3) and the final products and
/// sums (5): 27.
const CLOSED_FORM_OPERATIONS: f64 = 32.0;

/// Operations of a path that evaluates `Φ` or `φ`, each adding at most one ulp
/// to first order. libm-0.2.16's `erfc` takes an argument transform (2), two
/// Horner polynomials of degree at most 8 (32), their quotient (1), `exp` of a
/// split square (at most 8; `exp` itself errs below one ulp, `exp.rs`) and a
/// final scale (3): 46, with rational-fit errors below 2^-57.9 (`erf.rs`
/// regions 2-4). A form around it adds its own standardization, products and
/// Hermite recurrence of three operations per degree up to degree 6 (18): 64.
const SPECIAL_FUNCTION_OPERATIONS: f64 = 64.0;

/// Derivative orders `σ⁽⁰⁾ … σ⁽⁶⁾` the oracle integrands reach.
const ACTIVATION_ORDERS: usize = 7;

/// `E|Z|ᵐ` for `Z ~ N(0, 1)`: `(m − 1)!!`, times `√(2/π)` for odd `m`.
fn absolute_moment(order: usize) -> f64 {
    let mut value = if order % 2 == 0 { 1.0 } else { (2.0 / PI).sqrt() };
    let mut factor = order as f64 - 1.0;
    while factor > 0.0 {
        value *= factor;
        factor -= 2.0;
    }
    value
}

/// `max_x xᵐ φ(x) = mᵐᐟ² e^{−m/2}/√(2π)`, attained at `x = √m`.
fn peak_weighted_density(order: usize) -> f64 {
    let degree = order as f64;
    degree.powf(0.5 * degree) * (-0.5 * degree).exp() / TAU.sqrt()
}

/// `h Σ_j |jh|ᵐ φ(jh) ≤ E|Z|ᵐ + 3h max xᵐ φ(x)`. On each half-line the summand is
/// unimodal, and a unimodal sum exceeds its integral by at most one peak; the
/// node at zero adds one more.
fn discrete_moment(order: usize, step: f64) -> f64 {
    absolute_moment(order) + 3.0 * step * peak_weighted_density(order)
}

/// `2 ∫_L^∞ xᵐ φ(x) dx` for `L ≥ √m`, by
/// `∫_L^∞ xᵐ φ = L^{m−1} φ(L) + (m − 1) ∫_L^∞ x^{m−2} φ`.
fn tail_moment(order: usize, reach: f64) -> f64 {
    let mut lower = normal_cdf(-reach);
    if order == 0 {
        return 2.0 * lower;
    }
    let mut upper = normal_pdf(reach);
    for degree in 2..=order {
        let next = reach.powi(degree as i32 - 1) * normal_pdf(reach) + (degree as f64 - 1.0) * lower;
        lower = upper;
        upper = next;
    }
    2.0 * upper
}

fn factorial(count: usize) -> f64 {
    (1..=count).map(|index| index as f64).product()
}

/// The coefficients of `|He|_m`, low order first:
/// `|He|_{m+1}(x) = x |He|_m(x) + m |He|_{m−1}(x)`.
fn absolute_hermite_coefficients(degree: usize) -> Vec<f64> {
    let mut previous = vec![0.0; degree + 2];
    let mut current = vec![0.0; degree + 2];
    current[0] = 1.0;
    for order in 0..degree {
        let mut next = vec![0.0; degree + 2];
        for index in 0..=order {
            next[index + 1] += current[index];
        }
        for (slot, lower) in next.iter_mut().zip(&previous) {
            *slot += order as f64 * lower;
        }
        previous = current;
        current = next;
    }
    current.truncate(degree + 1);
    current
}

fn absolute_hermite(degree: usize, x: f64) -> f64 {
    absolute_hermite_coefficients(degree)
        .iter()
        .rev()
        .fold(0.0, |accumulated, coefficient| accumulated * x.abs() + coefficient)
}

/// A polynomial in `(s, t) = (|z₁|, |z₂|)` with nonnegative coefficients
/// `c[i][j]` of `sⁱ tʲ`.
#[derive(Clone, Debug)]
struct Majorant {
    coefficients: Vec<Vec<f64>>,
}

impl Majorant {
    /// `Σ_m p_m (α + β s + γ t)ᵐ` for nonnegative `p`, `α`, `β` and `γ`.
    fn of_affine(polynomial: &[f64], alpha: f64, beta: f64, gamma: f64) -> Self {
        let degree = polynomial.len() - 1;
        let mut coefficients = vec![vec![0.0; degree + 1]; degree + 1];
        for (order, coefficient) in polynomial.iter().enumerate() {
            for first in 0..=order {
                for second in 0..=(order - first) {
                    let constant = order - first - second;
                    let multinomial =
                        factorial(order) / (factorial(first) * factorial(second) * factorial(constant));
                    coefficients[first][second] += coefficient
                        * multinomial
                        * beta.powi(first as i32)
                        * gamma.powi(second as i32)
                        * alpha.powi(constant as i32);
                }
            }
        }
        Self { coefficients }
    }

    fn product(&self, other: &Self) -> Self {
        let rows = self.coefficients.len() + other.coefficients.len() - 1;
        let columns = self.coefficients[0].len() + other.coefficients[0].len() - 1;
        let mut coefficients = vec![vec![0.0; columns]; rows];
        for (first, row) in self.coefficients.iter().enumerate() {
            for (second, left) in row.iter().enumerate() {
                for (other_first, other_row) in other.coefficients.iter().enumerate() {
                    for (other_second, right) in other_row.iter().enumerate() {
                        coefficients[first + other_first][second + other_second] += left * right;
                    }
                }
            }
        }
        Self { coefficients }
    }

    /// `Σ c[i][j] μ(i) ν(j)` for moment sequences `μ` of `s` and `ν` of `t`.
    fn integrate(&self, first: impl Fn(usize) -> f64, second: impl Fn(usize) -> f64) -> f64 {
        self.coefficients
            .iter()
            .enumerate()
            .map(|(row, entries)| {
                entries
                    .iter()
                    .enumerate()
                    .map(|(column, coefficient)| coefficient * first(row) * second(column))
                    .sum::<f64>()
            })
            .sum()
    }
}

/// A polynomial `P` with `|σ⁽ᵏ⁾(ζ)| ≤ P(|Re ζ| + η)` wherever `|Im ζ| ≤ η`.
fn activation_majorant(order: usize, imaginary_reach: f64) -> Vec<f64> {
    let growth = (0.5 * imaginary_reach * imaginary_reach).exp();
    let density_bound = normal_pdf(0.0) * growth;
    let probability_bound = 1.0 + normal_pdf(0.0) * imaginary_reach * growth;
    match order {
        0 => vec![0.0, probability_bound],
        1 => vec![probability_bound, density_bound],
        _ => {
            let lower = absolute_hermite_coefficients(order - 2);
            absolute_hermite_coefficients(order)
                .iter()
                .enumerate()
                .map(|(index, upper)| density_bound * (upper + lower.get(index).copied().unwrap_or(0.0)))
                .collect()
        }
    }
}

/// `(σ⁽ᵏ⁾(x), m_k(x))` for `k < count ≤ ACTIVATION_ORDERS`, where `m_k` sums the
/// magnitudes of the terms: `σ = x Φ`, `σ' = Φ + x φ`, and
/// `σ⁽ᵏ⁾ = (−1)ᵏ φ (He_{k−2} − He_k)` for `k ≥ 2`.
fn activation_derivatives(x: f64, count: usize) -> [(f64, f64); ACTIVATION_ORDERS] {
    let density = normal_pdf(x);
    let probability = normal_cdf(x);
    let magnitude = x.abs();
    let mut derivatives = [(0.0, 0.0); ACTIVATION_ORDERS];
    derivatives[0] = (x * probability, magnitude * probability);
    derivatives[1] = (probability + x * density, probability + magnitude * density);
    let mut hermite = [0.0; ACTIVATION_ORDERS + 1];
    let mut absolute = [0.0; ACTIVATION_ORDERS + 1];
    hermite[0] = 1.0;
    absolute[0] = 1.0;
    hermite[1] = x;
    absolute[1] = magnitude;
    for degree in 1..count {
        hermite[degree + 1] = x * hermite[degree] - degree as f64 * hermite[degree - 1];
        absolute[degree + 1] = magnitude * absolute[degree] + degree as f64 * absolute[degree - 1];
    }
    for order in 2..count {
        let sign = if order % 2 == 0 { 1.0 } else { -1.0 };
        derivatives[order] = (
            sign * density * (hermite[order - 2] - hermite[order]),
            density * (absolute[order - 2] + absolute[order]),
        );
    }
    derivatives
}

/// First-order rounding of `σ⁽ᵏ⁾` at the rounded `x`: its operations on the
/// terms' magnitudes, plus `Φ(x) = ½ erfc(−x/√2)` reading a rounded quotient,
/// which moves `Φ` by at most `2ε |x| φ(x) ≤ 2ε m₁(x)`.
fn integrand_rounding(x: f64, derivatives: &[(f64, f64); ACTIVATION_ORDERS], order: usize) -> f64 {
    let operations = EPSILON * SPECIAL_FUNCTION_OPERATIONS * derivatives[order].1;
    match order {
        0 => operations + 2.0 * EPSILON * x.abs() * derivatives[1].1,
        1 => operations + 2.0 * EPSILON * derivatives[1].1,
        _ => operations,
    }
}

/// One trapezoidal axis: step `h`, the strip half-width `a` of its aliasing
/// bound, and `2J + 1` nodes `jh` reaching `L = Jh`.
struct Axis {
    step: f64,
    strip: f64,
    half_count: usize,
    reach: f64,
}

impl Axis {
    /// `growth` is `S` in the majorant's `e^{a² S/2}`; `degree` is the majorant's
    /// polynomial degree on this axis, whose tail bound needs `L ≥ √degree`.
    fn new(growth: f64, degree: usize) -> Self {
        let step = PI * (2.0 / (growth * ALIASING_NEPERS)).sqrt();
        let strip = (2.0 * ALIASING_NEPERS / growth).sqrt();
        let minimum_reach = (2.0 * ALIASING_NEPERS).sqrt().max((degree as f64).sqrt());
        let half_count = (minimum_reach / step).ceil() as usize;
        Self {
            step,
            strip,
            half_count,
            reach: half_count as f64 * step,
        }
    }

    fn count(&self) -> usize {
        2 * self.half_count + 1
    }

    fn node(&self, index: usize) -> f64 {
        (index as f64 - self.half_count as f64) * self.step
    }

    /// `2/(e^{2πa/h} − 1)`, Theorem 5.1's factor on `M`.
    fn aliasing_factor(&self) -> f64 {
        2.0 / (TAU * self.strip / self.step).exp_m1()
    }

    /// `e^{a²/2}`, the standard density's growth across the strip.
    fn density_growth(&self) -> f64 {
        (0.5 * self.strip * self.strip).exp()
    }
}

/// Neumaier's compensated sum with a running first-order rounding bound.
/// TwoSum errors are exact; recursively summing the `n` corrections errs by at
/// most `(n − 1) ε` times their summed magnitudes, themselves at most
/// `n ε Σ|term|/2`, and the final addition by `ε |sum|`.
#[derive(Clone, Copy, Debug, Default)]
struct CompensatedSum {
    sum: f64,
    correction: f64,
    absolute: f64,
    node_rounding: f64,
    count: usize,
}

impl CompensatedSum {
    fn add(&mut self, term: f64, rounding: f64) {
        let total = self.sum + term;
        self.correction += if self.sum.abs() >= term.abs() {
            (self.sum - total) + term
        } else {
            (term - total) + self.sum
        };
        self.sum = total;
        self.absolute += term.abs();
        self.node_rounding += rounding;
        self.count += 1;
    }

    fn value(&self) -> f64 {
        self.sum + self.correction
    }

    fn rounding(&self) -> f64 {
        let count = self.count as f64;
        self.node_rounding
            + EPSILON * self.value().abs()
            + count * count * EPSILON * EPSILON * self.absolute
    }
}

/// An oracle value with its four derived error bounds.
#[derive(Clone, Copy, Debug)]
struct Oracle {
    value: f64,
    aliasing: f64,
    truncation: f64,
    law: f64,
    rounding: f64,
}

impl Oracle {
    fn bound(&self) -> f64 {
        self.aliasing + self.truncation + self.law + self.rounding
    }
}

/// `E σ⁽ᵏ⁾(t + √v Z)` by the bounded trapezoidal rule.
fn smoothing_oracle(t: f64, variance: f64, order: usize) -> Oracle {
    let scale = variance.sqrt();
    let axis = Axis::new(1.0 + variance, order.max(1));
    let mut sum = CompensatedSum::default();
    for index in 0..axis.count() {
        let z = axis.node(index);
        let weight = axis.step * normal_pdf(z);
        let x = t + scale * z;
        let derivatives = activation_derivatives(x, order + 2);
        let weight_relative = EPSILON * (6.0 + z * z);
        let argument = EPSILON * (t.abs() + 3.0 * scale * z.abs());
        let rounding = weight
            * (weight_relative * derivatives[order].1
                + integrand_rounding(x, &derivatives, order)
                + derivatives[order + 1].1 * argument);
        sum.add(weight * derivatives[order].0, rounding);
    }
    let reach = scale * axis.strip;
    let aliasing = axis.aliasing_factor()
        * axis.density_growth()
        * Majorant::of_affine(&activation_majorant(order, reach), t.abs() + reach, scale, 0.0)
            .integrate(absolute_moment, absolute_moment);
    let real = |derivative_order: usize| {
        Majorant::of_affine(&activation_majorant(derivative_order, 0.0), t.abs(), scale, 0.0)
    };
    let truncation = real(order).integrate(|degree| tail_moment(degree, axis.reach), absolute_moment);
    // The rule integrates the variance (√v)², within the exactly rounded fused
    // residual (√v)² − v; ½ E σ⁽ᵏ⁺²⁾ is the smoothing's variance derivative.
    let variance_shift = scale.mul_add(scale, -variance).abs() * (1.0 + EPSILON);
    let law = 0.5 * variance_shift * real(order + 2).integrate(absolute_moment, absolute_moment);
    Oracle {
        value: sum.value(),
        aliasing,
        truncation,
        law,
        rounding: sum.rounding(),
    }
}

/// `K = E[σ(X) σ(Y)]` and `∂_r K = E[σ'(X) σ'(Y)]` by the bounded tensor rule.
struct PairOracles {
    kernel: Oracle,
    derivative: Oracle,
}

fn pair_oracles(
    mean_x: f64,
    mean_y: f64,
    variance_x: f64,
    variance_y: f64,
    covariance: f64,
) -> PairOracles {
    let lead = variance_x.sqrt();
    let coupling = if variance_x > 0.0 { covariance / lead } else { 0.0 };
    let conditional = (variance_y - coupling * coupling).max(0.0);
    let residual = conditional.sqrt();
    // X reads z₁ through `lead` and Y through `coupling`, so the strip on z₁
    // grows like e^{a²(1 + lead² + coupling²)/2}; Y alone reads z₂.
    let first = Axis::new(1.0 + variance_x + coupling * coupling, 2);
    let second = Axis::new(1.0 + conditional, 1);
    let second_weights = (0..second.count())
        .map(|index| {
            let z = second.node(index);
            (z, second.step * normal_pdf(z))
        })
        .collect::<Vec<_>>();
    let mut kernel = CompensatedSum::default();
    let mut derivative = CompensatedSum::default();
    for index in 0..first.count() {
        let first_node = first.node(index);
        let first_weight = first.step * normal_pdf(first_node);
        let x = mean_x + lead * first_node;
        let activation_x = activation_derivatives(x, 3);
        let argument_x = EPSILON * (mean_x.abs() + 3.0 * lead * first_node.abs());
        let rounding_x = [
            integrand_rounding(x, &activation_x, 0) + activation_x[1].1 * argument_x,
            integrand_rounding(x, &activation_x, 1) + activation_x[2].1 * argument_x,
        ];
        let shifted_y = mean_y + coupling * first_node;
        for (second_node, second_weight) in &second_weights {
            let weight = first_weight * second_weight;
            let y = shifted_y + residual * second_node;
            let activation_y = activation_derivatives(y, 3);
            let argument_y = EPSILON
                * (mean_y.abs() + 3.0 * (coupling * first_node).abs() + 3.0 * residual * second_node.abs());
            let rounding_y = [
                integrand_rounding(y, &activation_y, 0) + activation_y[1].1 * argument_y,
                integrand_rounding(y, &activation_y, 1) + activation_y[2].1 * argument_y,
            ];
            let weight_relative = EPSILON * (12.0 + first_node * first_node + second_node * second_node);
            for (accumulator, order) in [(&mut kernel, 0), (&mut derivative, 1)] {
                let term = weight * activation_x[order].0 * activation_y[order].0;
                let rounding = weight
                    * (weight_relative * activation_x[order].1 * activation_y[order].1
                        + rounding_x[order] * activation_y[order].1
                        + activation_x[order].1 * rounding_y[order]);
                accumulator.add(term, rounding);
            }
        }
    }
    let majorant = |order_x: usize, reach_x: f64, order_y: usize, reach_y: f64| {
        Majorant::of_affine(&activation_majorant(order_x, reach_x), mean_x.abs() + reach_x, lead, 0.0).product(
            &Majorant::of_affine(
                &activation_majorant(order_y, reach_y),
                mean_y.abs() + reach_y,
                coupling.abs(),
                residual,
            ),
        )
    };
    let aliasing = |order: usize| {
        // (I₁ − T₁) I₂: the strip on z₁ moves X by `lead a₁` and Y by `|coupling| a₁`.
        let first_strip = first.aliasing_factor()
            * first.density_growth()
            * majorant(order, lead * first.strip, order, coupling.abs() * first.strip)
                .integrate(absolute_moment, absolute_moment);
        // T₁ (I₂ − T₂): the strip on z₂ moves Y alone, and z₁ is summed over nodes.
        let second_strip = second.aliasing_factor()
            * second.density_growth()
            * majorant(order, 0.0, order, residual * second.strip)
                .integrate(|degree| discrete_moment(degree, first.step), absolute_moment);
        first_strip + second_strip
    };
    let truncation = |order: usize| {
        let real = majorant(order, 0.0, order, 0.0);
        real.integrate(
            |degree| tail_moment(degree, first.reach),
            |degree| discrete_moment(degree, second.step),
        ) + real.integrate(
            |degree| discrete_moment(degree, first.step),
            |degree| tail_moment(degree, second.reach),
        )
    };
    // The rule integrates variances lead² and coupling² + residual² and the
    // covariance lead·coupling. Each shift from the requested law is an exactly
    // rounded fused residual, widened by one ulp for its own rounding and, for
    // w, the inner residual's. ∂_v K = ½ E[σ''(X) σ(Y)], ∂_w K = ½ E[σ(X) σ''(Y)]
    // and ∂_r K = E[σ'(X) σ'(Y)], one order higher throughout for ∂_r K.
    let shift_v = lead.mul_add(lead, -variance_x).abs() * (1.0 + EPSILON);
    let shift_r = lead.mul_add(coupling, -covariance).abs() * (1.0 + EPSILON);
    let inner = coupling.mul_add(coupling, -variance_y);
    let outer = residual.mul_add(residual, inner);
    let shift_w = outer.abs() + EPSILON * (inner.abs() + outer.abs());
    let real_moment = |order_x: usize, order_y: usize| {
        majorant(order_x, 0.0, order_y, 0.0).integrate(absolute_moment, absolute_moment)
    };
    let law = |order: usize| {
        0.5 * shift_v * real_moment(order + 2, order)
            + shift_r * real_moment(order + 1, order + 1)
            + 0.5 * shift_w * real_moment(order, order + 2)
    };
    PairOracles {
        kernel: Oracle {
            value: kernel.value(),
            aliasing: aliasing(0),
            truncation: truncation(0),
            law: law(0),
            rounding: kernel.rounding(),
        },
        derivative: Oracle {
            value: derivative.value(),
            aliasing: aliasing(1),
            truncation: truncation(1),
            law: law(1),
            rounding: derivative.rounding(),
        },
    }
}

/// The contract on `T⁽ᵏ⁾`: the summed magnitudes of `t Φ(u) + (v/S) φ(u)`,
/// `Φ(u) + u φ(u)/A` or `φ(u) (A |He|_{k−2} + |He|_k)/(A Sᵏ⁻¹)`, with
/// `A = 1 + v`, `S = √A` and `u = t/S`, whose rounding moves `φ(u)` and `Φ(u)`
/// by `u²ε` and `(|u| + u²)ε`.
fn smoothing_contract(t: f64, variance: f64, order: usize) -> f64 {
    let total = 1.0 + variance;
    let root_total = total.sqrt();
    let u = t / root_total;
    let density = normal_pdf(u);
    let probability = normal_cdf(u);
    let magnitude = match order {
        0 => t.abs() * probability + variance / root_total * density,
        1 => probability + u.abs() * density / total,
        _ => {
            density * (total * absolute_hermite(order - 2, u) + absolute_hermite(order, u))
                / (total * root_total.powi(order as i32 - 1))
        }
    };
    EPSILON * (SPECIAL_FUNCTION_OPERATIONS + 2.0 * u * u + 2.0 * u.abs()) * magnitude
}

/// The discriminant pieces of a pair law: `A = 1 + v`, `B = 1 + w`, `vw − r²`
/// with both squares carried exactly, and `Δ = AB − r² = 1 + v + w + (vw − r²)`.
struct PairLaw {
    total_x: f64,
    total_y: f64,
    determinant: f64,
    discriminant: f64,
}

impl PairLaw {
    fn new(variance_x: f64, variance_y: f64, covariance: f64) -> Self {
        let product = variance_x * variance_y;
        let square = covariance * covariance;
        let determinant = (product - square)
            + (variance_x.mul_add(variance_y, -product) - covariance.mul_add(covariance, -square));
        Self {
            total_x: 1.0 + variance_x,
            total_y: 1.0 + variance_y,
            determinant,
            discriminant: 1.0 + variance_x + variance_y + determinant,
        }
    }
}

/// The contract on the zero-mean `K` and `∂_r K`: the summed magnitudes of
/// `r H + ((vw − r²) + r² (1/A + 1/B))/(2π √Δ)` and
/// `H + r (1/A + 1/B + 1/Δ)/(2π √Δ)`, `H = ¼ + arcsin(ρ)/(2π)`.
fn pair_contract(variance_x: f64, variance_y: f64, covariance: f64) -> (f64, f64) {
    let law = PairLaw::new(variance_x, variance_y, covariance);
    let root_totals = law.total_x.sqrt() * law.total_y.sqrt();
    let orthant = (law.discriminant.sqrt() / root_totals).atan2(-covariance / root_totals) / TAU;
    let density = 1.0 / (TAU * law.discriminant.sqrt());
    let inverse_totals = law.total_x.recip() + law.total_y.recip();
    let kernel = covariance.abs() * orthant
        + (law.determinant + covariance * covariance * inverse_totals) * density;
    let derivative =
        orthant + covariance.abs() * density * (inverse_totals + law.discriminant.recip());
    (
        EPSILON * CLOSED_FORM_OPERATIONS * kernel,
        EPSILON * CLOSED_FORM_OPERATIONS * derivative,
    )
}

/// The standardized pieces of a biased pair law: `h = b/√A`, `k = c/√B`,
/// `ρ = r/√(AB)`, `n = 1 − ρ² = Δ/(AB)`, `q_h = (k − ρh)/√n`, `q_k = (h − ρk)/√n`,
/// `Φ₂ = Φ₂(h, k; ρ)`, `ψ_h = φ(h) Φ(q_h)`, `ψ_k = φ(k) Φ(q_k)` and
/// `φ₂ = φ(h) φ(q_h)/√n`.
struct StandardizedPair {
    law: PairLaw,
    root_totals: f64,
    h: f64,
    k: f64,
    correlation: f64,
    complement: f64,
    root_complement: f64,
    quotient_h: f64,
    quotient_k: f64,
    orthant: f64,
    /// The bivariate normal owner's per-call bound on `orthant` (`gam_math::bivariate_normal`, "Contract").
    orthant_rounding: f64,
    tilted_h: f64,
    tilted_k: f64,
    density: f64,
}

impl StandardizedPair {
    fn new(mean_x: f64, mean_y: f64, variance_x: f64, variance_y: f64, covariance: f64) -> Self {
        let law = PairLaw::new(variance_x, variance_y, covariance);
        let root_x = law.total_x.sqrt();
        let root_y = law.total_y.sqrt();
        let root_totals = root_x * root_y;
        let h = mean_x / root_x;
        let k = mean_y / root_y;
        let correlation = covariance / root_totals;
        let complement = (law.discriminant / (law.total_x * law.total_y)).min(1.0);
        let root_complement = complement.sqrt();
        let quotient_h = (k - correlation * h) / root_complement;
        let quotient_k = (h - correlation * k) / root_complement;
        let orthant = bivariate_normal_cdf_with_complement(h, k, correlation, complement)
            .expect("standardized orthant of a valid pair law");
        Self {
            law,
            root_totals,
            h,
            k,
            correlation,
            complement,
            root_complement,
            quotient_h,
            quotient_k,
            orthant: orthant.value,
            orthant_rounding: orthant.rounding,
            tilted_h: normal_pdf(h) * normal_cdf(quotient_h),
            tilted_k: normal_pdf(k) * normal_cdf(quotient_k),
            density: normal_pdf(h) * normal_pdf(quotient_h) / root_complement,
        }
    }
}

/// The contract on the biased `K` and `∂_r K`, in the proposal's arrangement
/// `K = (bc + r) H + (cv + br/A) H_b + (bw + cr/B) H_c + (vw + r² q) H_bc` and
/// `∂_r K = H + (b H_b + r H_bc)/A + (c H_c + r H_bc)/B + (r/Δ + b̃ c̃) H_bc`, with
/// `H_b = ψ_h/√A`, `H_bc = φ₂/√(AB)`, `b̃ = (Bb − rc)/Δ` and `c̃ = (Ac − rb)/Δ`:
/// - first-order rounding on the summed term magnitudes;
/// - `ψ_h = φ(h) Φ(q_h)` reading a rounded `h`, which moves `φ(h)` by `h² ε`, and
///   a rounded `q_h`, whose numerator `k − ρh` errs by `ε (|k| + |ρh|)` before
///   division by `√n`; likewise `φ₂`, which reads `q_h` through `φ`;
/// - `Φ₂`'s owner per-call bound, with its partials times the rounding of `h`, `k`
///   and of `ρ` resolved from the complement, `ε (|ρ| + n)`;
/// - the cancellation in `b̃` and `c̃`.
fn biased_pair_contract(
    mean_x: f64,
    mean_y: f64,
    variance_x: f64,
    variance_y: f64,
    covariance: f64,
) -> (f64, f64) {
    let pair = StandardizedPair::new(mean_x, mean_y, variance_x, variance_y, covariance);
    let rounding = SPECIAL_FUNCTION_OPERATIONS * EPSILON;
    let (h, k, correlation) = (pair.h, pair.k, pair.correlation);
    let (total_x, total_y) = (pair.law.total_x, pair.law.total_y);
    let root_x = total_x.sqrt();
    let root_y = total_y.sqrt();
    let numerator_h = (k.abs() + (correlation * h).abs()) / pair.root_complement;
    let numerator_k = (h.abs() + (correlation * k).abs()) / pair.root_complement;
    let tilted_h_error = rounding
        * (pair.tilted_h * (1.0 + h * h + h.abs())
            + normal_pdf(h) * normal_pdf(pair.quotient_h) * (numerator_h + pair.quotient_h.abs()));
    let tilted_k_error = rounding
        * (pair.tilted_k * (1.0 + k * k + k.abs())
            + normal_pdf(k) * normal_pdf(pair.quotient_k) * (numerator_k + pair.quotient_k.abs()));
    let density_error = rounding
        * pair.density
        * (1.0 + h * h + h.abs() + pair.quotient_h.abs() * (numerator_h + pair.quotient_h.abs()));
    let orthant_error = pair.orthant_rounding
        + rounding
            * (pair.tilted_h * h.abs()
                + pair.tilted_k * k.abs()
                + pair.density * (correlation.abs() + pair.complement));
    let coefficient_h = (mean_y * variance_x + mean_x * covariance / total_x).abs() / root_x;
    let coefficient_k = (mean_x * variance_y + mean_y * covariance / total_y).abs() / root_y;
    let coupling =
        (pair.law.determinant + covariance * covariance * (total_x.recip() + total_y.recip())).abs()
            / pair.root_totals;
    let product = (mean_x * mean_y + covariance).abs();
    let kernel_terms = product * pair.orthant
        + coefficient_h * pair.tilted_h
        + coefficient_k * pair.tilted_k
        + coupling * pair.density;
    let kernel_contract = rounding * kernel_terms
        + product * orthant_error
        + coefficient_h * tilted_h_error
        + coefficient_k * tilted_k_error
        + coupling * density_error;
    let discriminant = pair.law.discriminant;
    let reduced_x = (total_y * mean_x - covariance * mean_y) / discriminant;
    let reduced_y = (total_x * mean_y - covariance * mean_x) / discriminant;
    let reduced_error = rounding
        * ((total_y * mean_x.abs() + (covariance * mean_y).abs()) * reduced_y.abs()
            + reduced_x.abs() * (total_x * mean_y.abs() + (covariance * mean_x).abs()))
        / discriminant;
    let bracket = (covariance / discriminant + reduced_x * reduced_y).abs() / pair.root_totals;
    let spread = covariance.abs() / pair.root_totals;
    let derivative_terms = pair.orthant
        + (mean_x.abs() / root_x * pair.tilted_h + spread * pair.density) / total_x
        + (mean_y.abs() / root_y * pair.tilted_k + spread * pair.density) / total_y
        + bracket * pair.density;
    let derivative_contract = rounding * derivative_terms
        + orthant_error
        + (mean_x.abs() / root_x * tilted_h_error + spread * density_error) / total_x
        + (mean_y.abs() / root_y * tilted_k_error + spread * density_error) / total_y
        + bracket * density_error
        + reduced_error * pair.density / pair.root_totals;
    (kernel_contract, derivative_contract)
}

/// The derivation's smoothing, arranged apart from the API's:
/// `T = s (u Φ(u) + φ(u)) − φ(u)/s`, `T' = Φ(u) + u φ(u)/s²` and
/// `T⁽ᵏ⁾ = (−1)ᵏ s^{−1−k} φ(u) (s² He_{k−2}(u) − He_k(u))`, `s = √(1 + v)`,
/// `u = t/s`, with its contract.
fn derived_smoothing(t: f64, variance: f64, order: usize) -> (f64, f64) {
    let scale = (1.0 + variance).sqrt();
    let u = t / scale;
    let density = normal_pdf(u);
    let probability = normal_cdf(u);
    let (value, magnitude) = match order {
        0 => (
            scale * (u * probability + density) - density / scale,
            scale * (u.abs() * probability + density) + density / scale,
        ),
        1 => (
            probability + u * density / (scale * scale),
            probability + u.abs() * density / (scale * scale),
        ),
        _ => {
            let mut hermite = vec![1.0, u];
            for degree in 1..order {
                hermite.push(u * hermite[degree] - degree as f64 * hermite[degree - 1]);
            }
            let sign = if order % 2 == 0 { 1.0 } else { -1.0 };
            let factor = scale.powi(-1 - order as i32) * density;
            (
                sign * factor * (scale * scale * hermite[order - 2] - hermite[order]),
                factor * (scale * scale * absolute_hermite(order - 2, u) + absolute_hermite(order, u)),
            )
        }
    };
    (
        value,
        EPSILON * (SPECIAL_FUNCTION_OPERATIONS + 2.0 * u * u + 2.0 * u.abs()) * magnitude,
    )
}

/// The derivation's zero-mean kernel and Price derivative:
/// `K = r/4 + r arcsin(ρ)/(2π) + (ρ²/√n + vw √n)/(2π √(AB))` and
/// `∂_r K = ¼ + arcsin(ρ)/(2π) + ρ (2 + v + w + 1/n)/(2π √n AB)`, with
/// `n = 1 − ρ² = Δ/(AB)` and the orthant `¼ + arcsin(ρ)/(2π)` formed as
/// `atan2(√n, −ρ)/(2π)`. Returns `((K, contract), (∂_r K, contract))`.
fn derived_zero_mean_pair(variance_x: f64, variance_y: f64, covariance: f64) -> ((f64, f64), (f64, f64)) {
    let law = PairLaw::new(variance_x, variance_y, covariance);
    let root_totals = law.total_x.sqrt() * law.total_y.sqrt();
    let totals = law.total_x * law.total_y;
    let correlation = covariance / root_totals;
    let root_complement = law.discriminant.sqrt() / root_totals;
    let orthant = root_complement.atan2(-correlation) / TAU;
    let spread = correlation * correlation / root_complement;
    let product = variance_x * variance_y * root_complement;
    let kernel = covariance * orthant + (spread + product) / (TAU * root_totals);
    let kernel_magnitude = covariance.abs() * orthant + (spread + product) / (TAU * root_totals);
    let slope = 2.0 + variance_x + variance_y + totals / law.discriminant;
    let derivative = orthant + correlation * slope / (TAU * root_complement * totals);
    let derivative_magnitude = orthant + correlation.abs() * slope / (TAU * root_complement * totals);
    (
        (kernel, EPSILON * CLOSED_FORM_OPERATIONS * kernel_magnitude),
        (derivative, EPSILON * CLOSED_FORM_OPERATIONS * derivative_magnitude),
    )
}

/// The derivation's biased kernel and Price derivative, in its own arrangement:
/// `K = √(AB) {(hk + ρ) Φ₂ + ((ρh + vk)/A) ψ_h + ((ρk + wh)/B) ψ_k + ((ρ² + vw n)/(AB)) φ₂}`
/// and `∂_r K = Φ₂ + (h ψ_h + ρ φ₂)/A + (k ψ_k + ρ φ₂)/B + (αγ + ρ/n) φ₂/(AB)`,
/// with `α = (h − ρk)/n` and `γ = (k − ρh)/n`. Returns `(K, ∂_r K)`; its term
/// magnitudes equal the proposal's term by term, so it carries the same contract.
fn derived_biased_pair(
    mean_x: f64,
    mean_y: f64,
    variance_x: f64,
    variance_y: f64,
    covariance: f64,
) -> (f64, f64) {
    let pair = StandardizedPair::new(mean_x, mean_y, variance_x, variance_y, covariance);
    let (h, k, correlation, complement) = (pair.h, pair.k, pair.correlation, pair.complement);
    let totals = pair.law.total_x * pair.law.total_y;
    let kernel = pair.root_totals
        * ((h * k + correlation) * pair.orthant
            + (correlation * h + variance_x * k) / pair.law.total_x * pair.tilted_h
            + (correlation * k + variance_y * h) / pair.law.total_y * pair.tilted_k
            + (correlation * correlation + variance_x * variance_y * complement) / totals * pair.density);
    let alpha = (h - correlation * k) / complement;
    let gamma = (k - correlation * h) / complement;
    let derivative = pair.orthant
        + (h * pair.tilted_h + correlation * pair.density) / pair.law.total_x
        + (k * pair.tilted_k + correlation * pair.density) / pair.law.total_y
        + (alpha * gamma + correlation / complement) * pair.density / totals;
    (kernel, derivative)
}

fn api_smoothing(t: f64, variance: f64) -> [f64; 5] {
    let mut derivatives = [f64::NAN; 5];
    gaussian_smoothing_derivatives(GaussianActivation::ExactGelu, t, variance, &mut derivatives)
        .expect("exact GELU smoothing of a valid law");
    derivatives
}

fn api_pair(mean_x: f64, mean_y: f64, variance_x: f64, variance_y: f64, covariance: f64) -> PairKernel {
    pair_kernel(
        GaussianActivation::ExactGelu,
        PreactivationPair {
            mean_x,
            mean_y,
            variance_x,
            variance_y,
            covariance,
            covariance_rounding: 0.0,
        },
    )
    .expect("exact GELU pair kernel of a valid law")
}

/// `(t, v)`: v = 0 and v → 0, both tails (u = −8, +5.7, −5.8, −1), large
/// smoothing variances and a large-norm reader.
const SMOOTHING_LAWS: [(f64, f64); 10] = [
    (0.7, 0.0),
    (0.4, 1.0e-12),
    (-1.3, 0.5),
    (2.5, 3.0),
    (-8.0, 1.0),
    (8.0, 1.0),
    (-24.0, 16.0),
    (5.0, 100.0),
    (-30.0, 900.0),
    (32.0, 4096.0),
];

/// `(v, w, r)`: moderate laws, a near-boundary covariance, independence, a zero
/// variance, v → 0, and the Cauchy-Schwarz boundary r² = vw at large norm with
/// ρ = 4096/4097, −4096/4097, −256/√(4097·17) and −10⁶/(10⁶ + 1).
const PAIR_LAWS: [(f64, f64, f64); 10] = [
    (1.0, 1.0, 0.5),
    (0.25, 4.0, -0.9),
    (9.0, 4.0, 5.994),
    (2.0, 3.0, 0.0),
    (0.0, 2.0, 0.0),
    (1.0e-12, 1.0, 5.0e-7),
    (4096.0, 4096.0, 4096.0),
    (4096.0, 4096.0, -4096.0),
    (4096.0, 16.0, -256.0),
    (1.0e6, 1.0e6, -1.0e6),
];

/// `(b, c, v, w, r)`: moderate laws of both covariance signs, both joint tails,
/// one pre-activation in its upper tail and the other in its lower, the boundary
/// r² = vw at v = w = 256 of both signs (ρ = ±256/257), v = w = r → 0, and a
/// zero variance.
const BIASED_PAIR_LAWS: [(f64, f64, f64, f64, f64); 8] = [
    (0.4, -0.9, 1.2, 0.7, 0.35),
    (-2.0, 1.5, 0.3, 2.0, -0.6),
    (-6.0, -5.0, 1.0, 1.5, 1.2),
    (6.0, -5.0, 1.0, 1.5, -1.2),
    (3.0, -2.0, 256.0, 256.0, 256.0),
    (3.0, -2.0, 256.0, 256.0, -256.0),
    (0.3, -0.2, 1.0e-12, 1.0e-12, 1.0e-12),
    (1.5, -0.7, 0.0, 2.0, 0.0),
];

fn assert_within(label: &str, law: &str, closed_value: f64, oracle: Oracle, contract: f64) {
    let tolerance = oracle.bound() + contract;
    let discrepancy = (closed_value - oracle.value).abs();
    assert!(
        discrepancy <= tolerance,
        "exact GELU {label} at {law}: closed form {closed_value:e} against oracle {:e} (discrepancy {discrepancy:e}, tolerance {tolerance:e} = aliasing {:e} + truncation {:e} + law {:e} + rounding {:e} + contract {contract:e})",
        oracle.value,
        oracle.aliasing,
        oracle.truncation,
        oracle.law,
        oracle.rounding
    );
}

#[test]
fn exact_gelu_smoothing_derivatives_match_the_bounded_trapezoid_oracle_into_the_tails() {
    for (t, variance) in SMOOTHING_LAWS {
        let closed = api_smoothing(t, variance);
        for (order, closed_value) in closed.iter().enumerate() {
            let oracle = smoothing_oracle(t, variance, order);
            let contract = smoothing_contract(t, variance, order);
            let tolerance = oracle.bound() + contract;
            let discrepancy = (closed_value - oracle.value).abs();
            assert!(
                discrepancy <= tolerance,
                "exact GELU T^({order}) at t = {t}, v = {variance}: closed form {closed_value:e} against oracle {:e} (discrepancy {discrepancy:e}, tolerance {tolerance:e} = aliasing {:e} + truncation {:e} + law {:e} + rounding {:e} + contract {contract:e})",
                oracle.value,
                oracle.aliasing,
                oracle.truncation,
                oracle.law,
                oracle.rounding
            );
        }
    }
}

#[test]
fn exact_gelu_zero_mean_pair_kernel_and_price_derivative_match_the_bounded_trapezoid_oracle() {
    for (variance_x, variance_y, covariance) in PAIR_LAWS {
        let closed = api_pair(0.0, 0.0, variance_x, variance_y, covariance);
        let oracles = pair_oracles(0.0, 0.0, variance_x, variance_y, covariance);
        let (kernel_contract, derivative_contract) = pair_contract(variance_x, variance_y, covariance);
        let law = format!("v = {variance_x}, w = {variance_y}, r = {covariance}");
        assert_within("K", &law, closed.value, oracles.kernel, kernel_contract);
        assert_within("∂_r K", &law, closed.covariance_derivative, oracles.derivative, derivative_contract);
    }
}

#[test]
fn exact_gelu_biased_pair_kernel_and_price_derivative_match_the_bounded_trapezoid_oracle() {
    for (mean_x, mean_y, variance_x, variance_y, covariance) in BIASED_PAIR_LAWS {
        let closed = api_pair(mean_x, mean_y, variance_x, variance_y, covariance);
        let oracles = pair_oracles(mean_x, mean_y, variance_x, variance_y, covariance);
        let (kernel_contract, derivative_contract) =
            biased_pair_contract(mean_x, mean_y, variance_x, variance_y, covariance);
        let law = format!("b = {mean_x}, c = {mean_y}, v = {variance_x}, w = {variance_y}, r = {covariance}");
        assert_within("K", &law, closed.value, oracles.kernel, kernel_contract);
        assert_within("∂_r K", &law, closed.covariance_derivative, oracles.derivative, derivative_contract);
    }
}

#[test]
fn exact_gelu_kernels_match_the_separately_derived_closed_forms() {
    for (t, variance) in SMOOTHING_LAWS {
        let closed = api_smoothing(t, variance);
        for (order, closed_value) in closed.iter().enumerate() {
            let (derived, derived_contract) = derived_smoothing(t, variance, order);
            let tolerance = smoothing_contract(t, variance, order) + derived_contract;
            let discrepancy = (closed_value - derived).abs();
            assert!(
                discrepancy <= tolerance,
                "exact GELU T^({order}) at t = {t}, v = {variance}: API {closed_value:e} against the derived form {derived:e} (discrepancy {discrepancy:e}, tolerance {tolerance:e})"
            );
        }
    }
    for (variance_x, variance_y, covariance) in PAIR_LAWS {
        let closed = api_pair(0.0, 0.0, variance_x, variance_y, covariance);
        let ((kernel, kernel_derived_contract), (derivative, derivative_derived_contract)) =
            derived_zero_mean_pair(variance_x, variance_y, covariance);
        let (kernel_contract, derivative_contract) = pair_contract(variance_x, variance_y, covariance);
        for (label, closed_value, derived, tolerance) in [
            ("K", closed.value, kernel, kernel_contract + kernel_derived_contract),
            (
                "∂_r K",
                closed.covariance_derivative,
                derivative,
                derivative_contract + derivative_derived_contract,
            ),
        ] {
            let discrepancy = (closed_value - derived).abs();
            assert!(
                discrepancy <= tolerance,
                "exact GELU {label} at v = {variance_x}, w = {variance_y}, r = {covariance}: API {closed_value:e} against the derived form {derived:e} (discrepancy {discrepancy:e}, tolerance {tolerance:e})"
            );
        }
    }
    for (mean_x, mean_y, variance_x, variance_y, covariance) in BIASED_PAIR_LAWS {
        let closed = api_pair(mean_x, mean_y, variance_x, variance_y, covariance);
        let (kernel, derivative) = derived_biased_pair(mean_x, mean_y, variance_x, variance_y, covariance);
        let (kernel_contract, derivative_contract) =
            biased_pair_contract(mean_x, mean_y, variance_x, variance_y, covariance);
        for (label, closed_value, derived, tolerance) in [
            ("K", closed.value, kernel, 2.0 * kernel_contract),
            ("∂_r K", closed.covariance_derivative, derivative, 2.0 * derivative_contract),
        ] {
            let discrepancy = (closed_value - derived).abs();
            assert!(
                discrepancy <= tolerance,
                "exact GELU {label} at b = {mean_x}, c = {mean_y}, v = {variance_x}, w = {variance_y}, r = {covariance}: API {closed_value:e} against the derived form {derived:e} (discrepancy {discrepancy:e}, tolerance {tolerance:e})"
            );
        }
    }
}

#[test]
fn the_oracle_tolerance_rejects_perturbed_closed_forms() {
    // T with the ReLU spread √v φ(u) in place of (v/√(1+v)) φ(u).
    let (t, variance) = (-1.3, 0.5);
    let oracle = smoothing_oracle(t, variance, 0);
    let root_total = (1.0 + variance).sqrt();
    let u = t / root_total;
    let perturbed = t * normal_cdf(u) + variance.sqrt() * normal_pdf(u);
    let tolerance = oracle.bound() + smoothing_contract(t, variance, 0);
    let control = (perturbed - oracle.value).abs();
    assert!(
        control > tolerance,
        "smoothing control at t = {t}, v = {variance}: the ReLU spread is within tolerance ({control:e} ≤ {tolerance:e})"
    );

    // K without q in (vw + r² q), and ∂_r K without Price's 1/Δ.
    let (variance_x, variance_y, covariance) = (9.0, 4.0, 5.994);
    let oracles = pair_oracles(0.0, 0.0, variance_x, variance_y, covariance);
    let ((kernel, kernel_derived_contract), (derivative, derivative_derived_contract)) =
        derived_zero_mean_pair(variance_x, variance_y, covariance);
    let law = PairLaw::new(variance_x, variance_y, covariance);
    let density = 1.0 / (TAU * law.discriminant.sqrt());
    let q_complement = variance_x / law.total_x + variance_y / law.total_y;
    let perturbed_kernel = kernel + covariance * covariance * q_complement * density;
    let kernel_tolerance = oracles.kernel.bound() + kernel_derived_contract;
    let kernel_control = (perturbed_kernel - oracles.kernel.value).abs();
    assert!(
        kernel_control > kernel_tolerance,
        "kernel control: (vw + r²) in place of (vw + r² q) is within tolerance ({kernel_control:e} ≤ {kernel_tolerance:e})"
    );
    let perturbed_derivative = derivative - covariance * density / law.discriminant;
    let derivative_tolerance = oracles.derivative.bound() + derivative_derived_contract;
    let derivative_control = (perturbed_derivative - oracles.derivative.value).abs();
    assert!(
        derivative_control > derivative_tolerance,
        "derivative control: dropping 1/Δ is within tolerance ({derivative_control:e} ≤ {derivative_tolerance:e})"
    );

    // The orthant as arccos(−ρ) of a rounded ρ, as ρ → −1: the tolerance resolves
    // the digits that form loses (#2946 comment 5716192200).
    let (variance_x, variance_y, covariance) = (4096.0, 4096.0, -4096.0);
    let oracles = pair_oracles(0.0, 0.0, variance_x, variance_y, covariance);
    let law = PairLaw::new(variance_x, variance_y, covariance);
    let correlation = covariance / (law.total_x.sqrt() * law.total_y.sqrt());
    let density = 1.0 / (TAU * law.discriminant.sqrt());
    let coupling = law.determinant + covariance * covariance * (law.total_x.recip() + law.total_y.recip());
    let arccos_kernel = covariance * (-correlation).acos() / TAU + coupling * density;
    let (kernel_contract, derivative_contract) = pair_contract(variance_x, variance_y, covariance);
    let arccos_tolerance = oracles.kernel.bound() + kernel_contract;
    let arccos_control = (arccos_kernel - oracles.kernel.value).abs();
    assert!(
        arccos_control > arccos_tolerance,
        "arccos control at v = w = −r = 4096: within tolerance ({arccos_control:e} ≤ {arccos_tolerance:e})"
    );
    // The atan2 orthant from the exact discriminant meets the same tolerance.
    let ((stable_kernel, stable_contract), (stable_derivative, stable_derivative_contract)) =
        derived_zero_mean_pair(variance_x, variance_y, covariance);
    let stable_tolerance = oracles.kernel.bound() + kernel_contract.max(stable_contract);
    let stable_discrepancy = (stable_kernel - oracles.kernel.value).abs();
    assert!(
        stable_discrepancy <= stable_tolerance,
        "stable orthant at v = w = −r = 4096: {stable_discrepancy:e} beyond {stable_tolerance:e}"
    );
    let stable_derivative_tolerance =
        oracles.derivative.bound() + derivative_contract.max(stable_derivative_contract);
    let stable_derivative_discrepancy = (stable_derivative - oracles.derivative.value).abs();
    assert!(
        stable_derivative_discrepancy <= stable_derivative_tolerance,
        "stable orthant derivative at v = w = −r = 4096: {stable_derivative_discrepancy:e} beyond {stable_derivative_tolerance:e}"
    );

    // Biased K with (cv + br) in place of (cv + br/A), and biased ∂_r K without
    // the b̃ c̃ H_bc term.
    let (mean_x, mean_y, variance_x, variance_y, covariance) = (0.4, -0.9, 1.2, 0.7, 0.35);
    let oracles = pair_oracles(mean_x, mean_y, variance_x, variance_y, covariance);
    let (kernel, derivative) = derived_biased_pair(mean_x, mean_y, variance_x, variance_y, covariance);
    let (kernel_contract, derivative_contract) =
        biased_pair_contract(mean_x, mean_y, variance_x, variance_y, covariance);
    let pair = StandardizedPair::new(mean_x, mean_y, variance_x, variance_y, covariance);
    let root_x = pair.law.total_x.sqrt();
    let unscaled_slope = mean_x * covariance * (1.0 - pair.law.total_x.recip()) * pair.tilted_h / root_x;
    let biased_kernel_tolerance = oracles.kernel.bound() + kernel_contract;
    let biased_kernel_control = (kernel + unscaled_slope - oracles.kernel.value).abs();
    assert!(
        biased_kernel_control > biased_kernel_tolerance,
        "biased kernel control: (cv + br) in place of (cv + br/A) is within tolerance ({biased_kernel_control:e} ≤ {biased_kernel_tolerance:e})"
    );
    let discriminant = pair.law.discriminant;
    let reduced_x = (pair.law.total_y * mean_x - covariance * mean_y) / discriminant;
    let reduced_y = (pair.law.total_x * mean_y - covariance * mean_x) / discriminant;
    let reduced_term = reduced_x * reduced_y * pair.density / pair.root_totals;
    let biased_derivative_tolerance = oracles.derivative.bound() + derivative_contract;
    let biased_derivative_control = (derivative - reduced_term - oracles.derivative.value).abs();
    assert!(
        biased_derivative_control > biased_derivative_tolerance,
        "biased derivative control: dropping b̃ c̃ H_bc is within tolerance ({biased_derivative_control:e} ≤ {biased_derivative_tolerance:e})"
    );
}
