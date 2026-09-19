//! The bivariate standard normal distribution function
//! `Φ₂(h, k; ρ) = P(X ≤ h, Y ≤ k)`, for standard normal `X, Y` with correlation
//! `ρ`, together with its density and its analytic partials (#2946).
//!
//! # Core rule, `|ρ| ≤ ½`
//!
//! Plackett's identity `∂_ρ Φ₂ = φ₂` integrated from `ρ = 0` with `r = sin θ`
//! (Drezner and Wesolowsky 1990) gives
//!
//! `Φ₂(h, k; ρ) = Φ(h)Φ(k) + (1/2π) ∫₀^α f(θ) dθ`, with `α = asin ρ`.
//!
//! The integrand is written with non-negative coefficients only:
//!
//! `f(θ) = exp(−a / cos²θ − b / (1 + σ sin θ))`,
//! with `a = (|h| − |k|)²/2`, `b = |hk|` and `σ = sign(hk)`.
//!
//! This is `(h² − 2hk sin θ + k²)/(2cos²θ)` split as `(|h| − |k|)²/2 + |hk|(1 − σ sin θ)`
//! over `(1 − sin θ)(1 + sin θ)`. It never cancels and never forms `∞ − ∞`.
//!
//! **Uniform bound.** On the diamond `|Re θ| + |Im θ| ≤ π/2`:
//! - `Re(1 + σ sin θ) = 1 + σ sin x cosh y ≥ 1 − cos t cosh t ≥ 0`, with `t = π/2 − |x| ≥ |y|`, because
//!   `cos t cosh t ≤ 1` on `[0, π/2]`;
//! - `Re cos²θ ≥ 0` follows from the same inequality.
//!
//! So `|f| ≤ 1` there for every `(h, k)`.
//!
//! **Order.** Map `θ = α(1 + x)/2`. The Bernstein ellipse `E_r` of `[−1, 1]` stays inside the diamond iff
//! `(|α|/2)(1 + √((r² + r⁻²)/2)) ≤ π/2`. The largest core angle `|α| = π/6` gives `r² + r⁻² = 50`.
//! Gauss-Legendre with `n` nodes misses `∫₋₁¹` by at most `(64/15) r⁻²ⁿ/(r² − 1)` (Trefethen 2008,
//! Thm 4.5). The order is the smallest `n` whose bound is at most `2ε`, the rounding scale of the same
//! sum (its weights sum to 2 and `0 < f ≤ 1`). A higher order would resolve digits the sum cannot hold.
//! The result is `n = 9` (`core_order`), and the core's truncation is at most `(|α|/4π)·2ε ≤ ε/12`.
//!
//! # Reductions, `|ρ| > ½`
//!
//! Rotate to independent `U = (X − Y)/√(2(1 − ρ))` and `V = (X + Y)/√(2(1 + ρ))`, and write
//! `c± = √((1 ± ρ)/2)`.
//!
//! - **`ρ > ½`.** The region `{X ≤ h, Y ≤ k}` splits along `U = u* = (h − k)/(2c₋)`:
//!   `Φ₂(h, k; ρ) = Φ₂(u*, k; −c₋) + Φ₂(−u*, h; −c₋)`.
//! - **`ρ < −½`.** The same region is `{V ≤ w*, lo(V) ≤ U ≤ hi(V)}`, with `w* = (h + k)/(2c₊)`:
//!   `Φ₂(h, k; ρ) = Φ₂(w*, h; c₊) − Φ₂(w*, −k; −c₊)`.
//!
//! Both are exact. The new correlations have modulus `√((1 − |ρ|)/2) < ½`, so every evaluation lands on the
//! core. `½` is not a knob: it is the fixed point `τ = √((1 − τ)/2)`, the smallest core domain the map
//! closes on.
//!
//! As `|ρ| → 1` the pieces tend to independence (`c → 0`). That is the regime where a fixed Gauss-Legendre
//! rule on `θ ∈ [0, asin ρ]` loses digits: the integrand's essential singularity at `θ = ±π/2` reaches the
//! interval. Genz (2004) handles `|ρ| > 0.925` with `s = √(1 − r²)`. That leaves `exp(−(h − k)²/2s²)` singular
//! at the endpoint `s = 0`, where no Bernstein ellipse exists. The rotation removes the endpoint instead.
//!
//! # Contract
//!
//! - **Truncation.** At most `ε/12` for `|ρ| ≤ ½`, and at most `ε/6` elsewhere.
//! - **Error bound.** [`BIVARIATE_NORMAL_CDF_ERROR_BOUND`] bounds truncation plus rounding of one evaluation at
//!   the computed arguments. [`BivariateNormalPartials`] carries a per-evaluation rounding bound next to each
//!   partial. Neither includes the caller's own argument error.
//! - **Rounding model.** Under round-to-nearest every `+ − × ÷ √` errs by at most `u = ε/2` of its result. The model
//!   adds one ulp, `2u`, per libm `exp`, `sin`, `asin` or `erfc`: the contract `gaussian_activation` states for its own
//!   bounds. A count of `k` rounded operations is carried as Wilkinson's `γ_k` ([`accumulation_growth`]), which sits
//!   above the first-order `k·u` and agrees with it to `O(u²)`.
//! - **Exact branches.** Infinite bounds and `ρ ∈ {−1, 0, 1}` are exact special cases.
//! - **Correlation input.** A caller who resolves `1 − ρ²` more finely than `ρ` passes it to the `_with_complement`
//!   entry points. Every `1 ∓ ρ` is then derived from it, and the input conditioning `φ₂·δρ` of a rounded `ρ`
//!   near `±1` never enters.
//! - **Projection.** Every result is projected onto `[0, 1]`, which contains the truth, so the projection never
//!   increases the error.
//! - **Relative accuracy.** The core does not claim it for `Φ₂ ≲ ε`. The sum `Φ(h)Φ(k) + T` cancels when `ρ < 0` in
//!   the lower tails, and so do the `ρ < −½` difference and the negative-correlation pieces of `ρ > ½`.
//!   [`bivariate_normal_cdf_with_complement_bounded`] certifies a relative bound where `ρ ≤ 0` and both constraints are
//!   active, through the positive form of `positive_form`.

mod positive_form;

use crate::double_double::SMALLEST_SUBNORMAL;
use crate::probability::{normal_cdf, normal_pdf};
use crate::roundoff::{UNIT_ROUNDOFF, accumulation_growth};
use crate::special::gauss_legendre;
use libm::{erf, erfc};
use std::f64::consts::{E, FRAC_PI_6, PI, SQRT_2};
use std::fmt;
use std::sync::LazyLock;

/// A bivariate normal argument outside the distribution's domain.
#[derive(Clone, Debug, PartialEq)]
pub enum BivariateNormalError {
    /// A bound was `NaN`. Both infinities are admissible bounds.
    NanBound { name: &'static str },
    /// The correlation was `NaN` or outside `[−1, 1]`.
    CorrelationOutsideUnitInterval { rho: f64 },
    /// The density and the partials exist only for `|ρ| < 1`.
    SingularCorrelation { rho: f64 },
    /// A caller-supplied `1 − ρ²` was `NaN` or outside `[0, 1]`.
    ComplementOutsideUnitInterval { complement: f64 },
}

impl fmt::Display for BivariateNormalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NanBound { name } => {
                write!(formatter, "bivariate normal bound {name} is NaN")
            }
            Self::CorrelationOutsideUnitInterval { rho } => write!(
                formatter,
                "bivariate normal correlation must lie in [-1, 1], got {rho}"
            ),
            Self::SingularCorrelation { rho } => write!(
                formatter,
                "bivariate normal density needs |rho| < 1, got {rho}"
            ),
            Self::ComplementOutsideUnitInterval { complement } => write!(
                formatter,
                "bivariate normal complement 1 - rho^2 must lie in [0, 1], got {complement}"
            ),
        }
    }
}

impl std::error::Error for BivariateNormalError {}

impl From<BivariateNormalError> for String {
    fn from(error: BivariateNormalError) -> Self {
        error.to_string()
    }
}

/// The analytic partials of `Φ₂(h, k; ρ)`, each with a first-order bound on its absolute rounding at the computed
/// arguments. The model is the one at [`BIVARIATE_NORMAL_CDF_ERROR_BOUND`], and the caller's argument error is not
/// included.
/// - `∂_h Φ₂ = φ(h) Φ(t)`, with `t = (k − ρh)/√(1 − ρ²)`, and symmetrically for `k`.
///   - The numerator `(k − h) + h(1 − ρ)` errs by `u|num| + u|k − h| + 3u|h(1 − ρ)|`, and `√(1 − ρ²)` by `3u`. So `t`
///     errs by at most `6u|t| + 4u|h|√((1 − ρ)/(1 + ρ)) ≤ 6u|t| + 4u|h|` (mirrored for `ρ < 0`).
///   - `Φ(t)` adds `2uΦ + 2u|t|φ(t)`, `φ(h)` adds `5u` of itself and the product `u`.
///   - The bound is `u·(8 ∂_hΦ₂ + φ(h)·φ(t)·(8|t| + 4|h|√((1 − ρ)/(1 + ρ))))`, with the computed `φ(t)`. It scales with the
///     partial: for `t ≤ 0` it is at most `u·(8 + (|t| + 1)(8|t| + 4|h|))` of it, because `φ(t)/Φ(t) ≤ |t| + 1` there.
/// - `∂_ρ Φ₂ = φ₂(h, k; ρ) = exp(−E)/(2π√(1 − ρ²))` (Plackett).
///   - The exponent `E = a/(1 − ρ²) + b/(1 ± ρ)` errs by `8uE`: `a` by `3u`, `b` by `u`, `1 − ρ²` by `3u`, the factor
///     by `2u`, the quotients and the addition.
///   - `exp` adds `2u`, the scale `4.5u` and the division `u`, so the bound is `u·φ₂·(8 + 8E)`.
/// - Gradual underflow quantizes a result by at most the smallest subnormal `2^−1074`, and a vanished `exp(−E)` by that
///   over the scale. Each bound adds it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BivariateNormalPartials {
    pub d_h: f64,
    pub d_k: f64,
    pub d_rho: f64,
    pub d_h_rounding: f64,
    pub d_k_rounding: f64,
    pub d_rho_rounding: f64,
}

/// Which guarantee a [`BoundedProbability`]'s `rounding` carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoundingContract {
    /// The value and its rounding come from a certified enclosure of `Φ₂` with a positive lower end, so `rounding`
    /// scales with the value and `rounding/value` bounds its relative error.
    Relative,
    /// `rounding` is [`BIVARIATE_NORMAL_CDF_ERROR_BOUND`], the core's absolute contract. It certifies no relative digit
    /// of a value at or below it.
    Absolute,
}

/// A probability with a bound on its error at the computed arguments.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundedProbability {
    pub value: f64,
    /// A bound on `|value − Φ₂|`, absolute whatever `contract` says.
    pub rounding: f64,
    pub contract: RoundingContract,
}

/// The largest `|asin ρ|` the core rule evaluates: `asin ½`.
const CORE_MAX_ANGLE: f64 = FRAC_PI_6;

/// `sup |x| φ(x) = 1/√(2πe) = 0.241970…`, rounded up.
const ARGUMENT_SENSITIVITY: f64 = 0.2420;

/// `sup φ₂(x, y; ρ) = 1/(2π√(1 − ρ²))` over `|ρ| ≤ ½`, which is `1/(π√3) = 0.183776…`, rounded up.
const CORE_DENSITY_MAXIMUM: f64 = 0.1838;

/// `gauss_legendre`'s node contract: within `4e-16` absolute. Its owner pins it at 8 and 16 nodes against a 50-digit
/// reference, and `core_rule_meets_the_gauss_legendre_contract_at_the_core_order` pins it at the core order.
const CORE_RULE_NODE_ERROR: f64 = 4.0e-16;

/// `gauss_legendre`'s weight contract: within `1e-14` of themselves, pinned as the node contract is.
const CORE_RULE_WEIGHT_ERROR: f64 = 1.0e-14;

/// One weighted node's value `f = exp(−E)` at `|ρ| ≤ ½`, first order.
/// - The angle `θ = α(1 + x)/2` errs by `4u|θ| + |α|ν/2` (the node, `1 + x`, `asin` and the product).
/// - `s = sin θ` errs by `2u|s| + δθ ≤ u + δθ`.
/// - `E = a/(1 − s²) + b/(1 + s)` errs by `E·(19u/3 + 2δs)`: `a` by `3u`, `b` by `u`, `1 − s² ≥ ¾`, `1 + s ≥ ½`, the
///   quotients and the addition.
/// - `f` errs by `2uf + f·δE ≤ 2u + (19u/3 + 2δs)/e`, since `f ≤ 1` and `fE ≤ 1/e`.
const NODE_VALUE_ERROR: f64 = {
    let angle = accumulation_growth(4) * CORE_MAX_ANGLE + CORE_MAX_ANGLE * CORE_RULE_NODE_ERROR / 2.0;
    let sine = accumulation_growth(2) / 2.0 + angle;
    accumulation_growth(2) + (accumulation_growth(19) / 3.0 + 2.0 * sine) / E
};

/// One core evaluation at computed `(h, k, ρ)` with `|ρ| ≤ ½`.
/// - `Φ(h)Φ(k)`: each factor errs by `2uΦ + 2u|x|φ(x)`, and the product adds `u`: at most `5u + 4u·sup|x|φ`.
/// - The node sum `S = Σ wᵢ fᵢ` has positive terms, `Σ wᵢ = 2` and `S ≤ 2`. It errs by `2·NODE_VALUE_ERROR` from the
///   values, `(ω + u)S` from the weights and products, and `8uS` from its partial sums.
/// - `α·S/(4π)` is at most `1/12`. `asin`, the product, `π` and the division add `5u` of it, and `|α|/4π ≤ 1/24`
///   multiplies the sum's error.
/// - The truncation adds at most `u/6`, and the final addition `u`.
const CORE_EVALUATION_ERROR: f64 = {
    let product = accumulation_growth(5) + accumulation_growth(4) * ARGUMENT_SENSITIVITY;
    let sum = 2.0 * NODE_VALUE_ERROR + 2.0 * (CORE_RULE_WEIGHT_ERROR + accumulation_growth(9));
    let integral = accumulation_growth(5) / 12.0 + sum / 24.0;
    product + integral + UNIT_ROUNDOFF / 6.0 + accumulation_growth(1)
};

/// The absolute error of one [`bivariate_normal_cdf`] or [`bivariate_normal_cdf_with_complement`] evaluation at its
/// computed arguments, `|computed − Φ₂(h, k; ρ)|`: truncation plus rounding, first order. It is about `4.0e-15`.
///
/// A reduction adds two core evaluations, and their arguments round inside it.
/// - `c = √((1 ∓ ρ)/2)` errs by `2uc ≤ u` (the factor, the halving and `√`). A core value moves by at most
///   `sup φ₂` per unit of correlation.
/// - The split `(h ∓ k)/(2c)` errs by `4u` of itself, which moves a core value by at most `4u·sup|x|φ`.
/// - The combination adds `u`.
///
/// The exact branches err by less. The projection onto `[0, 1]` never increases the error. The caller's own argument
/// error is not included.
pub const BIVARIATE_NORMAL_CDF_ERROR_BOUND: f64 = 2.0
    * (CORE_EVALUATION_ERROR
        + accumulation_growth(4) * ARGUMENT_SENSITIVITY
        + accumulation_growth(2) / 2.0 * CORE_DENSITY_MAXIMUM)
    + accumulation_growth(1);

/// A Gauss-Legendre rule stored on `[0, 1]`: nodes `(1 + x)/2` and the weights of `[−1, 1]`.
struct CoreRule {
    unit_nodes: Vec<f64>,
    weights: Vec<f64>,
}

impl CoreRule {
    fn with_order(order: usize) -> Self {
        let (nodes, weights) = gauss_legendre(order);
        Self {
            unit_nodes: nodes.iter().map(|node| 0.5 * (1.0 + node)).collect(),
            weights,
        }
    }
}

static CORE_RULE: LazyLock<CoreRule> = LazyLock::new(|| CoreRule::with_order(core_order()));

/// `r²` of the largest Bernstein ellipse whose image stays inside `|Re θ| + |Im θ| ≤ π/2` at the largest
/// core angle: `r² + r⁻² = 2s`, with `s = (π/α − 1)²`.
fn core_bernstein_radius_squared() -> f64 {
    let s = (PI / CORE_MAX_ANGLE - 1.0).powi(2);
    s + (s * s - 1.0).sqrt()
}

/// The smallest Gauss-Legendre order whose Trefethen bound `(64/15) r⁻²ⁿ/(r² − 1)` for an integrand bounded
/// by 1 on `E_r` is at most `2ε`, the rounding scale of the rule's own sum.
fn core_order() -> usize {
    let radius_squared = core_bernstein_radius_squared();
    let target = 2.0 * f64::EPSILON;
    ((64.0 / (15.0 * target * (radius_squared - 1.0))).ln() / radius_squared.ln()).ceil() as usize
}

fn validate(bounds: &[(&'static str, f64)], rho: f64) -> Result<(), BivariateNormalError> {
    if let Some(bound) = bounds.iter().find(|bound| bound.1.is_nan()) {
        return Err(BivariateNormalError::NanBound { name: bound.0 });
    }
    if !(-1.0..=1.0).contains(&rho) {
        return Err(BivariateNormalError::CorrelationOutsideUnitInterval { rho });
    }
    Ok(())
}

fn validate_complement(complement: f64) -> Result<(), BivariateNormalError> {
    if !(0.0..=1.0).contains(&complement) {
        return Err(BivariateNormalError::ComplementOutsideUnitInterval { complement });
    }
    Ok(())
}

/// A correlation with its singular factors `1 − ρ` and `1 + ρ`, and their product `1 − ρ²`.
///
/// - From `ρ` alone, each factor is formed directly, and is exact near its own singular end.
/// - From a caller's `1 − ρ²`, the product is the caller's value itself, and the vanishing factor is
///   `(1 − ρ²)/(1 + |ρ|)`. So a correlation that rounds to `±1` still carries the complement the caller resolved. A
///   complement whose vanishing factor underflows, such as the smallest subnormal at `|ρ| = 1`, keeps a nonzero
///   determinant instead of collapsing to `0·2`.
#[derive(Clone, Copy)]
struct Correlation {
    rho: f64,
    one_minus: f64,
    one_plus: f64,
    complement: f64,
}

impl Correlation {
    fn from_rho(rho: f64) -> Self {
        let (one_minus, one_plus) = (1.0 - rho, 1.0 + rho);
        Self {
            rho,
            one_minus,
            one_plus,
            complement: one_minus * one_plus,
        }
    }

    fn from_complement(rho: f64, complement: f64) -> Self {
        let far = 1.0 + rho.abs();
        let near = complement / far;
        let (one_minus, one_plus) = if rho >= 0.0 { (near, far) } else { (far, near) };
        Self {
            rho,
            one_minus,
            one_plus,
            complement,
        }
    }

    fn negated(self) -> Self {
        Self {
            rho: -self.rho,
            one_minus: self.one_plus,
            one_plus: self.one_minus,
            complement: self.complement,
        }
    }

    fn complement(self) -> f64 {
        self.complement
    }
}

/// `P(lower ≤ Z ≤ upper)` for standard normal `Z`, always as a difference of same-side tails. So a small
/// interval deep in either tail keeps its relative digits.
fn normal_interval_probability(lower: f64, upper: f64) -> f64 {
    if !(lower < upper) {
        return 0.0;
    }
    let probability = if lower >= 0.0 {
        0.5 * (erfc(lower / SQRT_2) - erfc(upper / SQRT_2))
    } else if upper <= 0.0 {
        0.5 * (erfc(-upper / SQRT_2) - erfc(-lower / SQRT_2))
    } else {
        0.5 * (erf(upper / SQRT_2) - erf(lower / SQRT_2))
    };
    probability.max(0.0)
}

/// The exact value when a bound is infinite, or when `ρ` is singular or zero. Otherwise `None`.
fn exact_branch(h: f64, k: f64, correlation: Correlation) -> Option<f64> {
    if h == f64::NEG_INFINITY || k == f64::NEG_INFINITY {
        Some(0.0)
    } else if h == f64::INFINITY {
        Some(normal_cdf(k))
    } else if k == f64::INFINITY {
        Some(normal_cdf(h))
    } else if correlation.one_minus == 0.0 {
        Some(normal_cdf(h.min(k)))
    } else if correlation.one_plus == 0.0 {
        Some(normal_interval_probability(-k, h))
    } else if correlation.rho == 0.0 {
        Some(normal_cdf(h) * normal_cdf(k))
    } else {
        None
    }
}

/// Drezner-Wesolowsky on `rule` for `|ρ| ≤ ½` (see the module docs for the integrand).
fn core_cdf(h: f64, k: f64, rho: f64, rule: &CoreRule) -> f64 {
    if let Some(value) = exact_branch(h, k, Correlation::from_rho(rho)) {
        return value;
    }
    let alpha = rho.asin();
    let a = 0.5 * (h.abs() - k.abs()).powi(2);
    let b = (h * k).abs();
    // `σ sin θ = sin(σθ)`, while `cos²θ` does not see the sign.
    let signed_alpha = if h * k < 0.0 { -alpha } else { alpha };
    let mut sum = 0.0;
    for (&unit, &weight) in rule.unit_nodes.iter().zip(&rule.weights) {
        let sine = (signed_alpha * unit).sin();
        sum += weight * (-(a / (1.0 - sine * sine) + b / (1.0 + sine))).exp();
    }
    (normal_cdf(h) * normal_cdf(k) + alpha * sum / (4.0 * PI)).clamp(0.0, 1.0)
}

/// `Φ₂(h, k; ρ)` on `rule`, through the exact reductions onto the core domain.
fn cdf_on_rule(h: f64, k: f64, correlation: Correlation, rule: &CoreRule) -> f64 {
    if let Some(value) = exact_branch(h, k, correlation) {
        return value;
    }
    let rho = correlation.rho;
    if rho > 0.5 {
        let c = (0.5 * correlation.one_minus).sqrt();
        let split = (h - k) / (2.0 * c);
        (core_cdf(split, k, -c, rule) + core_cdf(-split, h, -c, rule)).min(1.0)
    } else if rho < -0.5 {
        let c = (0.5 * correlation.one_plus).sqrt();
        let split = (h + k) / (2.0 * c);
        // Taking `h ≤ k` centres the conditional interval of `U` at or below zero. So the two pieces are
        // never both near one.
        let (low, high) = if h <= k { (h, k) } else { (k, h) };
        (core_cdf(split, low, c, rule) - core_cdf(split, -high, -c, rule)).clamp(0.0, 1.0)
    } else {
        core_cdf(h, k, rho, rule)
    }
}

/// `Φ₂(h, k; ρ) = P(X ≤ h, Y ≤ k)`. Bounds may be infinite, and `ρ ∈ [−1, 1]`. The error contract is in the
/// module docs.
pub fn bivariate_normal_cdf(h: f64, k: f64, rho: f64) -> Result<f64, BivariateNormalError> {
    validate(&[("h", h), ("k", k)], rho)?;
    Ok(cdf_on_rule(h, k, Correlation::from_rho(rho), &CORE_RULE))
}

/// `Φ₂(h, k; ρ)` for a caller who resolves `1 − ρ²` more finely than `ρ` itself, such as `Δ/(AB)` with
/// `Δ = AB − r²` formed by fma.
///
/// Every `1 ∓ ρ` is derived from `complement`, so a correlation rounded to `±1` keeps the complement's digits.
/// `complement == 0` selects the singular branch by the sign of `ρ`. `complement` must be `1 − ρ²` of the same
/// correlation, to rounding.
pub fn bivariate_normal_cdf_with_complement(
    h: f64,
    k: f64,
    rho: f64,
    complement: f64,
) -> Result<f64, BivariateNormalError> {
    validate(&[("h", h), ("k", k)], rho)?;
    validate_complement(complement)?;
    Ok(cdf_on_rule(
        h,
        k,
        Correlation::from_complement(rho, complement),
        &CORE_RULE,
    ))
}

/// `Φ₂(h, k; ρ)` from a caller-resolved `1 − ρ²`, as in [`bivariate_normal_cdf_with_complement`], with the smaller of
/// two derived error bounds.
///
/// - **Certified region:** `ρ ≤ 0`, finite `h` and `k`, `complement > 0`, and `α₁ = −(h − ρk)/c` and
///   `α₂ = −(k − ρh)/c` both resolved nonnegative (the apex is the design point). There the positive form (module
///   `positive_form`) encloses `Φ₂` with no cancellation, and its bound scales with the value.
/// - **Choice:** that bound is returned as [`RoundingContract::Relative`] when it is below
///   [`BIVARIATE_NORMAL_CDF_ERROR_BOUND`].
/// - **Outside the certified region**, or where the certificate declines (no admissible ellipse, a subnormal density, a
///   failed enclosure) or its bound is the larger: the value is exactly [`bivariate_normal_cdf_with_complement`]'s,
///   with `rounding = BIVARIATE_NORMAL_CDF_ERROR_BOUND` and [`RoundingContract::Absolute`]. That bound rests on one ulp
///   per `sin`, `asin` and `erfc` call, a measurement of the platform library rather than a derivation.
///
/// Neither bound includes the caller's argument error. In particular `complement` is taken as `1 − ρ²` of the same
/// correlation.
pub fn bivariate_normal_cdf_with_complement_bounded(
    h: f64,
    k: f64,
    rho: f64,
    complement: f64,
) -> Result<BoundedProbability, BivariateNormalError> {
    validate(&[("h", h), ("k", k)], rho)?;
    validate_complement(complement)?;
    if let Some(bounded) = positive_form::relative_orthant(h, k, rho, complement)
        && bounded.rounding < BIVARIATE_NORMAL_CDF_ERROR_BOUND
    {
        return Ok(bounded);
    }
    Ok(BoundedProbability {
        value: cdf_on_rule(
            h,
            k,
            Correlation::from_complement(rho, complement),
            &CORE_RULE,
        ),
        rounding: BIVARIATE_NORMAL_CDF_ERROR_BOUND,
        contract: RoundingContract::Absolute,
    })
}

/// `P(X ≤ h, lower ≤ Y ≤ upper)`.
///
/// A finite interval is a difference of two distribution functions taken on the tail side of zero: the
/// reflection `Y → −Y` when `lower ≥ 0`. So a small interval in the upper tail keeps its mass instead of
/// being subtracted from `Φ(h)`. It errs by at most `2·BIVARIATE_NORMAL_CDF_ERROR_BOUND + ε/2`.
pub fn bivariate_normal_interval_probability(
    h: f64,
    lower: f64,
    upper: f64,
    rho: f64,
) -> Result<f64, BivariateNormalError> {
    validate(&[("h", h), ("lower", lower), ("upper", upper)], rho)?;
    if !(lower < upper) {
        return Ok(0.0);
    }
    let (rule, correlation) = (&*CORE_RULE, Correlation::from_rho(rho));
    let reflected = correlation.negated();
    let probability = if lower == f64::NEG_INFINITY {
        cdf_on_rule(h, upper, correlation, rule)
    } else if upper == f64::INFINITY {
        cdf_on_rule(h, -lower, reflected, rule)
    } else if lower >= 0.0 {
        cdf_on_rule(h, -lower, reflected, rule) - cdf_on_rule(h, -upper, reflected, rule)
    } else {
        cdf_on_rule(h, upper, correlation, rule) - cdf_on_rule(h, lower, correlation, rule)
    };
    Ok(probability.clamp(0.0, 1.0))
}

/// `φ₂(h, k; ρ)` for `|ρ| < 1`, with its rounding bound (derived at [`BivariateNormalPartials`]). The quadratic
/// form is carried with non-negative terms, as in the core rule with `sin θ = ρ`.
fn density(h: f64, k: f64, correlation: Correlation) -> (f64, f64) {
    if h.is_infinite() || k.is_infinite() {
        return (0.0, 0.0);
    }
    let determinant = correlation.complement();
    let a = 0.5 * (h.abs() - k.abs()).powi(2);
    let b = (h * k).abs();
    // `1 + σρ` is the factor `1 + ρ` when `hk ≥ 0` and the factor `1 − ρ` when `hk < 0`.
    let signed_factor = if h * k < 0.0 {
        correlation.one_minus
    } else {
        correlation.one_plus
    };
    let exponent = a / determinant + b / signed_factor;
    let scale = 2.0 * PI * determinant.sqrt();
    let value = (-exponent).exp() / scale;
    // A positive value has a finite exponent. A vanished one errs by at most the smallest subnormal over the scale.
    let relative = if value > 0.0 {
        accumulation_growth(8) * value * (1.0 + exponent)
    } else {
        0.0
    };
    (value, relative + SMALLEST_SUBNORMAL * (1.0 + scale.recip()))
}

/// `φ₂(h, k; ρ)`, the bivariate standard normal density, for `|ρ| < 1`.
pub fn bivariate_normal_pdf(h: f64, k: f64, rho: f64) -> Result<f64, BivariateNormalError> {
    validate(&[("h", h), ("k", k)], rho)?;
    if rho.abs() == 1.0 {
        return Err(BivariateNormalError::SingularCorrelation { rho });
    }
    Ok(density(h, k, Correlation::from_rho(rho)).0)
}

/// `φ(x) Φ((y − ρx)/√(1 − ρ²))`. The numerator is `(y − x) + x(1 − ρ)` for `ρ ≥ 0` and `(y + x) − x(1 + ρ)`
/// for `ρ < 0`, so it keeps its digits as `|ρ| → 1`. Its rounding bound is derived at [`BivariateNormalPartials`].
fn conditional_partial(x: f64, y: f64, correlation: Correlation) -> (f64, f64) {
    let weight = normal_pdf(x);
    if weight == 0.0 {
        return (0.0, SMALLEST_SUBNORMAL);
    }
    let numerator = if correlation.rho >= 0.0 {
        (y - x) + x * correlation.one_minus
    } else {
        (y + x) - x * correlation.one_plus
    };
    let t = numerator / correlation.complement().sqrt();
    let value = weight * normal_cdf(t);
    // `Φ(t)` moves by `φ(t)` per unit of the argument's rounding. An infinite argument gives `Φ` exactly.
    let argument = if t.is_finite() {
        let (near, far) = if correlation.rho >= 0.0 {
            (correlation.one_minus, correlation.one_plus)
        } else {
            (correlation.one_plus, correlation.one_minus)
        };
        normal_pdf(t) * (accumulation_growth(8) * t.abs() + accumulation_growth(4) * x.abs() * (near / far).sqrt())
    } else {
        0.0
    };
    let rounding = accumulation_growth(8) * value + weight * argument + SMALLEST_SUBNORMAL;
    (value, rounding)
}

fn partials_of(h: f64, k: f64, correlation: Correlation) -> BivariateNormalPartials {
    let (d_h, d_h_rounding) = conditional_partial(h, k, correlation);
    let (d_k, d_k_rounding) = conditional_partial(k, h, correlation);
    let (d_rho, d_rho_rounding) = density(h, k, correlation);
    BivariateNormalPartials {
        d_h,
        d_k,
        d_rho,
        d_h_rounding,
        d_k_rounding,
        d_rho_rounding,
    }
}

/// The analytic partials `(∂_h, ∂_k, ∂_ρ)` of `Φ₂(h, k; ρ)`, for `|ρ| < 1`.
pub fn bivariate_normal_cdf_partials(
    h: f64,
    k: f64,
    rho: f64,
) -> Result<BivariateNormalPartials, BivariateNormalError> {
    validate(&[("h", h), ("k", k)], rho)?;
    if rho.abs() == 1.0 {
        return Err(BivariateNormalError::SingularCorrelation { rho });
    }
    Ok(partials_of(h, k, Correlation::from_rho(rho)))
}

/// The analytic partials of `Φ₂(h, k; ρ)` from a caller-resolved `1 − ρ²`, as in
/// [`bivariate_normal_cdf_with_complement`]. They exist for `complement > 0`.
pub fn bivariate_normal_cdf_partials_with_complement(
    h: f64,
    k: f64,
    rho: f64,
    complement: f64,
) -> Result<BivariateNormalPartials, BivariateNormalError> {
    validate(&[("h", h), ("k", k)], rho)?;
    validate_complement(complement)?;
    if complement == 0.0 {
        return Err(BivariateNormalError::SingularCorrelation { rho });
    }
    Ok(partials_of(h, k, Correlation::from_complement(rho, complement)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    /// A double-double `hi + lo`, with `|lo| ≤ ulp(hi)/2`: the reference arithmetic of the rounding tests.
    #[derive(Clone, Copy, Debug)]
    struct DoubleDouble {
        hi: f64,
        lo: f64,
    }

    impl DoubleDouble {
        fn from(value: f64) -> Self {
            Self { hi: value, lo: 0.0 }
        }

        /// Knuth's two-sum, `a + b` exactly.
        fn two_sum(a: f64, b: f64) -> Self {
            let sum = a + b;
            let virtual_b = sum - a;
            Self {
                hi: sum,
                lo: (a - (sum - virtual_b)) + (b - virtual_b),
            }
        }

        /// `hi + lo` renormalized, for `|hi| ≥ |lo|`.
        fn renormalized(hi: f64, lo: f64) -> Self {
            let sum = hi + lo;
            Self {
                hi: sum,
                lo: lo - (sum - hi),
            }
        }

        fn add(self, other: Self) -> Self {
            let head = Self::two_sum(self.hi, other.hi);
            Self::renormalized(head.hi, head.lo + self.lo + other.lo)
        }

        fn negated(self) -> Self {
            Self {
                hi: -self.hi,
                lo: -self.lo,
            }
        }

        fn sub(self, other: Self) -> Self {
            self.add(other.negated())
        }

        fn mul(self, other: Self) -> Self {
            let product = self.hi * other.hi;
            let residual = self.hi.mul_add(other.hi, -product);
            Self::renormalized(product, residual + (self.hi * other.lo + self.lo * other.hi))
        }

        fn div(self, other: Self) -> Self {
            let first = self.hi / other.hi;
            let remainder = self.sub(other.mul(Self::from(first)));
            let second = remainder.hi / other.hi;
            let rest = remainder.sub(other.mul(Self::from(second)));
            Self::renormalized(first, second).add(Self::from(rest.hi / other.hi))
        }

        fn sqrt(self) -> Self {
            let root = self.hi.sqrt();
            let remainder = self.sub(Self::from(root).mul(Self::from(root)));
            Self::renormalized(root, remainder.hi / (2.0 * root))
        }
    }

    const PI_DOUBLE_DOUBLE: DoubleDouble = DoubleDouble {
        hi: PI,
        lo: 1.224_646_799_147_353_2e-16,
    };

    /// `(P_n(x), P_n'(x))` by Bonnet's recurrence in double-double, with `P_n' = n(x P_n − P_{n−1})/(x² − 1)`.
    fn legendre_double_double(order: usize, x: DoubleDouble) -> (DoubleDouble, DoubleDouble) {
        let one = DoubleDouble::from(1.0);
        let (mut current, mut previous) = (one, DoubleDouble::from(0.0));
        for degree in 0..order {
            let j = degree as f64;
            let next = DoubleDouble::from(2.0 * j + 1.0)
                .mul(x)
                .mul(current)
                .sub(DoubleDouble::from(j).mul(previous))
                .div(DoubleDouble::from(j + 1.0));
            previous = current;
            current = next;
        }
        let slope = DoubleDouble::from(order as f64)
            .mul(x.mul(current).sub(previous))
            .div(x.mul(x).sub(one));
        (current, slope)
    }

    /// `(1 − ρ, 1 + ρ, 1 − ρ²)` exactly as the production correlation defines them from its computed inputs.
    fn reference_factors(rho: f64, complement: Option<f64>) -> (DoubleDouble, DoubleDouble, DoubleDouble) {
        match complement {
            None => {
                let one_minus = DoubleDouble::from(1.0).sub(DoubleDouble::from(rho));
                let one_plus = DoubleDouble::from(1.0).add(DoubleDouble::from(rho));
                (one_minus, one_plus, one_minus.mul(one_plus))
            }
            Some(c) => {
                let far = DoubleDouble::from(1.0).add(DoubleDouble::from(rho.abs()));
                let near = DoubleDouble::from(c).div(far);
                if rho >= 0.0 {
                    (near, far, DoubleDouble::from(c))
                } else {
                    (far, near, DoubleDouble::from(c))
                }
            }
        }
    }

    /// `φ(x)Φ(t)` with `t` carried in double-double, with an allowance for the reference's own `normal_cdf`
    /// rounding and product. Production and reference read identical `φ(x)` bits.
    fn reference_conditional_partial(
        x: f64,
        y: f64,
        rho: f64,
        factors: (DoubleDouble, DoubleDouble, DoubleDouble),
    ) -> (f64, f64) {
        let weight = normal_pdf(x);
        if weight == 0.0 {
            return (0.0, 0.0);
        }
        let (one_minus, one_plus, complement) = factors;
        let numerator = if rho >= 0.0 {
            DoubleDouble::from(y)
                .sub(DoubleDouble::from(x))
                .add(DoubleDouble::from(x).mul(one_minus))
        } else {
            DoubleDouble::from(y)
                .add(DoubleDouble::from(x))
                .sub(DoubleDouble::from(x).mul(one_plus))
        };
        let t = numerator.div(complement.sqrt());
        let density = normal_pdf(t.hi);
        let cdf = normal_cdf(t.hi) + density * t.lo;
        let value = weight * cdf;
        let allowance = weight * 2.0 * UNIT_ROUNDOFF * (cdf + density * t.hi.abs()) + 2.0 * UNIT_ROUNDOFF * value;
        (value, allowance)
    }

    /// `φ₂` with its exponent and scale carried in double-double, with an allowance for the reference's `exp`,
    /// its first-order correction and the division.
    fn reference_density(h: f64, k: f64, factors: (DoubleDouble, DoubleDouble, DoubleDouble)) -> (f64, f64) {
        let (one_minus, one_plus, complement) = factors;
        let gap = DoubleDouble::from(h.abs()).sub(DoubleDouble::from(k.abs()));
        let a = gap.mul(gap).mul(DoubleDouble::from(0.5));
        let product = DoubleDouble::from(h).mul(DoubleDouble::from(k));
        let b = if product.hi < 0.0 { product.negated() } else { product };
        let signed_factor = if h * k < 0.0 { one_minus } else { one_plus };
        let exponent = a.div(complement).add(b.div(signed_factor));
        let head = (-exponent.hi).exp() * (1.0 - exponent.lo);
        let scale = DoubleDouble::from(2.0).mul(PI_DOUBLE_DOUBLE).mul(complement.sqrt());
        let value = DoubleDouble::from(head).div(scale).hi;
        (
            value,
            3.0 * UNIT_ROUNDOFF * value + SMALLEST_SUBNORMAL * (1.0 + scale.hi.recip()),
        )
    }

    /// Neumaier's compensated sum, returned with `Σ|term|`. The rounding left over is the terms' own,
    /// at most a few ε of that magnitude.
    fn compensated_sum(terms: impl IntoIterator<Item = f64>) -> (f64, f64) {
        let (mut sum, mut compensation, mut magnitude) = (0.0_f64, 0.0_f64, 0.0_f64);
        for term in terms {
            let next = sum + term;
            compensation += if sum.abs() >= term.abs() {
                (sum - next) + term
            } else {
                (term - next) + sum
            };
            sum = next;
            magnitude += term.abs();
        }
        (sum + compensation, magnitude)
    }

    /// Composite Gauss-Legendre with `order` nodes per panel over `breaks`. It returns the integral and
    /// `Σ|term|`.
    fn composite(breaks: &[f64], order: usize, integrand: &dyn Fn(f64) -> f64) -> (f64, f64) {
        let (nodes, weights) = gauss_legendre(order);
        compensated_sum(breaks.windows(2).flat_map(|panel| {
            let (middle, half) = (0.5 * (panel[0] + panel[1]), 0.5 * (panel[1] - panel[0]));
            nodes
                .iter()
                .zip(weights.iter())
                .map(move |(&node, &weight)| half * weight * integrand(middle + half * node))
                .collect::<Vec<_>>()
        }))
    }

    /// An independent reference: the conditional form `∫_{−40}^{h} φ(x) Φ((k − ρx)/√(1 − ρ²)) dx`.
    ///
    /// - Panels: unit panels, plus panels graded geometrically about the conditional step `x = k/ρ`, of
    ///   width `√(1 − ρ²)/|ρ|`. So the step is resolved however close `|ρ|` is to 1.
    /// - Truncation: the mass below −40 is `Φ(−40)`, which is zero in binary64 (asserted by the caller).
    fn conditional_reference(h: f64, k: f64, rho: f64, order: usize) -> (f64, f64) {
        let (lower, upper) = (-40.0, h.min(40.0));
        if upper <= lower {
            return (0.0, 0.0);
        }
        let s = Correlation::from_rho(rho).complement().sqrt();
        let mut breaks = vec![lower, upper];
        let mut unit = lower + 1.0;
        while unit < upper {
            breaks.push(unit);
            unit += 1.0;
        }
        let (centre, width) = (k / rho, s / rho.abs());
        let mut offset = width;
        while offset < 80.0 {
            breaks.extend([centre - offset, centre + offset]);
            offset *= 2.0;
        }
        breaks.push(centre);
        breaks.retain(|point| *point >= lower && *point <= upper);
        breaks.sort_by(f64::total_cmp);
        breaks.dedup();
        composite(&breaks, order, &|x| {
            let numerator = if rho >= 0.0 {
                (k - x) + x * (1.0 - rho)
            } else {
                (k + x) - x * (1.0 + rho)
            };
            normal_pdf(x) * normal_cdf(numerator / s)
        })
    }

    /// The fixed-order Drezner-Wesolowsky rule on `θ ∈ [0, asin ρ]`, the form this module replaces. It is
    /// the positive control of the reference comparison.
    fn fixed_order_drezner_wesolowsky(h: f64, k: f64, rho: f64, order: usize) -> f64 {
        let (nodes, weights) = gauss_legendre(order);
        let alpha = rho.asin();
        let sum = nodes.iter().zip(weights.iter()).fold(0.0, |acc, (&node, &weight)| {
            let sine = (0.5 * alpha * (1.0 + node)).sin();
            acc + weight * ((sine * h * k - 0.5 * (h * h + k * k)) / (1.0 - sine * sine)).exp()
        });
        normal_cdf(h) * normal_cdf(k) + alpha * sum / (4.0 * PI)
    }

    #[test]
    fn core_order_is_the_smallest_meeting_the_bernstein_bound() {
        // The diamond claim rests on `cos t cosh t ≤ 1` over `[0, π/2]`.
        for step in 0..=1000 {
            let t = 0.5 * PI * f64::from(step) / 1000.0;
            assert!(t.cos() * t.cosh() <= 1.0 + f64::EPSILON, "t={t}");
        }
        let radius_squared = core_bernstein_radius_squared();
        let reach = 0.5
            * CORE_MAX_ANGLE
            * (1.0 + (0.5 * (radius_squared + 1.0 / radius_squared)).sqrt());
        assert!((reach - 0.5 * PI).abs() <= 8.0 * f64::EPSILON, "reach={reach}");
        let bound = |order: usize| {
            64.0 / 15.0 * radius_squared.powi(-(order as i32)) / (radius_squared - 1.0)
        };
        let order = core_order();
        assert_eq!(order, 9);
        assert!(bound(order) <= 2.0 * f64::EPSILON);
        assert!(bound(order - 1) > 2.0 * f64::EPSILON);
        assert_eq!(CORE_RULE.weights.len(), order);
    }

    #[test]
    fn core_rule_meets_its_truncation_bound_against_doubled_order() {
        let bounds = [-6.0, -3.0, -1.0, -0.25, 0.0, 0.5, 1.5, 3.0, 6.0];
        let rhos = [-0.5_f64, -0.35, -0.1, 0.1, 0.35, 0.5];
        let doubled = CoreRule::with_order(2 * core_order());
        let weak = CoreRule::with_order(4);
        let mut weak_exceeds = false;
        for &h in &bounds {
            for &k in &bounds {
                for &rho in &rhos {
                    let prefactor = rho.asin().abs() / (4.0 * PI);
                    // Truncation 2ε, plus the rounding of both sums (7ε a node) and of the two additions.
                    let allowance = prefactor
                        * (2.0 + 7.0 * (core_order() + doubled.weights.len()) as f64)
                        * f64::EPSILON
                        + 2.0 * f64::EPSILON;
                    let reference = core_cdf(h, k, rho, &doubled);
                    let production = core_cdf(h, k, rho, &CORE_RULE);
                    assert!(
                        (production - reference).abs() <= allowance,
                        "h={h} k={k} rho={rho} production={production:e} doubled={reference:e}"
                    );
                    weak_exceeds |= (core_cdf(h, k, rho, &weak) - reference).abs() > allowance;
                }
            }
        }
        assert!(weak_exceeds, "an order-4 rule must fail the same allowance somewhere");
    }

    #[test]
    fn cdf_matches_an_independent_conditional_reference_for_every_correlation() {
        assert_eq!(normal_cdf(-40.0), 0.0);
        let bounds = [-8.0, -4.0, -1.5, -0.3, 0.0, 0.7, 2.0, 5.0];
        let rhos = [
            -(1.0 - 1.0e-12),
            -(1.0 - 1.0e-6),
            -0.99,
            -0.925,
            -0.7,
            -0.5000001,
            -0.5,
            -0.3,
            -1.0e-12,
            1.0e-12,
            0.3,
            0.5,
            0.5000001,
            0.7,
            0.925,
            0.99,
            1.0 - 1.0e-6,
            1.0 - 1.0e-12,
        ];
        let check = |h: f64, k: f64, rho: f64| {
            let (reference, magnitude) = conditional_reference(h, k, rho, 40);
            let coarse = conditional_reference(h, k, rho, 20).0;
            let allowance = BIVARIATE_NORMAL_CDF_ERROR_BOUND + 4.0 * f64::EPSILON * magnitude + (reference - coarse).abs();
            let production = bivariate_normal_cdf(h, k, rho).unwrap();
            ((production - reference).abs(), allowance, reference)
        };
        for &h in &bounds {
            for &k in &bounds {
                for &rho in &rhos {
                    let (error, allowance, reference) = check(h, k, rho);
                    assert!(
                        error <= allowance,
                        "h={h} k={k} rho={rho} reference={reference:e} error={error:e} allowance={allowance:e}"
                    );
                }
            }
        }
        // Positive control: a nearly singular correlation with a nearly equal pair. There, the fixed
        // 20-point rule on θ misses the boundary layer at θ = π/2, while the rotated form does not.
        let (h, k, rho) = (0.25, 0.2505, 1.0 - 1.0e-12);
        let (error, allowance, reference) = check(h, k, rho);
        assert!(error <= allowance, "production error={error:e}");
        let fixed = fixed_order_drezner_wesolowsky(h, k, rho, 20);
        assert!(
            (fixed - reference).abs() > allowance,
            "the fixed 20-point rule must fail here: fixed={fixed:e} reference={reference:e}"
        );
    }

    #[test]
    fn reductions_agree_with_the_core_at_the_fixed_point() {
        let bounds = [-5.0, -1.2, 0.0, 0.8, 3.0];
        for &h in &bounds {
            for &k in &bounds {
                let positive = core_cdf(h, k, 0.5, &CORE_RULE);
                let split = h - k;
                let halved = core_cdf(split, k, -0.5, &CORE_RULE) + core_cdf(-split, h, -0.5, &CORE_RULE);
                assert!((positive - halved).abs() <= 1.5 * BIVARIATE_NORMAL_CDF_ERROR_BOUND, "h={h} k={k}");
                let negative = core_cdf(h, k, -0.5, &CORE_RULE);
                let (low, high) = if h <= k { (h, k) } else { (k, h) };
                let difference =
                    core_cdf(h + k, low, 0.5, &CORE_RULE) - core_cdf(h + k, -high, -0.5, &CORE_RULE);
                assert!((negative - difference).abs() <= 1.5 * BIVARIATE_NORMAL_CDF_ERROR_BOUND, "h={h} k={k}");
            }
        }
    }

    #[test]
    fn partials_integrate_back_to_the_distribution_function() {
        let panels = |lower: f64, upper: f64| -> Vec<f64> {
            (0..=8).map(|i| lower + (upper - lower) * f64::from(i) / 8.0).collect()
        };
        let agree = |exact: f64, integrand: &dyn Fn(f64) -> f64, breaks: &[f64]| {
            let (integral, magnitude) = composite(breaks, 40, integrand);
            let coarse = composite(breaks, 20, integrand).0;
            let allowance = 2.0 * BIVARIATE_NORMAL_CDF_ERROR_BOUND + 4.0 * f64::EPSILON * magnitude + (integral - coarse).abs();
            (exact - integral).abs() <= allowance
        };
        let cdf = |h: f64, k: f64, rho: f64| bivariate_normal_cdf(h, k, rho).unwrap();
        let partials = |h: f64, k: f64, rho: f64| bivariate_normal_cdf_partials(h, k, rho).unwrap();
        for &(h0, h1, k, rho) in &[
            (-3.0, 1.0, 0.4, 0.3),
            (-2.0, 2.0, -1.0, -0.8),
            (-1.0, 0.5, 0.5, 0.97),
            (0.0, 3.0, 1.0, -0.99),
        ] {
            let exact = cdf(h1, k, rho) - cdf(h0, k, rho);
            assert!(agree(exact, &|h| partials(h, k, rho).d_h, &panels(h0, h1)), "d_h {h0} {h1} {k} {rho}");
            assert!(agree(exact, &|h| partials(k, h, rho).d_k, &panels(h0, h1)), "d_k {h0} {h1} {k} {rho}");
        }
        for &(h, k, rho0, rho1) in &[
            (0.3, -0.7, -0.6, 0.4),
            (-1.2, -1.0, 0.2, 0.9),
            (1.5, -1.4, -0.95, -0.5),
        ] {
            let exact = cdf(h, k, rho1) - cdf(h, k, rho0);
            assert!(agree(exact, &|rho| partials(h, k, rho).d_rho, &panels(rho0, rho1)), "d_rho {h} {k}");
            assert!(agree(exact, &|rho| bivariate_normal_pdf(h, k, rho).unwrap(), &panels(rho0, rho1)));
        }
        // Positive control: the conditional partial with the correlation's sign flipped must fail.
        let (h0, h1, k, rho) = (-3.0, 1.0, 0.4, 0.3);
        let wrong = |h: f64| conditional_partial(h, k, Correlation::from_rho(-rho)).0;
        assert!(!agree(cdf(h1, k, rho) - cdf(h0, k, rho), &wrong, &panels(h0, h1)));
    }

    #[test]
    fn near_singular_correlation_approaches_the_limit_at_the_derived_rate() {
        // |Φ₂(ρ) − Φ₂(±1)| ≤ ∫ dr/(2π√(1 − r²)) = acos|ρ|/2π, and Φ₂ is non-decreasing in ρ.
        for &(h, k) in &[(0.4, 0.4), (-1.0, 2.0), (1.3, -0.2), (-2.5, -2.4), (2.0, -2.0)] {
            let upper = bivariate_normal_cdf(h, k, 1.0).unwrap();
            let lower = bivariate_normal_cdf(h, k, -1.0).unwrap();
            for gap in [1.0e-2_f64, 1.0e-6, 1.0e-10, 1.0e-14] {
                let rho = 1.0 - gap;
                let rate = rho.acos() / TAU + BIVARIATE_NORMAL_CDF_ERROR_BOUND;
                let near_upper = bivariate_normal_cdf(h, k, rho).unwrap();
                let near_lower = bivariate_normal_cdf(h, k, -rho).unwrap();
                assert!(near_upper <= upper + BIVARIATE_NORMAL_CDF_ERROR_BOUND && upper - near_upper <= rate, "h={h} k={k} gap={gap}");
                assert!(near_lower >= lower - BIVARIATE_NORMAL_CDF_ERROR_BOUND && near_lower - lower <= rate, "h={h} k={k} gap={gap}");
            }
        }
    }

    #[test]
    fn domain_errors_and_exact_branches() {
        assert_eq!(
            bivariate_normal_cdf(f64::NAN, 0.0, 0.1),
            Err(BivariateNormalError::NanBound { name: "h" })
        );
        assert!(matches!(
            bivariate_normal_cdf(0.0, 0.0, 1.01),
            Err(BivariateNormalError::CorrelationOutsideUnitInterval { .. })
        ));
        assert!(bivariate_normal_cdf(0.0, 0.0, f64::NAN).is_err());
        assert!(bivariate_normal_interval_probability(0.0, f64::NAN, 1.0, 0.2).is_err());
        assert!(matches!(
            bivariate_normal_pdf(0.0, 0.0, 1.0),
            Err(BivariateNormalError::SingularCorrelation { .. })
        ));
        assert!(bivariate_normal_cdf_partials(0.0, 0.0, -1.0).is_err());
        let (h, k) = (-0.35, 0.8);
        assert_eq!(bivariate_normal_cdf(h, k, 0.0).unwrap(), normal_cdf(h) * normal_cdf(k));
        assert_eq!(bivariate_normal_cdf(h, f64::INFINITY, 0.7).unwrap(), normal_cdf(h));
        assert_eq!(bivariate_normal_cdf(f64::NEG_INFINITY, k, -0.7).unwrap(), 0.0);
        // Sheppard's closed form at the origin.
        for rho in [-(1.0_f64 - 1.0e-13), -0.7, -0.2, 0.45, 0.8, 1.0 - 1.0e-13] {
            let expected = 0.25 + rho.asin() / TAU;
            let actual = bivariate_normal_cdf(0.0, 0.0, rho).unwrap();
            assert!((actual - expected).abs() <= BIVARIATE_NORMAL_CDF_ERROR_BOUND, "rho={rho} actual={actual:e}");
        }
    }

    #[test]
    fn interval_probability_retains_upper_tail_mass() {
        let expected = normal_cdf(-10.0) - normal_cdf(-12.0);
        for (h, rho, fraction) in [(f64::INFINITY, 0.3, 1.0), (0.0, 0.0, 0.5), (12.0, 1.0, 1.0), (0.0, -1.0, 1.0)] {
            let actual = bivariate_normal_interval_probability(h, 10.0, 12.0, rho).unwrap();
            assert!((actual / (fraction * expected) - 1.0).abs() <= 8.0 * f64::EPSILON, "h={h} rho={rho}");
        }
        let semi_infinite = bivariate_normal_interval_probability(0.0, 10.0, f64::INFINITY, 0.0).unwrap();
        assert!((semi_infinite / (0.5 * normal_cdf(-10.0)) - 1.0).abs() <= 8.0 * f64::EPSILON);
        let singular = bivariate_normal_cdf(12.0, -10.0, -1.0).unwrap();
        assert!((singular / expected - 1.0).abs() <= 8.0 * f64::EPSILON);
        for &(h, lower, upper, rho) in &[(0.3, -1.0, 0.5, 0.6), (-0.4, 0.2, 2.5, -0.8), (1.1, -3.0, -0.5, 0.95)] {
            let actual = bivariate_normal_interval_probability(h, lower, upper, rho).unwrap();
            let difference = bivariate_normal_cdf(h, upper, rho).unwrap() - bivariate_normal_cdf(h, lower, rho).unwrap();
            assert!((actual - difference).abs() <= 2.0 * BIVARIATE_NORMAL_CDF_ERROR_BOUND, "h={h} [{lower}, {upper}] rho={rho}");
        }
    }

    #[test]
    fn complement_route_agrees_with_the_plain_route() {
        let bounds = [-3.0_f64, -0.6, 0.0, 0.9, 2.5];
        let rhos = [-0.99_f64, -0.6, -0.2, 0.3, 0.7, 0.999];
        for &h in &bounds {
            for &k in &bounds {
                for &rho in &rhos {
                    let complement = Correlation::from_rho(rho).complement();
                    let plain = bivariate_normal_cdf(h, k, rho).unwrap();
                    let carried = bivariate_normal_cdf_with_complement(h, k, rho, complement).unwrap();
                    // Re-forming the vanishing factor by one division moves the rotation constant by ≤ ε relative.
                    // That moves the value by at most `(φ(u*)|u*| + φ₂·c)ε < ε`, on top of both calls' rounding.
                    assert!(
                        (plain - carried).abs() <= 2.0 * BIVARIATE_NORMAL_CDF_ERROR_BOUND + f64::EPSILON,
                        "h={h} k={k} rho={rho}"
                    );
                    let p = bivariate_normal_cdf_partials(h, k, rho).unwrap();
                    let q = bivariate_normal_cdf_partials_with_complement(h, k, rho, complement).unwrap();
                    // The exponent Q carries ≤ 3ε relative from the re-formed factor, and the arguments of Φ ≤ 2ε
                    // per unit of |x|. Each side also has its own few ulps.
                    let exponent = (h * h - 2.0 * rho * h * k + k * k) / (2.0 * complement);
                    let density_allowance = (8.0 + 6.0 * exponent) * f64::EPSILON * p.d_rho.max(q.d_rho);
                    assert!((p.d_rho - q.d_rho).abs() <= density_allowance, "d_rho h={h} k={k} rho={rho}");
                    assert!((p.d_h - q.d_h).abs() <= 4.0 * (1.0 + h.abs()) * f64::EPSILON, "d_h h={h} k={k} rho={rho}");
                    assert!((p.d_k - q.d_k).abs() <= 4.0 * (1.0 + k.abs()) * f64::EPSILON, "d_k h={h} k={k} rho={rho}");
                }
            }
        }
        assert!(matches!(
            bivariate_normal_cdf_with_complement(0.0, 0.0, 0.5, -1.0e-3),
            Err(BivariateNormalError::ComplementOutsideUnitInterval { .. })
        ));
        assert!(bivariate_normal_cdf_partials_with_complement(0.0, 0.0, 1.0, 0.0).is_err());
    }

    #[test]
    fn complement_keeps_the_digits_a_rounded_correlation_loses() {
        // The caller's correlation rounds to 1, while its complement is resolved.
        let (rho, complement) = (1.0_f64, 2.0e-17_f64);
        // Sheppard at the origin, through the half angle: Φ₂(0,0;ρ) = ½ − asin(√((1 − ρ)/2))/π, with 1 − ρ = c/2.
        let expected = 0.5 - (0.5 * complement.sqrt()).asin() / PI;
        let carried = bivariate_normal_cdf_with_complement(0.0, 0.0, rho, complement).unwrap();
        assert!((carried - expected).abs() <= BIVARIATE_NORMAL_CDF_ERROR_BOUND, "carried={carried:e} expected={expected:e}");
        let plain = bivariate_normal_cdf(0.0, 0.0, rho).unwrap();
        assert!((plain - expected).abs() > BIVARIATE_NORMAL_CDF_ERROR_BOUND, "the rounded correlation must lose these digits");

        // Partials at a nearly equal pair, against an independent route: ρ = √(1 − c), so
        // 1 − ρ = −expm1(½ log1p(−c)), and φ₂ = φ(h)·φ(t)/√c with t = (k − ρh)/√c.
        let (h, k) = (0.3_f64, 0.3 + 2.0e-9);
        let one_minus_rho = -libm::expm1(0.5 * libm::log1p(-complement));
        let t = ((k - h) + h * one_minus_rho) / complement.sqrt();
        let expected_d_h = normal_pdf(h) * normal_cdf(t);
        let expected_d_rho = normal_pdf(h) * normal_pdf(t) / complement.sqrt();
        let partials = bivariate_normal_cdf_partials_with_complement(h, k, rho, complement).unwrap();
        assert!(
            (partials.d_h - expected_d_h).abs() <= 4.0 * f64::EPSILON * normal_pdf(h),
            "d_h={:e} expected={expected_d_h:e}",
            partials.d_h
        );
        assert!(
            (partials.d_rho / expected_d_rho - 1.0).abs() <= 16.0 * f64::EPSILON,
            "d_rho={:e} expected={expected_d_rho:e}",
            partials.d_rho
        );
        assert!(bivariate_normal_cdf_partials(h, k, rho).is_err());
        // Positive control: the nearest double below 1 forms 1 − ρ² = 2.2e−16 instead of 2e−17, and misses d_h.
        let rounded = bivariate_normal_cdf_partials(h, k, 1.0 - 0.5 * f64::EPSILON).unwrap();
        assert!((rounded.d_h - expected_d_h).abs() > 4.0 * f64::EPSILON * normal_pdf(h));
    }

    #[test]
    fn core_rule_meets_the_gauss_legendre_contract_at_the_core_order() {
        let order = core_order();
        let (nodes, weights) = gauss_legendre(order);
        let mut resolves_offsets = true;
        for (&node, &weight) in nodes.iter().zip(&weights) {
            // Newton in double-double from the computed node converges quadratically to the true root.
            let mut root = DoubleDouble::from(node);
            let mut remaining = 3;
            while remaining > 0 {
                let (value, slope) = legendre_double_double(order, root);
                root = root.sub(value.div(slope));
                remaining -= 1;
            }
            let slope = legendre_double_double(order, root).1;
            let reference_weight = DoubleDouble::from(2.0)
                .div(DoubleDouble::from(1.0).sub(root.mul(root)).mul(slope.mul(slope)));
            let node_error = |candidate: f64| DoubleDouble::from(candidate).sub(root).hi.abs();
            let weight_error =
                |candidate: f64| DoubleDouble::from(candidate).sub(reference_weight).div(reference_weight).hi.abs();
            assert!(
                node_error(node) <= CORE_RULE_NODE_ERROR,
                "node {node:.17e} errs by {:e}",
                node_error(node)
            );
            assert!(
                weight_error(weight) <= CORE_RULE_WEIGHT_ERROR,
                "weight {weight:.17e} errs by {:e}",
                weight_error(weight)
            );
            // Positive control: the reference resolves an 8-ulp node offset and a weight moved by three contracts.
            resolves_offsets &= node_error(node + 8.0 * f64::EPSILON) > CORE_RULE_NODE_ERROR;
            resolves_offsets &= weight_error(weight * (1.0 + 3.0 * CORE_RULE_WEIGHT_ERROR)) > CORE_RULE_WEIGHT_ERROR;
        }
        assert!(resolves_offsets, "the double-double reference must resolve offsets beyond the contract");
    }

    #[test]
    fn cdf_error_bound_covers_the_complementary_orthant_identity() {
        // Φ₂(h, k; ρ) + Φ₂(h, −k; −ρ) = Φ(h). For |ρ| > ½ the two evaluations take different reductions, so their
        // errors do not cancel by construction. (On the core the two node sums are identical.)
        let bounds = [-8.0_f64, -3.0, -0.7, 0.0, 0.4, 2.0, 6.0];
        let rhos = [0.500_000_1_f64, 0.6, 0.9, 0.999, 1.0 - 1.0e-9, 1.0 - 1.0e-15];
        let violation = |h: f64, k: f64, rho: f64, complement: Option<f64>, rule: &CoreRule| {
            let (positive, negative) = match complement {
                None => (Correlation::from_rho(rho), Correlation::from_rho(-rho)),
                Some(c) => (
                    Correlation::from_complement(rho, c),
                    Correlation::from_complement(-rho, c),
                ),
            };
            let total = DoubleDouble::two_sum(
                cdf_on_rule(h, k, positive, rule),
                cdf_on_rule(h, -k, negative, rule),
            );
            let marginal = normal_cdf(h);
            let gap = total.sub(DoubleDouble::from(marginal)).hi.abs();
            // Φ(h) errs by 2uΦ + 2u|h|φ(h).
            let allowance = 2.0 * BIVARIATE_NORMAL_CDF_ERROR_BOUND
                + 2.0 * UNIT_ROUNDOFF * (marginal + h.abs() * normal_pdf(h));
            (gap, allowance)
        };
        let control_rule = CoreRule::with_order(3);
        let mut control_fails = false;
        for &h in &bounds {
            for &k in &bounds {
                for &rho in &rhos {
                    let complement = Correlation::from_rho(rho).complement();
                    for route in [None, Some(complement)] {
                        let (gap, allowance) = violation(h, k, rho, route, &CORE_RULE);
                        assert!(
                            gap <= allowance,
                            "h={h} k={k} rho={rho} route={route:?} gap={gap:e} allowance={allowance:e}"
                        );
                        control_fails |= violation(h, k, rho, route, &control_rule).0 > allowance;
                    }
                }
            }
        }
        assert!(control_fails, "an order-3 core rule must break the same allowance somewhere");
    }

    #[test]
    fn partial_rounding_bounds_cover_a_double_double_reference() {
        let bounds = [-10.0_f64, -4.5, -1.3, 0.0, 0.6, 2.2, 5.0, 9.5, 25.0];
        let rhos = [-0.9999_f64, -0.9, -0.4, 0.2, 0.7, 0.99, 0.999_999];
        let mut density_needs_its_exponent_term = false;
        let mut partial_bound_is_resolved = false;
        let mut check = |h: f64, k: f64, rho: f64, complement: Option<f64>| {
            let partials = match complement {
                None => bivariate_normal_cdf_partials(h, k, rho),
                Some(c) => bivariate_normal_cdf_partials_with_complement(h, k, rho, c),
            }
            .unwrap();
            let factors = reference_factors(rho, complement);
            for (production, rounding, (reference, allowance), name) in [
                (
                    partials.d_h,
                    partials.d_h_rounding,
                    reference_conditional_partial(h, k, rho, factors),
                    "d_h",
                ),
                (
                    partials.d_k,
                    partials.d_k_rounding,
                    reference_conditional_partial(k, h, rho, factors),
                    "d_k",
                ),
            ] {
                let gap = (production - reference).abs();
                assert!(
                    gap <= rounding + allowance,
                    "{name} h={h} k={k} rho={rho} c={complement:?} gap={gap:e} rounding={rounding:e}"
                );
                partial_bound_is_resolved |= gap > (rounding + allowance) / 64.0;
            }
            let (reference, allowance) = reference_density(h, k, factors);
            let gap = (partials.d_rho - reference).abs();
            assert!(
                gap <= partials.d_rho_rounding + allowance,
                "d_rho h={h} k={k} rho={rho} c={complement:?} gap={gap:e} rounding={:e}",
                partials.d_rho_rounding
            );
            density_needs_its_exponent_term |= gap > 8.0 * UNIT_ROUNDOFF * partials.d_rho + allowance;
        };
        for &h in &bounds {
            for &k in &bounds {
                for &rho in &rhos {
                    check(h, k, rho, None);
                    check(h, k, rho, Some(Correlation::from_rho(rho).complement()));
                }
            }
        }
        check(0.3, 0.3 + 2.0e-9, 1.0, Some(2.0e-17));
        // Positive controls: the exponent's 8uE term is needed somewhere, and some gap exceeds 1/64 of its bound.
        assert!(density_needs_its_exponent_term, "dropping the 8uE term must break the density bound somewhere");
        assert!(partial_bound_is_resolved, "the reference must resolve the partial bounds to 1/64");
    }

    #[test]
    fn partials_keep_a_subnormal_complement_resolved() {
        // At |ρ| = 1 the vanishing factor of the smallest subnormal complement rounds to zero. So a determinant formed
        // as the product of the factors collapses. It gave a zero scale, and NaN or infinite partials.
        let complement = f64::from_bits(1);
        assert_eq!((complement / 2.0) * 2.0, 0.0, "the factor product underflows at this complement");
        let root = complement.sqrt();
        // A tie at ρ = 1: t = 0, so ∂_hΦ₂ = ½φ(h), and φ₂ = exp(−h²/2)/(2π√c) is finite and representable.
        let h = 0.4_f64;
        let tie = bivariate_normal_cdf_partials_with_complement(h, h, 1.0, complement).unwrap();
        assert!(tie.d_h.is_finite() && tie.d_k.is_finite() && tie.d_rho.is_finite(), "{tie:?}");
        assert!((tie.d_h - 0.5 * normal_pdf(h)).abs() <= tie.d_h_rounding, "{tie:?}");
        assert_eq!(tie.d_k, tie.d_h);
        let expected_density = normal_pdf(h) / ((2.0 * PI).sqrt() * root);
        // The reference's own rounding: normal_pdf (5u) and the square root, product and division (3u).
        assert!(
            (tie.d_rho - expected_density).abs() <= tie.d_rho_rounding + 8.0 * UNIT_ROUNDOFF * expected_density,
            "d_rho={:e} expected={expected_density:e}",
            tie.d_rho
        );
        assert!(tie.d_h_rounding.is_finite() && tie.d_rho_rounding.is_finite(), "{tie:?}");
        // Opposite signs at ρ = 1: the step limits φ(h)·1[h < k] and φ(k)·1[k < h], and a vanishing density.
        let (x, y) = (0.3_f64, -0.2_f64);
        let split = bivariate_normal_cdf_partials_with_complement(x, y, 1.0, complement).unwrap();
        assert_eq!(split.d_h, 0.0, "{split:?}");
        assert!((split.d_k - normal_pdf(y)).abs() <= split.d_k_rounding, "{split:?}");
        assert_eq!(split.d_rho, 0.0, "{split:?}");
        assert!(split.d_rho_rounding.is_finite(), "{split:?}");
        // The value takes the exact singular branch.
        assert_eq!(
            bivariate_normal_cdf_with_complement(x, y, 1.0, complement).unwrap(),
            normal_cdf(y)
        );
    }

    #[test]
    fn partial_rounding_bounds_scale_with_the_partials_in_the_lower_tail() {
        // At the cells where fr-kernel's biased kernels need relative accuracy (ρ ≤ 0 with both constraints active),
        // t = (k − ρh)/√(1 − ρ²) ≤ 0 and φ(t)/Φ(t) ≤ |t| + 1. So each bound is at most u·(8 + (|t| + 1)(8|t| + 4|h|))
        // of its partial, plus the smallest subnormal.
        let cells = [
            (-3.0_f64, -3.0_f64, -0.9_f64),
            (-3.0, -3.0, -0.5),
            (-5.0, -5.0, -0.5),
            (-8.0, -8.0, -0.5),
            (-6.0, 1.0, -0.8),
            (-4.2426, 0.7071, -0.4),
        ];
        let mut sup_scale_certifies_nothing = false;
        for &(h, k, rho) in &cells {
            let partials = bivariate_normal_cdf_partials(h, k, rho).unwrap();
            let sigma = ((1.0 - rho) * (1.0 + rho)).sqrt();
            for (partial, rounding, x, y) in [
                (partials.d_h, partials.d_h_rounding, h, k),
                (partials.d_k, partials.d_k_rounding, k, h),
            ] {
                let t = (y - rho * x) / sigma;
                assert!(t <= 0.0, "cell ({h}, {k}, {rho}) is outside the lower-tail regime");
                let relative = UNIT_ROUNDOFF * (8.0 + (t.abs() + 1.0) * (8.0 * t.abs() + 4.0 * x.abs()));
                assert!(
                    rounding <= relative * partial + SMALLEST_SUBNORMAL,
                    "({h}, {k}, {rho}) partial={partial:e} rounding={rounding:e} relative={relative:e}"
                );
                // Positive control: the previous sup-scale bound u·φ(x)·8·sup|t|φ(t) exceeds the partial itself here,
                // so it certified no relative digit.
                sup_scale_certifies_nothing |= UNIT_ROUNDOFF * normal_pdf(x) * 8.0 * ARGUMENT_SENSITIVITY > partial;
            }
        }
        assert!(sup_scale_certifies_nothing, "the sup-scale bound must exceed some lower-tail partial");
    }
}
