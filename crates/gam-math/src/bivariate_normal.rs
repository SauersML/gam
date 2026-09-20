//! The bivariate standard normal distribution function
//! `Φ₂(h, k; ρ) = P(X ≤ h, Y ≤ k)`, for standard normal `X, Y` with correlation
//! `ρ`, together with its density and its analytic partials (#2946).
//!
//! # One route
//!
//! Every value of `Φ₂` comes from the apex tree of `apex_form` (#3158, #3253): exact branches for infinite bounds and
//! `ρ ∈ {−1, 0, 1}`, then an exact reduction onto leaves `Φ₂ = c·φ₂·K` whose integrands never cancel. Each value
//! carries an a-priori bound on its error at the computed arguments, so [`bivariate_normal_cdf`],
//! [`bivariate_normal_cdf_with_complement`] and [`bivariate_normal_interval_probability`] all return a
//! [`BoundedProbability`].
//!
//! # Contract
//!
//! - **Error bound.** The bound scales with the value however small it is (#3226). The tree's `exp`, Mills-ratio and
//!   `Φ` contracts, and the one difference it takes, at `ρ > 0` with its minuend below twice the value, are stated in
//!   `apex_form`. [`BivariateNormalPartials`] carries a per-evaluation rounding bound next to each partial. Neither
//!   includes the caller's own argument error.
//! - **Rounding model.** Under round-to-nearest every `+ − × ÷ √` errs by at most `u = ε/2` of its result. The
//!   partials add one ulp, `2u`, per libm `exp` or `erfc`: the contract `gaussian_activation` states for its own
//!   bounds. A count of `k` rounded operations is carried as Wilkinson's `γ_k` ([`accumulation_growth`]), which sits
//!   above the first-order `k·u` and agrees with it to `O(u²)`.
//! - **Exact branches.** Infinite bounds and `ρ ∈ {−1, 0, 1}` are exact special cases.
//! - **Correlation input.** A caller who resolves `1 − ρ²` more finely than `ρ` passes it to the `_with_complement`
//!   entry points. Every `1 ∓ ρ` is then derived from it, and the input conditioning `φ₂·δρ` of a rounded `ρ`
//!   near `±1` never enters. From `ρ` alone, `1 − ρ² = (1 − ρ)(1 + ρ)` is formed in binary64, and the bound charges
//!   that rounding.
//! - **Projection.** Every result is projected onto `[0, 1]`, which contains the truth, so the projection never
//!   increases the error.

mod apex_form;
#[cfg(test)]
mod positive_form_tests;

use crate::double_double::SMALLEST_SUBNORMAL;
use crate::probability::{normal_cdf, normal_pdf};
use crate::roundoff::{UNIT_ROUNDOFF, accumulation_growth};
use std::f64::consts::PI;
use std::fmt;

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
/// arguments. The model is the module's (Contract), and the caller's argument error is not included.
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

/// A probability with a bound on its error at the computed arguments.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundedProbability {
    pub value: f64,
    /// A bound on `|value − Φ₂|`.
    pub rounding: f64,
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

/// A correlation with its singular factors `1 − ρ` and `1 + ρ`, their product `1 − ρ²`, and a bound on that product's
/// relative error against `1 − ρ²` at the stored `ρ`.
///
/// - From `ρ` alone, each factor is formed directly, and is exact near its own singular end. The product rounds three
///   times: `(1 − ρ)(1 + ρ)` is within `γ₃` of `1 − ρ²`.
/// - From a caller's `1 − ρ²`, the product is the caller's value itself, taken as exact, and the vanishing factor is
///   `(1 − ρ²)/(1 + |ρ|)`. So a correlation that rounds to `±1` still carries the complement the caller resolved. A
///   complement whose vanishing factor underflows, such as the smallest subnormal at `|ρ| = 1`, keeps a nonzero
///   determinant instead of collapsing to `0·2`.
#[derive(Clone, Copy)]
struct Correlation {
    rho: f64,
    one_minus: f64,
    one_plus: f64,
    complement: f64,
    complement_error: f64,
}

impl Correlation {
    fn from_rho(rho: f64) -> Self {
        let (one_minus, one_plus) = (1.0 - rho, 1.0 + rho);
        Self {
            rho,
            one_minus,
            one_plus,
            complement: one_minus * one_plus,
            complement_error: accumulation_growth(3),
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
            complement_error: 0.0,
        }
    }

    fn negated(self) -> Self {
        Self {
            rho: -self.rho,
            one_minus: self.one_plus,
            one_plus: self.one_minus,
            ..self
        }
    }

    fn complement(self) -> f64 {
        self.complement
    }
}

/// `Φ₂(h, k; ρ) = P(X ≤ h, Y ≤ k)` through the apex tree, with a bound on its error at the computed arguments. Bounds
/// may be infinite, and `ρ ∈ [−1, 1]`. The bound charges the rounding of `1 − ρ²` formed from `ρ` (module docs).
pub fn bivariate_normal_cdf(h: f64, k: f64, rho: f64) -> Result<BoundedProbability, BivariateNormalError> {
    validate(&[("h", h), ("k", k)], rho)?;
    Ok(apex_form::orthant(h, k, Correlation::from_rho(rho)))
}

/// `Φ₂(h, k; ρ)` for a caller who resolves `1 − ρ²` more finely than `ρ` itself, such as `Δ/(AB)` with
/// `Δ = AB − r²` formed by fma, through the apex tree, with a bound on its error at the computed arguments.
///
/// Every `1 ∓ ρ` is derived from `complement`, so a correlation rounded to `±1` keeps the complement's digits.
/// `complement == 0` selects the singular branch by the sign of `ρ`. `complement` must be `1 − ρ²` of the same
/// correlation, to rounding: the bound takes it as exact and does not include the caller's argument error.
pub fn bivariate_normal_cdf_with_complement(
    h: f64,
    k: f64,
    rho: f64,
    complement: f64,
) -> Result<BoundedProbability, BivariateNormalError> {
    validate(&[("h", h), ("k", k)], rho)?;
    validate_complement(complement)?;
    Ok(apex_form::orthant(
        h,
        k,
        Correlation::from_complement(rho, complement),
    ))
}

/// `P(X ≤ h, lower ≤ Y ≤ upper)`, with a bound on its error at the computed arguments.
///
/// A finite interval is a difference of two orthants of the apex tree taken on the tail side of zero: the reflection
/// `Y → −Y` when `lower ≥ 0`. So a small interval in the upper tail keeps its mass instead of being subtracted from
/// `Φ(h)`. A half-infinite interval is one orthant, with the bound of [`bivariate_normal_cdf`]. A difference's bound is
/// the sum of its orthants' plus the subtraction's rounding.
pub fn bivariate_normal_interval_probability(
    h: f64,
    lower: f64,
    upper: f64,
    rho: f64,
) -> Result<BoundedProbability, BivariateNormalError> {
    validate(&[("h", h), ("lower", lower), ("upper", upper)], rho)?;
    if !(lower < upper) {
        return Ok(BoundedProbability {
            value: 0.0,
            rounding: 0.0,
        });
    }
    let correlation = Correlation::from_rho(rho);
    let reflected = correlation.negated();
    let difference = |minuend: BoundedProbability, subtrahend: BoundedProbability| {
        let value = minuend.value - subtrahend.value;
        BoundedProbability {
            value: value.clamp(0.0, 1.0),
            rounding: minuend.rounding + subtrahend.rounding + UNIT_ROUNDOFF * value.abs(),
        }
    };
    Ok(if lower == f64::NEG_INFINITY {
        apex_form::orthant(h, upper, correlation)
    } else if upper == f64::INFINITY {
        apex_form::orthant(h, -lower, reflected)
    } else if lower >= 0.0 {
        difference(
            apex_form::orthant(h, -lower, reflected),
            apex_form::orthant(h, -upper, reflected),
        )
    } else {
        difference(
            apex_form::orthant(h, upper, correlation),
            apex_form::orthant(h, lower, correlation),
        )
    })
}

/// `φ₂(h, k; ρ)` for `|ρ| < 1`, with its rounding bound (derived at [`BivariateNormalPartials`]). The quadratic
/// form is carried with non-negative terms, `a = (|h| − |k|)²/2` over `1 − ρ²` and `b = |hk|` over `1 + σρ`, with
/// `σ = sign(hk)`, so it never cancels and never forms `∞ − ∞`.
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
    use crate::probability::{
        NORMAL_CDF_RELATIVE_ERROR, NORMAL_CDF_UNDERFLOW_FLOOR, normal_cdf_and_pdf, normal_pdf_bounded,
    };
    use crate::special::gauss_legendre;
    use std::f64::consts::TAU;

    /// `sup |x| φ(x) = 1/√(2πe) = 0.241970…`, rounded up: the scale of a partial's bound taken at the sup instead of at
    /// the partial.
    const ARGUMENT_SENSITIVITY: f64 = 0.2420;

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

    /// An independent reference: the conditional form `∫_{−40}^{h} φ(x) Φ((k − ρx)/√(1 − ρ²)) dx`, with its value, the
    /// magnitude of its terms, and a first-order bound on its evaluation error. It shares no step with the apex tree but
    /// the certified Gauss-Legendre nodes.
    ///
    /// - Panels: in the distance `v = h − x` from the upper limit, unit panels in `x`, plus panels graded geometrically
    ///   about the conditional step `x = k/ρ`, of width `√(1 − ρ²)/|ρ|`. So the step is resolved however close `|ρ|` is
    ///   to 1. Every break is on the grid `2⁻⁴⁰` and `v = 0` is one, so with `v ≤ 128` every panel's midpoint and
    ///   half-width are exact and the panels tile `[0, V]` exactly.
    /// - Truncation: the mass below −40 is `Φ(−40)`, which is zero in binary64 (asserted by the caller). The rule's own
    ///   truncation is the caller's, from the same panels at half the order.
    /// - Evaluation: the certified rule's node and weight errors; `φ(x)` within `normal_pdf_bounded`'s bound; `Φ(t)` within
    ///   the table route's, and `t` within `δn/s + γ₄|t|` (the numerator's three roundings and one in `1 ∓ ρ`, then `√`,
    ///   the complement's `γ₃` and the quotient), which moves `Φ` by `φ(t)` per unit; a node's position error moves the
    ///   integrand `f` by `|f′| ≤ |x| f + φ(x)φ(t)|ρ|/s` per unit; the product and the compensated sum.
    fn conditional_reference(h: f64, k: f64, rho: f64, order: usize) -> (f64, f64, f64) {
        let (lower, upper) = (-40.0, h.min(40.0));
        if upper <= lower {
            return (0.0, 0.0, 0.0);
        }
        let s = Correlation::from_rho(rho).complement().sqrt();
        let grid = 2.0_f64.powi(40);
        let distance = |x: f64| ((upper - x) * grid).round() / grid;
        let mut breaks = vec![0.0, distance(lower)];
        let mut unit = lower + 1.0;
        while unit < upper {
            breaks.push(distance(unit));
            unit += 1.0;
        }
        let (centre, width) = (k / rho, s / rho.abs());
        let mut offset = width;
        while offset < 80.0 {
            breaks.extend([distance(centre - offset), distance(centre + offset)]);
            offset *= 2.0;
        }
        breaks.push(distance(centre));
        let far = distance(lower);
        breaks.retain(|point| *point >= 0.0 && *point <= far);
        breaks.sort_by(f64::total_cmp);
        breaks.dedup();
        let rule = apex_form::rule_of_order(order);
        let (one_minus, one_plus) = (1.0 - rho, 1.0 + rho);
        let mut terms = Vec::with_capacity(breaks.len() * order);
        let mut error = 0.0;
        for panel in breaks.windows(2) {
            let (middle, half) = (0.5 * (panel[0] + panel[1]), 0.5 * (panel[1] - panel[0]));
            for (&node, &weight) in rule.nodes.iter().zip(&rule.weights) {
                let v = middle + half * node;
                let x = upper - v;
                let (gap, lean) = if rho >= 0.0 { (k - x, x * one_minus) } else { (k + x, -(x * one_plus)) };
                let numerator = gap + lean;
                let numerator_error = UNIT_ROUNDOFF * (gap.abs() + 2.0 * lean.abs() + numerator.abs());
                let t = numerator / s;
                let t_error = numerator_error / s + accumulation_growth(4) * t.abs();
                let (cdf, cdf_density) = normal_cdf_and_pdf(t);
                let cdf_error = NORMAL_CDF_RELATIVE_ERROR * cdf + NORMAL_CDF_UNDERFLOW_FLOOR + cdf_density * t_error;
                let (pdf, pdf_error) = normal_pdf_bounded(x);
                let value = pdf * cdf;
                let scale = half * weight;
                let slope = x.abs() * value + pdf * cdf_density * rho.abs() / s;
                let position = half * rule.node_error + accumulation_growth(2) * (v.abs() + x.abs());
                terms.push(scale * value);
                error += scale * (pdf_error * cdf + pdf * cdf_error + slope * position + 2.0 * UNIT_ROUNDOFF * value);
            }
        }
        let (sum, magnitude) = compensated_sum(terms);
        let error = error + (rule.weight_relative_error + 4.0 * f64::EPSILON) * magnitude;
        (sum, magnitude, error)
    }

    /// The fixed-order Drezner-Wesolowsky rule on `θ ∈ [0, asin ρ]`, a form that loses digits as `|ρ| → 1`. It is
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
    fn cdf_matches_an_independent_conditional_reference_for_every_correlation() {
        assert_eq!(normal_cdf(-40.0), 0.0);
        let bounds = [-9.0, -8.0, -4.0, -1.5, -0.3, 0.0, 0.3, 0.7, 2.0, 5.0, 9.0];
        let rhos = [
            -(1.0 - 1.0e-12),
            -(1.0 - 1.0e-6),
            -0.999_999,
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
            0.999_999,
            1.0 - 1.0e-6,
            1.0 - 1.0e-12,
        ];
        // The reference errs by its evaluation bound and by its truncation, which the gap to the same panels at half the
        // order bounds. The production value errs by its own per-call bound.
        let check = |h: f64, k: f64, rho: f64| {
            let (reference, _, evaluation) = conditional_reference(h, k, rho, 40);
            let coarse = conditional_reference(h, k, rho, 20).0;
            let production = bivariate_normal_cdf(h, k, rho).unwrap();
            let allowance = production.rounding + evaluation + (reference - coarse).abs();
            ((production.value - reference).abs(), allowance, reference)
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
        // 20-point rule on θ misses the boundary layer at θ = π/2, while the apex tree does not.
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
    fn partials_integrate_back_to_the_distribution_function() {
        let panels = |lower: f64, upper: f64| -> Vec<f64> {
            (0..=8).map(|i| lower + (upper - lower) * f64::from(i) / 8.0).collect()
        };
        // `exact` is a difference of two bounded values: their bounds plus the subtraction's rounding.
        let difference = |upper: BoundedProbability, lower: BoundedProbability| {
            let value = upper.value - lower.value;
            (value, upper.rounding + lower.rounding + UNIT_ROUNDOFF * value.abs())
        };
        let agree = |(exact, rounding): (f64, f64), integrand: &dyn Fn(f64) -> f64, breaks: &[f64]| {
            let (integral, magnitude) = composite(breaks, 40, integrand);
            let coarse = composite(breaks, 20, integrand).0;
            let allowance = rounding + 4.0 * f64::EPSILON * magnitude + (integral - coarse).abs();
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
            let exact = difference(cdf(h1, k, rho), cdf(h0, k, rho));
            assert!(agree(exact, &|h| partials(h, k, rho).d_h, &panels(h0, h1)), "d_h {h0} {h1} {k} {rho}");
            assert!(agree(exact, &|h| partials(k, h, rho).d_k, &panels(h0, h1)), "d_k {h0} {h1} {k} {rho}");
        }
        for &(h, k, rho0, rho1) in &[
            (0.3, -0.7, -0.6, 0.4),
            (-1.2, -1.0, 0.2, 0.9),
            (1.5, -1.4, -0.95, -0.5),
        ] {
            let exact = difference(cdf(h, k, rho1), cdf(h, k, rho0));
            assert!(agree(exact, &|rho| partials(h, k, rho).d_rho, &panels(rho0, rho1)), "d_rho {h} {k}");
            assert!(agree(exact, &|rho| bivariate_normal_pdf(h, k, rho).unwrap(), &panels(rho0, rho1)));
        }
        // Positive control: the conditional partial with the correlation's sign flipped must fail.
        let (h0, h1, k, rho) = (-3.0, 1.0, 0.4, 0.3);
        let wrong = |h: f64| conditional_partial(h, k, Correlation::from_rho(-rho)).0;
        assert!(!agree(difference(cdf(h1, k, rho), cdf(h0, k, rho)), &wrong, &panels(h0, h1)));
    }

    #[test]
    fn near_singular_correlation_approaches_the_limit_at_the_derived_rate() {
        // |Φ₂(ρ) − Φ₂(±1)| ≤ ∫ dr/(2π√(1 − r²)) = acos|ρ|/2π, and Φ₂ is non-decreasing in ρ.
        for &(h, k) in &[(0.4, 0.4), (-1.0, 2.0), (1.3, -0.2), (-2.5, -2.4), (2.0, -2.0)] {
            let upper = bivariate_normal_cdf(h, k, 1.0).unwrap();
            let lower = bivariate_normal_cdf(h, k, -1.0).unwrap();
            for gap in [1.0e-2_f64, 1.0e-6, 1.0e-10, 1.0e-14] {
                let rho = 1.0 - gap;
                let rate = rho.acos() / TAU;
                let near_upper = bivariate_normal_cdf(h, k, rho).unwrap();
                let near_lower = bivariate_normal_cdf(h, k, -rho).unwrap();
                let (upper_slack, lower_slack) =
                    (upper.rounding + near_upper.rounding, lower.rounding + near_lower.rounding);
                assert!(
                    near_upper.value <= upper.value + upper_slack && upper.value - near_upper.value <= rate + upper_slack,
                    "h={h} k={k} gap={gap}"
                );
                assert!(
                    near_lower.value >= lower.value - lower_slack && near_lower.value - lower.value <= rate + lower_slack,
                    "h={h} k={k} gap={gap}"
                );
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
        assert!(matches!(
            bivariate_normal_cdf_with_complement(0.0, 0.0, 0.5, -1.0e-3),
            Err(BivariateNormalError::ComplementOutsideUnitInterval { .. })
        ));
        let (h, k) = (-0.35, 0.8);
        let phi = |x: f64| normal_cdf_and_pdf(x).0;
        assert_eq!(bivariate_normal_cdf(h, k, 0.0).unwrap().value, phi(h) * phi(k));
        assert_eq!(bivariate_normal_cdf(h, f64::INFINITY, 0.7).unwrap().value, phi(h));
        assert_eq!(
            bivariate_normal_cdf(f64::NEG_INFINITY, k, -0.7).unwrap(),
            BoundedProbability { value: 0.0, rounding: 0.0 }
        );
        // Sheppard's closed form at the origin. The reference `¼ + asin(ρ)/2π` errs by the one-ulp `asin`, `2π` and the
        // quotient, `4u` of the quotient, and by the addition's `u`.
        for rho in [-(1.0_f64 - 1.0e-13), -0.7, -0.2, 0.45, 0.8, 1.0 - 1.0e-13] {
            let quotient = rho.asin() / TAU;
            let expected = 0.25 + quotient;
            let reference_rounding = 4.0 * UNIT_ROUNDOFF * quotient.abs() + UNIT_ROUNDOFF * expected;
            let actual = bivariate_normal_cdf(0.0, 0.0, rho).unwrap();
            assert!(
                (actual.value - expected).abs() <= actual.rounding + reference_rounding,
                "rho={rho} actual={actual:?} expected={expected:e}"
            );
        }
    }

    #[test]
    fn interval_probability_retains_upper_tail_mass() {
        // `Φ(−10) − Φ(−12)` at 40 digits (mpmath).
        let truth = 7.619_853_022_384_043_953_895_664e-24;
        for (h, rho, fraction) in [(f64::INFINITY, 0.3, 1.0), (0.0, 0.0, 0.5), (12.0, 1.0, 1.0), (0.0, -1.0, 1.0)] {
            let actual = bivariate_normal_interval_probability(h, 10.0, 12.0, rho).unwrap();
            let target = fraction * truth;
            assert!(
                (actual.value - target).abs() <= actual.rounding + f64::EPSILON * target,
                "h={h} rho={rho} {actual:?} truth={target:e}"
            );
            assert!(actual.rounding <= 1.0e-13 * actual.value, "h={h} rho={rho} {actual:?}");
        }
        let semi_infinite = bivariate_normal_interval_probability(0.0, 10.0, f64::INFINITY, 0.0).unwrap();
        // `Φ(−10)/2` at 40 digits (mpmath).
        let half_tail = 0.5 * 7.619_853_024_160_526_065_973_343e-24;
        assert!((semi_infinite.value - half_tail).abs() <= semi_infinite.rounding + f64::EPSILON * half_tail);
        // The singular orthant is the same mass, `P(−k ≤ X ≤ h)`, taken on the tail side of zero.
        let singular = bivariate_normal_cdf(12.0, -10.0, -1.0).unwrap();
        assert!((singular.value - truth).abs() <= singular.rounding, "{singular:?} truth={truth:e}");
        assert!(singular.rounding <= 1.0e-13 * singular.value, "{singular:?}");
        for &(h, lower, upper, rho) in &[(0.3, -1.0, 0.5, 0.6), (-0.4, 0.2, 2.5, -0.8), (1.1, -3.0, -0.5, 0.95)] {
            let actual = bivariate_normal_interval_probability(h, lower, upper, rho).unwrap();
            let (above, below) = (bivariate_normal_cdf(h, upper, rho).unwrap(), bivariate_normal_cdf(h, lower, rho).unwrap());
            let difference = above.value - below.value;
            let allowed = actual.rounding + above.rounding + below.rounding + UNIT_ROUNDOFF * difference.abs();
            assert!((actual.value - difference).abs() <= allowed, "h={h} [{lower}, {upper}] rho={rho}");
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
                    // Re-forming the vanishing factor by one division moves it by ≤ ε relative, and with it the split's
                    // `ĉ` and the density's factor. That moves the value by at most `(φ(u*)|u*| + φ₂·c)ε < ε`, on top of
                    // both calls' own bounds.
                    assert!(
                        (plain.value - carried.value).abs() <= plain.rounding + carried.rounding + f64::EPSILON,
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
        assert!(bivariate_normal_cdf_partials_with_complement(0.0, 0.0, 1.0, 0.0).is_err());
    }

    #[test]
    fn complement_keeps_the_digits_a_rounded_correlation_loses() {
        // The caller's correlation rounds to 1, while its complement is resolved.
        let (rho, complement) = (1.0_f64, 2.0e-17_f64);
        // Sheppard at the origin, through the half angle: Φ₂(0,0;ρ) = ½ − asin(√((1 − ρ)/2))/π, with 1 − ρ = c/2. The
        // reference's own rounding is the subtraction's, `u` of ½, since its quotient is below 10⁻⁹.
        let expected = 0.5 - (0.5 * complement.sqrt()).asin() / PI;
        let reference_rounding = 2.0 * UNIT_ROUNDOFF * expected;
        let carried = bivariate_normal_cdf_with_complement(0.0, 0.0, rho, complement).unwrap();
        assert!(
            (carried.value - expected).abs() <= carried.rounding + reference_rounding,
            "carried={carried:?} expected={expected:e}"
        );
        let plain = bivariate_normal_cdf(0.0, 0.0, rho).unwrap();
        assert!(
            (plain.value - expected).abs() > plain.rounding + reference_rounding,
            "the rounded correlation must lose these digits: {plain:?}"
        );

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
    fn bounds_cover_the_complementary_orthant_identity() {
        // Φ₂(h, k; ρ) + Φ₂(h, −k; −ρ) = Φ(h). A constraint active in one orthant is inactive in the other, so the two take
        // different branches of the tree and their errors do not cancel by construction.
        let bounds = [-8.0_f64, -3.0, -0.7, 0.0, 0.4, 2.0, 6.0];
        let rhos = [
            -(1.0 - 1.0e-15),
            -0.999,
            -0.6,
            -0.2,
            0.2,
            0.500_000_1,
            0.9,
            0.999,
            1.0 - 1.0e-9,
            1.0 - 1.0e-15,
        ];
        // `sign` is the second orthant's correlation over `−ρ`: 1 is the identity, −1 its positive control.
        let violation = |h: f64, k: f64, rho: f64, complement: Option<f64>, sign: f64| {
            let (first, second) = match complement {
                None => (
                    bivariate_normal_cdf(h, k, rho).unwrap(),
                    bivariate_normal_cdf(h, -k, -sign * rho).unwrap(),
                ),
                Some(c) => (
                    bivariate_normal_cdf_with_complement(h, k, rho, c).unwrap(),
                    bivariate_normal_cdf_with_complement(h, -k, -sign * rho, c).unwrap(),
                ),
            };
            let total = DoubleDouble::two_sum(first.value, second.value);
            let marginal = normal_cdf_and_pdf(h).0;
            let gap = total.sub(DoubleDouble::from(marginal)).hi.abs();
            let allowance = first.rounding
                + second.rounding
                + NORMAL_CDF_RELATIVE_ERROR * marginal
                + NORMAL_CDF_UNDERFLOW_FLOOR;
            (gap, allowance)
        };
        let mut control_fails = false;
        for &h in &bounds {
            for &k in &bounds {
                for &rho in &rhos {
                    let complement = Correlation::from_rho(rho).complement();
                    for route in [None, Some(complement)] {
                        let (gap, allowance) = violation(h, k, rho, route, 1.0);
                        assert!(
                            gap <= allowance,
                            "h={h} k={k} rho={rho} route={route:?} gap={gap:e} allowance={allowance:e}"
                        );
                        let (gap, allowance) = violation(h, k, rho, route, -1.0);
                        control_fails |= gap > allowance;
                    }
                }
            }
        }
        assert!(control_fails, "the identity with the second correlation's sign dropped must fail somewhere");
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
            bivariate_normal_cdf_with_complement(x, y, 1.0, complement).unwrap().value,
            normal_cdf_and_pdf(y).0
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
