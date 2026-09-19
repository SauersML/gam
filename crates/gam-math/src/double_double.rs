//! Double-double arithmetic in binary64: the one owner of it in gam-math (#2946).
//!
//! A [`DoubleDouble`] is a pair `high + low` formed by error-free transformations, and it carries no bound. A
//! [`BoundedDoubleDouble`] pairs one with `rounding`, a rigorous upper bound on `|high + low − exact|`, where `exact` is
//! the value the operations that formed it would give on exact inputs.
//!
//! Every bound here rests only on IEEE-754 binary64 semantics under round-to-nearest, with no overflow:
//! - [`DoubleDouble::two_sum`] (Knuth, *TAOCP* Vol. 2, §4.2.2; Møller 1965) gives `s + t = a + b` exactly.
//! - [`two_product`] forms `p = fl(ab)` and `e = fma(a, b, −p)`. The fused multiply-add is correctly rounded, so
//!   `|p + e − ab|` is at most `η/2` with `η = 2⁻¹⁰⁷⁴`, and zero while `|ab| ≥ 2⁻⁹⁷⁰`.
//! - Any other correctly rounded `+ − × ÷ √` errs by at most `u·|x| + η/2` of its exact result `x`, with `u = ε/2`.
//!   Against the computed result that is at most `u·|fl(x)|/(1 − u) + η/2`.
//!
//! No published double-word constant enters a bound.
//!
//! **Rounding of the bounds themselves.** Each bound is a nonnegative expression evaluated in `f64`. With `m` rounded
//! operations, and one more for each `u·|fl(x)|` charged in place of `u·|x|`, the computed value can fall short of the
//! exact one by at most the factor `(1 − u)^m`. Since `(1 − u)^−m ≤ 1 + γ_m`, with Wilkinson's `γ_m = m·u/(1 − m·u)`,
//! every operation returns its formula times `1 + γ_{m+3}`; the three extra operations cover forming that factor and
//! the product. Each operation's comment states its `m`.

use crate::roundoff::{UNIT_ROUNDOFF, accumulation_growth, inflated};

/// The smallest positive subnormal, `η = 2⁻¹⁰⁷⁴`: the most any correctly rounded operation or fused remainder errs by
/// beyond its relative band, twice over.
pub(crate) const SMALLEST_SUBNORMAL: f64 = f64::from_bits(1);

/// A double-double number `high + low` with `|low| ≤ ulp(high)/2`. It carries no error bound.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DoubleDouble {
    pub(crate) high: f64,
    pub(crate) low: f64,
}

impl DoubleDouble {
    pub(crate) fn from_f64(value: f64) -> Self {
        Self { high: value, low: 0.0 }
    }

    /// Knuth's `2Sum`: `high + low = a + b` exactly, whatever the magnitudes.
    pub(crate) fn two_sum(a: f64, b: f64) -> Self {
        let sum = a + b;
        let virtual_b = sum - a;
        Self {
            high: sum,
            low: (a - (sum - virtual_b)) + (b - virtual_b),
        }
    }
}

/// `p = fl(ab)` and the remainder `e = fma(a, b, −p)`: `|p + e − ab| ≤ η/2`.
pub(crate) fn two_product(a: f64, b: f64) -> DoubleDouble {
    let product = a * b;
    DoubleDouble {
        high: product,
        low: a.mul_add(b, -product),
    }
}

/// A double-double value with a rigorous bound on its distance from the exact result.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BoundedDoubleDouble {
    pub(crate) value: DoubleDouble,
    pub(crate) rounding: f64,
}

impl BoundedDoubleDouble {
    /// π as a double-double. `high` is the double nearest π and `low` the double nearest `π − high`, and
    /// `|π − high − low| < 3.0e-33` from π's decimal expansion (`π − high − low = −2.9947698…e-33`).
    pub(crate) const PI: Self = Self {
        value: DoubleDouble {
            high: std::f64::consts::PI,
            low: 1.224_646_799_147_353_2e-16,
        },
        rounding: 3.0e-33,
    };

    pub(crate) fn exact(value: f64) -> Self {
        Self {
            value: DoubleDouble::from_f64(value),
            rounding: 0.0,
        }
    }

    /// `2Sum` on both words, the two carries rounded once each, renormalized by `2Sum`. Under cancellation every
    /// high-word step stays exact, so the charged rounding scales with the carries, not with the operands.
    /// `m = 7`: five rounded operations and two charged magnitudes.
    pub(crate) fn add(self, other: Self) -> Self {
        let head = DoubleDouble::two_sum(self.value.high, other.value.high);
        let tail = DoubleDouble::two_sum(self.value.low, other.value.low);
        let carry = head.low + tail.high;
        let first = DoubleDouble::two_sum(head.high, carry);
        let remainder = tail.low + first.low;
        let result = DoubleDouble::two_sum(first.high, remainder);
        Self {
            value: result,
            rounding: inflated(
                self.rounding
                    + other.rounding
                    + UNIT_ROUNDOFF * (carry.abs() + remainder.abs())
                    + SMALLEST_SUBNORMAL,
                7,
            ),
        }
    }

    /// `2Prod` of the high word; the low word's product and the carry are rounded once each.
    /// `m = 8`: six rounded operations and two charged magnitudes.
    pub(crate) fn mul_f64(self, factor: f64) -> Self {
        let head = two_product(self.value.high, factor);
        let low_product = self.value.low * factor;
        let carry = head.low + low_product;
        let result = DoubleDouble::two_sum(head.high, carry);
        Self {
            value: result,
            rounding: inflated(
                self.rounding * factor.abs()
                    + UNIT_ROUNDOFF * (low_product.abs() + carry.abs())
                    + 2.0 * SMALLEST_SUBNORMAL,
                8,
            ),
        }
    }

    /// `2Prod` of the high words; the cross products and the carries are rounded once each, and `low·low` is dropped
    /// and charged whole. The inputs' bounds enter through `δa·b + a·δb + δa·δb`.
    /// `m = 21`: sixteen rounded operations and five charged magnitudes.
    pub(crate) fn mul(self, other: Self) -> Self {
        let head = two_product(self.value.high, other.value.high);
        let cross_first = self.value.high * other.value.low;
        let cross_second = self.value.low * other.value.high;
        let cross = cross_first + cross_second;
        let carry = head.low + cross;
        let result = DoubleDouble::two_sum(head.high, carry);
        let dropped = (self.value.low * other.value.low).abs();
        let magnitude_self = self.value.high.abs() + self.value.low.abs();
        let magnitude_other = other.value.high.abs() + other.value.low.abs();
        Self {
            value: result,
            rounding: inflated(
                self.rounding * magnitude_other
                    + other.rounding * magnitude_self
                    + self.rounding * other.rounding
                    + UNIT_ROUNDOFF
                        * (cross_first.abs() + cross_second.abs() + cross.abs() + carry.abs())
                    + dropped
                    + 3.0 * SMALLEST_SUBNORMAL,
                21,
            ),
        }
    }

    /// `q = fl(high/d)`; the remainder `high − qd` through `2Prod` and the low word; the correction `remainder/d`.
    /// Exactly `a/d = q + (high − qd + low)/d`, so `q`'s own rounding never enters.
    /// `m = 14`: ten rounded operations and four charged magnitudes.
    pub(crate) fn div_f64(self, divisor: f64) -> Self {
        let quotient = self.value.high / divisor;
        let product = two_product(quotient, divisor);
        let remainder_high = self.value.high - product.high;
        let remainder_low = self.value.low - product.low;
        let remainder = remainder_high + remainder_low;
        let correction = remainder / divisor;
        let result = DoubleDouble::two_sum(quotient, correction);
        Self {
            value: result,
            rounding: inflated(
                (self.rounding
                    + UNIT_ROUNDOFF * (remainder_high.abs() + remainder_low.abs() + remainder.abs())
                    + 2.0 * SMALLEST_SUBNORMAL)
                    / divisor.abs()
                    + UNIT_ROUNDOFF * correction.abs()
                    + SMALLEST_SUBNORMAL,
                14,
            ),
        }
    }

    /// The double nearest `high + low`, which renormalization made `high`, and its bound. `m = 1`.
    pub(crate) fn to_f64(self) -> (f64, f64) {
        (
            self.value.high,
            inflated(self.rounding + self.value.low.abs(), 1),
        )
    }
}

/// The unbounded double-double arithmetic of the Gauss-Legendre root refinement (`special::gauss_legendre_certified`),
/// whose certificate reads the refined values exactly. `add` is the sloppy form: it can lose all relative accuracy under
/// cancellation, which the refinement's certificate does not rely on.
impl DoubleDouble {
    /// Fast2Sum: `high + low` renormalized, for `|high| ≥ |low|`.
    pub(crate) fn renormalized(high: f64, low: f64) -> Self {
        let sum = high + low;
        Self {
            high: sum,
            low: low - (sum - high),
        }
    }

    pub(crate) fn add(self, other: Self) -> Self {
        let head = Self::two_sum(self.high, other.high);
        Self::renormalized(head.high, head.low + self.low + other.low)
    }

    pub(crate) fn negated(self) -> Self {
        Self {
            high: -self.high,
            low: -self.low,
        }
    }

    pub(crate) fn sub(self, other: Self) -> Self {
        self.add(other.negated())
    }

    pub(crate) fn mul(self, other: Self) -> Self {
        let product = self.high * other.high;
        let residual = self.high.mul_add(other.high, -product);
        Self::renormalized(product, residual + (self.high * other.low + self.low * other.high))
    }

    /// Three quotient digits, each divided off the remainder.
    pub(crate) fn div(self, other: Self) -> Self {
        let first = self.high / other.high;
        let remainder = self.sub(other.mul(Self::from_f64(first)));
        let second = remainder.high / other.high;
        let rest = remainder.sub(other.mul(Self::from_f64(second)));
        Self::renormalized(first, second).add(Self::from_f64(rest.high / other.high))
    }
}

/// Appends the exact `TwoProductFMA` parts of `factor·value` for a double-double `value`: two products, four parts.
pub(crate) fn push_product_parts(parts: &mut Vec<f64>, factor: f64, value: DoubleDouble) {
    for component in [value.high, value.low] {
        let product = factor * component;
        parts.push(product);
        parts.push(factor.mul_add(component, -product));
    }
}

/// `Sum2` over `parts`, with an upper bound on its absolute error. `products` counts the `TwoProductFMA` calls that
/// formed the parts, each within `5·2^−1074` under underflow.
///
/// Returns `(res, error_bound)` with `|res − Σ exact| ≤ error_bound`. By Proposition 4.5 the error is at most
/// `u·|s| + γ²_{m−1}·S + 5η·products`, and `|s| ≤ |res| + error`. With `S` bounded above by
/// `fl(Σ|p|)/(1 − γ_{m−1})`, the bound solves to `(u·|res| + γ²·S + 5η·products)/(1 − u)` (Ogita, Rump and Oishi,
/// *Accurate sum and dot product*, SIAM J. Sci. Comput. 26(6), 2005, Algorithm 4.4).
pub(crate) fn accurate_sum(parts: &mut [f64], products: usize) -> (f64, f64) {
    let count = parts.len();
    if count == 0 {
        return (0.0, 0.0);
    }
    let gamma = accumulation_growth(count - 1);
    let magnitude = parts.iter().fold(0.0, |total, part| total + part.abs());
    let magnitude_bound = inflated(magnitude, count) / (1.0 - gamma);
    for index in 1..count {
        let pair = DoubleDouble::two_sum(parts[index], parts[index - 1]);
        parts[index] = pair.high;
        parts[index - 1] = pair.low;
    }
    let tail = parts[..count - 1].iter().fold(0.0, |total, part| total + part);
    let res = tail + parts[count - 1];
    let unit = UNIT_ROUNDOFF;
    let error = (unit * res.abs() + gamma * gamma * magnitude_bound + 5.0 * SMALLEST_SUBNORMAL * products as f64)
        / (1.0 - unit);
    (res, inflated(error, 6))
}

/// The bounded operations only the proofs read: `probability::normal_table`'s certified Taylor models and the
/// test oracles, since the table route replaced the production series that used them.
#[cfg(test)]
mod test_support {
    use super::{BoundedDoubleDouble, DoubleDouble, SMALLEST_SUBNORMAL, two_product};
    use crate::roundoff::{UNIT_ROUNDOFF, inflated};

    impl BoundedDoubleDouble {
        /// `ab` through [`two_product`], within `η/2`.
        pub(crate) fn product(a: f64, b: f64) -> Self {
            Self {
                value: two_product(a, b),
                rounding: SMALLEST_SUBNORMAL,
            }
        }

        pub(crate) fn negated(self) -> Self {
            Self {
                value: DoubleDouble {
                    high: -self.value.high,
                    low: -self.value.low,
                },
                rounding: self.rounding,
            }
        }

        pub(crate) fn sub(self, other: Self) -> Self {
            self.add(other.negated())
        }

        /// `s = √high`, the remainder `r = a − s²` through `2Prod`, and the correction `r/(2s)`. Requires `high > 0`.
        /// Exactly `√a − s = r/(√a + s)`. With `√a + s ≥ s` and `|√a − s| ≤ |r|/s`, replacing `r/(√a + s)` by
        /// `r_high/(2s)` moves it by at most `(bound_r + |r_low|)/s + |r_high|·(|r_high| + |r_low| + bound_r)/(2s³)`.
        /// `m = 14`: thirteen rounded operations and one charged magnitude.
        pub(crate) fn sqrt(self) -> Self {
            let root = self.value.high.sqrt();
            let remainder = self.sub(Self::product(root, root));
            let correction = remainder.value.high / (2.0 * root);
            let result = DoubleDouble::two_sum(root, correction);
            let remainder_magnitude =
                remainder.value.high.abs() + remainder.value.low.abs() + remainder.rounding;
            Self {
                value: result,
                rounding: inflated(
                    (remainder.rounding + remainder.value.low.abs()) / root
                        + remainder.value.high.abs() * remainder_magnitude / (2.0 * root * root * root)
                        + UNIT_ROUNDOFF * correction.abs()
                        + SMALLEST_SUBNORMAL,
                    14,
                ),
            }
        }
    }
}
