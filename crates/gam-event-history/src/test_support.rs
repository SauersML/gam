//! The crate's test-only rounding-bound tracker (#2961 bar ruling): two routes
//! to one quantity agree within the sum of their own forward rounding bounds,
//! above a magnitude floor.
#![cfg(test)]

use gam_math::nested_dual::JetField;

/// A value carried with the forward rounding bound of the route that
/// produced it. `scale` is the running error bound (Wilkinson; Higham,
/// *Accuracy and Stability of Numerical Algorithms*, ch. 3): for
/// `s = a op b`, `mu_s = |da s| mu_a + |db s| mu_b + |s|`, so the computed
/// value is within `eps * mu` of the exact one to first order. Cancellation
/// between inexact near-equal terms is charged through the propagated operand
/// bounds, while an exact subtraction of exact inputs adds only its own
/// result. A rounded constant charges its own magnitude.
///
/// A unary function charges one ulp of its result, `eps * max(|f|,
/// MIN_POSITIVE) >= ulp(f)`, plus the propagated `|f'| mu`.
/// - `crate::scalar::{sqrt, recip}`: derived. IEEE 754 squareRoot and
///   division are correctly rounded, within half an ulp.
/// - `crate::scalar::{exp, ln}`: CITED FROM THE RUNTIME LIBM (glibc 2.28
///   dbl-64 on MSI's Rocky 8.10), not derived from Rust's contract, which
///   leaves the precision of `f64::exp` and `f64::ln` unspecified. glibc
///   2.28's `sysdeps/ieee754/dbl-64/e_exp.c` (IBM uexp.c) states "Maximum ULP
///   error is 0.500008" (:281-282, :332), and `e_log.c` (IBM ulog.c) states
///   "max ULP error of b + c is ~0.502" (:113-114) and "max ULP error of
///   A + B is ~0.502" (:161). A test whose bar rests on this charge says so.
/// - Any other function passing its value through `compose_unary` (softplus,
///   expm1, lgamma, log Phi and their derivative stacks, erfc) has no such
///   statement: compose it from these functions and elementary operations,
///   or label a bar that relies on its charge as measured.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Bound {
    pub(crate) value: f64,
    pub(crate) scale: f64,
}

impl Bound {
    /// An exactly represented input.
    pub(crate) fn exact(value: f64) -> Self {
        Self { value, scale: 0.0 }
    }

    /// This route's own rounding bound.
    pub(crate) fn rounding(&self) -> f64 {
        f64::EPSILON * self.scale
    }

    /// The bar on the difference of two routes to one quantity: the sum of
    /// their own bounds.
    pub(crate) fn bar(&self, other: &Bound) -> f64 {
        self.rounding() + other.rounding()
    }
}

/// Scaling by zero or a power of two (including +-1) is exact whenever the
/// result is zero or normal.
fn exact_scaling(s: f64, value: f64) -> bool {
    s == 0.0 || (s.abs().log2().fract() == 0.0 && (value == 0.0 || value.is_normal()))
}

impl JetField for Bound {
    fn value(&self) -> f64 {
        self.value
    }
    fn add(&self, o: &Self) -> Self {
        let value = self.value + o.value;
        Self {
            value,
            scale: self.scale + o.scale + value.abs(),
        }
    }
    fn sub(&self, o: &Self) -> Self {
        let value = self.value - o.value;
        Self {
            value,
            scale: self.scale + o.scale + value.abs(),
        }
    }
    fn mul(&self, o: &Self) -> Self {
        let value = self.value * o.value;
        Self {
            value,
            scale: o.value.abs() * self.scale + self.value.abs() * o.scale + value.abs(),
        }
    }
    fn neg(&self) -> Self {
        Self {
            value: -self.value,
            ..*self
        }
    }
    fn scale(&self, s: f64) -> Self {
        let value = self.value * s;
        // Like a Sterbenz subtraction, an exact scaling adds only the
        // propagated bound.
        Self {
            value,
            scale: s.abs() * self.scale + if exact_scaling(s, value) { 0.0 } else { value.abs() },
        }
    }
    fn compose_unary(&self, d: [f64; 5]) -> Self {
        // One ulp of the result, including a subnormal one: see the type doc
        // for which functions this charge is derived or cited for.
        Self {
            value: d[0],
            scale: d[1].abs() * self.scale + d[0].abs().max(f64::MIN_POSITIVE),
        }
    }
    fn constant_like(&self, value: f64) -> Self {
        // Integers below 2^53 and powers of two are exact in source; every
        // other constant was rounded once.
        let exact = (value.fract() == 0.0 && value.abs() <= 2f64.powi(53)) || exact_scaling(value, value);
        Self {
            value,
            scale: if exact { 0.0 } else { value.abs() },
        }
    }
    fn with_value(&self, value: f64) -> Self {
        Self { value, ..*self }
    }
}

/// `|production - oracle| <= bar`, with the oracle above the bar.
pub(crate) fn agrees(production: &Bound, oracle: &Bound, name: &str) {
    let bar = production.bar(oracle);
    assert!(
        oracle.value.abs() > bar,
        "{name}: {} is below its bar {bar}",
        oracle.value
    );
    assert!(
        (production.value - oracle.value).abs() <= bar,
        "{name}: production {}, oracle {}, bar {bar}",
        production.value,
        oracle.value
    );
}
