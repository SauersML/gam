//! Overflow-safe scalar/array arithmetic and compensated-difference primitives
//! for the survival location-scale exact-Newton chain.
//!
//! Pure relocation from `survival_location_scale.rs` (issue #780
//! decomposition): the layered overflow guards (`safe_product`, `safe_sum2/3`,
//! `safe_product3`, `safe_hadamard_product`, `safe_linear_combo2_arrays`), the
//! numerically stable `softplus`, the weight-vector sanitizer, and the
//! compensated (two-difference) subtraction carrying an explicit roundoff
//! slack into the monotonicity gate. These are domain-agnostic numerical
//! primitives that depend on nothing in the rest of the module beyond the
//! family error type. No behavior change — bodies are byte-identical and the
//! entry points are re-imported by the parent so every call site is unchanged.

use super::SurvivalLocationScaleError;
use ndarray::Array1;

// Canonical stable softplus lives in `gam-linalg`; its sign-split identity
// already reproduces the explicit NaN/±inf guard arms this module used to
// carry (NaN → NaN, +inf → +inf, −inf → 0), so the alias is value-identical.
pub(super) use gam_linalg::utils::stable_softplus as softplus;

/// Layer 3 defense: clamp products that overflow to ±inf back to ±MAX.
/// With layer 1 (exp_neg_stable) active this should not trigger in normal
/// operation; it guards against edge cases where two independently large
/// (but sub-overflow) factors multiply to exceed MAX.
#[inline]
pub(super) fn safe_product(lhs: f64, rhs: f64) -> f64 {
    if lhs == 0.0 || rhs == 0.0 {
        0.0
    } else {
        let v = lhs * rhs;
        if v == f64::INFINITY {
            f64::MAX
        } else if v == f64::NEG_INFINITY {
            f64::MIN
        } else {
            v
        }
    }
}

/// Layer 3 defense: when a + b produces NaN from inf + (-inf), return 0.
///
/// In the survival chain, g = d_raw + qdot1 where both terms scale as
/// inv_sigma × (something).  When inv_sigma is very large, both terms
/// overflow independently even though their sum is finite.  Mapping
/// the cancellation to 0 is conservative: it says "the correction is
/// negligible", and the monotonicity guard in exact_row_kernel will floor
/// g upward if needed.
#[inline]
pub(super) fn safe_sum2(a: f64, b: f64) -> f64 {
    let sum = a + b;
    if sum.is_nan() {
        if a == 0.0 {
            return b;
        } else if b == 0.0 {
            return a;
        }
        if (a == f64::INFINITY && b == f64::NEG_INFINITY)
            || (a == f64::NEG_INFINITY && b == f64::INFINITY)
        {
            return 0.0;
        }
        sum
    } else {
        sum
    }
}

#[inline]
pub(super) fn safe_sum3(a: f64, b: f64, c: f64) -> f64 {
    safe_sum2(safe_sum2(a, b), c)
}

#[inline]
pub(super) fn safe_product3(a: f64, b: f64, c: f64) -> f64 {
    let mut factors = [a, b, c];
    factors.sort_by(|lhs, rhs| lhs.abs().total_cmp(&rhs.abs()));
    safe_product(safe_product(factors[0], factors[1]), factors[2])
}

pub(super) fn safe_hadamard_product(
    lhs: &Array1<f64>,
    rhs: &Array1<f64>,
) -> Result<Array1<f64>, SurvivalLocationScaleError> {
    if lhs.len() != rhs.len() {
        bail_dim_sls!(
            "safe_hadamard_product length mismatch: lhs has {}, rhs has {}",
            lhs.len(),
            rhs.len()
        );
    }
    let out = Array1::from_shape_fn(lhs.len(), |i| safe_product(lhs[i], rhs[i]));
    if out.iter().any(|value| value.is_nan()) {
        return Err(SurvivalLocationScaleError::NumericalFailure {
            reason: "safe_hadamard_product produced NaN values".to_string(),
        });
    }
    Ok(out)
}

pub(super) fn safe_linear_combo2_arrays(
    a: &Array1<f64>,
    b: &Array1<f64>,
    c: &Array1<f64>,
    d: &Array1<f64>,
) -> Result<Array1<f64>, SurvivalLocationScaleError> {
    if a.len() != b.len() || a.len() != c.len() || a.len() != d.len() {
        bail_dim_sls!(
            "safe_linear_combo2_arrays length mismatch: a={}, b={}, c={}, d={}",
            a.len(),
            b.len(),
            c.len(),
            d.len()
        );
    }
    let out = Array1::from_shape_fn(a.len(), |i| {
        safe_sum2(safe_product(a[i], b[i]), safe_product(c[i], d[i]))
    });
    if out.iter().any(|value| value.is_nan()) {
        return Err(SurvivalLocationScaleError::NumericalFailure {
            reason: "safe_linear_combo2_arrays produced NaN values".to_string(),
        });
    }
    Ok(out)
}

pub(super) fn sanitize_survival_weight_vector(weights: &Array1<f64>) -> Array1<f64> {
    Array1::from_shape_fn(weights.len(), |i| {
        let value = weights[i];
        if value.is_finite() {
            value
        } else if value == f64::INFINITY {
            f64::MAX
        } else if value == f64::NEG_INFINITY {
            f64::MIN
        } else {
            0.0
        }
    })
}

#[derive(Clone, Copy)]
pub(super) struct StableDifference {
    pub(super) value: f64,
    pub(super) roundoff_slack: f64,
    pub(super) operand_scale: f64,
}

#[inline]
fn two_diff(lhs: f64, rhs: f64) -> (f64, f64) {
    let high = lhs - rhs;
    let z = high - lhs;
    let low = (lhs - (high - z)) - (rhs + z);
    (high, low)
}

#[inline]
pub(super) fn compensated_difference(lhs: f64, rhs: f64) -> StableDifference {
    let operand_scale = lhs.abs().max(rhs.abs());
    if lhs.is_nan() || rhs.is_nan() {
        return StableDifference {
            value: f64::NAN,
            roundoff_slack: 0.0,
            operand_scale,
        };
    }
    if !lhs.is_finite() || !rhs.is_finite() {
        // Compensated subtraction is undefined for infinite operands.
        // Use a conservative slack: if the difference rounded to 0 (from
        // inf − inf via safe_sum2), the true value could be anywhere, so
        // make the slack large enough that the monotonicity guard will
        // clamp rather than hard-error.
        let diff = safe_sum2(lhs, -rhs);
        let slack = if diff == 0.0 && operand_scale > 0.0 {
            // inf − inf ≈ 0: the true difference is unknown; use a large
            // slack so the guard floor can absorb it.
            operand_scale
        } else {
            // One finite, one infinite, or both same-sign infinite:
            // the result is ±inf or a well-defined finite value.
            0.0
        };
        return StableDifference {
            value: diff,
            roundoff_slack: slack,
            operand_scale,
        };
    }
    let (high, low) = two_diff(lhs, rhs);
    if !high.is_finite() {
        return StableDifference {
            value: high,
            roundoff_slack: 0.0,
            operand_scale,
        };
    }
    let value = high + low;
    // |low| is the exact rounding error of the final lhs − rhs subtraction.
    // The 128ε term bounds accumulated upstream error: d_raw and qdot each
    // pass through ~45 chained safe_product / safe_sum operations, giving
    // ≤90ε × operand_scale total propagated error.  128 rounds up to the
    // next power of two for a conservative margin.
    let roundoff_slack = low.abs() + 128.0 * f64::EPSILON * operand_scale.max(value.abs());
    StableDifference {
        value,
        roundoff_slack,
        operand_scale,
    }
}
