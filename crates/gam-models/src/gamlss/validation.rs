//! Input/spec validation guards for the GAMLSS location-scale fitters.
//!
//! These are the pure precondition checks the entry points run before any
//! design assembly: length agreement between response/weights/offsets/designs,
//! finiteness and domain (`[0,1]` for binomial responses, non-negative
//! weights), the monotone-wiggle degree/knot-count invariant, and the
//! per-family term-spec validators that compose the primitives above. All
//! return `Result<(), String>` with byte-for-byte stable error strings; none
//! mutate state. Extracted verbatim from `gamlss.rs` (issue #780) — no
//! behavior change.

use super::{
    BinomialLocationScaleTermSpec, BinomialLocationScaleWiggleTermSpec, GamlssError,
    GaussianLocationScaleTermSpec, GaussianLocationScaleWiggleTermSpec,
};
use crate::parameter_block::ParameterBlockInput;
use gam_terms::smooth::TermCollectionSpec;
use ndarray::Array1;

pub(super) fn validate_len_match(name: &str, expected: usize, found: usize) -> Result<(), String> {
    if expected != found {
        return Err(GamlssError::DimensionMismatch {
            reason: format!("{name} length mismatch: expected {expected}, found {found}"),
        }
        .into());
    }
    Ok::<(), _>(())
}

pub(super) fn validateweights(weights: &Array1<f64>, context: &str) -> Result<(), String> {
    for (i, &w) in weights.iter().enumerate() {
        if !w.is_finite() || w < 0.0 {
            return Err(GamlssError::NonFinite {
                reason: format!(
                    "{context}: weights must be finite and non-negative; found weights[{i}]={w}"
                ),
            }
            .into());
        }
    }
    Ok(())
}

pub(super) fn validate_binomial_response(y: &Array1<f64>, context: &str) -> Result<(), String> {
    for (i, &yi) in y.iter().enumerate() {
        if !yi.is_finite() || !(0.0..=1.0).contains(&yi) {
            return Err(GamlssError::NonFinite {
                reason: format!(
                    "{context}: binomial response must be finite in [0,1]; found y[{i}]={yi}"
                ),
            }
            .into());
        }
    }
    Ok(())
}

#[inline]
pub(super) fn minimum_monotone_wiggle_knot_count(degree: usize) -> Result<usize, String> {
    degree
        .checked_add(1)
        .and_then(|order| order.checked_mul(2))
        .ok_or_else(|| "monotone wiggle knot-count overflow".to_string())
}

pub(super) fn validate_blockrows(
    name: &str,
    n: usize,
    block: &ParameterBlockInput,
) -> Result<(), String> {
    validate_len_match(
        &format!("block '{name}' offset vs response"),
        n,
        block.offset.len(),
    )?;
    validate_len_match(
        &format!("block '{name}' design rows vs response"),
        n,
        block.design.nrows(),
    )
}

pub(super) fn validate_term_datarows(
    context: &str,
    expected: usize,
    found: usize,
) -> Result<(), String> {
    if expected != found {
        return Err(GamlssError::DimensionMismatch { reason: format!(
            "{context}: data row count must match response length (expected {expected}, found {found})"
        ) }.into());
    }
    Ok::<(), _>(())
}

pub(super) fn validate_term_weights(
    data: ndarray::ArrayView2<'_, f64>,
    y_len: usize,
    weights: &Array1<f64>,
    context: &str,
) -> Result<(), String> {
    validate_term_datarows(context, y_len, data.nrows())?;
    validate_len_match("weights vs y", y_len, weights.len())?;
    validateweights(weights, context)
}

pub(super) fn validate_term_offset(
    y_len: usize,
    offset: &Array1<f64>,
    label: &str,
) -> Result<(), String> {
    validate_len_match(&format!("{label} vs y"), y_len, offset.len())?;
    for (row_idx, value) in offset.iter().enumerate() {
        if !value.is_finite() {
            return Err(GamlssError::NonFinite {
                reason: format!("{label} contains non-finite value at row {row_idx}: {value}"),
            }
            .into());
        }
    }
    Ok(())
}

/// Shared degree/knot-count guard for the location-scale *wiggle* term specs.
///
/// Both `validate_gaussian_location_scalewiggle_termspec` and
/// `validate_binomial_location_scalewiggle_termspec` enforce the identical
/// invariant — `wiggle_degree >= 2` and at least
/// `minimum_monotone_wiggle_knot_count(degree)` knots — with byte-for-byte
/// identical error strings. This is the single home for that check.
pub(super) fn validate_wiggle_degree_and_knots(
    context: &str,
    wiggle_degree: usize,
    wiggle_knots_len: usize,
) -> Result<(), String> {
    if wiggle_degree < 2 {
        return Err(GamlssError::ConstraintViolation {
            reason: format!("{context}: wiggle_degree must be >= 2, got {wiggle_degree}"),
        }
        .into());
    }
    let minimum_knots = minimum_monotone_wiggle_knot_count(wiggle_degree)?;
    if wiggle_knots_len < minimum_knots {
        return Err(GamlssError::DimensionMismatch {
            reason: format!(
                "{context}: wiggle_knots must have at least {minimum_knots} entries for degree {wiggle_degree}, got {wiggle_knots_len}"
            ),
        }
        .into());
    }
    Ok(())
}

pub(super) fn validate_gaussian_location_scale_termspec(
    data: ndarray::ArrayView2<'_, f64>,
    spec: &GaussianLocationScaleTermSpec,
    context: &str,
) -> Result<(), String> {
    validate_term_weights(data, spec.y.len(), &spec.weights, context)?;
    validate_term_offset(spec.y.len(), &spec.mean_offset, "mean_offset")?;
    validate_term_offset(spec.y.len(), &spec.log_sigma_offset, "log_sigma_offset")
}

pub(super) fn validate_gaussian_location_scalewiggle_termspec(
    data: ndarray::ArrayView2<'_, f64>,
    spec: &GaussianLocationScaleWiggleTermSpec,
    context: &str,
) -> Result<(), String> {
    let n = spec.y.len();
    validate_term_weights(data, n, &spec.weights, context)?;
    validate_term_offset(n, &spec.mean_offset, "mean_offset")?;
    validate_term_offset(n, &spec.log_sigma_offset, "log_sigma_offset")?;
    validate_blockrows("wiggle", n, &spec.wiggle_block)?;
    validate_wiggle_degree_and_knots(context, spec.wiggle_degree, spec.wiggle_knots.len())
}

pub(super) fn validate_binomial_location_scale_termspec(
    data: ndarray::ArrayView2<'_, f64>,
    spec: &BinomialLocationScaleTermSpec,
    context: &str,
) -> Result<(), String> {
    validate_term_weights(data, spec.y.len(), &spec.weights, context)?;
    validate_term_offset(spec.y.len(), &spec.threshold_offset, "threshold_offset")?;
    validate_term_offset(spec.y.len(), &spec.log_sigma_offset, "log_sigma_offset")?;
    validate_binomial_response(&spec.y, context)?;
    validate_binomial_log_sigma_identifiable(&spec.log_sigmaspec, context)?;
    Ok(())
}

pub(super) fn validate_binomial_location_scalewiggle_termspec(
    data: ndarray::ArrayView2<'_, f64>,
    spec: &BinomialLocationScaleWiggleTermSpec,
    context: &str,
) -> Result<(), String> {
    let n = spec.y.len();
    validate_term_weights(data, n, &spec.weights, context)?;
    validate_term_offset(n, &spec.threshold_offset, "threshold_offset")?;
    validate_term_offset(n, &spec.log_sigma_offset, "log_sigma_offset")?;
    validate_binomial_response(&spec.y, context)?;
    validate_binomial_log_sigma_identifiable(&spec.log_sigmaspec, context)?;
    validate_blockrows("wiggle", n, &spec.wiggle_block)?;
    gam_terms::inference::formula_dsl::require_binomial_inverse_link_supports_joint_wiggle(
        &spec.link_kind,
        context,
    )?;
    validate_wiggle_degree_and_knots(context, spec.wiggle_degree, spec.wiggle_knots.len())
}

pub(super) fn validate_binomial_log_sigma_identifiable(
    log_sigmaspec: &TermCollectionSpec,
    context: &str,
) -> Result<(), String> {
    if log_sigmaspec.linear_terms.is_empty()
        && log_sigmaspec.random_effect_terms.is_empty()
        && log_sigmaspec.smooth_terms.is_empty()
    {
        return Ok(());
    }

    Err(GamlssError::UnsupportedConfiguration {
        reason: format!(
            "{context}: Bernoulli binomial location-scale data identify only the composite q = -threshold / sigma; log_sigma must be intercept-only/fixed, not a free linear, random-effect, or smooth formula"
        ),
    }
    .into())
}
