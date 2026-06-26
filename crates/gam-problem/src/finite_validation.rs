//! Shared finite-value validation helpers for estimation/result contracts.

use crate::EstimationError;

pub fn ensure_finite_scalar_estimation(
    name: &str,
    value: f64,
) -> Result<(), EstimationError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(EstimationError::InvalidInput(format!(
            "{name} must be finite, got {value}"
        )))
    }
}

pub fn validate_all_finite_estimation<I>(
    label: &str,
    values: I,
) -> Result<(), EstimationError>
where
    I: IntoIterator<Item = f64>,
{
    for (idx, value) in values.into_iter().enumerate() {
        if !value.is_finite() {
            return Err(EstimationError::InvalidInput(format!(
                "{label}[{idx}] must be finite, got {value}"
            )));
        }
    }
    Ok(())
}

/// Public wrapper returning `String` errors for use outside the estimation module.
pub fn ensure_finite_scalar(name: &str, value: f64) -> Result<(), String> {
    ensure_finite_scalar_estimation(name, value).map_err(|err| err.to_string())
}

/// Public wrapper returning `String` errors for use outside the estimation module.
pub fn validate_all_finite<I: IntoIterator<Item = f64>>(
    label: &str,
    values: I,
) -> Result<(), String> {
    validate_all_finite_estimation(label, values).map_err(|err| err.to_string())
}
