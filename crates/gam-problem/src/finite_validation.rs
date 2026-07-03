//! Shared finite-value validation helpers for estimation/result contracts.

use crate::EstimationError;
use ndarray::Array1;

pub fn ensure_finite_scalar_estimation(name: &str, value: f64) -> Result<(), EstimationError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(EstimationError::InvalidInput(format!(
            "{name} must be finite, got {value}"
        )))
    }
}

pub fn validate_all_finite_estimation<I>(label: &str, values: I) -> Result<(), EstimationError>
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

#[inline]
pub fn bail_if_cached_beta_non_finite(beta: &Array1<f64>) -> Result<(), EstimationError> {
    if beta.iter().any(|v| !v.is_finite()) {
        return Err(EstimationError::InvalidInput(
            "cached inner beta contains non-finite entries".to_string(),
        ));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensure_finite_scalar_ok_for_finite() {
        assert!(ensure_finite_scalar("x", 0.0).is_ok());
        assert!(ensure_finite_scalar("x", -1.5).is_ok());
        assert!(ensure_finite_scalar("x", f64::MIN).is_ok());
    }

    #[test]
    fn ensure_finite_scalar_err_for_nan() {
        let e = ensure_finite_scalar("my_value", f64::NAN).unwrap_err();
        assert!(e.contains("my_value"), "error should mention the name: {e}");
    }

    #[test]
    fn ensure_finite_scalar_err_for_inf() {
        assert!(ensure_finite_scalar("v", f64::INFINITY).is_err());
        assert!(ensure_finite_scalar("v", f64::NEG_INFINITY).is_err());
    }

    #[test]
    fn validate_all_finite_ok_for_finite_slice() {
        assert!(validate_all_finite("vec", [1.0, 2.0, 3.0]).is_ok());
        assert!(validate_all_finite("empty", std::iter::empty()).is_ok());
    }

    #[test]
    fn validate_all_finite_err_reports_index() {
        let e = validate_all_finite("arr", [1.0, f64::NAN, 3.0]).unwrap_err();
        assert!(e.contains("arr[1]"), "error should mention arr[1]: {e}");
    }

    #[test]
    fn validate_all_finite_err_reports_inf() {
        let e = validate_all_finite("data", [0.0, f64::INFINITY]).unwrap_err();
        assert!(e.contains("data[1]"), "error should mention data[1]: {e}");
    }

    #[test]
    fn bail_if_cached_beta_ok_for_finite() {
        let beta = ndarray::array![1.0, -2.0, 3.0];
        assert!(bail_if_cached_beta_non_finite(&beta).is_ok());
    }

    #[test]
    fn bail_if_cached_beta_err_for_nan() {
        let beta = ndarray::array![1.0, f64::NAN];
        assert!(bail_if_cached_beta_non_finite(&beta).is_err());
    }
}
