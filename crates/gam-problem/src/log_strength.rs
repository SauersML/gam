//! Exact supported domain for logarithmic penalty strengths.
//!
//! Every smoothing precision has the form `lambda = exp(rho)`.  The value,
//! gradient, and Hessian with respect to `rho` agree only while that
//! exponentiation is evaluated exactly: clamping `rho` or flooring/ceilinging
//! `lambda` creates a constant tail with a fictitious nonzero derivative.
//! The inclusive `[-700, 700]` interval deliberately stays inside binary64's
//! finite, normal exponential range: its lower face avoids subnormal-strength
//! arithmetic and its upper face leaves overflow guard margin.  It is a solver
//! policy domain, not a claim about the widest representable binary64 input.
//! This module owns the single domain used by all penalty implementations.

/// Smallest supported logarithmic strength (inclusive).
pub const LOG_STRENGTH_MIN: f64 = -700.0;

/// Largest supported logarithmic strength (inclusive).
pub const LOG_STRENGTH_MAX: f64 = 700.0;

/// A logarithmic strength is outside the exact supported solver contract.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogStrengthDomainError {
    pub value: f64,
}

impl std::fmt::Display for LogStrengthDomainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "log strength must be finite and in [{LOG_STRENGTH_MIN}, {LOG_STRENGTH_MAX}]; got {}",
            self.value
        )
    }
}

impl std::error::Error for LogStrengthDomainError {}

/// Coordinate-aware failure returned when validating a vector of logarithmic
/// strengths.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IndexedLogStrengthDomainError {
    pub coordinate: usize,
    pub value: f64,
}

impl std::fmt::Display for IndexedLogStrengthDomainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "log strength coordinate {} must be finite and in [{LOG_STRENGTH_MIN}, {LOG_STRENGTH_MAX}]; got {}",
            self.coordinate, self.value
        )
    }
}

impl std::error::Error for IndexedLogStrengthDomainError {}

impl From<IndexedLogStrengthDomainError> for crate::EstimationError {
    fn from(error: IndexedLogStrengthDomainError) -> Self {
        Self::LogStrengthDomainViolation {
            coordinate: error.coordinate,
            value: error.value,
            lower: LOG_STRENGTH_MIN,
            upper: LOG_STRENGTH_MAX,
        }
    }
}

/// Validate a logarithmic strength without changing it.
#[inline]
pub fn validate_log_strength(log_strength: f64) -> Result<(), LogStrengthDomainError> {
    if log_strength.is_finite()
        && (LOG_STRENGTH_MIN..=LOG_STRENGTH_MAX).contains(&log_strength)
    {
        Ok(())
    } else {
        Err(LogStrengthDomainError {
            value: log_strength,
        })
    }
}

/// Validate a complete vector, reporting the deterministic smallest invalid
/// coordinate before any caller-visible computation begins.
pub fn validate_log_strengths(
    values: impl IntoIterator<Item = f64>,
) -> Result<(), IndexedLogStrengthDomainError> {
    for (coordinate, value) in values.into_iter().enumerate() {
        validate_log_strength(value)
            .map_err(|_| IndexedLogStrengthDomainError { coordinate, value })?;
    }
    Ok(())
}

/// Convert a complete vector atomically on the exact supported domain.
pub fn checked_exp_log_strengths(
    values: impl IntoIterator<Item = f64>,
) -> Result<Vec<f64>, IndexedLogStrengthDomainError> {
    let mut strengths = Vec::new();
    for (coordinate, value) in values.into_iter().enumerate() {
        strengths.push(
            checked_exp_log_strength(value)
                .map_err(|_| IndexedLogStrengthDomainError { coordinate, value })?,
        );
    }
    Ok(strengths)
}

/// Exponentiate a logarithmic strength on the exact closed solver domain.
///
/// No input is clamped and no output is floored or capped.  Thus the returned
/// value is exactly the one whose first and second `rho` derivatives are both
/// `exp(rho)`.
#[inline]
pub fn checked_exp_log_strength(
    log_strength: f64,
) -> Result<f64, LogStrengthDomainError> {
    validate_log_strength(log_strength)?;
    Ok(log_strength.exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_closed_domain_accepts_both_endpoints_without_saturation() {
        for endpoint in [LOG_STRENGTH_MIN, LOG_STRENGTH_MAX] {
            let strength = checked_exp_log_strength(endpoint).expect("closed endpoint");
            assert_eq!(strength.to_bits(), endpoint.exp().to_bits());
            assert!(strength.is_finite() && strength > 0.0);
        }
    }

    #[test]
    fn exact_closed_domain_rejects_unsupported_and_nonfinite_values() {
        for value in [
            LOG_STRENGTH_MIN - 1.0,
            LOG_STRENGTH_MAX + 1.0,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NAN,
        ] {
            assert_eq!(
                checked_exp_log_strength(value).unwrap_err().value.to_bits(),
                value.to_bits()
            );
        }
    }

    #[test]
    fn vector_validation_reports_the_smallest_bad_coordinate_atomically() {
        let values = [0.0, LOG_STRENGTH_MAX + 1.0, f64::NAN];
        let error = checked_exp_log_strengths(values).unwrap_err();
        assert_eq!(error.coordinate, 1);
        assert_eq!(error.value, LOG_STRENGTH_MAX + 1.0);
    }
}
