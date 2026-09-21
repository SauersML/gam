//! Exact supported domain for logarithmic penalty strengths.
//!
//! Every smoothing precision has the form `lambda = exp(rho)`.  The value,
//! gradient, and Hessian with respect to `rho` agree only while that
//! exponentiation is evaluated exactly: clamping `rho` or flooring/ceilinging
//! `lambda` creates a constant tail with a fictitious nonzero derivative.
//! The computation forms both `lambda = exp(rho)` and `1/lambda = exp(-rho)`
//! (a penalty and the prior covariance it implies), so the domain is where both
//! are finite, normal binary64 numbers: `|rho| <= -ln(f64::MIN_POSITIVE)
//! = 1022 ln 2`. Its faces are the edge of what the computation represents, not
//! a constraint of any model, so every face they bound is declared
//! [`crate::domain_face::DomainFaceKind::Representability`] (#2627).
//! This module owns the single domain used by all penalty implementations.

/// `ln √ε`: the log of the relative resolution of a criterion gradient carried
/// through an inverse whose conditioning is the strength ratio itself. Per
/// direction the ρ-gradient is the effective degrees of freedom `γ/(γ+λ)`
/// through `(H+λS)⁻¹`, whose condition in that direction is `λ/γ` once the
/// penalty dominates; a quantity through an inverse of condition `κ` holds
/// relative error `εκ`, and value and error cross at `λ/γ = 1/√ε`. Every
/// derived ρ-domain edge (#2812) is this many e-folds from the spectrum.
pub fn log_gradient_resolution() -> f64 {
    0.5 * f64::EPSILON.ln()
}

/// The precision box `[ln √ε, ln(1/√ε)]` around unit strength: the domain of
/// a coordinate whose penalty geometry cannot be projected, and the envelope a
/// seed is placed in before the domain is derived.
pub fn precision_box() -> (f64, f64) {
    (log_gradient_resolution(), -log_gradient_resolution())
}

/// Smallest supported logarithmic strength (inclusive): `ln(f64::MIN_POSITIVE)`,
/// the log of the smallest normal binary64 number. See [`LOG_STRENGTH_MAX`].
pub const LOG_STRENGTH_MIN: f64 = -LOG_STRENGTH_MAX;

/// Largest supported logarithmic strength (inclusive):
/// `-ln(f64::MIN_POSITIVE) = (1 - f64::MIN_EXP) ln 2 = 1022 ln 2`. At either face
/// `exp(rho)` and `exp(-rho)` are finite and normal: the reciprocal of the
/// smallest normal number is finite, so this face binds before `ln(f64::MAX)`.
pub const LOG_STRENGTH_MAX: f64 = (1 - f64::MIN_EXP) as f64 * std::f64::consts::LN_2;

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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhysicalStrengthDomainError {
    pub value: f64,
}

impl std::fmt::Display for PhysicalStrengthDomainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "physical strength must be positive and finite with its logarithm in [{LOG_STRENGTH_MIN}, {LOG_STRENGTH_MAX}]; got {}",
            self.value
        )
    }
}

impl std::error::Error for PhysicalStrengthDomainError {}

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
    if log_strength.is_finite() && (LOG_STRENGTH_MIN..=LOG_STRENGTH_MAX).contains(&log_strength) {
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
pub fn checked_exp_log_strength(log_strength: f64) -> Result<f64, LogStrengthDomainError> {
    validate_log_strength(log_strength)?;
    Ok(log_strength.exp())
}

/// Recover a canonical logarithmic coordinate without flooring or ceilinging
/// a physical strength.
pub fn checked_log_strength(strength: f64) -> Result<f64, PhysicalStrengthDomainError> {
    if !(strength.is_finite() && strength > 0.0) {
        return Err(PhysicalStrengthDomainError { value: strength });
    }
    let log_strength = strength.ln();
    validate_log_strength(log_strength)
        .map_err(|_| PhysicalStrengthDomainError { value: strength })?;
    Ok(log_strength)
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
    fn domain_faces_are_where_a_strength_and_its_reciprocal_stay_normal() {
        // The face is -ln(f64::MIN_POSITIVE), up to the one rounding of its
        // closed form 1022 ln 2.
        let from_log = -f64::MIN_POSITIVE.ln();
        assert!(
            (LOG_STRENGTH_MAX - from_log).abs() <= f64::EPSILON * from_log,
            "LOG_STRENGTH_MAX {LOG_STRENGTH_MAX:e} is not -ln(f64::MIN_POSITIVE) {from_log:e}"
        );
        assert_eq!(LOG_STRENGTH_MIN, -LOG_STRENGTH_MAX);
        for face in [LOG_STRENGTH_MIN, LOG_STRENGTH_MAX] {
            let strength = face.exp();
            let reciprocal = (-face).exp();
            assert!(
                strength.is_normal() && reciprocal.is_normal(),
                "at rho = {face:e}: exp(rho) = {strength:e} and exp(-rho) = {reciprocal:e} must \
                 both be finite normal binary64 numbers"
            );
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

    #[test]
    fn physical_strength_conversion_refuses_floor_and_ceiling_cases() {
        for value in [0.0, -1.0, f64::INFINITY, f64::NAN] {
            assert!(checked_log_strength(value).is_err());
        }
        for endpoint in [LOG_STRENGTH_MIN, LOG_STRENGTH_MAX] {
            assert!(checked_log_strength(endpoint.exp()).is_ok());
        }
    }
}
