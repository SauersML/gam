//! Validated dispersion/scale contract used by covariance and sampling code.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Why a dispersion value is part of a fitted model.
///
/// This tag is deliberately private: callers may inspect it through
/// [`Dispersion::is_estimated`], but cannot construct an unchecked value by
/// naming an enum variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum DispersionSource {
    Known,
    Estimated,
}

/// Serde representation for [`Dispersion`].
///
/// Deserialization goes through `TryFrom`, so persisted bytes cannot bypass the
/// same numerical invariant as an in-process constructor.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct DispersionWire {
    source: DispersionSource,
    phi: f64,
}

/// A validated response-level dispersion `phi`.
///
/// A known/fixed dispersion is finite and strictly positive. An estimated
/// dispersion is finite and non-negative: zero is a meaningful boundary result
/// for an exact fit, but any operation that divides by it must explicitly fail.
/// The distinction prevents a zero estimate from being silently promoted to a
/// tiny positive fixed scale.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "DispersionWire", into = "DispersionWire")]
pub struct Dispersion {
    source: DispersionSource,
    phi: f64,
}

#[derive(Clone, Copy, Debug, Error, PartialEq)]
pub enum DispersionError {
    #[error("dispersion phi must be finite, got {phi}")]
    NonFinite { phi: f64 },
    #[error("a known dispersion must be strictly positive, got {phi}")]
    NonPositiveKnown { phi: f64 },
    #[error("an estimated dispersion must be non-negative, got {phi}")]
    NegativeEstimate { phi: f64 },
    #[error("zero estimated dispersion has no finite reciprocal")]
    ZeroHasNoReciprocal,
    #[error("the reciprocal of dispersion phi={phi} is not representable as a finite f64")]
    ReciprocalNotRepresentable { phi: f64 },
    #[error("dispersion multiplier must be finite and strictly positive, got {multiplier}")]
    InvalidMultiplier { multiplier: f64 },
    #[error(
        "rescaling dispersion phi={phi} by multiplier={multiplier} is not representable as a finite f64"
    )]
    RescaleNotRepresentable { phi: f64, multiplier: f64 },
}

impl Dispersion {
    /// Exact unit fixed dispersion for fixed-scale likelihoods.
    pub const UNIT: Self = Self {
        source: DispersionSource::Known,
        phi: 1.0,
    };

    /// Exact boundary estimate produced by a zero-residual fit.
    pub const ZERO_ESTIMATE: Self = Self {
        source: DispersionSource::Estimated,
        phi: 0.0,
    };

    /// Construct a finite, strictly-positive fixed dispersion.
    #[inline]
    pub fn known(phi: f64) -> Result<Self, DispersionError> {
        if !phi.is_finite() {
            return Err(DispersionError::NonFinite { phi });
        }
        if phi <= 0.0 {
            return Err(DispersionError::NonPositiveKnown { phi });
        }
        Ok(Self {
            source: DispersionSource::Known,
            phi,
        })
    }

    /// Construct a finite, non-negative estimated dispersion.
    #[inline]
    pub fn estimated(phi: f64) -> Result<Self, DispersionError> {
        if !phi.is_finite() {
            return Err(DispersionError::NonFinite { phi });
        }
        if phi < 0.0 {
            return Err(DispersionError::NegativeEstimate { phi });
        }
        let phi = if phi == 0.0 { 0.0 } else { phi };
        Ok(Self {
            source: DispersionSource::Estimated,
            phi,
        })
    }

    /// Construct a dispersion from a finite positive precision/shape.
    ///
    /// This checks the division itself; a positive subnormal denominator whose
    /// reciprocal overflows is rejected instead of being floored.
    #[inline]
    pub fn from_reciprocal(value: f64, estimated: bool) -> Result<Self, DispersionError> {
        if !value.is_finite() {
            return Err(DispersionError::NonFinite { phi: value });
        }
        if value <= 0.0 {
            return Err(DispersionError::NonPositiveKnown { phi: value });
        }
        let phi = 1.0 / value;
        if !phi.is_finite() || phi == 0.0 {
            return Err(DispersionError::ReciprocalNotRepresentable { phi: value });
        }
        if estimated {
            Self::estimated(phi)
        } else {
            Self::known(phi)
        }
    }

    #[inline]
    pub const fn phi(self) -> f64 {
        self.phi
    }

    #[inline]
    pub const fn is_estimated(self) -> bool {
        matches!(self.source, DispersionSource::Estimated)
    }

    /// Whether this is the exact, validated boundary estimate `phi = 0`.
    #[inline]
    pub const fn is_zero_estimate(self) -> bool {
        self.is_estimated() && self.phi == 0.0
    }

    /// Return `1 / phi`, rejecting the exact boundary and overflow.
    #[inline]
    pub fn reciprocal(self) -> Result<f64, DispersionError> {
        if self.phi == 0.0 {
            return Err(DispersionError::ZeroHasNoReciprocal);
        }
        let reciprocal = 1.0 / self.phi;
        if !reciprocal.is_finite() {
            return Err(DispersionError::ReciprocalNotRepresentable { phi: self.phi });
        }
        Ok(reciprocal)
    }

    /// Return `sqrt(phi)`. Zero is represented exactly.
    #[inline]
    pub fn sqrt(self) -> f64 {
        self.phi.sqrt()
    }

    /// Rescale an estimated dispersion in place.
    ///
    /// Fixed dispersions are left unchanged and return `Ok(false)`. The product
    /// is checked before mutation so this operation is atomic on error.
    pub fn rescale_estimate(&mut self, multiplier: f64) -> Result<bool, DispersionError> {
        if !(multiplier.is_finite() && multiplier > 0.0) {
            return Err(DispersionError::InvalidMultiplier { multiplier });
        }
        if !self.is_estimated() {
            return Ok(false);
        }
        let phi = self.phi * multiplier;
        if !phi.is_finite() || (self.phi > 0.0 && phi == 0.0) {
            return Err(DispersionError::RescaleNotRepresentable {
                phi: self.phi,
                multiplier,
            });
        }
        self.phi = phi;
        Ok(true)
    }
}

impl From<Dispersion> for DispersionWire {
    fn from(value: Dispersion) -> Self {
        Self {
            source: value.source,
            phi: value.phi,
        }
    }
}

impl TryFrom<DispersionWire> for Dispersion {
    type Error = DispersionError;

    fn try_from(value: DispersionWire) -> Result<Self, Self::Error> {
        match value.source {
            DispersionSource::Known => Self::known(value.phi),
            DispersionSource::Estimated => Self::estimated(value.phi),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructors_enforce_distinct_domains() {
        assert_eq!(Dispersion::known(2.5).unwrap().phi(), 2.5);
        assert_eq!(
            Dispersion::estimated(0.0).unwrap(),
            Dispersion::ZERO_ESTIMATE
        );
        assert!(Dispersion::known(0.0).is_err());
        assert_eq!(
            Dispersion::estimated(-0.0).unwrap().phi().to_bits(),
            0.0_f64.to_bits()
        );
        assert!(Dispersion::estimated(-1.0).is_err());
        assert!(Dispersion::known(f64::NAN).is_err());
    }

    #[test]
    fn reciprocal_is_exactly_fallible_at_the_boundary() {
        assert_eq!(Dispersion::known(4.0).unwrap().reciprocal(), Ok(0.25));
        assert_eq!(
            Dispersion::ZERO_ESTIMATE.reciprocal(),
            Err(DispersionError::ZeroHasNoReciprocal)
        );
    }

    #[test]
    fn sqrt_preserves_zero_boundary() {
        assert_eq!(Dispersion::ZERO_ESTIMATE.sqrt(), 0.0);
        assert_eq!(Dispersion::estimated(9.0).unwrap().sqrt(), 3.0);
    }

    #[test]
    fn source_is_part_of_identity() {
        assert_ne!(
            Dispersion::known(1.0).unwrap(),
            Dispersion::estimated(1.0).unwrap()
        );
        assert!(!Dispersion::UNIT.is_estimated());
    }
}
