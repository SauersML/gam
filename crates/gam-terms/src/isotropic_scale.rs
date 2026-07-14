//! A scalar coordinate scale for isotropic Euclidean smooths.
//!
//! Isotropic kernels admit one uniform change of coordinate units.  Encoding
//! that scale as a vector made anisotropic states representable in the frozen
//! model even though the kernel and its scale contract require one value in
//! every direction.  `IsotropicScale` makes the geometric invariant explicit:
//! anisotropy belongs to the separate ARD parameters, never to this frame.

use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::fmt;

/// A positive, finite scalar whose reciprocal is also representable.
///
/// The field is private so construction, deserialization, and every frozen
/// replay path enforce the same invariant.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "f64", into = "f64")]
pub struct IsotropicScale(f64);

impl IsotropicScale {
    pub const ONE: Self = Self(1.0);

    pub fn new(value: f64) -> Result<Self, IsotropicScaleError> {
        if value.is_finite() && value > 0.0 && value.recip().is_finite() {
            Ok(Self(value))
        } else {
            Err(IsotropicScaleError { value })
        }
    }

    pub fn get(self) -> f64 {
        self.0
    }

    pub fn reciprocal(self) -> f64 {
        self.0.recip()
    }

    pub fn to_bits(self) -> u64 {
        self.0.to_bits()
    }

    /// Convert a coordinate-valued scalar from original to standardized units.
    pub fn to_standardized_units(self, value: f64) -> f64 {
        value * self.reciprocal()
    }

    /// Apply the uniform coordinate pullback in place.
    pub fn standardize(self, coordinates: &mut Array2<f64>) {
        let reciprocal = self.reciprocal();
        coordinates.mapv_inplace(|value| value * reciprocal);
    }
}

impl TryFrom<f64> for IsotropicScale {
    type Error = IsotropicScaleError;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<IsotropicScale> for f64 {
    fn from(value: IsotropicScale) -> Self {
        value.get()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IsotropicScaleError {
    value: f64,
}

impl fmt::Display for IsotropicScaleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "isotropic scale must be positive and finite with a finite reciprocal, got {}",
            self.value
        )
    }
}

impl std::error::Error for IsotropicScaleError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction_enforces_the_operational_invariant() {
        assert_eq!(IsotropicScale::new(1.0), Ok(IsotropicScale::ONE));
        assert!(IsotropicScale::new(f64::MIN_POSITIVE).is_ok());
        assert!(IsotropicScale::new(0.0).is_err());
        assert!(IsotropicScale::new(-1.0).is_err());
        assert!(IsotropicScale::new(f64::NAN).is_err());
        assert!(IsotropicScale::new(f64::INFINITY).is_err());
        assert!(IsotropicScale::new(f64::from_bits(1)).is_err());
    }

    #[test]
    fn wire_representation_is_a_checked_scalar() {
        let encoded = serde_json::to_string(&IsotropicScale::new(2.5).unwrap()).unwrap();
        assert_eq!(encoded, "2.5");
        assert_eq!(
            serde_json::from_str::<IsotropicScale>(&encoded).unwrap(),
            IsotropicScale::new(2.5).unwrap()
        );
        assert!(serde_json::from_str::<IsotropicScale>("[2.5,2.5]").is_err());
        assert!(serde_json::from_str::<IsotropicScale>("0.0").is_err());
        assert!(serde_json::from_str::<IsotropicScale>("-1.0").is_err());
    }
}
