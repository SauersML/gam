//! Dispersion/scale contract used by covariance and reference-distribution code.

use serde::{Deserialize, Serialize};

/// Dispersion contract used by inferential covariance and reference distributions.
///
/// `Known(phi)` is used for fixed-scale exponential-family fits such as
/// Poisson and Binomial (`phi = 1`). `Estimated(phi)` is used when the
/// residual/likelihood scale is estimated from the data, e.g. Gaussian
/// (`phi = sigma^2`) and Gamma (`phi = 1 / shape`). Stored covariance
/// matrices are scaled by this `phi`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Dispersion {
    Known(f64),
    Estimated(f64),
}

impl Dispersion {
    #[inline]
    pub const fn phi(self) -> f64 {
        match self {
            Self::Known(phi) | Self::Estimated(phi) => phi,
        }
    }

    #[inline]
    pub const fn is_estimated(self) -> bool {
        matches!(self, Self::Estimated(_))
    }
}

impl Default for Dispersion {
    fn default() -> Self {
        Self::Known(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phi_known_returns_inner_value() {
        assert_eq!(Dispersion::Known(2.5).phi(), 2.5);
    }

    #[test]
    fn phi_estimated_returns_inner_value() {
        assert_eq!(Dispersion::Estimated(0.5).phi(), 0.5);
    }

    #[test]
    fn is_estimated_false_for_known() {
        assert!(!Dispersion::Known(1.0).is_estimated());
    }

    #[test]
    fn is_estimated_true_for_estimated() {
        assert!(Dispersion::Estimated(0.1).is_estimated());
    }

    #[test]
    fn default_is_known_one() {
        let d = Dispersion::default();
        assert_eq!(d.phi(), 1.0);
        assert!(!d.is_estimated());
    }

    #[test]
    fn known_and_estimated_same_phi_are_not_equal() {
        assert_ne!(Dispersion::Known(1.0), Dispersion::Estimated(1.0));
    }
}
