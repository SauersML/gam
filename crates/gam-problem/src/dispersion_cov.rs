//! Newtype wrappers that disambiguate the two coefficient-space second-order
//! quantities used throughout inference.
//!
//! Background ("dispersion ownership"):
//!
//! The fitter stores two related matrices for a fitted model.
//!
//! * `FitInference::beta_covariance` is the posterior coefficient covariance
//!   `Vb = phi * H^{-1}`, with `H = X' W_H X + S(lambda)` and `phi` the
//!   dispersion parameter. This matrix is *already* multiplied by `phi`
//!   (see `solver/estimate.rs`'s `scaled_covariance` call).
//! * `FitInference::penalized_hessian` is the raw penalised Hessian `H`,
//!   with NO dispersion scaling.
//!
//! Several downstream consumers (HMC whitening, Laplace sampling, smooth
//! tests, etc.) have to know which of these representations they hold so
//! they apply `phi` exactly once. Passing both as bare `Array2<f64>` makes
//! that easy to get wrong: the same matrix shape can mean either thing,
//! and the compiler will not catch a missing — or duplicated —
//! `phi` factor.
//!
//! The lightweight newtypes below give us a way to label the convention at
//! API boundaries without changing the storage type of the existing
//! `FitInference` fields. Storage stays `Array2<f64>` to avoid cascading
//! changes into modules outside the dispersion-ownership refactor's scope
//! (pirls, families, GPU paths, main, etc.); callers that want to be
//! explicit can wrap with `PhiScaledCovariance::wrap` /
//! `UnscaledPrecision::wrap` at the boundary.
//!
//! `Dispersion` lives in `gam-problem` as the neutral scale contract.
//! `phi()` / `inv_phi()` / `sqrt_phi()` call-sites for sampling code.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::ops::{Deref, DerefMut};

pub use crate::Dispersion;

/// Compute standard errors from a covariance matrix (sqrt of diagonal).
pub fn se_from_covariance(cov: &Array2<f64>) -> Array1<f64> {
    Array1::from_iter(cov.diag().iter().map(|&v| v.max(0.0).sqrt()))
}

/// Posterior coefficient covariance `Vb = phi * H^{-1}` — the matrix users
/// see as `Cov(beta_hat)`. This newtype documents that `phi` has already
/// been multiplied in.
///
/// `#[serde(transparent)]` keeps the on-disk wire format identical to the
/// pre-newtype `Array2<f64>` storage so saved models round-trip cleanly.
/// `Deref<Target = Array2<f64>>` lets out-of-scope read sites continue
/// calling `Array2` methods (`.iter()`, `.nrows()`, `.dim()`, …) on the
/// wrapper without modification.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
#[serde(transparent)]
pub struct PhiScaledCovariance(pub Array2<f64>);

impl PhiScaledCovariance {
    /// Wrap an array that is known to already be on the `phi * H^{-1}`
    /// scale.
    #[inline]
    pub fn wrap(cov: Array2<f64>) -> Self {
        Self(cov)
    }

    /// Borrow the underlying `φ · H⁻¹` matrix without taking ownership.
    #[inline]
    pub fn as_array(&self) -> &Array2<f64> {
        &self.0
    }

    /// Consume the wrapper and return the raw `φ · H⁻¹` matrix.
    #[inline]
    pub fn into_array(self) -> Array2<f64> {
        self.0
    }
}

impl From<Array2<f64>> for PhiScaledCovariance {
    #[inline]
    fn from(cov: Array2<f64>) -> Self {
        Self(cov)
    }
}

impl From<PhiScaledCovariance> for Array2<f64> {
    #[inline]
    fn from(cov: PhiScaledCovariance) -> Self {
        cov.0
    }
}

impl Deref for PhiScaledCovariance {
    type Target = Array2<f64>;
    #[inline]
    fn deref(&self) -> &Array2<f64> {
        &self.0
    }
}

impl DerefMut for PhiScaledCovariance {
    #[inline]
    fn deref_mut(&mut self) -> &mut Array2<f64> {
        &mut self.0
    }
}

/// Raw penalised Hessian `H = X' W_H X + S(lambda)` with NO dispersion
/// scaling. Equivalent to `phi * Vb^{-1}` only when `phi == 1`. Use this
/// for whitening / precision-matrix paths, and pair it with a
/// [`Dispersion`] at the boundary if the consumer cares about `phi`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
#[serde(transparent)]
pub struct UnscaledPrecision(pub Array2<f64>);

impl UnscaledPrecision {
    /// Wrap an `Array2` that is already on the unscaled
    /// `H = XᵀW_H X + S(λ)` scale (no `φ` factor).  Caller is responsible
    /// for ensuring the matrix actually represents the penalised Hessian.
    #[inline]
    pub fn wrap(hessian: Array2<f64>) -> Self {
        Self(hessian)
    }

    /// Borrow the underlying penalised Hessian `H` without taking ownership.
    #[inline]
    pub fn as_array(&self) -> &Array2<f64> {
        &self.0
    }

    /// Consume the wrapper and return the raw `H` matrix.
    #[inline]
    pub fn into_array(self) -> Array2<f64> {
        self.0
    }
}

impl From<Array2<f64>> for UnscaledPrecision {
    #[inline]
    fn from(h: Array2<f64>) -> Self {
        Self(h)
    }
}

impl From<UnscaledPrecision> for Array2<f64> {
    #[inline]
    fn from(h: UnscaledPrecision) -> Self {
        h.0
    }
}

impl Deref for UnscaledPrecision {
    type Target = Array2<f64>;
    #[inline]
    fn deref(&self) -> &Array2<f64> {
        &self.0
    }
}

impl DerefMut for UnscaledPrecision {
    #[inline]
    fn deref_mut(&mut self) -> &mut Array2<f64> {
        &mut self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // ── se_from_covariance ────────────────────────────────────────────────────

    #[test]
    fn se_from_diagonal_matrix_is_sqrt_of_diagonal() {
        // cov = diag(4, 9) → se = [2, 3]
        let cov = array![[4.0_f64, 0.0], [0.0, 9.0]];
        let se = se_from_covariance(&cov);
        assert_eq!(se.len(), 2);
        assert!((se[0] - 2.0).abs() < 1e-14);
        assert!((se[1] - 3.0).abs() < 1e-14);
    }

    #[test]
    fn se_clamps_negative_diagonal_to_zero() {
        // A numerically-negative diagonal entry should clamp to 0 rather than NaN.
        let cov = array![[1.0_f64, 0.0], [0.0, -1e-15]];
        let se = se_from_covariance(&cov);
        assert!(se[1].is_finite());
        assert_eq!(se[1], 0.0);
    }

    // ── PhiScaledCovariance ───────────────────────────────────────────────────

    #[test]
    fn phi_scaled_covariance_wrap_and_as_array_round_trip() {
        let m = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let wrapped = PhiScaledCovariance::wrap(m.clone());
        assert_eq!(*wrapped.as_array(), m);
    }

    #[test]
    fn phi_scaled_covariance_deref_gives_array2() {
        let m = array![[5.0_f64]];
        let wrapped = PhiScaledCovariance::wrap(m.clone());
        assert_eq!(wrapped.nrows(), 1);
        assert_eq!(wrapped[[0, 0]], 5.0);
    }

    #[test]
    fn phi_scaled_covariance_into_array_consumes() {
        let m = array![[7.0_f64]];
        let wrapped = PhiScaledCovariance::wrap(m.clone());
        assert_eq!(wrapped.into_array(), m);
    }

    // ── UnscaledPrecision ─────────────────────────────────────────────────────

    #[test]
    fn unscaled_precision_wrap_and_as_array_round_trip() {
        let h = array![[2.0_f64, 0.0], [0.0, 3.0]];
        let wrapped = UnscaledPrecision::wrap(h.clone());
        assert_eq!(*wrapped.as_array(), h);
    }
}
