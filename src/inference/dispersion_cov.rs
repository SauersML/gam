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
//! `Dispersion` lives in `solver::estimate` and is re-exported here as the
//! single source of truth. The helper methods on the local
//! `DispersionExt` trait give terse `phi() / inv_phi() / sqrt_phi()`
//! call-sites for the sampling code; we do not duplicate the enum because
//! `solver::estimate::Dispersion` already implements `phi()` and
//! `is_estimated()`.

use ndarray::Array2;

pub use crate::solver::estimate::Dispersion;

/// Posterior coefficient covariance `Vb = phi * H^{-1}` — the matrix users
/// see as `Cov(beta_hat)`. This newtype documents that `phi` has already
/// been multiplied in.
#[derive(Clone, Debug)]
pub struct PhiScaledCovariance(pub Array2<f64>);

impl PhiScaledCovariance {
    /// Wrap an array that is known to already be on the `phi * H^{-1}`
    /// scale.
    #[inline]
    pub fn wrap(cov: Array2<f64>) -> Self {
        Self(cov)
    }

    #[inline]
    pub fn as_array(&self) -> &Array2<f64> {
        &self.0
    }

    #[inline]
    pub fn into_array(self) -> Array2<f64> {
        self.0
    }
}

/// Raw penalised Hessian `H = X' W_H X + S(lambda)` with NO dispersion
/// scaling. Equivalent to `phi * Vb^{-1}` only when `phi == 1`. Use this
/// for whitening / precision-matrix paths, and pair it with a
/// [`Dispersion`] at the boundary if the consumer cares about `phi`.
#[derive(Clone, Debug)]
pub struct UnscaledPrecision(pub Array2<f64>);

impl UnscaledPrecision {
    #[inline]
    pub fn wrap(hessian: Array2<f64>) -> Self {
        Self(hessian)
    }

    #[inline]
    pub fn as_array(&self) -> &Array2<f64> {
        &self.0
    }

    #[inline]
    pub fn into_array(self) -> Array2<f64> {
        self.0
    }
}

/// Extension methods on [`Dispersion`] used by the sampling code, kept here
/// so we do not need to touch the canonical definition in
/// `solver::estimate`. The conversions are all `phi`-aware: `inv_phi()`
/// and `sqrt_phi()` are floored away from zero so that downstream
/// arithmetic never produces `NaN` / `Inf` on a pathological zero
/// dispersion.
pub trait DispersionExt {
    fn inv_phi(self) -> f64;
    fn sqrt_phi(self) -> f64;
}

impl DispersionExt for Dispersion {
    #[inline]
    fn inv_phi(self) -> f64 {
        1.0 / self.phi().max(1e-300)
    }

    #[inline]
    fn sqrt_phi(self) -> f64 {
        self.phi().max(0.0).sqrt()
    }
}
