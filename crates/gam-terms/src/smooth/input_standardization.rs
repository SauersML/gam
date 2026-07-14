//! Isotropic input-scale estimation for Euclidean spatial smooths.

use crate::{IsotropicScale, basis::BasisError};
use ndarray::ArrayView2;

/// Compute an **isotropic** input scale for spatial inputs — a single spread
/// value applied to every covariate axis.
///
/// Standardizing to unit spread makes the Matérn/Duchon/thin-plate kernel — and
/// the `ψ = log κ = −log ℓ` REML length-scale optimizer that refines it —
/// operate in scale-free coordinates, so the fit is invariant to a covariate
/// rescale `x → a·x + b`. This matters in **one dimension too** (issue #1215): a
/// 1-D `s(x, bs="tp")` whose kernel ran in raw covariate units seeded and bounded
/// its `ψ`-optimizer off the raw magnitude, landing in a scale-dependent basin (a
/// clean bimodal step across `|a| ⋛ 1`). Standardizing removes that magnitude
/// from the optimizer's view, so the selected `ψ̂` (hence the fitted curve) is
/// scale-invariant. The frozen scale is replayed at predict, so original-unit
/// queries map onto the same standardized geometry.
///
/// The scale is the ROOT-MEAN-SQUARE of the per-axis standard deviations,
/// `σ_iso = √( (1/d)·Σ_a Var(x_a) ) = √(tr(Σ)/d)`, applied identically to every
/// axis. Because `tr(Σ)` is
/// invariant under an orthogonal (rotation/reflection) map of the covariates and
/// dividing every axis by the SAME scalar is a uniform scaling that commutes with
/// that map, the standardized geometry is a rigid rotation of the un-rotated one.
/// That is what an ISOTROPIC kernel — whose `r^{2m−d}`(log r) form depends only on
/// the Euclidean distance and whose polynomial null space `{1, x, …}` maps onto
/// itself under any orthogonal map — requires to stay exactly rotation-invariant
/// (gam#1818). Per-AXIS standardization (dividing `x_a` by its own `σ_a`) is the
/// diagonal of `Σ`, which is NOT orthogonally invariant, so it anisotropically
/// shears the cloud and breaks the isotropic kernel's rotation invariance; it also
/// contradicts the isotropic kernel's own contract of equal treatment of every
/// direction. Genuine per-axis relevance is instead the job of the anisotropic
/// length-scale / ARD machinery (`scale_dims`), not this preconditioner. For
/// `d = 1` the RMS reduces to the single axis's `σ`, preserving the #1215 behavior
/// exactly.
///
/// A scale cannot be inferred from an empty, singleton, non-finite, or
/// zero-spread cloud. Those are invalid geometries, not an identity-scale
/// fallback.
pub fn estimate_isotropic_scale(
    x: ArrayView2<'_, f64>,
) -> Result<IsotropicScale, BasisError> {
    let d = x.ncols();
    if d == 0 {
        return Err(BasisError::InvalidInput(
            "cannot estimate an isotropic scale without a coordinate axis".to_string(),
        ));
    }
    if x.nrows() < 2 {
        return Err(BasisError::InvalidInput(
            "cannot estimate an isotropic scale from fewer than two rows".to_string(),
        ));
    }

    let mut magnitude = 0.0_f64;
    for &value in x {
        if !value.is_finite() {
            return Err(BasisError::InvalidInput(
                "cannot estimate an isotropic scale from non-finite coordinates".to_string(),
            ));
        }
        magnitude = magnitude.max(value.abs());
    }
    if magnitude == 0.0 {
        return Err(BasisError::InvalidInput(
            "cannot estimate an isotropic scale from a zero-spread cloud".to_string(),
        ));
    }

    // Normalize first so finite input cannot overflow the variance pass. The
    // normalization cancels algebraically when the final scale is restored.
    let n = x.nrows() as f64;
    let mut var_sum = 0.0_f64;
    for j in 0..d {
        let col = x.column(j);
        let mean = col.iter().map(|&value| value / magnitude).sum::<f64>() / n;
        let var = col
            .iter()
            .map(|&value| (value / magnitude - mean).powi(2))
            .sum::<f64>()
            / (n - 1.0);
        var_sum += var;
    }
    let sigma = (var_sum / d as f64).sqrt() * magnitude;
    IsotropicScale::new(sigma).map_err(|error| {
        BasisError::InvalidInput(format!("cannot realize isotropic input scale: {error}"))
    })
}
