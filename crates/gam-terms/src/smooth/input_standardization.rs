//! Per-axis input standardization and length-scale compensation helpers for
//! the spatial smooth arm.
//!
//! Pure numeric helpers relocated verbatim from `smooth.rs` (issue #780
//! decomposition): per-column variance scales, in-place standardization, the
//! geometric-mean scale, and the kernel length-scale compensation maps that
//! keep the Matérn/Duchon/thin-plate range in original coordinates after
//! standardization. No behavior change — bodies are byte-identical and the
//! parent re-imports each name so every call site is unchanged.

use ndarray::{Array2, ArrayView2};

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
/// axis (the returned vector holds `d` copies of `σ_iso`). Because `tr(Σ)` is
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
/// Returns `None` only when there is no axis or too few rows to estimate a
/// spread, or when the caller already supplies frozen scales (prediction path).
pub fn compute_spatial_input_scales(x: ArrayView2<'_, f64>) -> Option<Vec<f64>> {
    let d = x.ncols();
    if d == 0 {
        return None;
    }
    let n = x.nrows() as f64;
    if n < 2.0 {
        return None;
    }
    let mut var_sum = 0.0_f64;
    for j in 0..d {
        let col = x.column(j);
        let mean = col.sum() / n;
        let var = col.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
        var_sum += var;
    }
    let sigma_iso = (var_sum / d as f64).sqrt().max(1e-12);
    Some(vec![sigma_iso; d])
}

/// Apply per-column standardization to a data matrix using precomputed scales.
pub fn apply_input_standardization(x: &mut Array2<f64>, scales: &[f64]) {
    for j in 0..x.ncols() {
        let inv = 1.0 / scales[j];
        x.column_mut(j).mapv_inplace(|v| v * inv);
    }
}

/// Geometric mean of strictly positive scales: `(∏ s_a)^(1/d)`.
///
/// Computed via log-sum-divide to avoid overflow / underflow when d is large
/// or when individual scales are small. The Matérn / Duchon / thin-plate
/// auto-standardization paths use this to compensate the user's
/// `length_scale` so the kernel range remains expressed in *original* data
/// coordinates after per-axis division by σ_a:
///
///   ‖x_std − c_std‖ / L_eff with L_eff = L_user / σ_geom
///
/// matches `‖x − c‖ / L_user` exactly for uniform σ_a (= σ_geom) and reduces
/// to the natural anisotropic-Mahalanobis preconditioning when σ_a vary —
/// the convention σ_geom = (∏σ_a)^(1/d) preserves the kernel volume scale.
fn geometric_mean_scale(scales: &[f64]) -> f64 {
    if scales.is_empty() {
        return 1.0;
    }
    let log_mean: f64 = scales.iter().map(|&s| s.ln()).sum::<f64>() / scales.len() as f64;
    log_mean.exp()
}

pub fn compensate_length_scale_for_standardization(length_scale: f64, scales: &[f64]) -> f64 {
    let sigma_geom = geometric_mean_scale(scales);
    if sigma_geom > 0.0 && sigma_geom.is_finite() {
        length_scale / sigma_geom
    } else {
        length_scale
    }
}
