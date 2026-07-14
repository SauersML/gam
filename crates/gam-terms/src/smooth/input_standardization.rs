//! Isotropic input-scale estimation for Euclidean spatial smooths.

use crate::{IsotropicScale, basis::BasisError};
use ndarray::ArrayView2;

#[derive(Default)]
struct CompensatedSum {
    sum: f64,
    correction: f64,
}

impl CompensatedSum {
    fn add(&mut self, value: f64) {
        let next = self.sum + value;
        self.correction += if self.sum.abs() >= value.abs() {
            (self.sum - next) + value
        } else {
            (value - next) + self.sum
        };
        self.sum = next;
    }

    fn total(&self) -> f64 {
        self.sum + self.correction
    }
}

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

    // Center before scaling. Scaling absolute coordinates first loses the
    // low-order bits that contain the spread when a tight cloud is translated
    // far from the origin. Referencing the first row is translation-invariant
    // and avoids the cancellation in a conventional `E[x^2] - E[x]^2`
    // variance. A single scale shared by every axis also preserves the exact
    // isotropic coordinate contract while keeping every Welford update bounded.
    let reference = x.row(0).to_vec();
    let mut max_abs_difference = 0.0_f64;
    let mut max_abs_coordinate = 0.0_f64;
    let mut difference_overflowed = false;
    for ((_, axis), &value) in x.indexed_iter() {
        if !value.is_finite() {
            return Err(BasisError::InvalidInput(
                "cannot estimate an isotropic scale from non-finite coordinates".to_string(),
            ));
        }
        max_abs_coordinate = max_abs_coordinate.max(value.abs());
        let difference = value - reference[axis];
        if difference.is_finite() {
            max_abs_difference = max_abs_difference.max(difference.abs());
        } else {
            // Finite operands can produce a non-finite subtraction only when
            // their mathematical separation exceeds f64::MAX. In that case,
            // divide the operands before subtracting below.
            difference_overflowed = true;
        }
    }
    if max_abs_difference == 0.0 && !difference_overflowed {
        return Err(BasisError::InvalidInput(
            "cannot estimate an isotropic scale from a zero-spread cloud".to_string(),
        ));
    }

    let normalization = if difference_overflowed {
        max_abs_coordinate
    } else {
        max_abs_difference
    };
    let mut trace = CompensatedSum::default();
    for axis in 0..d {
        let mut count = 0.0_f64;
        let mut mean = 0.0_f64;
        let mut m2 = CompensatedSum::default();
        for &value in x.column(axis) {
            let normalized = if difference_overflowed {
                value / normalization - reference[axis] / normalization
            } else {
                (value - reference[axis]) / normalization
            };
            count += 1.0;
            let delta = normalized - mean;
            mean += delta / count;
            m2.add(delta * (normalized - mean));
        }
        trace.add(m2.total() / (count - 1.0));
    }
    let normalized_trace = trace.total();
    if !(normalized_trace.is_finite() && normalized_trace > 0.0) {
        return Err(BasisError::InvalidInput(
            "cannot estimate an isotropic scale from a zero-spread cloud".to_string(),
        ));
    }
    let sigma = (normalized_trace / d as f64).sqrt() * normalization;
    IsotropicScale::new(sigma).map_err(|error| {
        BasisError::InvalidInput(format!("cannot realize isotropic input scale: {error}"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn assert_relative_eq(left: f64, right: f64, tolerance: f64) {
        let scale = left.abs().max(right.abs());
        assert!(
            (left - right).abs() <= tolerance * scale,
            "left={left:.17e}, right={right:.17e}, tolerance={tolerance:.3e}"
        );
    }

    #[test]
    fn scale_estimate_is_stable_under_a_huge_translation() {
        let spread = 2.0_f64.powi(450);
        let offset = 2.0_f64.powi(500);
        let source = array![[-3.0, -1.0], [-1.0, 2.0], [2.0, -2.0], [4.0, 1.0]] * spread;
        let translated = source.mapv(|value| value + offset);

        let source_scale = estimate_isotropic_scale(source.view()).unwrap().get();
        let translated_scale = estimate_isotropic_scale(translated.view()).unwrap().get();

        assert_relative_eq(translated_scale, source_scale, 8.0 * f64::EPSILON);
    }

    #[test]
    fn scale_estimate_tracks_extreme_finite_dilations() {
        let source = array![[-3.0, -1.0], [-1.0, 2.0], [2.0, -2.0], [4.0, 1.0]];
        let source_scale = estimate_isotropic_scale(source.view()).unwrap().get();

        for exponent in [-400, 400] {
            let dilation = 2.0_f64.powi(exponent);
            let dilated = source.mapv(|value| value * dilation);
            let dilated_scale = estimate_isotropic_scale(dilated.view()).unwrap().get();
            assert_relative_eq(
                dilated_scale,
                source_scale * dilation,
                8.0 * f64::EPSILON,
            );
        }
    }

    #[test]
    fn zero_spread_is_a_typed_geometry_error() {
        let constant = array![[7.0, -3.0], [7.0, -3.0], [7.0, -3.0]];
        assert!(matches!(
            estimate_isotropic_scale(constant.view()),
            Err(BasisError::InvalidInput(message)) if message.contains("zero-spread")
        ));
    }
}
