//! Rust-owned numerical helpers for the torch-only manifold-SAE routing lane.
//!
//! These kernels are intentionally small, deterministic primitives that the
//! Python torch module calls as FFI glue.  Keeping them here preserves the repo
//! doctrine that Python marshals tensors while Rust owns numeric rules.

use ndarray::{Array2, ArrayView1};

/// Lift a one-dimensional Duchon center vector into a deterministic `(K, d)`
/// low-discrepancy cloud in `[0, 1]^d`.
///
/// The first coordinate is the caller-provided 1-D center.  Remaining axes use
/// the generalized golden-ratio additive recurrence (`R_d`) keyed only to
/// `(K, d)`, matching the historical torch implementation bit-for-bit for f64
/// inputs before dtype conversion at the Python boundary.
#[must_use]
pub fn duchon_centers_nd(centers_1d: ArrayView1<'_, f64>, d: usize) -> Array2<f64> {
    let k = centers_1d.len();
    let width = d.max(1);
    let mut out = Array2::<f64>::zeros((k, width));
    for (row, center) in centers_1d.iter().enumerate() {
        out[(row, 0)] = *center;
    }
    if d <= 1 || k == 0 {
        return out;
    }

    // Historical R_d generalized golden-ratio fixed point: x^d = x + 1.
    // Thirty-two fixed-point refinements are retained deliberately for
    // bit-level continuity with the previous torch lane.  The count is not a
    // tuning knob: for the smallest routed multi-axis case (d=2) each step
    // roughly doubles correct digits near the fixed point, so 32 iterations is
    // beyond f64 mantissa resolution; larger d is contractive more quickly.
    let mut phi = 2.0_f64;
    for _ in 0..32 {
        phi = (1.0 + phi).powf(1.0 / d as f64);
    }
    for axis in 1..d {
        let alpha = (1.0 / phi).powi(axis as i32).rem_euclid(1.0);
        for row in 0..k {
            let idx = (row + 1) as f64;
            out[(row, axis)] = (idx * alpha + 0.5).rem_euclid(1.0);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::duchon_centers_nd;
    use ndarray::array;

    #[test]
    fn duchon_centers_nd_preserves_first_axis_and_shape() {
        let centers = array![0.0, 0.5, 1.0];
        let lifted = duchon_centers_nd(centers.view(), 3);
        assert_eq!(lifted.shape(), &[3, 3]);
        assert_eq!(lifted.column(0), centers);
    }

    #[test]
    fn duchon_centers_nd_handles_empty_and_one_dimensional_inputs() {
        let empty = array![];
        assert_eq!(duchon_centers_nd(empty.view(), 4).shape(), &[0, 4]);
        let centers = array![0.25, 0.75];
        assert_eq!(
            duchon_centers_nd(centers.view(), 1),
            centers.into_shape_clone((2, 1)).unwrap()
        );
    }
}
