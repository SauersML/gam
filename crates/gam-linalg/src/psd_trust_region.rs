//! Exact Euclidean trust-region solve for a symmetric positive-semidefinite model.
//!
//! For
//!
//! ```text
//! minimize  1/2 xᵀ G x - bᵀ x    subject to ||x||₂ <= radius,
//! ```
//!
//! with `G ⪰ 0`, the KKT solution is `(G + μI)x = b`, `μ >= 0`. A null-space
//! component of `b` is therefore not an error when the radius is finite: it makes
//! `μ > 0` and puts the solution on the boundary. This is deliberately distinct
//! from an unconstrained Moore–Penrose normal-equation solve, where such a
//! component proves the objective is unbounded.

use crate::LinalgError;
use crate::faer_ndarray::FaerEigh;
use faer::Side;
use ndarray::{Array1, ArrayView1, ArrayView2};

/// Overflow-safe Euclidean norm (the LAPACK `xLASSQ` recurrence).
fn scaled_norm(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut scale = 0.0_f64;
    let mut sumsq = 1.0_f64;
    for value in values {
        let magnitude = value.abs();
        if magnitude == 0.0 {
            continue;
        }
        if scale < magnitude {
            let ratio = scale / magnitude;
            sumsq = 1.0 + sumsq * ratio * ratio;
            scale = magnitude;
        } else {
            let ratio = magnitude / scale;
            sumsq += ratio * ratio;
        }
    }
    if scale == 0.0 {
        0.0
    } else {
        scale * sumsq.sqrt()
    }
}

/// Solve a convex PSD quadratic inside a finite Euclidean trust ball.
///
/// The eigensystem is scaled by `max(||G||₂, ||b||₂ / radius)` before the
/// secular solve, so its bracket is dimensionless and `O(1)` under arbitrary
/// common rescaling of `(G, b)`. Numerical-null RHS components participate in
/// the boundary solution instead of being discarded.
pub fn solve_psd_trust_region(
    gram: ArrayView2<'_, f64>,
    rhs: ArrayView1<'_, f64>,
    radius: f64,
) -> Result<Array1<f64>, LinalgError> {
    let dimension = gram.nrows();
    if dimension == 0 || gram.ncols() != dimension || rhs.len() != dimension {
        return Err(LinalgError::InvalidInput(format!(
            "PSD trust-region shape mismatch: gram={:?}, rhs={}",
            gram.dim(),
            rhs.len()
        )));
    }
    if !(radius.is_finite() && radius > 0.0) {
        return Err(LinalgError::InvalidInput(format!(
            "PSD trust-region radius must be finite and positive, got {radius}"
        )));
    }
    if gram.iter().any(|value| !value.is_finite())
        || rhs.iter().any(|value| !value.is_finite())
    {
        return Err(LinalgError::InvalidInput(
            "PSD trust-region inputs must be finite".to_string(),
        ));
    }

    let mut symmetric = gram.to_owned();
    symmetric += &gram.t();
    symmetric *= 0.5;
    let (eigenvalues, eigenvectors) = symmetric.eigh(Side::Lower).map_err(|error| {
        LinalgError::InvalidInput(format!(
            "PSD trust-region eigendecomposition failed: {error}"
        ))
    })?;
    let spectral_scale = eigenvalues
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    let rank_tolerance = dimension as f64 * f64::EPSILON * spectral_scale;
    let minimum_eigenvalue = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
    if minimum_eigenvalue < -rank_tolerance {
        return Err(LinalgError::InvalidInput(format!(
            "PSD trust-region matrix is not positive semidefinite: minimum eigenvalue \
             {minimum_eigenvalue:.17e}, backward-error tolerance {rank_tolerance:.17e}"
        )));
    }

    let projected = eigenvectors.t().dot(&rhs);
    let rhs_norm = scaled_norm(projected.iter().copied());
    if rhs_norm == 0.0 {
        return Ok(Array1::zeros(dimension));
    }

    // Try the canonical interior Moore–Penrose solution. A material RHS in a
    // numerical-null mode makes the unconstrained model unbounded, so it must go
    // through the finite-radius boundary solve below.
    let null_rhs_norm = scaled_norm(
        eigenvalues
            .iter()
            .zip(projected.iter())
            .filter_map(|(&eigenvalue, &coefficient)| {
                (eigenvalue <= rank_tolerance).then_some(coefficient)
            }),
    );
    let rhs_tolerance = dimension as f64 * f64::EPSILON * rhs_norm;
    if null_rhs_norm <= rhs_tolerance {
        let mut spectral_step = Array1::<f64>::zeros(dimension);
        for mode in 0..dimension {
            if eigenvalues[mode] > rank_tolerance {
                spectral_step[mode] = projected[mode] / eigenvalues[mode];
            }
        }
        let step_norm = scaled_norm(spectral_step.iter().copied());
        if step_norm.is_finite() && step_norm <= radius {
            return Ok(eigenvectors.dot(&spectral_step));
        }
    }

    // Dimensionless boundary equation. With
    //   s = max(||G||₂, ||b||₂/r), ell=λ/s, d=c/(r s), nu=μ/s,
    // the secular equation is ||d/(ell+nu)||₂ = 1 and `||d||₂ <= 1`.
    let shift_scale = spectral_scale.max(rhs_norm / radius);
    if !(shift_scale.is_finite() && shift_scale > 0.0) {
        return Err(LinalgError::InvalidInput(
            "PSD trust-region shift is not representable in f64".to_string(),
        ));
    }
    let scaled_eigenvalues = eigenvalues.mapv(|value| value.max(0.0) / shift_scale);
    let scaled_rhs = projected.mapv(|value| (value / shift_scale) / radius);
    let mut lower = 0.0_f64;
    let mut upper = scaled_norm(scaled_rhs.iter().copied());
    if !(upper.is_finite() && upper > 0.0) {
        return Err(LinalgError::InvalidInput(
            "PSD trust-region secular bracket is not representable in f64".to_string(),
        ));
    }

    let relative_tolerance = 64.0 * dimension as f64 * f64::EPSILON;
    let mut shift = 0.5 * upper;
    if shift == 0.0 {
        return Err(LinalgError::InvalidInput(
            "PSD trust-region secular shift underflowed".to_string(),
        ));
    }
    for _ in 0..100 {
        let norm = scaled_norm(
            scaled_rhs
                .iter()
                .zip(scaled_eigenvalues.iter())
                .map(|(&coefficient, &eigenvalue)| coefficient / (eigenvalue + shift)),
        );
        if !norm.is_finite() {
            lower = shift;
            shift = 0.5 * (lower + upper);
            continue;
        }
        if norm > 1.0 {
            lower = shift;
        } else {
            upper = shift;
        }
        if (norm - 1.0).abs() <= relative_tolerance
            || upper - lower <= relative_tolerance * upper
        {
            break;
        }

        // Safeguarded Newton on phi(nu)=1/||y(nu)||-1. Expressing the
        // derivative through normalized y avoids squaring a huge near-pole step.
        let inverse_norm = 1.0 / norm;
        let mut weighted_inverse_denominator = 0.0_f64;
        for (&coefficient, &eigenvalue) in scaled_rhs.iter().zip(scaled_eigenvalues.iter()) {
            let denominator = eigenvalue + shift;
            let normalized = (coefficient / denominator) * inverse_norm;
            weighted_inverse_denominator += normalized * normalized / denominator;
        }
        let phi = inverse_norm - 1.0;
        let phi_derivative = inverse_norm * weighted_inverse_denominator;
        let candidate = shift - phi / phi_derivative;
        shift = if candidate.is_finite() && candidate > lower && candidate < upper {
            candidate
        } else {
            0.5 * (lower + upper)
        };
    }

    // Use the feasible side of the final bracket. The spectral vector has norm
    // at most one, so multiplication by the finite radius cannot overflow.
    let spectral_step = Array1::from_iter(
        scaled_rhs
            .iter()
            .zip(scaled_eigenvalues.iter())
            .map(|(&coefficient, &eigenvalue)| {
                radius * coefficient / (eigenvalue + upper)
            }),
    );
    let step = eigenvectors.dot(&spectral_step);
    if step.iter().any(|value| !value.is_finite()) {
        return Err(LinalgError::InvalidInput(
            "PSD trust-region solution is not representable in f64".to_string(),
        ));
    }
    Ok(step)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

    fn norm(vector: &Array1<f64>) -> f64 {
        scaled_norm(vector.iter().copied())
    }

    #[test]
    fn null_space_rhs_takes_the_exact_boundary_descent_step() {
        let gram = Array2::<f64>::zeros((2, 2));
        let rhs = array![3.0, -4.0];
        let step = solve_psd_trust_region(gram.view(), rhs.view(), 0.25).unwrap();
        assert!((step[0] - 0.15).abs() < 1.0e-14);
        assert!((step[1] + 0.20).abs() < 1.0e-14);
        assert!((norm(&step) - 0.25).abs() < 1.0e-14);
        // (G + 20 I) step = rhs.
        for axis in 0..2 {
            assert!((20.0 * step[axis] - rhs[axis]).abs() < 1.0e-13);
        }
    }

    #[test]
    fn range_identified_interior_solution_is_the_moore_penrose_step() {
        let gram = array![[2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 4.0]];
        let rhs = array![2.0, 0.0, 8.0];
        let step = solve_psd_trust_region(gram.view(), rhs.view(), 3.0).unwrap();
        assert_eq!(step, array![1.0, 0.0, 2.0]);
    }

    #[test]
    fn boundary_solution_is_scale_equivariant_and_satisfies_common_shift_kkt() {
        let base_gram = array![[1.0, 0.0], [0.0, 4.0]];
        let base_rhs = array![2.0, 1.0];
        let reference =
            solve_psd_trust_region(base_gram.view(), base_rhs.view(), 0.5).unwrap();
        assert!((norm(&reference) - 0.5).abs() < 1.0e-13);
        let shift_0 = (base_rhs[0] - base_gram[[0, 0]] * reference[0]) / reference[0];
        let shift_1 = (base_rhs[1] - base_gram[[1, 1]] * reference[1]) / reference[1];
        assert!(shift_0 > 0.0);
        assert!((shift_0 - shift_1).abs() < 1.0e-12 * shift_0);

        for scale in [1.0e-200, 1.0, 1.0e200] {
            let gram = &base_gram * scale;
            let rhs = &base_rhs * scale;
            let step = solve_psd_trust_region(gram.view(), rhs.view(), 0.5).unwrap();
            for axis in 0..2 {
                assert!((step[axis] - reference[axis]).abs() < 1.0e-12);
            }
        }
    }

    #[test]
    fn genuinely_indefinite_model_is_rejected() {
        let gram = array![[-1.0, 0.0], [0.0, 2.0]];
        let rhs = array![1.0, 1.0];
        let error = solve_psd_trust_region(gram.view(), rhs.view(), 1.0).unwrap_err();
        assert!(error.to_string().contains("not positive semidefinite"));
    }
}
