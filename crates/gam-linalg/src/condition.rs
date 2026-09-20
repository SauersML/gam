//! Condition estimation from a factorization the caller already holds (#2901).
//!
//! [`estimate_inverse_one_norm`] is Hager's method with Higham's refinements
//! (N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
//! SIAM 2002, Algorithm 15.4; LAPACK `xLACN2`). A handful of solves with the
//! matrix and its transpose climb the vertices `e_j` of the unit 1-ball toward
//! the column of `A⁻¹` with the largest 1-norm.
//!
//! The value returned is `‖A⁻¹v‖₁` for vectors with `‖v‖₁ = 1`, so it is a LOWER
//! bound on `‖A⁻¹‖₁`. In practice it sits within a small factor of the true
//! norm, but nothing bounds how far below it can be. A band built on it is an
//! estimate of the arithmetic's error, not a certificate of it.

/// The 1-norm of `values`.
fn one_norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value.abs()).sum()
}

/// Estimate `‖A⁻¹‖₁` for a `dimension × dimension` matrix `A` from solves the
/// caller performs with its own factor.
///
/// `solve(v)` overwrites `v` with `A⁻¹v`, and `solve_transpose(v)` overwrites it
/// with `A⁻ᵀv`; for a symmetric `A` both are the same solve.
///
/// Each step takes `y = A⁻¹x`, the sign vector `ξ = sign(y)` and `z = A⁻ᵀξ`. When
/// `‖z‖∞ ≤ zᵀx`, `x` is a local maximum of `‖A⁻¹x‖₁` over the unit 1-ball and the
/// climb stops. Otherwise it moves to the vertex `e_j` with `j = argmax_j |z_j|`.
/// It also stops when a move does not increase the norm or repeats the sign
/// vector. Every move that continues strictly increases the estimate over the
/// finite set of vertices, so the climb terminates without an iteration cap.
///
/// A final alternating vector `b_i = (−1)^i·(1 + i/(n − 1))`, with `‖b‖₁ = 3n/2`,
/// contributes `‖A⁻¹b‖₁/‖b‖₁`. It catches the matrices on which the vertex climb
/// stops early.
///
/// A non-finite solve poisons the estimate rather than being discarded, so the
/// caller sees it and refuses.
pub fn estimate_inverse_one_norm<E>(
    dimension: usize,
    mut solve: impl FnMut(&mut [f64]) -> Result<(), E>,
    mut solve_transpose: impl FnMut(&mut [f64]) -> Result<(), E>,
) -> Result<f64, E> {
    if dimension == 0 {
        return Ok(0.0);
    }
    let n = dimension as f64;
    let mut x = vec![1.0 / n; dimension];
    let mut estimate = 0.0_f64;
    let mut previous_signs: Option<Vec<f64>> = None;
    loop {
        let mut y = x.clone();
        solve(&mut y)?;
        let norm = one_norm(&y);
        let signs: Vec<f64> = y
            .iter()
            .map(|&value| if value >= 0.0 { 1.0 } else { -1.0 })
            .collect();
        if let Some(previous) = previous_signs.as_ref() {
            if !(norm > estimate) || previous == &signs {
                estimate = estimate.max(norm);
                if !norm.is_finite() {
                    estimate = norm;
                }
                break;
            }
        }
        estimate = norm;
        let mut z = signs.clone();
        solve_transpose(&mut z)?;
        let mut largest = 0usize;
        for (index, value) in z.iter().enumerate() {
            if value.abs() > z[largest].abs() {
                largest = index;
            }
        }
        let z_dot_x: f64 = z.iter().zip(x.iter()).map(|(left, right)| left * right).sum();
        if !(z[largest].abs() > z_dot_x) {
            break;
        }
        x = vec![0.0; dimension];
        x[largest] = 1.0;
        previous_signs = Some(signs);
    }
    if dimension > 1 && estimate.is_finite() {
        let mut alternating: Vec<f64> = (0..dimension)
            .map(|index| {
                let magnitude = 1.0 + index as f64 / (n - 1.0);
                if index % 2 == 0 { magnitude } else { -magnitude }
            })
            .collect();
        solve(&mut alternating)?;
        let alternative = 2.0 * one_norm(&alternating) / (3.0 * n);
        estimate = if alternative.is_finite() {
            estimate.max(alternative)
        } else {
            alternative
        };
    }
    Ok(estimate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

    /// Solve through an explicit inverse: the test matrices are small enough to
    /// invert exactly by hand.
    fn inverse_solve(inverse: &Array2<f64>) -> impl FnMut(&mut [f64]) -> Result<(), String> + '_ {
        move |values: &mut [f64]| {
            let input = ndarray::Array1::from(values.to_vec());
            let output = inverse.dot(&input);
            values.copy_from_slice(output.as_slice().expect("contiguous"));
            Ok(())
        }
    }

    fn exact_one_norm(matrix: &Array2<f64>) -> f64 {
        (0..matrix.ncols())
            .map(|column| matrix.column(column).iter().map(|value| value.abs()).sum::<f64>())
            .fold(0.0_f64, f64::max)
    }

    /// On a diagonal matrix the largest column of `A⁻¹` is a vertex, and the climb
    /// reaches it exactly.
    #[test]
    fn a_diagonal_inverse_norm_is_found_exactly() {
        let inverse = Array2::from_diag(&array![1.0, 100.0, 0.01]);
        let estimate =
            estimate_inverse_one_norm(3, inverse_solve(&inverse), inverse_solve(&inverse)).unwrap();
        assert_eq!(estimate, 100.0);
    }

    /// The estimate never exceeds the true norm (it is `‖A⁻¹v‖₁` over unit
    /// vectors), and on this symmetric matrix it reaches the norm. `A = [[2, 1,
    /// 0], [1, 2, 1], [0, 1, 2]]` has `A⁻¹ = [[3, −2, 1], [−2, 4, −2], [1, −2, 3]]/4`,
    /// whose 1-norm is 2.
    #[test]
    fn the_estimate_is_a_lower_bound_that_reaches_a_symmetric_norm() {
        let inverse = array![[3.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 3.0]] / 4.0;
        let exact = exact_one_norm(&inverse);
        let estimate =
            estimate_inverse_one_norm(3, inverse_solve(&inverse), inverse_solve(&inverse)).unwrap();
        assert!(estimate <= exact * (1.0 + 3.0 * f64::EPSILON), "{estimate} above {exact}");
        assert!((estimate - exact).abs() <= 3.0 * f64::EPSILON * exact, "{estimate} vs {exact}");
    }

    /// A one-dimensional matrix has `‖A⁻¹‖₁ = 1/|a|`, and a failed solve is passed
    /// through to the caller.
    #[test]
    fn a_scalar_is_exact_and_a_failed_solve_propagates() {
        let inverse = array![[-0.25]];
        let estimate =
            estimate_inverse_one_norm(1, inverse_solve(&inverse), inverse_solve(&inverse)).unwrap();
        assert_eq!(estimate, 0.25);
        let refusal = estimate_inverse_one_norm(
            2,
            |_: &mut [f64]| Err("the factor refused".to_string()),
            |_: &mut [f64]| Err("the factor refused".to_string()),
        );
        assert_eq!(refusal.unwrap_err(), "the factor refused");
    }
}
