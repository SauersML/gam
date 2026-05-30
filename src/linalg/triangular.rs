//! Dense triangular and Cholesky-factor solves.
//!
//! A single home for the forward/back substitution kernels that several solver
//! and GPU host paths previously hand-rolled (one per call site). Given a
//! lower-triangular Cholesky factor `L` (so the symmetric positive-definite
//! system matrix is `A = L Lᵀ`), [`cholesky_solve_vector`] solves `A x = b` by a
//! forward solve `L y = b` followed by a back solve `Lᵀ x = y`. The matrix
//! variants apply the same kernel column by column.
//!
//! Callers supply `L` as a dense lower-triangular factor; only the
//! on-and-below-diagonal entries are read, and the diagonal must be nonzero
//! (the factor of a positive-definite matrix). No pivoting or zero-diagonal
//! guarding is done here — that belongs to the Cholesky factorization that
//! produced `L`.
//!
//! The entry points accept anything that converts into an `ArrayView`
//! (`&Array2`/`&Array1`, an owned view, a column view, …) so call sites pass
//! their factor and right-hand side without ceremony.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Solve `L y = b` (lower-triangular) for `y`.
fn forward_kernel(l: ArrayView2<'_, f64>, b: ArrayView1<'_, f64>) -> Array1<f64> {
    let n = l.nrows();
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[[i, k]] * y[k];
        }
        y[i] = sum / l[[i, i]];
    }
    y
}

/// Solve `Lᵀ x = y` (the transpose of the lower factor is upper-triangular) for
/// `x`. Only the on-and-below-diagonal entries of `L` are read.
fn back_kernel(l: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>) -> Array1<f64> {
    let n = l.nrows();
    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = y[i];
        for k in (i + 1)..n {
            sum -= l[[k, i]] * x[k];
        }
        x[i] = sum / l[[i, i]];
    }
    x
}

/// Solve the upper-triangular system `Lᵀ x = y` by back substitution, where `L`
/// is supplied as the lower-triangular factor.
pub(crate) fn back_substitution_lower_transpose<'l, 'y>(
    l: impl Into<ArrayView2<'l, f64>>,
    y: impl Into<ArrayView1<'y, f64>>,
) -> Array1<f64> {
    back_kernel(l.into(), y.into())
}

/// Solve `A x = b` where `A = L Lᵀ` and `L` is the lower Cholesky factor.
pub(crate) fn cholesky_solve_vector<'l, 'b>(
    l: impl Into<ArrayView2<'l, f64>>,
    b: impl Into<ArrayView1<'b, f64>>,
) -> Array1<f64> {
    let l = l.into();
    let y = forward_kernel(l, b.into());
    back_kernel(l, y.view())
}

/// Solve `A X = B` (multiple right-hand sides) where `A = L Lᵀ`, column by
/// column.
pub(crate) fn cholesky_solve_matrix<'l, 'b>(
    l: impl Into<ArrayView2<'l, f64>>,
    b: impl Into<ArrayView2<'b, f64>>,
) -> Array2<f64> {
    let l = l.into();
    let b = b.into();
    let n = l.nrows();
    let m = b.ncols();
    let mut out = Array2::<f64>::zeros((n, m));
    for c in 0..m {
        let y = forward_kernel(l, b.column(c));
        let x = back_kernel(l, y.view());
        out.column_mut(c).assign(&x);
    }
    out
}

/// Solve the lower-triangular system `L Y = B` (multiple right-hand sides) for
/// `Y`, column by column — forward substitution only, no back solve.
pub(crate) fn forward_substitution_lower_matrix<'l, 'b>(
    l: impl Into<ArrayView2<'l, f64>>,
    b: impl Into<ArrayView2<'b, f64>>,
) -> Array2<f64> {
    let l = l.into();
    let b = b.into();
    let n = l.nrows();
    let m = b.ncols();
    let mut out = Array2::<f64>::zeros((n, m));
    for c in 0..m {
        let y = forward_kernel(l, b.column(c));
        out.column_mut(c).assign(&y);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2, array};

    /// Lower-triangular Cholesky factor of a fixed SPD matrix, computed by hand
    /// so the tests do not depend on any factorization routine under test.
    fn fixture_factor() -> Array2<f64> {
        // A = [[4, 2, 2], [2, 5, 7], [2, 7, 19]] is SPD with exact integer L:
        // L = [[2, 0, 0], [1, 2, 0], [1, 3, 3]]  (L Lᵀ == A).
        array![[2.0, 0.0, 0.0], [1.0, 2.0, 0.0], [1.0, 3.0, 3.0]]
    }

    fn reconstruct_spd(l: &Array2<f64>) -> Array2<f64> {
        l.dot(&l.t())
    }

    #[test]
    fn forward_then_back_solves_the_spd_system() {
        let l = fixture_factor();
        let a = reconstruct_spd(&l);
        let x_true = array![1.5, -2.0, 0.75];
        let b = a.dot(&x_true);

        let x = cholesky_solve_vector(&l, &b);
        for i in 0..3 {
            assert!((x[i] - x_true[i]).abs() < 1e-12, "x[{i}] = {}", x[i]);
        }
    }

    #[test]
    fn forward_substitution_solves_lower_system() {
        let l = fixture_factor();
        let y_true = array![3.0, -1.0, 2.0];
        let b = l.dot(&y_true);
        let y = forward_kernel(l.view(), b.view());
        for i in 0..3 {
            assert!((y[i] - y_true[i]).abs() < 1e-12, "y[{i}] = {}", y[i]);
        }
    }

    #[test]
    fn back_substitution_solves_upper_system() {
        let l = fixture_factor();
        let x_true = array![0.5, 4.0, -3.0];
        // Lᵀ x = rhs.
        let rhs = l.t().dot(&x_true);
        let x = back_substitution_lower_transpose(&l, &rhs);
        for i in 0..3 {
            assert!((x[i] - x_true[i]).abs() < 1e-12, "x[{i}] = {}", x[i]);
        }
    }

    #[test]
    fn full_solve_is_forward_then_back_composed() {
        // cholesky_solve_vector must equal back(forward(b)) exactly (bit-for-bit),
        // since it is defined as that composition.
        let l = fixture_factor();
        let b = array![7.0, -2.5, 11.0];
        let y = forward_kernel(l.view(), b.view());
        let expected = back_substitution_lower_transpose(&l, &y);
        let got = cholesky_solve_vector(&l, &b);
        assert_eq!(got, expected);
    }

    #[test]
    fn matrix_solve_matches_per_column_vector_solve() {
        let l = fixture_factor();
        let b = array![[1.0, 0.0, 2.0], [0.0, 1.0, -1.0], [3.0, -2.0, 0.5]];
        let x = cholesky_solve_matrix(&l, &b);
        for c in 0..b.ncols() {
            let col = cholesky_solve_vector(&l, b.column(c));
            for r in 0..3 {
                assert_eq!(x[[r, c]], col[r], "mismatch at ({r},{c})");
            }
        }
    }

    #[test]
    fn matrix_solve_recovers_inverse() {
        // Solving A X = I yields A⁻¹; A A⁻¹ must be the identity.
        let l = fixture_factor();
        let a = reconstruct_spd(&l);
        let inv = cholesky_solve_matrix(&l, &Array2::<f64>::eye(3));
        let prod = a.dot(&inv);
        for i in 0..3 {
            for j in 0..3 {
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!((prod[[i, j]] - expect).abs() < 1e-12, "prod[{i},{j}]");
            }
        }
    }

    #[test]
    fn forward_matrix_matches_per_column_forward_solve() {
        let l = fixture_factor();
        let b = array![[2.0, -1.0], [5.0, 0.0], [-3.0, 4.0]];
        let y = forward_substitution_lower_matrix(&l, &b);
        for c in 0..b.ncols() {
            let col = forward_kernel(l.view(), b.column(c));
            for r in 0..3 {
                assert_eq!(y[[r, c]], col[r], "mismatch at ({r},{c})");
            }
        }
        // And it is the genuine forward solve: L Y == B.
        let recon = l.dot(&y);
        for i in 0..3 {
            for c in 0..b.ncols() {
                assert!((recon[[i, c]] - b[[i, c]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn one_by_one_system() {
        let l = array![[2.0_f64]];
        let b = array![6.0_f64];
        let x = cholesky_solve_vector(&l, &b);
        // A = 4, so x = 6/4 = 1.5.
        assert!((x[0] - 1.5).abs() < 1e-15);
    }

    #[test]
    fn empty_system_returns_empty() {
        let l = Array2::<f64>::zeros((0, 0));
        let b = Array1::<f64>::zeros(0);
        let x = cholesky_solve_vector(&l, &b);
        assert_eq!(x.len(), 0);
    }
}
