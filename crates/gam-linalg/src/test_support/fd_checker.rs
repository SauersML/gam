//! Central-difference finite-difference checking harness for tests.
//!
//! Test modules across the workspace repeatedly hand-roll the same central-
//! difference gradient check: clone the parameter vector, bump one coordinate by
//! `±eps`, evaluate a scalar objective, form `(f₊ − f₋) / (2·eps)`, and compare
//! against an analytic gradient component. This module captures the mechanical
//! shapes — a coordinate-wise scalar-objective gradient, a full central-difference
//! Hessian, and the matrix-level agreement assertions — behind named helpers so
//! each call site
//! routes through one audited implementation instead of an open-coded loop.
//!
//! These helpers own no model-layer types: they are `ndarray` in, `ndarray` out.
//! That is exactly why they live in `gam-linalg` (the leaf that owns the dense
//! array seam) rather than in the model-level `gam-test-support` crate. Any
//! crate needing an FD cross-check gets it from a leaf dependency it already
//! has, instead of dragging the entire model layer into its test build.
//!
//! The `ndarray`-shaped helpers here are *only* for tests. The scalar
//! self-certifying oracle they build on is not: it lives in
//! [`crate::numeric_derivative`] because the production outer-gradient FD
//! audit differences the real criterion with the same code, and is re-exported
//! below so there is one implementation rather than two.

use ndarray::{Array1, Array2};

// The self-certifying oracle itself is production numerics, not a test
// helper: the outer-gradient FD audit inside the optimizer differences the
// real criterion with it. It lives in `gam_linalg::numeric_derivative` and is
// re-exported here so the test-level checkers keep one path to it.
pub use crate::numeric_derivative::{FdDerivative, FdVerdict, RiddersConfig, ridders_derivative};

/// [`ridders_derivative`] applied to coordinate `coord` of a scalar objective
/// at `x`.
pub fn ridders_partial_derivative<F>(
    mut objective: F,
    x: &Array1<f64>,
    coord: usize,
    config: RiddersConfig,
) -> FdDerivative
where
    F: FnMut(&Array1<f64>) -> f64,
{
    assert!(
        coord < x.len(),
        "ridders_partial_derivative: coordinate {coord} out of range for length {}",
        x.len()
    );
    ridders_derivative(
        |t| {
            let mut probe = x.clone();
            probe[coord] += t;
            objective(&probe)
        },
        config,
    )
}

/// Central finite-difference gradient of a scalar objective at `x`.
///
/// For each coordinate `i`, returns `(f(x + eps·eᵢ) − f(x − eps·eᵢ)) / (2·eps)`.
/// `f` is evaluated `2·len(x)` times. The input slice is never mutated (each
/// evaluation operates on a fresh clone), so `f` may borrow `x`'s surroundings
/// freely.
pub fn numerical_gradient_central_diff<F>(mut f: F, x: &Array1<f64>, eps: f64) -> Array1<f64>
where
    F: FnMut(&Array1<f64>) -> f64,
{
    let mut grad = Array1::zeros(x.len());
    for i in 0..x.len() {
        let mut xp = x.clone();
        let mut xm = x.clone();
        xp[i] += eps;
        xm[i] -= eps;
        grad[i] = (f(&xp) - f(&xm)) / (2.0 * eps);
    }
    grad
}

/// Central finite-difference Hessian of a scalar objective at `x`.
///
/// Returns the dense `n×n` matrix whose `(i, j)` entry is the symmetric
/// four-point central difference
/// `(f(x + ε·eᵢ + ε·eⱼ) − f(x + ε·eᵢ − ε·eⱼ) − f(x − ε·eᵢ + ε·eⱼ) + f(x − ε·eᵢ − ε·eⱼ)) / (4·ε²)`.
/// For `i = j` this stencil degenerates to the `2ε`-spaced second difference
/// `(f(x + 2ε·eᵢ) − 2·f(x) + f(x − 2ε·eᵢ)) / (4·ε²)`, so the same expression
/// covers the diagonal without a special case. `f` is evaluated `4·n²` times
/// and the input is never mutated.
///
/// Every `(i, j)` and `(j, i)` entry is computed independently; the stencil is
/// symmetric in `i ↔ j` up to floating-point rounding, so callers that require
/// exact symmetry should average the result with its transpose.
pub fn numerical_hessian_central_diff<F>(mut f: F, x: &Array1<f64>, eps: f64) -> Array2<f64>
where
    F: FnMut(&Array1<f64>) -> f64,
{
    let n = x.len();
    let mut hess = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let mut pp = x.clone();
            let mut pm = x.clone();
            let mut mp = x.clone();
            let mut mm = x.clone();
            pp[i] += eps;
            pp[j] += eps;
            pm[i] += eps;
            pm[j] -= eps;
            mp[i] -= eps;
            mp[j] += eps;
            mm[i] -= eps;
            mm[j] -= eps;
            hess[[i, j]] = (f(&pp) - f(&pm) - f(&mp) + f(&mm)) / (4.0 * eps * eps);
        }
    }
    hess
}

/// Asserts that a finite difference dense matrix closely matches an analytically
/// computed directional derivative matrix, both in tolerance and in
/// component-wise sign.
pub fn assert_matrix_derivativefd(fd: &Array2<f64>, analytic: &Array2<f64>, tol: f64, label: &str) {
    assert_eq!(analytic.dim(), fd.dim(), "{} dimensions must match", label);
    for i in 0..analytic.nrows() {
        for j in 0..analytic.ncols() {
            let analytic_ij = analytic[[i, j]];
            let fd_ij = fd[[i, j]];
            let diff = (analytic_ij - fd_ij).abs();

            if analytic_ij.abs() > tol && fd_ij.abs() > tol {
                assert_eq!(
                    analytic_ij.signum(),
                    fd_ij.signum(),
                    "{} sign mismatch at ({}, {}): analytic={}, fd={}",
                    label,
                    i,
                    j,
                    analytic_ij,
                    fd_ij
                );
            }
            assert!(
                diff <= tol,
                "{} value mismatch at ({}, {}): analytic={}, fd={}, abs_diff={}, tol={}",
                label,
                i,
                j,
                analytic_ij,
                fd_ij,
                diff,
                tol
            );
        }
    }
}

/// Asserts that a finite difference dense matrix matches an analytically
/// computed directional derivative matrix to a *relative* tolerance
/// `rel_tol·(1 + |analytic|)`, plus component-wise sign agreement.
///
/// Use this (rather than the absolute-tolerance [`assert_matrix_derivativefd`])
/// when the comparison's dominant components are O(0.1–1) and the finite
/// difference is contaminated by a small, non-smooth solver channel — e.g. an
/// adaptive PIRLS stabilization ridge whose magnitude shifts discontinuously
/// across the ± FD re-solves. There the exact analytic IFT derivative (which
/// correctly excludes that solver-only ridge) and the FD disagree by a fixed
/// *fraction* of the component magnitude, not a fixed absolute amount, so an
/// absolute bound tuned for the small components is spuriously tight on the
/// large ones. The two underlying derivative channels are validated separately
/// against their own FDs, so this asserts the composite to the achievable
/// relative precision rather than weakening the per-channel checks (gam#855).
pub fn assert_matrix_derivativefd_rel(
    fd: &Array2<f64>,
    analytic: &Array2<f64>,
    rel_tol: f64,
    label: &str,
) {
    assert_eq!(analytic.dim(), fd.dim(), "{} dimensions must match", label);
    for i in 0..analytic.nrows() {
        for j in 0..analytic.ncols() {
            let analytic_ij = analytic[[i, j]];
            let fd_ij = fd[[i, j]];
            let tol = rel_tol * (1.0 + analytic_ij.abs());
            if analytic_ij.abs() > tol && fd_ij.abs() > tol {
                assert_eq!(
                    analytic_ij.signum(),
                    fd_ij.signum(),
                    "{} sign mismatch at ({}, {}): analytic={}, fd={}",
                    label,
                    i,
                    j,
                    analytic_ij,
                    fd_ij
                );
            }
            let diff = (analytic_ij - fd_ij).abs();
            assert!(
                diff <= tol,
                "{} value mismatch at ({}, {}): analytic={}, fd={}, abs_diff={}, rel_tol={}, tol={}",
                label,
                i,
                j,
                analytic_ij,
                fd_ij,
                diff,
                rel_tol,
                tol
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    /// `f(x) = ½·xᵀA x + bᵀx` with symmetric `A`, whose exact gradient is
    /// `A x + b`. Exercises the gradient and Hessian helpers against the closed form.
    #[test]
    fn quadratic_gradient_and_hessian_match_closed_form() {
        let a = array![[3.0, 0.5, -0.2], [0.5, 2.0, 0.4], [-0.2, 0.4, 1.5]];
        let b = array![0.3, -1.1, 0.7];
        let x = array![0.9, -0.4, 1.3];

        let objective = |v: &Array1<f64>| 0.5 * v.dot(&a.dot(v)) + b.dot(v);
        let analytic_grad = a.dot(&x) + &b;

        let eps = 1e-6;
        let fd = numerical_gradient_central_diff(objective, &x, eps);
        for i in 0..x.len() {
            assert_abs_diff_eq!(fd[i], analytic_grad[i], epsilon = 1e-6);
        }


        // Full central-difference Hessian recovers the constant curvature A.
        let hess_fd = numerical_hessian_central_diff(objective, &x, 1e-4);
        for i in 0..x.len() {
            for j in 0..x.len() {
                assert_abs_diff_eq!(hess_fd[[i, j]], a[[i, j]], epsilon = 1e-6);
            }
        }

        // The matrix-level assertions accept the agreeing pair …
        assert_matrix_derivativefd(&hess_fd, &a, 1e-5, "quadratic curvature");
        assert_matrix_derivativefd_rel(&hess_fd, &a, 1e-5, "quadratic curvature (relative)");
    }

    /// The coordinate wrapper differentiates the requested axis and only that
    /// axis.
    #[test]
    fn ridders_partial_picks_the_requested_coordinate() {
        let x = array![0.7, -1.3, 2.1];
        let objective = |v: &Array1<f64>| v[0] * v[1] + (v[2] * v[2]).sin();
        for (coord, exact) in [
            (0usize, x[1]),
            (1, x[0]),
            (2, 2.0 * x[2] * f64::cos(x[2] * x[2])),
        ] {
            let measured =
                ridders_partial_derivative(objective, &x, coord, RiddersConfig::default());
            assert!(
                (measured.value - exact).abs() <= 1e-9,
                "coord {coord}: {:.10e} vs {exact:.10e}",
                measured.value
            );
        }
    }

    /// … and reject a matrix that disagrees beyond tolerance, naming the entry
    /// so the failure localizes instead of just reporting "matrices differ".
    #[test]
    #[should_panic(expected = "identity curvature value mismatch at (1, 1)")]
    fn matrix_assert_rejects_a_disagreeing_entry() {
        let analytic = array![[1.0, 0.0], [0.0, 1.0]];
        let fd = array![[1.0, 0.0], [0.0, 1.5]];
        assert_matrix_derivativefd(&fd, &analytic, 1e-6, "identity curvature");
    }
}
