//! Central-difference finite-difference checking harness for tests.
//!
//! Test modules across the crate repeatedly hand-roll the same central-difference
//! gradient check: clone the parameter vector, bump one coordinate by `±eps`,
//! evaluate a scalar objective, form `(f₊ − f₋) / (2·eps)`, and compare against an
//! analytic gradient component. This module captures the two mechanical shapes —
//! a coordinate-wise scalar-objective gradient and a directional derivative of a
//! vector-valued map — behind named helpers so each call site routes through one
//! audited implementation instead of an open-coded loop.
//!
//! These helpers are *only* for tests. They are not part of any production solver
//! path; the production outer-gradient FD audit lives in
//! [`crate::solver::rho_optimizer::fd_audit`] and is a different (criterion-level,
//! diagnostic-logging) facility.

use ndarray::{Array1, Array2};

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

/// Directional central finite-difference of a vector-valued map `f` at `x` along
/// `direction`: `(f(x + eps·d) − f(x − eps·d)) / (2·eps)`.
///
/// This is the shape used to validate a Hessian-vector product or a directional
/// score derivative against an analytic operator action: pass the gradient/score
/// map as `f` and the probe vector as `direction`.
pub fn directional_central_diff<F>(
    mut f: F,
    x: &Array1<f64>,
    direction: &Array1<f64>,
    eps: f64,
) -> Array1<f64>
where
    F: FnMut(&Array1<f64>) -> Array1<f64>,
{
    assert_eq!(
        x.len(),
        direction.len(),
        "directional_central_diff: x and direction must have equal length"
    );
    let xp = x + &(direction * eps);
    let xm = x - &(direction * eps);
    (f(&xp) - f(&xm)) / (2.0 * eps)
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

/// Verify an analytic gradient against the central finite-difference of the
/// objective, coordinate by coordinate.
///
/// Each component must agree to `tol·(1 + |fd|)` — a mixed absolute/relative
/// bound that stays meaningful both where the gradient is `O(1)` and where it is
/// near zero. Returns `Err` naming the first failing coordinate (with both
/// values and the realized gap) so the test panic message localizes the
/// disagreement; returns `Ok(())` when every coordinate agrees.
pub fn verify_gradient_vs_fd<F>(
    objective: F,
    analytic_grad: &Array1<f64>,
    x: &Array1<f64>,
    eps: f64,
    tol: f64,
) -> Result<(), String>
where
    F: FnMut(&Array1<f64>) -> f64,
{
    if analytic_grad.len() != x.len() {
        return Err(format!(
            "verify_gradient_vs_fd: analytic gradient length {} != x length {}",
            analytic_grad.len(),
            x.len()
        ));
    }
    let fd = numerical_gradient_central_diff(objective, x, eps);
    for i in 0..x.len() {
        let bound = tol * (1.0 + fd[i].abs());
        let gap = (analytic_grad[i] - fd[i]).abs();
        if gap > bound {
            return Err(format!(
                "verify_gradient_vs_fd: coordinate {i} disagrees: analytic={:.6e}, fd={:.6e}, gap={:.3e}, tol={:.3e} (bound {:.3e})",
                analytic_grad[i], fd[i], gap, tol, bound
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    /// `f(x) = ½·xᵀA x + bᵀx` with symmetric `A`, whose exact gradient is
    /// `A x + b`. Exercises all three helpers against the closed form.
    #[test]
    fn quadratic_gradient_and_directional_match_closed_form() {
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

        verify_gradient_vs_fd(objective, &analytic_grad, &x, eps, 1e-5)
            .expect("analytic gradient matches FD of the quadratic");

        // Directional FD of the gradient map recovers the Hessian action A·d.
        let direction = array![0.6, -0.8, 0.2];
        let grad_map = |v: &Array1<f64>| a.dot(v) + &b;
        let hvp_fd = directional_central_diff(grad_map, &x, &direction, eps);
        let hvp_exact = a.dot(&direction);
        for i in 0..direction.len() {
            assert_abs_diff_eq!(hvp_fd[i], hvp_exact[i], epsilon = 1e-6);
        }

        // Full central-difference Hessian recovers the constant curvature A.
        let hess_fd = numerical_hessian_central_diff(objective, &x, 1e-4);
        for i in 0..x.len() {
            for j in 0..x.len() {
                assert_abs_diff_eq!(hess_fd[[i, j]], a[[i, j]], epsilon = 1e-6);
            }
        }
    }

    /// A wrong analytic gradient must be rejected with the offending coordinate
    /// named.
    #[test]
    fn verify_rejects_wrong_gradient() {
        let x = array![1.0, 2.0];
        let objective = |v: &Array1<f64>| v[0] * v[0] + v[1] * v[1];
        let exact = array![2.0, 4.0];
        verify_gradient_vs_fd(objective, &exact, &x, 1e-6, 1e-5).expect("exact gradient passes");

        let wrong = array![2.0, 4.5];
        let err = verify_gradient_vs_fd(objective, &wrong, &x, 1e-6, 1e-5)
            .expect_err("perturbed gradient must be rejected");
        assert!(
            err.contains("coordinate 1"),
            "error should name coord 1: {err}"
        );
    }
}
