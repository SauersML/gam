use faer::{Mat, Side};
use gam::linalg::faer_ndarray::factorize_symmetricwith_fallback;
use gam::linalg::matrix::{ConditionedDesign, DenseDesignMatrix, DesignMatrix, LinearOperator};
use ndarray::array;

#[test]
fn factorize_symmetric_with_fallback_returns_working_solve_after_cholesky_failure() {
    let a = Mat::from_fn(2, 2, |i, j| if i == j { 0.0 } else { 1.0 });
    let rhs = Mat::from_fn(2, 1, |i, _| if i == 0 { 2.0 } else { -3.0 });
    let factor = factorize_symmetricwith_fallback(a.as_ref(), Side::Lower)
        .expect("fallback factorization should succeed on indefinite symmetric input");
    let x = factor.solve(rhs.as_ref());
    let ax = &a * &x;
    let r0 = ax[(0, 0)] - rhs[(0, 0)];
    let r1 = ax[(1, 0)] - rhs[(1, 0)];
    let rnorm = (r0.abs() + r1.abs()).max(1e-16);
    let bnorm = rhs[(0, 0)].abs() + rhs[(1, 0)].abs();
    let berr = rnorm / bnorm.max(1e-16);
    assert!(
        berr <= 1e-8,
        "factorize_symmetric_with_fallback should produce a solve with backward error at most 1e-8"
    );
}

/// #3519: once Cholesky refuses, the solve must be backward stable. On
/// `A = [[1e-17, 1], [1, 1]]` (cond₂ ≈ 2.6) with b = (1, 2), an unpivoted
/// `L D Lᵀ` accepts the 1e-17 pivot, grows `l₂₁ = 1e17` and returns x = (0, 1),
/// whose normwise backward error `‖b − A x‖∞ / (‖A‖∞‖x‖∞ + ‖b‖∞)` is 1/4.
/// Bunch–Kaufman bounds it by `γ_{3n}·ρ_n` with `γ_k = k ε / (1 − k ε)` and
/// element growth `ρ_n ≤ (1 + 1/α)^{n−1}`, `α = (1 + √17)/8`.
#[test]
fn factorize_symmetric_with_fallback_is_backward_stable_on_tiny_leading_pivot_3519() {
    let a = Mat::from_fn(2, 2, |i, j| if i == 0 && j == 0 { 1e-17 } else { 1.0 });
    let rhs = Mat::from_fn(2, 1, |i, _| if i == 0 { 1.0 } else { 2.0 });
    let factor = factorize_symmetricwith_fallback(a.as_ref(), Side::Lower)
        .expect("a well-conditioned indefinite 2×2 must factor");
    let x = factor.solve(rhs.as_ref());
    let ax = &a * &x;
    let residual = (ax[(0, 0)] - rhs[(0, 0)])
        .abs()
        .max((ax[(1, 0)] - rhs[(1, 0)]).abs());
    let a_inf = (a[(0, 0)].abs() + a[(0, 1)].abs()).max(a[(1, 0)].abs() + a[(1, 1)].abs());
    let x_inf = x[(0, 0)].abs().max(x[(1, 0)].abs());
    let b_inf = rhs[(0, 0)].abs().max(rhs[(1, 0)].abs());
    let backward_error = residual / (a_inf * x_inf + b_inf);
    let n = 2.0_f64;
    let gamma = 3.0 * n * f64::EPSILON / (1.0 - 3.0 * n * f64::EPSILON);
    let alpha = (1.0 + 17.0_f64.sqrt()) / 8.0;
    let growth = (1.0 + 1.0 / alpha).powf(n - 1.0);
    assert!(
        backward_error <= gamma * growth,
        "fallback solve x = ({:.17e}, {:.17e}) has backward error {backward_error:.3e} above the Bunch–Kaufman bound {:.3e}",
        x[(0, 0)],
        x[(1, 0)],
        gamma * growth,
    );
}

/// #3519: the Bunch–Kaufman log-determinant is the log of a positive
/// semidefinite determinant and NaN once any eigenvalue is negative, even when
/// two negative eigenvalues make `det A` positive.
#[test]
fn factorize_symmetric_with_fallback_logdet_is_nan_off_the_semidefinite_cone_3519() {
    use gam::linalg::matrix::FactorizedSystem;
    // det = 1e-17 − 1 < 0: one negative eigenvalue.
    let indefinite = Mat::from_fn(2, 2, |i, j| if i == 0 && j == 0 { 1e-17 } else { 1.0 });
    let factor = factorize_symmetricwith_fallback(indefinite.as_ref(), Side::Lower)
        .expect("indefinite 2×2 factors");
    assert!(factor.logdet().is_nan());
    // −I₂: det = 1 but both eigenvalues are −1.
    let negative = Mat::from_fn(2, 2, |i, j| if i == j { -1.0 } else { 0.0 });
    let factor = factorize_symmetricwith_fallback(negative.as_ref(), Side::Lower)
        .expect("negative definite 2×2 factors");
    assert!(factor.logdet().is_nan());
}

#[test]
fn conditioned_design_operator_matches_explicit_column_conditioning_for_matvec() {
    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, -2.0]];
    let design = DesignMatrix::Dense(DenseDesignMatrix::from(x.clone()));
    let conditioned = ConditionedDesign::new(design, vec![(1, 1.0, 2.0)]);
    let v = array![2.0, -3.0];
    let got = conditioned.apply(&v);

    let mut xc = x.clone();
    xc.column_mut(1).mapv_inplace(|z| (z - 1.0) / 2.0);
    let expected = xc.dot(&v);
    let err = (&got - &expected)
        .iter()
        .fold(0.0_f64, |m, z| m.max(z.abs()));
    assert!(
        err <= 1e-12,
        "ConditionedDesignOperator matvec should match explicit lazy column rescaling"
    );
}
