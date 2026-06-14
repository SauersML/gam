//! Finite-difference verification of the Matérn operator-penalty log-κ
//! ψ-derivative (#1122).
//!
//! The existing FD coverage
//! (`basis_matern_log_kappa_first/second_derivative_fd.rs`) only checks the
//! *design-matrix* ψ-derivative. The isotropic-κ joint-REML stall in #1122 came
//! from the *penalty* ψ-derivative: the operator-collocation D₂ block's
//! mixed-curvature scalar `t` had a hand-written `4t + r·t_r` chain rule that
//! did not match the true `∂t/∂(log κ)` (it confused the fixed-ℓ radial
//! derivative with the fixed-r ψ-derivative). A wrong `∂S/∂ψ` makes the REML
//! gradient inconsistent with the REML objective, so the κ-optimizer never
//! drives its projected gradient to zero and exhausts the 80-iteration cap —
//! exactly the reported symptom. This test pins `∂S/∂ψ` (and `∂²S/∂ψ²`) to a
//! central finite difference of the forward penalty so the regression cannot
//! return.

use gam::terms::basis::{
    CenterStrategy, MaternBasisSpec, MaternNu, build_matern_basis,
    build_matern_basis_log_kappa_derivatives, build_matern_basis_log_kappasecond_derivative,
};
use ndarray::Array2;

fn dataset() -> Array2<f64> {
    Array2::from_shape_vec(
        (6, 2),
        vec![
            0.0, 0.0, 0.2, 0.4, 0.7, 0.1, 1.0, 0.8, 1.3, 1.1, 0.5, 0.9,
        ],
    )
    .unwrap()
}

fn spec_at(data: &Array2<f64>, rho: f64, nu: MaternNu) -> MaternBasisSpec {
    MaternBasisSpec {
        center_strategy: CenterStrategy::UserProvided(data.clone()),
        periodic: None,
        length_scale: (-rho).exp(),
        nu,
        include_intercept: false,
        double_penalty: false,
        identifiability: Default::default(),
        aniso_log_scales: None,
        nullspace_shrinkage_survived: None,
    }
}

fn penalty_at(data: &Array2<f64>, rho: f64, nu: MaternNu) -> Array2<f64> {
    let spec = spec_at(data, rho, nu);
    let penalties = build_matern_basis(data.view(), &spec).unwrap().penalties;
    assert_eq!(
        penalties.len(),
        1,
        "operator-penalty Matérn block should expose exactly one active penalty"
    );
    penalties.into_iter().next().unwrap()
}

fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    (a - b)
        .mapv(f64::abs)
        .iter()
        .fold(0.0_f64, |m, &v| m.max(v))
}

#[test]
fn matern_penalty_log_kappa_first_derivative_matches_finite_difference() {
    let data = dataset();
    let rho: f64 = 0.3;
    for nu in [MaternNu::ThreeHalves, MaternNu::FiveHalves, MaternNu::SevenHalves] {
        let spec = spec_at(&data, rho, nu);
        let analytic = build_matern_basis_log_kappa_derivatives(data.view(), &spec)
            .unwrap()
            .first
            .penalties_derivative;
        assert_eq!(analytic.len(), 1);
        let analytic = &analytic[0];

        let h = 1e-5;
        let num = (penalty_at(&data, rho + h, nu) - penalty_at(&data, rho - h, nu)) / (2.0 * h);

        let err = max_abs_diff(analytic, &num);
        let scale = analytic
            .iter()
            .chain(num.iter())
            .fold(0.0_f64, |m, &v| m.max(v.abs()))
            .max(1.0);
        assert!(
            err < 1e-4 * scale,
            "first penalty log-kappa derivative for nu={nu:?} should match \
             finite difference (rel tol 1e-4·scale={:.3e}); got max abs error {err}",
            1e-4 * scale,
        );
    }
}

#[test]
fn matern_penalty_log_kappa_second_derivative_matches_finite_difference() {
    let data = dataset();
    let rho: f64 = 0.3;
    for nu in [MaternNu::ThreeHalves, MaternNu::FiveHalves] {
        let spec = spec_at(&data, rho, nu);
        let analytic = build_matern_basis_log_kappasecond_derivative(data.view(), &spec)
            .unwrap()
            .penaltiessecond_derivative;
        assert_eq!(analytic.len(), 1);
        let analytic = &analytic[0];

        let h = 1e-4;
        let num = (penalty_at(&data, rho + h, nu) - 2.0 * penalty_at(&data, rho, nu)
            + penalty_at(&data, rho - h, nu))
            / (h * h);

        let err = max_abs_diff(analytic, &num);
        let scale = analytic
            .iter()
            .chain(num.iter())
            .fold(0.0_f64, |m, &v| m.max(v.abs()))
            .max(1.0);
        assert!(
            err < 5e-3 * scale,
            "second penalty log-kappa derivative for nu={nu:?} should match \
             finite difference (rel tol 5e-3·scale={:.3e}); got max abs error {err}",
            5e-3 * scale,
        );
    }
}
