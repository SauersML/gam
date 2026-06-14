//! Finite-difference verification of the Matérn operator-penalty log-κ
//! ψ-derivatives (#1122).
//!
//! The existing FD coverage
//! (`basis_matern_log_kappa_first/second_derivative_fd.rs`) only checks the
//! *design-matrix* ψ-derivative. The isotropic-κ joint-REML stall in #1122 came
//! from the *penalty* ψ-derivative.
//!
//! The Matérn operator-penalty path emits up to three normalized operator dials
//! — mass `S0`, tension `S1`, stiffness `S2` — and
//! `build_matern_basis(...).penalties` is index-aligned with
//! `build_matern_basis_log_kappa_derivatives(...).penalties_derivative`. This
//! test FD-checks EACH active block's first and second log-κ derivative against
//! a central difference of the forward (normalized) penalty, so a regression in
//! any single dial — and in particular the stiffness block's mixed-curvature
//! scalar `t` whose wrong chain rule caused the stall — is caught and localized.

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

/// Forward, normalized operator penalties at this log-κ (= `rho`). Index-aligned
/// with the ψ-derivative lists.
fn penalties_at(data: &Array2<f64>, rho: f64, nu: MaternNu) -> Vec<Array2<f64>> {
    let spec = spec_at(data, rho, nu);
    build_matern_basis(data.view(), &spec).unwrap().penalties
}

fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0_f64, |m, &v| m.max(v.abs()))
}

#[test]
fn matern_penalty_log_kappa_first_derivative_matches_finite_difference() {
    let data = dataset();
    let rho: f64 = 0.3;
    for nu in [
        MaternNu::ThreeHalves,
        MaternNu::FiveHalves,
        MaternNu::SevenHalves,
    ] {
        let spec = spec_at(&data, rho, nu);
        let analytic = build_matern_basis_log_kappa_derivatives(data.view(), &spec)
            .unwrap()
            .first
            .penalties_derivative;

        let h = 1e-5;
        let plus = penalties_at(&data, rho + h, nu);
        let minus = penalties_at(&data, rho - h, nu);
        assert_eq!(
            analytic.len(),
            plus.len(),
            "nu={nu:?}: ψ-derivative block count {} must match forward penalty \
             block count {}",
            analytic.len(),
            plus.len(),
        );

        for (block, da) in analytic.iter().enumerate() {
            let num = (&plus[block] - &minus[block]) / (2.0 * h);
            let err = max_abs(&(da - &num));
            let scale = max_abs(da).max(max_abs(&num)).max(1.0);
            assert!(
                err < 1e-4 * scale,
                "nu={nu:?} penalty block {block}: first log-κ derivative must match \
                 finite difference (rel tol 1e-4·scale={:.3e}); got max abs error {err:.3e}",
                1e-4 * scale,
            );
        }
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

        let h = 1e-4;
        let plus = penalties_at(&data, rho + h, nu);
        let mid = penalties_at(&data, rho, nu);
        let minus = penalties_at(&data, rho - h, nu);
        assert_eq!(analytic.len(), mid.len());

        for (block, da) in analytic.iter().enumerate() {
            let num = (&plus[block] - 2.0 * &mid[block] + &minus[block]) / (h * h);
            let err = max_abs(&(da - &num));
            let scale = max_abs(da).max(max_abs(&num)).max(1.0);
            assert!(
                err < 5e-3 * scale,
                "nu={nu:?} penalty block {block}: second log-κ derivative must match \
                 finite difference (rel tol 5e-3·scale={:.3e}); got max abs error {err:.3e}",
                5e-3 * scale,
            );
        }
    }
}
