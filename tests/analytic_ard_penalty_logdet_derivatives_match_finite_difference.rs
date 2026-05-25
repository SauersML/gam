use approx::assert_abs_diff_eq;
use gam::terms::analytic_penalties::{ARDPenalty, AnalyticPenaltyKind, PsiSlice};
use ndarray::array;
use std::sync::Arc;

#[test]
fn analytic_ard_penalty_logdet_derivatives_match_finite_difference() {
    let latent_dim = 3;
    let target = array![0.2_f64, -0.7, 0.5, 1.1, -0.3, 0.4, -0.9, 0.8, 0.6];
    let penalty = AnalyticPenaltyKind::Ard(Arc::new(ARDPenalty::new(
        PsiSlice::full(target.len(), Some(latent_dim)),
        latent_dim,
    )));
    let rho = array![0.0_f64, 0.3, -0.2];
    let frozen = penalty.freeze(target.clone(), rho.clone());

    let dense = frozen.as_dense();
    for i in 0..dense.nrows() {
        for j in 0..dense.ncols() {
            assert_abs_diff_eq!(dense[[i, j]], dense[[j, i]], epsilon = 1e-12);
            assert!(
                (dense[[i, j]] - dense[[j, i]]).abs() <= 1e-12,
                "ARD Hessian must be symmetric"
            );
        }
    }
    for i in 0..dense.nrows() {
        assert!(
            dense[[i, i]] >= 0.0,
            "ARD Hessian diagonal entries must be non-negative for PSD"
        );
    }

    let lambda = 1.37_f64;
    let eps = 1e-6;
    let f = |l: f64| {
        frozen
            .log_det_plus_lambda_i(l)
            .expect("log-det should exist for λ>0")
    };
    let d1_fd = (f(lambda + eps) - f(lambda - eps)) / (2.0 * eps);
    let d1_analytic = {
        let diag = frozen.diag();
        diag.iter().map(|&d| 1.0 / (d + lambda)).sum::<f64>()
    };
    assert_abs_diff_eq!(d1_analytic, d1_fd, epsilon = 1e-5);
    assert!(
        (d1_analytic - d1_fd).abs() <= 1e-5,
        "log|S+λI| first derivative should match finite differences"
    );

    let eps2 = 1e-4;
    let d2_fd = (f(lambda + eps2) - 2.0 * f(lambda) + f(lambda - eps2)) / (eps2 * eps2);
    let d2_analytic = {
        let diag = frozen.diag();
        -diag
            .iter()
            .map(|&d| 1.0 / ((d + lambda) * (d + lambda)))
            .sum::<f64>()
    };
    assert_abs_diff_eq!(d2_analytic, d2_fd, epsilon = 1e-3);
    assert!(
        (d2_analytic - d2_fd).abs() <= 1e-3,
        "log|S+λI| second derivative should match finite differences"
    );
}
