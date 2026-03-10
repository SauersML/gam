use gam::estimate::{
    SmoothingBfgsOptions, SmoothingOptimizerKind, optimize_log_smoothingwithmultistart,
    optimize_log_smoothingwithmultistartwithgradient,
};
use gam::seeding::{SeedConfig, SeedRiskProfile};
use ndarray::array;

#[test]
fn exactgradient_optimizer_matchesfd_on_quadraticobjective() {
    // Convex quadratic in rho-space with known minimizer at (1.5, -0.25).
    let objective = |rho: &ndarray::Array1<f64>| {
        let v0 = rho[0] - 1.5;
        let v1 = rho[1] + 0.25;
        Ok::<f64, gam::estimate::EstimationError>(0.5 * (v0 * v0 + 2.0 * v1 * v1))
    };
    let objectivewithgradient = |rho: &ndarray::Array1<f64>| {
        let v0 = rho[0] - 1.5;
        let v1 = rho[1] + 0.25;
        Ok::<(f64, ndarray::Array1<f64>), gam::estimate::EstimationError>((
            0.5 * (v0 * v0 + 2.0 * v1 * v1),
            array![v0, 2.0 * v1],
        ))
    };

    let opts = SmoothingBfgsOptions {
        max_iter: 200,
        tol: 1e-8,
        finite_diff_step: 1e-4,
        fdhessian_max_dim: usize::MAX,
        optimizer_kind: SmoothingOptimizerKind::Bfgs,
        seed_config: SeedConfig {
            bounds: (-4.0, 4.0),
            max_seeds: 16,
            screening_budget: 6,
            screen_max_inner_iterations: 5,
            risk_profile: SeedRiskProfile::Gaussian,
        },
    };

    let fd_res = optimize_log_smoothingwithmultistart(2, None, objective, &opts).unwrap();
    let exact_res = optimize_log_smoothingwithmultistartwithgradient(
        2,
        None,
        objectivewithgradient,
        &opts,
    )
    .unwrap();

    assert!((fd_res.rho[0] - 1.5).abs() < 1e-4);
    assert!((fd_res.rho[1] + 0.25).abs() < 1e-4);
    assert!((exact_res.rho[0] - 1.5).abs() < 1e-6);
    assert!((exact_res.rho[1] + 0.25).abs() < 1e-6);
    assert!(exact_res.finalgrad_norm <= fd_res.finalgrad_norm + 1e-8);
}
