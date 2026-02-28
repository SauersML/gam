use gam::estimate::{
    SmoothingBfgsOptions, optimize_log_smoothing_with_multistart,
    optimize_log_smoothing_with_multistart_with_gradient,
};
use gam::seeding::{SeedConfig, SeedRiskProfile};
use ndarray::array;

#[test]
fn exact_gradient_optimizer_matches_fd_on_quadratic_objective() {
    // Convex quadratic in rho-space with known minimizer at (1.5, -0.25).
    let objective = |rho: &ndarray::Array1<f64>| {
        let v0 = rho[0] - 1.5;
        let v1 = rho[1] + 0.25;
        Ok::<f64, gam::estimate::EstimationError>(0.5 * (v0 * v0 + 2.0 * v1 * v1))
    };
    let objective_with_gradient = |rho: &ndarray::Array1<f64>| {
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
        fd_hessian_max_dim: usize::MAX,
        seed_config: SeedConfig {
            bounds: (-4.0, 4.0),
            max_seeds: 16,
            screening_budget: 6,
            screen_max_inner_iterations: 5,
            risk_profile: SeedRiskProfile::Gaussian,
        },
    };

    let fd_res = optimize_log_smoothing_with_multistart(2, None, objective, &opts).unwrap();
    let exact_res = optimize_log_smoothing_with_multistart_with_gradient(
        2,
        None,
        objective_with_gradient,
        &opts,
    )
    .unwrap();

    assert!((fd_res.rho[0] - 1.5).abs() < 1e-4);
    assert!((fd_res.rho[1] + 0.25).abs() < 1e-4);
    assert!((exact_res.rho[0] - 1.5).abs() < 1e-6);
    assert!((exact_res.rho[1] + 0.25).abs() < 1e-6);
    assert!(exact_res.final_grad_norm <= fd_res.final_grad_norm + 1e-8);
}
