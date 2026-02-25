use gam::families::royston_parmar::{
    SurvivalLambdaOptimizerOptions, optimize_survival_lambdas_with_multistart,
    optimize_survival_lambdas_with_multistart_fd,
};
use ndarray::{Array1, array};

fn quadratic_objective(rho: &Array1<f64>) -> f64 {
    // Minimum at [0.3, -0.7]
    let target = array![0.3, -0.7];
    let d = rho - &target;
    0.5 * d.dot(&d)
}

#[test]
fn survival_optimizer_default_uses_exact_gradient_path() {
    let opts = SurvivalLambdaOptimizerOptions {
        max_iter: 120,
        tol: 1e-9,
        finite_diff_step: 1e-4,
        ..Default::default()
    };
    let heuristic = vec![1.0, 1.0];

    let exact = optimize_survival_lambdas_with_multistart(
        2,
        Some(&heuristic),
        |rho| {
            let target = array![0.3, -0.7];
            let grad = rho - &target;
            Ok((quadratic_objective(rho), grad))
        },
        &opts,
    )
    .expect("exact-gradient optimizer");

    let fd = optimize_survival_lambdas_with_multistart_fd(
        2,
        Some(&heuristic),
        |rho| Ok(quadratic_objective(rho)),
        &opts,
    )
    .expect("fd-gradient optimizer");

    let target = array![0.3, -0.7];
    assert!(
        (&exact.rho - &target).mapv(f64::abs).sum() < 1e-4,
        "exact path should converge near target, got {:?}",
        exact.rho
    );
    assert!(
        (&fd.rho - &target).mapv(f64::abs).sum() < 3e-3,
        "fd path should also converge near target, got {:?}",
        fd.rho
    );
    assert!(
        (exact.final_value - fd.final_value).abs() < 1e-6,
        "exact and fd final values should agree on smooth convex objective: exact={} fd={}",
        exact.final_value,
        fd.final_value
    );
}
