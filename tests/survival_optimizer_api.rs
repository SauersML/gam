use gam::families::royston_parmar::{
    SurvivalLambdaOptimizerOptions, optimize_survival_lambdaswithmultistart,
    optimize_survival_lambdaswithmultistartfd,
};
use gam::{FitOptions, LikelihoodFamily, fit_gam};
use ndarray::{Array1, array};

fn quadraticobjective(rho: &Array1<f64>) -> f64 {
    // Minimum at [0.3, -0.7]
    let target = array![0.3, -0.7];
    let d = rho - &target;
    0.5 * d.dot(&d)
}

#[test]
fn survival_optimizer_default_uses_exactgradient_path() {
    let opts = SurvivalLambdaOptimizerOptions {
        max_iter: 120,
        tol: 1e-9,
        finite_diff_step: 1e-4,
        ..Default::default()
    };
    let heuristic = vec![1.0, 1.0];

    let exact = optimize_survival_lambdaswithmultistart(
        2,
        Some(&heuristic),
        |rho| {
            let target = array![0.3, -0.7];
            let grad = rho - &target;
            Ok((quadraticobjective(rho), grad))
        },
        &opts,
    )
    .expect("exact-gradient optimizer");

    let fd = optimize_survival_lambdaswithmultistartfd(
        2,
        Some(&heuristic),
        |rho| Ok(quadraticobjective(rho)),
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

#[test]
fn survival_optimizer_exactgradient_contract_is_rho_space() {
    let opts = SurvivalLambdaOptimizerOptions {
        max_iter: 120,
        tol: 1e-9,
        finite_diff_step: 1e-4,
        ..Default::default()
    };
    let heuristic = vec![1.0];
    let target = 0.3_f64;

    let exact = optimize_survival_lambdaswithmultistart(
        1,
        Some(&heuristic),
        |rho| {
            let diff = rho[0] - target;
            Ok((0.5 * diff * diff, array![diff]))
        },
        &opts,
    )
    .expect("rho-gradient optimizer");

    let fd = optimize_survival_lambdaswithmultistartfd(
        1,
        Some(&heuristic),
        |rho| {
            let diff = rho[0] - target;
            Ok(0.5 * diff * diff)
        },
        &opts,
    )
    .expect("fd optimizer");

    assert!(
        (exact.rho[0] - target).abs() < 1e-6,
        "exact path should converge with dV/drho, got {:?}",
        exact.rho
    );
    assert!(
        (exact.rho[0] - fd.rho[0]).abs() < 1e-6,
        "exact rho-gradient path should match FD rho optimum: exact={:?} fd={:?}",
        exact.rho,
        fd.rho
    );
}

#[test]
fn fit_gam_rejects_royston_parmar_and_points_to_survival_api() {
    let x = array![[1.0], [1.0]];
    let y = array![0.0, 1.0];
    let w = array![1.0, 1.0];
    let offset = array![0.0, 0.0];
    let s_list = Vec::new();
    let opts = FitOptions {
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        max_iter: 60,
        tol: 1e-6,
        nullspace_dims: Vec::new(),
        linear_constraints: None,
        adaptive_regularization: None,
    };

    let err = match fit_gam(
        x.view(),
        y.view(),
        w.view(),
        offset.view(),
        &s_list,
        LikelihoodFamily::RoystonParmar,
        &opts,
    ) {
        Ok(_) => panic!("RoystonParmar should be rejected by fit_gam external-design path"),
        Err(err) => err,
    };

    let msg = err.to_string();
    assert!(
        msg.contains("fit_gam external design path does not support RoystonParmar"),
        "unexpected error message: {msg}"
    );
    assert!(
        msg.contains("use survival training APIs"),
        "error should direct callers to survival-specific APIs: {msg}"
    );
}
