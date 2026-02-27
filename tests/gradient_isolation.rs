use gam::estimate::{
    ExternalOptimOptions, evaluate_external_cost_and_ridge, evaluate_external_gradients,
    optimize_external_design,
};
use gam::types::LikelihoodFamily;
use ndarray::{Array1, Array2, array};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

fn make_binary_external_problem(
    seed: u64,
    n: usize,
    p: usize,
) -> (Array2<f64>, Array1<f64>, Array1<f64>, Vec<Array2<f64>>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        x[[i, 0]] = 1.0;
        for j in 1..p {
            x[[i, j]] = rng.random_range(-1.5..1.5);
        }
    }

    let mut beta = Array1::<f64>::zeros(p);
    beta[0] = -0.15;
    for j in 1..p {
        beta[j] = 0.3 / (j as f64).sqrt();
    }
    let eta = x.dot(&beta);
    let y = eta.mapv(|e| {
        let p = 1.0 / (1.0 + (-e).exp());
        if rng.random::<f64>() < p { 1.0 } else { 0.0 }
    });
    let w = Array1::<f64>::ones(n);

    let mut s = Array2::<f64>::zeros((p, p));
    for j in 1..p {
        s[[j, j]] = 1.0;
    }
    (x, y, w, vec![s])
}

fn default_logit_opts() -> ExternalOptimOptions {
    ExternalOptimOptions {
        family: LikelihoodFamily::BinomialLogit,
        tol: 1e-10,
        max_iter: 500,
        nullspace_dims: vec![1],
        linear_constraints: None,
        firth_bias_reduction: None,
    }
}

#[test]
fn analytic_gradient_sign_matches_local_cost_trend() {
    let (x, y, w, s_list) = make_binary_external_problem(11, 120, 8);
    let offset = Array1::<f64>::zeros(y.len());
    let opts = default_logit_opts();

    let (analytic, _fd) = evaluate_external_gradients(
        y.view(),
        w.view(),
        x.view(),
        offset.view(),
        &s_list,
        &opts,
        &array![12.0],
    )
    .expect("gradients");

    let h = 0.25;
    let c_minus = evaluate_external_cost_and_ridge(
        y.view(),
        w.view(),
        x.view(),
        offset.view(),
        &s_list,
        &opts,
        &array![12.0 - h],
    )
    .map(|(c, _)| c)
    .expect("cost-");
    let c_plus = evaluate_external_cost_and_ridge(
        y.view(),
        w.view(),
        x.view(),
        offset.view(),
        &s_list,
        &opts,
        &array![12.0 + h],
    )
    .map(|(c, _)| c)
    .expect("cost+");
    let trend = c_plus - c_minus;

    assert!(
        analytic[0].abs() > 1e-7 && trend.abs() > 1e-7,
        "uninformative gradient/trend: analytic={} trend={}",
        analytic[0],
        trend
    );
    assert_eq!(
        analytic[0] > 0.0,
        trend > 0.0,
        "analytic sign should match cost trend sign: analytic={:+.4e} trend={:+.4e}",
        analytic[0],
        trend
    );
}

#[test]
fn optimizer_reduces_external_objective() {
    let (x, y, w, s_list) = make_binary_external_problem(12, 140, 10);
    let offset = Array1::<f64>::zeros(y.len());
    let opts = default_logit_opts();

    let rho0 = array![0.0];
    let c0 = evaluate_external_cost_and_ridge(
        y.view(),
        w.view(),
        x.view(),
        offset.view(),
        &s_list,
        &opts,
        &rho0,
    )
    .map(|(c, _)| c)
    .expect("initial cost");

    let result = optimize_external_design(
        y.view(),
        w.view(),
        x.view(),
        offset.view(),
        s_list.clone(),
        &opts,
    )
    .expect("opt");

    let rho_opt = result.lambdas.mapv(f64::ln);
    let c1 = evaluate_external_cost_and_ridge(
        y.view(),
        w.view(),
        x.view(),
        offset.view(),
        &s_list,
        &opts,
        &rho_opt,
    )
    .map(|(c, _)| c)
    .expect("final cost");

    assert!(
        c1 <= c0 + 1e-6,
        "optimizer did not improve cost: c0={c0} c1={c1}"
    );
}

#[test]
fn gradient_components_remain_finite_across_rho_sweep() {
    let (x, y, w, s_list) = make_binary_external_problem(13, 160, 9);
    let offset = Array1::<f64>::zeros(y.len());
    let opts = default_logit_opts();
    for rho in [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0] {
        let (analytic, fd) = evaluate_external_gradients(
            y.view(),
            w.view(),
            x.view(),
            offset.view(),
            &s_list,
            &opts,
            &array![rho],
        )
        .expect("sweep gradient");
        assert!(
            analytic[0].is_finite(),
            "analytic gradient non-finite at rho={rho}"
        );
        assert!(fd[0].is_finite(), "fd gradient non-finite at rho={rho}");
    }
}
