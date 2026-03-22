use gam::estimate::{
    ExternalOptimOptions, evaluate_externalcost_andridge, evaluate_externalgradients,
    optimize_external_design,
};
use gam::smooth::BlockwisePenalty;
use gam::types::{LikelihoodFamily, SasLinkSpec};
use ndarray::{Array1, Array2, array};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

fn make_binary_external_problem(
    seed: u64,
    n: usize,
    p: usize,
) -> (Array2<f64>, Array1<f64>, Array1<f64>, Vec<BlockwisePenalty>) {
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
    (x, y, w, vec![BlockwisePenalty::new(0..p, s)])
}

fn default_logit_opts() -> ExternalOptimOptions {
    ExternalOptimOptions {
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        family: LikelihoodFamily::BinomialLogit,
        compute_inference: true,
        tol: 1e-10,
        max_iter: 500,
        nullspace_dims: vec![1],
        linear_constraints: None,
        firth_bias_reduction: None,
        penalty_shrinkage_floor: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
    }
}

#[test]
fn analytic_gradient_sign_matches_localcost_trend() {
    let (x, y, w, s_list) = make_binary_external_problem(11, 120, 8);
    let offset = Array1::<f64>::zeros(y.len());
    let opts = default_logit_opts();

    let (analytic, _) = evaluate_externalgradients(
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
    let c_minus = evaluate_externalcost_andridge(
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
    let c_plus = evaluate_externalcost_andridge(
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
fn optimizer_reduces_externalobjective() {
    let (x, y, w, s_list) = make_binary_external_problem(12, 140, 10);
    let offset = Array1::<f64>::zeros(y.len());
    let opts = default_logit_opts();

    let rho0 = array![0.0];
    let c0 = evaluate_externalcost_andridge(
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
    let c1 = evaluate_externalcost_andridge(
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
        let (analytic, fd) = evaluate_externalgradients(
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

#[test]
fn sas_helpercost_depends_on_link_state() {
    let (x, y, w, s_list) = make_binary_external_problem(21, 120, 6);
    let offset = Array1::<f64>::zeros(y.len());
    let rho = array![0.7];

    let mut opts_a = default_logit_opts();
    opts_a.family = LikelihoodFamily::BinomialSas;
    opts_a.sas_link = Some(SasLinkSpec {
        initial_epsilon: 0.0,
        initial_log_delta: 0.0,
    });

    let mut opts_b = opts_a.clone();
    opts_b.sas_link = Some(SasLinkSpec {
        initial_epsilon: 0.45,
        initial_log_delta: -0.6,
    });

    let cost_a = evaluate_externalcost_andridge(
        y.view(),
        w.view(),
        x.view(),
        offset.view(),
        &s_list,
        &opts_a,
        &rho,
    )
    .map(|(cost, _)| cost)
    .expect("sas cost with baseline state");
    let cost_b = evaluate_externalcost_andridge(
        y.view(),
        w.view(),
        x.view(),
        offset.view(),
        &s_list,
        &opts_b,
        &rho,
    )
    .map(|(cost, _)| cost)
    .expect("sas cost with shifted state");

    assert!(
        (cost_a - cost_b).abs() > 1e-6,
        "SAS helper cost should change with SAS link state, got cost_a={cost_a} cost_b={cost_b}"
    );
}

#[test]
fn conditioned_helpercost_matches_fittedobjective() {
    let n = 160usize;
    let mut x = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        let t = (i as f64 + 0.5) / n as f64;
        let x1 = -3.0 + 6.0 * t;
        x[[i, 0]] = 1.0;
        x[[i, 1]] = x1;
        x[[i, 2]] = (1.7 * x1).sin();
    }
    let beta = array![0.8, -1.1, 0.55];
    let y = x.dot(&beta);
    let w = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);

    let mut s = Array2::<f64>::zeros((3, 3));
    s[[2, 2]] = 1.0;
    let s_list = vec![BlockwisePenalty::new(0..3, s)];

    let opts = ExternalOptimOptions {
        family: LikelihoodFamily::GaussianIdentity,
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: true,
        max_iter: 80,
        tol: 1e-8,
        nullspace_dims: vec![2],
        linear_constraints: None,
        firth_bias_reduction: None,
        penalty_shrinkage_floor: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
    };

    let result = optimize_external_design(
        y.view(),
        w.view(),
        x.view(),
        offset.view(),
        s_list.clone(),
        &opts,
    )
    .expect("gaussian external fit");
    let rho = result.lambdas.mapv(f64::ln);
    let helpercost = evaluate_externalcost_andridge(
        y.view(),
        w.view(),
        x.view(),
        offset.view(),
        &s_list,
        &opts,
        &rho,
    )
    .map(|(cost, _)| cost)
    .expect("conditioned helper cost");

    assert!(
        (helpercost - result.reml_score).abs() < 1e-7,
        "conditioned helper cost should match fitted REML score: helper={helpercost} fit={}",
        result.reml_score
    );
}
