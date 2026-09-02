use gam::estimate::{ExternalOptimOptions, evaluate_externalcost_andridge, evaluate_externalgradient};
use gam::mixture_link::state_from_sasspec;
use gam::smooth::BlockwisePenalty;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, SasLinkSpec, StandardLink};
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
        family: binomial_logit_spec(),
        compute_inference: true,
        skip_rho_posterior_inference: false,
        tol: 1e-10,
        max_iter: 500,
        nullspace_dims: vec![1],
        linear_constraints: None,
        firth_bias_reduction: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persistent_warm_start_store: None,
    }
}

fn binomial_logit_spec() -> LikelihoodSpec {
    LikelihoodSpec::new(
        ResponseFamily::Binomial,
        InverseLink::Standard(StandardLink::Logit),
    )
}

fn binomial_sas_spec(sas_link: SasLinkSpec) -> LikelihoodSpec {
    LikelihoodSpec::new(
        ResponseFamily::Binomial,
        InverseLink::Sas(state_from_sasspec(sas_link).expect("valid SAS link state")),
    )
}

#[test]
fn analytic_gradient_sign_matches_localcost_trend() {
    let (x, y, w, s_list) = make_binary_external_problem(11, 120, 8);
    let offset = Array1::<f64>::zeros(y.len());
    let opts = default_logit_opts();

    let analytic = evaluate_externalgradient(
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
fn gradient_components_remain_finite_across_rho_sweep() {
    let (x, y, w, s_list) = make_binary_external_problem(13, 160, 9);
    let offset = Array1::<f64>::zeros(y.len());
    let opts = default_logit_opts();
    for rho in [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0] {
        let analytic = evaluate_externalgradient(
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
    }
}

#[test]
fn sas_helpercost_depends_on_link_state() {
    let (x, y, w, s_list) = make_binary_external_problem(21, 120, 6);
    let offset = Array1::<f64>::zeros(y.len());
    let rho = array![0.7];

    let mut opts_a = default_logit_opts();
    let sas_link_a = SasLinkSpec {
        initial_epsilon: 0.0,
        initial_log_delta: 0.0,
    };
    opts_a.family = binomial_sas_spec(sas_link_a);
    opts_a.sas_link = Some(sas_link_a);

    let mut opts_b = opts_a.clone();
    let sas_link_b = SasLinkSpec {
        initial_epsilon: 0.45,
        initial_log_delta: -0.6,
    };
    opts_b.family = binomial_sas_spec(sas_link_b);
    opts_b.sas_link = Some(sas_link_b);

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

