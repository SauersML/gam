use gam::construction::compute_penalty_square_roots;
use gam::estimate::{ExternalOptimOptions, evaluate_external_cost_and_ridge};
use gam::pirls::{PirlsConfig, fit_model_for_fixed_rho};
use gam::types::{LikelihoodFamily, LinkFunction, LogSmoothingParamsView};
use ndarray::{Array1, Array2, array};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

fn make_problem(seed: u64) -> (Array2<f64>, Array1<f64>, Array1<f64>, Vec<Array2<f64>>) {
    let n = 100;
    let p = 10;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        x[[i, 0]] = 1.0;
        for j in 1..p {
            x[[i, j]] = rng.random_range(-1.0..1.0);
        }
    }
    let beta = Array1::from_shape_fn(p, |j| if j == 0 { -0.1 } else { 0.2 / j as f64 });
    let eta = x.dot(&beta);
    let y = eta.mapv(|e| {
        let prob = 1.0 / (1.0 + (-e).exp());
        if rng.random::<f64>() < prob { 1.0 } else { 0.0 }
    });
    let w = Array1::<f64>::ones(n);
    let mut s = Array2::<f64>::zeros((p, p));
    for j in 1..p {
        s[[j, j]] = 1.0;
    }
    (x, y, w, vec![s])
}

fn fit_beta_norm(
    x: &Array2<f64>,
    y: &Array1<f64>,
    w: &Array1<f64>,
    rs: &[Array2<f64>],
    rho: f64,
    firth: bool,
) -> f64 {
    let cfg = PirlsConfig {
        link_function: LinkFunction::Logit,
        max_iterations: 500,
        convergence_tolerance: 1e-10,
        firth_bias_reduction: firth,
    };
    let offset = Array1::<f64>::zeros(y.len());
    let (fit, _) = fit_model_for_fixed_rho(
        LogSmoothingParamsView::new(array![rho].view()),
        x.view(),
        offset.view(),
        y.view(),
        w.view(),
        rs,
        None,
        None,
        x.ncols(),
        &cfg,
        None,
        None,
        None,
        None,
    )
    .expect("fit");
    fit.beta_transformed
        .dot(fit.beta_transformed.as_ref())
        .sqrt()
}

fn proxy_cost_with_pirls(
    x: &Array2<f64>,
    y: &Array1<f64>,
    w: &Array1<f64>,
    rs: &[Array2<f64>],
    s: &Array2<f64>,
    rho: f64,
    firth: bool,
) -> f64 {
    let cfg = PirlsConfig {
        link_function: LinkFunction::Logit,
        max_iterations: 500,
        convergence_tolerance: 1e-10,
        firth_bias_reduction: firth,
    };
    let offset = Array1::<f64>::zeros(y.len());
    let (fit, _) = fit_model_for_fixed_rho(
        LogSmoothingParamsView::new(array![rho].view()),
        x.view(),
        offset.view(),
        y.view(),
        w.view(),
        rs,
        None,
        None,
        x.ncols(),
        &cfg,
        None,
        None,
        None,
        None,
    )
    .expect("fit");
    let lambda = rho.exp();
    let b = fit.beta_transformed.as_ref().to_owned();
    let penalty = 0.5 * lambda * b.dot(&s.dot(&b));
    fit.deviance + penalty
}

#[test]
fn firth_fd_step_size_sensitivity() {
    let (x, y, w, s_list) = make_problem(31);
    let offset = Array1::<f64>::zeros(y.len());
    let opts = ExternalOptimOptions {
        family: LikelihoodFamily::BinomialLogit,
        tol: 1e-10,
        max_iter: 500,
        nullspace_dims: vec![1],
        linear_constraints: None,
        firth_bias_reduction: None,
    };
    let base_rho = 12.0;
    let cost_at = |rho: f64| -> f64 {
        evaluate_external_cost_and_ridge(
            y.view(),
            w.view(),
            x.view(),
            offset.view(),
            &s_list,
            &opts,
            &array![rho],
        )
        .map(|(c, _)| c)
        .expect("cost")
    };
    let wide_trend = cost_at(base_rho + 1.0) - cost_at(base_rho - 1.0);
    let trend_sign = wide_trend > 0.0;
    let step_sizes = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005];
    let mut consistent_count = 0;
    for &h in &step_sizes {
        let fd = (cost_at(base_rho + h) - cost_at(base_rho - h)) / (2.0 * h);
        if (fd > 0.0) == trend_sign {
            consistent_count += 1;
        }
    }
    assert!(consistent_count >= step_sizes.len() / 2);
}

#[test]
fn firth_beta_monotonicity_comparison() {
    let (x, y, w, s_list) = make_problem(31);
    let rs = compute_penalty_square_roots(&s_list).expect("roots");
    let deltas = [
        -0.010_f64, -0.005, -0.002, -0.001, 0.0, 0.001, 0.002, 0.005, 0.010,
    ];
    let betas_firth: Vec<f64> = deltas
        .iter()
        .map(|&d| fit_beta_norm(&x, &y, &w, &rs, 12.0 + d, true))
        .collect();
    let betas_no_firth: Vec<f64> = deltas
        .iter()
        .map(|&d| fit_beta_norm(&x, &y, &w, &rs, 12.0 + d, false))
        .collect();
    let count_sign_changes = |values: &[f64]| -> usize {
        values
            .windows(2)
            .filter(|w| (w[1] - w[0]).signum() != 0.0)
            .zip(values.windows(2).skip(1))
            .filter(|(a, b)| (a[1] - a[0]).signum() * (b[1] - b[0]).signum() < 0.0)
            .count()
    };
    let changes_firth = count_sign_changes(&betas_firth);
    let changes_no_firth = count_sign_changes(&betas_no_firth);
    assert!(changes_no_firth <= changes_firth || changes_no_firth <= 2);
}

#[test]
fn firth_cost_oscillation_vs_no_firth() {
    let (x, y, w, s_list) = make_problem(31);
    let rs = compute_penalty_square_roots(&s_list).expect("roots");
    let s = &s_list[0];
    let steps: Vec<f64> = (-20..=20).map(|i| i as f64 * 0.001).collect();
    let cost_firth: Vec<f64> = steps
        .iter()
        .map(|&d| proxy_cost_with_pirls(&x, &y, &w, &rs, s, 12.0 + d, true))
        .collect();
    let cost_no_firth: Vec<f64> = steps
        .iter()
        .map(|&d| proxy_cost_with_pirls(&x, &y, &w, &rs, s, 12.0 + d, false))
        .collect();
    let count_direction_changes = |costs: &[f64]| -> usize {
        let mut changes = 0;
        for i in 1..costs.len() - 1 {
            let left = costs[i] - costs[i - 1];
            let right = costs[i + 1] - costs[i];
            if left * right < 0.0 {
                changes += 1;
            }
        }
        changes
    };
    let firth_changes = count_direction_changes(&cost_firth);
    let no_firth_changes = count_direction_changes(&cost_no_firth);
    assert!(no_firth_changes <= firth_changes || no_firth_changes <= 5);
}
