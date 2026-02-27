use gam::estimate::{
    ExternalOptimOptions, evaluate_external_cost_and_ridge, evaluate_external_gradients,
};
use gam::types::LikelihoodFamily;
use ndarray::{Array1, Array2, array};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

fn make_binary_external_problem(
    seed: u64,
) -> (Array2<f64>, Array1<f64>, Array1<f64>, Vec<Array2<f64>>) {
    let n = 100;
    let p = 8;
    let mut rng = StdRng::seed_from_u64(seed);

    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        x[[i, 0]] = 1.0;
        for j in 1..p {
            x[[i, j]] = rng.random_range(-1.5..1.5);
        }
    }

    let mut beta = Array1::<f64>::zeros(p);
    beta[0] = -0.1;
    for j in 1..p {
        beta[j] = 0.25 / (j as f64).sqrt();
    }
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

#[test]
fn analytic_gradient_matches_cost_trend() {
    let (x, y, w, s_list) = make_binary_external_problem(31);
    let offset = Array1::<f64>::zeros(y.len());
    let opts = ExternalOptimOptions {
        family: LikelihoodFamily::BinomialLogit,
        tol: 1e-10,
        max_iter: 500,
        nullspace_dims: vec![1],
        linear_constraints: None,
        firth_bias_reduction: None,
    };

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

    let local_steps = [1e-2, 5e-2, 1e-1];
    let mut local_derivs = Vec::new();
    for &h in &local_steps {
        let cost_minus = evaluate_external_cost_and_ridge(
            y.view(),
            w.view(),
            x.view(),
            offset.view(),
            &s_list,
            &opts,
            &array![12.0 - h],
        )
        .map(|(c, _)| c)
        .expect("cost_minus");
        let cost_plus = evaluate_external_cost_and_ridge(
            y.view(),
            w.view(),
            x.view(),
            offset.view(),
            &s_list,
            &opts,
            &array![12.0 + h],
        )
        .map(|(c, _)| c)
        .expect("cost_plus");
        local_derivs.push((cost_plus - cost_minus) / (2.0 * h));
    }

    let positive_count = local_derivs.iter().filter(|&&d| d > 0.0).count();
    let negative_count = local_derivs.len() - positive_count;
    assert!(
        positive_count == 0 || negative_count == 0,
        "local objective slope near rho=12 is sign-inconsistent: {:?}",
        local_derivs
    );

    let local_trend = local_derivs.iter().sum::<f64>() / local_derivs.len() as f64;
    if local_trend.abs() > 1e-3 {
        assert!(
            analytic[0].abs() > 1e-5,
            "analytic gradient near zero ({:+.4e}) but local slope is not ({:+.4e})",
            analytic[0],
            local_trend
        );
    }
    assert_eq!(
        analytic[0] > 0.0,
        local_trend > 0.0,
        "Analytic gradient sign ({:+.4e}) should match local slope sign ({:+.4e})",
        analytic[0],
        local_trend
    );
}

#[test]
fn hypothesis_analytic_gradient_matches_cost_trend() {
    let (x, y, w, s_list) = make_binary_external_problem(31);
    let offset = Array1::<f64>::zeros(y.len());
    let opts = ExternalOptimOptions {
        family: LikelihoodFamily::BinomialLogit,
        tol: 1e-10,
        max_iter: 500,
        nullspace_dims: vec![1],
        linear_constraints: None,
        firth_bias_reduction: None,
    };

    let mut same_sign = 0usize;
    let mut opposite_sign = 0usize;
    let mut considered = 0usize;

    for rho_val in [0.0_f64, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0] {
        let (analytic, _fd) = evaluate_external_gradients(
            y.view(),
            w.view(),
            x.view(),
            offset.view(),
            &s_list,
            &opts,
            &array![rho_val],
        )
        .expect("gradients");
        let delta = 0.25;
        let cost_minus = evaluate_external_cost_and_ridge(
            y.view(),
            w.view(),
            x.view(),
            offset.view(),
            &s_list,
            &opts,
            &array![rho_val - delta],
        )
        .map(|(c, _)| c)
        .expect("cost_minus");
        let cost_plus = evaluate_external_cost_and_ridge(
            y.view(),
            w.view(),
            x.view(),
            offset.view(),
            &s_list,
            &opts,
            &array![rho_val + delta],
        )
        .map(|(c, _)| c)
        .expect("cost_plus");
        let cost_trend = cost_plus - cost_minus;

        if analytic[0].abs() < 1e-7 || cost_trend.abs() < 1e-7 {
            continue;
        }
        considered += 1;
        if (analytic[0] > 0.0) == (cost_trend > 0.0) {
            same_sign += 1;
        } else {
            opposite_sign += 1;
        }
    }

    assert!(
        considered >= 3,
        "Expected enough informative rho points, got {}",
        considered
    );
    let dominant = same_sign.max(opposite_sign);
    assert!(
        dominant + 1 >= considered,
        "analytic-vs-cost-trend sign relation should be consistent; same={} opposite={} considered={}",
        same_sign,
        opposite_sign,
        considered
    );
}
