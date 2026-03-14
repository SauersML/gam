use gam::pirls::update_glmvectors_by_family;
use gam::probability::normal_cdf;
use gam::types::GlmLikelihoodFamily;
use gam::{FitOptions, LikelihoodFamily, fit_gam, predict_gam};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

#[test]
fn probit_fit_and_predict_fast_integration() {
    let n = 400usize;
    let mut x = Array2::<f64>::zeros((n, 2));
    let mut y = Array1::<f64>::zeros(n);
    let mut rng = StdRng::seed_from_u64(7);

    for i in 0..n {
        let xi = -2.0 + 4.0 * (i as f64) / (n as f64 - 1.0);
        let eta = -0.3 + 1.1 * xi;
        let p = normal_cdf(eta);
        x[[i, 0]] = 1.0;
        x[[i, 1]] = xi;
        y[i] = if rng.random::<f64>() < p { 1.0 } else { 0.0 };
    }

    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);
    let mut s = Array2::<f64>::zeros((2, 2));
    s[[1, 1]] = 1.0;
    let s_list = vec![s];

    let fit = fit_gam(
        x.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &s_list,
        LikelihoodFamily::BinomialProbit,
        &FitOptions {
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            compute_inference: true,
            max_iter: 60,
            tol: 1e-6,
            nullspace_dims: vec![1],
            linear_constraints: None,
            adaptive_regularization: None,
            penalty_shrinkage_floor: None,
        },
    )
    .expect("probit fit should succeed");

    assert_eq!(fit.beta.len(), 2);
    assert_eq!(fit.lambdas.len(), 1);
    assert!(fit.edf_total().is_some_and(f64::is_finite));

    let pred = predict_gam(
        x.view(),
        fit.beta.view(),
        offset.view(),
        LikelihoodFamily::BinomialProbit,
    )
    .expect("probit predict should succeed");

    assert!(
        pred.mean
            .iter()
            .all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0)
    );

    let brier = (&pred.mean - &y)
        .mapv(|v| v * v)
        .mean()
        .unwrap_or(f64::INFINITY);
    assert!(
        brier < 0.25,
        "unexpectedly poor probit fit: brier={brier:.6e}"
    );
}

#[test]
fn probitworkingvectors_are_finite_for_extreme_eta() {
    let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0]);
    let eta = Array1::from_vec(vec![-100.0, -20.0, 0.0, 20.0, 100.0]);
    let w = Array1::ones(y.len());
    let mut mu = Array1::zeros(y.len());
    let mut weights = Array1::zeros(y.len());
    let mut z = Array1::zeros(y.len());

    update_glmvectors_by_family(
        y.view(),
        &eta,
        GlmLikelihoodFamily::BinomialProbit,
        w.view(),
        &mut mu,
        &mut weights,
        &mut z,
    )
    .expect("probit working-vector update should succeed");

    assert!(mu.iter().all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0));
    assert!(weights.iter().all(|v| v.is_finite() && *v >= 0.0));
    assert!(z.iter().all(|v| v.is_finite()));
}

#[test]
fn cloglog_fit_and_predict_fast_integration() {
    let n = 400usize;
    let mut x = Array2::<f64>::zeros((n, 2));
    let mut y = Array1::<f64>::zeros(n);
    let mut rng = StdRng::seed_from_u64(17);

    for i in 0..n {
        let xi = -2.0 + 4.0 * (i as f64) / (n as f64 - 1.0);
        let eta = -0.4 + 0.9 * xi;
        let z = eta.clamp(-30.0, 30.0);
        let p = 1.0 - (-(z.exp())).exp();
        x[[i, 0]] = 1.0;
        x[[i, 1]] = xi;
        y[i] = if rng.random::<f64>() < p { 1.0 } else { 0.0 };
    }

    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);
    let mut s = Array2::<f64>::zeros((2, 2));
    s[[1, 1]] = 1.0;
    let s_list = vec![s];

    let fit = fit_gam(
        x.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &s_list,
        LikelihoodFamily::BinomialCLogLog,
        &FitOptions {
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            compute_inference: true,
            max_iter: 60,
            tol: 1e-6,
            nullspace_dims: vec![1],
            linear_constraints: None,
            adaptive_regularization: None,
            penalty_shrinkage_floor: None,
        },
    )
    .expect("cloglog fit should succeed");

    let pred = predict_gam(
        x.view(),
        fit.beta.view(),
        offset.view(),
        LikelihoodFamily::BinomialCLogLog,
    )
    .expect("cloglog predict should succeed");

    assert!(
        pred.mean
            .iter()
            .all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0)
    );
}

#[test]
fn cloglogworkingvectors_are_finite_for_extreme_eta() {
    let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0]);
    let eta = Array1::from_vec(vec![-100.0, -20.0, 0.0, 20.0, 100.0]);
    let w = Array1::ones(y.len());
    let mut mu = Array1::zeros(y.len());
    let mut weights = Array1::zeros(y.len());
    let mut z = Array1::zeros(y.len());

    update_glmvectors_by_family(
        y.view(),
        &eta,
        GlmLikelihoodFamily::BinomialCLogLog,
        w.view(),
        &mut mu,
        &mut weights,
        &mut z,
    )
    .expect("cloglog working-vector update should succeed");

    assert!(mu.iter().all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0));
    assert!(weights.iter().all(|v| v.is_finite() && *v >= 0.0));
    assert!(z.iter().all(|v| v.is_finite()));
}
