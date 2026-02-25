use gam::{
    FitOptions, LikelihoodFamily, MeanIntervalMethod, PredictUncertaintyOptions,
    coefficient_uncertainty, fit_gam, predict_gam_with_uncertainty,
};
use ndarray::{Array1, Array2};

#[test]
fn fit_exposes_posterior_covariance_and_standard_errors() {
    let n = 120usize;
    let mut x = Array2::<f64>::zeros((n, 3));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let t = -1.0 + 2.0 * (i as f64) / (n as f64 - 1.0);
        x[[i, 0]] = 1.0;
        x[[i, 1]] = t;
        x[[i, 2]] = t * t;
        y[i] = 0.3 + 1.2 * t - 0.4 * t * t;
    }

    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);
    let mut s = Array2::<f64>::zeros((3, 3));
    s[[1, 1]] = 1.0;
    s[[2, 2]] = 1.0;

    let fit = fit_gam(
        x.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &[s],
        LikelihoodFamily::GaussianIdentity,
        &FitOptions {
            max_iter: 40,
            tol: 1e-6,
            nullspace_dims: vec![1],
        },
    )
    .expect("fit should succeed");

    let cov = fit
        .beta_covariance
        .as_ref()
        .expect("conditional covariance should be available");
    assert_eq!(cov.nrows(), fit.beta.len());
    assert_eq!(cov.ncols(), fit.beta.len());
    assert!(cov.iter().all(|v| v.is_finite()));

    let se = fit
        .beta_standard_errors
        .as_ref()
        .expect("standard errors should be available");
    assert_eq!(se.len(), fit.beta.len());
    assert!(se.iter().all(|v| v.is_finite() && *v >= 0.0));

    let coef_ci = coefficient_uncertainty(&fit, 0.95, true).expect("coefficient CI should work");
    assert_eq!(coef_ci.estimate.len(), fit.beta.len());
    assert_eq!(coef_ci.standard_error.len(), fit.beta.len());
    assert!(
        coef_ci
            .lower
            .iter()
            .zip(coef_ci.upper.iter())
            .all(|(&l, &u)| l.is_finite() && u.is_finite() && l <= u)
    );
}

#[test]
fn prediction_uncertainty_is_finite_and_well_shaped() {
    let n = 80usize;
    let mut x = Array2::<f64>::zeros((n, 2));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let t = -2.0 + 4.0 * (i as f64) / (n as f64 - 1.0);
        x[[i, 0]] = 1.0;
        x[[i, 1]] = t;
        y[i] = 1.0 / (1.0 + (-(-0.2 + 0.9 * t)).exp());
    }

    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);
    let mut s = Array2::<f64>::zeros((2, 2));
    s[[1, 1]] = 1.0;

    let fit = fit_gam(
        x.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &[s],
        LikelihoodFamily::BinomialLogit,
        &FitOptions {
            max_iter: 50,
            tol: 1e-6,
            nullspace_dims: vec![1],
        },
    )
    .expect("fit should succeed");

    let pred = predict_gam_with_uncertainty(
        x.view(),
        fit.beta.view(),
        offset.view(),
        LikelihoodFamily::BinomialLogit,
        &fit,
        &PredictUncertaintyOptions {
            confidence_level: 0.95,
            prefer_corrected_covariance: true,
            mean_interval_method: MeanIntervalMethod::TransformEta,
            include_observation_interval: true,
        },
    )
    .expect("prediction uncertainty should succeed");

    assert_eq!(pred.eta.len(), n);
    assert_eq!(pred.eta_standard_error.len(), n);
    assert_eq!(pred.mean_standard_error.len(), n);
    assert!(
        pred.eta_standard_error
            .iter()
            .all(|v| v.is_finite() && *v >= 0.0)
    );
    assert!(
        pred.mean_standard_error
            .iter()
            .all(|v| v.is_finite() && *v >= 0.0)
    );
    assert!(
        pred.mean_lower
            .iter()
            .zip(pred.mean_upper.iter())
            .all(|(&l, &u)| l.is_finite() && u.is_finite() && l <= u && l >= 0.0 && u <= 1.0)
    );
    assert!(pred.observation_lower.is_none());
    assert!(pred.observation_upper.is_none());
}

#[test]
fn gaussian_prediction_intervals_include_observation_noise() {
    let n = 100usize;
    let mut x = Array2::<f64>::zeros((n, 2));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let t = -1.0 + 2.0 * (i as f64) / (n as f64 - 1.0);
        x[[i, 0]] = 1.0;
        x[[i, 1]] = t;
        y[i] = 0.5 + 0.7 * t + 0.2 * (3.0 * t).sin();
    }
    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);
    let mut s = Array2::<f64>::zeros((2, 2));
    s[[1, 1]] = 1.0;

    let fit = fit_gam(
        x.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &[s],
        LikelihoodFamily::GaussianIdentity,
        &FitOptions {
            max_iter: 40,
            tol: 1e-6,
            nullspace_dims: vec![1],
        },
    )
    .expect("fit should succeed");

    let pred = predict_gam_with_uncertainty(
        x.view(),
        fit.beta.view(),
        offset.view(),
        LikelihoodFamily::GaussianIdentity,
        &fit,
        &PredictUncertaintyOptions::default(),
    )
    .expect("prediction uncertainty should succeed");

    let obs_lower = pred
        .observation_lower
        .as_ref()
        .expect("gaussian should return observation interval lower");
    let obs_upper = pred
        .observation_upper
        .as_ref()
        .expect("gaussian should return observation interval upper");
    assert_eq!(obs_lower.len(), n);
    assert_eq!(obs_upper.len(), n);
    assert!(
        obs_lower
            .iter()
            .zip(obs_upper.iter())
            .all(|(&l, &u)| l.is_finite() && u.is_finite() && l <= u)
    );
}
