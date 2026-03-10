use gam::estimate::FittedLinkParameters;
use gam::mixture_link::state_fromspec;
use gam::probability::try_inverse_link_array;
use gam::types::LinkComponent;
use gam::{
    FitOptions, InferenceCovarianceMode, LikelihoodFamily, MeanIntervalMethod,
    PredictUncertaintyOptions, coefficient_uncertainty, fit_gam, predict_gam_posterior_mean,
    predict_gamwith_uncertainty,
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
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            max_iter: 40,
            tol: 1e-6,
            nullspace_dims: vec![1],
            adaptive_regularization: None,
            linear_constraints: None,
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

    let coef_ci = coefficient_uncertainty(
        &fit,
        0.95,
        InferenceCovarianceMode::ConditionalPlusSmoothingPreferred,
    )
    .expect("coefficient CI should work");
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
fn prediction_uncertainty_is_finite_andwell_shaped() {
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
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            max_iter: 50,
            tol: 1e-6,
            nullspace_dims: vec![1],
            adaptive_regularization: None,
            linear_constraints: None,
        },
    )
    .expect("fit should succeed");

    let pred = predict_gamwith_uncertainty(
        x.view(),
        fit.beta.view(),
        offset.view(),
        LikelihoodFamily::BinomialLogit,
        &fit,
        &PredictUncertaintyOptions {
            confidence_level: 0.95,
            covariance_mode: InferenceCovarianceMode::ConditionalPlusSmoothingPreferred,
            mean_interval_method: MeanIntervalMethod::TransformEta,
            includeobservation_interval: true,
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
fn gaussian_prediction_intervals_includeobservation_noise() {
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
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            max_iter: 40,
            tol: 1e-6,
            nullspace_dims: vec![1],
            adaptive_regularization: None,
            linear_constraints: None,
        },
    )
    .expect("fit should succeed");

    let pred = predict_gamwith_uncertainty(
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

#[test]
fn posterior_mean_prediction_shrinks_extreme_logit_probabilities() {
    let n = 120usize;
    let mut x = Array2::<f64>::zeros((n, 2));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let t = -3.0 + 6.0 * (i as f64) / (n as f64 - 1.0);
        x[[i, 0]] = 1.0;
        x[[i, 1]] = t;
        y[i] = if t > 0.5 { 1.0 } else { 0.0 };
    }
    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);
    let mut s = Array2::<f64>::zeros((2, 2));
    s[[1, 1]] = 1e-2;

    let fit = fit_gam(
        x.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &[s],
        LikelihoodFamily::BinomialLogit,
        &FitOptions {
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            max_iter: 60,
            tol: 1e-6,
            nullspace_dims: vec![1],
            adaptive_regularization: None,
            linear_constraints: None,
        },
    )
    .expect("fit should succeed");
    let cov = fit
        .beta_covariance
        .as_ref()
        .expect("covariance should be available");
    let pred = predict_gam_posterior_mean(
        x.view(),
        fit.beta.view(),
        offset.view(),
        LikelihoodFamily::BinomialLogit,
        cov.view(),
    )
    .expect("posterior mean prediction should succeed");

    let eta_hi = pred.eta[n - 1];
    let map_hi = 1.0 / (1.0 + (-eta_hi).exp());
    let pm_hi = pred.mean[n - 1];
    assert!(pm_hi < map_hi);

    let eta_lo = pred.eta[0];
    let map_lo = 1.0 / (1.0 + (-eta_lo).exp());
    let pm_lo = pred.mean[0];
    assert!(pm_lo > map_lo);
}

#[test]
fn stateful_inverse_link_requires_state_for_sas_and_mixture() {
    let eta = Array1::from_vec(vec![-0.7, 0.0, 1.2]);
    let sas_err = try_inverse_link_array(LikelihoodFamily::BinomialSas, eta.view(), None)
        .expect_err("SAS inverse-link should require explicit sas_params");
    assert!(sas_err.to_string().contains("requires SAS link state"));

    let mix_err = try_inverse_link_array(LikelihoodFamily::BinomialMixture, eta.view(), None)
        .expect_err("Mixture inverse-link should require explicit mixture_state");
    assert!(mix_err.to_string().contains("requires mixture link state"));
}

#[test]
fn mixture_uncertainty_intervals_are_clamped_to_unit_interval() {
    let n = 100usize;
    let mut x = Array2::<f64>::zeros((n, 2));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let t = -3.0 + 6.0 * (i as f64) / (n as f64 - 1.0);
        x[[i, 0]] = 1.0;
        x[[i, 1]] = t;
        let p = 1.0 / (1.0 + (0.3 - 0.9 * t).exp());
        y[i] = p.clamp(0.02, 0.98);
    }

    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);
    let mut s = Array2::<f64>::zeros((2, 2));
    s[[1, 1]] = 1e-2;

    let fit_base = fit_gam(
        x.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &[s],
        LikelihoodFamily::BinomialLogit,
        &FitOptions {
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            max_iter: 80,
            tol: 1e-6,
            nullspace_dims: vec![1],
            adaptive_regularization: None,
            linear_constraints: None,
        },
    )
    .expect("base fit should succeed");

    let mut fit = fit_base.clone();
    let state = state_fromspec(&gam::types::MixtureLinkSpec {
        components: vec![
            LinkComponent::Probit,
            LinkComponent::Logit,
            LinkComponent::CLogLog,
        ],
        initial_rho: Array1::from_vec(vec![0.4, -0.2]),
    })
    .expect("valid synthetic mixture state");
    fit.fitted_link_parameters = FittedLinkParameters::Mixture {
        state,
        covariance: None,
    };

    let pred = predict_gamwith_uncertainty(
        x.view(),
        fit.beta.view(),
        offset.view(),
        LikelihoodFamily::BinomialMixture,
        &fit,
        &PredictUncertaintyOptions {
            confidence_level: 0.95,
            covariance_mode: InferenceCovarianceMode::ConditionalPlusSmoothingPreferred,
            mean_interval_method: MeanIntervalMethod::Delta,
            includeobservation_interval: false,
        },
    )
    .expect("mixture uncertainty prediction should succeed");

    assert!(
        pred.mean_lower
            .iter()
            .zip(pred.mean_upper.iter())
            .all(|(&l, &u)| l.is_finite() && u.is_finite() && l <= u && l >= 0.0 && u <= 1.0)
    );
}
