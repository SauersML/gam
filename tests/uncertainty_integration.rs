use gam::estimate::FittedLinkState;
use gam::estimate::{FitOptions, fit_gam};
use gam::mixture_link::state_fromspec;
use gam::predict::{
    InferenceCovarianceMode, MeanIntervalMethod, PredictUncertaintyOptions,
    coefficient_uncertainty, predict_gam_posterior_mean, predict_gamwith_uncertainty,
};
use gam::probability::try_inverse_link_array;
use gam::smooth::BlockwisePenalty;
use gam::types::LikelihoodFamily;
use gam::types::LinkComponent;
use ndarray::{Array1, Array2};

fn dense_penalty(local: Array2<f64>) -> BlockwisePenalty {
    let p = local.ncols();
    BlockwisePenalty::new(0..p, local)
}

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
        &[dense_penalty(s)],
        LikelihoodFamily::GaussianIdentity,
        &FitOptions {
            latent_cloglog: None,
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            compute_inference: true,
            max_iter: 40,
            tol: 1e-6,
            nullspace_dims: vec![1],
            adaptive_regularization: None,
            firth_bias_reduction: false,
            linear_constraints: None,
            penalty_shrinkage_floor: None,
            rho_prior: Default::default(),
            kronecker_penalty_system: None,
            kronecker_factored: None,
        },
    )
    .expect("fit should succeed");

    let cov = fit
        .beta_covariance()
        .expect("conditional covariance should be available");
    assert_eq!(cov.nrows(), fit.beta.len());
    assert_eq!(cov.ncols(), fit.beta.len());
    assert!(cov.iter().all(|v: &f64| v.is_finite()));

    // --- Posterior covariance must be symmetric within fp tolerance ---
    let p = fit.beta.len();
    let mut max_asym = 0.0_f64;
    let (mut asym_i, mut asym_j) = (0usize, 0usize);
    for i in 0..p {
        for j in (i + 1)..p {
            let d = (cov[[i, j]] - cov[[j, i]]).abs();
            if d > max_asym {
                max_asym = d;
                asym_i = i;
                asym_j = j;
            }
        }
    }
    assert!(
        max_asym < 1e-10,
        "posterior covariance not symmetric: max |C[i,j] - C[j,i]| = {max_asym} at \
         (i={asym_i}, j={asym_j})"
    );

    // --- Diagonal entries must be non-negative (variances) ---
    let mut min_diag = f64::INFINITY;
    let mut min_diag_i = 0usize;
    for i in 0..p {
        if cov[[i, i]] < min_diag {
            min_diag = cov[[i, i]];
            min_diag_i = i;
        }
    }
    assert!(
        min_diag >= -1e-12,
        "posterior covariance has negative diagonal entry at i={min_diag_i}: {min_diag}"
    );

    // --- Standard error consistency: se[i] == sqrt(cov[i,i]) ---
    let se = fit
        .beta_standard_errors()
        .expect("standard errors should be available");
    assert_eq!(se.len(), fit.beta.len());
    assert!(se.iter().all(|v: &f64| v.is_finite() && *v >= 0.0));
    for i in 0..p {
        let from_cov = cov[[i, i]].max(0.0).sqrt();
        let diff = (se[i] - from_cov).abs();
        assert!(
            diff < 1e-9 + 1e-9 * from_cov.max(se[i]),
            "se[{i}] inconsistent with sqrt(cov[{i},{i}]): se={se_i}, sqrt(diag)={from_cov}",
            se_i = se[i]
        );
    }

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
            .all(|(&l, &u): (&f64, &f64)| l.is_finite() && u.is_finite() && l <= u)
    );

    // --- Estimate must lie strictly inside CI for non-degenerate SEs ---
    for i in 0..coef_ci.estimate.len() {
        let est = coef_ci.estimate[i];
        let lo = coef_ci.lower[i];
        let hi = coef_ci.upper[i];
        assert!(
            lo - 1e-12 <= est && est <= hi + 1e-12,
            "coefficient {i} estimate {est} fell outside its 95% CI [{lo}, {hi}]"
        );
    }

    // --- 95% CI is wider than 80% CI for the same coefficient ---
    let coef_ci_80 = coefficient_uncertainty(
        &fit,
        0.80,
        InferenceCovarianceMode::ConditionalPlusSmoothingPreferred,
    )
    .expect("80% coefficient CI should also work");
    for i in 0..p {
        let width_95 = coef_ci.upper[i] - coef_ci.lower[i];
        let width_80 = coef_ci_80.upper[i] - coef_ci_80.lower[i];
        // Allow tiny equality slack for nullspace dimensions where SE may be
        // 0 and both intervals collapse to a point.
        assert!(
            width_95 + 1e-12 >= width_80,
            "95% CI width must not be narrower than 80% for coefficient {i}: \
             width_95={width_95}, width_80={width_80}"
        );
    }
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
        &[dense_penalty(s)],
        LikelihoodFamily::BinomialLogit,
        &FitOptions {
            latent_cloglog: None,
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            compute_inference: true,
            max_iter: 50,
            tol: 1e-6,
            nullspace_dims: vec![1],
            adaptive_regularization: None,
            firth_bias_reduction: false,
            linear_constraints: None,
            penalty_shrinkage_floor: None,
            rho_prior: Default::default(),
            kronecker_penalty_system: None,
            kronecker_factored: None,
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
            .all(|v: &f64| v.is_finite() && *v >= 0.0)
    );
    assert!(
        pred.mean_standard_error
            .iter()
            .all(|v: &f64| v.is_finite() && *v >= 0.0)
    );
    assert!(
        pred.mean_lower
            .iter()
            .zip(pred.mean_upper.iter())
            .all(|(&l, &u): (&f64, &f64)| l.is_finite()
                && u.is_finite()
                && l <= u
                && l >= 0.0
                && u <= 1.0)
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
        &[dense_penalty(s)],
        LikelihoodFamily::GaussianIdentity,
        &FitOptions {
            latent_cloglog: None,
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            compute_inference: true,
            max_iter: 40,
            tol: 1e-6,
            nullspace_dims: vec![1],
            adaptive_regularization: None,
            firth_bias_reduction: false,
            linear_constraints: None,
            penalty_shrinkage_floor: None,
            rho_prior: Default::default(),
            kronecker_penalty_system: None,
            kronecker_factored: None,
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
            .all(|(&l, &u): (&f64, &f64)| l.is_finite() && u.is_finite() && l <= u)
    );

    // --- Gaussian observation interval contracts ---
    // The observation interval includes the residual variance, so it must
    // be strictly wider than the mean interval for every prediction row.
    let mean_lower = &pred.mean_lower;
    let mean_upper = &pred.mean_upper;
    for i in 0..n {
        let mean_width = mean_upper[i] - mean_lower[i];
        let obs_width = obs_upper[i] - obs_lower[i];
        assert!(
            obs_width >= mean_width - 1e-12,
            "observation interval must be at least as wide as mean interval at row {i}: \
             mean_width={mean_width}, obs_width={obs_width}"
        );
        // The mean point estimate must sit inside the observation interval.
        assert!(
            obs_lower[i] - 1e-9 <= pred.mean[i] && pred.mean[i] <= obs_upper[i] + 1e-9,
            "row {i} mean {} not contained in observation interval [{}, {}]",
            pred.mean[i],
            obs_lower[i],
            obs_upper[i]
        );
    }
}

#[test]
fn posterior_mean_prediction_shrinks_extreme_logit_probabilities() {
    // Five rows spanning moderate (|eta| ~= 2) to extreme (|eta| ~= 6) logits
    // so we can check the shrinkage effect at multiple magnitudes — earlier
    // versions of the test only exercised the +-3 design point.
    let x = Array2::from_shape_vec(
        (5, 2),
        vec![1.0, -3.0, 1.0, -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 3.0],
    )
    .expect("design");
    let beta = Array1::from_vec(vec![0.0, 2.0]);
    let offset = Array1::zeros(5);
    let cov = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 0.0, 0.25]).expect("covariance");
    let pred = predict_gam_posterior_mean(
        x.view(),
        beta.view(),
        offset.view(),
        LikelihoodFamily::BinomialLogit,
        cov.view(),
    )
    .expect("posterior mean prediction should succeed");

    // Posterior-mean probabilities must be valid probabilities at every row.
    for (i, &p) in pred.mean.iter().enumerate() {
        assert!(
            (0.0..=1.0).contains(&p),
            "posterior-mean probability at row {i} fell outside [0,1]: {p}"
        );
    }

    // Shrinkage contract: PM(p) is closer to 0.5 than MAP(p) in the
    // saturating tails. Verify on every non-zero-eta row.
    for i in 0..pred.eta.len() {
        let eta = pred.eta[i];
        let map = 1.0 / (1.0 + (-eta).exp());
        let pm = pred.mean[i];
        if eta > 0.0 {
            assert!(
                pm < map + 1e-12,
                "row {i} (eta={eta}): posterior mean {pm} did not shrink below MAP {map}"
            );
        } else if eta < 0.0 {
            assert!(
                pm > map - 1e-12,
                "row {i} (eta={eta}): posterior mean {pm} did not shrink above MAP {map}"
            );
        } else {
            // eta = 0 -> MAP = 0.5; PM is also 0.5 by symmetry.
            assert!(
                (pm - 0.5).abs() < 1e-9,
                "eta=0 row {i} should give PM=0.5; got {pm}"
            );
        }
    }
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
        let t = -2.0 + 4.0 * (i as f64) / (n as f64 - 1.0);
        x[[i, 0]] = 1.0;
        x[[i, 1]] = t;
        y[i] = 1.0 / (1.0 + (-(-0.2 + 0.9 * t)).exp());
    }

    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);
    let mut s = Array2::<f64>::zeros((2, 2));
    s[[1, 1]] = 1.0;

    let fit_base = fit_gam(
        x.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &[dense_penalty(s)],
        LikelihoodFamily::BinomialLogit,
        &FitOptions {
            latent_cloglog: None,
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            compute_inference: true,
            max_iter: 80,
            tol: 1e-6,
            nullspace_dims: vec![1],
            adaptive_regularization: None,
            firth_bias_reduction: false,
            linear_constraints: None,
            penalty_shrinkage_floor: None,
            rho_prior: Default::default(),
            kronecker_penalty_system: None,
            kronecker_factored: None,
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
    fit.fitted_link = FittedLinkState::Mixture {
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
            .all(|(&l, &u): (&f64, &f64)| l.is_finite()
                && u.is_finite()
                && l <= u
                && l >= 0.0
                && u <= 1.0)
    );
}
