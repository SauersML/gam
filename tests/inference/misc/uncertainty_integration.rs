use gam::estimate::FittedLinkState;
use gam::estimate::FitOptions;
use gam::mixture_link::state_fromspec;
use gam_predict::{InferenceCovarianceMode, MeanIntervalMethod, PredictUncertaintyOptions, predict_gamwith_uncertainty};

use gam::smooth::BlockwisePenalty;
use gam::types::{
    InverseLink, LikelihoodSpec, LinkComponent, MixtureLinkSpec, ResponseFamily, StandardLink,
};
use ndarray::{Array1, Array2};

fn dense_penalty(local: Array2<f64>) -> BlockwisePenalty {
    let p = local.ncols();
    BlockwisePenalty::new(0..p, local)
}

fn gaussian_identity_likelihood() -> LikelihoodSpec {
    LikelihoodSpec::new(
        ResponseFamily::Gaussian,
        InverseLink::Standard(StandardLink::Identity),
    )
}

fn binomial_likelihood(link: StandardLink) -> LikelihoodSpec {
    LikelihoodSpec::new(ResponseFamily::Binomial, InverseLink::Standard(link))
}

fn mixture_likelihood(spec: &MixtureLinkSpec) -> LikelihoodSpec {
    LikelihoodSpec::new(
        ResponseFamily::Binomial,
        InverseLink::Mixture(state_fromspec(spec).expect("mixture state")),
    )
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
        &[dense_penalty(s.clone())],
        gaussian_identity_likelihood(),
        &FitOptions {
            resource_policy: gam_runtime::resource::ResourcePolicy::default_library(),
            latent_cloglog: None,
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            compute_inference: true,
            skip_rho_posterior_inference: false,
            max_iter: 40,
            tol: 1e-6,
            nullspace_dims: vec![1],
            adaptive_regularization: None,
            firth_bias_reduction: false,
            linear_constraints: None,
            rho_prior: Default::default(),
            kronecker_penalty_system: None,
            kronecker_factored: None,
            persistent_warm_start_store: None,
        },
    )
    .expect("fit should succeed");

    let cov = fit
        .beta_covariance()
        .expect("conditional covariance should be available");
    assert_eq!(cov.nrows(), fit.beta.len());
    assert_eq!(cov.ncols(), fit.beta.len());
    assert!(cov.iter().all(|v: &f64| v.is_finite()));
    let p = fit.beta.len();
    let dispersion = fit.dispersion().expect("dispersion should be stored");
    assert!(dispersion.is_estimated());
    let phi = dispersion.phi();
    assert!(phi.is_finite() && phi > 0.0);

    let fmat = fit
        .coefficient_influence()
        .expect("coefficient-space influence matrix should be stored");
    let trace_f: f64 = (0..fmat.nrows()).map(|i| fmat[[i, i]]).sum();
    assert!(
        (trace_f - fit.edf_total().expect("edf total")).abs() < 1e-6,
        "tr(F) = {trace_f} should equal edf_total = {:?}",
        fit.edf_total()
    );

    let mut xtwx = Array2::<f64>::zeros((p, p));
    let ww = fit.working_weights().expect("working weights");
    for i in 0..n {
        for a in 0..p {
            for b in 0..p {
                xtwx[[a, b]] += ww[i] * x[[i, a]] * x[[i, b]];
            }
        }
    }
    let mut h_inv = cov.clone();
    h_inv.mapv_inplace(|v| v / phi);
    let mut ve_expected = h_inv.dot(&xtwx).dot(&h_inv);
    ve_expected.mapv_inplace(|v| v * phi);
    let ve = fit
        .beta_covariance_ve()
        .expect("frequentist covariance should be stored when full covariance is available");
    let max_ve_diff = ve
        .iter()
        .zip(ve_expected.iter())
        .map(|(a, b): (&f64, &f64)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_ve_diff < 1e-8,
        "Ve should match H^-1 X'WX H^-1 * phi; max diff {max_ve_diff}"
    );

    // --- Posterior covariance must be symmetric within fp tolerance ---
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
        // Loose mixed absolute/relative tolerance: implementations may
        // store SE separately rather than recomputing sqrt(diag), so allow
        // ~1e-7 relative drift on top of fp roundoff at the SE magnitude.
        assert!(
            diff < 1e-9 + 1e-7 * from_cov.max(se[i]).max(1.0),
            "se[{i}] inconsistent with sqrt(cov[{i},{i}]): se={se_i}, sqrt(diag)={from_cov}, diff={diff}",
            se_i = se[i]
        );
    }

    // --- Covariance contract: Vb/Vp/Ve/F are explicitly exposed and scaled. ---
    let dispersion = fit.dispersion().expect("dispersion should be recorded");
    assert!(
        dispersion.is_estimated(),
        "Gaussian scale should be estimated"
    );
    assert!(dispersion.phi().is_finite() && dispersion.phi() > 0.0);

    let hessian = fit
        .penalized_hessian()
        .expect("penalized Hessian should be available");
    let ve = fit
        .beta_covariance_ve()
        .expect("frequentist covariance Ve should be available");
    let fmat = fit
        .coefficient_influence()
        .expect("coefficient influence matrix F should be available");
    assert_eq!(ve.dim(), cov.dim());
    assert_eq!(fmat.dim(), cov.dim());

    let mut weighted_penalty = s.clone();
    weighted_penalty.mapv_inplace(|v| v * fit.lambdas[0]);
    let mut xwx = hessian.clone();
    xwx -= &weighted_penalty;
    let h_inv_unscaled = cov.mapv(|v| v / dispersion.phi());
    let ve_expected = h_inv_unscaled.dot(&xwx).dot(&h_inv_unscaled) * dispersion.phi();
    let f_expected = h_inv_unscaled.dot(&xwx);
    for i in 0..p {
        for j in 0..p {
            assert!(
                (ve[[i, j]] - ve_expected[[i, j]]).abs() < 1e-7,
                "Ve mismatch at ({i},{j}): got {}, expected {}",
                ve[[i, j]],
                ve_expected[[i, j]]
            );
            assert!(
                (fmat[[i, j]] - f_expected[[i, j]]).abs() < 1e-7,
                "F mismatch at ({i},{j}): got {}, expected {}",
                fmat[[i, j]],
                f_expected[[i, j]]
            );
        }
    }
    let trace_f = fmat.diag().iter().copied().sum::<f64>();
    let edf_total = fit.edf_total().expect("edf_total should be available");
    assert!(
        (trace_f - edf_total).abs() < 1e-7,
        "tr(F) should equal edf_total: trace={trace_f}, edf_total={edf_total}"
    );

    let coef_ci = coefficient_uncertainty(&fit, 0.95, InferenceCovarianceMode::SmoothingCorrected)
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
    let coef_ci_80 =
        coefficient_uncertainty(&fit, 0.80, InferenceCovarianceMode::SmoothingCorrected)
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
fn noiseless_gaussian_smoothing_correction_is_a_valid_covariance_2490() {
    let n = 160usize;
    let mut x = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        let t = (i as f64 + 0.5) / n as f64;
        let x1 = -3.0 + 6.0 * t;
        x[[i, 0]] = 1.0;
        x[[i, 1]] = x1;
        x[[i, 2]] = (1.7 * x1).sin();
    }

    let truth = Array1::from_vec(vec![0.8, -1.1, 0.55]);
    let y = x.dot(&truth);
    let weights = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);

    let mut penalty = Array2::<f64>::zeros((3, 3));
    penalty[[2, 2]] = 1.0;

    let fit = fit_gam(
        x.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &[dense_penalty(penalty)],
        gaussian_identity_likelihood(),
        &FitOptions {
            skip_rho_posterior_inference: true,
            max_iter: 80,
            tol: 1e-8,
            nullspace_dims: vec![2],
            ..FitOptions::default()
        },
    )
    .expect("a noiseless penalized Gaussian fit must produce valid inference");

    let phi = fit
        .dispersion_phi()
        .expect("profiled Gaussian dispersion must be available");
    assert!(
        phi.is_finite() && phi > 0.0,
        "the penalized fit must retain its finite positive profiled dispersion, got {phi}"
    );

    let p = fit.beta.len();
    let conditional = fit
        .beta_covariance()
        .expect("fit must retain conditional covariance");
    let conditional_se = fit
        .beta_standard_errors()
        .expect("fit must retain conditional standard errors");
    assert_eq!(conditional.dim(), (p, p));
    assert_eq!(conditional_se.len(), p);
    assert!(conditional.iter().all(|value| value.is_finite()));
    for index in 0..p {
        let variance = conditional[[index, index]];
        assert!(
            variance >= 0.0,
            "conditional covariance diagonal {index} is negative: {variance}"
        );
        assert_eq!(
            conditional_se[index],
            variance.sqrt(),
            "conditional SE {index} must be derived from its reported covariance"
        );
    }

    let first_order = fit
        .smoothing_correction_first_order()
        .expect("fixture must exercise the first-order smoothing correction");
    assert_eq!(first_order.dim(), (p, p));
    assert!(first_order.iter().all(|value| value.is_finite()));
    assert!(
        first_order.diag().iter().all(|&variance| variance >= 0.0),
        "a Gram covariance must have non-negative diagonal: {first_order:?}"
    );
    assert!(
        first_order.diag().iter().any(|&variance| variance > 0.0),
        "fixture no longer exercises a non-vacuous smoothing correction"
    );

    let corrected = fit
        .beta_covariance_corrected()
        .expect("fit must retain corrected covariance");
    let corrected_se = fit
        .beta_standard_errors_corrected()
        .expect("fit must retain corrected standard errors");
    assert_eq!(corrected.dim(), (p, p));
    assert_eq!(corrected_se.len(), p);
    assert!(corrected.iter().all(|value| value.is_finite()));

    for index in 0..p {
        let variance = corrected[[index, index]];
        assert!(
            variance >= 0.0,
            "corrected covariance diagonal {index} is negative: {variance}"
        );
        assert_eq!(
            corrected_se[index],
            variance.sqrt(),
            "corrected SE {index} must be derived from its reported covariance"
        );
    }
}

/// Binomial trials per row for the uncertainty fixtures below.
///
/// These tests are about the SHAPE of the reported uncertainty, not about the
/// counts, so any trial count works — but it must exist. `weights = 1` with a
/// probability response is not a binomial observation at all, and the PIRLS
/// row-geometry contract says so rather than silently fitting it.
const TRIALS: f64 = 20.0;

#[test]
fn prediction_uncertainty_is_finite_andwell_shaped() {
    let n = 80usize;
    let mut x = Array2::<f64>::zeros((n, 2));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let t = -2.0 + 4.0 * (i as f64) / (n as f64 - 1.0);
        x[[i, 0]] = 1.0;
        x[[i, 1]] = t;
        // A binomial observation is (trials, successes), not a probability. With
        // unit weights `successes = weight * y` is the bare probability and is not
        // an integer, which `full_log_likelihood_row` correctly refuses ("fully-
        // normalized binomial trials/successes (exact integers required)"). Give
        // the row real trials and round the success count onto the grid they
        // define, so the signal is unchanged and the data is representable.
        let p = 1.0 / (1.0 + (-(-0.2 + 0.9 * t)).exp());
        y[i] = (p * TRIALS).round() / TRIALS;
    }

    let weights = Array1::from_elem(n, TRIALS);
    let offset = Array1::zeros(n);
    let mut s = Array2::<f64>::zeros((2, 2));
    s[[1, 1]] = 1.0;

    let fit = fit_gam(
        x.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &[dense_penalty(s)],
        binomial_likelihood(StandardLink::Logit),
        &FitOptions {
            resource_policy: gam_runtime::resource::ResourcePolicy::default_library(),
            latent_cloglog: None,
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            compute_inference: true,
            skip_rho_posterior_inference: false,
            max_iter: 50,
            tol: 1e-6,
            nullspace_dims: vec![1],
            adaptive_regularization: None,
            firth_bias_reduction: false,
            linear_constraints: None,
            rho_prior: Default::default(),
            kronecker_penalty_system: None,
            kronecker_factored: None,
            persistent_warm_start_store: None,
        },
    )
    .expect("fit should succeed");

    let pred = predict_gamwith_uncertainty(
        x.view(),
        fit.beta.view(),
        offset.view(),
        binomial_likelihood(StandardLink::Logit),
        &fit,
        &PredictUncertaintyOptions {
            confidence_level: 0.95,
            covariance_mode: InferenceCovarianceMode::SmoothingCorrected,
            mean_interval_method: MeanIntervalMethod::TransformEta,
            includeobservation_interval: true,
            apply_bias_correction: false,
            ..PredictUncertaintyOptions::default()
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
    // A binomial fit with `includeobservation_interval: true` DOES emit a
    // response-scale observation (prediction) band: the observation-interval
    // feature's symmetric-driver binomial arm (#811/#812) folds the conditional
    // Bernoulli variance p(1−p) into the mean SE and clamps to the [0,1] response
    // support. The band must therefore be present and well-shaped — finite,
    // ordered, and inside the probability support on every row (the earlier
    // `is_none()` expectation predated the observation-interval feature and was
    // masked by vacuous CI).
    let obs_lower = pred
        .observation_lower
        .as_ref()
        .expect("binomial observation band must be emitted when requested");
    let obs_upper = pred
        .observation_upper
        .as_ref()
        .expect("binomial observation band must be emitted when requested");
    assert_eq!(obs_lower.len(), n);
    assert_eq!(obs_upper.len(), n);
    assert!(
        obs_lower
            .iter()
            .zip(obs_upper.iter())
            .all(|(&l, &u): (&f64, &f64)| l.is_finite()
                && u.is_finite()
                && l <= u
                && l >= 0.0
                && u <= 1.0)
    );
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
        gaussian_identity_likelihood(),
        &FitOptions {
            resource_policy: gam_runtime::resource::ResourcePolicy::default_library(),
            latent_cloglog: None,
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            compute_inference: true,
            skip_rho_posterior_inference: false,
            max_iter: 40,
            tol: 1e-6,
            nullspace_dims: vec![1],
            adaptive_regularization: None,
            firth_bias_reduction: false,
            linear_constraints: None,
            rho_prior: Default::default(),
            kronecker_penalty_system: None,
            kronecker_factored: None,
            persistent_warm_start_store: None,
        },
    )
    .expect("fit should succeed");

    let pred = predict_gamwith_uncertainty(
        x.view(),
        fit.beta.view(),
        offset.view(),
        gaussian_identity_likelihood(),
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
        binomial_likelihood(StandardLink::Logit),
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

// `stateless_sas_inverse_link_is_rejected` was deleted: the type system now
// rejects `InverseLink::Standard(LinkFunction::Sas)` at compile time
// (`InverseLink::Standard` carries `StandardLink`, which has no `Sas`
// variant), so the runtime check it exercised is unreachable by
// construction.

#[test]
fn mixture_uncertainty_intervals_are_clamped_to_unit_interval() {
    let n = 100usize;
    let mut x = Array2::<f64>::zeros((n, 2));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let t = -2.0 + 4.0 * (i as f64) / (n as f64 - 1.0);
        x[[i, 0]] = 1.0;
        x[[i, 1]] = t;
        // A binomial observation is (trials, successes), not a probability. With
        // unit weights `successes = weight * y` is the bare probability and is not
        // an integer, which `full_log_likelihood_row` correctly refuses ("fully-
        // normalized binomial trials/successes (exact integers required)"). Give
        // the row real trials and round the success count onto the grid they
        // define, so the signal is unchanged and the data is representable.
        let p = 1.0 / (1.0 + (-(-0.2 + 0.9 * t)).exp());
        y[i] = (p * TRIALS).round() / TRIALS;
    }

    let weights = Array1::from_elem(n, TRIALS);
    let offset = Array1::zeros(n);
    let mut s = Array2::<f64>::zeros((2, 2));
    s[[1, 1]] = 1.0;

    let fit_base = fit_gam(
        x.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &[dense_penalty(s)],
        binomial_likelihood(StandardLink::Logit),
        &FitOptions {
            resource_policy: gam_runtime::resource::ResourcePolicy::default_library(),
            latent_cloglog: None,
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            compute_inference: true,
            skip_rho_posterior_inference: false,
            max_iter: 80,
            tol: 1e-6,
            nullspace_dims: vec![1],
            adaptive_regularization: None,
            firth_bias_reduction: false,
            linear_constraints: None,
            rho_prior: Default::default(),
            kronecker_penalty_system: None,
            kronecker_factored: None,
            persistent_warm_start_store: None,
        },
    )
    .expect("base fit should succeed");

    let mut fit = fit_base.clone();
    let mixture_spec = MixtureLinkSpec {
        components: vec![
            LinkComponent::Probit,
            LinkComponent::Logit,
            LinkComponent::CLogLog,
        ],
        initial_rho: Array1::from_vec(vec![0.4, -0.2]),
    };
    let state = state_fromspec(&mixture_spec).expect("valid synthetic mixture state");
    let likelihood = mixture_likelihood(&mixture_spec);
    fit.fitted_link = FittedLinkState::Mixture {
        state: state.clone(),
        covariance: None,
    };

    let pred = predict_gamwith_uncertainty(
        x.view(),
        fit.beta.view(),
        offset.view(),
        likelihood,
        &fit,
        &PredictUncertaintyOptions {
            confidence_level: 0.95,
            covariance_mode: InferenceCovarianceMode::SmoothingCorrected,
            mean_interval_method: MeanIntervalMethod::Delta,
            includeobservation_interval: false,
            apply_bias_correction: false,
            ..PredictUncertaintyOptions::default()
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
