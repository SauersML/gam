use gam::estimate::FittedLinkState;
use gam::estimate::{FitOptions, fit_gam};
use gam::inference::predict::{
    InferenceCovarianceMode, MeanIntervalMethod, PredictUncertaintyOptions,
    predict_gamwith_uncertainty,
};
use gam::mixture_link::{mixture_inverse_link_jet, sas_inverse_link_jet, state_fromspec};
use gam::smooth::BlockwisePenalty;
use gam::types::{LikelihoodFamily, LinkComponent, MixtureLinkSpec, SasLinkSpec};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

fn build_design(n: usize) -> Array2<f64> {
    let mut x = Array2::<f64>::zeros((n, 4));
    for i in 0..n {
        let t = (i as f64 + 0.5) / (n as f64);
        let x1 = -2.5 + 5.0 * t;
        x[[i, 0]] = 1.0;
        x[[i, 1]] = x1;
        x[[i, 2]] = (1.3 * x1).sin();
        x[[i, 3]] = 0.5 * x1 * x1 - 0.7;
    }
    x
}

fn one_penalty_for_non_intercept(p: usize) -> Vec<BlockwisePenalty> {
    let mut s = Array2::<f64>::zeros((p, p));
    for j in 1..p {
        s[[j, j]] = 1.0;
    }
    vec![BlockwisePenalty::new(0..p, s)]
}

fn base_fit_options() -> FitOptions {
    FitOptions {
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
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
    }
}

fn brier_score(p: &Array1<f64>, y: &Array1<f64>) -> f64 {
    (p - y).mapv(|v| v * v).mean().unwrap_or(f64::INFINITY)
}

fn coverage(true_p: &Array1<f64>, lo: &Array1<f64>, hi: &Array1<f64>) -> f64 {
    let mut hit = 0usize;
    for i in 0..true_p.len() {
        if lo[i] <= true_p[i] && true_p[i] <= hi[i] {
            hit += 1;
        }
    }
    hit as f64 / true_p.len() as f64
}

#[test]
fn sas_fit_recovery_and_calibration_system() {
    let n = 3000usize;
    let x = build_design(n);
    let beta_true = Array1::from_vec(vec![-0.35, 1.15, -0.65, 0.45]);
    let eps_true: f64 = 0.38;
    let log_delta_true: f64 = -0.30;
    let delta_true = log_delta_true.exp();
    let eta = x.dot(&beta_true);
    let p_true = eta.mapv(|e| sas_inverse_link_jet(e, eps_true, log_delta_true).mu);

    let mut rng = StdRng::seed_from_u64(9001);
    let y = p_true.mapv(|p| if rng.random::<f64>() < p { 1.0 } else { 0.0 });
    let w = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    let s_list = one_penalty_for_non_intercept(x.ncols());

    let mut opts = base_fit_options();
    opts.sas_link = Some(SasLinkSpec {
        initial_epsilon: 0.0,
        initial_log_delta: 0.0,
    });
    opts.optimize_sas = true;
    let fit = fit_gam(
        x.view(),
        y.view(),
        w.view(),
        offset.view(),
        &s_list,
        LikelihoodFamily::BinomialSas,
        &opts,
    )
    .expect("SAS fit");

    let (eps_hat, delta_hat) = match &fit.fitted_link {
        FittedLinkState::Sas { state, .. } => (state.epsilon, state.delta),
        other => panic!("expected SAS fitted state, got {other:?}"),
    };
    assert!(
        (eps_hat - eps_true).abs() < 0.20,
        "epsilon recovery off: hat={eps_hat:.4}, true={eps_true:.4}"
    );
    assert!(
        (delta_hat - delta_true).abs() < 0.20,
        "delta recovery off: hat={delta_hat:.4}, true={delta_true:.4}"
    );

    let pred = predict_gamwith_uncertainty(
        x.view(),
        fit.beta.view(),
        offset.view(),
        LikelihoodFamily::BinomialSas,
        &fit,
        &PredictUncertaintyOptions::default(),
    )
    .expect("SAS predict");
    let brier = brier_score(&pred.mean, &y);
    assert!(brier < 0.20, "SAS Brier too high: {brier:.4}");
    let calib_gap = (pred.mean.mean().unwrap_or(0.0) - y.mean().unwrap_or(0.0)).abs();
    assert!(
        calib_gap < 0.04,
        "SAS prevalence calibration gap too large: {calib_gap:.4}"
    );
}

#[test]
fn mixture_recovery_and_prediction_alignment_system() {
    let n = 2000usize;
    let x = build_design(n);
    let beta_true = Array1::from_vec(vec![-0.20, 1.0, -0.5, 0.3]);
    let mixspec_true = MixtureLinkSpec {
        components: vec![
            LinkComponent::Probit,
            LinkComponent::CLogLog,
            LinkComponent::Cauchit,
        ],
        initial_rho: Array1::from_vec(vec![1.0, -0.6]),
    };
    let mix_state_true = state_fromspec(&mixspec_true).expect("true mixture state");
    let eta = x.dot(&beta_true);
    let p_true = eta.mapv(|e| mixture_inverse_link_jet(&mix_state_true, e).mu);
    let mut rng = StdRng::seed_from_u64(12345);
    let y = p_true.mapv(|p| if rng.random::<f64>() < p { 1.0 } else { 0.0 });
    let w = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    let s_list = one_penalty_for_non_intercept(x.ncols());

    let mut opts = base_fit_options();
    opts.mixture_link = Some(MixtureLinkSpec {
        components: mixspec_true.components.clone(),
        initial_rho: Array1::zeros(2),
    });
    opts.optimize_mixture = true;
    let fit = fit_gam(
        x.view(),
        y.view(),
        w.view(),
        offset.view(),
        &s_list,
        LikelihoodFamily::BinomialMixture,
        &opts,
    )
    .expect("mixture fit");

    let pi_hat = match &fit.fitted_link {
        FittedLinkState::Mixture { state, .. } => state.pi.clone(),
        other => panic!("expected Mixture fitted state, got {other:?}"),
    };
    let simplex_sum = pi_hat.sum();
    assert!(
        (simplex_sum - 1.0).abs() < 1e-10 && pi_hat.iter().all(|&w| (0.0..=1.0).contains(&w)),
        "fitted mixture weights must be a valid simplex, got pi={pi_hat:?}"
    );

    let pred = predict_gamwith_uncertainty(
        x.view(),
        fit.beta.view(),
        offset.view(),
        LikelihoodFamily::BinomialMixture,
        &fit,
        &PredictUncertaintyOptions::default(),
    )
    .expect("mixture predict");
    let pred_rmsevs_truth = (&pred.mean - &p_true)
        .mapv(|v| v * v)
        .mean()
        .unwrap_or(f64::INFINITY)
        .sqrt();
    assert!(
        pred_rmsevs_truth < 0.11,
        "mixture truth-prob RMSE too high: {pred_rmsevs_truth:.4}"
    );
}

#[test]
fn posterior_mean_coverage_includes_sas_and_mixture() {
    let n_train = 1500usize;
    let n_test = 1000usize;
    let x_train = build_design(n_train);
    let x_test = build_design(n_test);
    let beta_true = Array1::from_vec(vec![-0.25, 0.95, -0.7, 0.35]);
    let eta_train = x_train.dot(&beta_true);
    let eta_test = x_test.dot(&beta_true);
    let p_train = eta_train.mapv(|e| 1.0 / (1.0 + (-e).exp()));
    let p_test = eta_test.mapv(|e| 1.0 / (1.0 + (-e).exp()));

    let mut rng = StdRng::seed_from_u64(4242);
    let y_train = p_train.mapv(|p| if rng.random::<f64>() < p { 1.0 } else { 0.0 });
    let w = Array1::<f64>::ones(n_train);
    let offset_train = Array1::<f64>::zeros(n_train);
    let offset_test = Array1::<f64>::zeros(n_test);
    let s_list = one_penalty_for_non_intercept(x_train.ncols());

    let opts_logit = base_fit_options();
    let fit_logit = fit_gam(
        x_train.view(),
        y_train.view(),
        w.view(),
        offset_train.view(),
        &s_list,
        LikelihoodFamily::BinomialLogit,
        &opts_logit,
    )
    .expect("logit fit");

    let opts_probit = base_fit_options();
    let fit_probit = fit_gam(
        x_train.view(),
        y_train.view(),
        w.view(),
        offset_train.view(),
        &s_list,
        LikelihoodFamily::BinomialProbit,
        &opts_probit,
    )
    .expect("probit fit");

    let mut opts_sas = base_fit_options();
    opts_sas.sas_link = Some(SasLinkSpec {
        initial_epsilon: 0.0,
        initial_log_delta: 0.0,
    });
    opts_sas.optimize_sas = true;
    opts_sas.max_iter = 120;
    opts_sas.tol = 1e-8;
    let fit_sas = fit_gam(
        x_train.view(),
        y_train.view(),
        w.view(),
        offset_train.view(),
        &s_list,
        LikelihoodFamily::BinomialSas,
        &opts_sas,
    )
    .expect("sas fit");

    let mut opts_mix = base_fit_options();
    opts_mix.mixture_link = Some(MixtureLinkSpec {
        components: vec![
            LinkComponent::Probit,
            LinkComponent::Logit,
            LinkComponent::CLogLog,
        ],
        initial_rho: Array1::zeros(2),
    });
    opts_mix.optimize_mixture = true;
    let fit_mix = fit_gam(
        x_train.view(),
        y_train.view(),
        w.view(),
        offset_train.view(),
        &s_list,
        LikelihoodFamily::BinomialMixture,
        &opts_mix,
    )
    .expect("mixture fit");

    let options = PredictUncertaintyOptions {
        confidence_level: 0.90,
        covariance_mode: InferenceCovarianceMode::ConditionalPlusSmoothingPreferred,
        mean_interval_method: MeanIntervalMethod::TransformEta,
        includeobservation_interval: false,
    };

    let pred_logit = predict_gamwith_uncertainty(
        x_test.view(),
        fit_logit.beta.view(),
        offset_test.view(),
        LikelihoodFamily::BinomialLogit,
        &fit_logit,
        &options,
    )
    .expect("logit pred");
    let pred_probit = predict_gamwith_uncertainty(
        x_test.view(),
        fit_probit.beta.view(),
        offset_test.view(),
        LikelihoodFamily::BinomialProbit,
        &fit_probit,
        &options,
    )
    .expect("probit pred");
    let pred_sas = predict_gamwith_uncertainty(
        x_test.view(),
        fit_sas.beta.view(),
        offset_test.view(),
        LikelihoodFamily::BinomialSas,
        &fit_sas,
        &options,
    )
    .expect("sas pred");
    let pred_mix = predict_gamwith_uncertainty(
        x_test.view(),
        fit_mix.beta.view(),
        offset_test.view(),
        LikelihoodFamily::BinomialMixture,
        &fit_mix,
        &options,
    )
    .expect("mix pred");

    let c_logit = coverage(&p_test, &pred_logit.mean_lower, &pred_logit.mean_upper);
    let c_probit = coverage(&p_test, &pred_probit.mean_lower, &pred_probit.mean_upper);
    let c_sas = coverage(&p_test, &pred_sas.mean_lower, &pred_sas.mean_upper);
    let c_mix = coverage(&p_test, &pred_mix.mean_lower, &pred_mix.mean_upper);

    let meanwidth_logit = (&pred_logit.mean_upper - &pred_logit.mean_lower)
        .mean()
        .unwrap_or(f64::INFINITY);
    let meanwidth_probit = (&pred_probit.mean_upper - &pred_probit.mean_lower)
        .mean()
        .unwrap_or(f64::INFINITY);
    let meanwidth_sas = (&pred_sas.mean_upper - &pred_sas.mean_lower)
        .mean()
        .unwrap_or(f64::INFINITY);
    let meanwidth_mix = (&pred_mix.mean_upper - &pred_mix.mean_lower)
        .mean()
        .unwrap_or(f64::INFINITY);

    for (name, c, w) in [
        ("logit", c_logit, meanwidth_logit),
        ("probit", c_probit, meanwidth_probit),
    ] {
        assert!(c >= 0.80, "{name} 90% coverage too low: {c:.3}");
        assert!(
            w < 0.60,
            "{name} intervals too wide on average: mean width={w:.3}"
        );
    }
    for (name, c, w) in [
        ("sas", c_sas, meanwidth_sas),
        ("mixture", c_mix, meanwidth_mix),
    ] {
        assert!(
            c >= 0.65,
            "{name} 90% coverage too low: {c:.3} (extra parameter uncertainty expected)"
        );
        assert!(
            w < 0.60,
            "{name} intervals too wide on average: mean width={w:.3}"
        );
    }
}
