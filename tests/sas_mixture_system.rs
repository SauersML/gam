use gam::estimate::{FitOptions, fit_gam, fit_gam_with_heuristic_lambdas};
use gam::inference::predict::{
    InferenceCovarianceMode, MeanIntervalMethod, PredictUncertaintyOptions,
    predict_gam_with_uncertainty,
};
use gam::mixture_link::{mixture_inverse_link_jet, sas_inverse_link_jet, state_from_spec};
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

fn one_penalty_for_non_intercept(p: usize) -> Vec<Array2<f64>> {
    let mut s = Array2::<f64>::zeros((p, p));
    for j in 1..p {
        s[[j, j]] = 1.0;
    }
    vec![s]
}

fn base_fit_options() -> FitOptions {
    FitOptions {
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        max_iter: 60,
        tol: 1e-6,
        nullspace_dims: vec![1],
        linear_constraints: None,
        adaptive_regularization: None,
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
    let n = 1800usize;
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

    let eps_hat = fit.sas_epsilon.expect("fitted epsilon");
    let delta_hat = fit.sas_delta.expect("fitted delta");
    assert!(
        (eps_hat - eps_true).abs() < 0.45,
        "epsilon recovery off: hat={eps_hat:.4}, true={eps_true:.4}"
    );
    assert!(
        (delta_hat - delta_true).abs() < 0.35,
        "delta recovery off: hat={delta_hat:.4}, true={delta_true:.4}"
    );

    let pred = predict_gam_with_uncertainty(
        x.view(),
        fit.beta.view(),
        offset.view(),
        LikelihoodFamily::BinomialSas,
        &fit,
        &PredictUncertaintyOptions::default(),
    )
    .expect("SAS predict");
    let brier = brier_score(&pred.mean, &y);
    assert!(brier < 0.22, "SAS Brier too high: {brier:.4}");
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
    let mix_spec_true = MixtureLinkSpec {
        components: vec![
            LinkComponent::Probit,
            LinkComponent::CLogLog,
            LinkComponent::Cauchit,
        ],
        initial_rho: Array1::from_vec(vec![1.0, -0.6]),
    };
    let mix_state_true = state_from_spec(&mix_spec_true).expect("true mixture state");
    let eta = x.dot(&beta_true);
    let p_true = eta.mapv(|e| mixture_inverse_link_jet(&mix_state_true, e).mu);
    let mut rng = StdRng::seed_from_u64(12345);
    let y = p_true.mapv(|p| if rng.random::<f64>() < p { 1.0 } else { 0.0 });
    let w = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    let s_list = one_penalty_for_non_intercept(x.ncols());

    let mut opts = base_fit_options();
    opts.mixture_link = Some(MixtureLinkSpec {
        components: mix_spec_true.components.clone(),
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

    let pi_hat = fit
        .mixture_link_weights
        .as_ref()
        .expect("mixture weights")
        .to_owned();
    let pi_true = mix_state_true.pi;
    let l1 = (&pi_hat - &pi_true).mapv(|v| v.abs()).sum();
    assert!(l1 < 0.55, "mixture weight recovery too far: l1={l1:.4}");

    let pred = predict_gam_with_uncertainty(
        x.view(),
        fit.beta.view(),
        offset.view(),
        LikelihoodFamily::BinomialMixture,
        &fit,
        &PredictUncertaintyOptions::default(),
    )
    .expect("mixture predict");
    let pred_rmse_vs_truth = (&pred.mean - &p_true)
        .mapv(|v| v * v)
        .mean()
        .unwrap_or(f64::INFINITY)
        .sqrt();
    assert!(
        pred_rmse_vs_truth < 0.11,
        "mixture truth-prob RMSE too high: {pred_rmse_vs_truth:.4}"
    );
}

#[test]
fn posterior_mean_coverage_includes_sas_and_mixture() {
    let n_train = 900usize;
    let n_test = 700usize;
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
        include_observation_interval: false,
    };

    let pred_logit = predict_gam_with_uncertainty(
        x_test.view(),
        fit_logit.beta.view(),
        offset_test.view(),
        LikelihoodFamily::BinomialLogit,
        &fit_logit,
        &options,
    )
    .expect("logit pred");
    let pred_probit = predict_gam_with_uncertainty(
        x_test.view(),
        fit_probit.beta.view(),
        offset_test.view(),
        LikelihoodFamily::BinomialProbit,
        &fit_probit,
        &options,
    )
    .expect("probit pred");
    let pred_sas = predict_gam_with_uncertainty(
        x_test.view(),
        fit_sas.beta.view(),
        offset_test.view(),
        LikelihoodFamily::BinomialSas,
        &fit_sas,
        &options,
    )
    .expect("sas pred");
    let pred_mix = predict_gam_with_uncertainty(
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

    for (name, c) in [
        ("logit", c_logit),
        ("probit", c_probit),
        ("sas", c_sas),
        ("mixture", c_mix),
    ] {
        assert!(
            (0.75..=0.98).contains(&c),
            "{name} 90% coverage out of range: {c:.3}"
        );
    }
}

#[test]
fn outer_profile_objective_stationary_near_fitted_sas_and_mixture_params() {
    let n = 260usize;
    let x = build_design(n);
    let w = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    let s_list = one_penalty_for_non_intercept(x.ncols());

    let beta_true = Array1::from_vec(vec![-0.3, 0.9, -0.55, 0.4]);
    let eta = x.dot(&beta_true);
    let mut rng = StdRng::seed_from_u64(2026);

    // SAS profile stationarity
    let p_sas = eta.mapv(|e| sas_inverse_link_jet(e, 0.25, -0.20).mu);
    let y_sas = p_sas.mapv(|p| if rng.random::<f64>() < p { 1.0 } else { 0.0 });
    let mut opts_sas = base_fit_options();
    opts_sas.sas_link = Some(SasLinkSpec {
        initial_epsilon: 0.0,
        initial_log_delta: 0.0,
    });
    opts_sas.optimize_sas = true;
    let fit_sas = fit_gam(
        x.view(),
        y_sas.view(),
        w.view(),
        offset.view(),
        &s_list,
        LikelihoodFamily::BinomialSas,
        &opts_sas,
    )
    .expect("sas fit");
    let eps_hat = fit_sas.sas_epsilon.expect("eps");
    let ld_hat = fit_sas.sas_log_delta.expect("ld");
    let h = 3e-2;
    let profile_sas = |eps: f64, ld: f64| -> f64 {
        let mut o = base_fit_options();
        o.sas_link = Some(SasLinkSpec {
            initial_epsilon: eps,
            initial_log_delta: ld,
        });
        o.optimize_sas = false;
        fit_gam(
            x.view(),
            y_sas.view(),
            w.view(),
            offset.view(),
            &s_list,
            LikelihoodFamily::BinomialSas,
            &o,
        )
        .expect("profile sas")
        .reml_score
    };
    let fd_eps = (profile_sas(eps_hat + h, ld_hat) - profile_sas(eps_hat - h, ld_hat)) / (2.0 * h);
    let fd_ld = (profile_sas(eps_hat, ld_hat + h) - profile_sas(eps_hat, ld_hat - h)) / (2.0 * h);
    assert!(
        fd_eps.abs() < 0.25,
        "SAS profile FD gradient epsilon too large: {fd_eps:.4}"
    );
    assert!(
        fd_ld.abs() < 0.25,
        "SAS profile FD gradient log_delta too large: {fd_ld:.4}"
    );

    // Mixture profile stationarity
    let true_mix_spec = MixtureLinkSpec {
        components: vec![
            LinkComponent::Probit,
            LinkComponent::CLogLog,
            LinkComponent::Cauchit,
        ],
        initial_rho: Array1::from_vec(vec![0.7, -0.4]),
    };
    let true_mix_state = state_from_spec(&true_mix_spec).expect("true mix");
    let p_mix = eta.mapv(|e| mixture_inverse_link_jet(&true_mix_state, e).mu);
    let y_mix = p_mix.mapv(|p| if rng.random::<f64>() < p { 1.0 } else { 0.0 });
    let mut opts_mix = base_fit_options();
    opts_mix.mixture_link = Some(MixtureLinkSpec {
        components: true_mix_spec.components.clone(),
        initial_rho: Array1::zeros(2),
    });
    opts_mix.optimize_mixture = true;
    let fit_mix = fit_gam(
        x.view(),
        y_mix.view(),
        w.view(),
        offset.view(),
        &s_list,
        LikelihoodFamily::BinomialMixture,
        &opts_mix,
    )
    .expect("mix fit");
    let rho_hat = fit_mix.mixture_link_rho.clone().expect("rho");
    let h_rho = 3e-2;
    let profile_mix = |rho: &Array1<f64>| -> f64 {
        let mut o = base_fit_options();
        o.mixture_link = Some(MixtureLinkSpec {
            components: true_mix_spec.components.clone(),
            initial_rho: rho.clone(),
        });
        o.optimize_mixture = false;
        fit_gam_with_heuristic_lambdas(
            x.view(),
            y_mix.view(),
            w.view(),
            offset.view(),
            &s_list,
            Some(fit_mix.lambdas.as_slice().unwrap_or(&[])),
            LikelihoodFamily::BinomialMixture,
            &o,
        )
        .expect("profile mix")
        .reml_score
    };
    for j in 0..rho_hat.len() {
        let mut rp = rho_hat.clone();
        let mut rm = rho_hat.clone();
        rp[j] += h_rho;
        rm[j] -= h_rho;
        let fd = (profile_mix(&rp) - profile_mix(&rm)) / (2.0 * h_rho);
        assert!(
            fd.abs() < 0.35,
            "Mixture profile FD gradient too large at rho[{j}]: {fd:.4}"
        );
    }
}
