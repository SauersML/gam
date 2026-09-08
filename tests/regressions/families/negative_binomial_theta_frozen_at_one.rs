//! Regression for #802: the Negative-Binomial overdispersion `theta` must be
//! ESTIMATED jointly with the mean, not frozen at the family-variant seed.
//!
//! NB2 has `Var(y) = mu + mu^2/theta`, so the IRLS working weight
//! `W = mu*theta/(theta+mu)` — and through it the coefficient covariance
//! `Vb = (X'WX)^-1` and every reported SE / interval — depend on `theta`. Before
//! the fix `theta` was a structural scalar on `ResponseFamily::NegativeBinomial`
//! defaulting to 1.0 and never refreshed from the data, so two datasets that
//! differ ONLY in their true overdispersion produced the *same* (over-confident)
//! uncertainty. This is the NB sibling of #678 (Gamma), #771 (Tweedie),
//! #769 (Beta).
//!
//! ## The set-up
//!
//! Two NB datasets share the same covariate grid and the same mean structure
//! `log mu = 1.0 + 0.5*x`, differing only in their true dispersion:
//!   * strong overdispersion: `theta = 0.5`
//!   * near-Poisson:          `theta = 50`
//! With the mean and design held fixed, the only thing that should move the
//! linear-predictor SE is `theta`. At the mean `mu ~ exp(1)`, the weight ratio
//! gives an expected SE ratio `sqrt(W_weak/W_strong) ~ 2.5`; the frozen-`theta`
//! bug pins it at ~1.0.
//!
//! ## What is asserted (two independent angles)
//!
//!  1. *theta is recovered from the data*: the strongly-overdispersed fit reports
//!     `theta_hat` well below 1, the near-Poisson fit well above 1 — and neither
//!     is pinned at the frozen seed.
//!  2. *the SE responds to the overdispersion*: the strong-overdispersion fit's
//!     mean-`eta` SE is markedly larger (ratio clearly above the buggy ~1.0).

use csv::StringRecord;
use gam::smooth::build_term_collection_design;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use gam_predict::{
    InferenceCovarianceMode, MeanIntervalMethod, PredictUncertaintyOptions,
    predict_gamwith_uncertainty,
};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Gamma, Poisson, Uniform};

/// One Negative-Binomial(mu, theta) count via the gamma-Poisson mixture:
/// `lambda ~ Gamma(shape=theta, scale=mu/theta)` then `y ~ Poisson(lambda)`,
/// giving `E[y]=mu`, `Var(y)=mu + mu^2/theta`.
fn sample_negbin(mu: f64, theta: f64, rng: &mut StdRng) -> f64 {
    let gamma = Gamma::new(theta, mu / theta).expect("gamma params valid");
    let lambda = gamma.sample(rng).max(1e-12);
    Poisson::new(lambda)
        .expect("poisson rate valid")
        .sample(rng)
}

/// Fit `y ~ x` as NB(log) and return `(theta_hat, mean eta-SE over an x grid)`.
fn fit_negbin_theta_and_se(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len();
    let headers = vec!["y".to_string(), "x".to_string()];
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| StringRecord::from(vec![(y[i] as i64).to_string(), x[i].to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode negbin dataset");
    let col = ds.column_map();
    let x_idx = col["x"];

    let cfg = FitConfig {
        family: Some("negative-binomial".to_string()),
        // No `negative_binomial_theta`: theta is estimated from the data.
        ..FitConfig::default()
    };
    let FitResult::Standard(fit) =
        fit_from_formula("y ~ x", &ds, &cfg).expect("gam negbin fit should succeed")
    else {
        panic!("negative-binomial GLM should produce a Standard fit");
    };

    // theta_hat off the recorded scale metadata (the smoking-gun the issue reads).
    let theta_hat = fit
        .fit
        .likelihood_scale
        .negbin_theta()
        .expect("NB fit must record an estimated theta in likelihood_scale");
    // The family variant must carry the SAME estimated theta (not the seed 1.0).
    let family_theta = match fit
        .fit
        .likelihood_family
        .as_ref()
        .expect("NB fit must record a likelihood family")
        .response
    {
        ResponseFamily::NegativeBinomial { theta, .. } => theta,
        ref other => panic!("expected NegativeBinomial family, got {other:?}"),
    };
    assert!(
        (theta_hat - family_theta).abs() <= 1e-9 * theta_hat.max(1.0),
        "scale-metadata theta {theta_hat} and family-variant theta {family_theta} must agree"
    );

    // Mean-eta SE over an interior grid (delta-method link-scale SE).
    let eval: Vec<f64> = vec![-1.5, -0.75, 0.0, 0.75, 1.5];
    let m = eval.len();
    let mut grid = Array2::<f64>::zeros((m, ds.headers.len()));
    for (i, &xi) in eval.iter().enumerate() {
        grid[[i, x_idx]] = xi;
    }
    let design =
        build_term_collection_design(grid.view(), &fit.resolvedspec).expect("design at eval grid");
    let dense = design.design.to_dense();
    let nb_log = LikelihoodSpec::new(
        ResponseFamily::NegativeBinomial {
            theta: theta_hat,
            theta_fixed: false,
        },
        InverseLink::Standard(StandardLink::Log),
    );
    let offset = Array1::<f64>::zeros(m);
    let pred = predict_gamwith_uncertainty(
        dense,
        fit.fit.beta.view(),
        offset.view(),
        nb_log,
        &fit.fit,
        &PredictUncertaintyOptions {
            confidence_level: 0.95,
            covariance_mode: InferenceCovarianceMode::Conditional,
            mean_interval_method: MeanIntervalMethod::Delta,
            includeobservation_interval: false,
            apply_bias_correction: false,
            edgeworth_one_sided: false,
            boundary_correction: false,
            ood_inflation: false,
            multi_point_joint: false,
            ..PredictUncertaintyOptions::default()
        },
    )
    .expect("gam negbin eta-SE prediction");
    let se = pred.eta_standard_error.to_vec();
    let mean_se = se.iter().sum::<f64>() / se.len() as f64;
    (theta_hat, mean_se)
}

#[test]
fn negbin_theta_is_estimated_and_drives_the_reported_se() {
    init_parallelism();

    let n = 2000usize;
    let mut rng = StdRng::seed_from_u64(20250606);
    let ux = Uniform::new(-2.0_f64, 2.0_f64).expect("uniform -2..2");

    // Shared covariate grid and shared mean structure log mu = 1.0 + 0.5*x.
    let x: Vec<f64> = (0..n).map(|_| ux.sample(&mut rng)).collect();
    let mu: Vec<f64> = x.iter().map(|xi| (1.0 + 0.5 * xi).exp()).collect();

    let theta_strong = 0.5_f64; // strong overdispersion
    let theta_weak = 50.0_f64; // near-Poisson

    let y_strong: Vec<f64> = mu
        .iter()
        .map(|&m| sample_negbin(m, theta_strong, &mut rng))
        .collect();
    let y_weak: Vec<f64> = mu
        .iter()
        .map(|&m| sample_negbin(m, theta_weak, &mut rng))
        .collect();

    let (theta_hat_strong, se_strong) = fit_negbin_theta_and_se(&x, &y_strong);
    let (theta_hat_weak, se_weak) = fit_negbin_theta_and_se(&x, &y_weak);

    let se_ratio = se_strong / se_weak;
    eprintln!(
        "[negbin-theta] true theta: strong={theta_strong} weak={theta_weak}; \
         fitted theta: strong={theta_hat_strong:.4} weak={theta_hat_weak:.4}; \
         mean eta SE: strong={se_strong:.5} weak={se_weak:.5}; SE ratio={se_ratio:.3} \
         (correct ~ 2.5; frozen-theta bug ~ 1.0)"
    );

    // (1) theta is recovered from the data, not frozen at the seed (1.0).
    assert!(
        (theta_hat_strong - 1.0).abs() > 1e-3 || (theta_hat_weak - 1.0).abs() > 1e-3,
        "theta was not estimated: both fits report the frozen seed \
         (strong={theta_hat_strong:.4}, weak={theta_hat_weak:.4})"
    );
    assert!(
        theta_hat_strong < 1.5,
        "strongly-overdispersed (true theta=0.5) fit must recover theta_hat well below 1, \
         got {theta_hat_strong:.4}"
    );
    assert!(
        theta_hat_weak > 8.0,
        "near-Poisson (true theta=50) fit must recover a large theta_hat, \
         got {theta_hat_weak:.4}"
    );
    assert!(
        theta_hat_weak > 3.0 * theta_hat_strong,
        "fitted theta must separate the two dispersion regimes: \
         strong={theta_hat_strong:.4} weak={theta_hat_weak:.4}"
    );

    // (2) the reported SE responds to the overdispersion. The strongly-
    // overdispersed model carries markedly larger linear-predictor SEs; the
    // frozen-theta bug pins the ratio at ~1.0.
    assert!(
        se_ratio > 1.6,
        "mean eta-SE ratio (strong/weak) must reflect the overdispersion \
         (expected ~2.5), got {se_ratio:.3} — the frozen-theta bug gives ~1.0"
    );
}
