use gam::matrix::SymmetricMatrix;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam_predict::linalg::PredictionCovarianceBackend;
use gam_predict::{
    InferenceCovarianceMode, MeanIntervalMethod, PredictPosteriorMeanResult,
    PredictUncertaintyOptions, enrich_posterior_mean_bounds,
    predict_gam_posterior_meanwith_backend, predict_gamwith_uncertainty,
};
use ndarray::{Array1, Array2, array};

fn like(response: ResponseFamily, link: StandardLink) -> LikelihoodSpec {
    LikelihoodSpec::new(response, InverseLink::Standard(link))
}

#[test]
fn predict_uncertainty_bounds_track_requested_alpha_level_for_logit() {
    let x = array![[1.0, -1.5], [1.0, -0.5], [1.0, 0.5], [1.0, 1.5]];
    let beta = array![0.2, 1.1];
    let offset = Array1::zeros(4);
    let cov = array![[0.12, 0.01], [0.01, 0.18]];
    let pred = predict_gamwith_uncertainty(
        x.view(),
        beta.view(),
        offset.view(),
        like(ResponseFamily::Binomial, StandardLink::Logit),
        &cov,
        &PredictUncertaintyOptions {
            confidence_level: 0.8,
            covariance_mode: InferenceCovarianceMode::Conditional,
            mean_interval_method: MeanIntervalMethod::TransformEta,
            ..PredictUncertaintyOptions::default()
        },
    )
    .expect("predict_gamwith_uncertainty should succeed for a simple logit design");
    for i in 0..pred.mean.len() {
        assert!(
            pred.mean_lower[i] <= pred.mean[i] && pred.mean[i] <= pred.mean_upper[i],
            "Posterior mean should fall inside its own requested 80% interval at row {i}"
        );
    }
}

#[test]
fn enrich_posterior_mean_bounds_clamps_domains_for_probability_and_count_families() {
    let mut beta_result = PredictPosteriorMeanResult {
        eta: array![-4.0, 4.0],
        eta_standard_error: array![3.0, 3.0],
        mean: array![0.2, 0.8],
        mean_standard_error: None,
        mean_lower: None,
        mean_upper: None,
        observation_lower: None,
        observation_upper: None,
    };
    enrich_posterior_mean_bounds(
        &mut beta_result,
        0.95,
        like(ResponseFamily::Beta { phi: 20.0 }, StandardLink::Logit),
        None,
    )
    .expect("beta-family bounds should be enrichable");
    for i in 0..2 {
        assert!(
            beta_result.mean_lower.as_ref().expect("lower")[i] >= 1e-10
                && beta_result.mean_upper.as_ref().expect("upper")[i] <= 1.0 - 1e-10,
            "Beta-family bounds should stay strictly inside (0,1) after clamping"
        );
    }

    let mut pois_result = PredictPosteriorMeanResult {
        eta: array![-3.0, 1.0],
        eta_standard_error: array![5.0, 5.0],
        mean: array![0.0, 0.0],
        mean_standard_error: None,
        mean_lower: None,
        mean_upper: None,
        observation_lower: None,
        observation_upper: None,
    };
    enrich_posterior_mean_bounds(
        &mut pois_result,
        0.95,
        like(ResponseFamily::Poisson, StandardLink::Log),
        None,
    )
    .expect("poisson-family bounds should be enrichable");
    assert!(
        pois_result
            .mean_lower
            .as_ref()
            .expect("lower")
            .iter()
            .all(|v| *v >= 0.0)
            && pois_result
                .mean_upper
                .as_ref()
                .expect("upper")
                .iter()
                .all(|v| *v >= 0.0),
        "Poisson-family bounds should never go below zero"
    );
}

#[test]
fn delta_method_variance_matches_posterior_simulation_for_small_logit_problem() {
    let x = array![[1.0, 0.8]];
    let beta = array![0.3, 0.7];
    let cov = array![[0.2, 0.05], [0.05, 0.15]];
    let offset = array![0.0];
    let pred = predict_gamwith_uncertainty(
        x.view(),
        beta.view(),
        offset.view(),
        like(ResponseFamily::Binomial, StandardLink::Logit),
        &cov,
        &PredictUncertaintyOptions {
            mean_interval_method: MeanIntervalMethod::Delta,
            covariance_mode: InferenceCovarianceMode::Conditional,
            ..PredictUncertaintyOptions::default()
        },
    )
    .expect("delta-method prediction should succeed");
    let delta_var = pred.mean_standard_error[0] * pred.mean_standard_error[0];

    let l11 = cov[[0, 0]].sqrt();
    let l21 = cov[[1, 0]] / l11;
    let l22 = (cov[[1, 1]] - l21 * l21).sqrt();
    let mut vals = Vec::new();
    let mut state = 1_u64;
    for _ in 0..20000 {
        state = (state * 48271) % 2147483647;
        let u1 = (state as f64) / 2147483647.0;
        state = (state * 48271) % 2147483647;
        let u2 = (state as f64) / 2147483647.0;
        let r = (-2.0 * u1.max(1e-12).ln()).sqrt();
        let z1 = r * (2.0 * std::f64::consts::PI * u2).cos();
        let z2 = r * (2.0 * std::f64::consts::PI * u2).sin();
        let b0 = beta[0] + l11 * z1;
        let b1 = beta[1] + l21 * z1 + l22 * z2;
        let eta = b0 + 0.8 * b1;
        vals.push(1.0 / (1.0 + (-eta).exp()));
    }
    let m = vals.iter().sum::<f64>() / vals.len() as f64;
    let mc_var = vals.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (vals.len() as f64 - 1.0);
    assert!(
        (delta_var - mc_var).abs() < 5e-3,
        "Delta-method variance should match dense posterior simulation within Monte Carlo error"
    );
}

#[test]
fn backend_variance_matches_between_dense_and_factorized_backends() {
    let x = array![[1.0, -1.0], [1.0, 1.5]];
    let beta = array![0.2, -0.4];
    let offset = array![0.0, 0.0];
    let cov = array![[0.3, 0.0], [0.0, 0.2]];
    let dense = PredictionCovarianceBackend::from_dense(cov.view());
    let h = Array2::from_diag(&array![1.0 / 0.3, 1.0 / 0.2]);
    let fact = PredictionCovarianceBackend::from_factorized_hessian(SymmetricMatrix::Dense(h))
        .expect("factorized backend should be constructible from diagonal Hessian");
    let a = predict_gam_posterior_meanwith_backend(
        x.view(),
        beta.view(),
        offset.view(),
        like(ResponseFamily::Binomial, StandardLink::Logit),
        &dense,
    )
    .expect("dense backend should predict");
    let b = predict_gam_posterior_meanwith_backend(
        x.view(),
        beta.view(),
        offset.view(),
        like(ResponseFamily::Binomial, StandardLink::Logit),
        &fact,
    )
    .expect("factorized backend should predict");
    for i in 0..a.eta_standard_error.len() {
        assert!(
            (a.eta_standard_error[i] - b.eta_standard_error[i]).abs() < 1e-9,
            "Backend-1 and backend-2 should agree on posterior eta variance within 1e-9"
        );
    }
}

#[test]
fn survival_uncertainty_bounds_stay_in_unit_interval() {
    let x = array![[1.0, -1.0], [1.0, 0.2], [1.0, 1.1]];
    let beta = array![0.3, 0.5];
    let cov = array![[0.1, 0.0], [0.0, 0.2]];
    let offset = Array1::zeros(3);
    let pred = predict_gamwith_uncertainty(
        x.view(),
        beta.view(),
        offset.view(),
        like(ResponseFamily::RoystonParmar, StandardLink::Log),
        &cov,
        &PredictUncertaintyOptions::default(),
    )
    .expect("survival uncertainty prediction should succeed");
    for i in 0..pred.mean.len() {
        assert!(
            pred.mean_lower[i] >= 0.0 && pred.mean_upper[i] <= 1.0,
            "Survival probability bounds should stay in [0,1] at every row"
        );
    }
}

#[test]
fn competing_risks_cif_bounds_are_probabilities_and_total_mass_is_valid() {
    use gam::families::survival::assemble_competing_risks_cif_from_endpoints;
    let times = array![0.0, 1.0, 2.0, 3.0];
    let endpoint_hazards = vec![
        array![[0.02, 0.03, 0.03, 0.04]],
        array![[0.01, 0.02, 0.03, 0.03]],
    ];
    let assembled = assemble_competing_risks_cif_from_endpoints(times.view(), &endpoint_hazards)
        .expect("Competing-risks CIF assembly should succeed for finite monotone times and positive hazards");
    for i in 0..times.len() {
        let s = assembled.overall_survival[[0, i]];
        let c1 = assembled.cif[0][[0, i]];
        let c2 = assembled.cif[1][[0, i]];
        assert!(
            (0.0..=1.0).contains(&s) && (0.0..=1.0).contains(&c1) && (0.0..=1.0).contains(&c2),
            "Each survival/CIF bound should remain a probability in [0,1]"
        );
        assert!(
            s + c1 + c2 <= 1.0 + 1e-9,
            "At each time, competing-risks CIF totals plus survival should not exceed one within tolerance"
        );
    }
}
