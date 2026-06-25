//! Bug hunt: the response-scale *observation* (prediction) interval is not
//! clamped to the family's response support, so a Poisson model — whose response
//! is a non-negative count (`y >= 0`, see
//! `ResponseFamily::response_support_requirement` in `src/types.rs`) — reports
//! observation-interval lower bounds that are strictly negative.
//!
//! The mean (confidence) interval respects the support: for a log-link family it
//! is built by transforming the η endpoints through `exp`, so it stays positive.
//! But the observation interval is the symmetric band `mu ± z·sigma` on the
//! response scale, assembled in
//! `src/inference/predict/interval_policy.rs::assemble_uncertainty_result`
//! (~lines 274–280):
//!
//! ```ignore
//! let half = obs.noise_sd.mapv(|s| z * s);
//! (Some(&mean - &half), Some(&mean + &half))   // <- no support clamp
//! ```
//!
//! Unlike the mean interval (`bounds.clamp_in_place`), this path never clamps to
//! the response support. For a Poisson mean `mu` small enough that
//! `mu - z·sqrt(mu) < 0`, the lower bound of the predictive interval for a count
//! falls below 0, which is outside the declared support and not a valid count.
//!
//! Observed (via the Python binding, family="poisson", interval=0.9,
//! observation_interval=True): observation_lower = [-0.677, -0.655, -0.413, ...].
//!
//! When the observation interval is floored at the response support (0 for
//! Poisson / NegBin / Tweedie, the lower edge of `response_support`), this test
//! passes unchanged.

use csv::StringRecord;
use gam_predict::{
    InferenceCovarianceMode, MeanIntervalMethod, PredictUncertaintyOptions,
    predict_gamwith_uncertainty,
};
use gam::smooth::build_term_collection_design;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Poisson, Uniform};

#[test]
fn poisson_observation_interval_stays_within_nonnegative_support() {
    init_parallelism();

    // Small-count Poisson: log mean = 0.4 + 0.6 x, so mu in roughly [0.3, 1.7]
    // over x in [-2, 2]. At the low end mu - z*sqrt(mu) is clearly negative.
    let n = 800usize;
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let ux = Uniform::new(-2.0_f64, 2.0_f64).expect("uniform");
    let x: Vec<f64> = (0..n).map(|_| ux.sample(&mut rng)).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let mu = (0.4 + 0.6 * xi).exp();
            Poisson::new(mu).expect("poisson").sample(&mut rng) as f64
        })
        .collect();

    let headers: Vec<String> = ["y", "x"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| StringRecord::from(vec![y[i].to_string(), x[i].to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let col = ds.column_map();
    let x_idx = col["x"];

    let cfg = FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    };
    let FitResult::Standard(fit) = fit_from_formula("y ~ s(x)", &ds, &cfg).expect("fit") else {
        panic!("expected a standard Poisson fit");
    };

    // Evaluate where the mean is smallest (left end), to make the symmetric band
    // dip below zero.
    let eval: Vec<f64> = vec![-2.0, -1.5, -1.0, -0.5, 0.0];
    let m = eval.len();
    let mut grid = Array2::<f64>::zeros((m, ds.headers.len()));
    for (i, &xi) in eval.iter().enumerate() {
        grid[[i, x_idx]] = xi;
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec).expect("design");
    let dense = design.design.to_dense();

    let poisson_log = LikelihoodSpec::new(
        ResponseFamily::Poisson,
        InverseLink::Standard(StandardLink::Log),
    );
    let offset = Array1::<f64>::zeros(m);
    let pred = predict_gamwith_uncertainty(
        dense,
        fit.fit.beta.view(),
        offset.view(),
        poisson_log,
        &fit.fit,
        &PredictUncertaintyOptions {
            confidence_level: 0.95,
            covariance_mode: InferenceCovarianceMode::Conditional,
            mean_interval_method: MeanIntervalMethod::Delta,
            includeobservation_interval: true,
            apply_bias_correction: false,
            edgeworth_one_sided: false,
            boundary_correction: false,
            ood_inflation: false,
            multi_point_joint: false,
            ..PredictUncertaintyOptions::default()
        },
    )
    .expect("predict with observation interval");

    let obs_lower = pred
        .observation_lower
        .expect("Poisson predictor should expose an observation interval");
    let means = pred.mean.to_vec();

    // The mean interval must be positive (sanity: the model is on the log scale).
    assert!(
        means.iter().all(|&mu| mu > 0.0),
        "Poisson fitted means should be positive: {means:?}"
    );

    // The predictive interval for a count must lie within the response support
    // y >= 0. Every observation lower bound must therefore be >= 0.
    for (i, &lo) in obs_lower.iter().enumerate() {
        assert!(
            lo >= 0.0,
            "Poisson observation-interval lower bound is below the y >= 0 support \
             at eval x={} (mu={}): observation_lower={lo}",
            eval[i],
            means[i],
        );
    }
}
