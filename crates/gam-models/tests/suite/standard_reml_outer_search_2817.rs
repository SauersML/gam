//! #2817: the standard REML outer search on an ordinary Gaussian additive model
//! stops on its own verdict inside one iteration budget.
//!
//! The issue's fit is `yg ~ s(x1) + s(x2) + s(x3) + s(x4) + te(x5, x6)`,
//! gaussian, n = 50 000, p = 93, 13 smoothing parameters. At that size the outer
//! Hessian takes the operator route (`large_n_moderate_p`), and the dense ARC
//! solve reaches it by materializing the operator. On that fit every seed ran
//! its 200-iteration budget and the ARC budget retry re-ran the sweep; at
//! 7ad913f69 the accepted tail steps bought 0.17 to 0.27 of the decrease their
//! model predicted and the certificate refused the fit (job 506109).
//!
//! `outer_iterations` is the total across every solver start the search ran:
//! seeds, budget retries and degraded plans. So a total under one budget means
//! no start exhausted it, which is the whole claim of the issue.

use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;
use gam_models::fit_orchestration::{FitConfig, FitResult, fit_from_formula};
use gam_solve::estimate::reml::reml_outer_engine::{
    MATRIX_FREE_OUTER_HESSIAN_DIM_AT_LARGE_N, MATRIX_FREE_OUTER_HESSIAN_LARGE_N_THRESHOLD,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

/// The outer iteration budget handed to the fit, and the bar its total is held to.
const MAX_ITER: usize = 200;
const NOISE_SD: f64 = 0.3;

/// Curvature on every margin and a genuine interaction on the tensor pair, so no
/// smoothing parameter belongs on a rail.
fn truth(x: &[f64; 6]) -> f64 {
    use std::f64::consts::PI;
    (PI * x[0]).sin()
        + x[1] * x[1]
        + 0.5 * x[2].powi(3)
        + (2.0 * x[3]).cos()
        + (PI * x[4]).sin() * x[5]
}

#[test]
fn gaussian_additive_fit_outer_search_stops_within_one_budget_2817() {
    let n = MATRIX_FREE_OUTER_HESSIAN_LARGE_N_THRESHOLD;
    let mut rng = StdRng::seed_from_u64(2817);
    let covariate = Uniform::new(-1.0_f64, 1.0).expect("valid uniform range");
    let noise = Normal::new(0.0, NOISE_SD).expect("valid normal");
    let headers: Vec<String> = ["yg", "x1", "x2", "x3", "x4", "x5", "x6"]
        .iter()
        .map(|name| name.to_string())
        .collect();
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let x: [f64; 6] = std::array::from_fn(|_| covariate.sample(&mut rng));
        let y = truth(&x) + noise.sample(&mut rng);
        let mut record = vec![y.to_string()];
        record.extend(x.iter().map(|value| value.to_string()));
        rows.push(StringRecord::from(record));
    }
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode the #2817 fixture");
    let config = FitConfig {
        family: Some("gaussian".to_string()),
        outer_max_iter: Some(MAX_ITER),
        ..FitConfig::default()
    };
    let fit = match fit_from_formula(
        "yg ~ s(x1) + s(x2) + s(x3) + s(x4) + te(x5, x6)",
        &data,
        &config,
    ) {
        Ok(FitResult::Standard(fit)) => fit,
        Ok(_) => panic!("a Gaussian additive formula must produce a standard fit"),
        Err(error) => panic!("#2817: the outer search must reach a certified optimum: {error}"),
    };
    let p = fit.design.design.ncols();
    eprintln!(
        "[#2817 acceptance] n={n} p={p} outer_iterations={} log_lambdas={:?}",
        fit.fit.outer_iterations, fit.fit.log_lambdas,
    );
    assert!(
        p >= MATRIX_FREE_OUTER_HESSIAN_DIM_AT_LARGE_N,
        "fixture precondition: the operator route at n = {n} needs p >= {}, got p = {p}",
        MATRIX_FREE_OUTER_HESSIAN_DIM_AT_LARGE_N,
    );
    assert!(
        fit.fit.outer_iterations < MAX_ITER,
        "#2817: the outer search spent {} iterations across its solver starts, at least one \
         whole budget of {MAX_ITER}, so some start ran out of budget instead of stopping on \
         its own verdict",
        fit.fit.outer_iterations,
    );
}
