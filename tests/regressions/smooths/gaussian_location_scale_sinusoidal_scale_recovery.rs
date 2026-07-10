//! Regression guard for the Gaussian location-scale over-smoothing failure
//! surfaced by `quality_vs_gamlss_gaussian_location_scale`.

use gam::estimate::BlockRole;
use gam::gamlss::GaussianLocationScaleFitResult;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{pearson, rmse};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;

const LOGB_SIGMA_FLOOR: f64 = 0.01;

fn next_unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn mean_truth(x: f64) -> f64 {
    (2.0 * std::f64::consts::PI * x).sin()
}

fn sigma_truth(x: f64) -> f64 {
    0.1 + 0.2 * (2.0 * std::f64::consts::PI * x).sin()
}

fn sinusoidal_location_scale_data(n: usize) -> (Vec<f64>, Vec<f64>) {
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut state = 42_u64;
    let mut x: Vec<f64> = (0..n).map(|_| next_unit(&mut state)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).expect("finite x"));

    let mut z = Vec::with_capacity(n);
    while z.len() < n {
        let u1 = next_unit(&mut state).max(1e-300);
        let u2 = next_unit(&mut state);
        let r = (-2.0 * u1.ln()).sqrt();
        z.push(r * (two_pi * u2).cos());
        if z.len() < n {
            z.push(r * (two_pi * u2).sin());
        }
    }

    let y = (0..n)
        .map(|i| mean_truth(x[i]) + sigma_truth(x[i]) * z[i])
        .collect();
    (x, y)
}

fn fit_sinusoidal_location_scale(x: &[f64], y: &[f64]) -> GaussianLocationScaleFitResult {
    let headers = vec!["x".to_string(), "y".to_string()];
    let rows: Vec<csv::StringRecord> = x
        .iter()
        .zip(y)
        .map(|(&x, &y)| csv::StringRecord::from(vec![format!("{x:.17e}"), format!("{y:.17e}")]))
        .collect();
    let data =
        encode_recordswith_inferred_schema(headers, rows).expect("encode location-scale fixture");
    let config = FitConfig {
        family: Some("gaussian".to_string()),
        noise_formula: Some("1 + s(x, bs='tp')".to_string()),
        ..FitConfig::default()
    };

    let result = fit_from_formula("y ~ s(x, bs='tp')", &data, &config)
        .expect("fit Gaussian location-scale model");
    let FitResult::GaussianLocationScale(result) = result else {
        panic!("expected GaussianLocationScale fit result");
    };
    result
}

fn fitted_channels(fit: &GaussianLocationScaleFitResult, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut grid = Array2::<f64>::zeros((x.len(), 2));
    for (i, &x) in x.iter().enumerate() {
        grid[[i, 0]] = x;
    }

    let beta_location = fit
        .fit
        .fit
        .block_by_role(BlockRole::Location)
        .expect("location block")
        .beta
        .clone();
    let beta_scale = fit
        .fit
        .fit
        .block_by_role(BlockRole::Scale)
        .expect("scale block")
        .beta
        .clone();

    let mean_design = build_term_collection_design(grid.view(), &fit.fit.meanspec_resolved)
        .expect("mean design at fixture grid");
    let scale_design = build_term_collection_design(grid.view(), &fit.fit.noisespec_resolved)
        .expect("scale design at fixture grid");

    let fitted_mu = mean_design.design.apply(&beta_location).to_vec();
    let fitted_log_sigma = scale_design
        .design
        .apply(&beta_scale)
        .iter()
        .map(|&eta| (fit.response_scale * LOGB_SIGMA_FLOOR + eta.exp()).ln())
        .collect();
    (fitted_mu, fitted_log_sigma)
}

#[test]
fn gaussian_location_scale_recovers_sinusoidal_mean_and_scale() {
    init_parallelism();

    let n = 200;
    let (x, y) = sinusoidal_location_scale_data(n);
    let fit = fit_sinusoidal_location_scale(&x, &y);
    let (fitted_mu, fitted_log_sigma) = fitted_channels(&fit, &x);

    let true_mu: Vec<f64> = x.iter().map(|&x| mean_truth(x)).collect();
    let true_log_sigma: Vec<f64> = x.iter().map(|&x| sigma_truth(x).abs().ln()).collect();
    let mean_noise = x.iter().map(|&x| sigma_truth(x).abs()).sum::<f64>() / n as f64;

    let rmse_mu = rmse(&fitted_mu, &true_mu);
    let corr_log_sigma = pearson(&fitted_log_sigma, &true_log_sigma);
    let rmse_log_sigma = rmse(&fitted_log_sigma, &true_log_sigma);

    let (lambdas, log_lambdas, edf_by_block, penalty_block_trace, edf_total) =
        if let Some(inference) = fit.fit.fit.inference.as_ref() {
            (
                fit.fit.fit.lambdas.to_vec(),
                fit.fit.fit.log_lambdas.to_vec(),
                inference.edf_by_block.clone(),
                inference.penalty_block_trace.clone(),
                inference.edf_total,
            )
        } else {
            (vec![], vec![], vec![], vec![], f64::NAN)
        };

    let mu_bar = 0.5 * mean_noise;
    assert!(
        rmse_mu < mu_bar && corr_log_sigma > 0.80,
        "Gaussian location-scale sinusoidal recovery regressed: \
         mu_rmse={rmse_mu:.6} must be < {mu_bar:.6}; \
         log_sigma_pearson={corr_log_sigma:.6} must be > 0.80; \
         log_sigma_rmse={rmse_log_sigma:.6}; \
         outer_converged=certified outer_iterations={}; \
         lambdas={lambdas:?}; log_lambdas={log_lambdas:?}; \
         edf_by_block={edf_by_block:?}; penalty_block_trace={penalty_block_trace:?}; \
         edf_total={edf_total}",
        fit.fit.fit.outer_iterations
    );
}
