//! Probe (NOT a permanent guard): measure the Gaussian location-scale
//! over-smoothing of the log-sigma surface (#1561) under several penalty
//! configurations, to localize whether the over-smoothing is driven by the
//! default null-space double penalty on the SCALE block (mgcv `gaulss` defaults
//! to `select=FALSE`, i.e. NO null-space penalty) vs the wiggliness lambda
//! selection itself. Prints pearson(log sigma, truth), rmse, lambdas, edf.

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

fn data(n: usize) -> (Vec<f64>, Vec<f64>) {
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

fn run_case(label: &str, mean_formula: &str, noise_formula: &str, n: usize) {
    let (x, y) = data(n);
    let headers = vec!["x".to_string(), "y".to_string()];
    let rows: Vec<csv::StringRecord> = x
        .iter()
        .zip(&y)
        .map(|(&x, &y)| csv::StringRecord::from(vec![format!("{x:.17e}"), format!("{y:.17e}")]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        noise_formula: Some(noise_formula.to_string()),
        ..FitConfig::default()
    };
    let result = match fit_from_formula(mean_formula, &ds, &cfg) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[{label}] FIT ERROR: {e}");
            return;
        }
    };
    let FitResult::GaussianLocationScale(GaussianLocationScaleFitResult { fit, .. }) = result else {
        eprintln!("[{label}] not a location-scale fit");
        return;
    };

    let beta_loc = fit
        .fit
        .block_by_role(BlockRole::Location)
        .expect("loc")
        .beta
        .clone();
    let beta_scale = fit
        .fit
        .block_by_role(BlockRole::Scale)
        .expect("scale")
        .beta
        .clone();

    let mut grid = Array2::<f64>::zeros((x.len(), 2));
    for (i, &t) in x.iter().enumerate() {
        grid[[i, 0]] = t;
    }
    let mean_design = build_term_collection_design(grid.view(), &fit.meanspec_resolved).expect("md");
    let scale_design =
        build_term_collection_design(grid.view(), &fit.noisespec_resolved).expect("sd");
    let gam_mu = mean_design.design.apply(&beta_loc).to_vec();
    let gam_eta_sigma = scale_design.design.apply(&beta_scale).to_vec();
    let gam_log_sigma: Vec<f64> = gam_eta_sigma
        .iter()
        .map(|&e| (LOGB_SIGMA_FLOOR + e.exp()).ln())
        .collect();

    let true_mu: Vec<f64> = x.iter().map(|&t| mean_truth(t)).collect();
    let true_log_sigma: Vec<f64> = x.iter().map(|&t| sigma_truth(t).abs().ln()).collect();

    let rmse_mu = rmse(&gam_mu, &true_mu);
    let corr = pearson(&gam_log_sigma, &true_log_sigma);
    let rmse_ls = rmse(&gam_log_sigma, &true_log_sigma);

    let (lambdas, log_lambdas, edf_by_block, edf_total) = if let Some(inf) = fit.fit.inference.as_ref()
    {
        (
            fit.fit.lambdas.to_vec(),
            fit.fit.log_lambdas.to_vec(),
            inf.edf_by_block.clone(),
            inf.edf_total,
        )
    } else {
        (vec![], vec![], vec![], f64::NAN)
    };

    eprintln!(
        "[{label}] pearson={corr:.5} rmse_ls={rmse_ls:.5} rmse_mu={rmse_mu:.5} \
         | lambdas={lambdas:?} log_lambdas={log_lambdas:?} \
         | edf_by_block={edf_by_block:?} edf_total={edf_total:.3} \
         | outer_conv={} iters={}",
        fit.fit.outer_converged, fit.fit.outer_iterations
    );
}

#[test]
fn probe_1561_locscale_penalty_configs() {
    init_parallelism();
    let n = 200;
    eprintln!("=== #1561 probe: Gaussian location-scale log-sigma recovery (n={n}) ===");
    // Baseline: default (double_penalty=true on both blocks).
    run_case("default", "y ~ s(x, bs='tp')", "1 + s(x, bs='tp')", n);
    // Scale block WITHOUT null-space double penalty (mgcv gaulss select=FALSE).
    run_case(
        "scale_nodbl",
        "y ~ s(x, bs='tp')",
        "1 + s(x, bs='tp', double_penalty=false)",
        n,
    );
    // BOTH blocks without null-space double penalty.
    run_case(
        "both_nodbl",
        "y ~ s(x, bs='tp', double_penalty=false)",
        "1 + s(x, bs='tp', double_penalty=false)",
        n,
    );
}
