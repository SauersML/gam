//! A Gaussian location-scale fit must recover a noise SD that is small relative
//! to the response's spread.
//!
//! The noise link is σ = b + exp(η), fitted on the response standardized by its
//! sample SD. The floor b used to be mgcv's `gaulss(b = 0.01)` constant, i.e. 1 %
//! of sd(y). A clean linear signal with σ at or below that level put the
//! likelihood's optimum at exp(η) → 0, the log-σ intercept ran off to −∞, and
//! every such fit ended in `InnerModeConvergenceError` — whatever unit the
//! response was recorded in, because the standardization makes the failure
//! depend only on σ / sd(y).
//!
//! The floor is now the recording-grid bound δ/√12 of the standardized response
//! (`gaussian_resolution_sigma_floor`), which lies below any noise level the data
//! can resolve. These fits must converge and report σ̂ within sampling error of
//! the residual SD of the true mean.

use gam::estimate::BlockRole;
use gam::gamlss::GaussianLocationScaleFitResult;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;

/// Deterministic uniform stream (64-bit LCG, top 53 bits).
fn next_unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

/// `y = 1 + 10·x + σ·z` with `x ~ U(0, 1)` and `z` standard normal (Box–Muller).
/// Returns `(x, y, oracle)`, where `oracle = sqrt(mean((σ·z)²))` is the residual
/// SD about the true mean: the noise scale this sample actually carries.
fn linear_signal(n: usize, sigma: f64, seed: u64) -> (Vec<f64>, Vec<f64>, f64) {
    let mut state = seed;
    let x: Vec<f64> = (0..n).map(|_| next_unit(&mut state)).collect();
    let noise: Vec<f64> = (0..n)
        .map(|_| {
            let u1 = 1.0 - next_unit(&mut state);
            let u2 = next_unit(&mut state);
            sigma * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        })
        .collect();
    let y = x.iter().zip(&noise).map(|(&xi, &e)| 1.0 + 10.0 * xi + e).collect();
    let oracle = (noise.iter().map(|e| e * e).sum::<f64>() / n as f64).sqrt();
    (x, y, oracle)
}

/// Fit `y ~ x` with an intercept-only noise model and return the fitted σ̂ in
/// raw units, `response_scale·sigma_floor + exp(η̂)`, with the raw floor.
fn fitted_noise_scale(x: &[f64], y: &[f64]) -> (f64, f64) {
    let headers = vec!["x".to_string(), "y".to_string()];
    let rows: Vec<csv::StringRecord> = x
        .iter()
        .zip(y)
        .map(|(xi, yi)| csv::StringRecord::from(vec![format!("{xi:.17e}"), format!("{yi:.17e}")]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode linear signal");
    let x_idx = ds.column_map()["x"];
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        noise_formula: Some("1".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ x", &ds, &cfg)
        .unwrap_or_else(|error| panic!("high-SNR Gaussian location-scale fit failed: {error}"));
    let FitResult::GaussianLocationScale(GaussianLocationScaleFitResult {
        fit,
        response_scale,
        sigma_floor,
        ..
    }) = result
    else {
        panic!("expected a Gaussian location-scale fit");
    };
    let beta_scale = fit
        .fit
        .block_by_role(BlockRole::Scale)
        .expect("scale (log-sigma) block present")
        .beta
        .clone();
    let mut grid = Array2::<f64>::zeros((1, ds.headers.len()));
    grid[[0, x_idx]] = 0.5;
    let noise_design = build_term_collection_design(grid.view(), &fit.noisespec_resolved)
        .expect("rebuild intercept-only noise design");
    let eta = noise_design.design.apply(&beta_scale)[0];
    let raw_floor = response_scale * sigma_floor;
    (raw_floor + eta.exp(), raw_floor)
}

#[test]
fn gaussian_location_scale_recovers_noise_far_below_one_percent_of_the_spread() {
    init_parallelism();
    let n = 600;
    // sd(y) ≈ 10/√12 ≈ 2.9, so these are σ / sd(y) ≈ 7e-3, 7e-4 and 7e-5: all
    // below the former fixed floor of 1 % of the spread.
    for (sigma, seed) in [(0.02, 7_u64), (2.0e-3, 11), (2.0e-4, 13)] {
        let (x, y, oracle) = linear_signal(n, sigma, seed);
        let (sigma_hat, raw_floor) = fitted_noise_scale(&x, &y);
        // The fitted σ̂ is the root mean square of the fitted residuals, whose
        // relative gap to the oracle is O(p/n) from the two fitted mean
        // coefficients, plus O(1/n) from the REML degrees-of-freedom correction.
        let relative = (sigma_hat / oracle - 1.0).abs();
        assert!(
            relative < 0.02,
            "σ = {sigma:e}: fitted σ̂ = {sigma_hat:e} but the sample's residual SD about \
             the true mean is {oracle:e} (relative error {relative:.3e})"
        );
        assert!(
            raw_floor < 0.01 * oracle,
            "σ = {sigma:e}: the raw σ floor {raw_floor:e} must sit far below the noise \
             the data resolve ({oracle:e})"
        );
    }
}
