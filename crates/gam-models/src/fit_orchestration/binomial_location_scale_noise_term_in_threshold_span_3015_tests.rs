#![cfg(test)]
//! Regression for gam#3015: a binomial location-scale fit whose noise formula holds a term
//! the threshold's Duchon smooth also spans was refused at every startup seed.
//!
//! The binomial log-σ design is residualized against the threshold design, so a noise
//! column that lies in the threshold's span, here `x3` in the affine null space of
//! `duchon(x2, x3)`, becomes a zero column. The identifiability audit then drops it, and
//! the solve runs on specs one column narrower than the designs the family was built
//! with. The family's exact joint curvature read its stored, pre-audit designs, so the
//! inner solve's Hessian was one column wider than its coefficients: "dense Hessian shape
//! mismatch: got 11x11, expected 10x10". Every location-scale family now takes its
//! designs from the specs it is handed.
//!
//! The pin fits gamfit's reproduction (issue #3015) with the shared noise column, and
//! with an independent noise column as the control, and requires both to fit and publish
//! the covariance a location-scale prediction needs.

use super::{FitConfig, FitResult, fit_from_formula};

const ROWS: usize = 800;

fn uniform(state: &mut u64) -> f64 {
    (gam_linalg::utils::splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn normal(state: &mut u64) -> f64 {
    let u1 = uniform(state).max(f64::MIN_POSITIVE);
    let u2 = uniform(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// gamfit's #3015 table: `P(y = 1) = (1 + tanh(η/1.6))/2` with
/// `η = (−0.3 + 0.5·x1 + sin x2)/exp(0.3·x3)`.
fn table() -> gam_data::EncodedDataset {
    let mut state = 3015u64;
    let headers = ["y", "x1", "x2", "x3", "x4"].map(String::from).to_vec();
    let records = (0..ROWS)
        .map(|_| {
            let x: Vec<f64> = (0..4).map(|_| normal(&mut state)).collect();
            let eta = (-0.3 + 0.5 * x[0] + x[1].sin()) / (0.3 * x[2]).exp();
            let p = 0.5 * (1.0 + (eta / 1.6).tanh());
            let y = if uniform(&mut state) < p { 1.0 } else { 0.0 };
            let mut row = vec![format!("{y}")];
            row.extend(x.iter().map(|v| format!("{v:.17e}")));
            csv::StringRecord::from(row)
        })
        .collect();
    gam_data::encode_recordswith_inferred_schema(headers, records).expect("encode the #3015 table")
}

fn fit(noise: &str) {
    let data = table();
    let config = FitConfig {
        family: Some("binomial".to_string()),
        link: Some("probit".to_string()),
        noise_formula: Some(noise.to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ x1 + duchon(x2, x3, centers=8)", &data, &config)
        .unwrap_or_else(|error| {
            panic!("#3015: the binomial location-scale fit with noise_formula={noise} must fit: {error}")
        });
    let FitResult::BinomialLocationScale(fitted) = result else {
        panic!("#3015: noise_formula={noise} must return a binomial location-scale fit");
    };
    let unified = &fitted.fit.fit;
    assert!(
        unified.convergence_evidence().inner_status().is_converged(),
        "#3015: noise_formula={noise}: the inner mode is not converged"
    );
    let covariance = unified
        .beta_covariance()
        .unwrap_or_else(|| panic!("#3015: noise_formula={noise}: no joint posterior covariance"));
    let width = unified.beta_flat().len();
    assert_eq!(
        covariance.dim(),
        (width, width),
        "#3015: noise_formula={noise}: the covariance must cover every saved coefficient"
    );
}

#[test]
fn a_noise_term_in_the_threshold_span_fits_3015() {
    fit("x3");
}

#[test]
fn a_noise_term_outside_the_threshold_span_fits_3015() {
    fit("x4");
}
