#![cfg(test)]
//! Regression for gam#3015: a binomial location-scale fit whose noise formula holds a term
//! the threshold's Duchon smooth also spans was refused at every startup seed.
//!
//! The binomial log-σ design was residualized against the threshold design, so a noise
//! column that lies in the threshold's span, here `x3` in the affine null space of
//! `duchon(x2, x3)`, became a zero column. The identifiability audit then dropped it, and
//! the solve ran on specs one column narrower than the designs the family was built
//! with. The family's exact joint curvature read its stored, pre-audit designs, so the
//! inner solve's Hessian was one column wider than its coefficients: "dense Hessian shape
//! mismatch: got 11x11, expected 10x10". Every location-scale family now takes its
//! designs from the specs it is handed, and no family residualizes its log-σ design: the
//! binomial likelihood identifies a log-σ column the threshold spans whenever the
//! threshold varies (`location_scale_log_sigma_design`), and the residualization had
//! erased that scale effect.
//!
//! The pins fit gamfit's reproduction (issue #3015) with the shared noise column, and
//! with an independent noise column as the control, and require both to fit and publish
//! the covariance a location-scale prediction needs. A third pin fits a probit truth whose
//! log-σ is linear in the shared column and requires the fit to recover that slope.

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

/// Rows of `(y, x1, x2, x3, x4)` with independent standard-normal covariates and
/// `P(y = 1) = probability(x)`.
fn table(rows: usize, probability: impl Fn(&[f64]) -> f64) -> gam_data::EncodedDataset {
    let mut state = 3015u64;
    let headers = ["y", "x1", "x2", "x3", "x4"].map(String::from).to_vec();
    let records = (0..rows)
        .map(|_| {
            let x: Vec<f64> = (0..4).map(|_| normal(&mut state)).collect();
            let y = if uniform(&mut state) < probability(&x) { 1.0 } else { 0.0 };
            let mut row = vec![format!("{y}")];
            row.extend(x.iter().map(|v| format!("{v:.17e}")));
            csv::StringRecord::from(row)
        })
        .collect();
    gam_data::encode_recordswith_inferred_schema(headers, records).expect("encode the #3015 table")
}

fn fit(data: &gam_data::EncodedDataset, noise: &str) -> gam_solve::model_types::UnifiedFitResult {
    let config = FitConfig {
        family: Some("binomial".to_string()),
        link: Some("probit".to_string()),
        noise_formula: Some(noise.to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ x1 + duchon(x2, x3, centers=8)", data, &config)
        .unwrap_or_else(|error| {
            panic!("#3015: the binomial location-scale fit with noise_formula={noise} must fit: {error}")
        });
    let FitResult::BinomialLocationScale(fitted) = result else {
        panic!("#3015: noise_formula={noise} must return a binomial location-scale fit");
    };
    let unified = fitted.fit.fit;
    assert!(
        unified.convergence_evidence().inner_status().is_converged(),
        "#3015: noise_formula={noise}: the inner mode is not converged"
    );
    let width = unified.beta_flat().len();
    let covariance = unified
        .beta_covariance()
        .unwrap_or_else(|| panic!("#3015: noise_formula={noise}: no joint posterior covariance"));
    assert_eq!(
        covariance.dim(),
        (width, width),
        "#3015: noise_formula={noise}: the covariance must cover every saved coefficient"
    );
    unified
}

/// gamfit's #3015 table: `P(y = 1) = (1 + tanh(η/1.6))/2` with
/// `η = (−0.3 + 0.5·x1 + sin x2)/exp(0.3·x3)`.
fn gamfit_table() -> gam_data::EncodedDataset {
    table(ROWS, |x| {
        let eta = (-0.3 + 0.5 * x[0] + x[1].sin()) / (0.3 * x[2]).exp();
        0.5 * (1.0 + (eta / 1.6).tanh())
    })
}

#[test]
fn a_noise_term_in_the_threshold_span_fits_3015() {
    fit(&gamfit_table(), "x3");
}

#[test]
fn a_noise_term_outside_the_threshold_span_fits_3015() {
    fit(&gamfit_table(), "x4");
}

/// A probit truth `P(y = 1) = Φ(−η_t·e^{−η_σ})` with a threshold free of `x3` and
/// `η_σ = 0.3·x3`. The threshold's Duchon smooth spans `x3`, and the likelihood still
/// identifies the log-σ slope because the threshold varies with `x1`. The fitted slope
/// is asymptotically normal about the truth with the posterior standard deviation the
/// fit reports, so it must lie within three of them of 0.3 and more than three of them
/// from 0. A residualized log-σ design carries no `x3` column and fails both.
#[test]
fn a_scale_effect_on_a_threshold_column_is_recovered_3015() {
    const SLOPE: f64 = 0.3;
    let data = table(20_000, |x| {
        let threshold = 0.3 - 0.5 * x[0] - x[1].sin();
        gam_math::probability::normal_cdf(-threshold * (-SLOPE * x[2]).exp())
    });
    let unified = fit(&data, "x3");
    let threshold_width = unified.beta_threshold().len();
    let log_sigma = unified.beta_log_sigma();
    assert_eq!(
        log_sigma.len(),
        2,
        "#3015: the log-σ block carries its intercept and the x3 slope the noise formula names"
    );
    let covariance = unified.beta_covariance().expect("posterior covariance");
    let slope = log_sigma[1];
    let sd = covariance[[threshold_width + 1, threshold_width + 1]].sqrt();
    assert!(
        sd.is_finite() && sd > 0.0,
        "#3015: the x3 log-σ slope has posterior standard deviation {sd}"
    );
    assert!(
        (slope - SLOPE).abs() <= 3.0 * sd && slope > 3.0 * sd,
        "#3015: fitted log-σ slope on x3 {slope} (posterior sd {sd}) against the truth {SLOPE}"
    );
}
