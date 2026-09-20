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
//! log-σ is linear in the shared column and requires the fit to recover that slope. A
//! fourth pins #3879: the binomial log-σ level is the exact scale gauge of
//! `q = −η_t·e^{−η_σ}` and is fixed structurally, so refitting with the scale covariate
//! in other units is an exact reparametrization and must give the same fit.

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
        1,
        "#3015: the log-σ block carries only the x3 slope the noise formula names; its level \
         is the scale gauge of q = -threshold/σ and is fixed at σ = 1 structurally (#3879)"
    );
    let covariance = unified.beta_covariance().expect("posterior covariance");
    let slope = log_sigma[0];
    let sd = covariance[[threshold_width, threshold_width]].sqrt();
    assert!(
        sd.is_finite() && sd > 0.0,
        "#3015: the x3 log-σ slope has posterior standard deviation {sd}"
    );
    assert!(
        (slope - SLOPE).abs() <= 3.0 * sd && slope > 3.0 * sd,
        "#3015: fitted log-σ slope on x3 {slope} (posterior sd {sd}) against the truth {SLOPE}"
    );
}

/// gam#3879: the binomial log-σ gauge used to be closed by a full-span identity ridge on
/// the log-σ coefficients, whose strength depends on the units of every log-σ covariate:
/// the same model fitted with `x3` and with `1000·x3` saw a different prior on the same
/// scale function. The gauge is now fixed structurally (no log-σ intercept), and the
/// remaining log-σ penalty is the formula-native linear-term penalty normalized by the
/// column's mean square, so `x3 ↦ 1000·x3` is an exact reparametrization of the
/// penalized criterion: the slope on `x3` must be exactly 1000 times the slope on
/// `1000·x3`, and the threshold must be unchanged. The two optima can differ only by the
/// solver's convergence error, so the comparison is made in units of the posterior
/// standard deviation the fit reports, at a hundredth of it: any difference in what the
/// model says about the data is excluded, while a unit-dependent prior, which moves the
/// estimate by an O(1) fraction of its sd, is not.
#[test]
fn the_binomial_scale_fit_does_not_depend_on_the_scale_covariates_units_3879() {
    const SLOPE: f64 = 0.3;
    const UNITS: f64 = 1000.0;
    let mut state = 3879u64;
    let headers = ["y", "x1", "x3", "x3k"].map(String::from).to_vec();
    let records = (0..ROWS)
        .map(|_| {
            let x1 = normal(&mut state);
            let x3 = normal(&mut state);
            let threshold = 0.3 - 0.8 * x1;
            let probability = gam_math::probability::normal_cdf(-threshold * (-SLOPE * x3).exp());
            let y = if uniform(&mut state) < probability { 1.0 } else { 0.0 };
            csv::StringRecord::from(vec![
                format!("{y}"),
                format!("{x1:.17e}"),
                format!("{x3:.17e}"),
                format!("{:.17e}", UNITS * x3),
            ])
        })
        .collect();
    let data = gam_data::encode_recordswith_inferred_schema(headers, records)
        .expect("encode the #3879 table");
    let fit_with = |noise: &str| {
        let config = FitConfig {
            family: Some("binomial".to_string()),
            link: Some("probit".to_string()),
            noise_formula: Some(noise.to_string()),
            ..FitConfig::default()
        };
        let result = fit_from_formula("y ~ s(x1)", &data, &config).unwrap_or_else(|error| {
            panic!("#3879: the binomial location-scale fit with noise_formula={noise} must fit: {error}")
        });
        let FitResult::BinomialLocationScale(fitted) = result else {
            panic!("#3879: noise_formula={noise} must return a binomial location-scale fit");
        };
        let unified = fitted.fit.fit;
        assert!(
            unified.convergence_evidence().inner_status().is_converged(),
            "#3879: noise_formula={noise}: the inner mode is not converged"
        );
        unified
    };
    let natural = fit_with("x3");
    let rescaled = fit_with("x3k");
    assert_eq!(
        natural.beta_log_sigma().len(),
        1,
        "#3879: the binomial log-σ block must carry the slope alone, with no gauge level"
    );
    assert_eq!(rescaled.beta_log_sigma().len(), 1);
    let width = natural.beta_threshold().len();
    assert_eq!(rescaled.beta_threshold().len(), width);
    let covariance = natural.beta_covariance().expect("posterior covariance");
    let sd = |index: usize| covariance[[index, index]].sqrt();

    let slope = natural.beta_log_sigma()[0];
    let rescaled_slope = UNITS * rescaled.beta_log_sigma()[0];
    assert!(
        (slope - rescaled_slope).abs() <= 1e-2 * sd(width),
        "#3879: log-σ slope on x3 is {slope} but {UNITS}× the slope on {UNITS}·x3 is \
         {rescaled_slope} (posterior sd {})",
        sd(width)
    );
    for (j, (a, b)) in natural
        .beta_threshold()
        .iter()
        .zip(rescaled.beta_threshold().iter())
        .enumerate()
    {
        assert!(
            (a - b).abs() <= 1e-2 * sd(j),
            "#3879: threshold coefficient {j} is {a} with x3 and {b} with {UNITS}·x3 \
             (posterior sd {})",
            sd(j)
        );
    }
}
