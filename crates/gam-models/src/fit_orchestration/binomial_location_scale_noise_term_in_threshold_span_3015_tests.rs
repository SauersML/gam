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
    table_with_x4_unit(rows, 1.0, probability)
}

/// [`table`] with the `x4` column recorded in units `x4_unit` times smaller: the
/// response is drawn from the standard-normal `x4`, and the column holds
/// `x4_unit·x4`.
fn table_with_x4_unit(
    rows: usize,
    x4_unit: f64,
    probability: impl Fn(&[f64]) -> f64,
) -> gam_data::EncodedDataset {
    let mut state = 3015u64;
    let headers = ["y", "x1", "x2", "x3", "x4"].map(String::from).to_vec();
    let records = (0..rows)
        .map(|_| {
            let x: Vec<f64> = (0..4).map(|_| normal(&mut state)).collect();
            let y = if uniform(&mut state) < probability(&x) { 1.0 } else { 0.0 };
            let mut row = vec![format!("{y}")];
            row.extend(x.iter().enumerate().map(|(j, v)| {
                let recorded = if j == 3 { x4_unit * v } else { *v };
                format!("{recorded:.17e}")
            }));
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
        "#3015: the log-σ block carries the x3 slope the noise formula names and no \
         intercept (σ = 1 at the reference pins the q = −η_t·e^{{−η_σ}} gauge, #3879)"
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

/// gam#3879: the binomial location-scale fit must not depend on the units of a
/// log-σ covariate. Recording `x4` as `1000·x4` is an exact reparametrization of the
/// linear log-σ block (`γ·x4 = (γ/1000)·(1000·x4)`), and the likelihood, the
/// threshold penalty and the REML criterion (whose log-determinant moves by a
/// ρ-independent constant) are all invariant under it, so both fits share one optimum:
/// the slope scales by exactly 1/1000 and every threshold coefficient is unchanged. A
/// full-span identity ridge on the raw log-σ coefficients charged the same σ(x) as
/// `λγ²` in one unit and `λ(γ/1000)²` in the other, and so fitted two different
/// models. The two solves differ only by their convergence tolerances, so they must
/// agree to a small fraction of each coefficient's own posterior standard deviation:
/// a difference far below the statistical resolution of the fit.
#[test]
fn the_fit_is_invariant_to_the_units_of_a_log_sigma_covariate_3879() {
    const UNIT: f64 = 1000.0;
    const SLOPE: f64 = 0.4;
    let probability = |x: &[f64]| {
        let threshold = 0.3 - 0.5 * x[0] - x[1].sin();
        gam_math::probability::normal_cdf(-threshold * (-SLOPE * x[3]).exp())
    };
    let native = fit(&table_with_x4_unit(2_000, 1.0, probability), "x4");
    let scaled = fit(&table_with_x4_unit(2_000, UNIT, probability), "x4");

    let width = native.beta_flat().len();
    assert_eq!(scaled.beta_flat().len(), width, "#3879: both fits carry the same layout");
    let threshold_width = native.beta_threshold().len();
    assert_eq!(
        native.beta_log_sigma().len(),
        1,
        "#3879: the log-σ block is the x4 slope alone, with no gauge intercept"
    );
    let native_cov = native.beta_covariance().expect("native posterior covariance");
    let scaled_cov = scaled.beta_covariance().expect("scaled posterior covariance");

    // Coefficient j of the native chart equals `unit_j` times coefficient j of the
    // scaled chart: 1 for every threshold coefficient, UNIT for the x4 slope.
    let native_beta = native.beta_flat();
    let scaled_beta = scaled.beta_flat();
    for j in 0..width {
        let unit = if j < threshold_width { 1.0 } else { UNIT };
        let sd = native_cov[[j, j]].sqrt();
        assert!(sd.is_finite() && sd > 0.0, "#3879: coefficient {j} has posterior sd {sd}");
        let gap = (native_beta[j] - unit * scaled_beta[j]).abs();
        assert!(
            gap <= 1e-2 * sd,
            "#3879: coefficient {j} moved with the units of x4: native {} vs rescaled {} \
             (gap {gap:.3e}, posterior sd {sd:.3e})",
            native_beta[j],
            unit * scaled_beta[j]
        );
        let scaled_sd = unit * scaled_cov[[j, j]].sqrt();
        assert!(
            (scaled_sd - sd).abs() <= 1e-2 * sd,
            "#3879: the posterior sd of coefficient {j} moved with the units of x4: \
             native {sd:.6e} vs rescaled {scaled_sd:.6e}"
        );
    }
    let slope = native.beta_log_sigma()[0];
    let slope_sd = native_cov[[threshold_width, threshold_width]].sqrt();
    assert!(
        (slope - SLOPE).abs() <= 3.0 * slope_sd,
        "#3879: fitted log-σ slope on x4 {slope} (posterior sd {slope_sd}) against the truth \
         {SLOPE}"
    );
}
