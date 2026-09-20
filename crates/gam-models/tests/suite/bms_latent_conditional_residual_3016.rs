//! gam#3016: a saved marginal-slope model returns the declared conditional law's
//! standardized residual `ζ = (z − m(a))/√v(a)` for new rows, through the map
//! the fit applied, so a held-out adequacy check can compare a stratum's law of
//! ζ with the training residual law instead of the raw score's law.
//!
//! The fixture is gam#2768's location-scale score on a raw scale, which m(a)
//! and v(a) absorb (the saved normalisation is the identity under the default
//! latent-z policy):
//!
//! ```text
//!     x, ζ ~ N(0,1) independent,   z = m·x + √(1−m²)·ζ,   z_raw = 3 + 2·z
//! ```
//!
//! so `E[z | x] = m·x` moves with the context, and ζ does not. The survival arm
//! draws event times from the probit transformation model on ζ, as gam#2768's
//! survival fixture does, and reads ζ from a frame that has no time columns.

use csv::StringRecord;
use gam_data::{EncodedDataset, encode_recordswith_inferred_schema};
use gam_linalg::utils::splitmix64;
use gam_math::probability::{normal_cdf, standard_normal_quantile};
use gam_models::fit_orchestration::{FitConfig, FitResult, fit_model, materialize};
use gam_models::inference::model::FittedModel;
use gam_models::inference::model_payload_builders::fit_formula_to_payload;
use ndarray::{Array1, Axis};
use std::collections::HashMap;

const N_TRAIN: usize = 2_000;
const N_HELD_OUT: usize = 2_000;
/// `Corr(z, x)`: the conditional-mean slope the calibration has to remove.
const M_SHIFT: f64 = 0.6;
/// The raw score is `RAW_CENTER + RAW_SCALE·z`, off the standard scale, so the
/// conditional map has to carry its location and scale.
const RAW_CENTER: f64 = 3.0;
const RAW_SCALE: f64 = 2.0;
const TRUE_SLOPE: f64 = 0.6;
const TRUE_BETA_X: f64 = 0.5;
const TRUE_INTERCEPT: f64 = -0.2;
/// Two-sided false-alarm probability of the held-out centring check.
const CENTRING_FALSE_ALARM: f64 = 1.0e-9;
/// The survival arm's `log t` coefficient in the marginal index.
const TRUE_LOG_TIME: f64 = 0.8;

fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(f64::MIN_POSITIVE);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// A uniform draw on the open interval (0, 1).
fn next_open_unit(state: &mut u64) -> f64 {
    ((splitmix64(state) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

struct Sample {
    dataset: EncodedDataset,
    x: Vec<f64>,
}

fn draw(n: usize, seed: u64) -> Sample {
    let headers = ["y", "z", "x"].iter().map(|s| s.to_string()).collect::<Vec<_>>();
    let residual_sd = (1.0 - M_SHIFT * M_SHIFT).sqrt();
    let c_true = (1.0 + TRUE_SLOPE * TRUE_SLOPE).sqrt();
    let mut state = seed;
    let mut rows = Vec::with_capacity(n);
    let mut x = Vec::with_capacity(n);
    for _ in 0..n {
        let xi = next_gauss(&mut state);
        let zeta = next_gauss(&mut state);
        let z = M_SHIFT * xi + residual_sd * zeta;
        let eta = (TRUE_INTERCEPT + TRUE_BETA_X * xi) * c_true + TRUE_SLOPE * zeta;
        let y = u8::from(next_unit(&mut state) < normal_cdf(eta));
        rows.push(StringRecord::from(vec![
            y.to_string(),
            format!("{:.17e}", RAW_CENTER + RAW_SCALE * z),
            format!("{xi:.17e}"),
        ]));
        x.push(xi);
    }
    Sample {
        dataset: encode_recordswith_inferred_schema(headers, rows)
            .expect("encode the #3016 sample"),
        x,
    }
}

fn bernoulli_config(latent_measure: &str) -> FitConfig {
    FitConfig {
        family: Some("bernoulli-marginal-slope".to_string()),
        z_column: Some("z".to_string()),
        slope_formula: Some("1".to_string()),
        latent_measure: Some(latent_measure.to_string()),
        ..FitConfig::default()
    }
}

fn fit_formula(sample: &Sample, formula: &str, latent_measure: &str) -> FittedModel {
    let payload = fit_formula_to_payload(
        formula.to_string(),
        &sample.dataset,
        &bernoulli_config(latent_measure),
    )
    .unwrap_or_else(|e| panic!("bernoulli fit {formula}, law {latent_measure}: {e}"));
    FittedModel::from_payload(payload)
}

fn fit(sample: &Sample, latent_measure: &str) -> FittedModel {
    fit_formula(sample, "y ~ x", latent_measure)
}

/// The score the fit itself trained on, read from the fit result before any
/// payload is built. The request is materialized as the payload builder
/// materializes it, so the two fits see the same design and the same rows.
fn fit_time_training_score(sample: &Sample, latent_measure: &str) -> Array1<f64> {
    let mut config = bernoulli_config(latent_measure);
    config.adaptive_resolution = Some(Vec::new());
    let materialized = materialize("y ~ x", &sample.dataset, &config)
        .unwrap_or_else(|e| panic!("materialize the bernoulli marginal-slope request: {e}"));
    match fit_model(materialized.request) {
        Ok(FitResult::BernoulliMarginalSlope(result)) => result.latent_score,
        Ok(_) => panic!("a bernoulli marginal-slope request returns a marginal-slope fit"),
        Err(e) => panic!("bernoulli marginal-slope fit: {e}"),
    }
}

fn residual(model: &FittedModel, sample: &Sample) -> Option<Array1<f64>> {
    model
        .latent_conditional_residual(sample.dataset.values.view(), &sample.dataset.column_map())
        .expect("the conditional residual of a saved marginal-slope model")
}

/// Ordinary least-squares slope of `values` on `x`, and its standard error.
fn slope_and_standard_error(values: &[f64], x: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let x_mean = x.iter().sum::<f64>() / n;
    let v_mean = values.iter().sum::<f64>() / n;
    let sxx = x.iter().map(|xi| (xi - x_mean).powi(2)).sum::<f64>();
    let sxv = x
        .iter()
        .zip(values)
        .map(|(xi, vi)| (xi - x_mean) * (vi - v_mean))
        .sum::<f64>();
    let slope = sxv / sxx;
    let rss = x
        .iter()
        .zip(values)
        .map(|(xi, vi)| (vi - v_mean - slope * (xi - x_mean)).powi(2))
        .sum::<f64>();
    (slope, (rss / (n - 2.0) / sxx).sqrt())
}

/// At the training rows the saved map reproduces the fit's own ζ moments: the
/// fit recorded the mean and sd of the sample it calibrated (unit weights). Each
/// is a sum over the same n values, so the two agree to the rounding of two
/// recursive sums, `(n−1)·ε·Σ|·|/n`. The survival fit result does not carry its
/// training score, so the survival arm pins the moments; the Bernoulli arm pins
/// the score itself.
fn assert_reproduces_the_fit_moments(model: &FittedModel, zeta_train: &Array1<f64>, label: &str) {
    let calibration = model
        .latent_z_conditional_calibration
        .as_ref()
        .expect("the declared location-scale law fits m(a) on this score");
    let n = zeta_train.len() as f64;
    let mean = zeta_train.sum() / n;
    let mean_band = f64::EPSILON * zeta_train.iter().map(|value| value.abs()).sum::<f64>();
    let sum_sq = zeta_train.iter().map(|value| (value - mean).powi(2)).sum::<f64>();
    let sd = (sum_sq / n).sqrt();
    let sd_band = (n + 2.0) * f64::EPSILON * sum_sq / n / (2.0 * sd);
    eprintln!(
        "[3016] {label} training ζ: mean {mean:.6e} (fit {:.6e}, band {mean_band:.1e}), \
         sd {sd:.12} (fit {:.12}, band {sd_band:.1e})",
        calibration.post_mean, calibration.post_sd
    );
    assert!(
        (mean - calibration.post_mean).abs() <= mean_band,
        "{label}: the training-row ζ mean {mean:.17e} must be the fit's {:.17e}",
        calibration.post_mean
    );
    assert!(
        (sd - calibration.post_sd).abs() <= sd_band,
        "{label}: the training-row ζ sd {sd:.17e} must be the fit's {:.17e}",
        calibration.post_sd
    );
}

/// On new rows ζ is conditionally centred: its slope on x is zero within the
/// held-out slope's own sampling error plus the calibration's (m̂ carries the
/// training sample's). The raw score's slope is 2m.
fn assert_conditionally_centred(
    zeta_held_out: &Array1<f64>,
    held_out: &EncodedDataset,
    x: &[f64],
    label: &str,
) {
    let (slope, held_out_se) =
        slope_and_standard_error(zeta_held_out.as_slice().expect("an owned 1-D array"), x);
    let calibration_se = held_out_se * (N_HELD_OUT as f64 / N_TRAIN as f64).sqrt();
    let critical = -standard_normal_quantile(CENTRING_FALSE_ALARM / 2.0)
        .expect("a finite quantile of a probability in (0, 1)");
    let band = critical * held_out_se.hypot(calibration_se);
    let raw: Vec<f64> = held_out.values.column(held_out.column_map()["z"]).to_vec();
    let (raw_slope, _) = slope_and_standard_error(&raw, x);
    eprintln!(
        "[3016] {label} held-out slope on x: ζ {slope:.4e} (band {band:.3e}), raw score \
         {raw_slope:.4}"
    );
    assert!(
        slope.abs() <= band,
        "{label}: held-out ζ must be conditionally centred: slope on x {slope:.4e} beyond \
         {band:.3e}"
    );
    assert!(
        raw_slope.abs() > band,
        "{label} control: the raw score's slope on x ({raw_slope:.4}) must be outside the same \
         band"
    );
}

#[test]
fn a_saved_model_returns_the_conditional_residual_its_fit_applied_3016() {
    let train = draw(N_TRAIN, 0x3016_0000_5EED_0001);
    let held_out = draw(N_HELD_OUT, 0x3016_0000_5EED_0002);
    let model = fit(&train, "conditional-location-scale");

    // 1. At the training rows ζ is the score the fit trained on, bit for bit:
    // the fit and the saved model run the one fitted score map.
    let zeta_train = residual(&model, &train).expect("a conditional law was consumed");
    let fit_time = fit_time_training_score(&train, "conditional-location-scale");
    let first_difference = zeta_train
        .iter()
        .zip(&fit_time)
        .position(|(saved, fitted)| saved.to_bits() != fitted.to_bits());
    assert_eq!(
        first_difference,
        None,
        "bernoulli: the saved model's training-row ζ must be the fit's own score bit for bit \
         (saved {:?}, fit {:?})",
        first_difference.map(|row| zeta_train[row]),
        first_difference.map(|row| fit_time[row])
    );
    assert_eq!(zeta_train.len(), fit_time.len());

    // 2. A save and load round trip returns the same ζ.
    let bytes = serde_json::to_vec(&model).expect("serialize the model");
    let loaded: FittedModel = serde_json::from_slice(&bytes).expect("parse the saved model");
    let zeta_loaded = residual(&loaded, &held_out).expect("a conditional law was consumed");

    // 3. On new rows ζ is conditionally centred.
    let zeta_held_out = residual(&model, &held_out).expect("a conditional law was consumed");
    assert_conditionally_centred(&zeta_held_out, &held_out.dataset, &held_out.x, "bernoulli");
    assert!(
        zeta_loaded
            .iter()
            .zip(&zeta_held_out)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "bernoulli: the reloaded model must return the saved model's ζ bit for bit"
    );

    // 4. A conditioning design of another width than the one m(a) and v(a) were
    // fitted on is refused rather than read against the wrong columns.
    let mut mismatched = loaded;
    let wider = fit_formula(&train, "y ~ smooth(x)", "conditional-location-scale");
    mismatched.resolved_termspec = wider.resolved_termspec.clone();
    let refusal = mismatched
        .latent_conditional_residual(
            held_out.dataset.values.view(),
            &held_out.dataset.column_map(),
        )
        .expect_err("a conditioning design of the wrong width must be refused");
    assert!(
        refusal.to_string().contains("basis columns"),
        "the refusal names the width mismatch: {refusal}"
    );

    // 5. A fit that consumed no conditional law returns none.
    let global = fit(&train, "global-empirical");
    assert!(
        global.latent_z_conditional_calibration.is_none(),
        "the global-empirical law fits no conditional map"
    );
    assert_eq!(residual(&global, &held_out), None);
}

fn draw_survival(n: usize, seed: u64) -> Sample {
    let headers = ["time", "event", "z", "x"].iter().map(|s| s.to_string()).collect::<Vec<_>>();
    let residual_sd = (1.0 - M_SHIFT * M_SHIFT).sqrt();
    let c_true = (1.0 + TRUE_SLOPE * TRUE_SLOPE).sqrt();
    let mut state = seed;
    let mut rows = Vec::with_capacity(n);
    let mut xs = Vec::with_capacity(n);
    for _ in 0..n {
        let x = next_gauss(&mut state);
        let zeta = next_gauss(&mut state);
        let z = M_SHIFT * x + residual_sd * zeta;
        xs.push(x);
        // Invert `Φ⁻¹(F(t | x, ζ)) = q(t, x)·c + b·ζ` at a uniform draw.
        let eta = standard_normal_quantile(next_open_unit(&mut state))
            .expect("a finite quantile of a probability in (0, 1)");
        let log_t =
            ((eta - TRUE_SLOPE * zeta) / c_true - TRUE_INTERCEPT - TRUE_BETA_X * x) / TRUE_LOG_TIME;
        let event_time = log_t.exp();
        let censor_time = 1.5 + 3.0 * next_unit(&mut state);
        let (time, event) = if event_time <= censor_time {
            (event_time, 1u8)
        } else {
            (censor_time, 0u8)
        };
        rows.push(StringRecord::from(vec![
            format!("{:.17e}", time.max(f64::EPSILON)),
            event.to_string(),
            format!("{:.17e}", RAW_CENTER + RAW_SCALE * z),
            format!("{x:.17e}"),
        ]));
    }
    Sample {
        dataset: encode_recordswith_inferred_schema(headers, rows)
            .expect("encode the #3016 survival sample"),
        x: xs,
    }
}

/// ζ from a frame that carries only the score and the covariate.
fn survival_residual(model: &FittedModel, sample: &Sample) -> Array1<f64> {
    let columns = sample.dataset.column_map();
    let covariates = sample.dataset.values.select(Axis(1), &[columns["z"], columns["x"]]);
    let covariate_map: HashMap<String, usize> =
        [("z".to_string(), 0), ("x".to_string(), 1)].into_iter().collect();
    model
        .latent_conditional_residual(covariates.view(), &covariate_map)
        .expect("the conditional residual of a saved survival marginal-slope model")
        .expect("a conditional law was consumed")
}

#[test]
fn a_saved_survival_model_returns_the_conditional_residual_without_time_columns_3016() {
    let train = draw_survival(N_TRAIN, 0x3016_0000_5EED_0003);
    let held_out = draw_survival(N_HELD_OUT, 0x3016_0000_5EED_0004);
    let config = FitConfig {
        survival_likelihood: Some("marginal-slope".to_string()),
        z_column: Some("z".to_string()),
        slope_formula: Some("smooth(x)".to_string()),
        baseline_target: "linear".to_string(),
        latent_measure: Some("conditional-location-scale".to_string()),
        ..FitConfig::default()
    };
    let payload = fit_formula_to_payload(
        "Surv(time, event) ~ smooth(x)".to_string(),
        &train.dataset,
        &config,
    )
    .unwrap_or_else(|e| panic!("survival marginal-slope fit: {e}"));
    let model = FittedModel::from_payload(payload);

    // The survival predictor conditions on the trailing covariate block of its
    // q-design, which is the design of the covariate term spec alone, so ζ
    // needs neither the time nor the event column.
    assert_reproduces_the_fit_moments(&model, &survival_residual(&model, &train), "survival");
    assert_conditionally_centred(
        &survival_residual(&model, &held_out),
        &held_out.dataset,
        &held_out.x,
        "survival",
    );
}
