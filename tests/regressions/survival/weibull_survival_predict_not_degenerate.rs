//! Regression for #897: single-cause `survival_likelihood = "weibull"` must not
//! predict the degenerate survival surface `S(t) ≡ 1` (cumulative hazard ≈ 0).
//!
//! Root cause (fixed in `src/families/survival_predict.rs`): the Weibull
//! single-cause fit carries its ENTIRE log-cumulative-hazard baseline in the
//! anchor-CENTERED `[1, log t]` linear time-basis coefficients and uses a
//! Linear (zero) parametric baseline offset. The fitted baseline is exactly
//! `Σ_k (b_k(t) − anchor_k)·β_time_k` (see `quality_vs_flexsurv_weibull_aft`,
//! which reconstructs the same quantity). The saved model also records a
//! `Weibull` baseline target (recovered scale/shape) purely for CIF/reporting.
//!
//! `predict_survival` previously rebuilt the time basis UN-centered AND re-added
//! the saved `Weibull` target as a parametric offset, so the baseline was
//! applied twice and against the wrong basis. The eta collapsed to a large
//! negative value, driving `H ≈ 0` and `S = exp(−H) ≈ 1` across the whole grid.
//!
//! This test fits high-mortality data through the real `gam fit` Weibull path,
//! loads the saved model, and drives the library predict surface
//! (`gam::families::survival::predict::predict_survival`) on an explicit time grid — the
//! exact code path the Python `model.predict(...).survival_at(grid)` FFI uses.
//! It asserts the predicted survival is finite, non-degenerate (`min S < 0.85`),
//! and monotone non-increasing in `t`. With the double-counting removed the
//! surface tracks the fitted Weibull and the assertions hold.

use std::path::Path;
use std::process::Command;

use csv::StringRecord;
use gam::encode_recordswith_inferred_schema;
use gam::families::survival::predict::{SurvivalPredictRequest, predict_survival};
use gam::inference::data::EncodedDataset;
use gam::inference::model::FittedModel;
use gam::test_support::cli_harness::run_or_panic;
use ndarray::Array1;

const N: usize = 500;

/// Deterministic right-censored data with high mortality (~87% events, median
/// observed time ~2.5), so any correctly-fitted survival model must predict
/// survival well below 1 inside the observed range. Mirrors the failing Python
/// spec `test_bug_hunt_weibull_survival_predict_degenerate_unit_survival.py`,
/// using a fixed pseudo-random generator so the fixture is reproducible without
/// pulling in an RNG dependency mismatch.
fn build_dataset() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // Simple deterministic LCG -> uniform(0,1); ample for a fixture.
    let mut state: u64 = 0x9E3779B97F4A7C15;
    let mut next_u01 = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };

    let shape = 1.5_f64;
    let mut age = Vec::with_capacity(N);
    let mut exit = Vec::with_capacity(N);
    let mut event = Vec::with_capacity(N);
    for _ in 0..N {
        let a = 40.0 + 35.0 * next_u01();
        let eta = -2.0 + 0.05 * (a - 55.0);
        let u = 1e-9_f64.max(next_u01());
        let t_lat = (-eta / shape).exp() * (-u.ln()).powf(1.0 / shape);
        let cens = (-next_u01().max(1e-12).ln() * 20.0).min(30.0);
        let ex = t_lat.min(cens);
        let ev = if t_lat <= cens { 1.0 } else { 0.0 };
        age.push(a);
        exit.push(ex);
        event.push(ev);
    }
    (age, exit, event)
}

fn write_training_csv(path: &Path, age: &[f64], exit: &[f64], event: &[f64]) {
    let mut writer = csv::Writer::from_path(path).expect("create training csv");
    writer
        .write_record(["entry", "exit", "event", "age"])
        .expect("write header");
    for i in 0..age.len() {
        writer
            .write_record([
                "0.0".to_string(),
                format!("{:.12}", exit[i]),
                format!("{}", event[i] as i64),
                format!("{:.12}", age[i]),
            ])
            .expect("write training row");
    }
    writer.flush().expect("flush training csv");
}

/// One predict row with a LARGE `exit` placeholder so every query time is inside
/// the surface frame (isolates the Weibull degeneracy from the grid-truncation
/// path) and a fixed `age` covariate.
fn predict_dataset(big_exit: f64) -> EncodedDataset {
    let headers = vec![
        "entry".to_string(),
        "exit".to_string(),
        "event".to_string(),
        "age".to_string(),
    ];
    let rows = vec![StringRecord::from(vec![
        "0.0".to_string(),
        format!("{big_exit:.12}"),
        "1".to_string(),
        "57.0".to_string(),
    ])];
    encode_recordswith_inferred_schema(headers, rows).expect("encode predict row")
}

#[test]
fn weibull_survival_predict_surface_is_not_degenerate_unit_survival() {
    let (age, exit, event) = build_dataset();
    let event_rate = event.iter().sum::<f64>() / event.len() as f64;
    assert!(
        event_rate > 0.5,
        "fixture must have substantial mortality, got event rate {event_rate}"
    );

    let dir = tempfile::tempdir().expect("create tempdir");
    let train_path = dir.path().join("train.csv");
    let model_path = dir.path().join("model.json");
    write_training_csv(&train_path, &age, &exit, &event);

    let mut fit_cmd = Command::new(gam::gam_binary!());
    fit_cmd
        .arg("fit")
        .arg(&train_path)
        .arg("Surv(entry, exit, event) ~ s(age)")
        .args(["--survival-likelihood", "weibull"])
        .arg("--out")
        .arg(&model_path);
    run_or_panic(fit_cmd, "gam fit Surv ~ s(age) (weibull)");
    assert!(model_path.is_file(), "gam fit did not write {model_path:?}");

    let model =
        FittedModel::load_from_path(&model_path).expect("load saved Weibull survival model");

    let big_exit = exit.iter().cloned().fold(f64::MIN, f64::max) + 5.0;
    let dataset = predict_dataset(big_exit);
    let col_map = dataset.column_map();
    let payload = model.payload();
    let training_headers = payload.training_headers.as_ref();
    let n = dataset.values.nrows();
    let primary_offset = Array1::<f64>::zeros(n);
    let noise_offset = Array1::<f64>::zeros(n);

    // All query times are inside the observed range; the model has a
    // well-defined survival surface here.
    let grid = [1.0_f64, 3.0, 6.0, 12.0];
    let request = SurvivalPredictRequest {
        model: &model,
        data: dataset.values.view(),
        col_map: &col_map,
        training_headers,
        primary_offset: &primary_offset,
        noise_offset: &noise_offset,
        time_grid: Some(&grid),
        with_uncertainty: false,
        estimand: gam::families::survival::predict::SurvivalPredictEstimand::Plugin,
    };
    let result = predict_survival(request).expect("library Weibull survival predict");

    assert_eq!(result.survival.nrows(), 1, "expected one prediction row");
    assert_eq!(
        result.survival.ncols(),
        grid.len(),
        "survival surface must cover every grid time"
    );

    let surv: Vec<f64> = result.survival.row(0).to_vec();
    assert!(
        surv.iter().all(|s| s.is_finite()),
        "Weibull predicted survival must be finite: {surv:?}"
    );

    let min_surv = surv.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        min_surv < 0.85,
        "Weibull single-cause prediction collapsed to a degenerate unit survival \
         surface: S(age=57) at {grid:?} = {surv:?} (cumulative hazard ~ 0). The \
         baseline must come from the anchor-centered linear time coefficients, \
         not a double-counted parametric offset (#897)."
    );

    for w in surv.windows(2) {
        assert!(
            w[1] <= w[0] + 1e-9,
            "Weibull survival surface must be monotone non-increasing in t: {surv:?}"
        );
    }
}
