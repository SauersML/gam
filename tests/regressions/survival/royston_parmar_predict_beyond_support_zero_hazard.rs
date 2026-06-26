//! Regression for #1564 (bug 2): saved Royston-Parmar (`transformation`)
//! survival prediction must not fail when the prediction grid extends past the
//! training support.
//!
//! Root cause: the RP baseline `log Λ(t)` is a monotone I-spline cumulative
//! hazard. Beyond its last interior knot every I-spline basis is flat, so the
//! time-derivative `d(log Λ)/dt` is exactly `0` and the instantaneous hazard
//! there is `0` (the survival curve is locally constant — the model has no
//! information past its support and accumulates no further hazard). The saved
//! predict path drove `royston_parmar_survival_hazard_components`, whose guard
//! required a STRICTLY positive derivative and therefore aborted with
//! `eta_t=0` on every grid node past the support. A multi-smooth clinical fit
//! evaluated on a fixed 25-point grid hits this on its tail nodes, so the whole
//! prediction failed.
//!
//! The fix relaxes the guard to accept `eta_derivative >= 0` (still rejecting
//! NaN and genuinely-negative / non-monotone slopes) and maps the zero-boundary
//! to a zero hazard, matching the probit / marginal-slope sibling guard.
//!
//! This test fits the default `transformation` (Royston-Parmar) likelihood
//! through the real `gam fit` path, loads the saved model, and drives the
//! library predict surface on a grid that deliberately extends to 3× the
//! largest observed time — the exact code path the Python
//! `model.predict(...).survival_at(grid)` FFI uses. It asserts the surface is
//! finite, monotone, in `[0, 1]`, that the tail (beyond support) is flat with
//! exactly-zero hazard, and — critically — that predict returns `Ok` at all.

use std::path::Path;
use std::process::Command;

use csv::StringRecord;
use gam::encode_recordswith_inferred_schema;
use gam::families::survival::predict::{SurvivalPredictRequest, predict_survival};
use gam::inference::data::EncodedDataset;
use gam::inference::model::FittedModel;
use gam::test_support::cli_harness::run_or_panic;
use ndarray::Array1;

const N: usize = 220;

/// Deterministic right-censored Weibull-latent data with two covariates and a
/// moderate event rate (~0.35), echoing the clinical fits in the #1564 report.
fn build_dataset() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut state: u64 = 0xD1B54A32D192ED03;
    let mut next_u01 = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };

    let shape = 1.3_f64;
    let mut age = Vec::with_capacity(N);
    let mut x1 = Vec::with_capacity(N);
    let mut exit = Vec::with_capacity(N);
    let mut event = Vec::with_capacity(N);
    for _ in 0..N {
        let a = 40.0 + 30.0 * next_u01();
        let cov = 2.0 * next_u01() - 1.0;
        let eta = -3.0 + 0.04 * (a - 55.0) + 0.6 * cov;
        let u = 1e-9_f64.max(next_u01());
        let t_lat = (-eta / shape).exp() * (-u.ln()).powf(1.0 / shape);
        let cens = -next_u01().max(1e-12).ln() * 9.0;
        let ex = t_lat.min(cens).max(1e-3);
        let ev = if t_lat <= cens { 1.0 } else { 0.0 };
        age.push(a);
        x1.push(cov);
        exit.push(ex);
        event.push(ev);
    }
    (age, x1, exit, event)
}

fn write_training_csv(path: &Path, age: &[f64], x1: &[f64], exit: &[f64], event: &[f64]) {
    let mut writer = csv::Writer::from_path(path).expect("create training csv");
    writer
        .write_record(["entry", "exit", "event", "age", "x1"])
        .expect("write header");
    for i in 0..age.len() {
        writer
            .write_record([
                "0.0".to_string(),
                format!("{:.12}", exit[i]),
                format!("{}", event[i] as i64),
                format!("{:.12}", age[i]),
                format!("{:.12}", x1[i]),
            ])
            .expect("write training row");
    }
    writer.flush().expect("flush training csv");
}

/// Predict rows with a large `exit` placeholder so every grid time stays inside
/// the surface frame (isolating the beyond-support flat-hazard path from any
/// grid-truncation behavior).
fn predict_dataset(big_exit: f64) -> EncodedDataset {
    let headers = vec![
        "entry".to_string(),
        "exit".to_string(),
        "event".to_string(),
        "age".to_string(),
        "x1".to_string(),
    ];
    let rows = vec![
        StringRecord::from(vec![
            "0.0".to_string(),
            format!("{big_exit:.12}"),
            "1".to_string(),
            "57.0".to_string(),
            "0.4".to_string(),
        ]),
        StringRecord::from(vec![
            "0.0".to_string(),
            format!("{big_exit:.12}"),
            "1".to_string(),
            "63.0".to_string(),
            "-0.5".to_string(),
        ]),
    ];
    encode_recordswith_inferred_schema(headers, rows).expect("encode predict rows")
}

#[test]
fn royston_parmar_saved_predict_beyond_support_does_not_fail() {
    let (age, x1, exit, event) = build_dataset();
    let event_rate = event.iter().sum::<f64>() / event.len() as f64;
    assert!(
        (0.2..0.7).contains(&event_rate),
        "fixture must have a moderate event rate, got {event_rate}"
    );

    let dir = tempfile::tempdir().expect("create tempdir");
    let train_path = dir.path().join("train.csv");
    let model_path = dir.path().join("model.json");
    write_training_csv(&train_path, &age, &x1, &exit, &event);

    // Default survival likelihood is `transformation` (Royston-Parmar) with an
    // I-spline baseline log-cumulative-hazard — exactly the configuration the
    // #1564 report fit.
    let mut fit_cmd = Command::new(gam::gam_binary!());
    fit_cmd
        .arg("fit")
        .arg(&train_path)
        .arg("Surv(entry, exit, event) ~ s(age) + s(x1)")
        .arg("--out")
        .arg(&model_path);
    run_or_panic(fit_cmd, "gam fit Surv ~ s(age)+s(x1) (transformation)");
    assert!(model_path.is_file(), "gam fit did not write {model_path:?}");

    let model = FittedModel::load_from_path(&model_path).expect("load saved RP survival model");

    let max_exit = exit.iter().cloned().fold(f64::MIN, f64::max);
    // The final nodes sit at 2× and 3× the largest observed time, i.e. well past
    // the I-spline support, where the log-cumulative-hazard is flat.
    let grid = [
        0.5,
        2.0,
        max_exit * 0.5,
        max_exit,
        max_exit * 2.0,
        max_exit * 3.0,
    ];
    let big_exit = max_exit * 3.0 + 5.0;
    let dataset = predict_dataset(big_exit);
    let col_map = dataset.column_map();
    let payload = model.payload();
    let training_headers = payload.training_headers.as_ref();
    let n = dataset.values.nrows();
    let primary_offset = Array1::<f64>::zeros(n);
    let noise_offset = Array1::<f64>::zeros(n);

    let request = SurvivalPredictRequest {
        model: &model,
        data: dataset.values.view(),
        col_map: &col_map,
        training_headers,
        primary_offset: &primary_offset,
        noise_offset: &noise_offset,
        time_grid: Some(&grid),
        with_uncertainty: false,
    };

    // The core assertion: predict must NOT error on the beyond-support tail.
    let result = predict_survival(request)
        .expect("RP saved predict must succeed past the training support (#1564)");

    assert_eq!(result.survival.nrows(), n, "one survival row per predict row");
    assert_eq!(
        result.survival.ncols(),
        grid.len(),
        "survival surface must cover every grid time"
    );

    for r in 0..n {
        let surv: Vec<f64> = result.survival.row(r).to_vec();
        let haz: Vec<f64> = result.hazard.row(r).to_vec();
        let cum: Vec<f64> = result.cumulative_hazard.row(r).to_vec();

        assert!(
            surv.iter().all(|s| s.is_finite() && (0.0..=1.0).contains(s)),
            "survival must be finite and in [0,1]: {surv:?}"
        );
        assert!(
            haz.iter().all(|h| h.is_finite() && *h >= 0.0),
            "hazard must be finite and non-negative: {haz:?}"
        );
        assert!(
            cum.iter().all(|c| c.is_finite() && *c >= 0.0),
            "cumulative hazard must be finite and non-negative here: {cum:?}"
        );
        for w in surv.windows(2) {
            assert!(
                w[1] <= w[0] + 1e-9,
                "survival must be monotone non-increasing in t: {surv:?}"
            );
        }

        // The two tail nodes (2× and 3× max observed time) are past the
        // I-spline support: the cumulative hazard is flat, so the instantaneous
        // hazard is exactly zero and the survival probability stops decreasing.
        let last = grid.len() - 1;
        assert_eq!(
            haz[last], 0.0,
            "hazard beyond support must be exactly 0 (flat I-spline), got {}",
            haz[last]
        );
        assert!(
            (cum[last] - cum[last - 1]).abs() <= 1e-9,
            "cumulative hazard must be flat across the beyond-support tail: \
             {} vs {}",
            cum[last - 1],
            cum[last]
        );
        assert!(
            (surv[last] - surv[last - 1]).abs() <= 1e-9,
            "survival must be flat across the beyond-support tail: {} vs {}",
            surv[last - 1],
            surv[last]
        );
    }
}
