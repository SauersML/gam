//! Regression for #1564 (bug 2): saved Royston-Parmar (`transformation`)
//! survival prediction must not fail at the top of its default time grid.
//!
//! Root cause: the RP baseline `log Λ(t)` is a monotone I-spline cumulative
//! hazard. At the top boundary of the fitted knot span the I-spline saturates
//! (the rightmost retained basis reaches its plateau and the would-be-rising
//! tail columns were dropped as constant over training), so the time-derivative
//! `d(log Λ)/dt` is exactly `0` there — and the instantaneous hazard is `0`
//! (the survival curve is locally flat). The saved predict path drove
//! `royston_parmar_survival_hazard_components`, whose guard required a STRICTLY
//! positive derivative and therefore aborted with `eta_t=0`.
//!
//! The Python prediction surface always evaluates a grid whose top node sits at
//! `max_observed_exit * (1 + 1e-6)` (see `default_survival_time_grid`), i.e.
//! exactly in this saturated regime, so **every** transformation-RP surface
//! prediction failed on its last grid node — the user saw no survival curves at
//! all. The fix relaxes the guard to accept `eta_derivative >= 0` (still
//! rejecting NaN and genuinely-negative / non-monotone slopes) and maps the
//! zero boundary to a zero hazard, matching the probit / marginal-slope guard.
//!
//! This test reproduces the failure on real data: the UCI Heart Failure Clinical
//! Records dataset (Chicco & Jurman, 2020; CC BY 4.0) with the exact
//! multi-smooth formula from the #1564 report. It fits through the real `gam
//! fit` path, loads the saved model, and drives the library predict surface on
//! the default-style grid — the exact code path the Python
//! `model.predict(...).survival_at(grid)` FFI uses. It asserts predict returns
//! `Ok` (the regression), that the surface is finite / monotone / in `[0, 1]`,
//! and — to prove the `eta_t=0` path is genuinely exercised — that at least one
//! grid node carries an exactly-zero hazard atop a finite positive cumulative
//! hazard.

use std::path::Path;
use std::process::Command;

use csv::StringRecord;
use gam::encode_recordswith_inferred_schema;
use gam::families::survival::predict::{SurvivalPredictRequest, predict_survival};
use gam::inference::data::EncodedDataset;
use gam::inference::model::FittedModel;
use gam::test_support::cli_harness::run_or_panic;
use ndarray::Array1;

/// gam-format Royston-Parmar survival fixture derived from the UCI Heart Failure
/// Clinical Records dataset (n=299). Columns mirror the #1564 bug-2 formula.
const HEART_FAILURE_CSV: &str = include_str!("../../fixtures/survival/heart_failure_rp.csv");

const SURVIVAL_FORMULA: &str = "Surv(entry, exit, event) ~ s(age) \
    + s(log_creatinine_phosphokinase) + s(ejection_fraction) + s(log_platelets) \
    + s(log_serum_creatinine) + s(serum_sodium) + linear(anaemia) \
    + linear(diabetes) + linear(high_blood_pressure) + linear(sex) + linear(smoking)";

/// Parse the fixture into (header, rows-of-cells).
fn fixture_records() -> (Vec<String>, Vec<Vec<String>>) {
    let mut reader = csv::Reader::from_reader(HEART_FAILURE_CSV.as_bytes());
    let headers: Vec<String> = reader
        .headers()
        .expect("fixture header")
        .iter()
        .map(|s| s.to_string())
        .collect();
    let rows: Vec<Vec<String>> = reader
        .records()
        .map(|r| r.expect("fixture row").iter().map(|s| s.to_string()).collect())
        .collect();
    (headers, rows)
}

/// Build a small predict frame from the first `k` subjects with a large `exit`
/// placeholder so the surface frame is never the binding constraint on the grid.
fn predict_dataset(headers: &[String], rows: &[Vec<String>], k: usize, big_exit: f64) -> EncodedDataset {
    let exit_idx = headers.iter().position(|h| h == "exit").expect("exit column");
    let event_idx = headers.iter().position(|h| h == "event").expect("event column");
    let records: Vec<StringRecord> = rows
        .iter()
        .take(k)
        .map(|row| {
            let mut cells = row.clone();
            cells[exit_idx] = format!("{big_exit:.6}");
            cells[event_idx] = "1".to_string();
            StringRecord::from(cells)
        })
        .collect();
    encode_recordswith_inferred_schema(headers.to_vec(), records).expect("encode predict rows")
}

#[test]
fn royston_parmar_saved_predict_at_grid_top_does_not_fail() {
    let (headers, rows) = fixture_records();
    let exit_idx = headers.iter().position(|h| h == "exit").expect("exit column");
    let max_exit = rows
        .iter()
        .map(|row| row[exit_idx].parse::<f64>().expect("numeric exit"))
        .fold(f64::MIN, f64::max);
    assert!(max_exit > 0.0, "fixture must have positive exit times");

    let dir = tempfile::tempdir().expect("create tempdir");
    let train_path = dir.path().join("train.csv");
    let model_path = dir.path().join("model.json");
    std::fs::write(&train_path, HEART_FAILURE_CSV).expect("write training fixture");

    // Default survival likelihood is `transformation` (Royston-Parmar) with an
    // I-spline baseline log-cumulative-hazard — exactly the #1564 configuration.
    let mut fit_cmd = Command::new(gam::gam_binary!());
    fit_cmd
        .arg("fit")
        .arg(&train_path)
        .arg(SURVIVAL_FORMULA)
        .arg("--out")
        .arg(&model_path);
    run_or_panic(fit_cmd, "gam fit multi-smooth Royston-Parmar survival");
    assert!(model_path.is_file(), "gam fit did not write {model_path:?}");

    let model = FittedModel::load_from_path(Path::new(&model_path)).expect("load saved RP model");

    // The default prediction grid (`default_survival_time_grid`): 64 linear
    // nodes from 0 to `max_exit * (1 + 1e-6)`. The top node lands in the
    // saturated I-spline regime where `d(log Λ)/dt == 0`.
    let hi = max_exit * (1.0 + 1.0e-6);
    let step = hi / 63.0;
    let grid: Vec<f64> = (0..64).map(|i| step * (i as f64)).collect();

    let dataset = predict_dataset(&headers, &rows, 6, max_exit + 5.0);
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

    // The core regression: predict must NOT abort with `eta_t=0` at the grid top.
    let result = predict_survival(request)
        .expect("RP saved predict must succeed at the default grid top (#1564)");

    assert_eq!(result.survival.nrows(), n, "one survival row per predict row");
    assert_eq!(result.survival.ncols(), grid.len(), "surface covers every grid time");

    let mut zero_hazard_nodes = 0usize;
    for r in 0..n {
        let surv: Vec<f64> = result.survival.row(r).to_vec();
        let haz: Vec<f64> = result.hazard.row(r).to_vec();
        let cum: Vec<f64> = result.cumulative_hazard.row(r).to_vec();

        assert!(
            surv.iter()
                .all(|s| s.is_finite() && (0.0..=1.0).contains(s)),
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

        // Count the exactly-zero-hazard nodes (the `eta_t=0` boundary) that sit
        // atop a finite positive cumulative hazard. Each one is a node that the
        // pre-fix strict `> 0.0` guard would have rejected.
        for j in 0..grid.len() {
            if haz[j] == 0.0 && cum[j] > 1e-9 {
                zero_hazard_nodes += 1;
            }
        }
    }

    assert!(
        zero_hazard_nodes > 0,
        "expected at least one exactly-zero-hazard grid node (the #1564 eta_t=0 \
         boundary); without one this fixture would not guard the regression"
    );
}
