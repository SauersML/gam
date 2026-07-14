//! Persistence acceptance for #2302: a fresh `gam fit` process must save the
//! exact O(n) spline-scan posterior, and fresh `gam predict` / `gam report`
//! processes must consume that state without a dense fit or training design.

use gam::inference::model::FittedModel;
use std::path::Path;
use std::process::{Command, Output};

fn run_success(command: &mut Command, label: &str) -> Output {
    let output = command
        .output()
        .unwrap_or_else(|error| panic!("failed to spawn {label}: {error}"));
    assert!(
        output.status.success(),
        "{label} failed with status {}\n--- stdout ---\n{}\n--- stderr ---\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    output
}

fn write_training_fixture(path: &Path) {
    let mut writer = csv::Writer::from_path(path).expect("create training fixture");
    writer
        .write_record(["x", "y"])
        .expect("write training header");
    for i in 0..48 {
        let x = i as f64 / 47.0;
        let deterministic_noise = 0.025 * ((i % 7) as f64 - 3.0);
        let y = (6.0 * x).sin() + 0.35 * x + deterministic_noise;
        writer
            .write_record([x.to_string(), y.to_string()])
            .expect("write training row");
    }
    writer.flush().expect("flush training fixture");
}

fn write_query_fixture(path: &Path, query: &[f64]) {
    let mut writer = csv::Writer::from_path(path).expect("create query fixture");
    writer.write_record(["x"]).expect("write query header");
    for &x in query {
        writer
            .write_record([x.to_string()])
            .expect("write query row");
    }
    writer.flush().expect("flush query fixture");
}

#[test]
fn fresh_processes_replay_saved_scan_for_predict_and_report() {
    let scratch = tempfile::tempdir().expect("create isolated scratch directory");
    let train_path = scratch.path().join("train.csv");
    let query_path = scratch.path().join("query.csv");
    let model_path = scratch.path().join("scan_model.gam");
    let prediction_path = scratch.path().join("prediction.csv");
    let report_path = scratch.path().join("scan_model.report.html");
    let query = [-0.25, 0.0, 0.375, 0.8, 1.0, 1.25];
    write_training_fixture(&train_path);
    write_query_fixture(&query_path, &query);

    let fit_output = run_success(
        Command::new(env!("CARGO_BIN_EXE_gam"))
            .current_dir(scratch.path())
            .arg("fit")
            .arg(&train_path)
            .arg("y ~ s(x, bs=\"ps\", degree=3, penalty_order=2, double_penalty=False)")
            .arg("--family")
            .arg("gaussian")
            .arg("--out")
            .arg(&model_path),
        "gam fit",
    );
    assert!(
        String::from_utf8_lossy(&fit_output.stdout).contains("spline-scan fit"),
        "fit must visibly select the exact spline-scan route"
    );

    // This is the decisive representation check: predict/report receive neither
    // responses nor a dense design, and the persisted model has no dense fit to
    // fall back to. Their only executable posterior is `SavedSplineScan::state`.
    let saved = FittedModel::load_from_path(&model_path).expect("load fitted scan payload");
    assert!(saved.payload().spline_scan.is_some());
    assert!(saved.payload().fit_result.is_none());
    assert!(saved.payload().unified.is_none());
    assert!(saved.payload().resolved_termspec.is_none());
    let (feature_column, scan) = saved
        .saved_spline_scan()
        .expect("restore persisted scan state")
        .expect("model must carry the scan representation");
    assert_eq!(feature_column, "x");

    run_success(
        Command::new(env!("CARGO_BIN_EXE_gam"))
            .current_dir(scratch.path())
            .arg("predict")
            .arg(&model_path)
            .arg(&query_path)
            .arg("--out")
            .arg(&prediction_path)
            .arg("--uncertainty")
            // #2296: the scan posterior variance is conditional on the
            // profiled smoothing parameter; the default corrected request is
            // an honest typed refusal on scan models.
            .arg("--covariance-mode")
            .arg("conditional")
            .arg("--level")
            .arg("0.9"),
        "gam predict",
    );

    let mut predictions = csv::Reader::from_path(&prediction_path).expect("read prediction CSV");
    assert_eq!(
        predictions
            .headers()
            .expect("prediction header")
            .iter()
            .collect::<Vec<_>>(),
        vec!["eta", "mean", "std_error", "mean_lower", "mean_upper"]
    );
    let rows = predictions
        .records()
        .collect::<Result<Vec<_>, _>>()
        .expect("parse prediction rows");
    assert_eq!(rows.len(), query.len());
    let z = gam::probability::standard_normal_quantile(0.95).expect("90% normal quantile");
    for ((row, &x), row_index) in rows.iter().zip(&query).zip(0..) {
        let (mean, variance) = scan.predict(x).expect("direct saved-scan prediction");
        let se = variance.max(0.0).sqrt();
        let expected = [
            format!("{mean:.12}"),
            format!("{mean:.12}"),
            format!("{se:.12}"),
            format!("{:.12}", mean - z * se),
            format!("{:.12}", mean + z * se),
        ];
        assert_eq!(
            row.iter().collect::<Vec<_>>(),
            expected.iter().map(String::as_str).collect::<Vec<_>>(),
            "fresh-process CLI prediction drifted from the persisted scan at row {row_index} (x={x})"
        );
    }

    // No DATA argument is supplied: a fresh report process has only the model
    // file, so a successful scalar report necessarily comes from the scan state.
    run_success(
        Command::new(env!("CARGO_BIN_EXE_gam"))
            .current_dir(scratch.path())
            .arg("report")
            .arg(&model_path),
        "gam report",
    );
    let report = std::fs::read_to_string(&report_path).expect("read scan report");
    assert!(report.contains("Exact O(n) state-space spline scan for s(x)"));
    assert!(report.contains("no coefficient table is shown"));
    assert!(report.contains(&format!("EDF={:.3}", scan.edf())));
    assert!(report.contains(&format!("knots={}", scan.knots.len())));
}
