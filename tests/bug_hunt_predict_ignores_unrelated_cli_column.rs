//! Regression for #840, CLI angle.
//!
//! `gam predict` loaded the new-data file with the *full* training schema, so a
//! column the formula never references — a label / ID / grouping column kept in
//! the file for bookkeeping — was strict-validated against the training levels.
//! A held-out prediction file whose value for that column never appeared during
//! training aborted predict with
//! `unseen level '…' in categorical column '…'`, the classic
//! leave-one-group-out / bootstrap foot-gun, naming a column the user never put
//! in the formula.
//!
//! The fix projects the prediction frame onto the model's referenced columns
//! (`FittedModel::prediction_required_columns`) before encoding, so unrelated
//! columns are ignored — matching mgcv / glm semantics and the PyFFI surface.
//!
//! This test fits `y ~ s(x)` from a CSV that *also* carries an unrelated
//! categorical `g`, then predicts on a frame whose `g` holds a brand-new level.
//! Before the fix the predict aborts; after it, predict succeeds and `g` does
//! not influence the result.

use gam::test_support::cli_harness::{read_prediction_means, write_predict_csv_rows};
use std::path::Path;
use std::process::Command;

fn write_training_csv(path: &Path) {
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand_distr::{Distribution, Normal};
    let mut rng = StdRng::seed_from_u64(11);
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let mut writer = csv::Writer::from_path(path).expect("create training csv");
    writer.write_record(["x", "y", "g"]).expect("write header");
    // Training carries the unrelated label column `g` with levels b, c, d only.
    let levels = ["b", "c", "d"];
    let n = 300usize;
    for i in 0..n {
        let x = (i as f64) / ((n - 1) as f64);
        let y = (2.0 * std::f64::consts::PI * x).sin() + noise.sample(&mut rng);
        let g = levels[i % levels.len()];
        writer
            .write_record([format!("{x:.10}"), format!("{y:.10}"), g.to_string()])
            .expect("write training row");
    }
    writer.flush().expect("flush training csv");
}

fn fit_model(train_path: &Path, model_path: &Path) {
    let mut fit_cmd = Command::new(gam::gam_binary!());
    fit_cmd
        .arg("fit")
        .arg(train_path)
        .arg("y ~ s(x)")
        .args(["--family", "gaussian"])
        .arg("--out")
        .arg(model_path);
    let fit_out = fit_cmd.output().expect("spawn gam fit");
    assert!(
        fit_out.status.success(),
        "`gam fit 'y ~ s(x)'` failed.\n--- stderr ---\n{}",
        String::from_utf8_lossy(&fit_out.stderr),
    );
    assert!(model_path.is_file(), "gam fit did not write {model_path:?}");
}

fn predict(model_path: &Path, predict_path: &Path, out_path: &Path) -> std::process::Output {
    let mut predict_cmd = Command::new(gam::gam_binary!());
    predict_cmd
        .arg("predict")
        .arg(model_path)
        .arg(predict_path)
        .arg("--out")
        .arg(out_path);
    predict_cmd.output().expect("spawn gam predict")
}

#[test]
fn predict_ignores_unrelated_column_with_unseen_level() {
    let dir = tempfile::tempdir().expect("create tempdir");
    let train_path = dir.path().join("train.csv");
    let model_path = dir.path().join("model.json");

    write_training_csv(&train_path);
    fit_model(&train_path, &model_path);

    // Held-out frame: `g = "a"` is a brand-new level, never seen in training.
    let predict_path = dir.path().join("predict_unseen.csv");
    write_predict_csv_rows(
        &predict_path,
        ["x", "y", "g"],
        [(0.25, "a"), (0.5, "a")]
            .iter()
            .map(|&(x, g)| [format!("{x:.10}"), "0.0".to_string(), g.to_string()]),
    );
    let out_path = dir.path().join("pred_unseen.csv");
    let out = predict(&model_path, &predict_path, &out_path);
    assert!(
        out.status.success(),
        "`gam predict` aborted on an unrelated column's held-out level (#840).\n\
         --- stderr ---\n{}",
        String::from_utf8_lossy(&out.stderr),
    );
    let preds = read_prediction_means(&out_path);
    assert_eq!(preds.len(), 2, "expected one prediction per row");
    assert!(
        preds.iter().all(|p| p.is_finite()),
        "predictions must be finite: {preds:?}"
    );

    // The unrelated column must not influence the prediction: predicting the
    // same x with a *seen* level must match predicting it with the unseen one.
    let seen_path = dir.path().join("predict_seen.csv");
    write_predict_csv_rows(
        &seen_path,
        ["x", "y", "g"],
        [(0.25, "b"), (0.5, "c")]
            .iter()
            .map(|&(x, g)| [format!("{x:.10}"), "0.0".to_string(), g.to_string()]),
    );
    let seen_out_path = dir.path().join("pred_seen.csv");
    let seen_out = predict(&model_path, &seen_path, &seen_out_path);
    assert!(seen_out.status.success(), "predict with seen levels failed");
    let seen_preds = read_prediction_means(&seen_out_path);
    for (a, b) in preds.iter().zip(seen_preds.iter()) {
        assert!(
            (a - b).abs() < 1e-9,
            "an unrelated column changed the prediction: unseen={a}, seen={b}"
        );
    }
}

#[test]
fn predict_still_errors_on_missing_required_column() {
    let dir = tempfile::tempdir().expect("create tempdir");
    let train_path = dir.path().join("train.csv");
    let model_path = dir.path().join("model.json");

    write_training_csv(&train_path);
    fit_model(&train_path, &model_path);

    // A predict frame missing the required predictor `x` must still error — the
    // projection drops unrelated columns, it does not mask genuinely missing
    // ones.
    let bad_path = dir.path().join("predict_missing_x.csv");
    {
        let mut writer = csv::Writer::from_path(&bad_path).expect("create csv");
        writer.write_record(["g", "y"]).expect("header");
        writer.write_record(["b", "0.0"]).expect("row");
        writer.flush().expect("flush");
    }
    let out_path = dir.path().join("pred_missing.csv");
    let out = predict(&model_path, &bad_path, &out_path);
    assert!(
        !out.status.success(),
        "`gam predict` should fail when the required column 'x' is absent"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains('x'),
        "error should mention the missing required column 'x': {stderr}"
    );
}
