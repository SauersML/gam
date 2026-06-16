//! Regression for #807, numeric-`by` angle.
//!
//! The companion test `bug_hunt_by_factor_smooth_column_not_loaded_from_cli`
//! covers a *factor* `by=` smooth. The same root cause — `collect_term_column_names`
//! / `parsed_terms_reference_column` reading only a smooth's positional `vars`
//! and ignoring `options["by"]` — equally broke a **numeric** (continuous)
//! varying-coefficient smooth `s(x, by=z)`, where the by-variable scales the
//! smooth rather than splitting it into per-level curves. This test exercises
//! that distinct surface from the CLI end to end: `s(x, by=z)` must load `z`,
//! fit, and recover the varying coefficient.
//!
//! Truth: `y = z · sin(2πx)`. At `x = 0.25`, `sin(2π·0.25) = +1`, so the model's
//! prediction collapses to the supplied `z`. We probe `z = +1` and `z = −1` and
//! assert the predictions track `z` (positive for `z=+1`, negative for `z=−1`,
//! both near ±1). Before the fix the fit aborts with
//! `column 'z' not found in data. Available columns: [x, y]`.

use gam::test_support::cli_harness::{read_prediction_means, write_predict_csv_rows};
use std::path::Path;
use std::process::Command;

fn write_training_csv(path: &Path) {
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand_distr::{Distribution, Normal, Uniform};
    let mut rng = StdRng::seed_from_u64(3);
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let zdist = Uniform::new(-2.0, 2.0).expect("uniform");
    let mut writer = csv::Writer::from_path(path).expect("create training csv");
    writer.write_record(["x", "y", "z"]).expect("write header");
    let n = 400usize;
    for i in 0..n {
        let x = (i as f64) / ((n - 1) as f64);
        let z = zdist.sample(&mut rng);
        let y = z * (2.0 * std::f64::consts::PI * x).sin() + noise.sample(&mut rng);
        writer
            .write_record([format!("{x:.10}"), format!("{y:.10}"), format!("{z:.10}")])
            .expect("write training row");
    }
    writer.flush().expect("flush training csv");
}

#[test]
fn numeric_by_smooth_loads_its_by_column_and_recovers_varying_coefficient() {
    let dir = tempfile::tempdir().expect("create tempdir");
    let train_path = dir.path().join("train.csv");
    let predict_path = dir.path().join("predict.csv");
    let model_path = dir.path().join("model.json");
    let out_path = dir.path().join("pred_out.csv");

    write_training_csv(&train_path);

    // Fit the numeric varying-coefficient smooth straight from the CLI, with `z`
    // referenced ONLY through `by=z` (no redundant `+ z`).
    let mut fit_cmd = Command::new(gam::gam_binary!());
    fit_cmd
        .arg("fit")
        .arg(&train_path)
        .arg("y ~ s(x, by=z)")
        .args(["--family", "gaussian"])
        .arg("--out")
        .arg(&model_path);
    let fit_out = fit_cmd.output().expect("spawn gam fit");
    assert!(
        fit_out.status.success(),
        "`gam fit 'y ~ s(x, by=z)'` failed — the numeric by= column was not loaded.\n\
         --- stdout ---\n{}\n--- stderr ---\n{}",
        String::from_utf8_lossy(&fit_out.stdout),
        String::from_utf8_lossy(&fit_out.stderr),
    );
    assert!(model_path.is_file(), "gam fit did not write {model_path:?}");

    // x = 0.25: sin(2π·0.25) = +1, so prediction ≈ z. Probe z = +1 and z = −1.
    let probes: [(f64, f64); 2] = [(0.25, 1.0), (0.25, -1.0)];
    write_predict_csv_rows(
        &predict_path,
        ["x", "y", "z"],
        probes
            .iter()
            .map(|&(x, z)| [format!("{x:.10}"), "0.0".to_string(), format!("{z:.10}")]),
    );

    let mut predict_cmd = Command::new(gam::gam_binary!());
    predict_cmd
        .arg("predict")
        .arg(&model_path)
        .arg(&predict_path)
        .arg("--out")
        .arg(&out_path);
    let pred_out = predict_cmd.output().expect("spawn gam predict");
    assert!(
        pred_out.status.success(),
        "`gam predict` failed for the numeric by-smooth model.\n--- stderr ---\n{}",
        String::from_utf8_lossy(&pred_out.stderr),
    );

    let preds = read_prediction_means(&out_path);
    assert_eq!(preds.len(), 2, "expected one prediction per probe");
    let (pred_pos, pred_neg) = (preds[0], preds[1]);

    // The varying coefficient is f(0.25) = sin(2π·0.25) = +1, so prediction
    // tracks the supplied z: positive for z=+1, negative for z=−1, both near ±1.
    assert!(
        pred_pos > 0.5,
        "z=+1 at x=0.25 should recover z·sin(2π·0.25)=+1, got {pred_pos:.4}"
    );
    assert!(
        pred_neg < -0.5,
        "z=-1 at x=0.25 should recover z·sin(2π·0.25)=-1, got {pred_neg:.4}"
    );
}
