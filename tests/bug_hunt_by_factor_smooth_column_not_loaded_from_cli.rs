//! Bug: a factor-`by` smooth `s(x, by=g)` is **unfittable from the `gam` CLI**
//! because the `by=` grouping variable is never added to the set of columns the
//! CLI reads from the input file. The fit aborts before any numerics with
//! `column 'g' not found in data. Available columns: [x, y]`.
//!
//! Root cause: `collect_term_column_names` (`src/main.rs:8013`) builds the
//! CLI's required-column set from the parsed formula. For a smooth term it does
//!
//! ```ignore
//! ParsedTerm::Smooth { vars, .. } => { out.extend(vars.iter().cloned()); }
//! ```
//!
//! i.e. it collects only the smooth's `vars` (here `["x"]`) and ignores the
//! `by=` variable, which the parser stores in `ParsedTerm::Smooth.options` as
//! `{"by": "g"}` (`src/inference/formula_dsl.rs:1114`), not in `vars`. The
//! design builder *does* consume it from there (`options.get("by")` in
//! `src/terms/term_builder.rs:427` and `:1281`), so the column is genuinely
//! required — but `required_columns_for_fit` → `required_columns_for_formula`
//! (`src/main.rs:8059,8036`) never lists it, the CLI loads the file with only
//! `{x, y}`, and `validate_response`/column resolution (`src/inference/data.rs:137`,
//! `src/solver/workflow.rs:175`) rejects the fit with the "not found" error.
//!
//! The functionality itself is fully implemented and correct: declaring the
//! by-variable a second time as a parametric term — `y ~ s(x, by=g) + g` —
//! makes `collect_term_column_names` pick it up via the `ParsedTerm::Linear`
//! arm, the column loads, and the fit recovers each level's curve to ~1e-2
//! (verified). So the smooth works; it is merely unreachable from the CLI
//! without a redundant `+ g`.
//!
//! The same gap also affects a numeric (continuous) `by=z` varying-coefficient
//! smooth and the sibling helper `parsed_terms_reference_column`
//! (`src/inference/formula_dsl.rs:1153`), which likewise only inspects `vars`.
//!
//! Reproduction (confirmed against the `gam` CLI):
//!
//! ```text
//!   $ gam fit data.csv 'y ~ s(x, by=g)' --out model.json
//!   error: column 'g' not found in data. Available columns: [x, y]
//!   $ gam fit data.csv 'y ~ s(x, by=g) + g' --out model.json   # workaround
//!   saved model: model.json
//! ```
//!
//! Expected: `s(x, by=g)` loads `g` and fits, recovering the per-level curves.
//!
//! This test fits the by-smooth, then predicts each level at a point where the
//! two true curves have opposite sign, and asserts the recovered predictions
//! have the correct, distinct signs — so it fails today at the fit step and
//! passes once the `by=` column is added to the required-column set, with no
//! edits.

use gam::test_support::cli_harness::{read_prediction_means, write_predict_csv_rows};
use std::path::Path;
use std::process::Command;

/// Per-level truth: `g="a"` follows `+sin(2πx)`, `g="b"` follows `-sin(2πx)`.
fn truth(x: f64, level: &str) -> f64 {
    let s = (2.0 * std::f64::consts::PI * x).sin();
    if level == "a" { s } else { -s }
}

/// Noisy training data on `x ∈ [0, 1]` for two factor levels.
fn write_training_csv(path: &Path) {
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand_distr::{Distribution, Normal};
    let mut rng = StdRng::seed_from_u64(7);
    let noise = Normal::new(0.0, 0.08).expect("normal");
    let mut writer = csv::Writer::from_path(path).expect("create training csv");
    writer.write_record(["x", "y", "g"]).expect("write header");
    let n = 400usize;
    for i in 0..n {
        let x = (i as f64) / ((n - 1) as f64);
        let level = if i % 2 == 0 { "a" } else { "b" };
        let y = truth(x, level) + noise.sample(&mut rng);
        writer
            .write_record([format!("{x:.10}"), format!("{y:.10}"), level.to_string()])
            .expect("write training row");
    }
    writer.flush().expect("flush training csv");
}

#[test]
fn by_factor_smooth_loads_its_by_column_and_recovers_per_level_curves() {
    let dir = tempfile::tempdir().expect("create tempdir");
    let train_path = dir.path().join("train.csv");
    let predict_path = dir.path().join("predict.csv");
    let model_path = dir.path().join("model.json");
    let out_path = dir.path().join("pred_out.csv");

    write_training_csv(&train_path);

    // Fit the by-factor smooth straight from the CLI, with `g` referenced ONLY
    // through `by=g` (no redundant `+ g`). This is the exact documented syntax
    // (`s(x, by=g)`) used throughout the test suite.
    let mut fit_cmd = Command::new(gam::gam_binary!());
    fit_cmd
        .arg("fit")
        .arg(&train_path)
        .arg("y ~ s(x, by=g)")
        .args(["--family", "gaussian"])
        .arg("--out")
        .arg(&model_path);
    let fit_out = fit_cmd.output().expect("spawn gam fit");
    assert!(
        fit_out.status.success(),
        "`gam fit 'y ~ s(x, by=g)'` failed — the by= column was not loaded \
         (collect_term_column_names ignores ParsedTerm::Smooth.options[\"by\"]).\n\
         --- stdout ---\n{}\n--- stderr ---\n{}",
        String::from_utf8_lossy(&fit_out.stdout),
        String::from_utf8_lossy(&fit_out.stderr),
    );
    assert!(model_path.is_file(), "gam fit did not write {model_path:?}");

    // x = 0.25: sin(2π·0.25) = +1, so level "a" ≈ +1 and level "b" ≈ −1.
    let probes: [(f64, &str); 2] = [(0.25, "a"), (0.25, "b")];
    write_predict_csv_rows(
        &predict_path,
        ["x", "y", "g"],
        probes
            .iter()
            .map(|&(x, g)| [format!("{x:.10}"), "0.0".to_string(), g.to_string()]),
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
        "`gam predict` failed for the by-factor smooth model.\n--- stderr ---\n{}",
        String::from_utf8_lossy(&pred_out.stderr),
    );

    let preds = read_prediction_means(&out_path);
    assert_eq!(preds.len(), 2, "expected one prediction per probe");
    let (pred_a, pred_b) = (preds[0], preds[1]);

    // The two levels' curves are mirror images, so at x = 0.25 they must have
    // opposite signs and both be near ±1. A loose 0.5 threshold makes this a
    // sign/identifiability check, not a precision test.
    assert!(
        pred_a > 0.5,
        "level a at x=0.25 should recover +sin(2π·0.25)=+1, got {pred_a:.4}"
    );
    assert!(
        pred_b < -0.5,
        "level b at x=0.25 should recover -sin(2π·0.25)=-1, got {pred_b:.4}"
    );
}
