//! Bug: a **parametric linear** term cannot extrapolate, because predict-time
//! input handling silently clamps every continuous covariate to the training
//! range before the design is built.
//!
//! `FittedModel::axis_clip_to_training_ranges` (`src/inference/model.rs:1990`)
//! clips each continuous new-data column to the `(min, max)` observed during
//! training. It deliberately *excludes* categorical/binary columns and periodic
//! axes (see the doc comment and `training_periodic_axes`), but it does **not**
//! exclude columns that feed an ordinary linear term. The clip is
//! applied to all continuous columns at `src/inference/predict/input.rs:227`
//! (`build_predict_input_for_model_inner`), which is the single predict entry
//! point reached by both the `gam predict` CLI (`src/main.rs:2741`) and the
//! Python `gamfit.predict` path (`crates/gam-pyffi/src/lib.rs:24556`).
//!
//! For a smooth/basis term, clamping the input to avoid wild basis blow-up is
//! at least arguable. For a *linear* term `η = β0 + β1·x`, it is not: linear
//! extrapolation is the entire mathematical contract of the term, and clamping
//! the input turns `predict` into a piecewise-constant plateau outside the
//! training hull. It is also internally inconsistent: the raw design path
//! (`build_term_collection_design`, exercised by `extrapolation_does_not_explode.rs`)
//! extrapolates as expected, while the `FittedModel` predict pipeline clamps —
//! so the same model yields different predictions depending on the entry point.
//!
//! Reproduction (confirmed against the `gam` CLI): fit `y ~ x` on a noise-free
//! line `y = 0.5 + 1.25·x` with training `x ∈ [-2, 2]`, then predict outside
//! that range. Observed predictions are flat-clamped to the boundary fitted
//! values:
//!
//! ```text
//!   x=-10  pred=-2.0000   (clamped to value at x=-2; linear truth = -12.0)
//!   x=  3  pred= 3.0000   (clamped to value at x= 2; linear truth =   4.25)
//!   x= 10  pred= 3.0000   (clamped to value at x= 2; linear truth =  13.0)
//! ```
//!
//! Expected: predictions follow `β0 + β1·x` everywhere, i.e. `pred(10) ≈ 13.0`.
//!
//! When the clip is taught to skip columns that feed parametric/linear terms
//! (the same way it already skips categorical/binary/periodic columns), this
//! test passes without edits.

use gam::test_support::cli_harness::{fit_then_predict_gaussian, write_predict_csv_rows};
use std::path::Path;

const SLOPE: f64 = 1.25;
const INTERCEPT: f64 = 0.5;
const TRAIN_LO: f64 = -2.0;
const TRAIN_HI: f64 = 2.0;

fn truth(x: f64) -> f64 {
    INTERCEPT + SLOPE * x
}

/// Noise-free training line `y = 0.5 + 1.25·x` on a dense grid over
/// `x ∈ [-2, 2]`. No noise means the recovered coefficients reproduce the
/// line to numerical precision, so the only thing the assertions can catch is
/// the predict-time clamp.
fn write_training_csv(path: &Path) {
    let mut writer = csv::Writer::from_path(path).expect("create training csv");
    writer.write_record(["x", "y"]).expect("write header");
    let n = 41usize;
    for i in 0..n {
        let x = TRAIN_LO + (TRAIN_HI - TRAIN_LO) * (i as f64) / ((n - 1) as f64);
        writer
            .write_record([format!("{x:.12}"), format!("{:.12}", truth(x))])
            .expect("write training row");
    }
    writer.flush().expect("flush training csv");
}

#[test]
fn linear_term_predict_extrapolates_instead_of_clamping_to_training_range() {
    let dir = tempfile::tempdir().expect("create tempdir");
    let train_path = dir.path().join("train.csv");
    let predict_path = dir.path().join("predict.csv");
    let model_path = dir.path().join("model.json");
    let out_path = dir.path().join("pred_out.csv");

    write_training_csv(&train_path);

    // A mix of in-hull and out-of-hull probes. The in-hull points anchor the
    // test (they must always be correct); the out-of-hull points are where the
    // clamp bug shows up.
    let probes: [f64; 9] = [-10.0, -5.0, -3.0, 0.0, 1.0, 2.0, 3.0, 5.0, 10.0];
    // `y` is a placeholder (predict ignores it); the schema just needs the column.
    write_predict_csv_rows(
        &predict_path,
        ["x", "y"],
        probes
            .iter()
            .map(|&x| [format!("{x:.12}"), "0.0".to_string()]),
    );

    let preds =
        fit_then_predict_gaussian(&train_path, "y ~ x", &model_path, &predict_path, &out_path);
    assert_eq!(
        preds.len(),
        probes.len(),
        "expected one prediction per probe row"
    );

    // Sanity anchor: in-hull predictions must reproduce the line. If these
    // ever fail the model itself is wrong and the extrapolation assertions
    // below would be meaningless — so check them first.
    for (&x, &p) in probes.iter().zip(preds.iter()) {
        if (TRAIN_LO..=TRAIN_HI).contains(&x) {
            assert!(
                (p - truth(x)).abs() < 1e-3,
                "in-hull prediction is wrong: x={x} pred={p} expected≈{} \
                 (model failed to recover the line y=0.5+1.25x)",
                truth(x)
            );
        }
    }

    // The bug: out-of-hull predictions are clamped to the boundary fitted
    // value (truth(±2) = ±2.0 / 3.0) instead of following β0 + β1·x. Assert
    // genuine linear extrapolation. Tolerance is loose (0.05) so this is not a
    // precision test — a clamped prediction misses by whole units (e.g. at
    // x=10 the gap is 13.0 vs 3.0).
    let mut violations = Vec::<String>::new();
    for (&x, &p) in probes.iter().zip(preds.iter()) {
        if (TRAIN_LO..=TRAIN_HI).contains(&x) {
            continue;
        }
        let expected = truth(x);
        if (p - expected).abs() > 0.05 {
            violations.push(format!(
                "x={x:+.1}: pred={p:+.4} expected≈{expected:+.4} \
                 (looks clamped to boundary value {:+.4})",
                truth(x.clamp(TRAIN_LO, TRAIN_HI))
            ));
        }
    }
    assert!(
        violations.is_empty(),
        "linear term `y ~ x` did not extrapolate — predict-time input was \
         clamped to the training range [{TRAIN_LO}, {TRAIN_HI}] \
         (FittedModel::axis_clip_to_training_ranges):\n  - {}",
        violations.join("\n  - "),
    );
}
