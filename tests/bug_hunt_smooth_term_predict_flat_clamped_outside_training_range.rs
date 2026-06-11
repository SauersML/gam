//! Bug: a **spline smooth** term `s(x)` is flat-clamped (frozen at its boundary
//! fitted value) when predicting outside the training range, instead of
//! extrapolating with the boundary slope the way the B-spline basis is built to.
//!
//! `FittedModel::axis_clip_to_training_ranges` (`src/inference/model.rs:2007`)
//! clamps every *continuous* new-data column to the `(min, max)` observed during
//! training before the design is built, at the single predict entry point
//! `build_predict_input_for_model_inner` (`src/inference/predict/input.rs:227`)
//! reached by both the `gam predict` CLI and the Python `gamfit.predict` path.
//! It deliberately *exempts* periodic axes (`training_periodic_axes`),
//! parametric/linear axes (`training_linear_axes`), random-effect axes
//! (`training_random_effect_axes`), and sphere latitude — but it does **not**
//! exempt a column that feeds an ordinary spline smooth `s(x)`.
//!
//! That clamp silently defeats the basis layer's own linear-extension machinery.
//! `apply_dense_bspline_extrapolation` / `apply_linear_extension_from_first_derivative`
//! (`src/terms/basis.rs:534,390`) evaluate the smooth outside the knot domain as
//! `B_ext(z) = B(z_clamped) + (z - z_clamped) · B'(z_clamped)`, i.e. a bounded
//! *linear* extension off the boundary slope — exactly mgcv's behaviour. But the
//! predict pipeline hands the basis an already-clamped `z`, so `z == z_clamped`,
//! `needs_ext` is false, and the linear extension never fires. The result is a
//! piecewise-constant plateau outside the training hull with a prediction SE
//! that is frozen at the boundary — the precise harm the `training_linear_axes`
//! exemption was added to avoid (see its doc comment), now reproduced for the
//! most common term type of all.
//!
//! It is also internally inconsistent: the raw design path
//! `build_term_collection_design` (exercised by `extrapolation_does_not_explode.rs`)
//! extrapolates the smooth in a bounded, non-flat way, while the `FittedModel`
//! predict pipeline clamps — so the same model yields different predictions
//! depending on the entry point. This is the smooth-term sibling of
//! `bug_hunt_predict_linear_term_clamped_to_training_range.rs` (the linear-term
//! case, fixed by the `training_linear_axes` exemption).
//!
//! Reproduction (confirmed against the `gam` CLI): fit `y ~ s(x)` on a
//! noise-free line `y = 0.5 + 1.25·x` with training `x ∈ [0, 2]`. In-range the
//! smooth recovers the line to numerical precision (slope ≈ 1.25), so its
//! boundary slope at `x = 2` is unambiguously ≈ 1.25. Predicting outside the
//! range:
//!
//! ```text
//!   x=1.0  pred=+1.7500   (in range; truth +1.750)
//!   x=2.0  pred=+3.0000   (boundary; truth +3.000)
//!   x=3.0  pred=+3.0000   (FLAT-CLAMPED; linear-extension truth +4.250)
//!   x=4.0  pred=+3.0000   (FLAT-CLAMPED; linear-extension truth +5.500)
//!   x=6.0  pred=+3.0000   (FLAT-CLAMPED; linear-extension truth +8.000)
//! ```
//!
//! Expected: the smooth continues off its boundary slope, so predictions keep
//! moving outside the hull (`pred(6) − pred(2) ≈ 1.25 · 4`), not freeze.
//!
//! When the clip is taught to skip columns that feed a spline-smooth term (the
//! same way it already skips parametric/periodic/random-effect/sphere columns),
//! the linear extension fires and this test passes without edits.

use gam::test_support::cli_harness::{fit_then_predict_gaussian, write_predict_csv_rows};
use std::path::Path;

const SLOPE: f64 = 1.25;
const INTERCEPT: f64 = 0.5;
const TRAIN_LO: f64 = 0.0;
const TRAIN_HI: f64 = 2.0;

fn truth(x: f64) -> f64 {
    INTERCEPT + SLOPE * x
}

/// Noise-free training line `y = 0.5 + 1.25·x` on a dense grid over
/// `x ∈ [0, 2]`. No noise means the smooth recovers the line (and hence its
/// boundary slope) to numerical precision, so the only thing the out-of-hull
/// assertions can catch is the predict-time clamp.
fn write_training_csv(path: &Path) {
    let mut writer = csv::Writer::from_path(path).expect("create training csv");
    writer.write_record(["x", "y"]).expect("write header");
    let n = 200usize;
    for i in 0..n {
        let x = TRAIN_LO + (TRAIN_HI - TRAIN_LO) * (i as f64) / ((n - 1) as f64);
        writer
            .write_record([format!("{x:.12}"), format!("{:.12}", truth(x))])
            .expect("write training row");
    }
    writer.flush().expect("flush training csv");
}

#[test]
fn smooth_term_predict_extrapolates_instead_of_flat_clamping_to_training_range() {
    let dir = tempfile::tempdir().expect("create tempdir");
    let train_path = dir.path().join("train.csv");
    let predict_path = dir.path().join("predict.csv");
    let model_path = dir.path().join("model.json");
    let out_path = dir.path().join("pred_out.csv");

    write_training_csv(&train_path);

    // Two in-hull anchors (to measure the recovered slope) and three out-of-hull
    // probes to the right, where the clamp bug shows up.
    let probes: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 6.0];
    // `y` is a placeholder (predict ignores it); the schema just needs the column.
    write_predict_csv_rows(
        &predict_path,
        ["x", "y"],
        probes
            .iter()
            .map(|&x| [format!("{x:.12}"), "0.0".to_string()]),
    );

    let preds = fit_then_predict_gaussian(
        &train_path,
        "y ~ s(x)",
        &model_path,
        &predict_path,
        &out_path,
    );
    assert_eq!(
        preds.len(),
        probes.len(),
        "expected one prediction per probe row"
    );
    let pred_at = |x: f64| {
        let idx = probes
            .iter()
            .position(|&p| (p - x).abs() < 1e-9)
            .unwrap_or_else(|| panic!("probe {x} not found"));
        preds[idx]
    };

    // Sanity anchor: the in-hull smooth must recover the line, so its slope over
    // [1, 2] is ≈ 1.25. If this fails the model itself is wrong and the
    // extrapolation assertions below would be meaningless — check it first.
    let in_hull_slope = pred_at(2.0) - pred_at(1.0); // ≈ SLOPE over a unit step
    assert!(
        (in_hull_slope - SLOPE).abs() < 0.1,
        "in-hull smooth did not recover the linear trend: slope over [1,2] = {in_hull_slope:.4}, \
         expected ≈ {SLOPE} (pred(1)={:.4}, pred(2)={:.4}); extrapolation check would be moot",
        pred_at(1.0),
        pred_at(2.0),
    );

    // The bug: out-of-hull predictions are frozen at the boundary fitted value
    // (truth(2) = 3.0) instead of continuing off the boundary slope. A correct
    // linear extension grows by ≈ in_hull_slope per unit x; a flat clamp grows
    // by exactly 0. The threshold below (half the recovered slope per unit x) is
    // far above the clamp's 0 and far below the true extension, so this is a
    // robust "is it flat?" check, not a precision test.
    let boundary = pred_at(TRAIN_HI); // value at x = 2 (the clamp target)
    let mut violations = Vec::<String>::new();
    for &x in probes.iter().filter(|&&x| x > TRAIN_HI) {
        let p = pred_at(x);
        let observed_step = p - boundary; // expected ≈ in_hull_slope · (x - 2)
        let min_step = 0.5 * in_hull_slope * (x - TRAIN_HI);
        if observed_step < min_step {
            violations.push(format!(
                "x={x:+.1}: pred={p:+.4} moved only {observed_step:+.4} off the boundary value \
                 {boundary:+.4} (need ≥ {min_step:+.4}; linear-extension truth ≈ {:+.4}) \
                 — looks flat-clamped",
                truth(x)
            ));
        }
    }
    assert!(
        violations.is_empty(),
        "smooth term `s(x)` did not extrapolate — predict-time input was clamped to the \
         training range [{TRAIN_LO}, {TRAIN_HI}] by FittedModel::axis_clip_to_training_ranges, \
         defeating the basis linear-extension (apply_linear_extension_from_first_derivative):\n  - {}",
        violations.join("\n  - "),
    );
}
