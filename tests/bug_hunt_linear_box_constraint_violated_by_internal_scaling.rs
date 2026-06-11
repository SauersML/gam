//! Bug (#791): a box-constrained **parametric linear** coefficient —
//! `linear(x, min=.., max=..)` or its sugar `constrain(x, min=.., max=..)` — is
//! not actually held inside `[min, max]` on the reported / saved / prediction
//! scale when the predictor is not already standardized.
//!
//! The bound is wired as an active-set inequality on the *internally
//! conditioned* (centered + scaled) design column, but the inverse-coordinate
//! transform of the constraint used the wrong power of the column scale:
//! `transform_constraint_matrix_to_internal` (`src/solver/estimate.rs`)
//! multiplied the constraint column by `scale` (the back-transform factor)
//! instead of dividing by it (its transpose). The active set then enforced
//! `scale·β_int ≤ ub` ⟹ `β_int ≤ ub/scale`, so the reported coefficient
//! `β = β_int/scale ≤ ub/scale²` escaped the box by exactly `1/scale²`.
//!
//! Reproduction (confirmed against the `gam` CLI): noise-free line `y = 2 + 5·x`
//! on an even grid over `x ∈ [-1, 1]` — the predictor is deliberately *not*
//! standardized (population std ≈ 0.5774, so `scale² ≈ 1/3`). The unconstrained
//! slope is 5, far above `max = 1`, so the box must bind. Before the fix:
//!
//! ```text
//!   y ~ linear(x, min=0, max=1)    -> reported/predicted slope ≈ 2.93   (asked ≤ 1)
//!   y ~ constrain(x, min=0, max=1) -> reported/predicted slope ≈ 2.93   (asked ≤ 1)
//!   y ~ bounded(x, min=0, max=1)   -> reported/predicted slope ≈ 1.00   (box honored)
//! ```
//!
//! `bounded(x, min, max)` uses an exact interval transform on the user-scale
//! coefficient and correctly reports slope ≈ 1 on the same data, so the box is
//! achievable — the two documented ways to box a linear coefficient must agree.
//!
//! Passes once the internal-coordinate transform of the bound uses the
//! canonical back-transform `M` (divide by `scale`), so the active set enforces
//! `(1/scale)·β_int ≤ ub`, giving reported `β = β_int/scale ≤ ub`.

use gam::test_support::cli_harness::{fit_then_predict_gaussian, write_predict_csv_rows};
use std::path::Path;

const INTERCEPT: f64 = 2.0;
const SLOPE: f64 = 5.0;
const TRAIN_LO: f64 = -1.0;
const TRAIN_HI: f64 = 1.0;
const BOX_MAX: f64 = 1.0;

/// Noise-free training line `y = 2 + 5·x` on an even grid over `x ∈ [-1, 1]`.
/// The predictor is intentionally unstandardized (population std ≈ 0.5774) so
/// the column scale differs from 1 and the buggy `1/scale²` escape is visible.
fn write_training_csv(path: &Path) {
    let mut writer = csv::Writer::from_path(path).expect("create training csv");
    writer.write_record(["x", "y"]).expect("write header");
    let n = 41usize;
    for i in 0..n {
        let x = TRAIN_LO + (TRAIN_HI - TRAIN_LO) * (i as f64) / ((n - 1) as f64);
        let y = INTERCEPT + SLOPE * x;
        writer
            .write_record([format!("{x:.12}"), format!("{y:.12}")])
            .expect("write training row");
    }
    writer.flush().expect("flush training csv");
}

/// Fit `y ~ <formula>` on the noise-free unstandardized line, predict at x∈{0,1},
/// and return the reported linear-term slope `pred(1) - pred(0)`.
fn reported_slope(dir: &Path, label: &str, formula: &str) -> f64 {
    let train_path = dir.join("train.csv");
    let predict_path = dir.join("predict.csv");
    let model_path = dir.join(format!("model_{label}.json"));
    let out_path = dir.join(format!("pred_{label}.csv"));

    write_training_csv(&train_path);
    // Two in-hull probes whose `x` gap is 1.0, so `pred(x1) - pred(x0)` is exactly
    // the reported linear-term slope. `y` is a placeholder (predict ignores it).
    write_predict_csv_rows(
        &predict_path,
        ["x", "y"],
        [0.0_f64, 1.0_f64]
            .into_iter()
            .map(|x| [format!("{x:.12}"), "0.0".to_string()]),
    );

    let preds =
        fit_then_predict_gaussian(&train_path, formula, &model_path, &predict_path, &out_path);
    assert_eq!(
        preds.len(),
        2,
        "expected two prediction rows for `{formula}`"
    );
    preds[1] - preds[0]
}

#[test]
fn linear_box_constraint_holds_on_reported_scale() {
    let tmp = tempfile::tempdir().expect("create tempdir");
    let dir = tmp.path();

    // Anchor: `bounded()` uses an exact interval transform on the user-scale
    // coefficient and is known to honor the box on this data. If this ever
    // drifts the data itself stopped exercising a binding box, so check first.
    let bounded_slope = reported_slope(dir, "bounded", "y ~ bounded(x, min=0, max=1)");
    assert!(
        (bounded_slope - BOX_MAX).abs() < 1e-2,
        "anchor failed: bounded(x, min=0, max=1) slope={bounded_slope:.6}, expected ≈ {BOX_MAX} \
         (the box is supposed to bind here — true unconstrained slope is {SLOPE})",
    );

    // The two documented active-set box forms must agree with the anchor: the
    // reported / predicted slope must lie inside [0, 1], not escape to 1/scale².
    for (label, formula) in [
        ("linear", "y ~ linear(x, min=0, max=1)"),
        ("constrain", "y ~ constrain(x, min=0, max=1)"),
    ] {
        let slope = reported_slope(dir, label, formula);
        assert!(
            slope <= BOX_MAX + 1e-3,
            "box constraint violated: `{formula}` reports/predicts slope={slope:.6}, \
             which is OUTSIDE the requested box [0, 1] (escaped toward 1/scale² ≈ {:.6}); \
             bounded() honors the same box at slope={bounded_slope:.6}",
            SLOPE / 3.0,
        );
        // It must also still bind near the upper bound (a degenerate fit that
        // collapsed the slope to ~0 would vacuously satisfy the line above).
        assert!(
            slope >= BOX_MAX - 1e-2,
            "`{formula}` slope={slope:.6} did not bind at the upper bound {BOX_MAX} \
             (the unconstrained slope {SLOPE} should push the active set to the boundary)",
        );
    }
}
