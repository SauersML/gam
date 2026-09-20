//! Bug (#791): a box-constrained **parametric linear** coefficient —
//! `linear(x, min=.., max=..)` — is
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
//! slope is 5, far above `max = 1`, so the MAP must bind. Before the fix:
//!
//! ```text
//!   y ~ linear(x, min=0, max=1)    -> reported/predicted slope ≈ 2.93   (asked ≤ 1)
//!   y ~ linear(x, min=0, max=1)    -> reported/predicted slope ≈ 2.93   (asked ≤ 1)
//!   constrained MAP                -> slope = 1.00
//! ```
//!
//! A constrained fit publishes two distinct quantities. The persisted
//! `constrained_posterior.mode` is the boundary MAP; `fit.beta` and default
//! prediction are the truncated posterior mean required by SPEC, which is
//! strictly interior at finite curvature. This regression therefore verifies
//! binding on the mode and verifies support—not boundary equality—on the
//! reported and predicted posterior mean.
//!
//! Passes once the internal-coordinate transform of the bound uses the
//! canonical back-transform `M` (divide by `scale`), so the active set enforces
//! `(1/scale)·β_int ≤ ub`, giving reported `β = β_int/scale ≤ ub`.

use gam::inference::model::FittedModel;
use gam::test_support::cli_harness::write_predict_csv_rows;
use std::path::Path;
use std::process::Command;

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

/// Fit on the noise-free unstandardized line and return
/// `(persisted_mode, reported_mean, predicted_mean_slope)` for the linear term.
fn coefficient_estimands(dir: &Path, label: &str, formula: &str) -> (f64, f64, f64) {
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

    let fit_output = Command::new(gam::gam_binary!())
        .arg("fit")
        .arg(&train_path)
        .arg(formula)
        .args(["--family", "gaussian"])
        .arg("--out")
        .arg(&model_path)
        .output()
        .expect("spawn gam fit");
    assert!(
        fit_output.status.success(),
        "gam fit `{formula}` failed:\n{}",
        String::from_utf8_lossy(&fit_output.stderr)
    );

    let model = FittedModel::load_from_path(&model_path).expect("load constrained model");
    let fit = model.unified().expect("saved model carries unified fit");
    let posterior = fit
        .geometry
        .as_ref()
        .and_then(|geometry| geometry.constrained_posterior.as_ref())
        .expect("linear coefficient box persists its constrained posterior");
    let mode = posterior.mode[1];
    let reported = fit.beta[1];

    let predict_output = Command::new(gam::gam_binary!())
        .arg("predict")
        .arg(&model_path)
        .arg(&predict_path)
        .arg("--out")
        .arg(&out_path)
        .output()
        .expect("spawn gam predict");
    assert!(
        predict_output.status.success(),
        "gam predict `{formula}` failed:\n{}",
        String::from_utf8_lossy(&predict_output.stderr)
    );

    let mut reader = csv::Reader::from_path(&out_path).expect("open predictions csv");
    let headers = reader.headers().expect("predict csv headers").clone();
    let mean_idx = headers
        .iter()
        .position(|header| header == "posterior_mean")
        .expect("standard prediction publishes the explicit posterior_mean column");
    let predictions: Vec<f64> = reader
        .records()
        .map(|record| {
            record.expect("predict csv row")[mean_idx]
                .parse::<f64>()
                .expect("numeric posterior mean")
        })
        .collect();
    assert_eq!(
        predictions.len(),
        2,
        "expected two prediction rows for `{formula}`"
    );
    (mode, reported, predictions[1] - predictions[0])
}

#[test]
fn linear_box_constraint_holds_on_reported_scale() {
    let tmp = tempfile::tempdir().expect("create tempdir");
    let dir = tmp.path();

    // The box must bind at the MAP, while the distinct reported/default-
    // prediction estimand remains inside the requested box.
    for (label, formula) in [("linear", "y ~ linear(x, min=0, max=1)")] {
        let (mode, reported, predicted) = coefficient_estimands(dir, label, formula);
        assert!(
            (mode - BOX_MAX).abs() < 1e-8,
            "`{formula}` MAP must bind at the upper bound {BOX_MAX}, got {mode:.12}"
        );
        assert!(
            (0.0..=BOX_MAX).contains(&reported),
            "box constraint violated: `{formula}` reports posterior mean {reported:.6} \
             outside [0, 1] (the old scaling bug escaped toward 1/scale² ≈ {:.6})",
            SLOPE / 3.0,
        );
        assert!(
            reported < mode,
            "`{formula}` reported {reported:.12}; a finite truncated-posterior mean must \
             be strictly interior to its boundary MAP {mode:.12}"
        );
        assert!(
            (predicted - reported).abs() <= 1e-9 * reported.abs().max(1.0),
            "`{formula}` default prediction slope {predicted:.12} must equal the persisted \
             posterior-mean coefficient {reported:.12}"
        );
    }
}
