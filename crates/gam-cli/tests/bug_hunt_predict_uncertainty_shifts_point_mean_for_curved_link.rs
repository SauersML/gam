//! Bug hunt (#1787): `gam predict --uncertainty` shifted the point `mean` for
//! curved-link families.
//!
//! For a model with a curved inverse link (binomial/logit, and any family where
//! `FittedModel::prediction_uses_posterior_mean()` is true) the CLI's default
//! `gam predict` reports the posterior mean `E[link⁻¹(X·β)]` (the documented
//! default), but `gam predict --uncertainty` wrongly switched to the plug-in
//! `link⁻¹(η̂)`. That violates the #398 invariant: the point prediction is a
//! property of the model + inputs, never of whether an interval was requested —
//! `--uncertainty` only ADDS `std_error`/`mean_lower`/`mean_upper` columns and
//! must never shift `mean` or `linear_predictor`. The Python FFI upholds this;
//! the CLI was the sole discrepant path.
//!
//! ROOT CAUSE: `run_predict_unified` in `run_predict.rs` computed
//! `let nonlinear = model.prediction_uses_posterior_mean();` but the
//! `--uncertainty` arm ignored it, unconditionally building
//! `PredictUncertaintyOptions { mean_interval_method: TransformEta, .. }` and
//! calling `predict_full_uncertainty`, which reports the plug-in
//! `apply_family_inverse_link(&eta, ..)` as `mean`. The no-interval arm instead
//! routes through `predict_posterior_mean` when `nonlinear`.
//!
//! FIX: when `--uncertainty` is set AND `nonlinear`, route through
//! `predict_posterior_mean` (mirroring the Python FFI and the no-interval arm),
//! so the interval columns are added on top of the same posterior-mean point.
//!
//! This test fits a binomial/logit model through the CLI, predicts on the same
//! new data with and without `--uncertainty`, and asserts the `mean` columns are
//! identical (tol 1e-9). Before the fix the two differed by ~1.45e-2.

use std::process::Command;

fn parse_mean_column(csv: &str) -> Vec<f64> {
    let mut lines = csv.lines();
    let header = lines.next().expect("prediction CSV has a header row");
    let mean_idx = header
        .split(',')
        .position(|h| h.trim() == "mean")
        .expect("prediction CSV has a `mean` column");
    lines
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.split(',')
                .nth(mean_idx)
                .expect("row has a mean cell")
                .trim()
                .parse::<f64>()
                .expect("mean cell parses as f64")
        })
        .collect()
}

#[test]
fn predict_uncertainty_does_not_shift_point_mean_for_curved_link() {
    let train = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/bug_hunt_uncertainty_point_swap_train.csv"
    );
    let newdata = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/bug_hunt_uncertainty_point_swap_newdata.csv"
    );

    let model = tempfile::Builder::new()
        .suffix(".gam")
        .tempfile()
        .expect("temp model path");
    let out_plain = tempfile::Builder::new()
        .suffix(".csv")
        .tempfile()
        .expect("temp plain output path");
    let out_uncertainty = tempfile::Builder::new()
        .suffix(".csv")
        .tempfile()
        .expect("temp uncertainty output path");

    // Fit a binomial/logit model (curved inverse link => posterior-mean default).
    let fit = Command::new(env!("CARGO_BIN_EXE_gam"))
        .arg("fit")
        .arg(train)
        .arg("y ~ s(x)")
        .arg("--family")
        .arg("binomial-logit")
        .arg("--out")
        .arg(model.path())
        .output()
        .expect("spawn gam fit");
    assert!(
        fit.status.success(),
        "gam fit (binomial-logit) failed (exit {:?}).\nstderr tail: {}",
        fit.status.code(),
        String::from_utf8_lossy(&fit.stderr)
            .lines()
            .rev()
            .take(8)
            .collect::<Vec<_>>()
            .join("\n")
    );

    // Predict WITHOUT --uncertainty (the documented default point: posterior mean).
    let plain = Command::new(env!("CARGO_BIN_EXE_gam"))
        .arg("predict")
        .arg(model.path())
        .arg(newdata)
        .arg("--out")
        .arg(out_plain.path())
        .output()
        .expect("spawn gam predict (plain)");
    assert!(
        plain.status.success(),
        "gam predict (plain) failed (exit {:?}).\nstderr tail: {}",
        plain.status.code(),
        String::from_utf8_lossy(&plain.stderr)
            .lines()
            .rev()
            .take(8)
            .collect::<Vec<_>>()
            .join("\n")
    );

    // Predict WITH --uncertainty. This must ADD std_error/bounds only, never
    // shift the point `mean`.
    let uncertain = Command::new(env!("CARGO_BIN_EXE_gam"))
        .arg("predict")
        .arg(model.path())
        .arg(newdata)
        .arg("--uncertainty")
        .arg("--out")
        .arg(out_uncertainty.path())
        .output()
        .expect("spawn gam predict --uncertainty");
    assert!(
        uncertain.status.success(),
        "gam predict --uncertainty failed (exit {:?}).\nstderr tail: {}",
        uncertain.status.code(),
        String::from_utf8_lossy(&uncertain.stderr)
            .lines()
            .rev()
            .take(8)
            .collect::<Vec<_>>()
            .join("\n")
    );

    let plain_csv = std::fs::read_to_string(out_plain.path()).expect("read plain predictions");
    let uncertain_csv =
        std::fs::read_to_string(out_uncertainty.path()).expect("read uncertainty predictions");

    let plain_mean = parse_mean_column(&plain_csv);
    let uncertain_mean = parse_mean_column(&uncertain_csv);

    assert_eq!(
        plain_mean.len(),
        uncertain_mean.len(),
        "row counts differ between plain and --uncertainty predictions"
    );
    assert!(!plain_mean.is_empty(), "no prediction rows were produced");

    let mut max_abs = 0.0_f64;
    for (a, b) in plain_mean.iter().zip(uncertain_mean.iter()) {
        max_abs = max_abs.max((a - b).abs());
    }

    // #398/#1787 invariant: `--uncertainty` never shifts the point `mean`.
    assert!(
        max_abs <= 1e-9,
        "gam predict --uncertainty shifted the point mean by up to {max_abs:.3e} \
         for a binomial/logit (curved inverse link) model; the point prediction \
         must be the posterior-mean point reported by the default `gam predict`, \
         with --uncertainty only ADDING std_error/mean_lower/mean_upper columns \
         (#398, #1787)."
    );
}
