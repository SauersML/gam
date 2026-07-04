//! Regression (#2115): `gam predict --uncertainty` shifted the point `mean`
//! for the LINEAR / identity-link arm.
//!
//! Companion to `bug_hunt_predict_uncertainty_shifts_point_mean_for_curved_link`
//! (#1787), which pins the CURVED (posterior-mean) arm. The two `run_predict_unified`
//! uncertainty arms are independent: the curved arm routes through
//! `predict_posterior_mean`, while the linear/identity arm builds
//! `PredictUncertaintyOptions { mean_interval_method: TransformEta, .. }`. That arm
//! passed `apply_bias_correction: !args.no_bias_correction` (default `true`),
//! recentring η by `X·H⁻¹S(λ̂)β̂` — so requesting an interval silently moved the
//! reported `mean`/`linear_predictor` (~2.5%) relative to the plain plug-in point
//! that plain `gam predict` and the Python FFI report. #2115 pinned
//! `apply_bias_correction: false` in that arm.
//!
//! The #398 invariant is family-agnostic: the point prediction is a property of
//! the model + inputs, never of whether an interval was requested. This test
//! guards the identity-link arm that the curved-link test never exercises — fit
//! a Gaussian smooth through the CLI, predict on the same new data with and
//! without `--uncertainty`, and assert the `mean` columns are bit-for-bit equal
//! (tol 1e-9). The fixture carries a structured wiggle the penalized smooth
//! cannot fully absorb, so the residual — and hence the pre-fix bias-correction
//! recentring — is genuinely non-zero.

use std::process::Command;

fn parse_named_column(csv: &str, name: &str) -> Vec<f64> {
    let mut lines = csv.lines();
    let header = lines.next().expect("prediction CSV has a header row");
    let idx = header
        .split(',')
        .position(|h| h.trim() == name)
        .unwrap_or_else(|| panic!("prediction CSV has a `{name}` column"));
    lines
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.split(',')
                .nth(idx)
                .unwrap_or_else(|| panic!("row has a `{name}` cell"))
                .trim()
                .parse::<f64>()
                .unwrap_or_else(|_| panic!("`{name}` cell parses as f64"))
        })
        .collect()
}

fn stderr_tail(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes)
        .lines()
        .rev()
        .take(8)
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn predict_uncertainty_does_not_shift_point_mean_for_linear_link() {
    let train = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/bug_hunt_uncertainty_point_swap_gaussian_train.csv"
    );
    // The newdata grid (x only) is family-agnostic — reuse the curved-link fixture.
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

    // Fit a Gaussian (identity link) smooth: the linear/identity uncertainty arm.
    let fit = Command::new(env!("CARGO_BIN_EXE_gam"))
        .arg("fit")
        .arg(train)
        .arg("y ~ s(x)")
        .arg("--family")
        .arg("gaussian")
        .arg("--out")
        .arg(model.path())
        .output()
        .expect("spawn gam fit");
    assert!(
        fit.status.success(),
        "gam fit (gaussian) failed (exit {:?}).\nstderr tail: {}",
        fit.status.code(),
        stderr_tail(&fit.stderr)
    );

    // Predict WITHOUT --uncertainty (the plain plug-in point).
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
        stderr_tail(&plain.stderr)
    );

    // Predict WITH --uncertainty. This must ADD std_error/bounds only, never
    // shift the point `mean` (or `linear_predictor`).
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
        stderr_tail(&uncertain.stderr)
    );

    let plain_csv = std::fs::read_to_string(out_plain.path()).expect("read plain predictions");
    let uncertain_csv =
        std::fs::read_to_string(out_uncertainty.path()).expect("read uncertainty predictions");

    let plain_mean = parse_named_column(&plain_csv, "mean");
    let uncertain_mean = parse_named_column(&uncertain_csv, "mean");

    assert_eq!(
        plain_mean.len(),
        uncertain_mean.len(),
        "row counts differ between plain and --uncertainty predictions"
    );
    assert!(!plain_mean.is_empty(), "no prediction rows were produced");

    let max_abs = plain_mean
        .iter()
        .zip(uncertain_mean.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    // #398/#2115 invariant: `--uncertainty` never shifts the identity-link point.
    assert!(
        max_abs <= 1e-9,
        "gam predict --uncertainty shifted the point mean by up to {max_abs:.3e} for a \
         gaussian (identity link) model; the point prediction must equal the plain \
         `gam predict` plug-in point, with --uncertainty only ADDING \
         std_error/mean_lower/mean_upper columns (#398, #2115)."
    );
}
