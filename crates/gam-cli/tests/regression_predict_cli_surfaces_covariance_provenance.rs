//! Regression (#2296): `gam predict --uncertainty` gives no user-facing record
//! of which coefficient covariance definition (`Vb` conditional-on-λ̂ vs the
//! smoothing-parameter-corrected `Vc`) produced the reported band. The engine
//! (`gam-predict::InferenceCovarianceMode`) already refuses to substitute one
//! for the other silently — `SmoothingCorrected` hard-errors rather than
//! falling back when the fit lacks the corrected covariance — but nothing
//! outside `gam-pyffi`'s dataset-predict payload surfaced *which* source was
//! actually used. This test pins the CLI's `covariance_provenance_note`
//! (`crates/gam-cli/src/main/model_summary.rs`), wired into the
//! "wrote predictions" line in `run_predict_unified`
//! (`crates/gam-cli/src/main/run_predict.rs`): `--uncertainty` runs must now
//! echo `[covariance=<source>]` naming the exact resolved source, and a plain
//! point-only predict (which never consults a coefficient covariance for a
//! linear/identity-link model) must not.

use std::process::Command;

fn stderr_tail(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes)
        .lines()
        .rev()
        .take(8)
        .collect::<Vec<_>>()
        .join("\n")
}

fn run_predict(
    model: &std::path::Path,
    newdata: &str,
    out: &std::path::Path,
    extra_args: &[&str],
) -> std::process::Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_gam"));
    cmd.arg("predict")
        .arg(model)
        .arg(newdata)
        .arg("--out")
        .arg(out);
    for arg in extra_args {
        cmd.arg(arg);
    }
    cmd.output().expect("spawn gam predict")
}

#[test]
fn predict_uncertainty_surfaces_resolved_covariance_source() {
    let train = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/bug_hunt_uncertainty_point_swap_gaussian_train.csv"
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
    let out_corrected = tempfile::Builder::new()
        .suffix(".csv")
        .tempfile()
        .expect("temp corrected output path");
    let out_conditional = tempfile::Builder::new()
        .suffix(".csv")
        .tempfile()
        .expect("temp conditional output path");

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

    // Plain point-only predict on a linear/identity-link model never consults
    // a coefficient covariance, so the "wrote predictions" line must carry no
    // `[covariance=...]` tag at all.
    let plain = run_predict(model.path(), newdata, out_plain.path(), &[]);
    assert!(
        plain.status.success(),
        "gam predict (plain) failed (exit {:?}).\nstderr tail: {}",
        plain.status.code(),
        stderr_tail(&plain.stderr)
    );
    let plain_stdout = String::from_utf8_lossy(&plain.stdout);
    assert!(
        !plain_stdout.contains("[covariance="),
        "plain point-only predict should not report a covariance source \
         (none was consulted), got stdout: {plain_stdout}"
    );

    // `--uncertainty` with the DEFAULT (smoothing-corrected) covariance mode
    // must label the resolved source. A REML-fit smooth carries the
    // smoothing-corrected covariance, so this must succeed and name it
    // explicitly rather than leaving the reader to guess.
    let corrected = run_predict(
        model.path(),
        newdata,
        out_corrected.path(),
        &["--uncertainty", "--covariance-mode", "corrected"],
    );
    assert!(
        corrected.status.success(),
        "gam predict --uncertainty --covariance-mode corrected failed (exit {:?}).\n\
         stderr tail: {}",
        corrected.status.code(),
        stderr_tail(&corrected.stderr)
    );
    let corrected_stdout = String::from_utf8_lossy(&corrected.stdout);
    assert!(
        corrected_stdout.contains("[covariance=smoothing-corrected]"),
        "--covariance-mode corrected must echo the resolved source \
         (#2296: no silent, unlabeled downgrade to conditional), got stdout: {corrected_stdout}"
    );

    // `--covariance-mode conditional` must label the OTHER source, proving
    // the tag tracks what was actually requested/resolved rather than being
    // a constant string.
    let conditional = run_predict(
        model.path(),
        newdata,
        out_conditional.path(),
        &["--uncertainty", "--covariance-mode", "conditional"],
    );
    assert!(
        conditional.status.success(),
        "gam predict --uncertainty --covariance-mode conditional failed (exit {:?}).\n\
         stderr tail: {}",
        conditional.status.code(),
        stderr_tail(&conditional.stderr)
    );
    let conditional_stdout = String::from_utf8_lossy(&conditional.stdout);
    assert!(
        conditional_stdout.contains("[covariance=conditional]"),
        "--covariance-mode conditional must echo `conditional`, not the corrected \
         source, got stdout: {conditional_stdout}"
    );
}
