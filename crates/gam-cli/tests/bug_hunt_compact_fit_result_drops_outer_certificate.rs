//! Bug hunt: `gam fit ... --out FILE` on an ordinary smooth-term Gaussian
//! model fails at save time with
//!
//!   error: Fit assembly rejected a non-converged optimization state: ...
//!   outer status outer iterations ran without an analytic stationarity
//!   certificate, ...
//!
//! even though the printed fit summary reports `status=Converged` and the
//! outer smoothing-parameter search genuinely certified before assembly.
//!
//! ROOT CAUSE: `compact_fit_result_for_batch` (run_fit.rs) resets the whole
//! `FitArtifacts` struct to `Default` to reclaim memory from the heavy
//! `pirls` diagnostic payload, but that also wipes the small
//! `criterion_certificate` field. `save_to_path` immediately re-validates the
//! compacted fit via `validate_numeric_finiteness`, which requires a present
//! certificate whenever `outer_iterations > 0` (#934) — so every standard
//! `--out` fit with a real smoothing-parameter search failed at save time
//! despite having converged and certified moments earlier.
//!
//! This test fits a plain `y ~ s(x)` Gaussian model to `--out` and asserts
//! the CLI succeeds and actually writes the model file.

use std::process::Command;

#[test]
fn standard_smooth_fit_saves_successfully_with_out() {
    let fixture = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/bug_hunt_uncertainty_point_swap_gaussian_train.csv"
    );
    let out = tempfile::Builder::new()
        .suffix(".gam")
        .tempfile()
        .expect("temp output path");

    let output = Command::new(env!("CARGO_BIN_EXE_gam"))
        .arg("fit")
        .arg(fixture)
        .arg("y ~ s(x)")
        .arg("--family")
        .arg("gaussian")
        .arg("--out")
        .arg(out.path())
        .output()
        .expect("spawn gam fit");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        output.status.success(),
        "gam fit y ~ s(x) --out FILE failed (exit {:?}).\nstdout tail: {}\nstderr tail: {}",
        output.status.code(),
        stdout.lines().rev().take(10).collect::<Vec<_>>().join("\n"),
        stderr.lines().rev().take(10).collect::<Vec<_>>().join("\n")
    );
    assert!(
        !stderr.contains("without an analytic stationarity certificate"),
        "compact_fit_result_for_batch regressed: the outer criterion \
         certificate was dropped before save-time revalidation.\nstderr tail: {}",
        stderr.lines().rev().take(10).collect::<Vec<_>>().join("\n")
    );

    let saved = std::fs::metadata(out.path()).expect("model file must exist after a successful fit");
    assert!(saved.len() > 0, "saved model file must be non-empty");
}
