//! Reproduces a CTN monotonicity-boundary regression observed live with
//! `gam fit --transformation-normal --scale-dimensions` followed by
//! `gam predict ...`: the fit is accepted, but at predict time the analytic
//! lower bound on `h'(y, x)` falls below `TRANSFORMATION_MONOTONICITY_EPS`
//! and prediction errors out. This test FAILs RED until that regression is
//! fixed.
//!
//! Recipe (matches `bench/aniso_demo/run_demo.py`):
//!   - n = 300, seed = 7, 5 standard-normal PCs
//!   - pgs_raw = 1.4·PC1 + 0.6·(PC1²−1) + 1.0·PC3 + 0.4·tanh(PC3) + 0.6·N(0,1)
//!   - formula `pgs_raw ~ duchon(pc1_std..pc5_std, centers=12, order=1,
//!     power=8, length_scale=1)`
//!   - `--transformation-normal --scale-dimensions`
//!
//! The error message we want to fail on (raised by
//! `src/inference/predict_input.rs:340-350`):
//!   "prediction failed: transformation-normal fit is non-monotone in y for
//!    at least one observation. Analytic lower bound on h'(y, x) is 1.000e-8,
//!    threshold 1e-8."

use std::path::{Path, PathBuf};
use std::process::Command;

const N: usize = 300;
const SEED: u64 = 7;
const PC_DIM: usize = 5;
const CENTERS: usize = 12;
const DUCHON_ORDER: usize = 1;
const DUCHON_POWER: usize = 8;
const DUCHON_LENGTH: f64 = 1.0;

fn write_demo_fixture_with_python(csv_path: &Path) {
    let script = format!(
        r#"
import csv
import sys
from pathlib import Path

import numpy as np

N = {N}
SEED = {SEED}
PC_DIM = {PC_DIM}

csv_path = Path(sys.argv[1])

rng = np.random.default_rng(SEED)
pcs = rng.standard_normal((N, PC_DIM))
shift = (
    1.4 * pcs[:, 0]
    + 0.6 * (pcs[:, 0] ** 2 - 1.0)
    + 1.0 * pcs[:, 2]
    + 0.4 * np.tanh(pcs[:, 2])
)
pgs_raw = shift + 0.6 * rng.standard_normal(N)

with csv_path.open("w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["pgs_raw"] + ["pc%d_std" % (i + 1) for i in range(PC_DIM)])
    for i in range(N):
        writer.writerow(["%.6f" % pgs_raw[i]] + ["%.6f" % pcs[i, j] for j in range(PC_DIM)])
"#
    );
    let mut command = Command::new("python3");
    command.arg("-c").arg(script).arg(csv_path);
    run_command(command, "generate ctn monotonicity fixture");
}

fn formula() -> String {
    format!(
        "pgs_raw ~ duchon(pc1_std, pc2_std, pc3_std, pc4_std, pc5_std, \
         centers={CENTERS}, order={DUCHON_ORDER}, power={DUCHON_POWER}, \
         length_scale={DUCHON_LENGTH})"
    )
}

fn gam_binary() -> PathBuf {
    option_env!("CARGO_BIN_EXE_gam")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/debug/gam"))
}

fn run_command(mut command: Command, label: &str) {
    let output = command.output().unwrap_or_else(|err| {
        panic!("failed to run {label}: {err}");
    });
    if !output.status.success() {
        panic!(
            "{label} failed with status {}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn ctn_predict_succeeds_when_fit_is_accepted() {
    let workdir = tempfile::tempdir().expect("create tempdir for ctn monotonicity test");
    let csv_path = workdir.path().join("data.csv");
    let model_path = workdir.path().join("model.json");
    let pred_path = workdir.path().join("pred.csv");

    write_demo_fixture_with_python(&csv_path);

    let mut fit = Command::new(gam_binary());
    fit.current_dir(env!("CARGO_MANIFEST_DIR"))
        .arg("fit")
        .arg("--transformation-normal")
        .arg("--scale-dimensions")
        .arg("--out")
        .arg(&model_path)
        .arg(&csv_path)
        .arg(formula());
    run_command(fit, "ctn fit (transformation-normal + scale-dimensions)");

    // The bug: predict refuses the fit because the analytic lower bound on
    // h'(y, x) collapses to the monotonicity threshold. We assert predict
    // succeeds; this assertion fails red until the regression is fixed.
    let mut predict = Command::new(gam_binary());
    predict
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .arg("predict")
        .arg(&model_path)
        .arg(&csv_path)
        .arg("--out")
        .arg(&pred_path);
    let output = predict
        .output()
        .expect("failed to spawn `gam predict` for ctn monotonicity test");
    assert!(
        output.status.success(),
        "predict should succeed for a valid CTN fit, but failed.\n\
         stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}
