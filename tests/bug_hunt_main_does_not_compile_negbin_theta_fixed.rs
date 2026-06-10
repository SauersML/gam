//! Bug: the `gam` crate **does not compile at HEAD**, so nothing that links it —
//! this whole `tests/` suite, the `gam` CLI binary, the `gamfit` Python wheel,
//! any downstream consumer — can be built.
//!
//! Root cause (files/lines read): the latest commit on `main`
//! (`a414a6c25 fix(#982,#983-core,#978): …`) widened two core data-model types
//! but left ~23 of their consumers on the old shape:
//!
//!  * `src/types.rs:362-376` added a field to the negative-binomial family:
//!    `ResponseFamily::NegativeBinomial { theta, theta_fixed: bool }` (the
//!    "fixed θ" knob of #983), and `src/types.rs:1507` added the enum variant
//!    `LikelihoodScaleMetadata::FixedNegBinTheta { theta }`.
//!  * Match arms across the crate still destructure the old
//!    `NegativeBinomial { theta }` without `theta_fixed`, e.g.
//!    `src/inference/generative.rs:108`, `src/inference/hmc.rs:1334`,
//!    `src/inference/hmc.rs:5069`, `src/inference/predict/mod.rs:5753`,
//!    `src/inference/quadrature.rs:3320`, `src/inference/sample.rs:152`,
//!    `src/solver/estimate.rs:750` (E0027: "pattern does not mention field
//!    `theta_fixed`"), and struct literals still omit it, e.g.
//!    `src/inference/hmc.rs:553` `NegativeBinomial { theta: 1.0 }` (E0063).
//!    A `match` over `LikelihoodScaleMetadata` (`src/solver/estimate.rs:4740`)
//!    is left non-exhaustive for the new `FixedNegBinTheta` variant (E0004).
//!  * The sibling #978 commit likewise added `frozen_global_orthogonality` to
//!    `SmoothBasisSpec`/`FactorSmoothSpec` and `unabsorbed_global_orthogonality`
//!    to `SmoothTerm` without updating every initializer (E0063 at
//!    `src/terms/smooth.rs:6505/6823/7719`, `src/terms/term_builder.rs:1509`).
//!
//! `cargo build` (debug or release) therefore fails with 23 errors
//! (`E0027` / `E0063` / `E0004`) and "could not compile `gam` (lib)". The
//! published wheel / prebuilt CLI predate the break, but the source tree at HEAD
//! is red.
//!
//! Per the established pattern for build-break tickets (see e.g. the closed
//! `gpu_err!`-unscoped and `uv.lock`-stale tickets), this test is a real
//! end-to-end fit of the *neighbour* of the broken data-model change: a
//! negative-binomial smooth. While the crate does not compile the test (like
//! everything else) cannot build, so it cannot pass; once every `theta_fixed` /
//! `frozen_global_orthogonality` consumer is updated and the crate compiles,
//! this fit runs and its assertion holds unchanged.
//!
//! The assertion is meaningful, not a tautology: the negative-binomial smooth
//! must recover a known log-mean surface `μ(x) = exp(0.4 + 1.2·sin(2π·x))` from
//! overdispersed counts to better than R² = 0.80 (it actually reaches ≈ 0.97 in
//! sample), which a degenerate / intercept-only fit could not.
//!
//! Related: see the sibling `nonnegative(x)` active-set KKT-abort ticket filed
//! in the same run (its test also cannot build until this compiles).

use std::path::{Path, PathBuf};
use std::process::Command;

fn gam_binary() -> PathBuf {
    option_env!("CARGO_BIN_EXE_gam")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/debug/gam"))
}

const N: usize = 600;

/// True log-mean surface the negative-binomial smooth must recover.
fn true_mean(x: f64) -> f64 {
    (0.4 + 1.2 * (2.0 * std::f64::consts::PI * x).sin()).exp()
}

/// Deterministic overdispersed counts on a low-discrepancy `x` grid. `x_i =
/// frac((i+1)·φ)`; the count is `round(μ(x_i) + jitter)` with a bounded,
/// NB-scale jitter from a fixed LCG, so the data are fully reproducible (no RNG
/// crate, no seed plumbing) yet genuinely overdispersed around `μ`.
fn write_training_csv(path: &Path) {
    let phi = (5.0_f64.sqrt() - 1.0) / 2.0;
    let mut state: u64 = 12345;
    let mut writer = csv::Writer::from_path(path).expect("create training csv");
    writer.write_record(["x", "y"]).expect("write header");
    for i in 0..N {
        let x = (((i + 1) as f64) * phi).fract();
        let mu = true_mean(x);
        // 31-bit LCG → u ∈ [0,1)
        state = (1_103_515_245u64.wrapping_mul(state).wrapping_add(12345)) & 0x7fff_ffff;
        let u = state as f64 / 0x7fff_ffff as f64;
        let nb_sd = (mu + mu * mu / 3.0).sqrt();
        let y = (mu + (u - 0.5) * 2.0 * nb_sd).round().max(0.0);
        writer
            .write_record([format!("{x:.12}"), format!("{y}")])
            .expect("write training row");
    }
    writer.flush().expect("flush training csv");
}

fn read_means(path: &Path) -> Vec<f64> {
    let mut reader = csv::Reader::from_path(path).expect("open predictions csv");
    let headers = reader.headers().expect("predict csv headers").clone();
    let mean_idx = headers
        .iter()
        .position(|h| h == "mean")
        .expect("predict csv has a mean column");
    reader
        .records()
        .map(|rec| {
            rec.expect("predict csv row")[mean_idx]
                .parse::<f64>()
                .expect("numeric prediction")
        })
        .collect()
}

#[test]
fn negative_binomial_smooth_recovers_true_mean_surface() {
    let tmp = tempfile::tempdir().expect("create tempdir");
    let dir = tmp.path();
    let train = dir.join("nb_train.csv");
    let model = dir.join("nb_model.json");
    let pred = dir.join("nb_pred.csv");
    write_training_csv(&train);

    let fit = Command::new(gam_binary())
        .arg("fit")
        .arg(&train)
        .arg("y ~ smooth(x)")
        .args(["--family", "negative-binomial"])
        .arg("--out")
        .arg(&model)
        .output()
        .expect("spawn gam fit");
    assert!(
        fit.status.success() && model.is_file(),
        "gam fit (negative-binomial smooth) failed — the crate must compile and fit; stderr:\n{}",
        String::from_utf8_lossy(&fit.stderr)
    );

    let predict = Command::new(gam_binary())
        .arg("predict")
        .arg(&model)
        .arg(&train)
        .arg("--out")
        .arg(&pred)
        .status()
        .expect("spawn gam predict");
    assert!(predict.success(), "gam predict failed");

    // R² of the predicted mean against the KNOWN mean surface μ(x).
    let phi = (5.0_f64.sqrt() - 1.0) / 2.0;
    let truth: Vec<f64> = (0..N)
        .map(|i| true_mean((((i + 1) as f64) * phi).fract()))
        .collect();
    let preds = read_means(&pred);
    assert_eq!(preds.len(), N, "expected one prediction per training row");

    let mean_t = truth.iter().sum::<f64>() / N as f64;
    let ss_res: f64 = preds
        .iter()
        .zip(&truth)
        .map(|(p, t)| (p - t) * (p - t))
        .sum();
    let ss_tot: f64 = truth.iter().map(|t| (t - mean_t) * (t - mean_t)).sum();
    let r2 = 1.0 - ss_res / ss_tot;
    assert!(
        r2 > 0.80,
        "negative-binomial smooth must recover μ(x)=exp(0.4+1.2·sin(2πx)); R²={r2:.4} (expected ≳0.97)"
    );
}
