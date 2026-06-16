//! Bug: the `gam` crate **does not compile at HEAD**, so nothing that links it —
//! this whole `tests/` suite, the `gam` CLI binary, the `gamfit` Python wheel,
//! any downstream consumer — can be built.
//!
//! `main` is *chronically* red: in the concurrent commit stream, every few
//! commits introduce a fresh compile error that a sibling repairs minutes later,
//! so the crate is non-compiling at most snapshots. Observed, distinct breaks
//! while this ticket was being written (each fixed before the next appeared):
//!
//!  1. #983/#978 widened `ResponseFamily::NegativeBinomial` with a `theta_fixed`
//!     field and added `LikelihoodScaleMetadata::FixedNegBinTheta` /
//!     `SmoothBasisSpec::frozen_global_orthogonality` without updating ~23 match
//!     arms / initializers (E0027 / E0063 / E0004).
//!  2. #935 (`0058c08bb`, the new `FitSensitivity` operator) passed a `&Array2`
//!     where a view was required at `src/solver/sensitivity.rs:109`
//!     (E0277: `cholesky_solve_vector(factor, rhs)` — needs `factor.view()`).
//!  3. The SAE frontier commits (#985/#986/#987) leave `-D warnings`-fatal dead
//!     code and a privacy leak — the live break at the SHA this test is
//!     committed at:
//!       * `src/solver/outer_strategy.rs:5473` / `:4436` — `error: type
//!         `OuterConfig` is more private than the item
//!         `run_per_atom_efs_if_frontier``;
//!       * `src/solver/reml/per_atom_efs.rs:188` — assoc fns `new` /
//!         `fully_coupled` never used;
//!       * `src/terms/sae_candidate_index.rs:271` — field `bits_per_table`
//!         never read;
//!       * `src/terms/sae/corpus/shard_reader.rs:170` — field `path` never read.
//!     `cargo build` → "could not compile `gam` (lib) due to 4 previous errors".
//!
//! The published wheel / prebuilt CLI predate the breaks, but the source tree at
//! HEAD does not build, so the whole `tests/` suite + `gamfit` wheel cannot be
//! built. Per the established pattern for build-break tickets (see the closed
//! `gpu_err!`-unscoped and `uv.lock`-stale tickets), this test is a real
//! end-to-end negative-binomial smooth fit: while the crate does not compile the
//! test (like everything else) cannot build, so it cannot pass; once the tree is
//! green this fit runs and its assertion holds unchanged. The durable ask is a
//! green-`main` gate (a `cargo build`/`clippy -D warnings` check before merge),
//! independent of which specific edit is red on any given day.
//!
//! The assertion is meaningful, not a tautology: the negative-binomial smooth
//! must recover a known log-mean surface `μ(x) = exp(0.4 + 1.2·sin(2π·x))` from
//! overdispersed counts to better than R² = 0.80 (it actually reaches ≈ 0.97 in
//! sample), which a degenerate / intercept-only fit could not.
//!
//! Related: see the sibling `nonnegative(x)` active-set KKT-abort ticket filed
//! in the same run (its test also cannot build until this compiles).

use std::path::Path;
use std::process::Command;

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

    let fit = Command::new(gam::gam_binary!())
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

    let predict = Command::new(gam::gam_binary!())
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
