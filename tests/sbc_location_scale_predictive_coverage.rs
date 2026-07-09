//! Standing coverage gate (issue #1891): the Gaussian location-scale predictive
//! interval, which exercises the heteroscedastic scale surface σ(x).
//!
//! This gate targets the #1561 genus directly: a Gaussian location-scale model
//! fits both a mean smooth μ(x) and a log-scale smooth log σ(x), and #1561
//! documents the log-σ smooth being *over-smoothed* (flattened toward a
//! constant). An over-smoothed σ(x) is too small where the true noise is large,
//! so a predictive interval built from the reported σ̂(x) under-covers a new
//! observation in the high-noise regions — a calibration failure that a
//! mean-only band never sees. Averaged over the domain the net effect is
//! anti-conservative, which this gate detects.
//!
//! The location-scale predictor is only constructible from a persisted model
//! (its prediction-time design materialization lives in the saved-model
//! runtime), so — unlike the standard/GLM gates — this one drives the real
//! `gam` CLI end to end: `gam fit ... --predict-noise "s(x)"` then
//! `gam predict ... --uncertainty`, parsing the reported `mean`, `sigma`,
//! `mean_lower`, `mean_upper`. That is exactly the user-facing surface.
//!
//! Coverage sweep (audit mode 1): draw a smooth mean and a smooth (varying)
//! log-scale truth from a prior, simulate heteroscedastic Gaussian data, fit,
//! then at one independent interior point draw a genuinely NEW observation
//! y_new ~ N(μ(x⋆), σ(x⋆)²) and check whether it lies in the predictive interval
//!   μ̂(x⋆) ± z·√(se_μ̂(x⋆)² + σ̂(x⋆)²)
//! (mean uncertainty + observation scatter — the proper predictive interval,
//! reconstructed from the CLI's mean band and σ column). Audited by the shared
//! Wilson verdict over the 80/90/95 sweep; only anti-conservative under-coverage
//! gates, over-coverage is reported.
//!
//! Runtime: small n and R with a single fit + single predict per replication
//! (the level sweep is reconstructed analytically from the level-independent
//! se_μ̂ and σ̂, so no extra CLI spawns), fixed seeds, deterministic.

use std::path::Path;
use std::process::Command;

use gam_test_support::calibration::{
    CalibrationRng, CoverageClass, audit_coverage, standard_normal_quantile,
};

/// Resolve the `gam` CLI binary for this profile. Mirrors the `gam_binary!`
/// macro but as a direct call — `option_env!` must be expanded at this crate's
/// call site (it resolves to `None` for the root package's integration targets,
/// so the runtime resolver locates the binary next to the test's profile dir).
fn gam_bin() -> std::path::PathBuf {
    gam_test_support::cli_harness::resolve_gam_binary(option_env!("CARGO_BIN_EXE_gam"))
}

const N_TRAIN: usize = 120;
const N_REPLICATIONS: usize = 40;
const NOMINAL_LEVELS: [f64; 3] = [0.80, 0.90, 0.95];
/// The single level the CLI is asked for; the mean-band SE is recovered from it
/// and the other levels are reconstructed analytically (Gaussian-symmetric band).
const PREDICT_LEVEL: f64 = 0.95;
const SEED: u64 = 0x1891_1_0C5_CA1E;

/// Smooth mean + smooth (varying) log-scale truth, drawn from the prior.
struct LocScaleTruth {
    mean_center: f64,
    mean_amp: f64,
    mean_freq: f64,
    mean_phase: f64,
    log_sigma_center: f64,
    log_sigma_amp: f64,
    log_sigma_freq: f64,
    log_sigma_phase: f64,
}

impl LocScaleTruth {
    fn draw(rng: &mut CalibrationRng) -> Self {
        Self {
            mean_center: -0.5 + 1.0 * rng.uniform_open01(),
            mean_amp: 0.8 + 0.8 * rng.uniform_open01(),
            mean_freq: 0.7 + 0.8 * rng.uniform_open01(),
            mean_phase: rng.uniform_open01(),
            // exp(center) ≈ 0.25 .. 0.45: a modest baseline noise level.
            log_sigma_center: -1.4 + 0.6 * rng.uniform_open01(),
            // Genuine heteroscedasticity: σ varies by up to exp(±0.7) ≈ ×2 across x.
            log_sigma_amp: 0.4 + 0.4 * rng.uniform_open01(),
            log_sigma_freq: 0.7 + 0.8 * rng.uniform_open01(),
            log_sigma_phase: rng.uniform_open01(),
        }
    }

    fn mean(&self, x: f64) -> f64 {
        let tau = std::f64::consts::TAU;
        self.mean_center + self.mean_amp * (tau * (self.mean_freq * x + self.mean_phase)).sin()
    }

    fn sigma(&self, x: f64) -> f64 {
        let tau = std::f64::consts::TAU;
        (self.log_sigma_center
            + self.log_sigma_amp * (tau * (self.log_sigma_freq * x + self.log_sigma_phase)).sin())
        .exp()
    }
}

fn training_grid(n: usize) -> Vec<f64> {
    (0..n).map(|i| i as f64 / (n - 1) as f64).collect()
}

fn write_train_csv(path: &Path, x: &[f64], y: &[f64]) {
    let mut w = csv::Writer::from_path(path).expect("create train csv");
    w.write_record(["x", "y"]).expect("header");
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        w.write_record([xi.to_string(), yi.to_string()])
            .expect("row");
    }
    w.flush().expect("flush train csv");
}

fn stderr_tail(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes)
        .lines()
        .rev()
        .take(10)
        .collect::<Vec<_>>()
        .join("\n")
}

/// Parse one named column out of a `gam predict` CSV.
fn read_named_column(path: &Path, name: &str) -> Vec<f64> {
    let mut reader = csv::Reader::from_path(path).expect("open prediction csv");
    let headers = reader.headers().expect("prediction headers").clone();
    let idx = headers
        .iter()
        .position(|h| h == name)
        .unwrap_or_else(|| panic!("prediction csv missing `{name}` column: {headers:?}"));
    reader
        .records()
        .map(|rec| {
            let rec = rec.expect("prediction row");
            rec[idx]
                .parse::<f64>()
                .unwrap_or_else(|_| panic!("non-numeric `{name}` cell: {:?}", &rec[idx]))
        })
        .collect()
}

/// Fit a Gaussian location-scale model and predict with uncertainty at
/// `PREDICT_LEVEL`. Returns `(mean, sigma, mean_lower, mean_upper)` per row.
fn fit_and_predict(
    train_path: &Path,
    model_path: &Path,
    out_path: &Path,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let fit = Command::new(gam_bin())
        .arg("fit")
        .arg(train_path)
        .arg("y ~ s(x)")
        .args(["--predict-noise", "s(x)"])
        .args(["--family", "gaussian"])
        .arg("--out")
        .arg(model_path)
        .output()
        .expect("spawn gam fit (location-scale)");
    assert!(
        fit.status.success(),
        "gam fit --predict-noise failed (exit {:?}).\nstderr tail:\n{}",
        fit.status.code(),
        stderr_tail(&fit.stderr)
    );
    let predict = Command::new(gam_bin())
        .arg("predict")
        .arg(model_path)
        .arg(train_path)
        .arg("--uncertainty")
        .args(["--level", &PREDICT_LEVEL.to_string()])
        .arg("--out")
        .arg(out_path)
        .output()
        .expect("spawn gam predict (location-scale)");
    assert!(
        predict.status.success(),
        "gam predict --uncertainty failed (exit {:?}).\nstderr tail:\n{}",
        predict.status.code(),
        stderr_tail(&predict.stderr)
    );
    (
        read_named_column(out_path, "mean"),
        read_named_column(out_path, "sigma"),
        read_named_column(out_path, "mean_lower"),
        read_named_column(out_path, "mean_upper"),
    )
}

#[test]
fn location_scale_predictive_interval_covers_new_observation_at_nominal() {
    let x = training_grid(N_TRAIN);
    let interior_lo = N_TRAIN / 10;
    let interior_hi = N_TRAIN - N_TRAIN / 10;
    let span = interior_hi - interior_lo;

    let tmp = tempfile::tempdir().expect("tempdir");
    let train_path = tmp.path().join("train.csv");
    let model_path = tmp.path().join("model.gam");
    let out_path = tmp.path().join("pred.csv");

    let z_predict = standard_normal_quantile(0.5 + PREDICT_LEVEL / 2.0);

    let mut rng = CalibrationRng::new(SEED);
    let mut hits = [0usize; NOMINAL_LEVELS.len()];
    let mut positive_width_seen = false;

    for _ in 0..N_REPLICATIONS {
        let truth = LocScaleTruth::draw(&mut rng);
        let y: Vec<f64> = x
            .iter()
            .map(|&xi| truth.mean(xi) + truth.sigma(xi) * rng.standard_normal())
            .collect();
        write_train_csv(&train_path, &x, &y);

        let (mean, sigma, lower, upper) = fit_and_predict(&train_path, &model_path, &out_path);
        assert_eq!(mean.len(), N_TRAIN, "prediction row count mismatch");

        // One independent interior evaluation point per replication.
        let j = interior_lo + (rng.uniform_open01() * span as f64) as usize % span;
        // Mean-band SE at this row, recovered from the (Gaussian-symmetric) band.
        let se_mean = (upper[j] - lower[j]) / (2.0 * z_predict);
        let sigma_hat = sigma[j];
        assert!(
            se_mean.is_finite() && sigma_hat.is_finite() && sigma_hat > 0.0,
            "degenerate location-scale prediction at row {j}: se_mean={se_mean}, sigma={sigma_hat}"
        );
        if se_mean > 0.0 || sigma_hat > 0.0 {
            positive_width_seen = true;
        }
        // A genuinely NEW observation at the evaluation point.
        let y_new = truth.mean(x[j]) + truth.sigma(x[j]) * rng.standard_normal();

        for (level_idx, &level) in NOMINAL_LEVELS.iter().enumerate() {
            let z = standard_normal_quantile(0.5 + level / 2.0);
            // Proper predictive half-width: mean uncertainty ⊕ observation scatter.
            let half = z * (se_mean * se_mean + sigma_hat * sigma_hat).sqrt();
            if mean[j] - half <= y_new && y_new <= mean[j] + half {
                hits[level_idx] += 1;
            }
        }
    }

    assert!(
        positive_width_seen,
        "every predictive interval had zero width — the surface is not producing \
         a real interval"
    );

    let mut failures = Vec::new();
    for (level_idx, &level) in NOMINAL_LEVELS.iter().enumerate() {
        let verdict = audit_coverage(hits[level_idx], N_REPLICATIONS, level);
        if verdict.class == CoverageClass::AntiConservative {
            failures.push(format!(
                "level {level}: empirical={:.4} (hits {}/{}), Wilson CI=[{:.4},{:.4}], \
                 nominal ABOVE the CI by {:.4} — anti-conservative (over-smoothed log-σ / \
                 the #1561 signature)",
                verdict.empirical,
                verdict.hits,
                verdict.replications,
                verdict.ci_lo,
                verdict.ci_hi,
                -verdict.slack(),
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "location-scale predictive interval under-covers a new observation:\n{}",
        failures.join("\n")
    );
}
