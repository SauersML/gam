//! Standing coverage gate (issue #1891): the survival model's survival-probability
//! credible band S(t | x).
//!
//! Completes the harness's coverage of the library's uncertainty surfaces with
//! the survival family. A survival fit estimates the survival function S(t | x)
//! from right-censored data; its reported band must contain the true survival
//! probability at the nominal rate. Miscalibration here (an over/under-confident
//! hazard/survival band) is the survival instance of the #1869–#1878 genus.
//!
//! The survival prediction machinery (time-basis / baseline-hazard surface,
//! per-row survival-at-time queries) is built around the persisted model, so —
//! like the location-scale gate — this drives the real `gam` CLI end to end:
//! `gam fit "Surv(time, event) ~ s(x)"` then `gam predict --uncertainty` on a
//! newdata frame carrying an explicit query time, parsing `survival_prob` and
//! its bounds (`mean_lower`/`mean_upper`). Driving the CLI (rather than
//! reconstructing the time anchoring in-process) means the query-time semantics
//! are handled by the tested path, so the true-S comparison is unambiguous.
//!
//! Coverage experiment: draw a smooth covariate effect on the Weibull scale from
//! a prior, simulate right-censored survival times with a KNOWN true survival
//! function S(t | x) = exp(-(t/λ(x))^k), fit, and at one independent interior
//! covariate value check whether the true S(t⋆ | x⋆) at a fixed query time lies
//! inside the reported band. Audited by the shared Wilson verdict; only
//! anti-conservative under-coverage gates.
//!
//! Runtime: small n/R, one fit + one predict per replication, fixed seeds. This
//! gate audits the tightest 95% level (the most sensitive); it extends to the
//! full 80/90/95 sweep by predicting once per level.

use std::path::Path;
use std::process::Command;

use gam_test_support::calibration::{CalibrationRng, CoverageClass, audit_coverage};

/// Weibull shape (fixed); the smooth covariate effect enters through the scale.
const WEIBULL_SHAPE: f64 = 1.3;
/// Query time at which survival coverage is audited — chosen so S(t⋆ | x) sits
/// in an informative mid-range (neither ~0 nor ~1) across the covariate range.
const QUERY_TIME: f64 = 0.7;
/// Nominal level audited (the tightest / most sensitive of the 80/90/95 sweep).
const NOMINAL_LEVEL: f64 = 0.95;

const N_TRAIN: usize = 150;
const N_REPLICATIONS: usize = 30;
const SEED: u64 = 0x1891_5_012_71A1;

/// Resolve the `gam` CLI binary for this profile (see the location-scale gate
/// for why `option_env!` is expanded at this call site).
fn gam_bin() -> std::path::PathBuf {
    gam_test_support::cli_harness::resolve_gam_binary(option_env!("CARGO_BIN_EXE_gam"))
}

/// A smooth covariate effect on the log Weibull scale, drawn from the prior.
struct SurvivalTruth {
    log_scale_center: f64,
    log_scale_amp: f64,
    log_scale_freq: f64,
    log_scale_phase: f64,
}

impl SurvivalTruth {
    fn draw(rng: &mut CalibrationRng) -> Self {
        Self {
            // exp(center) ≈ 0.82 .. 1.22: scale O(1) so QUERY_TIME is interior.
            log_scale_center: -0.2 + 0.4 * rng.uniform_open01(),
            log_scale_amp: 0.3 + 0.4 * rng.uniform_open01(),
            log_scale_freq: 0.7 + 0.8 * rng.uniform_open01(),
            log_scale_phase: rng.uniform_open01(),
        }
    }

    fn scale(&self, x: f64) -> f64 {
        let tau = std::f64::consts::TAU;
        (self.log_scale_center
            + self.log_scale_amp * (tau * (self.log_scale_freq * x + self.log_scale_phase)).sin())
        .exp()
    }

    /// True survival function S(t | x) = exp(-(t/λ(x))^k).
    fn survival(&self, t: f64, x: f64) -> f64 {
        (-(t / self.scale(x)).powf(WEIBULL_SHAPE)).exp()
    }

    /// One Weibull event time T | x via inverse-CDF: T = λ(x)·(-ln U)^{1/k}.
    fn draw_event_time(&self, x: f64, rng: &mut CalibrationRng) -> f64 {
        self.scale(x) * (-(rng.uniform_open01()).ln()).powf(1.0 / WEIBULL_SHAPE)
    }
}

fn training_grid(n: usize) -> Vec<f64> {
    (0..n).map(|i| i as f64 / (n - 1) as f64).collect()
}

fn stderr_tail(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes)
        .lines()
        .rev()
        .take(10)
        .collect::<Vec<_>>()
        .join("\n")
}

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

/// Write a right-censored survival training frame: columns `x, time, event`.
fn write_train_csv(path: &Path, x: &[f64], time: &[f64], event: &[f64]) {
    let mut w = csv::Writer::from_path(path).expect("create train csv");
    w.write_record(["x", "time", "event"]).expect("header");
    for i in 0..x.len() {
        w.write_record([x[i].to_string(), time[i].to_string(), event[i].to_string()])
            .expect("row");
    }
    w.flush().expect("flush train csv");
}

/// Write a newdata frame that queries survival at `QUERY_TIME` for each x on the
/// grid (the exit-time column the survival predictor reads is `time`). `event`
/// is a dummy column so the frame carries the same schema as training.
fn write_newdata_csv(path: &Path, x: &[f64]) {
    let mut w = csv::Writer::from_path(path).expect("create newdata csv");
    w.write_record(["x", "time", "event"]).expect("header");
    for &xi in x {
        w.write_record([xi.to_string(), QUERY_TIME.to_string(), "0".to_string()])
            .expect("row");
    }
    w.flush().expect("flush newdata csv");
}

#[test]
fn survival_probability_band_covers_truth_at_nominal() {
    let x = training_grid(N_TRAIN);
    let interior_lo = N_TRAIN / 10;
    let interior_hi = N_TRAIN - N_TRAIN / 10;
    let span = interior_hi - interior_lo;

    let tmp = tempfile::tempdir().expect("tempdir");
    let train_path = tmp.path().join("train.csv");
    let newdata_path = tmp.path().join("newdata.csv");
    let model_path = tmp.path().join("model.gam");
    let out_path = tmp.path().join("pred.csv");
    // Newdata (grid × QUERY_TIME) is identical every replication.
    write_newdata_csv(&newdata_path, &x);

    let mut rng = CalibrationRng::new(SEED);
    let mut hits = 0usize;
    let mut positive_width_seen = false;
    let mut censoring_fraction_seen = 0.0f64;

    for _ in 0..N_REPLICATIONS {
        let truth = SurvivalTruth::draw(&mut rng);
        // Right-censored simulation: T ~ Weibull(k, λ(x)); independent uniform
        // censoring keeps a healthy event fraction while exercising censoring.
        let mut time = Vec::with_capacity(N_TRAIN);
        let mut event = Vec::with_capacity(N_TRAIN);
        let mut censored = 0usize;
        for &xi in &x {
            let t_event = truth.draw_event_time(xi, &mut rng);
            let c = 0.3 + 2.4 * rng.uniform_open01();
            if t_event <= c {
                time.push(t_event);
                event.push(1.0);
            } else {
                time.push(c);
                event.push(0.0);
                censored += 1;
            }
        }
        censoring_fraction_seen = censored as f64 / N_TRAIN as f64;
        write_train_csv(&train_path, &x, &time, &event);

        let fit = Command::new(gam_bin())
            .arg("fit")
            .arg(&train_path)
            .arg("Surv(time, event) ~ s(x)")
            .arg("--out")
            .arg(&model_path)
            .output()
            .expect("spawn gam fit (survival)");
        assert!(
            fit.status.success(),
            "gam fit Surv(time, event) ~ s(x) failed (exit {:?}).\nstderr tail:\n{}",
            fit.status.code(),
            stderr_tail(&fit.stderr)
        );

        let predict = Command::new(gam_bin())
            .arg("predict")
            .arg(&model_path)
            .arg(&newdata_path)
            .arg("--uncertainty")
            .args(["--level", &NOMINAL_LEVEL.to_string()])
            .arg("--out")
            .arg(&out_path)
            .output()
            .expect("spawn gam predict (survival)");
        assert!(
            predict.status.success(),
            "gam predict --uncertainty (survival) failed (exit {:?}).\nstderr tail:\n{}",
            predict.status.code(),
            stderr_tail(&predict.stderr)
        );

        let lower = read_named_column(&out_path, "mean_lower");
        let upper = read_named_column(&out_path, "mean_upper");
        assert_eq!(lower.len(), N_TRAIN, "prediction row count mismatch");

        let j = interior_lo + (rng.uniform_open01() * span as f64) as usize % span;
        assert!(
            lower[j].is_finite() && upper[j].is_finite() && upper[j] >= lower[j],
            "degenerate survival band at row {j}: [{}, {}]",
            lower[j],
            upper[j]
        );
        if upper[j] - lower[j] > 0.0 {
            positive_width_seen = true;
        }
        let s_true = truth.survival(QUERY_TIME, x[j]);
        if lower[j] <= s_true && s_true <= upper[j] {
            hits += 1;
        }
    }

    assert!(
        positive_width_seen,
        "every survival band had zero width — the surface is not producing a real band"
    );
    // Sanity that the simulation actually exercised censoring (a degenerate
    // all-events sim would not test the survival likelihood's censoring path).
    assert!(
        censoring_fraction_seen > 0.05 && censoring_fraction_seen < 0.75,
        "unexpected censoring fraction {censoring_fraction_seen:.3} — check the simulation"
    );

    let verdict = audit_coverage(hits, N_REPLICATIONS, NOMINAL_LEVEL);
    assert!(
        verdict.class != CoverageClass::AntiConservative,
        "survival probability band under-covers the truth at level {NOMINAL_LEVEL}: \
         empirical={:.4} (hits {}/{}), Wilson CI=[{:.4},{:.4}], nominal ABOVE the CI by {:.4} \
         — anti-conservative",
        verdict.empirical,
        verdict.hits,
        verdict.replications,
        verdict.ci_lo,
        verdict.ci_hi,
        -verdict.slack(),
    );
}
