//! Regression for #2129: the CLI `gam predict` path must not double-count the
//! parametric Weibull baseline for `--survival-likelihood weibull`.
//!
//! A Weibull proportional-hazards model with a linear predictor `x` has
//! log-cumulative-hazard `log H(t | x) = k·log t − k·log λ + β·x`, so at a fixed
//! covariate value `eta = log H(t)` is linear in `log t` with slope exactly the
//! Weibull shape `k`, and `S(t) = exp(−exp(eta))`.
//!
//! Root cause (fixed in `crates/gam-cli/src/main/run_predict.rs`): the Weibull
//! single-cause fit carries its ENTIRE baseline in the anchor-CENTERED
//! `[1, log t]` linear time-basis coefficients (with `beta_time[1] ≈ k`) and
//! uses a Linear (zero) parametric offset. The save path re-encodes that same
//! baseline a SECOND time as a parametric `Weibull` target (recovered
//! scale/shape) purely for CIF/reporting. The CLI predict path previously
//! rebuilt the linear basis UN-centered AND re-added the saved `Weibull` target
//! as a parametric offset, so `k·log t` was counted twice: the fitted
//! log-cumulative-hazard slope came back ~2·k (survival decayed ~2× too fast)
//! and — because the un-centered basis reintroduced the unidentified constant
//! column whose posterior variance is enormous — the default posterior-mean
//! survival collapsed to a flat ≈ 0.5 at every time. The library predict path
//! already avoided this (#897); this test pins the CLI path.
//!
//! The fix mirrors the fit (and the library predict path): center the linear
//! time basis at the survival anchor and carry a zero parametric offset, so the
//! CLI predict reproduces the fitted `beta_time[1]·(log t − log anchor)`.
//!
//! The `transformation` fit on identical data is a passing control (its baseline
//! lives entirely in a structural time basis with a zero parametric offset), so
//! any failure below is specific to the Weibull likelihood mode.

use std::path::Path;
use std::process::Command;

use gam::test_support::cli_harness::run_or_panic;

const TRUE_SHAPE: f64 = 1.5;
const TRUE_SCALE: f64 = 10.0;
const TRUE_BETA: f64 = 0.8;
const GRID_TIMES: [f64; 4] = [2.0, 5.0, 10.0, 20.0];
const N: usize = 2000;

/// Deterministic Weibull proportional-hazards sample:
///   H(t | x) = (t / λ)^k · exp(β·x), drawn by inverse-CDF
///   t = λ · exp(−β·x / k) · (−log u)^(1/k), with light admin censoring.
fn build_training_rows() -> Vec<(f64, i64, f64)> {
    // Deterministic LCG -> uniform(0,1); ample for a reproducible fixture.
    let mut state: u64 = 0x2545F4914F6CDD1D;
    let mut next_u01 = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Keep strictly inside (0, 1) so log/sqrt draws stay finite.
        (((state >> 11) as f64) / ((1u64 << 53) as f64)).clamp(1e-12, 1.0 - 1e-12)
    };

    let mut rows = Vec::with_capacity(N);
    for _ in 0..N {
        // Standard-normal covariate via Box-Muller.
        let u1 = next_u01();
        let u2 = next_u01();
        let x = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();

        let u = next_u01();
        let t_event =
            TRUE_SCALE * (-TRUE_BETA * x / TRUE_SHAPE).exp() * (-u.ln()).powf(1.0 / TRUE_SHAPE);
        let admin = 40.0; // light administrative censoring (~2%)
        let exit = t_event.min(admin);
        let event = i64::from(t_event <= admin);
        rows.push((exit, event, x));
    }
    rows
}

fn write_training_csv(path: &Path, rows: &[(f64, i64, f64)]) {
    let mut writer = csv::Writer::from_path(path).expect("create training csv");
    writer
        .write_record(["t0", "t1", "event", "x"])
        .expect("write header");
    for (exit, event, x) in rows {
        writer
            .write_record([
                "0.0".to_string(),
                format!("{exit:.12}"),
                event.to_string(),
                format!("{x:.12}"),
            ])
            .expect("write training row");
    }
    writer.flush().expect("flush training csv");
}

fn write_grid_csv(path: &Path) {
    let mut writer = csv::Writer::from_path(path).expect("create grid csv");
    writer
        .write_record(["t0", "t1", "event", "x"])
        .expect("write grid header");
    for t in GRID_TIMES {
        writer
            .write_record([
                "0.0".to_string(),
                format!("{t:.12}"),
                "1".to_string(),
                "0.0".to_string(),
            ])
            .expect("write grid row");
    }
    writer.flush().expect("flush grid csv");
}

/// Read a named column of a `gam predict` output CSV as `f64`s.
fn read_column(path: &Path, name: &str) -> Vec<f64> {
    let mut reader = csv::Reader::from_path(path).expect("open predictions csv");
    let headers = reader.headers().expect("predict csv headers").clone();
    let idx = headers
        .iter()
        .position(|h| h == name)
        .unwrap_or_else(|| panic!("predict csv missing `{name}` column: {headers:?}"));
    reader
        .records()
        .map(|rec| {
            let rec = rec.expect("predict csv row");
            rec[idx]
                .parse::<f64>()
                .unwrap_or_else(|_| panic!("non-numeric `{name}`: {:?}", &rec[idx]))
        })
        .collect()
}

fn fit(train: &Path, likelihood: &str, model: &Path) {
    let mut cmd = Command::new(gam::gam_binary!());
    cmd.arg("fit")
        .arg(train)
        .arg("Surv(t0,t1,event) ~ x")
        .args(["--survival-likelihood", likelihood])
        .arg("--out")
        .arg(model);
    run_or_panic(cmd, &format!("gam fit Surv ~ x ({likelihood})"));
    assert!(
        model.is_file(),
        "gam fit ({likelihood}) did not write {model:?}"
    );
}

fn predict(model: &Path, grid: &Path, out: &Path, mode: &str) {
    let mut cmd = Command::new(gam::gam_binary!());
    cmd.arg("predict")
        .arg(model)
        .arg(grid)
        .args(["--mode", mode])
        .arg("--out")
        .arg(out);
    run_or_panic(cmd, &format!("gam predict ({mode})"));
}

/// Least-squares slope of `y` regressed on `x`.
fn slope(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let sxy: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
    let sxx: f64 = x.iter().map(|a| (a - mx).powi(2)).sum();
    sxy / sxx
}

fn true_survival() -> [f64; 4] {
    GRID_TIMES.map(|t| (-(t / TRUE_SCALE).powf(TRUE_SHAPE)).exp())
}

#[test]
fn weibull_cli_predict_baseline_is_not_double_counted() {
    let log_t: Vec<f64> = GRID_TIMES.iter().map(|t| t.ln()).collect();
    let true_s = true_survival();

    let dir = tempfile::tempdir().expect("create tempdir");
    let train = dir.path().join("surv_train.csv");
    let grid = dir.path().join("surv_grid.csv");
    write_training_csv(&train, &build_training_rows());
    write_grid_csv(&grid);

    let m_tf = dir.path().join("m_tf.json");
    let m_wb = dir.path().join("m_wb.json");
    fit(&train, "transformation", &m_tf);
    fit(&train, "weibull", &m_wb);

    let p_tf = dir.path().join("p_tf.csv");
    let p_wb = dir.path().join("p_wb.csv");
    let p_wb_pm = dir.path().join("p_wb_pm.csv");
    predict(&m_tf, &grid, &p_tf, "map");
    predict(&m_wb, &grid, &p_wb, "map");
    predict(&m_wb, &grid, &p_wb_pm, "posterior-mean");

    // ── Control: transformation mode recovers the true Weibull baseline, which
    // proves the data, the CLI, and the survival predict path are sound. ──
    let tf_eta = read_column(&p_tf, "eta");
    let tf_slope = slope(&log_t, &tf_eta);
    assert!(
        (tf_slope - TRUE_SHAPE).abs() < 0.4,
        "control failed: transformation-mode log-cumulative-hazard slope {tf_slope:.3} \
         should recover the true Weibull shape {TRUE_SHAPE}",
    );
    let tf_surv = read_column(&p_tf, "survival_prob");
    for (got, want) in tf_surv.iter().zip(true_s.iter()) {
        assert!(
            (got - want).abs() < 0.05,
            "control failed: transformation-mode survival {tf_surv:?} should match \
             the truth {true_s:?}",
        );
    }

    // ── The bug: Weibull-mode `eta = log H(t)` must be linear in log t with
    // slope equal to the true shape k. Before the fix it comes back ≈ 2·k
    // because the baseline is applied twice (linear time design + re-encoded
    // parametric offset). ──
    let wb_eta = read_column(&p_wb, "eta");
    let wb_slope = slope(&log_t, &wb_eta);
    assert!(
        (wb_slope - TRUE_SHAPE).abs() < 0.4,
        "Weibull-mode survival baseline is DOUBLE-COUNTED: fitted log-cumulative-hazard \
         slope in log t is {wb_slope:.3}, but a Weibull model with shape k={TRUE_SHAPE} \
         must have slope k. The observed slope is ~2·k ({}), i.e. the baseline k·log(t) \
         is added twice. Transformation control recovered slope {tf_slope:.3}.",
        2.0 * TRUE_SHAPE,
    );

    // ── Consequence: the map-mode survival curve must track the truth. ──
    let wb_surv = read_column(&p_wb, "survival_prob");
    for (got, want) in wb_surv.iter().zip(true_s.iter()) {
        assert!(
            (got - want).abs() < 0.1,
            "Weibull-mode (map) survival {wb_surv:?} is grossly wrong vs the truth \
             {true_s:?} because the baseline is double-counted",
        );
    }

    // ── Second symptom: the DEFAULT posterior-mean mode must not collapse the
    // survival curve to a flat ≈ 0.5 at every time (the un-centered,
    // unidentified constant time column had an enormous posterior variance). ──
    let wb_pm_surv = read_column(&p_wb_pm, "survival_prob");
    let max_dev_from_half = wb_pm_surv
        .iter()
        .map(|s| (s - 0.5).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_dev_from_half > 0.1,
        "Weibull-mode default (posterior-mean) survival collapsed to a flat ≈ 0.5 at \
         every time: {wb_pm_surv:?} (truth {true_s:?}). A valid fit must produce a \
         non-degenerate, time-varying survival curve.",
    );
}
