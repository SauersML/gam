//! Extended regression for #2129: the CLI `gam predict` path must recover the
//! Weibull baseline (never double-count it) across the regimes the canonical
//! repro in `weibull_cli_predict_baseline_double_counted_2129.rs` does NOT
//! exercise. The canonical test pins a single point — increasing hazard
//! (`k = 1.5`), single-cause, `t0 = 0`, no timewiggle. Each assertion below
//! attacks the same root cause (the CLI double-counting the Weibull baseline)
//! from a *different* angle, so a regression that slips past the k=1.5 fixture
//! is still caught:
//!
//! * `decreasing_hazard` — `k = 0.7` (< 1). A different point in parameter
//!   space: the true slope 0.7 and the double-counted slope 1.4 straddle the
//!   0.4 tolerance from opposite sides, so a re-introduced double-count is
//!   unambiguous, and this guards against a fix that happened to work only for
//!   `k > 1`.
//!
//! * `left_truncated_entry` — delayed entry (`t0 > 0`, left truncation). The
//!   fix centers BOTH the entry-time and the exit-time linear designs at the
//!   survival anchor. The canonical fixture has `t0 = 0`, so its entry design
//!   is trivially zero and a regression that centered only the exit design
//!   would pass it while silently corrupting `H(t1) − H(t0)` for truncated
//!   data. This fixture makes the entry-time centering observable.
//!
//! * `with_timewiggle` — the MIRROR branch of the predict gate
//!   `weibull_baseline_in_beta = Weibull && !timewiggle`. With a learned
//!   `timewiggle(...)` the baseline lives in a *parametric* Weibull offset (the
//!   time basis is `None`) and the offset MUST be applied — the exact opposite
//!   of the no-timewiggle case, where it must be dropped. No other test drives
//!   the CLI Weibull+timewiggle predict path, so a change that broke the gate
//!   (e.g. inverting the condition, or zeroing the offset unconditionally)
//!   would go uncaught. Here it would collapse the baseline to `S ≈ 1`.
//!
//! All three fit through the real `gam fit` Weibull path and predict at the
//! baseline covariate (`x = 0`), asserting `eta = log H(t)` is linear in `log t`
//! with slope ≈ the true shape `k` and the survival curve tracks the truth.

use std::path::Path;
use std::process::Command;

use gam::test_support::cli_harness::run_or_panic;

const GRID_TIMES: [f64; 4] = [2.0, 5.0, 10.0, 20.0];

/// Deterministic Weibull proportional-hazards sample with optional left
/// truncation. `H(t | x) = (t / scale)^shape · exp(beta·x)`, drawn by
/// inverse-CDF: `t = scale · exp(−beta·x / shape) · (−log u)^(1/shape)`.
///
/// When `entry_max > 0` each subject is assigned a delayed entry time
/// `t0 ~ U(0, entry_max)` and rows with `t_event ≤ t0` are rejected (left
/// truncation: only subjects who survive to their entry are observed). The
/// delayed-entry risk set means the fitted baseline is still the untruncated
/// population Weibull, so the `x = 0` prediction grid recovers the same
/// `S(t) = exp(−(t/scale)^shape)`.
fn generate_weibull_rows(
    seed: u64,
    n: usize,
    shape: f64,
    scale: f64,
    beta: f64,
    entry_max: f64,
    admin: f64,
) -> Vec<(f64, f64, i64, f64)> {
    let mut state = seed;
    let mut next_u01 = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((state >> 11) as f64) / ((1u64 << 53) as f64)).clamp(1e-12, 1.0 - 1e-12)
    };

    let mut rows = Vec::with_capacity(n);
    // Bound the rejection loop so a mis-parameterised fixture fails loudly
    // instead of hanging; `admin`/`entry_max` here keep acceptance well above
    // 80%, so `50·n` draws is never reached in practice.
    let mut guard = 0usize;
    let max_draws = 50 * n + 1000;
    while rows.len() < n {
        guard += 1;
        assert!(
            guard < max_draws,
            "left-truncation rejection sampling failed to reach {n} rows; check fixture params"
        );
        // Standard-normal covariate via Box-Muller.
        let u1 = next_u01();
        let u2 = next_u01();
        let x = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();

        let u = next_u01();
        let t_event = scale * (-beta * x / shape).exp() * (-u.ln()).powf(1.0 / shape);

        let entry = if entry_max > 0.0 {
            next_u01() * entry_max
        } else {
            0.0
        };
        if entry_max > 0.0 && t_event <= entry {
            continue; // left truncation: unobserved (did not survive to entry)
        }

        let exit = t_event.min(admin);
        let event = i64::from(t_event <= admin);
        rows.push((entry, exit, event, x));
    }
    rows
}

fn write_training_csv(path: &Path, rows: &[(f64, f64, i64, f64)]) {
    let mut writer = csv::Writer::from_path(path).expect("create training csv");
    writer
        .write_record(["t0", "t1", "event", "x"])
        .expect("write header");
    for (entry, exit, event, x) in rows {
        writer
            .write_record([
                format!("{entry:.12}"),
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

fn fit(train: &Path, formula: &str, model: &Path) {
    let mut cmd = Command::new(gam::gam_binary!());
    cmd.arg("fit")
        .arg(train)
        .arg(formula)
        .args(["--survival-likelihood", "weibull"])
        .arg("--out")
        .arg(model);
    run_or_panic(cmd, &format!("gam fit `{formula}` (weibull)"));
    assert!(model.is_file(), "gam fit did not write {model:?}");
}

fn predict_map(model: &Path, grid: &Path, out: &Path) {
    let mut cmd = Command::new(gam::gam_binary!());
    cmd.arg("predict")
        .arg(model)
        .arg(grid)
        .args(["--mode", "map"])
        .arg("--out")
        .arg(out);
    run_or_panic(cmd, "gam predict (map)");
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

fn true_survival(shape: f64, scale: f64) -> [f64; 4] {
    GRID_TIMES.map(|t| (-(t / scale).powf(shape)).exp())
}

/// Fit the given Weibull `formula` on data with the given true parameters,
/// predict at `x = 0`, and assert the baseline is recovered (slope ≈ shape,
/// survival ≈ truth). `entry_max > 0` selects the left-truncated regime.
fn assert_weibull_baseline_recovered(
    label: &str,
    formula: &str,
    seed: u64,
    n: usize,
    shape: f64,
    scale: f64,
    beta: f64,
    entry_max: f64,
    admin: f64,
) {
    let dir = tempfile::tempdir().expect("create tempdir");
    let train = dir.path().join("surv_train.csv");
    let grid = dir.path().join("surv_grid.csv");
    write_training_csv(
        &train,
        &generate_weibull_rows(seed, n, shape, scale, beta, entry_max, admin),
    );
    write_grid_csv(&grid);

    let model = dir.path().join("m_wb.json");
    let pred = dir.path().join("p_wb.csv");
    fit(&train, formula, &model);
    predict_map(&model, &grid, &pred);

    let log_t: Vec<f64> = GRID_TIMES.iter().map(|t| t.ln()).collect();
    let eta = read_column(&pred, "eta");
    let got_slope = slope(&log_t, &eta);
    assert!(
        (got_slope - shape).abs() < 0.4,
        "[{label}] Weibull-mode log-cumulative-hazard slope in log t is \
         {got_slope:.3}, but a Weibull model with shape k={shape} must have \
         slope k. A double-counted baseline would give ~2·k ({:.3}); a dropped \
         baseline would give ~0.",
        2.0 * shape,
    );

    let surv = read_column(&pred, "survival_prob");
    let true_s = true_survival(shape, scale);
    for (got, want) in surv.iter().zip(true_s.iter()) {
        assert!(
            (got - want).abs() < 0.1,
            "[{label}] Weibull-mode (map) survival {surv:?} does not track the \
             truth {true_s:?}; the baseline is mis-counted.",
        );
    }
}

/// Decreasing hazard (`k < 1`): the true slope (0.7) and the double-counted
/// slope (1.4) sit on opposite sides of the tolerance, so this catches any
/// re-introduced double-count that a `k > 1` fixture might tolerate.
#[test]
fn weibull_cli_predict_decreasing_hazard_baseline_not_double_counted() {
    assert_weibull_baseline_recovered(
        "decreasing-hazard k=0.7",
        "Surv(t0,t1,event) ~ x",
        0x1234_5678_9abc_def0,
        2500,
        0.7,  // shape (decreasing hazard)
        8.0,  // scale
        0.5,  // beta
        0.0,  // no left truncation
        60.0, // admin censoring
    );
}

/// Left-truncated / delayed-entry data (`t0 > 0`): exercises the entry-time
/// design centering, which the `t0 = 0` canonical fixture cannot.
#[test]
fn weibull_cli_predict_left_truncated_entry_baseline_not_double_counted() {
    assert_weibull_baseline_recovered(
        "left-truncated entry",
        "Surv(t0,t1,event) ~ x",
        0x0fed_cba9_8765_4321,
        3200,
        1.8,  // shape
        10.0, // scale
        0.7,  // beta
        4.0,  // delayed entry ~ U(0, 4) with left truncation
        40.0, // admin censoring
    );
}

/// The MIRROR branch of the predict gate: with a learned `timewiggle(...)` the
/// baseline lives in the parametric Weibull offset (time basis is `None`) and
/// the offset MUST be applied — the exact opposite of the no-timewiggle case.
#[test]
fn weibull_cli_predict_with_timewiggle_baseline_not_double_counted() {
    assert_weibull_baseline_recovered(
        "with timewiggle",
        "Surv(t0,t1,event) ~ x + timewiggle(internal_knots=6)",
        0x2545_f491_4f6c_dd1d,
        2500,
        1.5,  // shape
        10.0, // scale
        0.8,  // beta
        0.0,  // no left truncation
        40.0, // admin censoring
    );
}
