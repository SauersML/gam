//! Regression for #898: a purely-parametric survival model
//! `Surv(entry, exit, event) ~ x` (only linear, non-smooth covariates) must be
//! predictable. It previously fit fine but `predict` bailed with
//! `linear term 'x' feature column 3 out of bounds for 3 columns` because the
//! saved linear-term `feature_cols` were left at TRAINING column indices while
//! the survival predict frame drops the response/event column (an off-by-the-
//! dropped-column indexing error). Fixed in 82f184bb6 by remapping
//! `feature_cols` (not just the singular `feature_col`) at predict time.
//!
//! This test exercises the end-to-end fit -> save -> library predict path
//! (`gam::families::survival::predict::predict_survival`, the same entry point the Python
//! `model.predict` FFI uses) and asserts the predict SUCCEEDS, returns a valid
//! survival surface in `[0, 1]` that is monotone non-increasing in `t`, and
//! genuinely varies with the parametric covariate `x`.

use std::path::Path;
use std::process::Command;

use csv::StringRecord;
use gam::encode_recordswith_inferred_schema;
use gam::families::survival::predict::{SurvivalPredictRequest, predict_survival};
use gam::inference::data::EncodedDataset;
use gam::inference::model::FittedModel;
use gam::test_support::cli_harness::run_or_panic;
use ndarray::Array1;

const N: usize = 300;

/// Deterministic right-censored Weibull-shaped data with a single linear
/// covariate `x` whose true effect is a positive log-hazard slope, so the
/// fitted survival curve must differ across `x`.
fn build_dataset() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // Explicit-state RNG helpers (not nested closures: a closure capturing
    // another closure's `&mut state` holds a persistent borrow that conflicts
    // with direct calls to the inner closure).
    fn next_u01(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64) / ((1u64 << 53) as f64)
    }
    // Box-Muller for a standard normal covariate.
    fn next_normal(state: &mut u64) -> f64 {
        let u1 = next_u01(state).max(1e-12);
        let u2 = next_u01(state);
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
    let mut state: u64 = 0x243F6A8885A308D3;

    let shape = 1.5_f64;
    let mut x = Vec::with_capacity(N);
    let mut exit = Vec::with_capacity(N);
    let mut event = Vec::with_capacity(N);
    for _ in 0..N {
        let xi = next_normal(&mut state);
        let eta = -1.5 + 0.7 * xi;
        let u = next_u01(&mut state).max(1e-9);
        let t_lat = (-eta / shape).exp() * (-u.ln()).powf(1.0 / shape);
        let cens = (-next_u01(&mut state).max(1e-12).ln() * 12.0).min(20.0);
        let ex = t_lat.min(cens);
        let ev = if t_lat <= cens { 1.0 } else { 0.0 };
        x.push(xi);
        exit.push(ex);
        event.push(ev);
    }
    (x, exit, event)
}

fn write_training_csv(path: &Path, x: &[f64], exit: &[f64], event: &[f64]) {
    let mut writer = csv::Writer::from_path(path).expect("create training csv");
    writer
        .write_record(["entry", "exit", "event", "x"])
        .expect("write header");
    for i in 0..x.len() {
        writer
            .write_record([
                "0.0".to_string(),
                format!("{:.12}", exit[i]),
                format!("{}", event[i] as i64),
                format!("{:.12}", x[i]),
            ])
            .expect("write training row");
    }
    writer.flush().expect("flush training csv");
}

/// One predict row at the given covariate value with a large `exit` placeholder
/// so every query time is inside the surface frame.
fn predict_dataset(x_val: f64, big_exit: f64) -> EncodedDataset {
    let headers = vec![
        "entry".to_string(),
        "exit".to_string(),
        "event".to_string(),
        "x".to_string(),
    ];
    let rows = vec![StringRecord::from(vec![
        "0.0".to_string(),
        format!("{big_exit:.12}"),
        "1".to_string(),
        format!("{x_val:.12}"),
    ])];
    encode_recordswith_inferred_schema(headers, rows).expect("encode predict row")
}

fn predict_surface(model: &FittedModel, dataset: &EncodedDataset, grid: &[f64]) -> Vec<f64> {
    let col_map = dataset.column_map();
    let training_headers = model.payload().training_headers.as_ref();
    let n = dataset.values.nrows();
    let primary_offset = Array1::<f64>::zeros(n);
    let noise_offset = Array1::<f64>::zeros(n);
    let request = SurvivalPredictRequest {
        model,
        data: dataset.values.view(),
        col_map: &col_map,
        training_headers,
        primary_offset: &primary_offset,
        noise_offset: &noise_offset,
        time_grid: Some(grid),
        with_uncertainty: false,
    };
    let result = predict_survival(request)
        .expect("parametric Surv(...) ~ x must build a survival prediction design and predict");
    result.survival.row(0).to_vec()
}

#[test]
fn parametric_only_survival_model_predicts_a_valid_varying_surface() {
    let (x, exit, event) = build_dataset();
    let dir = tempfile::tempdir().expect("create tempdir");
    let train_path = dir.path().join("train.csv");
    let model_path = dir.path().join("model.json");
    write_training_csv(&train_path, &x, &exit, &event);

    let mut fit_cmd = Command::new(gam::gam_binary!());
    fit_cmd
        .arg("fit")
        .arg(&train_path)
        // Parametric-only formula: a single linear covariate, no smooth. This is
        // the design path that previously crashed at predict (#898).
        .arg("Surv(entry, exit, event) ~ x")
        .args(["--survival-likelihood", "weibull"])
        .arg("--out")
        .arg(&model_path);
    run_or_panic(fit_cmd, "gam fit Surv(...) ~ x (weibull, parametric-only)");
    assert!(model_path.is_file(), "gam fit did not write {model_path:?}");

    let model = FittedModel::load_from_path(&model_path).expect("load parametric survival model");

    let big_exit = exit.iter().cloned().fold(f64::MIN, f64::max) + 5.0;
    let grid = [0.5_f64, 1.0, 2.0, 4.0];

    // Predict at two covariate values. The headline assertion is that predict
    // SUCCEEDS at all (the #898 crash); the surface checks confirm it is a valid
    // survival function that responds to the covariate.
    let surv_lo = predict_surface(&model, &predict_dataset(-1.0, big_exit), &grid);
    let surv_hi = predict_surface(&model, &predict_dataset(1.0, big_exit), &grid);

    for (label, surv) in [("x=-1", &surv_lo), ("x=+1", &surv_hi)] {
        assert_eq!(surv.len(), grid.len(), "{label}: surface width mismatch");
        assert!(
            surv.iter()
                .all(|s| s.is_finite() && (0.0..=1.0).contains(s)),
            "{label}: survival surface must lie in [0, 1]: {surv:?}"
        );
        for w in surv.windows(2) {
            assert!(
                w[1] <= w[0] + 1e-9,
                "{label}: survival surface must be monotone non-increasing in t: {surv:?}"
            );
        }
    }

    // The parametric covariate `x` has a genuine (positive log-hazard) effect, so
    // the two survival curves must differ -- a flat covariate would mean the
    // remapped design silently zeroed the linear term.
    let max_gap = surv_lo
        .iter()
        .zip(surv_hi.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_gap > 1e-3,
        "parametric covariate `x` had no effect on the survival surface \
         (S(x=-1)={surv_lo:?} vs S(x=+1)={surv_hi:?}); the linear term may have \
         been dropped by the predict-time design remap (#898)."
    );
}
