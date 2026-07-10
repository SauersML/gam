//! Regression for #898: a purely-parametric survival model
//! `Surv(entry, exit, event) ~ x` (only a linear, non-smooth covariate) must be
//! predictable end-to-end.
//!
//! It used to FIT fine but `model.predict(...)` bailed with
//! `linear term 'x' feature column 3 out of bounds for 3 columns`: the saved
//! linear-term `feature_cols` were left at TRAINING column indices while the
//! survival predict frame drops the response/time columns, so the remapped
//! design indexed a stale absolute position (an off-by-the-dropped-column
//! indexing error, the same indexing-fragility family as the `:` interaction
//! builder). Fixed in 82f184bb6 by remapping the `feature_cols` VEC at predict
//! time, not just the singular `feature_col`
//! (`TermCollectionSpec::remap_feature_columns`, `src/terms/smooth.rs`).
//!
//! This test pins the DEFAULT survival likelihood mode — `gam fit` with no
//! `--survival-likelihood` flag is the `transformation` baseline, which is the
//! literal `gh issue view 898` invocation and the parametric mode that does not
//! yet have its own parametric-predict regression file (the sibling guards
//! `bug_hunt_parametric_survival_predict_design_columns.rs` and
//! `bug_hunt_reduced_aft_location_scale_predict_surface.rs` cover the `weibull`
//! and `location-scale` modes). It fits `Surv(t, d) ~ x`, SAVES + LOADS the
//! model, drives the library predict surface (`gam::families::survival::predict::predict_survival`,
//! the exact entry point the Python `model.predict` FFI uses) on a time grid at
//! two covariate values, and asserts predict SUCCEEDS and returns a valid
//! survival surface in `[0, 1]` that is monotone non-increasing in `t` and
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
/// fitted survival curve must differ across `x`. Mirrors the #898 repro's DGP
/// (shape 1.5, log-hazard slope 0.7) so the fit is well posed.
fn build_dataset() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // Explicit-state LCG (no RNG crate dependency; reproducible across runs).
    fn next_u01(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn next_normal(state: &mut u64) -> f64 {
        let u1 = next_u01(state).max(1e-12);
        let u2 = next_u01(state);
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
    let mut state: u64 = 0x1234_5678_9ABC_DEF0;

    const SHAPE: f64 = 1.5;
    const ETA0: f64 = -1.5;
    const SLOPE: f64 = 0.7;

    let mut x = Vec::with_capacity(N);
    let mut exit = Vec::with_capacity(N);
    let mut event = Vec::with_capacity(N);
    for _ in 0..N {
        let xi = next_normal(&mut state);
        // Weibull latent time: T = exp(-eta/shape) * (-log U)^(1/shape).
        let eta = ETA0 + SLOPE * xi;
        let u = next_u01(&mut state).clamp(1e-9, 1.0 - 1e-12);
        let t_lat = (-eta / SHAPE).exp() * (-u.ln()).powf(1.0 / SHAPE);
        // Independent exponential censoring, capped, so most subjects event.
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

/// One predict row at the given covariate value with a placeholder exit time
/// (the survival surface is evaluated on an explicit `time_grid`, not `exit`).
fn predict_dataset(x_val: f64) -> EncodedDataset {
    let headers = vec![
        "entry".to_string(),
        "exit".to_string(),
        "event".to_string(),
        "x".to_string(),
    ];
    let rows = vec![StringRecord::from(vec![
        "0.0".to_string(),
        "1.0".to_string(),
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
        estimand: gam::families::survival::predict::SurvivalPredictEstimand::Plugin,
    };
    let result = predict_survival(request).expect(
        "parametric Surv(...) ~ x must build a survival prediction design and predict (#898)",
    );
    result.survival.row(0).to_vec()
}

#[test]
fn parametric_survival_linear_covariate_predicts_a_valid_varying_surface() {
    let (x, exit, event) = build_dataset();
    let dir = tempfile::tempdir().expect("create tempdir");
    let train_path = dir.path().join("train.csv");
    let model_path = dir.path().join("model.json");
    write_training_csv(&train_path, &x, &exit, &event);

    let mut fit_cmd = Command::new(gam::gam_binary!());
    fit_cmd
        .arg("fit")
        .arg(&train_path)
        // Parametric-only formula: the linear covariate `x` is the design path
        // that previously crashed at predict (#898). No `--survival-likelihood`
        // flag → the DEFAULT `transformation` baseline, the bare `gh issue view
        // 898` invocation.
        .arg("Surv(entry, exit, event) ~ x")
        .arg("--out")
        .arg(&model_path);
    run_or_panic(
        fit_cmd,
        "gam fit Surv(...) ~ x (default transformation, parametric-only)",
    );
    assert!(model_path.is_file(), "gam fit did not write {model_path:?}");

    let model = FittedModel::load_from_path(&model_path).expect("load parametric survival model");

    // Fresh time grid spanning the body of the distribution.
    let grid = [0.25_f64, 0.5, 1.0, 2.0, 4.0];

    // Two distinct covariate values, neither equal to a training row's `x`.
    let surv_lo = predict_surface(&model, &predict_dataset(-1.0), &grid);
    let surv_hi = predict_surface(&model, &predict_dataset(1.0), &grid);

    for (label, surv) in [("x=-1", &surv_lo), ("x=+1", &surv_hi)] {
        assert_eq!(surv.len(), grid.len(), "{label}: surface width mismatch");
        assert!(
            surv.iter()
                .all(|s| s.is_finite() && (0.0..=1.0).contains(s)),
            "{label}: survival surface must lie in [0, 1]: {surv:?}"
        );
        // A survival surface must be monotone non-increasing in t.
        for w in surv.windows(2) {
            assert!(
                w[1] <= w[0] + 1e-9,
                "{label}: survival surface must be monotone non-increasing in t: {surv:?}"
            );
        }
        // And genuinely move across the grid (not a degenerate constant).
        let span = surv[0] - surv[surv.len() - 1];
        assert!(
            span > 0.1,
            "{label}: survival barely changes across t={grid:?}: {surv:?} (span={span:.4})"
        );
    }

    // The covariate has a true positive log-hazard slope, so higher `x` ⇒ higher
    // hazard ⇒ SHORTER survival: S(t | x=+1) < S(t | x=-1) at every interior time.
    // The headline #898 guard is that the two surfaces DIFFER at all — a stale
    // dropped covariate column would make the design ignore `x` and the curves
    // coincide.
    for (i, (&lo, &hi)) in surv_lo.iter().zip(surv_hi.iter()).enumerate() {
        assert!(
            hi <= lo + 1e-9,
            "higher-hazard covariate must not lengthen survival at t={}: \
             S(x=-1)={lo:.4} S(x=+1)={hi:.4}",
            grid[i]
        );
    }
    let max_x_gap = surv_lo
        .iter()
        .zip(surv_hi.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_x_gap > 0.05,
        "predicted survival barely depends on x — the parametric covariate column \
         was likely dropped/stale at predict time (#898): \
         S(x=-1)={surv_lo:?} S(x=+1)={surv_hi:?}"
    );
}
