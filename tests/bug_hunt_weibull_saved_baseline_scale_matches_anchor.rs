//! Regression for #899: the SAVED Weibull baseline `scale` for single-cause
//! survival (no learned timewiggle) must be recovered from the IDENTIFIED
//! anchor, not the stale/unidentified constant-column coefficient `beta[0]`.
//!
//! Root cause (fixed in `src/main.rs` and `src/solver/workflow.rs`): the fit
//! centers the `[1, log t]` linear time basis at the survival time anchor
//! (`center_survival_time_designs_at_anchor`), which zeroes the constant column,
//! so `beta[0]` is unidentified (left at its stale seed). The model carries the
//! baseline as `eta(t) = beta[1]·(log t − log anchor)`, exactly the Weibull form
//! `eta(t) = shape·(log t − log scale)` with `shape = beta[1]` and
//! `scale = anchor`. `fitted_weibull_baseline_from_linear_time_beta` previously
//! reconstructed `scale = exp(−beta[0]/shape)`, reading the stale `beta[0]`, so
//! the saved `scale` was wrong (it collapses toward `1.0` independent of the
//! anchor). Any consumer that rebuilds `H0(t) = (t/scale)^shape` from the saved
//! scale (e.g. competing-risks CIF) was then misled.
//!
//! This test fits a left-truncated Weibull model through the real `gam fit`
//! path, where the time anchor is the earliest entry age and is well away from
//! `1.0`, then loads the saved model and asserts:
//!   1. the saved baseline `scale` equals the saved `survival_time_anchor`
//!      (the identified reconstruction), NOT the degenerate `≈ 1.0` the stale
//!      `beta[0]` produced; and
//!   2. the baseline cumulative hazard `H0(t) = (t/scale)^shape` rebuilt from
//!      the saved scale/shape is consistent with the survival surface
//!      `predict_survival` derives from the fitted coefficients directly: the
//!      predicted hazard at the reference covariate tracks the saved-baseline
//!      hazard up to a single covariate-driven multiplicative shift that is
//!      constant across the time grid.

use std::path::Path;
use std::process::Command;

use csv::StringRecord;
use gam::encode_recordswith_inferred_schema;
use gam::inference::data::EncodedDataset;
use gam::inference::model::FittedModel;
use gam::survival_predict::{SurvivalPredictRequest, predict_survival};
use gam::test_support::cli_harness::run_or_panic;
use ndarray::Array1;

const N: usize = 500;

/// Deterministic LEFT-TRUNCATED Weibull data: entry ages are pushed to a
/// positive left-tail (entry ∈ [30, 50]), so the time anchor (earliest entry
/// age) is comfortably away from `1.0`. The stale-`beta[0]` reconstruction
/// would yield a saved scale near `1.0`; the correct reconstruction yields the
/// anchor.
fn build_dataset() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut state: u64 = 0x243F6A8885A308D3;
    let mut next_u01 = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };

    let shape = 1.5_f64;
    let mut entry = Vec::with_capacity(N);
    let mut exit = Vec::with_capacity(N);
    let mut event = Vec::with_capacity(N);
    for _ in 0..N {
        // Left-truncation: subjects enter at a positive age, observed only after.
        let en = 30.0 + 20.0 * next_u01();
        let eta = -2.0 + 0.04 * (en - 40.0);
        let u = 1e-9_f64.max(next_u01());
        // Latent additional lifetime past entry.
        let extra = (-eta / shape).exp() * (-u.ln()).powf(1.0 / shape);
        let t_lat = en + extra.max(1e-6);
        let cens = en + (-next_u01().max(1e-12).ln() * 25.0).min(40.0);
        let ex = t_lat.min(cens);
        let ev = if t_lat <= cens { 1.0 } else { 0.0 };
        entry.push(en);
        exit.push(ex);
        event.push(ev);
    }
    (entry, exit, event)
}

fn write_training_csv(path: &Path, entry: &[f64], exit: &[f64], event: &[f64]) {
    let mut writer = csv::Writer::from_path(path).expect("create training csv");
    writer
        .write_record(["entry", "exit", "event"])
        .expect("write header");
    for i in 0..entry.len() {
        writer
            .write_record([
                format!("{:.12}", entry[i]),
                format!("{:.12}", exit[i]),
                format!("{}", event[i] as i64),
            ])
            .expect("write training row");
    }
    writer.flush().expect("flush training csv");
}

/// One predict row with a large `exit` placeholder so every grid time is inside
/// the surface frame.
fn predict_dataset(entry: f64, big_exit: f64) -> EncodedDataset {
    let headers = vec!["entry".to_string(), "exit".to_string(), "event".to_string()];
    let rows = vec![StringRecord::from(vec![
        format!("{entry:.12}"),
        format!("{big_exit:.12}"),
        "1".to_string(),
    ])];
    encode_recordswith_inferred_schema(headers, rows).expect("encode predict row")
}

#[test]
fn weibull_saved_baseline_scale_recovered_from_anchor_not_stale_beta0() {
    let (entry, exit, event) = build_dataset();
    let event_rate = event.iter().sum::<f64>() / event.len() as f64;
    assert!(
        event_rate > 0.4,
        "fixture must have substantial mortality, got event rate {event_rate}"
    );

    let dir = tempfile::tempdir().expect("create tempdir");
    let train_path = dir.path().join("train.csv");
    let model_path = dir.path().join("model.json");
    write_training_csv(&train_path, &entry, &exit, &event);

    // Intercept-only baseline so the entire log-cumulative-hazard sits in the
    // anchor-centered linear time coefficients (no covariate smooth to dilute
    // the baseline-recovery assertion).
    let mut fit_cmd = Command::new(gam::gam_binary!());
    fit_cmd
        .arg("fit")
        .arg(&train_path)
        .arg("Surv(entry, exit, event) ~ 1")
        .args(["--survival-likelihood", "weibull"])
        .arg("--out")
        .arg(&model_path);
    run_or_panic(fit_cmd, "gam fit Surv(entry, exit, event) ~ 1 (weibull)");
    assert!(model_path.is_file(), "gam fit did not write {model_path:?}");

    let model =
        FittedModel::load_from_path(&model_path).expect("load saved Weibull survival model");
    let payload = model.payload();

    let anchor = payload
        .survival_time_anchor
        .expect("saved Weibull model must persist survival_time_anchor");
    let saved_scale = payload
        .survival_baseline_scale
        .expect("saved Weibull model must persist a baseline scale");
    let saved_shape = payload
        .survival_baseline_shape
        .expect("saved Weibull model must persist a baseline shape");

    assert!(
        anchor > 5.0,
        "left-truncated fixture must anchor at a positive left-tail entry age, got {anchor}"
    );
    assert!(
        saved_shape.is_finite() && saved_shape > 0.0,
        "saved Weibull shape must be a finite positive value, got {saved_shape}"
    );

    // The identified reconstruction sets `scale = anchor`. The old, buggy
    // `scale = exp(-beta[0]/shape)` reads the stale (≈0) constant-column
    // coefficient and collapses the saved scale toward 1.0, far below the
    // left-tail anchor.
    let rel_err = (saved_scale - anchor).abs() / anchor.max(1e-12);
    assert!(
        rel_err < 1e-9,
        "saved Weibull baseline scale must equal the identified time anchor \
         (scale = anchor): saved_scale = {saved_scale}, anchor = {anchor} \
         (rel_err = {rel_err:.3e}). A scale near 1.0 means it was rebuilt from \
         the stale, unidentified constant-column coefficient beta[0] (#899)."
    );
    assert!(
        saved_scale > 5.0,
        "saved Weibull scale collapsed toward the degenerate beta[0] value \
         (~1.0) instead of the left-tail anchor {anchor}: {saved_scale} (#899)"
    );

    // Objective consistency: the baseline cumulative hazard rebuilt from the
    // saved scale/shape, H0(t) = (t/scale)^shape, must track the hazard the
    // (beta-direct) predict surface produces, up to a single covariate-driven
    // multiplicative shift that is CONSTANT across the time grid. (For the
    // intercept-only model the shift is exactly the constant covariate effect.)
    let ref_entry = anchor;
    let big_exit = exit.iter().cloned().fold(f64::MIN, f64::max) + 5.0;
    let dataset = predict_dataset(ref_entry, big_exit);
    let col_map = dataset.column_map();
    let training_headers = payload.training_headers.as_ref();
    let n = dataset.values.nrows();
    let primary_offset = Array1::<f64>::zeros(n);
    let noise_offset = Array1::<f64>::zeros(n);

    let grid = [anchor * 1.2, anchor * 1.6, anchor * 2.0, anchor * 2.6];
    let request = SurvivalPredictRequest {
        model: &model,
        data: dataset.values.view(),
        col_map: &col_map,
        training_headers,
        primary_offset: &primary_offset,
        noise_offset: &noise_offset,
        time_grid: Some(&grid),
        with_uncertainty: false,
    };
    let result = predict_survival(request).expect("library Weibull survival predict");
    let surv: Vec<f64> = result.survival.row(0).to_vec();
    assert!(
        surv.iter().all(|s| s.is_finite() && *s > 0.0 && *s <= 1.0),
        "predicted survival must lie in (0, 1]: {surv:?}"
    );

    // Predicted cumulative hazard from the beta-direct surface.
    let h_pred: Vec<f64> = surv.iter().map(|s| -s.ln()).collect();
    // Saved-baseline cumulative hazard.
    let h0: Vec<f64> = grid
        .iter()
        .map(|&t| (t / saved_scale).powf(saved_shape))
        .collect();

    // The ratio H_pred(t) / H0(t) must be (nearly) constant across the grid:
    // both are proportional with a single time-independent covariate shift.
    let ratios: Vec<f64> = h_pred
        .iter()
        .zip(h0.iter())
        .map(|(hp, h0)| hp / h0.max(1e-300))
        .collect();
    assert!(
        ratios.iter().all(|r| r.is_finite() && *r > 0.0),
        "hazard ratio must be finite positive: pred={h_pred:?}, H0={h0:?}, ratios={ratios:?}"
    );
    let mean_ratio = ratios.iter().sum::<f64>() / ratios.len() as f64;
    let max_dev = ratios
        .iter()
        .map(|r| (r - mean_ratio).abs() / mean_ratio)
        .fold(0.0_f64, f64::max);
    assert!(
        max_dev < 1e-3,
        "H0(t) = (t/scale)^shape rebuilt from the SAVED baseline scale/shape \
         must be proportional to the beta-direct predicted hazard across the \
         whole grid (constant covariate shift). Observed time-varying ratio \
         (max relative deviation {max_dev:.3e}) means the saved scale is \
         inconsistent with the fitted baseline — the #899 stale-beta[0] bug. \
         ratios={ratios:?}, H_pred={h_pred:?}, H0={h0:?}"
    );
}
