//! End-to-end CAPABILITY test: restricted mean survival time (RMST).
//!
//! RMST(tau) = \int_0^tau S(t) dt = E[min(T, tau)] is the standard clinical-trial
//! survival summary (`survRM2::rmst2`, lifelines `restricted_mean_survival_time`,
//! flexsurv `rmst_*`). gam predicts a full survival surface but, until now,
//! exposed no restricted-mean output; `SurvivalPredictResult::restricted_mean_
//! survival_time` adds it as a derived quantity (trapezoid over the prediction
//! grid). This test pins that capability end to end: fit a known Weibull,
//! predict the survival surface, integrate it to RMST, and recover the truth.
//!
//! DATA-GENERATING PROCESS (truth known exactly).
//!   Weibull latent time with shape k and covariate-dependent log-hazard:
//!     S(t | x) = exp( -(t / scale(x))^k ),   scale(x) = exp( -(eta0 + slope*x)/k ).
//!   The TRUE restricted mean at horizon tau is the analytic integral
//!     RMST_true(tau | x) = \int_0^tau exp( -(t/scale(x))^k ) dt,
//!   computed here to machine precision by dense Simpson quadrature of the KNOWN
//!   curve (this is ground truth, not a reference-tool output).
//!
//! OBJECTIVE METRIC (truth recovery): gam's RMST — obtained by integrating gam's
//! OWN recovered survival surface via the new `restricted_mean_survival_time`
//! method on the predict grid — must match RMST_true within a principled bar at
//! two covariate values.
//!
//! BASELINE TO MATCH-OR-BEAT: lifelines `WeibullAFTFitter` is fit on the same
//! data and its `restricted_mean_survival_time(tau)` (lifelines' own RMST output)
//! is computed at the same covariates; gam's RMST error vs the truth must be no
//! worse than lifelines' by more than 10%.
//!
//! No skip path: a missing `lifelines` is a real failure.

use std::path::Path;
use std::process::Command;

use csv::StringRecord;
use gam::encode_recordswith_inferred_schema;
use gam::families::survival::predict::{SurvivalPredictRequest, predict_survival};
use gam::inference::data::EncodedDataset;
use gam::inference::model::FittedModel;
use gam::test_support::cli_harness::run_or_panic;
use gam::test_support::reference::{Column, run_python};
use ndarray::Array1;

const N: usize = 600;
const SHAPE: f64 = 1.4;
const ETA0: f64 = -1.2;
const SLOPE: f64 = 0.6;
const TAU: f64 = 3.0;

fn scale_of(x: f64) -> f64 {
    (-(ETA0 + SLOPE * x) / SHAPE).exp()
}

/// True S(t | x) for the generating Weibull.
fn true_survival(t: f64, x: f64) -> f64 {
    let s = scale_of(x);
    (-(t / s).powf(SHAPE)).exp()
}

/// Dense-Simpson ground-truth RMST of the KNOWN Weibull curve on [0, tau].
fn true_rmst(x: f64, tau: f64) -> f64 {
    let m = 20_000usize; // even
    let h = tau / m as f64;
    let mut acc = true_survival(0.0, x) + true_survival(tau, x);
    for i in 1..m {
        let t = i as f64 * h;
        acc += if i % 2 == 1 { 4.0 } else { 2.0 } * true_survival(t, x);
    }
    acc * h / 3.0
}

fn build_dataset() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
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
    let mut state: u64 = 0xC0FF_EE12_3456_789A;

    let mut x = Vec::with_capacity(N);
    let mut exit = Vec::with_capacity(N);
    let mut event = Vec::with_capacity(N);
    for _ in 0..N {
        let xi = next_normal(&mut state) * 0.7;
        let s = scale_of(xi);
        let u = next_u01(&mut state).clamp(1e-9, 1.0 - 1e-12);
        // Weibull inverse-CDF: T = scale * (-ln U)^(1/shape).
        let t_lat = s * (-u.ln()).powf(1.0 / SHAPE);
        // Light administrative censoring well beyond tau so RMST(tau) is estimable.
        let cens = 8.0;
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

/// gam's RMST at horizon `tau` for a single covariate value, integrating gam's
/// recovered survival surface via the new `restricted_mean_survival_time` method.
fn gam_rmst(model: &FittedModel, dataset: &EncodedDataset, grid: &[f64], tau: f64) -> f64 {
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
    let result = predict_survival(request).expect("RMST capability: predict survival surface");
    let rmst = result
        .restricted_mean_survival_time(tau)
        .expect("restricted_mean_survival_time must return an RMST vector");
    rmst[0]
}

#[test]
fn gam_recovers_rmst_truth_match_or_beat_lifelines() {
    let (x, exit, event) = build_dataset();
    let dir = tempfile::tempdir().expect("create tempdir");
    let train_path = dir.path().join("train.csv");
    let model_path = dir.path().join("model.json");
    write_training_csv(&train_path, &x, &exit, &event);

    let mut fit_cmd = Command::new(gam::gam_binary!());
    fit_cmd
        .arg("fit")
        .arg(&train_path)
        .arg("Surv(entry, exit, event) ~ x")
        .arg("--survival-likelihood")
        .arg("weibull")
        .arg("--out")
        .arg(&model_path);
    run_or_panic(fit_cmd, "gam fit Weibull Surv(...) ~ x for RMST");
    assert!(model_path.is_file(), "gam fit did not write {model_path:?}");

    let model = FittedModel::load_from_path(&model_path).expect("load Weibull survival model");

    // A grid fine enough that the trapezoid RMST of the recovered smooth Weibull
    // curve is accurate to well under the recovery bar, ending exactly at tau so
    // no extrapolation is needed.
    let grid: Vec<f64> = (1..=60).map(|k| TAU * k as f64 / 60.0).collect();

    let x_eval = [-0.8_f64, 0.8_f64];
    let mut gam_abs_err = 0.0_f64;
    let mut gam_vals = Vec::new();
    let mut truth_vals = Vec::new();
    for &xv in &x_eval {
        let g = gam_rmst(&model, &predict_dataset(xv), &grid, TAU);
        let t = true_rmst(xv, TAU);
        gam_abs_err = gam_abs_err.max((g - t).abs());
        gam_vals.push(g);
        truth_vals.push(t);
    }

    // ---- lifelines RMST baseline on the SAME data ------------------------
    let ref_res = run_python(
        &[
            Column::new("t", &exit),
            Column::new("d", &event),
            Column::new("x", &x),
        ],
        r#"
import numpy as np, pandas as pd
from lifelines import WeibullAFTFitter
df = pd.DataFrame({"t": np.asarray(t,float), "d": np.asarray(d,float), "x": np.asarray(x,float)})
aft = WeibullAFTFitter()
aft.fit(df, duration_col="t", event_col="d")
tau = 3.0
out = []
for xv in [-0.8, 0.8]:
    row = pd.DataFrame({"x": [xv]})
    # lifelines restricted_mean_survival_time integrates the predicted survival fn.
    from lifelines.utils import restricted_mean_survival_time as rmst
    sf = aft.predict_survival_function(row, times=np.linspace(0, tau, 2001))
    out.append(rmst(sf, t=tau))
emit("rmst_ref", out)
"#,
    );
    let rmst_ref = ref_res.vector("rmst_ref");

    let mut ref_abs_err = 0.0_f64;
    for (i, &xv) in x_eval.iter().enumerate() {
        ref_abs_err = ref_abs_err.max((rmst_ref[i] - true_rmst(xv, TAU)).abs());
    }

    eprintln!(
        "RMST(tau={TAU}) recovery: gam={:?} truth={:?} lifelines={:?} | maxerr gam={:.4} lifelines={:.4}",
        gam_vals, truth_vals, rmst_ref, gam_abs_err, ref_abs_err
    );

    // ---- truth-recovery assertion ----------------------------------------
    // RMST is an integral so it is well determined; n=600 with light censoring
    // pins it tightly. Bar is a small fraction of tau.
    assert!(
        gam_abs_err < 0.12,
        "gam RMST failed to recover truth: max |RMST_gam - RMST_true| = {gam_abs_err:.4} (tau={TAU})"
    );

    // ---- match-or-beat lifelines -----------------------------------------
    assert!(
        gam_abs_err <= ref_abs_err * 1.10 + 1e-2,
        "gam RMST recovery ({gam_abs_err:.4}) worse than lifelines ({ref_abs_err:.4}) by >10%"
    );
}
