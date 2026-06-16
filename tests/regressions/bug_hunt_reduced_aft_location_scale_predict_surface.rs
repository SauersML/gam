//! Regression for #892: a SAVED reduced-AFT location-scale survival model must
//! predict the correct `S(t|x)`.
//!
//! The reduced parametric-AFT fit (lognormal / loglogistic `Surv(...) ~ x`)
//! removes the time warp entirely (`h ≡ 0`, zero free time columns) and carries
//! the σ-scaled `log t` baseline as a per-row LOCATION shift `η_t → η_t − log t`,
//! so the standardized residual is `u = (log t − μ)/σ` and σ is identified (the
//! survreg / lifelines AFT gauge). The fit is correct, but `predict_survival`
//! used to reconstruct the survival surface from the time-warp coefficients —
//! which are now EMPTY — producing an `S(t|x)` with no `log t` dependence (flat /
//! wrong) for every saved reduced-AFT model.
//!
//! The fix (src/families/survival_predict.rs `predict_survival_location_scale_batch`)
//! detects the regime from the saved payload (empty time-warp β + no timewiggle)
//! and MIRRORS the fit's location shift: per query time it adds `−log t` (floored
//! at `SURVIVAL_TIME_FLOOR`) to the location channel with `h ≡ 0`, so the
//! predicted residual reproduces `(log t − μ)/σ`.
//!
//! This test fits lognormal-AFT data through the real `gam fit` location-scale
//! path, SAVES + LOADS the model, drives the library predict surface
//! (`gam::families::survival::predict::predict_survival`) on a time grid at two covariate
//! values, and asserts the surface is finite, monotone non-increasing in `t`,
//! varies with `x`, and tracks the analytic lognormal survival truth. Without the
//! `−log t` mirror the surface is flat in `t` and fails every check.

use gam::test_support::cli_harness::run_or_panic;
use std::path::Path;
use std::process::Command;

use csv::StringRecord;
use gam::encode_recordswith_inferred_schema;
use gam::inference::data::EncodedDataset;
use gam::inference::model::FittedModel;
use gam::families::survival::predict::{SurvivalPredictRequest, predict_survival};
use ndarray::Array1;

const N: usize = 400;
// True lognormal AFT: log T = mu(x) + sigma * Z, Z ~ N(0,1).
const A: f64 = 1.0; // intercept of mu(x)
const B: f64 = 0.6; // slope on x
const SIGMA_TRUE: f64 = 0.5;

/// Standard normal CDF via erf, matching the lognormal survival the location-
/// scale gaussian residual family implements.
fn normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

/// Abramowitz & Stegun 7.1.26 erf approximation (max abs error ~1.5e-7) — ample
/// for a 5e-2-scale survival-surface tolerance.
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

/// True lognormal survival `S(t|x) = 1 - Phi((log t - mu(x)) / sigma)`.
fn lognormal_survival(t: f64, mu: f64, sigma: f64) -> f64 {
    if t <= 0.0 {
        return 1.0;
    }
    1.0 - normal_cdf((t.ln() - mu) / sigma)
}

/// Deterministic lognormal-AFT data with right censoring.
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
    fn next_normal(state: &mut u64) -> f64 {
        let u1 = next_u01(state).max(1e-12);
        let u2 = next_u01(state);
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
    let mut state: u64 = 0xB5297A4D_68E31DA4;

    let mut x = Vec::with_capacity(N);
    let mut exit = Vec::with_capacity(N);
    let mut event = Vec::with_capacity(N);
    for _ in 0..N {
        let xi = if next_u01(&mut state) < 0.5 {
            -1.0
        } else {
            1.0
        };
        let mu = A + B * xi;
        let t_lat = (mu + SIGMA_TRUE * next_normal(&mut state)).exp();
        // Light censoring so most subjects have an event and sigma is identified.
        let cens = (-next_u01(&mut state).max(1e-12).ln() * 30.0).min(60.0);
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
    writer.write_record(["t", "d", "x"]).expect("write header");
    for i in 0..x.len() {
        writer
            .write_record([
                format!("{:.12}", exit[i]),
                format!("{}", event[i] as i64),
                format!("{:.12}", x[i]),
            ])
            .expect("write training row");
    }
    writer.flush().expect("flush training csv");
}

fn predict_dataset(x_val: f64, big_exit: f64) -> EncodedDataset {
    let headers = vec!["t".to_string(), "d".to_string(), "x".to_string()];
    let rows = vec![StringRecord::from(vec![
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
        .expect("reduced-AFT location-scale survival predict must succeed");
    result.survival.row(0).to_vec()
}

#[test]
fn reduced_aft_location_scale_predicts_correct_log_t_surface() {
    let (x, exit, event) = build_dataset();
    let dir = tempfile::tempdir().expect("create tempdir");
    let train_path = dir.path().join("train.csv");
    let model_path = dir.path().join("model.json");
    write_training_csv(&train_path, &x, &exit, &event);

    let mut fit_cmd = Command::new(gam::gam_binary!());
    fit_cmd
        .arg("fit")
        .arg(&train_path)
        // Lognormal location-scale AFT: the reduced parametric-AFT regime fires
        // (a single parametric covariate, constant scale, no smooth time warp).
        // `--survival-likelihood location-scale` with the default Gaussian
        // residual on log-time IS the lognormal AFT; the `survmodel(...)` term
        // is library-transformation-only syntax the CLI one-hazard fitter
        // rejects, so the formula is the bare `Surv(t, d) ~ x`.
        .arg("Surv(t, d) ~ x")
        .args(["--survival-likelihood", "location-scale"])
        .arg("--out")
        .arg(&model_path);
    run_or_panic(fit_cmd, "gam fit lognormal location-scale AFT");
    assert!(model_path.is_file(), "gam fit did not write {model_path:?}");

    let model = FittedModel::load_from_path(&model_path).expect("load reduced-AFT LS model");

    let big_exit = exit.iter().cloned().fold(f64::MIN, f64::max) + 5.0;
    // Query times spanning the body of the lognormal distribution at both x.
    let grid = [1.0_f64, 2.0, 4.0, 8.0, 16.0];

    let surv_lo = predict_surface(&model, &predict_dataset(-1.0, big_exit), &grid);
    let surv_hi = predict_surface(&model, &predict_dataset(1.0, big_exit), &grid);

    for (label, surv) in [("x=-1", &surv_lo), ("x=+1", &surv_hi)] {
        assert_eq!(surv.len(), grid.len(), "{label}: surface width mismatch");
        assert!(
            surv.iter()
                .all(|s| s.is_finite() && (0.0..=1.0).contains(s)),
            "{label}: survival surface must lie in [0, 1]: {surv:?}"
        );
        // The whole point of #892: S(t|x) must DECREASE in t. Before the fix the
        // location channel carried no `log t`, so the surface was flat in t.
        for w in surv.windows(2) {
            assert!(
                w[1] <= w[0] + 1e-9,
                "{label}: survival surface must be monotone non-increasing in t \
                 (a flat surface means the `-log t` location shift is missing): {surv:?}"
            );
        }
        // And it must genuinely move across the grid (not a constant).
        let span = surv[0] - surv[surv.len() - 1];
        assert!(
            span > 0.2,
            "{label}: survival barely changes across t={grid:?}: {surv:?} (span={span:.4}) \
             — the predicted surface is missing its `log t` dependence (#892)"
        );
    }

    // Higher x ⇒ larger mu ⇒ longer survival: S(t | x=+1) > S(t | x=-1) at every
    // interior time. (B>0.)
    for (i, (&lo, &hi)) in surv_lo.iter().zip(surv_hi.iter()).enumerate() {
        assert!(
            hi > lo - 1e-9,
            "survival must increase with x at t={}: S(x=-1)={lo:.4} S(x=+1)={hi:.4}",
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
        "predicted survival barely depends on x: S(x=-1)={surv_lo:?} S(x=+1)={surv_hi:?}"
    );

    // Track the analytic lognormal truth. The intercept (gauge) is recovered by
    // the fit, so we compare against the TRUE generating surface directly; a
    // loose 0.12 rel-l2 admits finite-sample + censoring noise while still
    // catching a flat / mis-scaled surface (which sits at rel-l2 ~ 0.3+).
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for (xv, surv) in [(-1.0_f64, &surv_lo), (1.0_f64, &surv_hi)] {
        let mu = A + B * xv;
        for (j, &t) in grid.iter().enumerate() {
            let truth = lognormal_survival(t, mu, SIGMA_TRUE);
            num += (surv[j] - truth).powi(2);
            den += truth.powi(2);
        }
    }
    let rel_l2 = (num / den).sqrt();
    assert!(
        rel_l2 <= 0.12,
        "predicted reduced-AFT survival surface diverges from the lognormal truth: \
         rel_l2={rel_l2:.4} S(x=-1)={surv_lo:?} S(x=+1)={surv_hi:?}"
    );
}
