//! End-to-end TRUTH RECOVERY for the joint competing-risks survival fit
//! (#1025): `Surv(time, event) ~ x` with event codes `{0, 1, 2}` and
//! `survival_likelihood = "transformation"` — the documented competing-risks
//! entry point that used to abort in the identifiability audit before the
//! cause-priority/audit and inner-solver fixes.
//!
//! OBJECTIVE METRIC ASSERTED (reference-as-truth, #904): we SIMULATE two
//! competing causes with KNOWN Weibull cause-specific hazards
//!
//!     h_k(t | x) = shape_k * t^(shape_k - 1) / scale_k(x)^shape_k,
//!     scale_1(x) = exp( 0.2 + 0.7 x),   shape_1 = 1.2,
//!     scale_2(x) = exp(-0.2 - 0.7 x),   shape_2 = 1.4,
//!
//! plus independent exponential right-censoring, fit the joint cause-specific
//! model through the real `gam fit` CLI path (the exact pipeline the Python
//! `gamfit.fit(...)` FFI drives), and score gam's per-subject cause-specific
//! cumulative incidence against the ANALYTIC truth
//!
//!     CIF_k(t | x) = ∫_0^t h_k(u | x) · S(u | x) du,
//!     S(u | x)     = exp(-H_1(u | x) - H_2(u | x)),
//!
//! evaluated by fine quadrature (truth we construct ourselves — no external
//! reference tool is treated as ground truth). The primary assertion is a
//! cause-specific CIF RMSE bound over a (time × covariate) grid; secondary
//! assertions pin the structural competing-risks identities (each CIF in
//! [0, 1] and monotone, overall survival monotone, and the total-probability
//! identity S_overall(t) + Σ_k CIF_k(t) = 1) and a covariate-direction
//! catastrophe guard (the fitted cause-2 incidence must rise with x, as the
//! truth does, by a non-trivial fraction of the true gap).
//!
//! Bounds are calibrated to leave honest headroom over the observed error of a
//! correct fit while failing on any qualitative regression (audit refusal,
//! inner-solve collapse, baseline mix-up between the twin time bases, or a
//! flipped covariate effect). We never weaken a bound to force a pass.

use std::path::Path;
use std::process::Command;

use csv::StringRecord;
use gam::encode_recordswith_inferred_schema;
use gam::families::survival::predict::{
    SurvivalPredictRequest, SurvivalPredictionCovarianceMode, predict_competing_risks_survival,
};
use gam::inference::data::EncodedDataset;
use gam::inference::model::FittedModel;
use gam::test_support::cli_harness::run_or_panic;
use ndarray::Array1;

const N: usize = 900;
const SHAPE_1: f64 = 1.2;
const SHAPE_2: f64 = 1.4;

fn scale_1(x: f64) -> f64 {
    (0.2 + 0.7 * x).exp()
}

fn scale_2(x: f64) -> f64 {
    (-0.2 - 0.7 * x).exp()
}

/// Deterministic LCG -> uniform(0,1); reproducible without an RNG dependency.
fn lcg_u01(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

/// Weibull(shape, scale) draw via inverse CDF: scale * (-ln U)^(1/shape).
fn weibull_draw(u: f64, shape: f64, scale: f64) -> f64 {
    scale * (-(u.max(1e-12)).ln()).powf(1.0 / shape)
}

/// Two-cause competing-risks sample: latent cause times T1, T2 with the known
/// cause-specific hazards above, censoring C ~ Exp(mean 8); observed
/// (min(T1, T2, C), cause indicator 0/1/2).
fn simulate() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut state: u64 = 0x1025_1025_2026_0612;
    let mut time = Vec::with_capacity(N);
    let mut event = Vec::with_capacity(N);
    let mut xs = Vec::with_capacity(N);
    for _ in 0..N {
        let x = lcg_u01(&mut state);
        let t1 = weibull_draw(lcg_u01(&mut state), SHAPE_1, scale_1(x));
        let t2 = weibull_draw(lcg_u01(&mut state), SHAPE_2, scale_2(x));
        let c = -8.0 * lcg_u01(&mut state).max(1e-12).ln();
        let obs = t1.min(t2).min(c);
        let ev = if obs == c {
            0.0
        } else if t1 < t2 {
            1.0
        } else {
            2.0
        };
        time.push(obs);
        event.push(ev);
        xs.push(x);
    }
    (time, event, xs)
}

fn write_training_csv(path: &Path, time: &[f64], event: &[f64], x: &[f64]) {
    let mut writer = csv::Writer::from_path(path).expect("create training csv");
    writer
        .write_record(["time", "event", "x"])
        .expect("write header");
    for i in 0..time.len() {
        writer
            .write_record([
                format!("{:.12}", time[i]),
                format!("{}", event[i] as i64),
                format!("{:.12}", x[i]),
            ])
            .expect("write training row");
    }
    writer.flush().expect("flush training csv");
}

/// True cause-specific CIFs at the requested times via cumulative trapezoid
/// quadrature of h_k(u) * S(u) on a fine u-grid (step 2.5e-4 — discretization
/// error O(1e-7), negligible against the asserted RMSE scale).
fn true_cifs(x: f64, times: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let t_max = times.iter().cloned().fold(0.0_f64, f64::max);
    let step = 2.5e-4;
    let n_steps = (t_max / step).ceil() as usize;
    let a1 = scale_1(x);
    let a2 = scale_2(x);
    let hazard_1 = |u: f64| SHAPE_1 * u.powf(SHAPE_1 - 1.0) / a1.powf(SHAPE_1);
    let hazard_2 = |u: f64| SHAPE_2 * u.powf(SHAPE_2 - 1.0) / a2.powf(SHAPE_2);
    let total_survival = |u: f64| (-(u / a1).powf(SHAPE_1) - (u / a2).powf(SHAPE_2)).exp();
    let mut cif1 = vec![0.0_f64; n_steps + 1];
    let mut cif2 = vec![0.0_f64; n_steps + 1];
    let mut grid = vec![0.0_f64; n_steps + 1];
    let mut prev1 = 0.0_f64; // h_1(0)·S(0) = 0 for shape > 1
    let mut prev2 = 0.0_f64;
    for j in 1..=n_steps {
        let u = step * j as f64;
        let s = total_survival(u);
        let f1 = hazard_1(u) * s;
        let f2 = hazard_2(u) * s;
        cif1[j] = cif1[j - 1] + 0.5 * step * (prev1 + f1);
        cif2[j] = cif2[j - 1] + 0.5 * step * (prev2 + f2);
        grid[j] = u;
        prev1 = f1;
        prev2 = f2;
    }
    let read_at = |cif: &[f64], t: f64| -> f64 {
        let idx = ((t / step).round() as usize).min(n_steps);
        assert!(
            (grid[idx] - t).abs() <= step,
            "query time off the quadrature grid"
        );
        cif[idx]
    };
    (
        times.iter().map(|&t| read_at(&cif1, t)).collect(),
        times.iter().map(|&t| read_at(&cif2, t)).collect(),
    )
}

fn predict_dataset(x_eval: &[f64]) -> EncodedDataset {
    let headers = vec!["time".to_string(), "event".to_string(), "x".to_string()];
    let rows: Vec<StringRecord> = x_eval
        .iter()
        .map(|&x| StringRecord::from(vec!["1.0".to_string(), "1".to_string(), format!("{x:.12}")]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode predict rows")
}

#[test]
fn joint_competing_risks_transformation_recovers_true_cause_specific_cifs() {
    let (time, event, x) = simulate();
    let n_events_1 = event.iter().filter(|&&e| e == 1.0).count();
    let n_events_2 = event.iter().filter(|&&e| e == 2.0).count();
    assert!(
        n_events_1 > 80 && n_events_2 > 80,
        "fixture must populate both causes well, got cause1={n_events_1} cause2={n_events_2}"
    );

    let dir = tempfile::tempdir().expect("create tempdir");
    let train_path = dir.path().join("train.csv");
    let model_path = dir.path().join("model.json");
    write_training_csv(&train_path, &time, &event, &x);

    // The documented competing-risks entry point: event codes 1..K trigger the
    // joint cause-specific transformation fit. This is the call that #1025
    // reported as always refused by the identifiability audit.
    let mut fit_cmd = Command::new(gam::gam_binary!());
    fit_cmd
        .arg("fit")
        .arg(&train_path)
        .arg("Surv(time, event) ~ x")
        .args(["--survival-likelihood", "transformation"])
        .arg("--out")
        .arg(&model_path);
    run_or_panic(fit_cmd, "gam fit joint competing risks (transformation)");
    assert!(model_path.is_file(), "gam fit did not write {model_path:?}");

    let model = FittedModel::load_from_path(&model_path).expect("load saved competing-risks model");

    let x_eval = [0.1, 0.25, 0.5, 0.75, 0.9];
    let grid = [0.3, 0.6, 0.9, 1.2, 1.6, 2.0];
    let dataset = predict_dataset(&x_eval);
    let col_map = dataset.column_map();
    let payload = model.payload();
    let training_headers = payload.training_headers.as_ref();
    let n_rows = dataset.values.nrows();
    let primary_offset = Array1::<f64>::zeros(n_rows);
    let noise_offset = Array1::<f64>::zeros(n_rows);

    let request = SurvivalPredictRequest {
        model: &model,
        data: dataset.values.view(),
        col_map: &col_map,
        training_headers,
        primary_offset: &primary_offset,
        noise_offset: &noise_offset,
        time_grid: Some(&grid),
        with_uncertainty: false,
        estimand: gam::families::survival::predict::SurvivalPredictEstimand::Plugin,
    };
    let result =
        predict_competing_risks_survival(request, SurvivalPredictionCovarianceMode::Conditional)
            .expect("competing-risks survival predict");

    assert_eq!(result.cif.len(), 2, "expected K=2 cause-specific CIFs");
    assert_eq!(result.overall_survival.nrows(), n_rows);
    assert_eq!(result.overall_survival.ncols(), grid.len());
    for cif in &result.cif {
        assert_eq!(cif.nrows(), n_rows);
        assert_eq!(cif.ncols(), grid.len());
    }

    // Structural identities: bounds, monotonicity, total probability.
    for i in 0..n_rows {
        for j in 0..grid.len() {
            let s = result.overall_survival[[i, j]];
            assert!(
                s.is_finite() && (-1e-9..=1.0 + 1e-9).contains(&s),
                "overall survival out of [0,1]: S[{i},{j}]={s}"
            );
            if j > 0 {
                assert!(
                    s <= result.overall_survival[[i, j - 1]] + 1e-9,
                    "overall survival must be non-increasing in t (row {i})"
                );
            }
            let mut cif_sum = 0.0;
            for (k, cif) in result.cif.iter().enumerate() {
                let f = cif[[i, j]];
                assert!(
                    f.is_finite() && (-1e-9..=1.0 + 1e-9).contains(&f),
                    "CIF_{k} out of [0,1] at [{i},{j}]: {f}"
                );
                if j > 0 {
                    assert!(
                        f >= cif[[i, j - 1]] - 1e-9,
                        "CIF_{k} must be non-decreasing in t (row {i})"
                    );
                }
                cif_sum += f;
            }
            let total = s + cif_sum;
            assert!(
                (total - 1.0).abs() < 1e-3,
                "total-probability identity violated at [{i},{j}]: S + sum CIF = {total}"
            );
        }
    }

    // PRIMARY: cause-specific CIF truth recovery (RMSE over the time × x grid).
    let mut sq_err = [0.0_f64; 2];
    let mut count = 0usize;
    let mut truth = vec![[0.0_f64; 2]; n_rows * grid.len()];
    for (i, &xv) in x_eval.iter().enumerate() {
        let (t1, t2) = true_cifs(xv, &grid);
        for j in 0..grid.len() {
            truth[i * grid.len() + j] = [t1[j], t2[j]];
            sq_err[0] += (result.cif[0][[i, j]] - t1[j]).powi(2);
            sq_err[1] += (result.cif[1][[i, j]] - t2[j]).powi(2);
            count += 1;
        }
    }
    let rmse_1 = (sq_err[0] / count as f64).sqrt();
    let rmse_2 = (sq_err[1] / count as f64).sqrt();
    assert!(
        rmse_1 <= 0.06,
        "cause-1 CIF truth-recovery RMSE too large: {rmse_1:.4} > 0.06"
    );
    assert!(
        rmse_2 <= 0.06,
        "cause-2 CIF truth-recovery RMSE too large: {rmse_2:.4} > 0.06"
    );

    // Covariate-direction catastrophe guard: cause 2's hazard rises with x, so
    // its true incidence gap CIF_2(t | x=0.9) - CIF_2(t | x=0.1) is large and
    // positive at t=1.2; the fit must recover a substantial share of it (a
    // flipped or flattened covariate effect fails here even if pointwise RMSE
    // sneaks under the bound).
    let j_mid = 3; // t = 1.2
    let true_gap_2 = truth[(n_rows - 1) * grid.len() + j_mid][1] - truth[j_mid][1];
    assert!(
        true_gap_2 > 0.05,
        "fixture sanity: true cause-2 incidence gap should be material, got {true_gap_2:.4}"
    );
    let fit_gap_2 = result.cif[1][[n_rows - 1, j_mid]] - result.cif[1][[0, j_mid]];
    assert!(
        fit_gap_2 >= 0.25 * true_gap_2,
        "fitted cause-2 covariate effect lost: gap {fit_gap_2:.4} vs true {true_gap_2:.4}"
    );
}
