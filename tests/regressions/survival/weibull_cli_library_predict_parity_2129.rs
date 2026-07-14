//! Root-cause-class regression for #2129: the CLI `gam predict` survival surface
//! must AGREE with the in-process library `predict_survival` surface for the
//! same saved single-cause Weibull model.
//!
//! #2129 was, at root, the CLI predict path *diverging* from the library predict
//! path: the library already centered the `[1, log t]` linear time basis at the
//! survival anchor and carried a zero parametric baseline offset (its #897 fix),
//! while the CLI rebuilt the basis un-centered AND re-added the recovered Weibull
//! baseline as a parametric offset, double-counting `k·log t`. Each path is now
//! guarded on its own — the library surface by
//! `weibull_survival_predict_not_degenerate` (#897) and the CLI surface by
//! `weibull_cli_predict_baseline_double_counted_2129` / `_regimes_2129` (#2129) —
//! but nothing pins the two paths to EACH OTHER. A future change that re-broke
//! only one path (e.g. reintroduced the CLI offset, or dropped the library
//! centering) could still satisfy a truth-tracking tolerance on its own fixture
//! while silently re-opening the divergence this test forbids.
//!
//! This test fits ONE model through the real `gam fit` Weibull path, then
//! predicts the same `x = 0` time grid two ways — through the `gam predict` CLI
//! and through `gam::families::survival::predict::predict_survival` in-process —
//! and asserts the two survival curves (and the log-cumulative-hazard `eta`)
//! match to a tight numerical tolerance. Under the old double-count the CLI eta
//! was ~2× the library eta, so the surfaces disagreed grossly; parity holds only
//! while both paths carry the baseline exactly once.

use std::path::Path;
use std::process::Command;

use csv::StringRecord;
use gam::encode_recordswith_inferred_schema;
use gam::families::survival::predict::{
    SurvivalPredictRequest, SurvivalPredictionCovarianceMode, predict_survival,
};
use gam::inference::data::EncodedDataset;
use gam::inference::model::FittedModel;
use gam::test_support::cli_harness::run_or_panic;
use ndarray::Array1;

const GRID_TIMES: [f64; 5] = [2.0, 4.0, 7.0, 12.0, 20.0];
const TRUE_SHAPE: f64 = 1.4;
const TRUE_SCALE: f64 = 9.0;
const TRUE_BETA: f64 = 0.6;
const N: usize = 2500;

/// Deterministic Weibull proportional-hazards sample (inverse-CDF draw), matching
/// the generator style of the sibling #2129 fixtures.
fn build_training_rows() -> Vec<(f64, i64, f64)> {
    let mut state: u64 = 0xD1B54A32D192ED03;
    let mut next_u01 = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((state >> 11) as f64) / ((1u64 << 53) as f64)).clamp(1e-12, 1.0 - 1e-12)
    };

    let mut rows = Vec::with_capacity(N);
    for _ in 0..N {
        let u1 = next_u01();
        let u2 = next_u01();
        let x = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
        let u = next_u01();
        let t_event =
            TRUE_SCALE * (-TRUE_BETA * x / TRUE_SHAPE).exp() * (-u.ln()).powf(1.0 / TRUE_SHAPE);
        let admin = 45.0;
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

/// CLI predict grid: one row per grid time at the baseline covariate `x = 0`.
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

/// Library predict input: a SINGLE row at `x = 0` with a large `exit` placeholder
/// (so every queried grid time is inside the surface frame), queried on
/// `time_grid = GRID_TIMES`.
fn library_predict_dataset(big_exit: f64) -> EncodedDataset {
    let headers = vec![
        "t0".to_string(),
        "t1".to_string(),
        "event".to_string(),
        "x".to_string(),
    ];
    let rows = vec![StringRecord::from(vec![
        "0.0".to_string(),
        format!("{big_exit:.12}"),
        "1".to_string(),
        "0.0".to_string(),
    ])];
    encode_recordswith_inferred_schema(headers, rows).expect("encode library predict row")
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

#[test]
fn weibull_cli_and_library_predict_surfaces_agree() {
    let dir = tempfile::tempdir().expect("create tempdir");
    let train = dir.path().join("surv_train.csv");
    let grid = dir.path().join("surv_grid.csv");
    let model_path = dir.path().join("m_wb.json");
    let cli_pred = dir.path().join("p_cli.csv");
    write_training_csv(&train, &build_training_rows());
    write_grid_csv(&grid);

    // Fit once through the real Weibull CLI path.
    let mut fit_cmd = Command::new(gam::gam_binary!());
    fit_cmd
        .arg("fit")
        .arg(&train)
        .arg("Surv(t0,t1,event) ~ x")
        .args(["--survival-likelihood", "weibull"])
        .arg("--out")
        .arg(&model_path);
    run_or_panic(fit_cmd, "gam fit Surv(t0,t1,event) ~ x (weibull)");
    assert!(model_path.is_file(), "gam fit did not write {model_path:?}");

    // Path A — the CLI predict surface at x = 0.
    let mut predict_cmd = Command::new(gam::gam_binary!());
    predict_cmd
        .arg("predict")
        .arg(&model_path)
        .arg(&grid)
        .args(["--mode", "map"])
        .arg("--out")
        .arg(&cli_pred);
    run_or_panic(predict_cmd, "gam predict (map)");
    let cli_survival = read_column(&cli_pred, "survival_prob");
    let cli_eta = read_column(&cli_pred, "eta");

    // Path B — the in-process library predict surface for the same model, same
    // x = 0 covariate, same time grid.
    let model =
        FittedModel::load_from_path(&model_path).expect("load saved Weibull survival model");
    let payload = model.payload();
    let big_exit = GRID_TIMES[GRID_TIMES.len() - 1] + 5.0;
    let dataset = library_predict_dataset(big_exit);
    let col_map = dataset.column_map();
    let training_headers = payload.training_headers.as_ref();
    let n = dataset.values.nrows();
    let primary_offset = Array1::<f64>::zeros(n);
    let noise_offset = Array1::<f64>::zeros(n);
    let request = SurvivalPredictRequest {
        model: &model,
        data: dataset.values.view(),
        col_map: &col_map,
        training_headers,
        primary_offset: &primary_offset,
        noise_offset: &noise_offset,
        time_grid: Some(&GRID_TIMES),
        with_uncertainty: false,
        estimand: gam::families::survival::predict::SurvivalPredictEstimand::Plugin,
    };
    let lib = predict_survival(request, SurvivalPredictionCovarianceMode::Conditional).expect("library Weibull survival predict");
    assert_eq!(
        lib.survival.nrows(),
        1,
        "expected one library prediction row"
    );
    assert_eq!(
        lib.survival.ncols(),
        GRID_TIMES.len(),
        "library survival surface must cover every grid time"
    );
    let lib_survival: Vec<f64> = lib.survival.row(0).to_vec();
    // eta = log H(t) = log(cumulative hazard).
    let lib_eta: Vec<f64> = lib
        .cumulative_hazard
        .row(0)
        .iter()
        .map(|h| h.ln())
        .collect();

    assert_eq!(
        cli_survival.len(),
        GRID_TIMES.len(),
        "CLI row count mismatch"
    );

    // The two paths compute the identical quantity from the identical saved
    // coefficients; they must agree to a tight numerical tolerance. Under the
    // old double-count the CLI eta was ~2× the library eta, so this diff was
    // O(1) in eta and O(0.5) in survival — many orders of magnitude above the
    // tolerance here.
    let mut max_surv_diff = 0.0_f64;
    let mut max_eta_diff = 0.0_f64;
    for i in 0..GRID_TIMES.len() {
        max_surv_diff = max_surv_diff.max((cli_survival[i] - lib_survival[i]).abs());
        max_eta_diff = max_eta_diff.max((cli_eta[i] - lib_eta[i]).abs());
    }
    assert!(
        max_surv_diff < 1e-6,
        "CLI and library Weibull predict survival surfaces diverge: \
         cli = {cli_survival:?}, lib = {lib_survival:?} (max |Δ| = {max_surv_diff:.3e}). \
         The two predict paths must carry the baseline identically (#2129).",
    );
    assert!(
        max_eta_diff < 1e-6,
        "CLI and library Weibull predict log-cumulative-hazard (eta) diverge: \
         cli = {cli_eta:?}, lib = {lib_eta:?} (max |Δ| = {max_eta_diff:.3e}). \
         A ~2× ratio here is the #2129 double-count resurfacing on the CLI path.",
    );
}
