//! Regression for #896: the predicted survival surface for fixed covariates and
//! an explicit set of query times is a property of the fitted model and the
//! query times alone. It must NOT depend on the prediction frame's `exit`
//! placeholder, and an ordinary in-fitted-range query time must never be
//! silently collapsed to the `t → ∞` asymptote (`S = 0`).
//!
//! Root cause of the user-visible bug (fixed in `crates/gam-pyffi/src/lib.rs`,
//! `default_survival_time_grid`): the Python `survival_at(times)` surface grid
//! was bounded above by `max(exit)` of the PREDICTION rows, so a small
//! placeholder `exit` shrank the grid and the asymptotic extrapolation past the
//! grid's right edge then forced in-range times to `S = 0`.
//!
//! The library evaluator `gam::families::survival::predict::predict_survival` — the path
//! every FFI surface query ultimately drives — is the contract the grid fix
//! relies on: when handed an explicit `time_grid` it must evaluate the model at
//! exactly those times, reusing the SAVED training-time basis (knots / anchor),
//! independent of the per-row `age_exit`. This test pins that contract directly:
//! it fits a real survival model, then predicts the SAME covariates on the SAME
//! explicit grid twice — once with a tiny `exit` placeholder (`1.0`) and once
//! with a large one (`exit.max() + 5`) — and asserts the two surfaces are
//! identical and the in-range times are not collapsed to zero. Before the fix
//! the small-placeholder surface truncated; with the evaluator honouring the
//! explicit grid the two surfaces agree to floating-point tolerance.
//!
//! The fit uses `--survival-likelihood weibull`, whose baseline log-cumulative-
//! hazard is the `[1, log t]` linear time basis with a strictly positive
//! derivative `d/dt log Λ = 1/t > 0` at every `t > 0`. That keeps the test
//! squarely on the #896 contract (placeholder-independence of the surface) and
//! away from the unrelated degenerate-baseline regime: a flexible
//! `transformation`/Royston–Parmar I-spline fit on a small synthetic dataset can
//! legitimately drive its baseline log-cumhaz derivative to exactly `0` in a flat
//! region, where `royston_parmar_survival_hazard_components` correctly rejects
//! `η_t = 0` (a zero hazard rate is not a valid RP hazard). That rejection is a
//! property of the synthetic baseline shape, not of the exit placeholder, so it
//! is out of scope for the #896 surface-independence contract; Weibull's
//! positive log-t baseline slope avoids it.

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

const N: usize = 600;

/// Deterministic right-censored Weibull-shaped data with substantial mortality
/// (so the survival surface drops well below 1 inside the observed range),
/// mirroring the failing Python spec
/// `test_bug_hunt_survival_surface_truncated_by_prediction_exit.py`. A fixed LCG
/// keeps the fixture reproducible with no RNG dependency.
fn build_dataset() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut state: u64 = 0x2545F4914F6CDD1D;
    let mut next_u01 = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };

    let shape = 1.5_f64;
    let mut age = Vec::with_capacity(N);
    let mut exit = Vec::with_capacity(N);
    let mut event = Vec::with_capacity(N);
    for _ in 0..N {
        let a = 40.0 + 35.0 * next_u01();
        // High-hazard log-linear AFT (matches the #897 fixture's mortality):
        // most subjects fail inside the observed window, so a correct surface is
        // genuinely curved and S(t) drops below 1 at in-range times.
        let eta = -2.0 + 0.05 * (a - 55.0);
        let u = 1e-9_f64.max(next_u01());
        let t_lat = (-eta / shape).exp() * (-u.ln()).powf(1.0 / shape);
        let cens = (-next_u01().max(1e-12).ln() * 20.0).min(30.0);
        let ex = t_lat.min(cens);
        let ev = if t_lat <= cens { 1.0 } else { 0.0 };
        age.push(a);
        exit.push(ex);
        event.push(ev);
    }
    (age, exit, event)
}

fn write_training_csv(path: &Path, age: &[f64], exit: &[f64], event: &[f64]) {
    let mut writer = csv::Writer::from_path(path).expect("create training csv");
    writer
        .write_record(["entry", "exit", "event", "age"])
        .expect("write header");
    for i in 0..age.len() {
        writer
            .write_record([
                "0.0".to_string(),
                format!("{:.12}", exit[i]),
                format!("{}", event[i] as i64),
                format!("{:.12}", age[i]),
            ])
            .expect("write training row");
    }
    writer.flush().expect("flush training csv");
}

/// One predict row at a fixed `age` covariate with the supplied `exit`
/// placeholder. The covariates and query grid are identical across calls; only
/// the placeholder `exit` differs.
fn predict_dataset(exit_placeholder: f64) -> EncodedDataset {
    let headers = vec![
        "entry".to_string(),
        "exit".to_string(),
        "event".to_string(),
        "age".to_string(),
    ];
    let rows = vec![StringRecord::from(vec![
        "0.0".to_string(),
        format!("{exit_placeholder:.12}"),
        "1".to_string(),
        "57.0".to_string(),
    ])];
    encode_recordswith_inferred_schema(headers, rows).expect("encode predict row")
}

/// Drive `predict_survival` on `grid` for a single predict row carrying the
/// given `exit` placeholder, returning the survival and cumulative-hazard rows.
fn surface_for_exit(
    model: &FittedModel,
    exit_placeholder: f64,
    grid: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let dataset = predict_dataset(exit_placeholder);
    let col_map = dataset.column_map();
    let payload = model.payload();
    let training_headers = payload.training_headers.as_ref();
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
    let result = predict_survival(request, SurvivalPredictionCovarianceMode::Conditional).expect("library survival predict");
    assert_eq!(result.survival.nrows(), 1, "expected one prediction row");
    assert_eq!(
        result.survival.ncols(),
        grid.len(),
        "survival surface must cover every grid time"
    );
    (
        result.survival.row(0).to_vec(),
        result.cumulative_hazard.row(0).to_vec(),
    )
}

#[test]
fn survival_surface_is_independent_of_prediction_exit_placeholder() {
    let (age, exit, event) = build_dataset();

    let dir = tempfile::tempdir().expect("create tempdir");
    let train_path = dir.path().join("train.csv");
    let model_path = dir.path().join("model.json");
    write_training_csv(&train_path, &age, &exit, &event);

    let mut fit_cmd = Command::new(gam::gam_binary!());
    fit_cmd
        .arg("fit")
        .arg(&train_path)
        .arg("Surv(entry, exit, event) ~ s(age)")
        .args(["--survival-likelihood", "weibull"])
        .arg("--out")
        .arg(&model_path);
    run_or_panic(fit_cmd, "gam fit Surv ~ s(age) (weibull)");
    assert!(model_path.is_file(), "gam fit did not write {model_path:?}");

    let model =
        FittedModel::load_from_path(&model_path).expect("load saved Weibull survival model");

    // All query times are well inside the fitted time range (median exit ~2.5,
    // max ~16), so the model has a well-defined, curved survival surface here.
    let grid = [1.0_f64, 3.0, 6.0, 12.0];
    let big_exit = exit.iter().cloned().fold(f64::MIN, f64::max) + 5.0;

    // Surface evaluated with a tiny exit placeholder (the #896 trigger) vs. a
    // large one. Both drive the SAME explicit grid against the SAME saved model
    // and covariates, so they must agree.
    let (surv_small, cum_small) = surface_for_exit(&model, 1.0, &grid);
    let (surv_big, cum_big) = surface_for_exit(&model, big_exit, &grid);

    for (i, (&s_small, &s_big)) in surv_small.iter().zip(surv_big.iter()).enumerate() {
        assert!(
            (s_small - s_big).abs() <= 1e-9,
            "survival at t={} depends on the prediction row's `exit` placeholder: \
             S(exit=1)={surv_small:?} vs S(exit={big_exit})={surv_big:?} (#896). The \
             explicit-grid surface must be a function of the model and query times \
             alone.",
            grid[i]
        );
    }
    for (i, (&c_small, &c_big)) in cum_small.iter().zip(cum_big.iter()).enumerate() {
        assert!(
            (c_small - c_big).abs() <= 1e-9,
            "cumulative hazard at t={} depends on the `exit` placeholder: \
             H(exit=1)={cum_small:?} vs H(exit={big_exit})={cum_big:?} (#896).",
            grid[i]
        );
    }

    // The #896 truncation manifested as the SMALL-placeholder surface collapsing
    // to the `t → ∞` asymptote (S ≈ 0) at query times PAST the placeholder, while
    // the large-placeholder surface stayed on the real curve. That exact divergence
    // is already caught to 1e-9 by the placeholder-independence loop above: a
    // truncated small-placeholder column could not match the big-placeholder one.
    // We additionally anchor non-degeneracy on the EARLIEST in-range time, which
    // for this median-~2.5 model sits deep in the high-survival region — a uniform
    // collapse to S ≈ 0 everywhere would fail this. Late grid times (t = 12) are
    // LEGITIMATELY near zero for a high-mortality fit (S(12) ≈ 2.5e-4 is the model's
    // true survival, not a truncation), so they are deliberately NOT asserted above
    // a floor — doing so would contradict the real curve rather than detect #896.
    assert!(
        surv_small[0] > 0.5,
        "earliest in-range time t={} collapsed toward the asymptote (S≈0): \
         S={surv_small:?} (#896)",
        grid[0]
    );
    // And the surface is genuinely curved (a non-degenerate fit), so the
    // placeholder-independence above is meaningful and not the trivial S≡1 case.
    let min_surv = surv_big.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        min_surv < 0.85,
        "survival surface is degenerate (≈1) over the grid, so the test would not \
         exercise the #896 truncation contract: S={surv_big:?}"
    );

    // Surfaces are valid survival functions: finite, in [0,1], and monotone
    // non-increasing, with S = exp(-H) holding on the shared grid.
    for surf in [&surv_small, &surv_big] {
        assert!(
            surf.iter()
                .all(|s| s.is_finite() && (0.0..=1.0).contains(s)),
            "survival surface must be finite and in [0,1]: {surf:?}"
        );
        for w in surf.windows(2) {
            assert!(
                w[1] <= w[0] + 1e-9,
                "survival surface must be monotone non-increasing in t: {surf:?}"
            );
        }
    }
    for (&s, &h) in surv_big.iter().zip(cum_big.iter()) {
        assert!(
            (s - (-h).exp()).abs() <= 1e-5,
            "S = exp(-H) must hold on the shared grid: S={s}, H={h}, exp(-H)={}",
            (-h).exp()
        );
    }
}
