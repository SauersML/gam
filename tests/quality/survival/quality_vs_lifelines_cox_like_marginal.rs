//! End-to-end OBJECTIVE quality: gam's **survival marginal-slope** family (a
//! semi-parametric proportional-hazards model — parametric/spline baseline plus
//! a smooth covariate effect on the survival index) measured by its **held-out
//! predictive discrimination** (Harrell's concordance index), with
//! `lifelines.CoxPHFitter` demoted to a baseline-to-match-or-beat on that same
//! held-out metric.
//!
//! ## The objective metric: held-out concordance (Harrell's C)
//!
//! This is real survival data with **no known ground-truth hazard**, so the
//! honest objective claim is *predictive accuracy on data the model never saw*
//! (objective category #2 / #4). We make a deterministic, fixed-seed train/test
//! split (no randomness — a reproducible interleaved partition by row index),
//! fit gam on the **train** rows only, and score the **test** rows with gam's own
//! forward map. The quality metric is **Harrell's concordance index** on the test
//! set: over all comparable pairs (the earlier-time subject had an event), the
//! fraction whose predicted risk order agrees with the observed event order. C=1
//! is perfect risk ranking, C=0.5 is a coin flip. Concordance is censoring-aware,
//! rank-based, and link-agnostic, so it compares the *predictive quality* of two
//! differently-parameterized hazard models on a common, objective footing — it
//! does NOT reward gam for reproducing lifelines' fitted numbers.
//!
//! ### gam's predicted risk score
//!
//! gam's survival link is `S(t | z) = Φ(−η)`,
//!   η = q(t)·c(g) + (probit_scale · g) · z_std,
//! where `z` is the modeled covariate (here EJECTION_FRACTION), `g` is the per-row
//! slope (`baseline_slope + slope_design·β_slope`, with
//! `slope = s(age, bs='tp', k=4)` — an age-modulated EF effect; the z column
//! itself is structurally reserved as the latent score and cannot appear in the
//! slope surface), and SEX + AGE enter the marginal block. The cumulative
//! hazard is `Λ = −log Φ(−η)`, strictly increasing in η. We evaluate the
//! saved model's posterior-mean cumulative hazard at a common 100-day horizon
//! through the production survival predictor. This includes the learned time,
//! marginal and slope terms and integrates their coefficient uncertainty.
//!
//! ## Data — real, identical rows to both engines
//!
//! `heart_failure_clinical_records_dataset.csv` (n=299: 96 deaths, 203 censored,
//! i.e. a ~32% event rate / ~68% right-censoring rate). Event is
//! `DEATH_EVENT`, follow-up is `time` (days). Right-censored shorthand
//! `Surv(time, DEATH_EVENT)`. Covariates: `ejection_fraction` is the modeled
//! smooth covariate (gam's latent score `z`; Cox's continuous covariate), `sex`
//! and `age` enter linearly. The SAME deterministic train rows fit both engines;
//! the SAME test rows are scored by both.
//!
//! ## Assertions — objective, never "close to the reference's output"
//!
//!   1. **Absolute discrimination bar (PRIMARY)**: gam's held-out concordance
//!      `C_test(gam) ≥ 0.62`. EJECTION_FRACTION + AGE + SEX are clinically
//!      predictive of heart-failure mortality; a model with real signal must beat
//!      a coin flip by a clear margin. This is gam's own predictive quality, not a
//!      comparison to anyone.
//!   2. **Match-or-beat the mature baseline (ACCURACY)**: `C_test(gam) ≥
//!      C_test(cox) − 0.03`. lifelines' CoxPHFitter is fit on the identical train
//!      rows and scored on the identical test rows; gam must be at least as good a
//!      risk-discriminator (within a small tolerance for the genuine link
//!      difference). gam is allowed to *win*; it is not allowed to lose materially.
//!   3. **Survival-structure invariant (STRUCTURE)**: gam's reconstructed
//!      posterior-mean cumulative hazard is finite and strictly positive, and
//!      across the held-out EF range it is **monotone** in the covariate
//!      (successive Λ over sorted EF are non-increasing within numerical eps),
//!      i.e. gam encodes a single coherent protective EF gradient — a real
//!      property of the fitted survival function, asserted directly.
//!
//! We do NOT assert pointwise closeness of gam's HR curve to Cox's exp(β·Δ); two
//! different links need not coincide, and matching a peer tool's noisy fit proves
//! nothing. The quality bars are the original issue's acceptance requirements.

use gam::test_support::reference::{Column, run_python};
use gam::{FitConfig, init_parallelism, load_csvwith_inferred_schema};
use gam_models::inference::model::FittedModel;
use gam_models::inference::model_payload_builders::fit_formula_to_payload;
use gam_models::survival::predict::{SurvivalPredictEstimand, SurvivalPredictRequest, SurvivalPredictionCovarianceMode, harrell_concordance, predict_survival};
use ndarray::{Array1, Axis};
use std::path::Path;
use std::time::Instant;

#[test]
fn gam_marginal_slope_heldout_concordance_matches_or_beats_lifelines_coxph() {
    init_parallelism();
    let ds = load_csvwith_inferred_schema(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/heart_failure_clinical_records_dataset.csv"))).expect("heart-failure observations");
    assert_eq!(ds.values.nrows(), 299);
    let columns = ds.column_map();
    let train_rows: Vec<usize> = (0..299).filter(|i| i % 3 != 0).collect();
    let test_rows: Vec<usize> = (0..299).filter(|i| i % 3 == 0).collect();
    let mut train = ds.clone();
    train.values = ds.values.select(Axis(0), &train_rows);
    let test = ds.values.select(Axis(0), &test_rows);
    let config = FitConfig {
        survival_likelihood: Some("marginal-slope".into()),
        z_column: Some("ejection_fraction".into()),
        slope_formula: Some("s(age, bs='tp', k=4)".into()),
        baseline_target: "linear".into(),
        ..FitConfig::default()
    };
    let started = Instant::now();
    let model = FittedModel::from_payload(fit_formula_to_payload("Surv(time, DEATH_EVENT) ~ sex + age".into(), &train, &config).expect("converged survival model payload"));
    let fit_seconds = started.elapsed().as_secs_f64();
    eprintln!("#1082 survival fit: {fit_seconds:.3} seconds");
    // Exercise the saved model's actual posterior-mean survival surface at a
    // common follow-up horizon, including the fitted marginal and slope terms.
    let evaluate = |values: &ndarray::Array2<f64>| {
        let offset = Array1::zeros(values.nrows());
        predict_survival(SurvivalPredictRequest {
            model: &model,
            data: values.view(),
            col_map: &columns,
            training_headers: Some(&train.headers),
            primary_offset: &offset,
            noise_offset: &offset,
            time_grid: Some(&[100.0]),
            with_uncertainty: false,
            estimand: SurvivalPredictEstimand::PosteriorMean,
        }, SurvivalPredictionCovarianceMode::SmoothingCorrected).expect("production survival prediction")
    };
    let predicted = evaluate(&test);
    let gam_risk = predicted.cumulative_hazard.column(0).to_vec();
    assert!(gam_risk.iter().all(|&risk| risk.is_finite() && risk > 0.0));
    let test_time = test.column(columns["time"]).to_vec();
    let test_event = test.column(columns["DEATH_EVENT"]).to_vec();
    let gam_c = harrell_concordance(&test_time, &test_event, &gam_risk).expect("comparable held-out pairs");
    let names = ["time", "DEATH_EVENT", "ejection_fraction", "sex", "age"];
    let training_columns: Vec<Vec<f64>> = names.iter().map(|name| train.values.column(columns[*name]).to_vec()).collect();
    let testing_columns: Vec<Vec<f64>> = names.iter().map(|name| test.column(columns[*name]).to_vec()).collect();
    let train_names: Vec<String> = names.iter().map(|name| format!("train_{name}")).collect();
    let test_names: Vec<String> = names.iter().map(|name| format!("test_{name}")).collect();
    let reference_columns: Vec<Column<'_>> = train_names.iter().zip(&training_columns).chain(test_names.iter().zip(&testing_columns)).map(|(name, values)| Column::new(name, values)).collect();
    let reference = run_python(&reference_columns, r#"
from lifelines import CoxPHFitter
names = ['time', 'DEATH_EVENT', 'ejection_fraction', 'sex', 'age']
train = pd.DataFrame({name: df['train_' + name] for name in names}).dropna()
test = pd.DataFrame({name: df['test_' + name] for name in names}).dropna()
model = CoxPHFitter().fit(train, duration_col='time', event_col='DEATH_EVENT')
emit('risk', np.asarray(model.predict_partial_hazard(test)).reshape(-1))
"#);
    assert_eq!(reference.vector("risk").len(), test_rows.len());
    let cox_c = harrell_concordance(&test_time, &test_event, reference.vector("risk")).expect("Cox comparable pairs");
    eprintln!("#1082 heart failure: posterior_mean_concordance={gam_c}, cox_concordance={cox_c}");
    assert!(gam_c >= 0.62, "held-out concordance={gam_c}");
    assert!(gam_c >= cox_c - 0.03, "gam={gam_c}, Cox={cox_c}");

    let mut grid = ndarray::Array2::from_shape_fn((61, ds.headers.len()), |(_, column)| train.values[[0, column]]);
    for i in 0..grid.nrows() {
        grid[[i, columns["age"]]] = 60.0;
        grid[[i, columns["sex"]]] = 0.0;
        grid[[i, columns["ejection_fraction"]]] = 20.0 + i as f64;
    }
    let surface = evaluate(&grid);
    let hazard = surface.cumulative_hazard.column(0).to_vec();
    assert!(hazard.iter().all(|&h| h.is_finite() && h > 0.0));
    assert!(hazard.windows(2).all(|pair| pair[1] <= pair[0] + 1e-10), "ejection fraction must have a coherent protective gradient");
    assert!(fit_seconds <= 120.0, "#1082 survival fit exceeded 120 seconds: {fit_seconds:.3}");
}
