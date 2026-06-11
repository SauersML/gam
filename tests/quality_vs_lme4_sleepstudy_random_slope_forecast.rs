//! End-to-end quality on a REAL longitudinal dataset: gam's random-slope model
//! must FORECAST held-out within-subject observations at least as accurately as
//! `lme4::lmer` (the mature mixed-model reference, demoted here to a
//! baseline-to-match-or-beat).
//!
//! DATA — the classic `sleepstudy` panel (Belenky et al. 2003, distributed with
//! lme4): average reaction time (ms) on a psychomotor-vigilance task for 18
//! subjects measured once per day across 10 days of sleep deprivation (Days
//! 0..9). A perfectly balanced grouped/longitudinal design — repeated measures
//! nested within subject, with a within-group covariate (Days). Each subject has
//! their own baseline reaction time AND their own rate of deterioration, the
//! textbook random-intercept + random-slope situation.
//!   Source: https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/lme4/sleepstudy.csv
//!
//! USE-CASE / OBJECTIVE METRIC (the pass criterion): HELD-OUT PREDICTION. We
//! split each subject's series into a TRAIN window (Days 0..7) and a future TEST
//! window (Days 8..9), then ask each engine to forecast the unseen later days
//! from the subject's own early trajectory. This is exactly what a random-slope
//! model buys over a population-only fit: by partially pooling each subject's
//! slope it predicts where THAT subject is heading. We assert, on the held-out
//! Day-8/Day-9 reaction times:
//!   1. gam's forecast RMSE clears an ABSOLUTE bar tied to the data scale (well
//!      under the marginal SD of Reaction — a population-mean forecast that
//!      ignored the subject would do no better than that SD), AND
//!   2. gam's forecast RMSE is no worse than 1.10× lme4's on the SAME split.
//!
//! lme4 is fit on the IDENTICAL train rows and scored on the IDENTICAL test rows
//! purely as a BASELINE TO MATCH-OR-BEAT — never as the output to replicate.
//! A genuine shortfall here (gam over-shrinking the per-subject slope, or
//! collapsing subjects toward the population line) drives the forecast RMSE up
//! and fails the test: a real bug in gam's random-effect machinery, not a
//! tolerance knob.
//!
//! `Days + s(Days, Subject, bs="re")` is the GAM/mgcv random-slope model: a
//! fixed population intercept and Days trend, plus a parametric per-subject
//! random intercept+slope `[1, Days]` penalized by an iid ridge (one variance).
//! The ridge shrinks each subject's intercept and slope toward the FIXED
//! population values — the exact analogue of lmer's `Days + (Days | Subject)`
//! (modulo lme4's separate intercept/slope variances and their correlation).

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use std::path::Path;

const SLEEPSTUDY_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/sleepstudy.csv");

/// Days 0..=7 train each subject; Days 8 and 9 are the held-out forecast targets.
const LAST_TRAIN_DAY: f64 = 7.0;

/// One parsed row of the real sleepstudy table.
struct Obs {
    reaction: f64,
    days: f64,
    /// Subject id verbatim from the CSV (e.g. "308"); used as the grouping key.
    subject: String,
}

/// Parse `bench/datasets/sleepstudy.csv` (columns: rownames,Reaction,Days,Subject).
/// Rows are returned in FILE ORDER so the subject blocks stay contiguous and the
/// categorical level coding is identical for gam and lme4.
fn read_sleepstudy() -> Vec<Obs> {
    let text = std::fs::read_to_string(Path::new(SLEEPSTUDY_CSV)).expect("read sleepstudy.csv");
    let mut out = Vec::new();
    for (li, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || li == 0 {
            continue; // skip header
        }
        let cols: Vec<&str> = line.split(',').collect();
        assert_eq!(
            cols.len(),
            4,
            "sleepstudy row {li} has {} columns, expected 4",
            cols.len()
        );
        out.push(Obs {
            reaction: cols[1].parse().expect("Reaction column is numeric"),
            days: cols[2].parse().expect("Days column is numeric"),
            subject: cols[3].to_string(),
        });
    }
    assert_eq!(out.len(), 180, "sleepstudy.csv should carry 180 rows");
    out
}

#[test]
fn gam_sleepstudy_random_slope_forecasts_held_out_days_vs_lme4() {
    init_parallelism();

    let data = read_sleepstudy();

    // Stable subject ordering by first appearance, so the encoded categorical
    // level index for gam matches the order we iterate subjects for prediction.
    let mut subjects: Vec<String> = Vec::new();
    for obs in &data {
        if !subjects.contains(&obs.subject) {
            subjects.push(obs.subject.clone());
        }
    }
    let n_subjects = subjects.len();
    assert_eq!(n_subjects, 18, "sleepstudy has 18 subjects");

    // ---- train / test split (identical rows handed to both engines) --------
    // gam consumes a categorical Subject; its schema inferrer classifies a column
    // whose every value parses as f64 as Continuous, so we prefix the id with "S"
    // to force the Categorical kind the `re` factor requires. lme4 receives the numeric
    // subject code plus the same Days/Reaction values and rebuilds the factor.
    let mut train_rows = Vec::<StringRecord>::new();
    let mut train_days_r = Vec::<f64>::new();
    let mut train_subj_r = Vec::<f64>::new();
    let mut train_react_r = Vec::<f64>::new();

    // Held-out test points, captured both as a gam prediction grid and as plain
    // truth vectors for RMSE. Stored grouped by subject so the gam design grid
    // and lme4's per-subject prediction line up element-for-element.
    let mut test_subject_idx = Vec::<usize>::new();
    let mut test_days = Vec::<f64>::new();
    let mut test_truth = Vec::<f64>::new();

    for obs in &data {
        let subj_idx = subjects
            .iter()
            .position(|s| s == &obs.subject)
            .expect("subject was collected above");
        let subj_code = obs.subject.parse::<f64>().expect("subject id is numeric");
        if obs.days <= LAST_TRAIN_DAY {
            train_rows.push(StringRecord::from(vec![
                format!("{:.17e}", obs.days),
                format!("S{}", obs.subject),
                format!("{:.17e}", obs.reaction),
            ]));
            train_days_r.push(obs.days);
            train_subj_r.push(subj_code);
            train_react_r.push(obs.reaction);
        } else {
            test_subject_idx.push(subj_idx);
            test_days.push(obs.days);
            test_truth.push(obs.reaction);
        }
    }
    let n_test = test_truth.len();
    assert_eq!(
        n_test,
        n_subjects * 2,
        "expected 2 held-out days (8,9) per subject"
    );

    // Marginal SD of the held-out reaction times: the yardstick for the absolute
    // bar. A forecast that ignored the subject entirely (population mean) would
    // have RMSE ~ this SD; a working random-slope model must beat it handily.
    let test_mean = test_truth.iter().sum::<f64>() / n_test as f64;
    let test_sd = (test_truth
        .iter()
        .map(|v| (v - test_mean) * (v - test_mean))
        .sum::<f64>()
        / (n_test as f64 - 1.0))
        .sqrt();

    // ---- fit gam on the train window: Reaction ~ Days + s(Days, Subject, bs="re") ----
    let headers = vec![
        "Days".to_string(),
        "Subject".to_string(),
        "Reaction".to_string(),
    ];
    let ds = encode_recordswith_inferred_schema(headers, train_rows)
        .expect("encode sleepstudy train window");
    let col = ds.column_map();
    let days_idx = col["Days"];
    let subj_idx_col = col["Subject"];

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    // gam's random-slope model: a fixed population trend `Days` plus the
    // parametric random intercept+slope `s(Days, Subject, bs="re")` — the exact
    // GAM/mgcv analogue of lme4's `Days + (Days | Subject)`. The `re` random
    // effect shrinks each subject's slope toward the FIXED population slope (the
    // mixed-model BLUP), which is what makes a held-out within-subject forecast
    // borrow strength. (A `bs="fs"` factor smooth is the WRONG model here: it is
    // a wiggly per-subject curve shrunk toward zero, not toward the population
    // slope, and its cubic extrapolation overshoots — both gam's and mgcv's `fs`
    // forecast ~50 ms on this split, well above lme4's 42.6, so `fs` cannot meet
    // the bar by construction. The mature comparator for a random slope is the
    // random-effect model, and that is what we benchmark here.)
    let result = fit_from_formula("Reaction ~ Days + s(Days, Subject, bs=\"re\")", &ds, &cfg)
        .unwrap_or_else(|e| panic!("gam random-slope fit on sleepstudy: {e:?}"));
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for a gaussian random-slope model");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // Build the held-out prediction grid: each test (Subject, Day). The factor
    // column carries the encoded level index so each row is evaluated against ITS
    // OWN subject's smooth block; identity link => design*beta = predicted mean.
    let mut grid = Array2::<f64>::zeros((n_test, ds.headers.len()));
    for (row, (&subj_idx, &day)) in test_subject_idx.iter().zip(&test_days).enumerate() {
        grid[[row, days_idx]] = day;
        grid[[row, subj_idx_col]] = subj_idx as f64;
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild design at held-out grid");
    let gam_pred: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    assert_eq!(gam_pred.len(), n_test, "gam prediction length mismatch");

    // ---- fit the SAME model with lme4 on the SAME train rows ----------------
    // The R body receives ONLY the train rows (all four columns equal length),
    // rebuilds Subject as a factor, fits the correlated random intercept + slope
    // (Days | Subject), then forecasts the held-out (Subject, Day) grid in the
    // exact iteration order we used above so the emitted vector aligns with
    // gam_pred / test_truth element-for-element.
    let test_subj_code_r: Vec<f64> = test_subject_idx
        .iter()
        .map(|&i| subjects[i].parse::<f64>().expect("subject id numeric"))
        .collect();
    // Everything travels in ONE run_r call so train and forecast grid cannot
    // drift apart. All four columns must be equal length, so we stack the test
    // grid below the train rows into a single frame: an `is_train` flag separates
    // them and Reaction is NA on the test rows (lmer fits on `is_train==1`,
    // predicts on the rest). The test rows keep their stored order, which is the
    // same order gam_pred and test_truth use.
    let combined_len = train_days_r.len() + n_test;
    let mut all_days = train_days_r.clone();
    all_days.extend_from_slice(&test_days);
    let mut all_subj = train_subj_r.clone();
    all_subj.extend_from_slice(&test_subj_code_r);
    let mut all_react = train_react_r.clone();
    all_react.extend(std::iter::repeat(f64::NAN).take(n_test));
    let mut is_train = vec![1.0_f64; train_days_r.len()];
    is_train.extend(std::iter::repeat(0.0).take(n_test));
    assert_eq!(all_days.len(), combined_len);
    assert_eq!(all_subj.len(), combined_len);
    assert_eq!(all_react.len(), combined_len);
    assert_eq!(is_train.len(), combined_len);

    let r = run_r(
        &[
            Column::new("Days", &all_days),
            Column::new("Subject", &all_subj),
            Column::new("Reaction", &all_react),
            Column::new("is_train", &is_train),
        ],
        r#"
        suppressPackageStartupMessages(library(lme4))
        df$Subject <- factor(df$Subject)
        train <- df[df$is_train == 1, ]
        test  <- df[df$is_train == 0, ]
        m <- lmer(Reaction ~ Days + (Days | Subject), data = train,
                  control = lmerControl(check.conv.singular = "ignore"))
        # Forecast the held-out rows in their stored order (the test frame keeps
        # the row order of the combined data, which is exactly the order gam and
        # the Rust truth vector use).
        pred <- predict(m, newdata = test)
        emit("pred", as.numeric(pred))
        "#,
    );
    let lme4_pred = r.vector("pred");
    assert_eq!(
        lme4_pred.len(),
        n_test,
        "lme4 returned {} forecasts, expected {n_test}",
        lme4_pred.len()
    );

    // ---- held-out forecast errors: gam (primary) and lme4 (baseline) -------
    let gam_rmse = rmse(&gam_pred, &test_truth);
    let lme4_rmse = rmse(lme4_pred, &test_truth);

    eprintln!(
        "sleepstudy random-slope forecast (Days 0..7 train, 8..9 test): \
         subjects={n_subjects} n_test={n_test} gam_edf={gam_edf:.2}\n  \
         held-out RMSE  gam={gam_rmse:.3} ms  lme4={lme4_rmse:.3} ms  \
         (marginal SD of held-out Reaction = {test_sd:.3} ms)"
    );

    // (1) ABSOLUTE — a working random-slope forecaster must beat the
    // subject-blind population-mean forecast by a wide margin. That naive
    // baseline has RMSE ~ the marginal SD of the held-out reaction times; we
    // require gam to come in under 70% of that SD. Forecasting two days past the
    // training window is genuine extrapolation, so the bar is generous in
    // absolute terms while still proving the per-subject trend was learned.
    let abs_bar = 0.70 * test_sd;
    assert!(
        gam_rmse <= abs_bar,
        "gam held-out forecast missed the bar: RMSE={gam_rmse:.3} ms > {abs_bar:.3} ms \
         (0.70 * marginal SD {test_sd:.3})"
    );

    // (2) MATCH-OR-BEAT — on the IDENTICAL train/test split, gam must forecast no
    // worse than 1.10× the mature mixed-model reference. lme4 is the accuracy
    // baseline here, never the target: we beat-or-match its held-out error, we do
    // not reproduce its fit.
    assert!(
        gam_rmse <= lme4_rmse * 1.10,
        "gam held-out forecast worse than lme4 baseline: gam={gam_rmse:.3} ms > 1.10*lme4={:.3} ms",
        lme4_rmse * 1.10
    );
}
