//! End-to-end quality: gam's cyclic cubic smooth (`cc()` / `bs="cc"`) must
//! PREDICT a real seasonal cycle on held-out data — and enforce the periodic
//! wrap — at least as well as **mgcv**, the mature, standard GAM implementation.
//!
//! DATASET: Nottingham Castle monthly average air temperatures, Jan 1920 – Dec
//! 1939 (240 months), R's `datasets::nottem` exported by the Rdatasets project:
//! https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/nottem.csv
//! Vincent Arel-Bundock's Rdatasets index:
//! https://vincentarelbundock.github.io/Rdatasets/datasets.html
//! The raw `time` column is a decimal year; we materialised a clean
//! `year,month,temp` CSV (`bench/datasets/nottem_monthly_temp.csv`) where
//! `month in 1..=12`. Temperature has a strong, smooth annual cycle (cold
//! Jan/Feb/Dec ~32–44 °F, warm Jun/Jul/Aug ~57–66 °F): a textbook use-case for a
//! cyclic smooth over month-of-year, where December must join back to January.
//!
//! This is a HELD-OUT PREDICTION test on REAL data — there is no analytic ground
//! truth, so objective quality is out-of-sample accuracy. A deterministic split
//! holds out every 4th row (60 of 240). We fit `temp ~ cc(month)` on the
//! training rows by REML, predict the held-out months, and assert on gam's OWN
//! predictions:
//!
//!   1. PREDICTION (primary, tool-free): held-out `R2 >= 0.85`. The seasonal
//!      signal is strong and clean, so a competent cyclic smooth explains the
//!      vast majority of held-out variance — far above the constant-mean
//!      predictor (R2 = 0) and high enough to catch under/over-smoothing.
//!   2. MATCH-OR-BEAT (accuracy): gam's held-out RMSE is no worse than mgcv's by
//!      more than 10% — `gam_test_rmse <= mgcv_test_rmse * 1.10`. mgcv fits the
//!      SAME training rows and predicts the SAME held-out months; it is a
//!      baseline to match-or-beat on ACCURACY, never a fit to reproduce.
//!   3. STRUCTURE — periodic seam continuity: gam genuinely enforces the wrap,
//!      so its fitted cyclic smooth agrees at the period endpoints
//!      `fit(month=1) == fit(month=13)` to 1e-6. This is the defining property
//!      of a cyclic basis (`bs="cc"`) and is asserted directly on gam's own fit.
//!
//! The same rows, in the same order, with the identical month encoding (period
//! [1, 13), so the Dec→Jan seam sits at the knots) are handed to both gam and
//! mgcv. "rel_l2 to mgcv's in-sample fit" is computed for context only and is
//! deliberately NOT a pass criterion.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pad_to, r2, relative_l2, rmse, run_r};
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::Array2;
use std::path::Path;

const NOTTEM_CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/bench/datasets/nottem_monthly_temp.csv"
);

/// Month-of-year cyclic period: months 1..=12, with the seam (knots) at 1 and
/// 13, so December (12) wraps continuously back to January (1). Both gam's
/// `cc(..., period_start, period_end)` and mgcv's `knots=c(1,13)` use these.
const PERIOD_START: f64 = 1.0;
const PERIOD_END: f64 = 13.0;
/// Cyclic-basis dimension. With only 12 distinct months, k=8 is comfortably
/// resolved by mgcv's `bs="cc"` and leaves ample smoothing freedom.
const K: usize = 8;

#[test]
fn gam_cyclic_predicts_nottem_seasonal_cycle_vs_mgcv() {
    init_parallelism();

    // ---- load the Nottingham monthly-temperature dataset (month -> temp) ---
    let ds = load_csvwith_inferred_schema(Path::new(NOTTEM_CSV)).expect("load nottem CSV");
    let col = ds.column_map();
    let month_idx = col["month"];
    let temp_idx = col["temp"];
    let month: Vec<f64> = ds.values.column(month_idx).to_vec();
    let temp: Vec<f64> = ds.values.column(temp_idx).to_vec();
    let n = month.len();
    assert_eq!(n, 240, "nottem should have 240 monthly rows, got {n}");

    // ---- deterministic train/test split: every 4th row is held out --------
    let is_test = |i: usize| i % 4 == 0;
    let train_rows: Vec<usize> = (0..n).filter(|&i| !is_test(i)).collect();
    let test_rows: Vec<usize> = (0..n).filter(|&i| is_test(i)).collect();
    assert!(
        train_rows.len() == 180 && test_rows.len() == 60,
        "split sizes: train={} test={}",
        train_rows.len(),
        test_rows.len()
    );

    let train_month: Vec<f64> = train_rows.iter().map(|&i| month[i]).collect();
    let train_temp: Vec<f64> = train_rows.iter().map(|&i| temp[i]).collect();
    let test_month: Vec<f64> = test_rows.iter().map(|&i| month[i]).collect();
    let test_temp: Vec<f64> = test_rows.iter().map(|&i| temp[i]).collect();

    // Build a training-only dataset by sub-setting the encoded rows; headers,
    // schema and column kinds are unchanged, so the formula resolves identically.
    let p = ds.headers.len();
    let mut train_values = Array2::<f64>::zeros((train_rows.len(), p));
    for (out_row, &src_row) in train_rows.iter().enumerate() {
        for c in 0..p {
            train_values[[out_row, c]] = ds.values[[src_row, c]];
        }
    }
    let mut train_ds = ds.clone();
    train_ds.values = train_values;

    // ---- fit gam on TRAIN: temp ~ cc(month), REML over period [1, 13) ------
    // The DSL parses option values with a plain f64 parse (no expression eval),
    // so the period bounds are passed as numeric literals.
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let formula =
        format!("temp ~ cc(month, k={K}, period_start={PERIOD_START}, period_end={PERIOD_END})");
    let result = fit_from_formula(&formula, &train_ds, &cfg).expect("gam cyclic fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for a gaussian cyclic smooth");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // gam predictions at the held-out months: rebuild the design from the frozen
    // spec (identity link => design*beta = predicted mean).
    let mut test_grid = Array2::<f64>::zeros((test_rows.len(), p));
    for (i, &m) in test_month.iter().enumerate() {
        test_grid[[i, month_idx]] = m;
    }
    let test_design = build_term_collection_design(test_grid.view(), &fit.resolvedspec)
        .expect("rebuild design at held-out months");
    let gam_test_pred: Vec<f64> = test_design.design.apply(&fit.fit.beta).to_vec();

    // ---- periodic-wrap check on gam's OWN fit: fit(month=1) == fit(month=13).
    // A genuine cyclic basis has identical design rows one full period apart, so
    // the fitted seasonal curve must close on itself across the Dec->Jan seam.
    let mut seam_grid = Array2::<f64>::zeros((2, p));
    seam_grid[[0, month_idx]] = PERIOD_START; // month = 1   (January edge)
    seam_grid[[1, month_idx]] = PERIOD_END; //   month = 13  (== one period later)
    let seam_design = build_term_collection_design(seam_grid.view(), &fit.resolvedspec)
        .expect("rebuild design at period seam");
    let seam_fit: Vec<f64> = seam_design.design.apply(&fit.fit.beta).to_vec();
    let wrap_gap = (seam_fit[0] - seam_fit[1]).abs();

    // ---- fit the SAME model on TRAIN with mgcv bs="cc", predict SAME TEST ---
    // mgcv is the mature baseline. The harness exposes one data.frame per call,
    // so the held-out months ride along padded into a parallel column and we
    // read both mgcv's in-sample fitted values (context) and its held-out
    // predictions (accuracy) back.
    let train_r = run_r(
        &[
            Column::new("month", &train_month),
            Column::new("temp", &train_temp),
            Column::new("test_month", &pad_to(&test_month, train_month.len())),
            Column::new("test_n", &vec![test_month.len() as f64; train_month.len()]),
        ],
        &format!(
            r#"
            suppressPackageStartupMessages(library(mgcv))
            m <- gam(temp ~ s(month, bs = "cc", k = {K}), data = df, method = "REML",
                     knots = list(month = c({PERIOD_START}, {PERIOD_END})))
            emit("fitted", as.numeric(fitted(m)))
            emit("edf", sum(m$edf))
            k <- df$test_n[1]
            newd <- data.frame(month = df$test_month[1:k])
            emit("test_pred", as.numeric(predict(m, newdata = newd)))
            "#
        ),
    );
    let mgcv_train_fitted = train_r.vector("fitted").to_vec();
    let mgcv_edf = train_r.scalar("edf");
    let mgcv_test_pred = train_r.vector("test_pred");
    assert_eq!(
        mgcv_test_pred.len(),
        test_rows.len(),
        "mgcv held-out prediction length mismatch"
    );

    // ---- objective metrics on gam's OWN predictions -----------------------
    let gam_test_r2 = r2(&gam_test_pred, &test_temp);
    let gam_test_rmse = rmse(&gam_test_pred, &test_temp);
    let mgcv_test_rmse = rmse(mgcv_test_pred, &test_temp);

    // Context-only diagnostic: closeness of gam's in-sample fit vs mgcv's. NOT a
    // pass criterion (matching another tool's fit proves nothing about truth).
    let mut gam_train_grid = Array2::<f64>::zeros((train_rows.len(), p));
    for (i, &m) in train_month.iter().enumerate() {
        gam_train_grid[[i, month_idx]] = m;
    }
    let gam_train_design = build_term_collection_design(gam_train_grid.view(), &fit.resolvedspec)
        .expect("rebuild design at training months");
    let gam_train_fitted: Vec<f64> = gam_train_design.design.apply(&fit.fit.beta).to_vec();
    let insample_rel = relative_l2(&gam_train_fitted, &mgcv_train_fitted);

    eprintln!(
        "nottem cc(month): n_train={} n_test={} gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} \
         gam_test_R2={gam_test_r2:.4} gam_test_rmse={gam_test_rmse:.4} \
         mgcv_test_rmse={mgcv_test_rmse:.4} wrap_gap={wrap_gap:.3e} \
         (context: in-sample rel_l2 vs mgcv={insample_rel:.4})",
        train_rows.len(),
        test_rows.len(),
    );

    // 1) PREDICTION (primary): gam's cyclic smooth explains the held-out cycle.
    assert!(
        gam_test_r2 >= 0.85,
        "gam held-out predictive R2 too low: {gam_test_r2:.4} (< 0.85)"
    );

    // 2) MATCH-OR-BEAT (accuracy): gam no worse than mgcv on held-out RMSE.
    assert!(
        gam_test_rmse <= mgcv_test_rmse * 1.10,
        "gam held-out RMSE {gam_test_rmse:.4} exceeds mgcv {mgcv_test_rmse:.4} * 1.10"
    );

    // 3) STRUCTURE — periodic seam continuity: fit(1) must equal fit(13).
    assert!(
        wrap_gap < 1e-6,
        "cyclic wrap not enforced: |fit(month=1) - fit(month=13)| = {wrap_gap:.3e}"
    );

    // ---- complexity sanity: edf in a seasonal-appropriate range (not matched).
    assert!(
        gam_edf > 1.0 && gam_edf < (K as f64),
        "gam effective dof out of sane range: {gam_edf:.3}"
    );
}
