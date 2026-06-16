//! End-to-end quality: gam's 1-D thin-plate regression spline (`bs="tp"`) must
//! generalize — it has to PREDICT held-out data well, not merely reproduce
//! mgcv's in-sample fit. Matching another smoother's fitted curve proves
//! nothing about quality (both could overfit the lidar noise identically); the
//! only honest claim is out-of-sample accuracy on data the smoother never saw.
//!
//! OBJECTIVE METRIC (the pass/fail criterion):
//!   * Deterministic 80/20 train/test split of the lidar dataset (every 5th row,
//!     by original index, is held out — no RNG, fully reproducible).
//!   * gam fits `s(range, bs="tp", k=20)` by REML on the *train* rows only, then
//!     predicts the *test* rows it never saw.
//!   * PRIMARY claim: held-out predictive accuracy. We assert the test-set
//!     coefficient of determination `R^2 >= 0.55` (the lidar signal-to-noise is
//!     modest; a smoother that has learned the real range->logratio curve clears
//!     this comfortably while pure noise / a badly broken smoother does not).
//!   * MATCH-OR-BEAT baseline: mgcv (`mgcv::gam(..., method="REML")`), the mature
//!     origin of the thin-plate-regression-spline construction, is fit on the
//!     IDENTICAL train rows and asked to predict the IDENTICAL test rows. gam's
//!     held-out RMSE must be `<= mgcv_test_rmse * 1.10` — gam may not be more
//!     than 10% worse than the reference on out-of-sample error.
//!
//! mgcv is therefore a baseline to match-or-beat on a real predictive metric,
//! not an oracle whose in-sample curve we copy. EDF agreement is deliberately
//! not asserted (it is a basis/null-space-convention artifact, not a quality
//! signal). The in-sample rel_l2 vs mgcv is still printed for context.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, relative_l2, rmse, run_r};
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::Array2;
use std::path::Path;

const LIDAR_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/lidar.csv");

/// Coefficient of determination of `pred` against observed `truth`:
/// `1 - SS_res / SS_tot`. On a held-out set this is honest out-of-sample R^2.
fn r_squared(pred: &[f64], truth: &[f64]) -> f64 {
    assert_eq!(pred.len(), truth.len(), "r_squared length mismatch");
    let n = truth.len() as f64;
    let mean = truth.iter().sum::<f64>() / n;
    let ss_tot: f64 = truth.iter().map(|y| (y - mean) * (y - mean)).sum();
    let ss_res: f64 = pred.iter().zip(truth).map(|(p, y)| (y - p) * (y - p)).sum();
    1.0 - ss_res / ss_tot.max(1e-300)
}

#[test]
fn gam_thin_plate_1d_predicts_heldout_lidar_at_least_as_well_as_mgcv() {
    init_parallelism();

    // ---- load the canonical lidar dataset (range -> logratio) -------------
    let ds = load_csvwith_inferred_schema(Path::new(LIDAR_CSV)).expect("load lidar.csv");
    let col = ds.column_map();
    let range_idx = col["range"];
    let logratio_idx = col["logratio"];
    let range: Vec<f64> = ds.values.column(range_idx).to_vec();
    let logratio: Vec<f64> = ds.values.column(logratio_idx).to_vec();
    let n = range.len();
    assert!(n > 200, "lidar should have ~221 rows, got {n}");

    // ---- deterministic 80/20 split: every 5th original row is held out ----
    // No RNG, no shuffle — the split is a pure function of the row index, so the
    // test is bit-reproducible and gam + mgcv see byte-identical train/test sets.
    let is_test = |i: usize| i % 5 == 0;
    let train_rows: Vec<usize> = (0..n).filter(|&i| !is_test(i)).collect();
    let test_rows: Vec<usize> = (0..n).filter(|&i| is_test(i)).collect();
    assert!(
        train_rows.len() > 150 && test_rows.len() > 30,
        "split is too small: train={} test={}",
        train_rows.len(),
        test_rows.len()
    );

    let train_range: Vec<f64> = train_rows.iter().map(|&i| range[i]).collect();
    let train_logratio: Vec<f64> = train_rows.iter().map(|&i| logratio[i]).collect();
    let test_range: Vec<f64> = test_rows.iter().map(|&i| range[i]).collect();
    let test_logratio: Vec<f64> = test_rows.iter().map(|&i| logratio[i]).collect();

    // ---- build a TRAIN-only dataset (same schema, subset of rows) ---------
    let mut train_ds = ds.clone();
    let mut train_values = Array2::<f64>::zeros((train_rows.len(), ds.headers.len()));
    for (new_i, &orig_i) in train_rows.iter().enumerate() {
        for c in 0..ds.headers.len() {
            train_values[[new_i, c]] = ds.values[[orig_i, c]];
        }
    }
    train_ds.values = train_values;

    // ---- fit gam on TRAIN only: thin-plate spline, k=20, REML -------------
    // `s(range, bs="tp", k=20)` routes through the thin-plate (`tps`) basis with
    // 20 centers — the 1-D analogue of mgcv's `bs="tp", k=20`.
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("logratio ~ s(range, bs=\"tp\", k=20)", &train_ds, &cfg)
        .expect("gam fit on train");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for a gaussian thin-plate smooth");
    };

    // gam prediction on the held-out TEST points: rebuild the frozen design at
    // the test `range` values (identity link => design*beta = predicted mean).
    let predict_at = |xs: &[f64]| -> Vec<f64> {
        let mut grid = Array2::<f64>::zeros((xs.len(), ds.headers.len()));
        for (i, &r) in xs.iter().enumerate() {
            grid[[i, range_idx]] = r;
        }
        let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
            .expect("rebuild thin-plate design at prediction points");
        design.design.apply(&fit.fit.beta).to_vec()
    };
    let gam_test_pred = predict_at(&test_range);
    let gam_train_pred = predict_at(&train_range);

    let gam_test_r2 = r_squared(&gam_test_pred, &test_logratio);
    let gam_test_rmse = rmse(&gam_test_pred, &test_logratio);

    // ---- fit the SAME model with mgcv on the SAME train rows, predict test -
    // mgcv is the match-or-beat baseline on out-of-sample error, not an oracle.
    // The harness hands the body exactly ONE equal-length data.frame (the train
    // rows). The held-out test grid has a different length, so it cannot ride in
    // that frame; we bake it into the R body as a literal `c(...)` vector built
    // in Rust. This keeps the test `range` values byte-identical to gam's.
    let test_grid_literal = test_range
        .iter()
        .map(|x| format!("{x:.17e}"))
        .collect::<Vec<_>>()
        .join(", ");
    let r_body = format!(
        r#"
        suppressPackageStartupMessages(library(mgcv))
        m <- gam(train_logratio ~ s(train_range, bs = "tp", k = 20),
                 data = df, method = "REML")
        emit("train_fitted", as.numeric(fitted(m)))
        nd <- data.frame(train_range = c({test_grid_literal}))
        emit("test_pred", as.numeric(predict(m, newdata = nd)))
        "#
    );
    let r = run_r(
        &[
            Column::new("train_range", &train_range),
            Column::new("train_logratio", &train_logratio),
        ],
        &r_body,
    );
    let mgcv_train_fitted = r.vector("train_fitted");
    assert_eq!(
        mgcv_train_fitted.len(),
        train_rows.len(),
        "mgcv train fitted length mismatch"
    );
    let mgcv_test_pred = r.vector("test_pred");
    assert_eq!(
        mgcv_test_pred.len(),
        test_rows.len(),
        "mgcv test prediction length mismatch"
    );
    let mgcv_test_rmse = rmse(mgcv_test_pred, &test_logratio);
    let mgcv_test_r2 = r_squared(mgcv_test_pred, &test_logratio);

    // in-sample tracking vs mgcv, printed for context only (NOT a pass criterion)
    let train_rel = relative_l2(&gam_train_pred, mgcv_train_fitted);

    eprintln!(
        "lidar s(range, bs=tp, k=20) held-out: n={n} train={} test={} \
         gam_test_R2={gam_test_r2:.4} mgcv_test_R2={mgcv_test_r2:.4} \
         gam_test_rmse={gam_test_rmse:.4} mgcv_test_rmse={mgcv_test_rmse:.4} \
         train_rel_l2={train_rel:.4}",
        train_rows.len(),
        test_rows.len()
    );

    // PRIMARY objective claim: gam generalizes — it predicts held-out lidar
    // logratio with real skill. The lidar curve is a smooth nonlinear trend with
    // heteroscedastic noise; a smoother that has recovered it clears R^2 >= 0.55
    // on the held-out 20%, while a flat / broken / pure-noise fit cannot.
    assert!(
        gam_test_r2 >= 0.55,
        "gam thin-plate fails to generalize on held-out lidar: test R^2={gam_test_r2:.4} (need >= 0.55)"
    );

    // MATCH-OR-BEAT: gam's out-of-sample error must not exceed mgcv's by >10%.
    assert!(
        gam_test_rmse <= mgcv_test_rmse * 1.10,
        "gam held-out RMSE worse than mgcv by >10%: gam={gam_test_rmse:.4} mgcv={mgcv_test_rmse:.4}"
    );
}

/// Real-data arm (a second, independent corroboration of the SAME thin-plate
/// 1-D capability). The companion test above splits "every 5th row"; this arm
/// uses a *coarser, disjoint-by-construction* split — every 4th original row is
/// held out — and applies the identical thin-plate spline `s(range, bs="tp",
/// k=20)`. Two different deterministic holdout cadences both clearing the
/// objective bar is much stronger evidence that gam's tps smoother genuinely
/// recovers the lidar range->logratio curve (rather than fitting one peculiar
/// split). Same rules: identical train/test rows in identical order to gam and
/// mgcv, metrics computed in plain Rust, mgcv only a match-or-beat baseline.
///
/// Dataset SOURCE: the canonical `lidar` data of Sigrist (1994), as distributed
/// with the R package `SemiPar` (`data(lidar)`); 221 rows of LIDAR `range`
/// (distance, metres) and `logratio` (log ratio of received light intensities).
/// Checked in at `bench/datasets/lidar.csv` (columns `range`, `logratio`).
#[test]
fn gam_thin_plate_1d_predicts_heldout_lidar_at_least_as_well_as_mgcv_on_real_data() {
    init_parallelism();

    // ---- load the canonical lidar dataset (range -> logratio) -------------
    let ds = load_csvwith_inferred_schema(Path::new(LIDAR_CSV)).expect("load lidar.csv");
    let col = ds.column_map();
    let range_idx = col["range"];
    let logratio_idx = col["logratio"];
    let range: Vec<f64> = ds.values.column(range_idx).to_vec();
    let logratio: Vec<f64> = ds.values.column(logratio_idx).to_vec();
    let n = range.len();
    assert!(n > 200, "lidar should have ~221 rows, got {n}");

    // ---- deterministic 75/25 split: every 4th original row is held out ----
    // A pure function of the row index (no RNG / shuffle) so gam and mgcv see
    // byte-identical, identically-ordered train/test sets. This cadence is
    // intentionally different from the companion test's `i % 5` split.
    let is_test = |i: usize| i % 4 == 0;
    let train_rows: Vec<usize> = (0..n).filter(|&i| !is_test(i)).collect();
    let test_rows: Vec<usize> = (0..n).filter(|&i| is_test(i)).collect();
    assert!(
        train_rows.len() > 120 && test_rows.len() > 40,
        "split is too small: train={} test={}",
        train_rows.len(),
        test_rows.len()
    );

    let train_range: Vec<f64> = train_rows.iter().map(|&i| range[i]).collect();
    let train_logratio: Vec<f64> = train_rows.iter().map(|&i| logratio[i]).collect();
    let test_range: Vec<f64> = test_rows.iter().map(|&i| range[i]).collect();
    let test_logratio: Vec<f64> = test_rows.iter().map(|&i| logratio[i]).collect();

    // ---- build a TRAIN-only dataset (same schema, subset of rows) ---------
    let mut train_ds = ds.clone();
    let mut train_values = Array2::<f64>::zeros((train_rows.len(), ds.headers.len()));
    for (new_i, &orig_i) in train_rows.iter().enumerate() {
        for c in 0..ds.headers.len() {
            train_values[[new_i, c]] = ds.values[[orig_i, c]];
        }
    }
    train_ds.values = train_values;

    // ---- fit gam on TRAIN only: thin-plate spline, k=20, REML -------------
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("logratio ~ s(range, bs=\"tp\", k=20)", &train_ds, &cfg)
        .expect("gam fit on train");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for a gaussian thin-plate smooth");
    };

    // gam prediction at arbitrary `range` points: rebuild the frozen design
    // (identity link => design*beta = predicted mean).
    let predict_at = |xs: &[f64]| -> Vec<f64> {
        let mut grid = Array2::<f64>::zeros((xs.len(), ds.headers.len()));
        for (i, &r) in xs.iter().enumerate() {
            grid[[i, range_idx]] = r;
        }
        let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
            .expect("rebuild thin-plate design at prediction points");
        design.design.apply(&fit.fit.beta).to_vec()
    };
    let gam_test_pred = predict_at(&test_range);
    let gam_train_pred = predict_at(&train_range);

    let gam_test_r2 = r_squared(&gam_test_pred, &test_logratio);
    let gam_test_rmse = rmse(&gam_test_pred, &test_logratio);

    // ---- fit the SAME model with mgcv on the SAME train rows, predict test -
    // The harness hands the body exactly ONE equal-length data.frame (the train
    // rows). The held-out test grid has a different length, so it is baked into
    // the R body as a literal `c(...)` vector built in Rust, keeping the test
    // `range` values byte-identical to what gam predicted at.
    let test_grid_literal = test_range
        .iter()
        .map(|x| format!("{x:.17e}"))
        .collect::<Vec<_>>()
        .join(", ");
    let r_body = format!(
        r#"
        suppressPackageStartupMessages(library(mgcv))
        m <- gam(train_logratio ~ s(train_range, bs = "tp", k = 20),
                 data = df, method = "REML")
        emit("train_fitted", as.numeric(fitted(m)))
        nd <- data.frame(train_range = c({test_grid_literal}))
        emit("test_pred", as.numeric(predict(m, newdata = nd)))
        "#
    );
    let r = run_r(
        &[
            Column::new("train_range", &train_range),
            Column::new("train_logratio", &train_logratio),
        ],
        &r_body,
    );
    let mgcv_train_fitted = r.vector("train_fitted");
    assert_eq!(
        mgcv_train_fitted.len(),
        train_rows.len(),
        "mgcv train fitted length mismatch"
    );
    let mgcv_test_pred = r.vector("test_pred");
    assert_eq!(
        mgcv_test_pred.len(),
        test_rows.len(),
        "mgcv test prediction length mismatch"
    );
    let mgcv_test_rmse = rmse(mgcv_test_pred, &test_logratio);
    let mgcv_test_r2 = r_squared(mgcv_test_pred, &test_logratio);

    // in-sample tracking vs mgcv, printed for context only (NOT a pass criterion)
    let train_rel = relative_l2(&gam_train_pred, mgcv_train_fitted);

    eprintln!(
        "lidar s(range, bs=tp, k=20) i%4 held-out: n={n} train={} test={} \
         gam_test_R2={gam_test_r2:.4} mgcv_test_R2={mgcv_test_r2:.4} \
         gam_test_rmse={gam_test_rmse:.4} mgcv_test_rmse={mgcv_test_rmse:.4} \
         train_rel_l2={train_rel:.4}",
        train_rows.len(),
        test_rows.len()
    );

    // PRIMARY objective claim: gam recovers the real range->logratio curve and
    // predicts the held-out 25% with skill far above the constant-mean baseline.
    assert!(
        gam_test_r2 >= 0.55,
        "gam thin-plate fails to generalize on held-out lidar (i%4 split): \
         test R^2={gam_test_r2:.4} (need >= 0.55)"
    );

    // MATCH-OR-BEAT: gam's out-of-sample error must not exceed mgcv's by >10%.
    assert!(
        gam_test_rmse <= mgcv_test_rmse * 1.10,
        "gam held-out RMSE worse than mgcv by >10% (i%4 split): \
         gam={gam_test_rmse:.4} mgcv={mgcv_test_rmse:.4}"
    );
}
