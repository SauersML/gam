//! End-to-end quality: gam's penalized Gaussian smooth must PREDICT well on
//! held-out data — not merely reproduce mgcv's in-sample fit.
//!
//! The lidar benchmark (`logratio ~ s(range)`) is real data with no known
//! ground-truth function, so the objective quality of a smoother is its
//! out-of-sample predictive accuracy. We make a deterministic train/test split
//! (every 4th row held out), fit `s(range)` by REML on the training rows only,
//! predict the held-out rows, and assert OBJECTIVE metrics on gam's own
//! predictions:
//!
//!   PRIMARY (objective, tool-free): held-out coefficient of determination
//!     `test_R2 >= 0.55` — gam's smooth genuinely explains held-out variance,
//!     well above the constant-mean predictor (R2 = 0).
//!
//!   BASELINE (match-or-beat): mgcv (the mature, standard GAM implementation)
//!     fits the SAME training rows and predicts the SAME held-out rows; gam's
//!     held-out RMSE must be no worse than `mgcv_test_rmse * 1.10`. mgcv is a
//!     baseline to match-or-beat on accuracy, NOT a fitted target to reproduce.
//!
//! "rel_l2 to mgcv's in-sample fit" is still computed and printed for context,
//! but is deliberately NOT a pass criterion: matching another tool's fitted
//! output proves nothing about correctness — both could overfit alike.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, QualityPair, pad_to, r2, relative_l2, rmse, run_r};
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::Array2;
use std::path::Path;

const LIDAR_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/lidar.csv");

#[test]
fn gam_smooth_predicts_lidar_better_than_baseline() {
    init_parallelism();

    // ---- load the canonical lidar dataset (range -> logratio) -------------
    let ds = load_csvwith_inferred_schema(Path::new(LIDAR_CSV)).expect("load lidar.csv");
    let col = ds.column_map();
    let range_idx = col["range"];
    let logratio_idx = col["logratio"];
    let range: Vec<f64> = ds.values.column(range_idx).to_vec();
    let logratio: Vec<f64> = ds.values.column(logratio_idx).to_vec();
    let n = range.len();
    assert!(n > 100, "lidar should have ~221 rows, got {n}");

    // ---- deterministic train/test split: every 4th row is held out -------
    let is_test = |i: usize| i % 4 == 0;
    let train_rows: Vec<usize> = (0..n).filter(|&i| !is_test(i)).collect();
    let test_rows: Vec<usize> = (0..n).filter(|&i| is_test(i)).collect();
    assert!(
        train_rows.len() > 100 && test_rows.len() > 30,
        "split sizes: train={} test={}",
        train_rows.len(),
        test_rows.len()
    );

    let train_range: Vec<f64> = train_rows.iter().map(|&i| range[i]).collect();
    let train_logratio: Vec<f64> = train_rows.iter().map(|&i| logratio[i]).collect();
    let test_range: Vec<f64> = test_rows.iter().map(|&i| range[i]).collect();
    let test_logratio: Vec<f64> = test_rows.iter().map(|&i| logratio[i]).collect();

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

    // ---- fit gam on TRAIN: logratio ~ s(range), REML ----------------------
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("logratio ~ s(range)", &train_ds, &cfg).expect("gam fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // gam predictions at the held-out `range` points: rebuild the design from
    // the frozen spec (identity link => design*beta = predicted mean).
    let mut test_grid = Array2::<f64>::zeros((test_rows.len(), p));
    for (i, &r) in test_range.iter().enumerate() {
        test_grid[[i, range_idx]] = r;
    }
    let test_design = build_term_collection_design(test_grid.view(), &fit.resolvedspec)
        .expect("rebuild design at held-out points");
    let gam_test_pred: Vec<f64> = test_design.design.apply(&fit.fit.beta).to_vec();

    // ---- fit the SAME model on TRAIN with mgcv, predict the SAME TEST -----
    // mgcv is the mature baseline; we read back its in-sample fitted values
    // (for context only), edf, and held-out predictions in a SINGLE R subprocess
    // so we pay only one mgcv fit cost.  Test ranges ride along as a padded
    // parallel column; only the first `test_n` entries are read back in R.
    let r = run_r(
        &[
            Column::new("range", &train_range),
            Column::new("logratio", &train_logratio),
            Column::new("test_range", &pad_to(&test_range, train_range.len())),
            Column::new("test_n", &vec![test_range.len() as f64; train_range.len()]),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        m <- gam(logratio ~ s(range), data = df, method = "REML")
        emit("fitted", as.numeric(fitted(m)))
        emit("edf", sum(m$edf))
        k <- df$test_n[1]
        newd <- data.frame(range = df$test_range[1:k])
        emit("test_pred", as.numeric(predict(m, newdata = newd)))
        "#,
    );
    let mgcv_train_fitted = r.vector("fitted").to_vec();
    let mgcv_edf = r.scalar("edf");
    let mgcv_test_pred = r.vector("test_pred");
    assert_eq!(
        mgcv_test_pred.len(),
        test_rows.len(),
        "mgcv held-out prediction length mismatch"
    );

    // ---- objective metrics on gam's OWN predictions -----------------------
    let gam_test_r2 = r2(&gam_test_pred, &test_logratio);
    let gam_test_rmse = rmse(&gam_test_pred, &test_logratio);
    let mgcv_test_rmse = rmse(mgcv_test_pred, &test_logratio);

    // Context-only diagnostic: closeness of gam's in-sample fit vs mgcv's. NOT
    // a pass criterion.
    let mut gam_train_grid = Array2::<f64>::zeros((train_rows.len(), p));
    for (i, &r) in train_range.iter().enumerate() {
        gam_train_grid[[i, range_idx]] = r;
    }
    let gam_train_design = build_term_collection_design(gam_train_grid.view(), &fit.resolvedspec)
        .expect("rebuild design at training points");
    let gam_train_fitted: Vec<f64> = gam_train_design.design.apply(&fit.fit.beta).to_vec();
    let insample_rel = relative_l2(&gam_train_fitted, &mgcv_train_fitted);

    eprintln!(
        "lidar s(range) held-out: n_train={} n_test={} gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} \
         gam_test_R2={gam_test_r2:.4} gam_test_rmse={gam_test_rmse:.4} \
         mgcv_test_rmse={mgcv_test_rmse:.4} (context: in-sample rel_l2 vs mgcv={insample_rel:.4})",
        train_rows.len(),
        test_rows.len(),
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "families",
            "quality_vs_mgcv_gaussian_smooth::default_basis",
            "test_rmse",
            gam_test_rmse,
            "mgcv",
            mgcv_test_rmse,
        )
        .line()
    );

    // ---- PRIMARY objective assertion: gam predicts the held-out signal -----
    // The lidar smooth is strongly nonlinear with a clear signal; a competent
    // smoother explains well over half the held-out variance. R2 >= 0.55 is far
    // above the constant-mean baseline (0) and would catch under/over-smoothing.
    assert!(
        gam_test_r2 >= 0.55,
        "gam's held-out predictive R2 too low: {gam_test_r2:.4} (< 0.55)"
    );

    // ---- BASELINE (match-or-beat): no worse than mgcv on held-out RMSE -----
    assert!(
        gam_test_rmse <= mgcv_test_rmse * 1.10,
        "gam held-out RMSE {gam_test_rmse:.4} exceeds mgcv {mgcv_test_rmse:.4} * 1.10"
    );

    // ---- complexity sanity: edf in a signal-appropriate range (not matched) -
    assert!(
        gam_edf > 1.0 && gam_edf < 30.0,
        "gam effective dof out of sane range: {gam_edf:.3}"
    );
}

/// Real-data arm for the SAME Gaussian P-spline smooth capability, kept beside
/// the canonical default-`s(range)` test above so neither weakens the other.
///
/// Dataset SOURCE: `bench/datasets/lidar.csv` — the classic LIDAR scatterplot
/// (range in metres vs. log-ratio of received light from two laser sources),
/// distributed with R's `SemiPar` package (Ruppert, Wand & Carroll,
/// *Semiparametric Regression*, 2003). Real measurements, so the true mean
/// function is unknown; objective quality is held-out predictive accuracy.
///
/// This arm fits gam's explicit P-spline basis `s(range, bs="ps")` (degree-3
/// B-spline with a 2nd-order difference penalty — gam's default penalized
/// smooth construction) by REML, and judges it OUT-OF-SAMPLE:
///
///   PRIMARY (objective, tool-free): held-out `test_R2 >= 0.55` on a DISJOINT
///     split from the test above (every 5th row held out), well above the
///     constant-mean predictor's R2 = 0.
///
///   BASELINE (match-or-beat): mgcv fits the IDENTICAL training rows with the
///     SAME `s(range, bs="ps")` basis and predicts the IDENTICAL held-out rows;
///     gam's held-out RMSE must be no worse than `mgcv_test_rmse * 1.10`.
#[test]
fn gam_smooth_predicts_lidar_better_than_baseline_on_real_data() {
    init_parallelism();

    // ---- load the canonical lidar dataset (range -> logratio) -------------
    let ds = load_csvwith_inferred_schema(Path::new(LIDAR_CSV)).expect("load lidar.csv");
    let col = ds.column_map();
    let range_idx = col["range"];
    let logratio_idx = col["logratio"];
    let range: Vec<f64> = ds.values.column(range_idx).to_vec();
    let logratio: Vec<f64> = ds.values.column(logratio_idx).to_vec();
    let n = range.len();
    assert!(n > 100, "lidar should have ~221 rows, got {n}");

    // ---- deterministic train/test split: every 5th row held out (disjoint
    //      pattern from the default-`s(range)` test, which holds out every 4th)
    let is_test = |i: usize| i % 5 == 0;
    let train_rows: Vec<usize> = (0..n).filter(|&i| !is_test(i)).collect();
    let test_rows: Vec<usize> = (0..n).filter(|&i| is_test(i)).collect();
    assert!(
        train_rows.len() > 100 && test_rows.len() > 30,
        "split sizes: train={} test={}",
        train_rows.len(),
        test_rows.len()
    );

    let train_range: Vec<f64> = train_rows.iter().map(|&i| range[i]).collect();
    let train_logratio: Vec<f64> = train_rows.iter().map(|&i| logratio[i]).collect();
    let test_range: Vec<f64> = test_rows.iter().map(|&i| range[i]).collect();
    let test_logratio: Vec<f64> = test_rows.iter().map(|&i| logratio[i]).collect();

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

    // ---- fit gam on TRAIN: logratio ~ s(range, bs="ps"), REML -------------
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("logratio ~ s(range, bs=\"ps\")", &train_ds, &cfg).expect("gam fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // gam predictions at the held-out `range` points (identity link).
    let mut test_grid = Array2::<f64>::zeros((test_rows.len(), p));
    for (i, &r) in test_range.iter().enumerate() {
        test_grid[[i, range_idx]] = r;
    }
    let test_design = build_term_collection_design(test_grid.view(), &fit.resolvedspec)
        .expect("rebuild design at held-out points");
    let gam_test_pred: Vec<f64> = test_design.design.apply(&fit.fit.beta).to_vec();

    // ---- mgcv baseline: SAME train rows, SAME basis, SAME held-out rows ----
    let r = run_r(
        &[
            Column::new("range", &train_range),
            Column::new("logratio", &train_logratio),
            Column::new("test_range", &pad_to(&test_range, train_range.len())),
            Column::new("test_n", &vec![test_range.len() as f64; train_range.len()]),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        m <- gam(logratio ~ s(range, bs = "ps"), data = df, method = "REML")
        emit("edf", sum(m$edf))
        k <- df$test_n[1]
        newd <- data.frame(range = df$test_range[1:k])
        emit("test_pred", as.numeric(predict(m, newdata = newd)))
        "#,
    );
    let mgcv_edf = r.scalar("edf");
    let mgcv_test_pred = r.vector("test_pred");
    assert_eq!(
        mgcv_test_pred.len(),
        test_rows.len(),
        "mgcv held-out prediction length mismatch"
    );

    // ---- objective metrics on gam's OWN held-out predictions --------------
    let gam_test_r2 = r2(&gam_test_pred, &test_logratio);
    let gam_test_rmse = rmse(&gam_test_pred, &test_logratio);
    let mgcv_test_rmse = rmse(mgcv_test_pred, &test_logratio);

    eprintln!(
        "lidar s(range,bs=ps) held-out: n_train={} n_test={} gam_edf={gam_edf:.3} \
         mgcv_edf={mgcv_edf:.3} gam_test_R2={gam_test_r2:.4} gam_test_rmse={gam_test_rmse:.4} \
         mgcv_test_rmse={mgcv_test_rmse:.4}",
        train_rows.len(),
        test_rows.len(),
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "families",
            "quality_vs_mgcv_gaussian_smooth::ps_basis",
            "test_rmse",
            gam_test_rmse,
            "mgcv",
            mgcv_test_rmse,
        )
        .line()
    );

    // ---- PRIMARY objective assertion: gam predicts the held-out signal -----
    assert!(
        gam_test_r2 >= 0.55,
        "gam's held-out predictive R2 too low: {gam_test_r2:.4} (< 0.55)"
    );

    // ---- BASELINE (match-or-beat): no worse than mgcv on held-out RMSE -----
    assert!(
        gam_test_rmse <= mgcv_test_rmse * 1.10,
        "gam held-out RMSE {gam_test_rmse:.4} exceeds mgcv {mgcv_test_rmse:.4} * 1.10"
    );

    // ---- complexity sanity: edf in a signal-appropriate range (not matched) -
    assert!(
        gam_edf > 1.0 && gam_edf < 30.0,
        "gam effective dof out of sane range: {gam_edf:.3}"
    );
}
