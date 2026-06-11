//! End-to-end quality: gam's 1-D **p-spline** smooth (`bs='ps'`) is judged by
//! its OBJECTIVE held-out predictive accuracy on real data, not by how closely
//! it reproduces mgcv's fitted curve.
//!
//! mgcv's `bs='ps'` is the canonical Eilers-Marx penalized B-spline: a cubic
//! (degree-3) B-spline basis with a discrete *second-order difference* penalty
//! on adjacent coefficients (mgcv defaults `m=c(2,2)` -> degree 3, penalty
//! order 2). gam's `s(range, bs='ps')` builds exactly this basis
//! (`term_builder`: `"ps"` -> B-spline, `degree=3`, `penalty_order=2`).
//!
//! We use the canonical `lidar` smoothing benchmark (`logratio ~ range`) and a
//! deterministic train/test split (every 5th row, by index, is held out). The
//! p-spline is fit on the training rows ONLY and used to predict the held-out
//! `range` values. The OBJECTIVE metric asserted is the held-out predictive
//! accuracy of gam's OWN predictions:
//!   1. test R^2 >= 0.55 — the smooth must explain the majority of held-out
//!      variance in `logratio` (the lidar signal is strong but noisy; a flat
//!      mean scores 0 and a wiggly overfit scores poorly out-of-sample, so a
//!      solid positive R^2 is a real generalization claim), and
//!   2. gam's held-out RMSE <= mgcv's held-out RMSE * 1.10 — mgcv (fit on the
//!      identical training rows, predicting the identical test rows) is a
//!      BASELINE TO MATCH-OR-BEAT on out-of-sample accuracy, never a target to
//!      reproduce.
//!
//! The primary claim is that gam's p-spline GENERALIZES (recovers the latent
//! signal well enough to predict unseen points), not that it mimics mgcv. We
//! still compute mgcv's in-sample fit and print the train-grid rel_l2 for
//! context via `eprintln!`, but that closeness is never a pass criterion.

use gam::data::EncodedDataset;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, r2, relative_l2, run_r};
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::Array2;
use std::path::Path;

const LIDAR_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/lidar.csv");

/// Root-mean-square error of `pred` against `truth`.
fn rmse_pair(pred: &[f64], truth: &[f64]) -> f64 {
    assert_eq!(pred.len(), truth.len(), "rmse length mismatch");
    let n = pred.len() as f64;
    let s: f64 = pred.iter().zip(truth).map(|(p, y)| (p - y) * (p - y)).sum();
    (s / n.max(1.0)).sqrt()
}

#[test]
fn gam_pspline_generalizes_on_lidar() {
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

    // ---- deterministic train/test split: every 5th row is held out -------
    // Index-based split is fully reproducible (no RNG) and identical for gam
    // and mgcv, so both engines see the EXACT same training rows and are
    // scored on the EXACT same held-out rows.
    let test_mask: Vec<bool> = (0..n).map(|i| i % 5 == 0).collect();
    let train_rows: Vec<usize> = (0..n).filter(|&i| !test_mask[i]).collect();
    let test_rows: Vec<usize> = (0..n).filter(|&i| test_mask[i]).collect();
    let n_train = train_rows.len();
    let n_test = test_rows.len();
    assert!(
        n_train > 100 && n_test > 20,
        "split should leave a healthy train/test partition: train={n_train} test={n_test}"
    );

    let range_train: Vec<f64> = train_rows.iter().map(|&i| range[i]).collect();
    let range_test: Vec<f64> = test_rows.iter().map(|&i| range[i]).collect();
    let logratio_test: Vec<f64> = test_rows.iter().map(|&i| logratio[i]).collect();

    // ---- build a TRAIN-ONLY dataset for gam -------------------------------
    // Clone the schema/headers/kinds and replace the value matrix with just the
    // training rows so gam never sees the held-out points during fitting.
    let mut train_values = Array2::<f64>::zeros((n_train, ds.headers.len()));
    for (new_row, &orig) in train_rows.iter().enumerate() {
        train_values.row_mut(new_row).assign(&ds.values.row(orig));
    }
    let train_ds = EncodedDataset {
        headers: ds.headers.clone(),
        values: train_values,
        schema: ds.schema.clone(),
        column_kinds: ds.column_kinds.clone(),
    };

    // ---- fit with gam on TRAIN ONLY: logratio ~ s(range, bs='ps', k=15) ---
    // bs='ps' -> cubic B-spline basis with 2nd-order difference penalty,
    // exactly matching mgcv's p-spline default; k=15 fixes the basis dim; REML
    // selects the smoothing parameter.
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("logratio ~ s(range, bs='ps', k=15)", &train_ds, &cfg)
        .expect("gam p-spline fit on training rows");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for a gaussian p-spline smooth");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // gam predictions at the held-out `range`: rebuild the design from the
    // frozen training spec at the test points (identity link => design*beta is
    // the predicted mean). These are gam's OWN out-of-sample predictions.
    let mut test_grid = Array2::<f64>::zeros((n_test, ds.headers.len()));
    for (i, &r) in range_test.iter().enumerate() {
        test_grid[[i, range_idx]] = r;
    }
    let test_design = build_term_collection_design(test_grid.view(), &fit.resolvedspec)
        .expect("rebuild design at held-out points");
    let gam_test_pred: Vec<f64> = test_design.design.apply(&fit.fit.beta).to_vec();

    // gam predictions on the training grid (context only, for the rel_l2 print).
    let mut train_grid = Array2::<f64>::zeros((n_train, ds.headers.len()));
    for (i, &r) in range_train.iter().enumerate() {
        train_grid[[i, range_idx]] = r;
    }
    let train_design = build_term_collection_design(train_grid.view(), &fit.resolvedspec)
        .expect("rebuild design at training points");
    let gam_train_fit: Vec<f64> = train_design.design.apply(&fit.fit.beta).to_vec();

    // ---- fit the SAME model with mgcv on the SAME train rows --------------
    // mgcv is the BASELINE: fit on the identical training partition, predict
    // the identical held-out `range`. m=c(2,2) pins mgcv to a cubic B-spline
    // (degree 3) with a 2nd-order difference penalty so the basis cannot drift.
    //
    // The reference harness writes all columns into a single CSV (one row per
    // index), so every column handed to `run_r` must have the SAME length. We
    // therefore pass the FULL dataset (`range`, `logratio`) plus an `is_test`
    // mask column (all length `n`) and let R reconstruct the IDENTICAL split
    // internally: train where is_test==0, predict where is_test==1. This keeps
    // the train/test partition bit-identical to gam's while satisfying the
    // equal-length-columns contract.
    let is_test: Vec<f64> = (0..n)
        .map(|i| if test_mask[i] { 1.0 } else { 0.0 })
        .collect();
    let r = run_r(
        &[
            Column::new("range", &range),
            Column::new("logratio", &logratio),
            Column::new("is_test", &is_test),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        train <- df$is_test == 0
        test  <- df$is_test == 1
        tr <- data.frame(range = df$range[train], logratio = df$logratio[train])
        te <- data.frame(range = df$range[test])
        m <- gam(logratio ~ s(range, bs = "ps", k = 15, m = c(2, 2)), data = tr, method = "REML")
        emit("test_pred", as.numeric(predict(m, newdata = te)))
        emit("train_fit", as.numeric(fitted(m)))
        emit("edf", sum(m$edf))
        "#,
    );
    let mgcv_test_pred = r.vector("test_pred");
    let mgcv_train_fit = r.vector("train_fit");
    let mgcv_edf = r.scalar("edf");
    assert_eq!(
        mgcv_test_pred.len(),
        n_test,
        "mgcv held-out prediction length mismatch"
    );
    assert_eq!(
        mgcv_train_fit.len(),
        n_train,
        "mgcv training-fit length mismatch"
    );

    // ---- objective held-out metrics on gam's OWN predictions --------------
    let gam_test_r2 = r2(&gam_test_pred, &logratio_test);
    let gam_test_rmse = rmse_pair(&gam_test_pred, &logratio_test);
    let mgcv_test_rmse = rmse_pair(mgcv_test_pred, &logratio_test);

    // Context only (NOT a pass criterion): how close the two in-sample fits and
    // out-of-sample predictions land. A real divergence would also show up as a
    // failed objective bar above.
    let train_rel = relative_l2(&gam_train_fit, mgcv_train_fit);
    let test_rel = relative_l2(&gam_test_pred, mgcv_test_pred);
    eprintln!(
        "lidar s(range, bs='ps', k=15) held-out: n_train={n_train} n_test={n_test} \
         gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} \
         gam_test_R2={gam_test_r2:.4} gam_test_rmse={gam_test_rmse:.4} \
         mgcv_test_rmse={mgcv_test_rmse:.4} train_rel_l2={train_rel:.4} test_rel_l2={test_rel:.4}"
    );

    // ---- assertions: OBJECTIVE predictive quality -------------------------
    // (1) gam's p-spline must GENERALIZE: it explains the majority of held-out
    //     variance. 0.55 is a principled floor for the lidar signal — well
    //     above a flat-mean predictor (R^2=0) and far above what an overfit
    //     wiggly curve scores out-of-sample.
    assert!(
        gam_test_r2 >= 0.55,
        "gam p-spline does not generalize on held-out lidar: test R^2 = {gam_test_r2:.4} (need >= 0.55)"
    );
    // (2) MATCH-OR-BEAT the mature baseline on out-of-sample accuracy: gam's
    //     held-out RMSE is no worse than mgcv's by more than 10%.
    assert!(
        gam_test_rmse <= mgcv_test_rmse * 1.10,
        "gam held-out RMSE {gam_test_rmse:.4} exceeds mgcv {mgcv_test_rmse:.4} by > 10%"
    );
    // (3) sane complexity (NOT edf-matching): the selected smooth uses more
    //     than a line but well under the basis dimension k=15.
    assert!(
        gam_edf > 1.0 && gam_edf < 15.0,
        "gam effective complexity out of range: edf = {gam_edf:.3} (expected 1 < edf < 15)"
    );
}

/// Second real-data arm exercising the SAME cubic p-spline (`s(range, bs='ps')`)
/// capability on the canonical `lidar` smoothing benchmark, under a DIFFERENT
/// deterministic holdout and a DIFFERENT objective lens.
///
/// Dataset SOURCE: the classic LIDAR (light-detection-and-ranging) smoothing
/// benchmark of Sigrist (1994), distributed with the SemiPar R package
/// (`data(lidar)`); `range` is the distance traveled before light is reflected
/// back to source, `logratio` is the log ratio of received light from two
/// laser sources. Real measurements: there is no analytic ground-truth curve,
/// so objective quality is purely out-of-sample predictive accuracy.
///
/// Holdout (distinct from the every-5th split above so the two arms are NOT
/// redundant): every 4th row by index is held out. The p-spline is fit on the
/// training rows ONLY and used to predict the held-out `range`. Objective bars
/// asserted on gam's OWN predictions:
///   1. ABSOLUTE held-out accuracy: test RMSE <= 0.08 logratio units. The
///      observed lidar noise floor is ~0.05-0.07; an RMSE under 0.08 means gam
///      tracks the latent curve to within roughly the irreducible noise and is
///      a tool-free correctness claim (a flat-mean predictor scores ~0.18).
///   2. MATCH-OR-BEAT mgcv (the mature canonical p-spline, fit on the identical
///      training rows, predicting the identical held-out rows): gam's held-out
///      RMSE <= mgcv's held-out RMSE * 1.10. mgcv is a baseline to match-or-beat
///      on accuracy, never an output to reproduce.
/// We also assert test R^2 >= 0.55 as a generalization floor; the in-sample
/// rel_l2 vs mgcv is printed for context only and is NOT a pass criterion.
#[test]
fn gam_pspline_generalizes_on_lidar_on_real_data() {
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
    // Index-based, no RNG; identical rows in identical order go to gam and mgcv.
    let test_mask: Vec<bool> = (0..n).map(|i| i % 4 == 0).collect();
    let train_rows: Vec<usize> = (0..n).filter(|&i| !test_mask[i]).collect();
    let test_rows: Vec<usize> = (0..n).filter(|&i| test_mask[i]).collect();
    let n_train = train_rows.len();
    let n_test = test_rows.len();
    assert!(
        n_train > 100 && n_test > 40,
        "split should leave a healthy train/test partition: train={n_train} test={n_test}"
    );

    let range_train: Vec<f64> = train_rows.iter().map(|&i| range[i]).collect();
    let range_test: Vec<f64> = test_rows.iter().map(|&i| range[i]).collect();
    let logratio_test: Vec<f64> = test_rows.iter().map(|&i| logratio[i]).collect();

    // ---- build a TRAIN-ONLY dataset for gam -------------------------------
    let mut train_values = Array2::<f64>::zeros((n_train, ds.headers.len()));
    for (new_row, &orig) in train_rows.iter().enumerate() {
        train_values.row_mut(new_row).assign(&ds.values.row(orig));
    }
    let train_ds = EncodedDataset {
        headers: ds.headers.clone(),
        values: train_values,
        schema: ds.schema.clone(),
        column_kinds: ds.column_kinds.clone(),
    };

    // ---- fit with gam on TRAIN ONLY: logratio ~ s(range, bs='ps', k=15) ---
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("logratio ~ s(range, bs='ps', k=15)", &train_ds, &cfg)
        .expect("gam p-spline fit on training rows");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for a gaussian p-spline smooth");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // gam's OWN out-of-sample predictions at the held-out `range`.
    let mut test_grid = Array2::<f64>::zeros((n_test, ds.headers.len()));
    for (i, &r) in range_test.iter().enumerate() {
        test_grid[[i, range_idx]] = r;
    }
    let test_design = build_term_collection_design(test_grid.view(), &fit.resolvedspec)
        .expect("rebuild design at held-out points");
    let gam_test_pred: Vec<f64> = test_design.design.apply(&fit.fit.beta).to_vec();

    // gam in-sample fit on the training grid (context only).
    let mut train_grid = Array2::<f64>::zeros((n_train, ds.headers.len()));
    for (i, &r) in range_train.iter().enumerate() {
        train_grid[[i, range_idx]] = r;
    }
    let train_design = build_term_collection_design(train_grid.view(), &fit.resolvedspec)
        .expect("rebuild design at training points");
    let gam_train_fit: Vec<f64> = train_design.design.apply(&fit.fit.beta).to_vec();

    // ---- fit the SAME model with mgcv on the SAME train rows --------------
    // Equal-length-columns contract: pass the FULL dataset plus an `is_test`
    // mask and reconstruct the identical split inside R (see the first arm).
    let is_test: Vec<f64> = (0..n)
        .map(|i| if test_mask[i] { 1.0 } else { 0.0 })
        .collect();
    let r = run_r(
        &[
            Column::new("range", &range),
            Column::new("logratio", &logratio),
            Column::new("is_test", &is_test),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        train <- df$is_test == 0
        test  <- df$is_test == 1
        tr <- data.frame(range = df$range[train], logratio = df$logratio[train])
        te <- data.frame(range = df$range[test])
        m <- gam(logratio ~ s(range, bs = "ps", k = 15, m = c(2, 2)), data = tr, method = "REML")
        emit("test_pred", as.numeric(predict(m, newdata = te)))
        emit("train_fit", as.numeric(fitted(m)))
        emit("edf", sum(m$edf))
        "#,
    );
    let mgcv_test_pred = r.vector("test_pred");
    let mgcv_train_fit = r.vector("train_fit");
    let mgcv_edf = r.scalar("edf");
    assert_eq!(
        mgcv_test_pred.len(),
        n_test,
        "mgcv held-out prediction length mismatch"
    );
    assert_eq!(
        mgcv_train_fit.len(),
        n_train,
        "mgcv training-fit length mismatch"
    );

    // ---- objective held-out metrics on gam's OWN predictions --------------
    let gam_test_r2 = r2(&gam_test_pred, &logratio_test);
    let gam_test_rmse = rmse_pair(&gam_test_pred, &logratio_test);
    let mgcv_test_rmse = rmse_pair(mgcv_test_pred, &logratio_test);

    // Context only (NOT a pass criterion).
    let train_rel = relative_l2(&gam_train_fit, mgcv_train_fit);
    let test_rel = relative_l2(&gam_test_pred, mgcv_test_pred);
    eprintln!(
        "lidar(every-4th) s(range, bs='ps', k=15) held-out: n_train={n_train} n_test={n_test} \
         gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} \
         gam_test_R2={gam_test_r2:.4} gam_test_rmse={gam_test_rmse:.4} \
         mgcv_test_rmse={mgcv_test_rmse:.4} train_rel_l2={train_rel:.4} test_rel_l2={test_rel:.4}"
    );

    // ---- assertions: OBJECTIVE predictive quality -------------------------
    // (1) ABSOLUTE held-out accuracy bar (tool-free): RMSE within the lidar
    //     noise floor. A flat-mean predictor scores ~0.18, so <= 0.08 is a
    //     strong, real generalization claim.
    assert!(
        gam_test_rmse <= 0.08,
        "gam held-out RMSE {gam_test_rmse:.4} exceeds the absolute lidar accuracy bar (0.08)"
    );
    // (2) generalization floor on explained held-out variance.
    assert!(
        gam_test_r2 >= 0.55,
        "gam p-spline does not generalize on held-out lidar: test R^2 = {gam_test_r2:.4} (need >= 0.55)"
    );
    // (3) MATCH-OR-BEAT the mature baseline on out-of-sample RMSE.
    assert!(
        gam_test_rmse <= mgcv_test_rmse * 1.10,
        "gam held-out RMSE {gam_test_rmse:.4} exceeds mgcv {mgcv_test_rmse:.4} by > 10%"
    );
    // (4) sane complexity (NOT edf-matching).
    assert!(
        gam_edf > 1.0 && gam_edf < 15.0,
        "gam effective complexity out of range: edf = {gam_edf:.3} (expected 1 < edf < 15)"
    );
}
