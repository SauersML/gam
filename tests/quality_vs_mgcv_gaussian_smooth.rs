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
use gam::test_support::reference::{Column, relative_l2, rmse, run_r};
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::Array2;
use std::path::Path;

const LIDAR_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/lidar.csv");

/// Coefficient of determination of `pred` against observed `truth`, relative to
/// the mean predictor: `1 - SS_res / SS_tot`. R2 = 1 is perfect, R2 = 0 matches
/// predicting the held-out mean, R2 < 0 is worse than the mean.
fn r2(pred: &[f64], truth: &[f64]) -> f64 {
    assert_eq!(pred.len(), truth.len(), "r2 length mismatch");
    let n = truth.len() as f64;
    let mean = truth.iter().sum::<f64>() / n;
    let ss_res: f64 = pred.iter().zip(truth).map(|(p, t)| (t - p) * (t - p)).sum();
    let ss_tot: f64 = truth.iter().map(|t| (t - mean) * (t - mean)).sum();
    1.0 - ss_res / ss_tot.max(1e-300)
}

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
    // mgcv is the mature baseline; we read back its held-out predictions (to
    // compare accuracy) and its in-sample fitted values (for context only).
    let train_r = run_r(
        &[
            Column::new("range", &train_range),
            Column::new("logratio", &train_logratio),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        m <- gam(logratio ~ s(range), data = df, method = "REML")
        emit("fitted", as.numeric(fitted(m)))
        emit("edf", sum(m$edf))
        "#,
    );
    let mgcv_train_fitted = train_r.vector("fitted").to_vec();
    let mgcv_edf = train_r.scalar("edf");

    // Re-fit on train and predict the held-out rows. The harness exposes one
    // data.frame per call, so we pass train range/logratio plus the test range
    // padded into a parallel column and predict on it.
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
        k <- df$test_n[1]
        newd <- data.frame(range = df$test_range[1:k])
        emit("test_pred", as.numeric(predict(m, newdata = newd)))
        "#,
    );
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

/// Right-pad `v` with its last value (or 0.0 when empty) to length `len`, so it
/// can ride along as a column of the reference data.frame. Only the first
/// `v.len()` entries are read back inside the R body.
fn pad_to(v: &[f64], len: usize) -> Vec<f64> {
    assert!(v.len() <= len, "pad target {len} shorter than source {}", v.len());
    let fill = v.last().copied().unwrap_or(0.0);
    let mut out = v.to_vec();
    out.resize(len, fill);
    out
}
