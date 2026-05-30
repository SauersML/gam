//! End-to-end quality: gam's REML/Laplace latent-Gaussian smoothing must
//! GENERALIZE — it has to predict held-out observations well, not merely
//! reproduce another engine's in-sample fit.
//!
//! Objective metric asserted (no known truth: `lidar` is real data, so this is
//! the predictive-accuracy case). We make a deterministic train/test split of
//! the canonical `lidar` benchmark — every 5th row (by ascending `range`) is
//! held out as the test set, the remainder is training — fit gam's smooth
//! `logratio ~ s(range, bs='tp')` on the training rows only, predict at the
//! held-out `range` values, and assert:
//!
//!   1. ABSOLUTE held-out accuracy: test R^2 >= 0.55. The lidar signal is a
//!      smooth descending curve with a heteroscedastic noisy tail; a smoother
//!      that has genuinely learned the latent function explains the majority of
//!      held-out variance. This is gam's own out-of-sample skill, measured on
//!      gam's own predictions — it is true whether or not any reference exists.
//!
//!   2. MATCH-OR-BEAT a mature Bayesian smoother on held-out RMSE: gam's test
//!      RMSE must be <= 1.10 * (R-INLA's test RMSE). R-INLA — the reference
//!      scalable nested-Laplace engine for latent-Gaussian models — fits the
//!      SAME structural model (an `rw2` smooth = the discrete 2nd-derivative
//!      penalty, the thin-plate analogue) on the IDENTICAL training rows and
//!      predicts the same held-out points (test rows entered with `y = NA`, the
//!      standard INLA prediction route). INLA is here purely as a competitive
//!      BASELINE on an objective accuracy metric, NOT as a thing to imitate:
//!      we never require gam to reproduce INLA's fitted values, only to predict
//!      held-out data at least as accurately.
//!
//! Why this is a quality claim and the old one was not: the previous test
//! asserted relative_l2(gam_mean, inla_mean) < 0.05 and matching posterior SDs
//! — i.e. "gam does the same thing as INLA". Two engines can agree closely and
//! both be over- or under-smoothed; agreement proves nothing about correctness.
//! Held-out predictive accuracy on a fixed split is an objective, falsifiable
//! quality bar that a wrong smoother fails.
//!
//! NOTE on licensing: R-INLA is distributed under a non-commercial license
//! (academic/research use). This is a scientific benchmarking test that fits a
//! published smoothing dataset to validate gam's approximate-Bayesian
//! inference; it is not a production or commercial comparison.
//!
//! Data: the canonical `lidar` smoothing benchmark (`logratio ~ s(range)`,
//! n = 221), fed identically to both engines.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, relative_l2, rmse, run_r};
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::Array2;
use std::path::Path;

const LIDAR_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/lidar.csv");

/// Deterministic split stride: every `TEST_STRIDE`-th row (in ascending
/// `range` order) is a held-out test point; the rest train. Fixed, seedless,
/// reproducible.
const TEST_STRIDE: usize = 5;

#[test]
fn gam_smooth_predicts_heldout_lidar_at_least_as_well_as_inla() {
    init_parallelism();

    // ---- load the canonical lidar dataset (range -> logratio) -------------
    let ds = load_csvwith_inferred_schema(Path::new(LIDAR_CSV)).expect("load lidar.csv");
    let col = ds.column_map();
    let range_idx = col["range"];
    let logratio_idx = col["logratio"];
    let n = ds.values.nrows();
    assert!(n > 100, "lidar should have ~221 rows, got {n}");

    // Sort row order by ascending range so the deterministic stride split is
    // well spread across the covariate domain (lidar ships pre-sorted, but we
    // do not rely on that).
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        ds.values[[a, range_idx]]
            .partial_cmp(&ds.values[[b, range_idx]])
            .expect("range values are finite")
    });

    // Deterministic train/test partition over the sorted order.
    let mut train_rows: Vec<usize> = Vec::new();
    let mut test_rows: Vec<usize> = Vec::new();
    for (rank, &row) in order.iter().enumerate() {
        if rank % TEST_STRIDE == TEST_STRIDE - 1 {
            test_rows.push(row);
        } else {
            train_rows.push(row);
        }
    }
    let n_train = train_rows.len();
    let n_test = test_rows.len();
    assert!(n_test > 20, "need a meaningful held-out set, got {n_test}");

    let test_range: Vec<f64> = test_rows.iter().map(|&i| ds.values[[i, range_idx]]).collect();
    let test_y: Vec<f64> = test_rows.iter().map(|&i| ds.values[[i, logratio_idx]]).collect();
    let train_y: Vec<f64> =
        train_rows.iter().map(|&i| ds.values[[i, logratio_idx]]).collect();

    // ---- fit gam on the TRAINING rows only --------------------------------
    let mut train_ds = ds.clone();
    let mut train_vals = Array2::<f64>::zeros((n_train, ds.headers.len()));
    for (new_i, &row) in train_rows.iter().enumerate() {
        for c in 0..ds.headers.len() {
            train_vals[[new_i, c]] = ds.values[[row, c]];
        }
    }
    train_ds.values = train_vals;

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("logratio ~ s(range, bs='tp')", &train_ds, &cfg).expect("gam fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit");
    };

    // gam held-out predictions: rebuild the frozen design at the TEST `range`
    // values (identity link => design*beta = predicted mean).
    let mut grid = Array2::<f64>::zeros((n_test, ds.headers.len()));
    for (i, &r) in test_range.iter().enumerate() {
        grid[[i, range_idx]] = r;
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild design at held-out points");
    let gam_pred: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    assert_eq!(gam_pred.len(), n_test, "gam prediction length");

    // ---- INLA baseline: same model, same split, predict the test points ---
    // Full-length data frame in ascending-range order with a per-row `idx` =
    // 1..N driving the rw2 latent field. Training rows carry observed y; held-
    // out rows carry NA, so INLA's posterior at those indices is a genuine
    // out-of-sample prediction (no test y leaks into the fit). We emit the
    // fitted means at the held-out indices, aligned to gam's test order.
    let order_range: Vec<f64> = order.iter().map(|&i| ds.values[[i, range_idx]]).collect();
    let order_logratio: Vec<f64> = order
        .iter()
        .enumerate()
        .map(|(rank, &i)| {
            if rank % TEST_STRIDE == TEST_STRIDE - 1 {
                f64::NAN // held out from the INLA fit
            } else {
                ds.values[[i, logratio_idx]]
            }
        })
        .collect();
    // 1-based held-out positions within the sorted order, for INLA to report.
    let test_positions: Vec<f64> = (0..order.len())
        .filter(|rank| rank % TEST_STRIDE == TEST_STRIDE - 1)
        .map(|rank| (rank + 1) as f64)
        .collect();
    assert_eq!(test_positions.len(), n_test, "INLA test-position count");

    let r = run_r(
        &[
            Column::new("srange", &order_range),
            Column::new("slogratio", &order_logratio),
            Column::new("testpos", &test_positions),
        ],
        r#"
        suppressPackageStartupMessages(library(INLA))
        # df columns: srange (ascending), slogratio (y with NA at held-out
        # rows), testpos (1-based held-out positions, padded with NA to the
        # data-frame length by read.csv). Recover the held-out positions.
        testpos <- df$testpos[!is.na(df$testpos)]
        N <- nrow(df)
        dat <- data.frame(y = df$slogratio, idx = seq_len(N))
        m <- inla(
            y ~ f(idx, model = "rw2", scale.model = TRUE, constr = TRUE),
            family = "gaussian",
            data = dat,
            control.predictor = list(compute = TRUE),
            control.compute = list(config = FALSE)
        )
        # Posterior mean of the linear predictor at the held-out indices =
        # INLA's out-of-sample prediction (those y were NA during the fit).
        fv <- m$summary.fitted.values$mean
        emit("pred", fv[testpos])
        "#,
    );
    let inla_pred = r.vector("pred");
    assert_eq!(inla_pred.len(), n_test, "INLA prediction length mismatch");

    // ---- objective metrics on gam's OWN held-out predictions --------------
    let y_bar = test_y.iter().sum::<f64>() / (n_test as f64);
    let ss_tot: f64 = test_y.iter().map(|y| (y - y_bar) * (y - y_bar)).sum();
    let ss_res: f64 = test_y
        .iter()
        .zip(gam_pred.iter())
        .map(|(y, p)| (y - p) * (y - p))
        .sum();
    let gam_test_r2 = 1.0 - ss_res / ss_tot.max(1e-300);
    let gam_test_rmse = rmse(&gam_pred, &test_y);
    let inla_test_rmse = rmse(inla_pred, &test_y);

    // Context only (NOT a pass criterion): how close the two engines' held-out
    // predictions happen to be. Printed, never asserted.
    let pred_rel_l2 = relative_l2(&gam_pred, inla_pred);

    eprintln!(
        "lidar held-out s(range): n_train={n_train} n_test={n_test} \
         gam_test_R2={gam_test_r2:.4} gam_test_RMSE={gam_test_rmse:.4} \
         inla_test_RMSE={inla_test_rmse:.4} rmse_ratio={:.4} \
         pred_rel_l2(gam,inla)={pred_rel_l2:.4} train_mean_y={:.4}",
        gam_test_rmse / inla_test_rmse.max(1e-300),
        train_y.iter().sum::<f64>() / (n_train as f64),
    );

    // (1) ABSOLUTE quality: gam explains the majority of held-out variance.
    assert!(
        gam_test_r2 >= 0.55,
        "gam held-out R^2 too low: {gam_test_r2:.4} (bound 0.55) — the smooth \
         is not generalizing to unseen lidar points"
    );

    // (2) MATCH-OR-BEAT INLA on held-out RMSE (baseline, not a target to copy).
    assert!(
        gam_test_rmse <= 1.10 * inla_test_rmse,
        "gam held-out RMSE {gam_test_rmse:.4} exceeds 1.10 * INLA RMSE \
         {inla_test_rmse:.4} (= {:.4}) — gam predicts unseen lidar points worse \
         than the mature Bayesian smoother",
        1.10 * inla_test_rmse
    );
}
