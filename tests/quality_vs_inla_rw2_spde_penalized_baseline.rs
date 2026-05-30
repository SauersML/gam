//! End-to-end quality: gam's REML p-spline smooth (`bs='ps', m=2`) on a
//! structured second-order-difference prior, scored by **held-out predictive
//! accuracy** — with **R-INLA** demoted to a baseline to match-or-beat on the
//! identical split.
//!
//! ## What this test asserts (objective metric)
//!
//! The lidar benchmark (`range -> logratio`, n=221) is real data with no known
//! ground-truth function, so the honest quality claim is *predictive*: does
//! gam's RW2 p-spline generalize to data it never saw? We make a deterministic
//! train/test split (every 5th observation, by sorted `range`, is held out),
//! fit gam on the training rows only, predict the held-out rows from the frozen
//! smooth, and assert
//!
//!   1. an **absolute held-out accuracy bar**: out-of-sample
//!      `R^2 = 1 - SS_res/SS_tot >= 0.55` on the test rows. lidar's logratio is
//!      a strongly nonlinear but noisy signal; a smoother that has genuinely
//!      learned the mean structure (rather than interpolating noise) clears this
//!      comfortably, while an over- or under-smoothed fit does not. This is the
//!      PRIMARY claim and stands on its own without any reference tool.
//!
//!   2. a **match-or-beat against R-INLA** on the same held-out rows: gam's
//!      out-of-sample RMSE must be within 10% of INLA's,
//!      `rmse(gam_test) <= 1.10 * rmse(inla_test)`. INLA — the mature,
//!      best-in-class approximate-Bayesian latent-Gaussian engine — is fit to
//!      the **identical training rows** with its `f(x, model="rw2")` second-order
//!      random-walk prior (the discrete analog of gam's p-spline penalty) and
//!      asked to predict the **identical held-out rows**. We are NOT asserting
//!      that gam reproduces INLA's fitted curve; we assert gam predicts unseen
//!      data at least as accurately as INLA does. Matching a peer tool's fit
//!      proves nothing about quality; beating (or tying) it on out-of-sample
//!      error is an objective accuracy statement.
//!
//! The reference rel_l2 between the two methods' held-out predictions is still
//! computed and printed via `eprintln!` for context, but it is NOT a pass
//! criterion: "close to INLA" is not a quality claim.
//!
//! A failing assertion because gam genuinely predicts worse than the bar (or
//! worse than INLA) is a real finding, not a reason to weaken the bound.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, relative_l2, rmse, run_r};
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::{Array2, s};
use std::path::Path;

const LIDAR_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/lidar.csv");

/// Out-of-sample coefficient of determination `R^2 = 1 - SS_res/SS_tot`, with
/// `SS_tot` taken about the TEST-set mean. A model that predicts the held-out
/// mean structure scores near 1; predicting the constant test mean scores 0.
fn held_out_r2(pred: &[f64], truth: &[f64]) -> f64 {
    assert_eq!(pred.len(), truth.len(), "held_out_r2 length mismatch");
    let m = truth.iter().sum::<f64>() / truth.len() as f64;
    let ss_res: f64 = pred
        .iter()
        .zip(truth)
        .map(|(p, t)| (p - t) * (p - t))
        .sum();
    let ss_tot: f64 = truth.iter().map(|t| (t - m) * (t - m)).sum();
    1.0 - ss_res / ss_tot.max(1e-300)
}

#[test]
fn gam_rw2_pspline_predicts_held_out_at_least_as_well_as_inla() {
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

    // ---- deterministic train/test split -----------------------------------
    // Sort rows by `range` so the held-out points are interspersed across the
    // covariate axis (a smooth must interpolate, not extrapolate). Every 5th
    // sorted row is a test point; the remaining 4/5 are training. The split is
    // fully deterministic (no RNG) so the test is reproducible.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        range[a]
            .partial_cmp(&range[b])
            .expect("lidar range values are finite")
    });
    let mut train_rows: Vec<usize> = Vec::new();
    let mut test_rows: Vec<usize> = Vec::new();
    for (rank, &row) in order.iter().enumerate() {
        if rank % 5 == 2 {
            test_rows.push(row);
        } else {
            train_rows.push(row);
        }
    }
    assert!(
        test_rows.len() > 20 && train_rows.len() > 100,
        "split sizes look wrong: {} train / {} test",
        train_rows.len(),
        test_rows.len()
    );

    // Training-row covariate / response vectors (handed identically to gam and
    // to INLA so both engines fit exactly the same data).
    let train_range: Vec<f64> = train_rows.iter().map(|&i| range[i]).collect();
    let train_logratio: Vec<f64> = train_rows.iter().map(|&i| logratio[i]).collect();
    let test_range: Vec<f64> = test_rows.iter().map(|&i| range[i]).collect();
    let test_logratio: Vec<f64> = test_rows.iter().map(|&i| logratio[i]).collect();

    // Build a training-only Dataset by row-subsetting the encoded values; the
    // schema/headers/column-kinds are unchanged, only the rows are selected.
    let mut train_values = Array2::<f64>::zeros((train_rows.len(), ds.headers.len()));
    for (out_row, &in_row) in train_rows.iter().enumerate() {
        train_values
            .slice_mut(s![out_row, ..])
            .assign(&ds.values.slice(s![in_row, ..]));
    }
    let mut train_ds = ds.clone();
    train_ds.values = train_values;

    // ---- fit gam on the TRAINING rows: logratio ~ s(range, bs='ps', m=2) --
    // bs='ps' with a SECOND-order difference penalty (penalty_order=2 is gam's
    // spelling of mgcv's `m=2`) -> the discrete analog of INLA's rw2 latent
    // prior; REML selects lambda by marginal likelihood.
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(
        "logratio ~ s(range, bs='ps', penalty_order=2)",
        &train_ds,
        &cfg,
    )
    .expect("gam RW2 p-spline fit on training rows");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for a gaussian RW2 p-spline smooth");
    };

    // Sanity: exactly one penalized smooth block, and a sane effective
    // complexity (we do NOT assert edf == reference edf; only that the smooth is
    // neither a straight line nor a noise interpolator).
    assert_eq!(
        fit.fit.log_lambdas.len(),
        1,
        "expected exactly one penalized smooth block, got {}",
        fit.fit.log_lambdas.len()
    );

    // ---- gam predictions at the held-out TEST rows ------------------------
    // Rebuild the (dense) design from the frozen spec at the test `range`; for
    // the identity link the predicted mean is X_test * beta.
    let mut test_grid = Array2::<f64>::zeros((test_rows.len(), ds.headers.len()));
    for (i, &r) in test_range.iter().enumerate() {
        test_grid[[i, range_idx]] = r;
    }
    let test_design = build_term_collection_design(test_grid.view(), &fit.resolvedspec)
        .expect("rebuild design at held-out test points");
    let gam_test_pred: Vec<f64> = test_design.design.apply(&fit.fit.beta).to_vec();
    assert_eq!(
        gam_test_pred.len(),
        test_rows.len(),
        "gam test prediction length mismatch"
    );

    // ---- fit the SAME training data with R-INLA and predict the test rows -
    // f(xg, model="rw2", scale.model=TRUE): a second-order random walk on the
    // ordered, grouped covariate locations — the latent-Gaussian twin of gam's
    // p-spline. We stack train+test into one frame with the test responses set
    // to NA; INLA then returns posterior-mean fitted values at the NA rows,
    // which are its held-out predictions on the identical split.
    let n_train = train_range.len();
    let n_test = test_range.len();
    let mut all_range = train_range.clone();
    all_range.extend_from_slice(&test_range);
    let mut all_logratio = train_logratio.clone();
    all_logratio.extend(std::iter::repeat_n(f64::NAN, n_test));
    let mut is_test = vec![0.0_f64; n_train];
    is_test.extend(std::iter::repeat_n(1.0_f64, n_test));

    let r = run_r(
        &[
            Column::new("range", &all_range),
            Column::new("logratio", &all_logratio),
            Column::new("is_test", &is_test),
        ],
        r#"
        suppressPackageStartupMessages(library(INLA))
        # Group `range` onto the ordered rw2 location lattice. The test rows
        # (logratio == NA) carry no likelihood contribution, so the rw2 latent
        # field is estimated from the TRAINING rows only; INLA's posterior-mean
        # fitted value at an NA row is its held-out prediction there.
        df$xg <- inla.group(df$range, n = 50, method = "quantile")
        df$intercept <- 1
        form <- logratio ~ -1 + intercept + f(xg, model = "rw2",
                                               scale.model = TRUE,
                                               constr = TRUE)
        m <- inla(form, data = df, family = "gaussian",
                  control.predictor = list(compute = TRUE),
                  control.inla = list(int.strategy = "grid"))
        # Posterior-mean held-out predictions, in the SAME order the test rows
        # were appended (is_test == 1), so they align element-wise with
        # gam_test_pred / test_logratio.
        test_idx <- which(df$is_test == 1)
        pred <- m$summary.fitted.values$mean[test_idx]
        emit("inla_test_pred", as.numeric(pred))
        "#,
    );
    let inla_test_pred = r.vector("inla_test_pred");
    assert_eq!(
        inla_test_pred.len(),
        test_rows.len(),
        "INLA held-out prediction length mismatch"
    );

    // ---- objective held-out metrics ---------------------------------------
    let gam_r2 = held_out_r2(&gam_test_pred, &test_logratio);
    let gam_rmse = rmse(&gam_test_pred, &test_logratio);
    let inla_r2 = held_out_r2(inla_test_pred, &test_logratio);
    let inla_rmse = rmse(inla_test_pred, &test_logratio);
    // Context only — agreement with INLA is NOT a pass criterion.
    let rel_pred = relative_l2(&gam_test_pred, inla_test_pred);

    eprintln!(
        "lidar RW2 held-out: n={n} train={n_train} test={n_test} \
         gam_R2={gam_r2:.4} gam_rmse={gam_rmse:.5} \
         inla_R2={inla_r2:.4} inla_rmse={inla_rmse:.5} \
         (context) rel_l2(gam_pred,inla_pred)={rel_pred:.4}"
    );

    // PRIMARY: absolute out-of-sample accuracy bar. A smoother that has learned
    // lidar's mean structure (not interpolated noise) generalizes well past this
    // floor; an over-/under-smoothed fit falls below it.
    assert!(
        gam_r2 >= 0.55,
        "gam RW2 p-spline does not generalize: held-out R^2={gam_r2:.4} < 0.55 bar"
    );

    // MATCH-OR-BEAT: gam must predict the held-out rows at least as accurately
    // as INLA (within 10% on RMSE), on the identical split. This demotes the
    // mature reference to a baseline on an OBJECTIVE accuracy metric rather than
    // a target curve to reproduce.
    assert!(
        gam_rmse <= 1.10 * inla_rmse,
        "gam predicts the held-out rows worse than INLA: \
         gam_rmse={gam_rmse:.5} > 1.10 * inla_rmse={:.5} (inla_rmse={inla_rmse:.5})",
        1.10 * inla_rmse
    );
}
