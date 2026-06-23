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
use gam::test_support::reference::{
    Column, held_out_r2, r_package_available, relative_l2, rmse, run_r,
};
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::{Array2, s};
use std::path::Path;

const LIDAR_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/lidar.csv");
// Source: R `datasets::quakes` (Harvard PRIM-H earthquake catalogue, locations
// near Fiji, 1964 onward), shipped here as bench/datasets/quakes.csv with
// columns rownames,lat,long,depth,mag,stations (n = 1000).
const QUAKES_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/quakes.csv");

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

    // Sanity: the single `s(range, bs='ps')` term is penalized, and a sane
    // effective complexity (we do NOT assert edf == reference edf; only that the
    // smooth is neither a straight line nor a noise interpolator). gam's default
    // p-spline is *double-penalized* — one smoothing parameter on the range
    // (second-difference) penalty plus one on the null-space shrinkage penalty
    // (the `double_penalty=true` default in `term_builder.rs`) — so a healthy
    // fit carries 1..=2 smoothing parameters, not exactly 1. We assert the term
    // is penalized at all (no degenerate unpenalized fit) rather than pinning an
    // implementation-specific block count.
    assert!(
        (1..=2).contains(&fit.fit.log_lambdas.len()),
        "expected the single p-spline term to be penalized (1 range + optional \
         null-space shrinkage block), got {} smoothing parameters",
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

    // Environmental gate: R-INLA is provisioned best-effort in CI and is
    // frequently unavailable. When it is, we cannot run the match-or-beat arm,
    // but gam's tool-free absolute accuracy claim still stands on its own and we
    // assert it here (the IDENTICAL held-out R^2 >= 0.55 bar this test asserts
    // below) rather than silently skipping the whole test.
    if !r_package_available("INLA") {
        let gam_r2 = held_out_r2(&gam_test_pred, &test_logratio);
        eprintln!(
            "R-INLA unavailable — asserting gam's tool-free absolute quality only \
             (skipping match-or-beat arm): gam_R2={gam_r2:.4}"
        );
        assert!(
            gam_r2 >= 0.55,
            "gam RW2 p-spline does not generalize: held-out R^2={gam_r2:.4} < 0.55 bar"
        );
        return;
    }

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

/// REAL-DATA arm: the same penalized-smoothness capability (a 2-D thin-plate
/// spatial smooth, the planar twin of the 1-D RW2 p-spline above), exercised on
/// the `quakes` earthquake catalogue and scored by **held-out predictive
/// accuracy** with **R-INLA's SPDE** spatial model demoted to a match-or-beat
/// baseline.
///
/// ## What this test asserts (objective metric)
///
/// `quakes` (n = 1000, near Fiji) has no known ground-truth surface, so the
/// honest quality claim is predictive: does gam's 2-D smooth of magnitude over
/// geographic location generalize to quakes it never saw? We make a fully
/// deterministic train/test split (every 5th row, by original file order, is
/// held out), fit `mag ~ s(long, lat, bs='tp')` on the training rows only,
/// predict the held-out rows from the frozen smooth, and assert
///
///   1. an **absolute held-out accuracy bar**: out-of-sample
///      `R^2 = 1 - SS_res/SS_tot >= 0.20` on the test rows. Earthquake
///      magnitude varies smoothly but noisily with subduction-zone geometry;
///      a spatial smooth that has learned that structure (rather than the grand
///      mean) clears this comfortably while predicting the constant mean scores
///      0. This PRIMARY claim stands alone without any reference tool.
///
///   2. a **match-or-beat against R-INLA's SPDE** on the same held-out rows:
///      gam's out-of-sample RMSE must be within 10% of INLA's,
///      `gam_rmse <= 1.10 * inla_rmse`. INLA's stochastic-PDE Matern field
///      (`inla.spde2.pcmatern` on a Delaunay mesh of the training locations) is
///      the best-in-class continuous spatial-smoothing engine and the 2-D
///      continuous analog of gam's thin-plate basis; it is fit to the IDENTICAL
///      training rows and asked to predict the IDENTICAL held-out longitudes /
///      latitudes. We assert gam predicts unseen quakes at least as accurately
///      as INLA, not that gam reproduces INLA's fitted surface.
///
/// A failing assertion because gam genuinely predicts worse than the bar (or
/// worse than INLA) is a real finding, not a reason to weaken the bound.
#[test]
fn gam_rw2_pspline_predicts_held_out_at_least_as_well_as_inla_on_real_data() {
    init_parallelism();

    // ---- load the quakes catalogue (long, lat -> mag spatial 2-D) ---------
    let ds = load_csvwith_inferred_schema(Path::new(QUAKES_CSV)).expect("load quakes.csv");
    let col = ds.column_map();
    let long_idx = col["long"];
    let lat_idx = col["lat"];
    let mag_idx = col["mag"];
    let long: Vec<f64> = ds.values.column(long_idx).to_vec();
    let lat: Vec<f64> = ds.values.column(lat_idx).to_vec();
    let mag: Vec<f64> = ds.values.column(mag_idx).to_vec();
    let n = long.len();
    assert!(n > 500, "quakes should have ~1000 rows, got {n}");

    // ---- deterministic train/test split: every 5th row held out ----------
    // Original file order; row r is a test point iff r % 5 == 2. Fully
    // deterministic (no RNG), so the split is reproducible and identical for
    // gam and INLA.
    let mut train_rows: Vec<usize> = Vec::new();
    let mut test_rows: Vec<usize> = Vec::new();
    for r in 0..n {
        if r % 5 == 2 {
            test_rows.push(r);
        } else {
            train_rows.push(r);
        }
    }
    assert!(
        test_rows.len() > 150 && train_rows.len() > 600,
        "split sizes look wrong: {} train / {} test",
        train_rows.len(),
        test_rows.len()
    );

    let train_long: Vec<f64> = train_rows.iter().map(|&i| long[i]).collect();
    let train_lat: Vec<f64> = train_rows.iter().map(|&i| lat[i]).collect();
    let train_mag: Vec<f64> = train_rows.iter().map(|&i| mag[i]).collect();
    let test_long: Vec<f64> = test_rows.iter().map(|&i| long[i]).collect();
    let test_lat: Vec<f64> = test_rows.iter().map(|&i| lat[i]).collect();
    let test_mag: Vec<f64> = test_rows.iter().map(|&i| mag[i]).collect();

    // Training-only dataset: row-subset the encoded values; schema/headers/
    // column-kinds are unchanged so the formula resolves identically.
    let p = ds.headers.len();
    let mut train_values = Array2::<f64>::zeros((train_rows.len(), p));
    for (out_row, &in_row) in train_rows.iter().enumerate() {
        train_values
            .slice_mut(s![out_row, ..])
            .assign(&ds.values.slice(s![in_row, ..]));
    }
    let mut train_ds = ds.clone();
    train_ds.values = train_values;

    // ---- fit gam on TRAIN: mag ~ s(long, lat, bs='tp'), REML --------------
    // A 2-D thin-plate regression spline over geographic location: the planar,
    // isotropic analog of the 1-D RW2 smoothness penalty, with REML selecting
    // the smoothing parameter by marginal likelihood.
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("mag ~ s(long, lat, bs='tp')", &train_ds, &cfg)
        .expect("gam 2-D thin-plate spatial fit on training rows");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for a gaussian 2-D thin-plate smooth");
    };

    // ---- gam predictions at the held-out TEST rows ------------------------
    // Rebuild the design from the frozen spec at the test (long, lat); identity
    // link => predicted mean is X_test * beta.
    let mut test_grid = Array2::<f64>::zeros((test_rows.len(), p));
    for i in 0..test_rows.len() {
        test_grid[[i, long_idx]] = test_long[i];
        test_grid[[i, lat_idx]] = test_lat[i];
    }
    let test_design = build_term_collection_design(test_grid.view(), &fit.resolvedspec)
        .expect("rebuild 2-D design at held-out test points");
    let gam_test_pred: Vec<f64> = test_design.design.apply(&fit.fit.beta).to_vec();
    assert_eq!(
        gam_test_pred.len(),
        test_rows.len(),
        "gam test prediction length mismatch"
    );

    // ---- fit the SAME training data with R-INLA's SPDE, predict the test --
    // inla.spde2.pcmatern on a Delaunay mesh of the TRAINING locations: a
    // continuous Gaussian-Markov random field (Matern via the stochastic PDE),
    // the 2-D continuous twin of gam's thin-plate basis. We stack train+test
    // into one frame with test responses NA; the test rows carry no likelihood,
    // so the field is estimated from TRAINING rows only and INLA's posterior-
    // mean fitted value at an NA row is its held-out prediction there.
    let n_train = train_long.len();
    let n_test = test_long.len();
    let mut all_long = train_long.clone();
    all_long.extend_from_slice(&test_long);
    let mut all_lat = train_lat.clone();
    all_lat.extend_from_slice(&test_lat);
    let mut all_mag = train_mag.clone();
    all_mag.extend(std::iter::repeat_n(f64::NAN, n_test));
    let mut is_test = vec![0.0_f64; n_train];
    is_test.extend(std::iter::repeat_n(1.0_f64, n_test));

    // Environmental gate: R-INLA is provisioned best-effort in CI and is
    // frequently unavailable. When it is, we cannot run the match-or-beat arm,
    // but gam's tool-free absolute accuracy claim still stands on its own and we
    // assert it here (the IDENTICAL held-out R^2 >= 0.20 bar this test asserts
    // below) rather than silently skipping the whole test.
    if !r_package_available("INLA") {
        let gam_r2 = held_out_r2(&gam_test_pred, &test_mag);
        eprintln!(
            "R-INLA unavailable — asserting gam's tool-free absolute quality only \
             (skipping match-or-beat arm): gam_R2={gam_r2:.4}"
        );
        assert!(
            gam_r2 >= 0.20,
            "gam 2-D spatial smooth does not generalize: held-out R^2={gam_r2:.4} < 0.20 bar"
        );
        return;
    }

    let r = run_r(
        &[
            Column::new("long", &all_long),
            Column::new("lat", &all_lat),
            Column::new("mag", &all_mag),
            Column::new("is_test", &is_test),
        ],
        r#"
        suppressPackageStartupMessages(library(INLA))
        # Mesh over the TRAINING locations only (test rows are predicted via the
        # projector, never used to build the mesh or estimate the field).
        train <- df[df$is_test == 0, ]
        test  <- df[df$is_test == 1, ]
        loc.train <- cbind(train$long, train$lat)
        loc.test  <- cbind(test$long,  test$lat)
        mesh <- inla.mesh.2d(loc = loc.train,
                             max.edge = c(0.75, 3),
                             cutoff = 0.3)
        spde <- inla.spde2.pcmatern(mesh = mesh,
                                    prior.range = c(1, 0.5),
                                    prior.sigma = c(1, 0.5))
        A.train <- inla.spde.make.A(mesh = mesh, loc = loc.train)
        A.test  <- inla.spde.make.A(mesh = mesh, loc = loc.test)
        s.index <- inla.spde.make.index(name = "spatial", n.spde = spde$n.spde)
        stk.train <- inla.stack(
          data = list(mag = train$mag),
          A = list(A.train, 1),
          effects = list(c(s.index, list(intercept = 1)), list()),
          tag = "train")
        stk.test <- inla.stack(
          data = list(mag = rep(NA, nrow(test))),
          A = list(A.test, 1),
          effects = list(c(s.index, list(intercept = 1)), list()),
          tag = "test")
        stk <- inla.stack(stk.train, stk.test)
        form <- mag ~ -1 + intercept + f(spatial, model = spde)
        m <- inla(form, data = inla.stack.data(stk), family = "gaussian",
                  control.predictor = list(A = inla.stack.A(stk), compute = TRUE))
        idx.test <- inla.stack.index(stk, tag = "test")$data
        pred <- m$summary.fitted.values$mean[idx.test]
        # Emitted in the SAME order the test rows were appended, so they align
        # element-wise with gam_test_pred / test_mag.
        emit("inla_test_pred", as.numeric(pred))
        "#,
    );
    let inla_test_pred = r.vector("inla_test_pred");
    assert_eq!(
        inla_test_pred.len(),
        test_rows.len(),
        "INLA SPDE held-out prediction length mismatch"
    );

    // ---- objective held-out metrics ---------------------------------------
    let gam_r2 = held_out_r2(&gam_test_pred, &test_mag);
    let gam_rmse = rmse(&gam_test_pred, &test_mag);
    let inla_r2 = held_out_r2(inla_test_pred, &test_mag);
    let inla_rmse = rmse(inla_test_pred, &test_mag);
    // Context only — agreement with INLA is NOT a pass criterion.
    let rel_pred = relative_l2(&gam_test_pred, inla_test_pred);

    eprintln!(
        "quakes 2-D spatial held-out: n={n} train={n_train} test={n_test} \
         gam_R2={gam_r2:.4} gam_rmse={gam_rmse:.5} \
         inla_R2={inla_r2:.4} inla_rmse={inla_rmse:.5} \
         (context) rel_l2(gam_pred,inla_pred)={rel_pred:.4}"
    );

    // PRIMARY: absolute out-of-sample accuracy bar. A spatial smooth that has
    // learned the subduction-zone magnitude structure (not the grand mean)
    // clears this floor; a degenerate fit falls below it.
    assert!(
        gam_r2 >= 0.20,
        "gam 2-D spatial smooth does not generalize: held-out R^2={gam_r2:.4} < 0.20 bar"
    );

    // MATCH-OR-BEAT: gam must predict the held-out rows at least as accurately
    // as INLA's SPDE (within 10% on RMSE), on the identical split.
    assert!(
        gam_rmse <= 1.10 * inla_rmse,
        "gam predicts the held-out quakes worse than INLA SPDE: \
         gam_rmse={gam_rmse:.5} > 1.10 * inla_rmse={:.5} (inla_rmse={inla_rmse:.5})",
        1.10 * inla_rmse
    );
}

// ============================ #1074 DIAGNOSTIC ============================
// TEMPORARY: localize the rw2/lidar held-out R^2≈0.02 failure. R-free. Dumps
// train R^2, held-out R^2, edf, lambdas, and sample (range, pred, actual).
#[test]
fn diag_rw2_lidar_1074() {
    init_parallelism();
    let ds = load_csvwith_inferred_schema(Path::new(LIDAR_CSV)).expect("load lidar.csv");
    let col = ds.column_map();
    let range_idx = col["range"];
    let logratio_idx = col["logratio"];
    let range: Vec<f64> = ds.values.column(range_idx).to_vec();
    let logratio: Vec<f64> = ds.values.column(logratio_idx).to_vec();
    let n = range.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| range[a].partial_cmp(&range[b]).unwrap());
    let mut train_rows: Vec<usize> = Vec::new();
    let mut test_rows: Vec<usize> = Vec::new();
    for (rank, &row) in order.iter().enumerate() {
        if rank % 5 == 2 { test_rows.push(row); } else { train_rows.push(row); }
    }
    let train_range: Vec<f64> = train_rows.iter().map(|&i| range[i]).collect();
    let train_logratio: Vec<f64> = train_rows.iter().map(|&i| logratio[i]).collect();
    let test_range: Vec<f64> = test_rows.iter().map(|&i| range[i]).collect();
    let test_logratio: Vec<f64> = test_rows.iter().map(|&i| logratio[i]).collect();
    let mut train_values = Array2::<f64>::zeros((train_rows.len(), ds.headers.len()));
    for (out_row, &in_row) in train_rows.iter().enumerate() {
        train_values.slice_mut(s![out_row, ..]).assign(&ds.values.slice(s![in_row, ..]));
    }
    let mut train_ds = ds.clone();
    train_ds.values = train_values;
    let cfg = FitConfig { family: Some("gaussian".to_string()), ..FitConfig::default() };
    let result = fit_from_formula("logratio ~ s(range, bs='ps', penalty_order=2)", &train_ds, &cfg).unwrap();
    let FitResult::Standard(fit) = result else { panic!() };

    // train-row predictions (in-sample)
    let mut train_grid = Array2::<f64>::zeros((train_rows.len(), ds.headers.len()));
    for (i, &r) in train_range.iter().enumerate() { train_grid[[i, range_idx]] = r; }
    let train_design = build_term_collection_design(train_grid.view(), &fit.resolvedspec).unwrap();
    let train_pred: Vec<f64> = train_design.design.apply(&fit.fit.beta).to_vec();

    // held-out predictions
    let mut test_grid = Array2::<f64>::zeros((test_rows.len(), ds.headers.len()));
    for (i, &r) in test_range.iter().enumerate() { test_grid[[i, range_idx]] = r; }
    let test_design = build_term_collection_design(test_grid.view(), &fit.resolvedspec).unwrap();
    let test_pred: Vec<f64> = test_design.design.apply(&fit.fit.beta).to_vec();

    let train_r2 = held_out_r2(&train_pred, &train_logratio);
    let test_r2 = held_out_r2(&test_pred, &test_logratio);
    eprintln!(
        "[#1074-rw2] n={n} n_train={} n_test={} edf={:.3} log_lambdas={:?} train_R2={:.4} heldout_R2={:.4}",
        train_rows.len(), test_rows.len(), fit.fit.edf_total().unwrap(),
        fit.fit.log_lambdas.iter().map(|v| (v*1000.0).round()/1000.0).collect::<Vec<_>>(),
        train_r2, test_r2,
    );
    eprintln!("[#1074-rw2] range_min={:.1} range_max={:.1} beta_len={} design_ncol={}",
        range.iter().cloned().fold(f64::INFINITY, f64::min),
        range.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        fit.fit.beta.len(), test_design.design.ncols());
    // sample test points sorted by range
    let mut idx: Vec<usize> = (0..test_rows.len()).collect();
    idx.sort_by(|&a, &b| test_range[a].partial_cmp(&test_range[b]).unwrap());
    for &i in idx.iter().step_by(idx.len().max(8) / 8) {
        eprintln!("[#1074-rw2]   range={:.1} pred={:.4} actual={:.4}", test_range[i], test_pred[i], test_logratio[i]);
    }
}
