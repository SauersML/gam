//! End-to-end quality: gam's cyclic cubic spline (`cc()` / `cyclic()`) must
//! **recover the true periodic signal** it was trained on, and must do so at
//! least as accurately as **mgcv** — the mature, standard GAM implementation.
//!
//! This is a TRUTH-RECOVERY test. The data is generated from a known function
//! `g(t) = sin(t)` corrupted by additive Gaussian noise of known scale
//! `sigma = 0.1`: `h = sin(t) + 0.1*noise`, `t in [0, 2π)`. The objective
//! quality of a smoother is how close its fitted curve lands to that hidden
//! truth — NOT how close it lands to some other tool's (equally noisy) fit.
//!
//! mgcv's `bs="cc"` is the canonical cyclic cubic regression spline; gam
//! exposes the same construction through
//! `cc(t, k=12, period_start=0, period_end=2*pi)`, a `PeriodicUniform` cubic
//! B-spline with a `Cyclic` boundary. Both fit by REML against a Gaussian
//! likelihood. The same `(t, h)` samples (n=100, seed=42) are handed to both.
//!
//! ASSERTIONS (all objective):
//!   1. TRUTH RECOVERY (primary): the RMSE of gam's fitted curve against the
//!      true `sin(t)`, on a dense grid over one period, is below the noise
//!      scale — `rmse(gam, sin) <= 0.5*sigma = 0.05`. A good smoother strips
//!      most of the noise, so its error sits well under one sigma.
//!   2. MATCH-OR-BEAT (accuracy): gam's truth-recovery error is no worse than
//!      mgcv's by more than 10% — `rmse(gam, sin) <= rmse(mgcv, sin)*1.10`.
//!      The mature tool is a baseline to match-or-beat on ACCURACY, not an
//!      oracle whose noisy output gam must reproduce.
//!   3. STRUCTURE — periodic seam continuity: gam genuinely enforces the wrap,
//!      `fit(0) == fit(2π)` to 1e-6. This is the defining property of a cyclic
//!      basis and is asserted directly on gam's own fit.
//!
//! The reference rel-L2 (gam vs mgcv) is still computed and printed for
//! context, but it is NOT a pass criterion.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pad_to, r2, relative_l2, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
    load_csvwith_inferred_schema,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use std::f64::consts::PI;
use std::path::Path;

/// Real monthly-temperature series, `nottem` from R's `datasets` package
/// (average air temperature at Nottingham Castle, 1920-1939, in degrees F),
/// reshaped to one row per (year, month). Source: R `datasets::nottem`.
const NOTTEM_CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/bench/datasets/nottem_monthly_temp.csv"
);

#[test]
fn gam_cyclic_cubic_matches_mgcv_on_sine() {
    init_parallelism();

    // ---- synthetic periodic data: t in [0,2π), h = sin(t) + 0.1*noise ------
    // Generated once and handed IDENTICALLY to gam and mgcv.
    let n = 100usize;
    let period = 2.0 * PI;
    let mut rng = StdRng::seed_from_u64(42);
    let noise = Normal::new(0.0, 1.0).expect("normal");
    let t: Vec<f64> = (0..n).map(|i| period * i as f64 / n as f64).collect();
    let h: Vec<f64> = t
        .iter()
        .map(|&ti| ti.sin() + 0.1 * noise.sample(&mut rng))
        .collect();

    let headers: Vec<String> = vec!["t".to_string(), "h".to_string()];
    let rows = t
        .iter()
        .zip(h.iter())
        .map(|(a, b)| csv::StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect::<Vec<_>>();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode cyclic dataset");
    let col = ds.column_map();
    let t_idx = col["t"];

    // ---- fit with gam: h ~ cc(t, k=12, period_start=0, period_end=2π) ------
    // The DSL parses option values with a plain f64 parse (no expression eval),
    // so `2*pi` must be passed as a numeric literal for `period_end`.
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let formula = format!("h ~ cc(t, k=12, period_start=0, period_end={period:.17})");
    let result = fit_from_formula(&formula, &ds, &cfg).expect("gam cyclic fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for a gaussian cyclic smooth");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // Evaluate gam's fitted function on a dense grid over one period [0, 2π).
    // We also append the seam point t=2π so we can verify the wrap directly.
    let grid_n = 200usize;
    let mut grid_t: Vec<f64> = (0..grid_n)
        .map(|i| period * i as f64 / grid_n as f64)
        .collect();
    grid_t.push(period); // last entry is exactly one period after grid_t[0]==0

    let mut design_pts = Array2::<f64>::zeros((grid_t.len(), ds.headers.len()));
    for (i, &gt) in grid_t.iter().enumerate() {
        design_pts[[i, t_idx]] = gt;
    }
    let design = build_term_collection_design(design_pts.view(), &fit.resolvedspec)
        .expect("rebuild design at grid points");
    let gam_fitted: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();

    // Periodic-wrap check: fitted(0) must equal fitted(2π). This is the
    // defining guarantee of a cyclic basis (bs="cc") — value continuity across
    // the seam — and is exact up to floating point for a true wrapped spline.
    let wrap_gap = (gam_fitted[0] - gam_fitted[grid_n]).abs();

    // Fitted values on the in-period grid only (drop the appended seam point).
    let gam_grid_fit = &gam_fitted[..grid_n];

    // ---- fit the SAME model with mgcv bs="cc" (the mature reference) -------
    let r = run_r(
        &[Column::new("t", &t), Column::new("h", &h)],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        m <- gam(h ~ s(t, bs = "cc", k = 12), data = df, method = "REML",
                 knots = list(t = seq(0, 2 * pi, length = 12)))
        gridn <- 200
        gt <- (2 * pi) * (0:(gridn - 1)) / gridn
        pr <- as.numeric(predict(m, newdata = data.frame(t = gt)))
        emit("fitted", pr)
        emit("edf", sum(m$edf))
        "#,
    );
    let mgcv_fitted = r.vector("fitted");
    let mgcv_edf = r.scalar("edf");
    assert_eq!(
        mgcv_fitted.len(),
        grid_n,
        "mgcv grid prediction length mismatch"
    );

    // ---- objective quality: recovery of the TRUE signal sin(t) -------------
    // The grid points are noise-free abscissae, so the hidden truth at each is
    // exactly sin(grid_t[i]). Compare both smoothers' fits to that truth.
    let truth: Vec<f64> = grid_t[..grid_n].iter().map(|&gt| gt.sin()).collect();
    let gam_truth_rmse = rmse(gam_grid_fit, &truth);
    let mgcv_truth_rmse = rmse(mgcv_fitted, &truth);

    // Reference closeness is computed for CONTEXT only — never a pass criterion.
    let rel_to_mgcv = relative_l2(gam_grid_fit, mgcv_fitted);

    eprintln!(
        "cyclic cc(t): n={n} sigma=0.1 gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} \
         rmse(gam,sin)={gam_truth_rmse:.5} rmse(mgcv,sin)={mgcv_truth_rmse:.5} \
         rel_l2(gam,mgcv)={rel_to_mgcv:.5} wrap_gap={wrap_gap:.3e}"
    );

    // 1) TRUTH RECOVERY (primary): the fitted curve must land close to sin(t).
    // With sigma=0.1 noise over n=100, a good periodic smoother removes most of
    // the noise; its curve-vs-truth RMSE should sit comfortably below half a
    // sigma. This asserts gam's OWN accuracy against ground truth.
    let sigma = 0.1;
    assert!(
        gam_truth_rmse <= 0.5 * sigma,
        "gam cyclic fit does not recover sin(t): rmse(gam,sin)={gam_truth_rmse:.5} > {:.5}",
        0.5 * sigma
    );

    // 2) MATCH-OR-BEAT (accuracy): gam must be at least as accurate as mgcv at
    // recovering the truth, allowing a 10% slack for basis/centering gaps.
    assert!(
        gam_truth_rmse <= mgcv_truth_rmse * 1.10,
        "gam less accurate than mgcv at recovering sin(t): \
         rmse(gam,sin)={gam_truth_rmse:.5} > 1.10*rmse(mgcv,sin)={:.5}",
        mgcv_truth_rmse * 1.10
    );

    // 3) STRUCTURE — periodic seam continuity: a genuine cyclic basis has
    // identical design rows at t and t+period, so the fit must wrap exactly.
    assert!(
        wrap_gap < 1e-6,
        "cyclic wrap not enforced: |fit(0) - fit(2π)| = {wrap_gap:.3e}"
    );
}

/// REAL-DATA arm of the cyclic-cubic capability: the SAME `cc()` periodic
/// smooth, now exercised on `nottem` monthly temperatures (no known truth).
///
/// The annual temperature cycle is the textbook periodic signal: month 1..=12
/// wraps back to month 1 the next year, so `temp ~ cc(month)` must capture a
/// smooth seasonal curve that genuinely PREDICTS held-out months — not merely
/// reproduce mgcv's in-sample fit.
///
/// We make a deterministic train/test split (every 4th row held out), fit a
/// cyclic cubic spline by REML on the training rows only, predict the held-out
/// rows, and assert OBJECTIVE metrics on gam's OWN predictions:
///
///   PRIMARY (objective, tool-free): held-out coefficient of determination
///     `test_R2 >= 0.85` — the seasonal cycle dominates the variance, so a
///     competent periodic smoother explains the vast majority of held-out
///     variation, far above the constant-mean predictor (R2 = 0).
///
///   BASELINE (match-or-beat): mgcv `bs="cc"` fits the SAME training rows and
///     predicts the SAME held-out rows; gam's held-out RMSE must be no worse
///     than `mgcv_test_rmse * 1.10`. mgcv is a baseline to match-or-beat on
///     accuracy, NOT a fitted target to reproduce.
///
///   STRUCTURE — periodic seam continuity: gam's fitted curve wraps exactly,
///     `fit(month=1) == fit(month=13)` to 1e-6, the defining cyclic property.
#[test]
fn gam_cyclic_cubic_matches_mgcv_on_sine_on_real_data() {
    init_parallelism();

    // ---- load real monthly temperatures: month (1..12) -> temp ------------
    let ds = load_csvwith_inferred_schema(Path::new(NOTTEM_CSV)).expect("load nottem csv");
    let col = ds.column_map();
    let month_idx = col["month"];
    let temp_idx = col["temp"];
    let month: Vec<f64> = ds.values.column(month_idx).to_vec();
    let temp: Vec<f64> = ds.values.column(temp_idx).to_vec();
    let n = month.len();
    assert!(n > 200, "nottem should have ~240 rows, got {n}");

    // ---- deterministic train/test split: every 4th row is held out -------
    let is_test = |i: usize| i % 4 == 0;
    let train_rows: Vec<usize> = (0..n).filter(|&i| !is_test(i)).collect();
    let test_rows: Vec<usize> = (0..n).filter(|&i| is_test(i)).collect();
    assert!(
        train_rows.len() > 150 && test_rows.len() > 50,
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

    // ---- fit gam on TRAIN: temp ~ cc(month, k=12, period 1..13), REML ------
    // The period spans month=1 to month=13 (== month 1 of the next cycle), so
    // the cyclic basis wraps December back onto January. The DSL parses option
    // values as plain f64 literals, so the period bounds are numeric literals.
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let formula = "temp ~ cc(month, k=12, period_start=1, period_end=13)";
    let result = fit_from_formula(formula, &train_ds, &cfg).expect("gam cyclic fit on nottem");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for a gaussian cyclic smooth");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // gam predictions at the held-out months: rebuild the design from the
    // frozen spec (identity link => design*beta = predicted mean). Append the
    // seam pair (month=1, month=13) so we can verify the wrap on gam's own fit.
    let n_test = test_rows.len();
    let mut pred_grid = Array2::<f64>::zeros((n_test + 2, p));
    for (i, &m) in test_month.iter().enumerate() {
        pred_grid[[i, month_idx]] = m;
    }
    pred_grid[[n_test, month_idx]] = 1.0;
    pred_grid[[n_test + 1, month_idx]] = 13.0;
    let pred_design = build_term_collection_design(pred_grid.view(), &fit.resolvedspec)
        .expect("rebuild design at held-out + seam points");
    let gam_pred_all: Vec<f64> = pred_design.design.apply(&fit.fit.beta).to_vec();
    let gam_test_pred = &gam_pred_all[..n_test];
    let wrap_gap = (gam_pred_all[n_test] - gam_pred_all[n_test + 1]).abs();

    // ---- fit the SAME model on TRAIN with mgcv bs="cc", predict TEST -------
    // One data.frame per call: train month/temp plus the test months padded into
    // a parallel column (only the first `test_n` entries are read back).
    let train_r = run_r(
        &[
            Column::new("month", &train_month),
            Column::new("temp", &train_temp),
            Column::new("test_month", &pad_to(&test_month, train_month.len())),
            Column::new("test_n", &vec![n_test as f64; train_month.len()]),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        m <- gam(temp ~ s(month, bs = "cc", k = 12), data = df, method = "REML",
                 knots = list(month = seq(1, 13, length = 12)))
        emit("edf", sum(m$edf))
        k <- df$test_n[1]
        newd <- data.frame(month = df$test_month[1:k])
        emit("test_pred", as.numeric(predict(m, newdata = newd)))
        emit("fitted", as.numeric(fitted(m)))
        "#,
    );
    let mgcv_edf = train_r.scalar("edf");
    let mgcv_test_pred = train_r.vector("test_pred");
    let mgcv_train_fitted = train_r.vector("fitted").to_vec();
    assert_eq!(
        mgcv_test_pred.len(),
        n_test,
        "mgcv held-out prediction length mismatch"
    );

    // ---- objective metrics on gam's OWN predictions -----------------------
    let gam_test_r2 = r2(gam_test_pred, &test_temp);
    let gam_test_rmse = rmse(gam_test_pred, &test_temp);
    let mgcv_test_rmse = rmse(mgcv_test_pred, &test_temp);

    // Context-only diagnostic: closeness of gam's in-sample fit vs mgcv's. NOT
    // a pass criterion.
    let mut gam_train_grid = Array2::<f64>::zeros((train_rows.len(), p));
    for (i, &m) in train_month.iter().enumerate() {
        gam_train_grid[[i, month_idx]] = m;
    }
    let gam_train_design = build_term_collection_design(gam_train_grid.view(), &fit.resolvedspec)
        .expect("rebuild design at training points");
    let gam_train_fitted: Vec<f64> = gam_train_design.design.apply(&fit.fit.beta).to_vec();
    let insample_rel = relative_l2(&gam_train_fitted, &mgcv_train_fitted);

    eprintln!(
        "nottem cc(month) held-out: n_train={} n_test={} gam_edf={gam_edf:.3} \
         mgcv_edf={mgcv_edf:.3} gam_test_R2={gam_test_r2:.4} \
         gam_test_rmse={gam_test_rmse:.4} mgcv_test_rmse={mgcv_test_rmse:.4} \
         wrap_gap={wrap_gap:.3e} (context: in-sample rel_l2 vs mgcv={insample_rel:.4})",
        train_rows.len(),
        n_test,
    );

    // ---- PRIMARY objective assertion: gam predicts the seasonal cycle ------
    // The annual temperature cycle dominates the variance; a competent periodic
    // smoother explains the bulk of held-out variation. R2 >= 0.85 is far above
    // the constant-mean baseline (0) and would catch under/over-smoothing.
    assert!(
        gam_test_r2 >= 0.85,
        "gam's held-out predictive R2 too low: {gam_test_r2:.4} (< 0.85)"
    );

    // ---- BASELINE (match-or-beat): no worse than mgcv on held-out RMSE -----
    assert!(
        gam_test_rmse <= mgcv_test_rmse * 1.10,
        "gam held-out RMSE {gam_test_rmse:.4} exceeds mgcv {mgcv_test_rmse:.4} * 1.10"
    );

    // ---- STRUCTURE — periodic seam continuity on gam's OWN fit -------------
    assert!(
        wrap_gap < 1e-6,
        "cyclic wrap not enforced on real data: |fit(month=1) - fit(month=13)| = {wrap_gap:.3e}"
    );
}
