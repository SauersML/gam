//! End-to-end quality: gam's cyclic cubic spline (`cc()` / `cyclic()`) must
//! **recover the true periodic signal** it was trained on, and must do so at
//! least as accurately as **mgcv** — the mature, standard GAM implementation.
//!
//! The synthetic arm is a TRUTH-RECOVERY test. The data is generated from a known
//! function `g(t) = sin(t)` corrupted by additive Gaussian noise of known scale
//! `sigma = 0.1`: `h = sin(t) + 0.1*noise`, `t in [0, 2π)`. The objective quality
//! of a smoother is how close its fitted curve lands to that hidden truth.
//!
//! mgcv's `bs="cc"` is the canonical cyclic cubic regression spline; gam exposes
//! the same construction through `cc(t, k=12, period_start=0, period_end=2*pi)`.
//! Both fit by REML against a Gaussian likelihood.
//!
//! #2395 K-averaging: the former single seed / single hold-out put the gam-vs-mgcv
//! margin on a knife-edge whose sign flipped across seeds/splits (pure sampling
//! noise). The synthetic arm now averages truth recovery over K noise seeds (fixed
//! t-grid + truth, only the noise draw varies); the real-data arm averages held-out
//! accuracy over K random partitions. gam and mgcv are scored on the SAME K
//! datasets/partitions (identical per-seed response columns / fold masks shipped
//! into the R body), so the paired comparison stays honest; averaging a
//! lower-variance metric against the same bars is strictly harder than the former
//! single sample, never a weakening. The split-invariant seam-continuity structure
//! check is kept on one representative fit.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, QualityPair, r2, rmse, run_r};
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

/// #2395: K noise seeds (synthetic arm) / K random partitions (real-data arm).
/// Both fits (cc on n=100 / n=240) are sub-millisecond, so 2*K=20 fits are
/// trivially inside the fast envelope while cutting the metric's std error ~3.2x.
const K_SPLITS: usize = 10;
/// Held-out fraction per partition for the real-data arm (~75/25).
const HOLDOUT: f64 = 0.25;

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

/// Deterministic uniform(0,1) hash of (row, split) via splitmix64 — row `i` is in
/// the TEST fold of partition `split` iff it maps below `HOLDOUT`. No RNG dep; gam
/// and mgcv, fed the SAME masks, partition byte-identically.
fn is_heldout(i: usize, split: usize) -> bool {
    let mut z = (i as u64)
        .wrapping_add((split as u64).wrapping_mul(0x9E3779B97F4A7C15))
        .wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 31;
    let u = ((z >> 11) as f64) / ((1u64 << 53) as f64);
    u < HOLDOUT
}

#[test]
fn gam_cyclic_cubic_matches_mgcv_on_sine() {
    init_parallelism();

    // ---- synthetic periodic data: t in [0,2π), h = sin(t) + 0.1*noise ------
    // The t-grid and the truth are fixed; each of K seeds draws a fresh noise
    // vector, and gam + mgcv see identical draws per seed.
    let n = 100usize;
    let period = 2.0 * PI;
    let sigma = 0.1_f64;
    let t: Vec<f64> = (0..n).map(|i| period * i as f64 / n as f64).collect();

    // Dense grid over one period (+ appended seam point for the wrap check).
    let grid_n = 200usize;
    let mut grid_t: Vec<f64> = (0..grid_n)
        .map(|i| period * i as f64 / grid_n as f64)
        .collect();
    grid_t.push(period);
    let truth: Vec<f64> = grid_t[..grid_n].iter().map(|&gt| gt.sin()).collect();

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let formula = format!("h ~ cc(t, k=12, period_start=0, period_end={period:.17})");

    let mut gam_rmses = Vec::with_capacity(K_SPLITS);
    let mut h_cols: Vec<Vec<f64>> = Vec::with_capacity(K_SPLITS);
    let mut h_names: Vec<String> = Vec::with_capacity(K_SPLITS);
    let mut wrap_gap_repr = f64::NAN;
    let mut gam_edf_repr = f64::NAN;

    for s in 0..K_SPLITS {
        let mut rng = StdRng::seed_from_u64(42 + s as u64);
        let noise = Normal::new(0.0, 1.0).expect("normal");
        let h: Vec<f64> = t.iter().map(|&ti| ti.sin() + sigma * noise.sample(&mut rng)).collect();

        let headers: Vec<String> = vec!["t".to_string(), "h".to_string()];
        let rows = t
            .iter()
            .zip(h.iter())
            .map(|(a, b)| csv::StringRecord::from(vec![a.to_string(), b.to_string()]))
            .collect::<Vec<_>>();
        let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode cyclic dataset");
        let t_idx = ds.column_map()["t"];

        let result = fit_from_formula(&formula, &ds, &cfg).expect("gam cyclic fit");
        let FitResult::Standard(fit) = result else {
            panic!("expected a standard GAM fit for a gaussian cyclic smooth");
        };
        let mut design_pts = Array2::<f64>::zeros((grid_t.len(), ds.headers.len()));
        for (i, &gt) in grid_t.iter().enumerate() {
            design_pts[[i, t_idx]] = gt;
        }
        let design = build_term_collection_design(design_pts.view(), &fit.resolvedspec)
            .expect("rebuild design at grid points");
        let gam_fitted: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
        gam_rmses.push(rmse(&gam_fitted[..grid_n], &truth));

        if s == 0 {
            wrap_gap_repr = (gam_fitted[0] - gam_fitted[grid_n]).abs();
            gam_edf_repr = fit.fit.edf_total().expect("gam reports total edf");
        }

        h_cols.push(h);
        h_names.push(format!("h{s}"));
    }

    // ---- mgcv on the SAME K noise draws (t + K response columns) -----------
    let mut columns: Vec<Column> = vec![Column::new("t", &t)];
    for (name, data) in h_names.iter().zip(h_cols.iter()) {
        columns.push(Column::new(name, data));
    }
    let r = run_r(
        &columns,
        &format!(
            r#"
            suppressPackageStartupMessages(library(mgcv))
            K <- {K_SPLITS}
            gridn <- {grid_n}
            gt <- (2 * pi) * (0:(gridn - 1)) / gridn
            truth <- sin(gt)
            rmses <- numeric(K)
            for (s in 0:(K - 1)) {{
              hs <- df[[paste0("h", s)]]
              tr <- data.frame(t = df$t, h = hs)
              m <- gam(h ~ s(t, bs = "cc", k = 12), data = tr, method = "REML",
                       knots = list(t = seq(0, 2 * pi, length = 12)))
              pr <- as.numeric(predict(m, newdata = data.frame(t = gt)))
              rmses[s + 1] <- sqrt(mean((pr - truth)^2))
            }}
            emit("mgcv_rmses", rmses)
            "#
        ),
    );
    let mgcv_rmses = r.vector("mgcv_rmses");
    assert_eq!(mgcv_rmses.len(), K_SPLITS, "mgcv per-seed rmse count mismatch");

    let gam_truth_rmse = mean(&gam_rmses);
    let mgcv_truth_rmse = mean(mgcv_rmses);

    eprintln!(
        "cyclic cc(t) #2395 K={K_SPLITS}-seed avg: n={n} sigma={sigma} gam_edf(split0)={gam_edf_repr:.3} \
         rmse(gam,sin)_avg={gam_truth_rmse:.5} rmse(mgcv,sin)_avg={mgcv_truth_rmse:.5} \
         wrap_gap(split0)={wrap_gap_repr:.3e}"
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "smooths",
            "quality_vs_mgcv_cyclic_cubic",
            "truth_rmse",
            gam_truth_rmse,
            "mgcv",
            mgcv_truth_rmse,
        )
        .line()
    );

    // 1) TRUTH RECOVERY (primary): averaged fitted curve lands close to sin(t).
    assert!(
        gam_truth_rmse <= 0.5 * sigma,
        "gam cyclic fit does not recover sin(t): averaged rmse(gam,sin)={gam_truth_rmse:.5} > {:.5}",
        0.5 * sigma
    );

    // 2) MATCH-OR-BEAT (accuracy): gam at least as accurate as mgcv (10% slack).
    assert!(
        gam_truth_rmse <= mgcv_truth_rmse * 1.10,
        "gam less accurate than mgcv at recovering sin(t): \
         averaged rmse(gam,sin)={gam_truth_rmse:.5} > 1.10*rmse(mgcv,sin)={:.5}",
        mgcv_truth_rmse * 1.10
    );

    // 3) STRUCTURE — periodic seam continuity: fit(0) must equal fit(2π).
    assert!(
        wrap_gap_repr < 1e-6,
        "cyclic wrap not enforced: |fit(0) - fit(2π)| = {wrap_gap_repr:.3e}"
    );
}

/// REAL-DATA arm of the cyclic-cubic capability: the SAME `cc()` periodic smooth,
/// exercised on `nottem` monthly temperatures (no known truth) under #2395 K-split
/// averaging over K random train/test partitions.
///
///   PRIMARY (objective, tool-free): AVERAGED held-out `test_R2 >= 0.85` — the
///     seasonal cycle dominates the variance, so a competent periodic smoother
///     explains the vast majority of held-out variation.
///
///   BASELINE (match-or-beat): mgcv `bs="cc"` fits the SAME training rows and
///     predicts the SAME held-out rows of each partition; gam's AVERAGED held-out
///     RMSE must be no worse than `mgcv_rmse_avg * 1.10`.
///
///   STRUCTURE — periodic seam continuity: gam's fitted curve wraps exactly,
///     `fit(month=1) == fit(month=13)` to 1e-6 (split-invariant, on one fit).
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
    let p = ds.headers.len();

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let formula = "temp ~ cc(month, k=12, period_start=1, period_end=13)";

    let mut gam_rmses = Vec::with_capacity(K_SPLITS);
    let mut gam_r2s = Vec::with_capacity(K_SPLITS);
    let mut fold_data: Vec<Vec<f64>> = Vec::with_capacity(K_SPLITS);
    let mut fold_names: Vec<String> = Vec::with_capacity(K_SPLITS);
    let mut wrap_gap_repr = f64::NAN;
    let mut gam_edf_repr = f64::NAN;

    for split in 0..K_SPLITS {
        let train_rows: Vec<usize> = (0..n).filter(|&i| !is_heldout(i, split)).collect();
        let test_rows: Vec<usize> = (0..n).filter(|&i| is_heldout(i, split)).collect();
        assert!(
            train_rows.len() > 150 && test_rows.len() > 30,
            "#2395 cyclic split {split} degenerate: train={} test={}",
            train_rows.len(),
            test_rows.len()
        );
        let test_month: Vec<f64> = test_rows.iter().map(|&i| month[i]).collect();
        let test_temp: Vec<f64> = test_rows.iter().map(|&i| temp[i]).collect();

        let mut train_values = Array2::<f64>::zeros((train_rows.len(), p));
        for (out_row, &src_row) in train_rows.iter().enumerate() {
            for c in 0..p {
                train_values[[out_row, c]] = ds.values[[src_row, c]];
            }
        }
        let mut train_ds = ds.clone();
        train_ds.values = train_values;

        let result = fit_from_formula(formula, &train_ds, &cfg).expect("gam cyclic fit on nottem");
        let FitResult::Standard(fit) = result else {
            panic!("expected a standard GAM fit for a gaussian cyclic smooth");
        };

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
        gam_rmses.push(rmse(gam_test_pred, &test_temp));
        gam_r2s.push(r2(gam_test_pred, &test_temp));

        if split == 0 {
            wrap_gap_repr = (gam_pred_all[n_test] - gam_pred_all[n_test + 1]).abs();
            gam_edf_repr = fit.fit.edf_total().expect("gam reports total edf");
        }

        fold_data.push(
            (0..n)
                .map(|i| if is_heldout(i, split) { 1.0 } else { 0.0 })
                .collect(),
        );
        fold_names.push(format!("fold{split}"));
    }

    // ---- mgcv on the SAME K partitions (full data.frame + K fold masks) ----
    let mut columns: Vec<Column> = vec![
        Column::new("month", &month),
        Column::new("temp", &temp),
    ];
    for (name, data) in fold_names.iter().zip(fold_data.iter()) {
        columns.push(Column::new(name, data));
    }
    let r = run_r(
        &columns,
        &format!(
            r#"
            suppressPackageStartupMessages(library(mgcv))
            K <- {K_SPLITS}
            rmses <- numeric(K)
            for (s in 0:(K - 1)) {{
              fold <- df[[paste0("fold", s)]]
              tr <- df[fold < 0.5, ]
              te <- df[fold >= 0.5, ]
              m <- gam(temp ~ s(month, bs = "cc", k = 12), data = tr, method = "REML",
                       knots = list(month = seq(1, 13, length = 12)))
              p <- as.numeric(predict(m, newdata = data.frame(month = te$month)))
              rmses[s + 1] <- sqrt(mean((p - te$temp)^2))
            }}
            emit("mgcv_rmses", rmses)
            "#
        ),
    );
    let mgcv_rmses = r.vector("mgcv_rmses");
    assert_eq!(mgcv_rmses.len(), K_SPLITS, "mgcv per-split rmse count mismatch");

    let gam_test_rmse = mean(&gam_rmses);
    let gam_test_r2 = mean(&gam_r2s);
    let mgcv_test_rmse = mean(mgcv_rmses);

    eprintln!(
        "nottem cc(month) #2395 K={K_SPLITS}-split avg: gam_edf(split0)={gam_edf_repr:.3} \
         gam_test_R2_avg={gam_test_r2:.4} gam_test_rmse_avg={gam_test_rmse:.4} \
         mgcv_test_rmse_avg={mgcv_test_rmse:.4} wrap_gap(split0)={wrap_gap_repr:.3e}"
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "smooths",
            "quality_vs_mgcv_cyclic_cubic::test",
            "test_rmse",
            gam_test_rmse,
            "mgcv",
            mgcv_test_rmse,
        )
        .line()
    );

    // ---- PRIMARY: gam predicts the seasonal cycle (averaged held-out R2) ----
    assert!(
        gam_test_r2 >= 0.85,
        "gam's averaged held-out predictive R2 too low: {gam_test_r2:.4} (< 0.85)"
    );

    // ---- BASELINE (match-or-beat): no worse than mgcv on averaged held-out RMSE.
    assert!(
        gam_test_rmse <= mgcv_test_rmse * 1.10,
        "gam averaged held-out RMSE {gam_test_rmse:.4} exceeds mgcv {mgcv_test_rmse:.4} * 1.10"
    );

    // ---- STRUCTURE — periodic seam continuity on gam's OWN fit -------------
    assert!(
        wrap_gap_repr < 1e-6,
        "cyclic wrap not enforced on real data: |fit(month=1) - fit(month=13)| = {wrap_gap_repr:.3e}"
    );
}
