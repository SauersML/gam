//! End-to-end OBJECTIVE quality: gam's Gaussian location-scale fit with a
//! *linear* mean and a *smooth* log-sigma is judged by its HELD-OUT predictive
//! quality on the canonical heteroscedastic `lidar` benchmark — not by whether
//! it reproduces another tool's fitted curve.
//!
//! Objective metric asserted
//! --------------------------
//! The dataset (`range -> logratio`) has no known generating truth, so the
//! honest quality question is *predictive*: held out a deterministic 20% of the
//! rows (`row_index % 5 == 0`), fit gam on the remaining 80%, predict the test
//! rows, and score the predictions with the model's OWN proper scoring rule —
//! the Gaussian location-scale negative log-likelihood
//!
//!     nll_i = 0.5*log(2*pi) + log(sigma_i) + 0.5*((y_i - mu_i)/sigma_i)^2
//!
//! averaged over the held-out rows. This is the natural objective for a
//! *distributional* model: it rewards a well-placed mean AND a well-calibrated
//! conditional spread simultaneously (a too-narrow sigma where the data are
//! noisy, or a too-wide sigma where they are tight, is penalized). A model that
//! merely fits the mean while getting the heteroscedastic scale wrong scores
//! poorly here, which is exactly the failure mode this test must catch.
//!
//! Two assertions, both objective:
//!   1. ABSOLUTE: held-out mean fit explains real signal — test R^2 of `mu`
//!      against `logratio` is >= 0.55. (lidar's mean is a strong, nearly
//!      monotone trend; an R^2 this high cannot be reached by a degenerate or
//!      mis-separated mean block, but is comfortably below what a correct linear
//!      mean achieves, so it is a floor, not a ceiling.)
//!   2. MATCH-OR-BEAT (baseline): gam's held-out mean per-point NLL is no worse
//!      than `mgcv::gam(family = gaulss())`'s held-out NLL by more than a small
//!      additive margin (0.05 nats/point). mgcv's gaulss is the mature reference
//!      for Gaussian location-scale regression; here it is demoted from "gam
//!      must reproduce its curve" to "gam must predict at least as well as it on
//!      genuinely held-out data". Beating it is allowed and welcome.
//!
//! Both engines are fit on the IDENTICAL training rows and scored on the
//! IDENTICAL test rows. mgcv `gaulss()` models the *reciprocal* sd through
//! `1/sigma = b + exp(eta_sigma)` (default `b = 0.01`); gam floors `sigma` via
//! `sigma = LOGB_SIGMA_FLOOR + exp(eta)` with `LOGB_SIGMA_FLOOR = 0.01`. The
//! parameterizations differ, but the NLL is computed in the convention-free
//! physical `(mu, sigma)` coordinates for both, so the comparison is fair.
//!
//! The reference is retained purely as a BASELINE TO MATCH-OR-BEAT on held-out
//! NLL. The PRIMARY claim is gam's own held-out predictive quality. Neither
//! bound is to be relaxed to force a pass — a genuine held-out shortfall failing
//! is a real signal about gam's two-block separation or penalty application.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::solver::estimate::BlockRole;
use gam::test_support::reference::{Column, relative_l2, run_r};
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::{Array2, s};
use std::path::Path;

const LIDAR_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/lidar.csv");

/// gam's sigma link offset (`sigma = LOGB_SIGMA_FLOOR + exp(eta)`). Numerically
/// equal to mgcv `gaulss()`'s default `b = 0.01` (mgcv places it on `1/sigma`).
const LOGB_SIGMA_FLOOR: f64 = 0.01;

/// Every 5th row (0-based) is held out for testing; the rest train. Deterministic.
const TEST_STRIDE: usize = 5;

/// Mean per-point Gaussian negative log-likelihood of held-out `(y, mu, sigma)`.
/// This is the model's own proper scoring rule for a Gaussian location-scale
/// fit; lower is better. Computed in physical `(mu, sigma)` coordinates so it is
/// convention-free across the two engines' differing sigma links.
fn gaussian_nll(y: &[f64], mu: &[f64], sigma: &[f64]) -> f64 {
    assert_eq!(y.len(), mu.len(), "nll: y/mu length mismatch");
    assert_eq!(y.len(), sigma.len(), "nll: y/sigma length mismatch");
    let half_log_2pi = 0.5 * (2.0 * std::f64::consts::PI).ln();
    let n = y.len() as f64;
    let total: f64 = y
        .iter()
        .zip(mu)
        .zip(sigma)
        .map(|((&yi, &mui), &si)| {
            let z = (yi - mui) / si;
            half_log_2pi + si.ln() + 0.5 * z * z
        })
        .sum();
    total / n
}

/// Coefficient of determination of `pred` against `truth`: `1 - SSE/SST`.
fn r_squared(truth: &[f64], pred: &[f64]) -> f64 {
    assert_eq!(truth.len(), pred.len(), "r2: length mismatch");
    let n = truth.len() as f64;
    let mean = truth.iter().sum::<f64>() / n;
    let sse: f64 = truth
        .iter()
        .zip(pred)
        .map(|(&t, &p)| (t - p) * (t - p))
        .sum();
    let sst: f64 = truth.iter().map(|&t| (t - mean) * (t - mean)).sum();
    1.0 - sse / sst.max(1e-300)
}

#[test]
fn gam_gaulss_linear_mean_smooth_sigma_predicts_lidar_at_least_as_well_as_mgcv() {
    init_parallelism();

    // ---- load the canonical lidar dataset (range -> logratio) -------------
    let ds = load_csvwith_inferred_schema(Path::new(LIDAR_CSV)).expect("load lidar.csv");
    let col = ds.column_map();
    let range_idx = col["range"];
    let logratio_idx = col["logratio"];
    let range_all: Vec<f64> = ds.values.column(range_idx).to_vec();
    let logratio_all: Vec<f64> = ds.values.column(logratio_idx).to_vec();
    let n = range_all.len();
    assert!(n > 100, "lidar should have ~221 rows, got {n}");

    // ---- deterministic 80/20 train/test split (every 5th row -> test) -----
    let train_rows: Vec<usize> = (0..n).filter(|i| i % TEST_STRIDE != 0).collect();
    let test_rows: Vec<usize> = (0..n).filter(|i| i % TEST_STRIDE == 0).collect();
    assert!(
        test_rows.len() > 20,
        "expected a sizeable held-out set, got {}",
        test_rows.len()
    );

    // Build the training dataset by selecting the training rows; schema and
    // column kinds are row-count-independent metadata, so only `values` changes.
    let mut train_ds = ds.clone();
    let mut train_values = Array2::<f64>::zeros((train_rows.len(), ds.values.ncols()));
    for (out_i, &row) in train_rows.iter().enumerate() {
        train_values
            .slice_mut(s![out_i, ..])
            .assign(&ds.values.slice(s![row, ..]));
    }
    train_ds.values = train_values;

    let range_train: Vec<f64> = train_rows.iter().map(|&i| range_all[i]).collect();
    let logratio_train: Vec<f64> = train_rows.iter().map(|&i| logratio_all[i]).collect();
    let range_test: Vec<f64> = test_rows.iter().map(|&i| range_all[i]).collect();
    let logratio_test: Vec<f64> = test_rows.iter().map(|&i| logratio_all[i]).collect();

    // ---- fit gam on TRAIN: linear mean + smooth log-sigma, REML -----------
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        noise_formula: Some("s(range)".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("logratio ~ range", &train_ds, &cfg).expect("gam gaulss fit");
    let FitResult::GaussianLocationScale(fit) = result else {
        panic!("expected a GaussianLocationScale fit result for a Gaussian noise_formula model");
    };

    // The mean (Location) block must be exactly the two-coefficient line
    // (intercept + range); the scale block must carry a multi-column smooth.
    let mean_block = fit
        .fit
        .fit
        .block_by_role(BlockRole::Location)
        .expect("Gaussian location-scale fit must carry a Location (mean) block");
    let beta_mean = mean_block.beta.clone();
    assert_eq!(
        beta_mean.len(),
        2,
        "linear mean `logratio ~ range` must materialize exactly 2 coefficients \
         (intercept + slope), got {}",
        beta_mean.len()
    );
    let scale_block = fit
        .fit
        .fit
        .block_by_role(BlockRole::Scale)
        .expect("smooth noise_formula must fit a retrievable Scale (log-sigma) block");
    let beta_scale = scale_block.beta.clone();
    assert!(
        beta_scale.len() >= 2,
        "smooth `noise_formula=\"s(range)\"` must materialize a multi-coefficient scale \
         basis, got {} coefficient(s)",
        beta_scale.len()
    );

    // ---- predict gam at the held-out TEST rows ----------------------------
    // Rebuild each block's design at the test inputs from the FROZEN resolved
    // specs (same knots/centering as the fit) and apply the block coefficients.
    let mut test_grid = Array2::<f64>::zeros((test_rows.len(), ds.headers.len()));
    for (i, &r) in range_test.iter().enumerate() {
        test_grid[[i, range_idx]] = r;
    }
    let mean_design = build_term_collection_design(test_grid.view(), &fit.fit.meanspec_resolved)
        .expect("rebuild mean design at test rows");
    let noise_design = build_term_collection_design(test_grid.view(), &fit.fit.noisespec_resolved)
        .expect("rebuild noise design at test rows");
    assert_eq!(
        mean_design.design.ncols(),
        beta_mean.len(),
        "test mean design columns ({}) must match mean coefficient count ({})",
        mean_design.design.ncols(),
        beta_mean.len()
    );
    assert_eq!(
        noise_design.design.ncols(),
        beta_scale.len(),
        "test noise design columns ({}) must match scale coefficient count ({})",
        noise_design.design.ncols(),
        beta_scale.len()
    );

    // Mean is identity-link: response-scale mu = X_mean * beta_mean.
    let gam_mu: Vec<f64> = mean_design.design.apply(&beta_mean).to_vec();
    // log-sigma link: sigma = LOGB_SIGMA_FLOOR + exp(eta_scale).
    let eta_scale = noise_design.design.apply(&beta_scale);
    let gam_sigma: Vec<f64> = eta_scale
        .iter()
        .map(|&e| LOGB_SIGMA_FLOOR + e.exp())
        .collect();

    // ---- gam held-out objective scores ------------------------------------
    let gam_nll = gaussian_nll(&logratio_test, &gam_mu, &gam_sigma);
    let gam_r2 = r_squared(&logratio_test, &gam_mu);

    // ---- fit the SAME model with mgcv gaulss on the SAME train rows -------
    // gaulss(): mu formula linear, sigma formula smooth s(range, bs="tp").
    // predict(type="response") returns [mu, 1/sigma] at the test rows; we score
    // the held-out NLL in physical (mu, sigma) coordinates for a fair comparison.
    let r = run_r(
        &[
            Column::new("range", &range_train),
            Column::new("logratio", &logratio_train),
            Column::new("range_test", &pad_to(&range_test, range_train.len())),
            Column::new("logratio_test", &pad_to(&logratio_test, range_train.len())),
        ],
        &format!(
            r#"
            suppressPackageStartupMessages(library(mgcv))
            ntest <- {ntest}
            m <- gam(list(logratio ~ range, ~ s(range, bs = "tp")),
                     family = gaulss(), data = df, method = "REML")
            rt <- df$range_test[1:ntest]
            yt <- df$logratio_test[1:ntest]
            nd <- data.frame(range = rt)
            pr <- predict(m, newdata = nd, type = "response")
            mu <- as.numeric(pr[, 1])
            inv_sigma <- as.numeric(pr[, 2])
            sigma <- 1 / inv_sigma
            z <- (yt - mu) / sigma
            nll <- 0.5 * log(2 * pi) + log(sigma) + 0.5 * z * z
            emit("nll", mean(nll))
            emit("mu", mu)
            "#,
            ntest = test_rows.len()
        ),
    );
    let mgcv_nll = r.scalar("nll");
    let mgcv_mu = r.vector("mu");
    assert_eq!(mgcv_mu.len(), test_rows.len(), "mgcv mu length mismatch");

    // For context only (NOT a pass criterion): how close gam's held-out mean
    // tracks mgcv's. Closeness to the reference is deliberately not asserted.
    let mu_rel_to_mgcv = relative_l2(&gam_mu, mgcv_mu);

    eprintln!(
        "lidar gaulss held-out (n_train={} n_test={}): gam_R2={gam_r2:.4} \
         gam_NLL={gam_nll:.4} mgcv_NLL={mgcv_nll:.4} (gam-mgcv={:.4}) \
         mu_rel_l2_vs_mgcv={mu_rel_to_mgcv:.4}",
        train_rows.len(),
        test_rows.len(),
        gam_nll - mgcv_nll
    );

    // ---- OBJECTIVE assertion 1: gam recovers real held-out signal ---------
    assert!(
        gam_r2 >= 0.55,
        "gam's held-out mean explains too little of lidar's signal: \
         test R^2={gam_r2:.4} (floor 0.55)"
    );

    // ---- OBJECTIVE assertion 2: match-or-beat mgcv on held-out NLL --------
    // The distributional proper scoring rule rewards a good mean AND a good
    // conditional scale. gam must predict at least as well as the mature
    // reference on genuinely unseen data (small additive slack for solver/basis
    // convention differences); beating it is welcome.
    assert!(
        gam_nll <= mgcv_nll + 0.05,
        "gam's held-out Gaussian location-scale NLL is worse than mgcv gaulss: \
         gam_NLL={gam_nll:.4} > mgcv_NLL={mgcv_nll:.4} + 0.05"
    );
}

/// The reference harness requires every column to share the data's row count.
/// The held-out test vectors are shorter than the training frame, so pad them
/// to the training length by repeating the last value; the R body slices back
/// to `1:ntest`, so the padding is never read.
fn pad_to(v: &[f64], n: usize) -> Vec<f64> {
    let mut out = v.to_vec();
    let last = *v.last().expect("non-empty test vector");
    while out.len() < n {
        out.push(last);
    }
    out.truncate(n);
    out
}
