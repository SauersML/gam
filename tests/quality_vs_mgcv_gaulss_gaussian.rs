//! End-to-end quality: gam's Gaussian location-scale fit with a *linear* mean
//! and a *smooth* log-sigma must match `mgcv::gam(family = gaulss())` — the
//! mature, standard reference for Gaussian location-scale (distributional)
//! regression with arbitrary link functions and smooth terms in either
//! parameter — on the canonical heteroscedastic `lidar` benchmark.
//!
//! Why this is the right reference and the right comparison
//! --------------------------------------------------------
//! mgcv's `gaulss()` family models the Gaussian mean `mu` and the *reciprocal*
//! standard deviation through a pair of linear predictors, with the sigma link
//! `eta_sigma = log(1/sigma - b)` (default offset `b = 0.01`), i.e.
//! `1/sigma = b + exp(eta_sigma)`. gam instead places the same `b = 0.01` offset
//! on the *standard-deviation* side: `sigma = LOGB_SIGMA_FLOOR + exp(eta)` with
//! `LOGB_SIGMA_FLOOR = 0.01`. These two links are NOT algebraically identical —
//! mgcv's offset caps `sigma` from *above* (`sigma <= 1/b = 100`) while gam's
//! floors it from *below* (`sigma >= b = 0.01`) — but both are link-scale
//! parameterizations of the *same physical conditional standard deviation*
//! `sigma(range)` estimated by REML on the same data. The honest, convention-free
//! comparison is therefore on `log(sigma)` itself (the response/physical
//! quantity), NOT on the internal `eta`: each engine's `eta` lives in a different
//! coordinate, but `log(sigma)` is the same object. A grid-aligned comparison of
//! the fitted `mu(range)` and `log(sigma)(range)` — not just predictive scores —
//! is the test of whether gam's blockwise solver correctly separates one
//! *linear* (unpenalized) mean block from one *smooth* (penalized) log-sigma
//! block, each with its own penalty.
//!
//! Where the two offsets place the floor matters only where the true conditional
//! `sigma` is comparable to `b = 0.01`. On lidar `sigma(range)` ranges from roughly
//! 0.02 at the low-range end to ~0.4 at the high-range end (sd(logratio) ~ 0.28),
//! so `exp(eta)` dominates the 0.01 offset across nearly the whole grid and the
//! two parameterizations track the same `log(sigma)`; the asymmetric floor only
//! contributes a little slack at the smallest-`sigma` (most negative `log sigma`)
//! end, which `relative_l2` weights heavily. That residual offset asymmetry — not
//! a solver bug — is the expected source of whatever slack the bound below allows.
//!
//! The lidar dataset (`range -> logratio`) is the textbook heteroscedastic
//! smoothing example: the conditional spread of `logratio` grows with `range`,
//! so the smooth sigma model is genuinely exercised. We fit `logratio ~ range`
//! (a straight-line mean) with a smooth `noise_formula = "s(range)"` in gam, and
//! the matching `gaulss` two-formula model in mgcv, then compare on a 90-point
//! grid spanning the observed `range`.
//!
//! Bounds (principled, not loosened)
//! ---------------------------------
//! * Mean: both engines fit the *same* two-parameter line by REML-weighted least
//!   squares (weights = 1/sigma^2). The fitted lines must essentially coincide,
//!   so `relative_l2(mu) < 0.012` is tight yet leaves room for the small weight
//!   differences induced by each engine's smooth-sigma estimate.
//! * log-sigma: both fit a thin-plate smooth log-sigma under REML, each with a
//!   `b = 0.01` offset (gam on `sigma`, mgcv on `1/sigma`). Comparing the
//!   convention-free `log(sigma)` over the grid, `relative_l2 < 0.06` allows for
//!   (a) basis/null-space-convention slack between gam's default thin-plate basis
//!   and mgcv's `bs = "tp"`, and (b) the offset-placement asymmetry at the
//!   small-`sigma` end (gam floors `sigma` at 0.01, mgcv caps `1/sigma` floor),
//!   which can only matter where `sigma ~ 0.01`. The bound is still tight enough
//!   that a genuinely wrong smooth shape, a swapped/duplicated block, or a missing
//!   penalty would blow well past it.
//!
//! A genuine divergence failing either bound is a real signal about gam's
//! two-block separation or penalty application — the bounds are NOT to be
//! relaxed to make it pass.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::solver::estimate::BlockRole;
use gam::test_support::reference::{Column, relative_l2, run_r};
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::Array2;
use std::path::Path;

const LIDAR_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/lidar.csv");

/// gam's sigma link offset (`sigma = LOGB_SIGMA_FLOOR + exp(eta)`). Numerically
/// equal to mgcv `gaulss()`'s default `b = 0.01`, though mgcv places that offset
/// on `1/sigma` rather than `sigma`; we compare the convention-free `log(sigma)`,
/// so the offsets coincide except at the smallest-`sigma` end of the grid.
const LOGB_SIGMA_FLOOR: f64 = 0.01;

#[test]
fn gam_gaulss_linear_mean_smooth_sigma_matches_mgcv_on_lidar() {
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

    // ---- fit with gam: linear mean + smooth log-sigma, REML ---------------
    // mean predictor: logratio ~ range  (a straight line, identity link)
    // scale predictor: s(range)         (thin-plate smooth on log-sigma)
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        noise_formula: Some("s(range)".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("logratio ~ range", &ds, &cfg).expect("gam gaulss fit");
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

    // ---- evaluation grid: 90 points spanning [range.min, range.max] -------
    let rmin = range.iter().copied().fold(f64::INFINITY, f64::min);
    let rmax = range.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let grid_n = 90usize;
    let grid_range: Vec<f64> = (0..grid_n)
        .map(|i| rmin + (rmax - rmin) * (i as f64) / ((grid_n - 1) as f64))
        .collect();

    // Rebuild each block's design at the grid from the FROZEN resolved specs
    // (same knots/centering as the fit) and apply the block coefficients.
    let mut grid = Array2::<f64>::zeros((grid_n, ds.headers.len()));
    for (i, &r) in grid_range.iter().enumerate() {
        grid[[i, range_idx]] = r;
    }
    let mean_design = build_term_collection_design(grid.view(), &fit.fit.meanspec_resolved)
        .expect("rebuild mean design at grid");
    let noise_design = build_term_collection_design(grid.view(), &fit.fit.noisespec_resolved)
        .expect("rebuild noise design at grid");
    assert_eq!(
        mean_design.design.ncols(),
        beta_mean.len(),
        "grid mean design columns ({}) must match mean coefficient count ({})",
        mean_design.design.ncols(),
        beta_mean.len()
    );
    assert_eq!(
        noise_design.design.ncols(),
        beta_scale.len(),
        "grid noise design columns ({}) must match scale coefficient count ({})",
        noise_design.design.ncols(),
        beta_scale.len()
    );

    // Mean is identity-link: response-scale mu = X_mean * beta_mean.
    let gam_mu: Vec<f64> = mean_design.design.apply(&beta_mean).to_vec();
    // log-sigma link: sigma = LOGB_SIGMA_FLOOR + exp(eta_scale); compare log(sigma).
    let eta_scale = noise_design.design.apply(&beta_scale);
    let gam_log_sigma: Vec<f64> = eta_scale
        .iter()
        .map(|&e| (LOGB_SIGMA_FLOOR + e.exp()).ln())
        .collect();

    // ---- fit the SAME model with mgcv gaulss (the mature reference) -------
    // gaulss(): mu formula linear, sigma formula smooth s(range, bs="tp").
    // predict(type="response") returns a matrix whose first column is mu and
    // whose second column is 1/sigma (gaulss models the reciprocal sd), so
    // sigma = 1 / col2 and log(sigma) = -log(col2). mgcv's default offset b = 0.01
    // shares gam's numeric value but sits on 1/sigma rather than sigma; we compare
    // the convention-free log(sigma), so the placement difference shows up only as
    // small-sigma slack, not a coordinate mismatch.
    let r = run_r(
        &[
            Column::new("range", &range),
            Column::new("logratio", &logratio),
            Column::new("grid_range", &grid_range_padded(&grid_range, n)),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        m <- gam(list(logratio ~ range, ~ s(range, bs = "tp")),
                 family = gaulss(), data = df, method = "REML")
        g <- df$grid_range[1:90]
        nd <- data.frame(range = g)
        pr <- predict(m, newdata = nd, type = "response")
        mu <- as.numeric(pr[, 1])
        inv_sigma <- as.numeric(pr[, 2])
        emit("mu", mu)
        emit("log_sigma", -log(inv_sigma))
        "#,
    );
    let mgcv_mu = r.vector("mu");
    let mgcv_log_sigma = r.vector("log_sigma");
    assert_eq!(mgcv_mu.len(), grid_n, "mgcv mu length mismatch");
    assert_eq!(
        mgcv_log_sigma.len(),
        grid_n,
        "mgcv log_sigma length mismatch"
    );

    // ---- compare on the grid ----------------------------------------------
    let mu_rel = relative_l2(&gam_mu, mgcv_mu);
    let log_sigma_rel = relative_l2(&gam_log_sigma, mgcv_log_sigma);

    eprintln!(
        "lidar gaulss (linear mean + smooth log-sigma): n={n} grid={grid_n} \
         mean_beta={:?} scale_dim={} mu_rel_l2={mu_rel:.5} log_sigma_rel_l2={log_sigma_rel:.5}",
        beta_mean.to_vec(),
        beta_scale.len()
    );

    // Both engines fit the SAME straight-line mean by REML-weighted least
    // squares; the fitted lines must essentially coincide.
    assert!(
        mu_rel < 0.012,
        "linear mean diverges from mgcv gaulss: rel_l2(mu)={mu_rel:.5} (bound 0.012)"
    );
    // Both fit a thin-plate smooth log-sigma under REML; the two engines place
    // the b = 0.01 offset on opposite sides (gam on sigma, mgcv on 1/sigma), so on
    // the convention-free log(sigma) we allow basis-convention slack plus the
    // small-sigma offset asymmetry, but nothing close to a wrong-shape failure.
    assert!(
        log_sigma_rel < 0.06,
        "smooth log-sigma diverges from mgcv gaulss: rel_l2(log sigma)={log_sigma_rel:.5} \
         (bound 0.06)"
    );
}

/// The reference harness requires every column to have the same row count as the
/// data (`n` lidar rows), but the evaluation grid only has 90 points. Pad the
/// grid out to length `n` by repeating its last value; the R body slices back to
/// the first 90 entries (`df$grid_range[1:90]`), so the padding is never read.
fn grid_range_padded(grid_range: &[f64], n: usize) -> Vec<f64> {
    let mut v = grid_range.to_vec();
    let last = *grid_range.last().expect("non-empty grid");
    while v.len() < n {
        v.push(last);
    }
    v.truncate(n);
    v
}
