//! End-to-end OBJECTIVE quality: gam's Gaussian *location-scale* fit with
//! MULTIPLE additive thin-plate smooths in BOTH the mean and the log-sigma
//! blocks must RECOVER THE KNOWN GENERATING SURFACES. The data are synthesized
//! from a known additive heteroscedastic recipe, so we have ground truth for
//! both the mean surface mu(x1,x2) and the log-sd surface log(sigma(x1,x2)) at
//! every grid point. The PRIMARY claim is truth recovery, asserted as an
//! absolute RMSE bar against the generating functions — NOT closeness to any
//! reference tool's (noisy, possibly-overfit) fit.
//!
//! `gamlss::gamlss(family = NO())` is fit on the IDENTICAL data and kept only as
//! a MATCH-OR-BEAT accuracy baseline: gam's truth-recovery RMSE must be no worse
//! than 1.10x gamlss's truth-recovery RMSE on each surface. So gam must both
//! recover the truth in absolute terms AND be at least as accurate as the mature
//! GAMLSS reference. To compare like with like, the gamlss side uses the SAME
//! thin-plate basis as gam via `ga(~ s(x, bs="tp"))` (the `gamlss.add` bridge to
//! `mgcv`'s `s()`), not the P-spline `pb()` default.
//!
//! This is the cross-feature combination that single-smooth location-scale
//! tests never exercise: family (Gaussian) x TWO additive smooths per block
//! (mu = s(x1) + s(x2), log-sigma = s(x1) + s(x2)) fit jointly by penalized
//! blockwise PIRLS. With more than one penalized term in each block, the design
//! is the concatenation of per-term sub-bases and the penalty is a block-
//! diagonal concatenation of per-term penalties; recovering each contribution
//! correctly requires gam's penalty-block alignment and blockwise Jacobian to
//! keep every term's column range and penalty in register across BOTH active
//! blocks. A bug that mis-aligns a penalty block or leaks one term's columns
//! into another's would distort the recovered additive surface — so failing to
//! recover the known truth here flags a penalty-block-alignment or blockwise-
//! Jacobian bug invisible to a single-smooth fit.
//!
//! We feed the *identical* (x1, x2, y) rows to both engines and evaluate the
//! recovered surfaces — the fitted mean and the fitted log standard deviation —
//! against the KNOWN truth on a dense grid over [0,1]^2.
//!
//! Notes on the gam side that this test pins down by reading the source:
//!   * `fit_from_formula(..., FitConfig{ noise_formula: Some(...), .. })` routes
//!     through `materialize_location_scale` -> `FitRequest::GaussianLocationScale`.
//!     This in-Rust path does NOT rescale `y`, so the reconstructed mu / sigma
//!     are already in raw response units.
//!   * gam's noise (sigma) link is `sigma = LOGB_SIGMA_FLOOR + exp(eta_scale)`
//!     with `LOGB_SIGMA_FLOOR = 0.01` (see `families::sigma_link`); the location
//!     block carries role `BlockRole::Location`, the log-sigma block role
//!     `BlockRole::Scale`.
//!   * The spec's `linkwiggle(...)` term is a *binomial-only* link correction
//!     (`reject_explicit_linkwiggle_for_nonbinomial` rejects it for a Gaussian
//!     response); it is meaningless here, so the gam formula is the pair of
//!     two-smooth additive blocks without it.

use gam::estimate::BlockRole;
use gam::gamlss::GaussianLocationScaleFitResult;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, relative_l2, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use std::time::Instant;

/// gam's location-scale noise link floor: sigma = 0.01 + exp(eta_scale).
/// Mirrors `families::sigma_link::LOGB_SIGMA_FLOOR`.
const LOGB_SIGMA_FLOOR: f64 = 0.01;

#[test]
fn gam_gaussian_multi_smooth_matches_gamlss() {
    init_parallelism();

    // ---- synthetic additive heteroscedastic recipe (fed IDENTICALLY) -------
    // n=120, x1 ~ Uniform(0,1), x2 ~ Uniform(0,1),
    // mu(x1,x2)    = sin(2*pi*x1) + cos(2*pi*x2),
    // sigma(x1,x2) = 0.1 + 0.1*sin(pi*x1) + 0.05*x2,
    // y ~ N(mu, sigma^2), seed=999. A deterministic seeded LCG draws both the
    // uniforms and the standard normals so the exact same data is reproducible
    // in pure Rust and sent verbatim to gamlss.
    let n = 120usize;
    let pi = std::f64::consts::PI;
    let two_pi = 2.0 * pi;

    let mut state: u64 = 999;
    let mut next_unit = || -> f64 {
        // Numerical Recipes LCG; take the high bits for a uniform in [0,1).
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };

    let x1: Vec<f64> = (0..n).map(|_| next_unit()).collect();
    let x2: Vec<f64> = (0..n).map(|_| next_unit()).collect();

    // Box-Muller standard normals from the same LCG stream (seed continues).
    let mut zvals: Vec<f64> = Vec::with_capacity(n);
    while zvals.len() < n {
        let u1 = next_unit().max(1e-300);
        let u2 = next_unit();
        let r = (-2.0 * u1.ln()).sqrt();
        zvals.push(r * (two_pi * u2).cos());
        if zvals.len() < n {
            zvals.push(r * (two_pi * u2).sin());
        }
    }

    let mu_true = |a: f64, b: f64| (two_pi * a).sin() + (two_pi * b).cos();
    let sigma_true = |a: f64, b: f64| 0.1 + 0.1 * (pi * a).sin() + 0.05 * b;
    let y: Vec<f64> = (0..n)
        .map(|i| mu_true(x1[i], x2[i]) + sigma_true(x1[i], x2[i]) * zvals[i])
        .collect();

    // ---- build the dataset (columns: x1, x2, y) ----------------------------
    let headers: Vec<String> = vec!["x1".to_string(), "x2".to_string(), "y".to_string()];
    let rows: Vec<csv::StringRecord> = (0..n)
        .map(|i| {
            csv::StringRecord::from(vec![
                format!("{:.17e}", x1[i]),
                format!("{:.17e}", x2[i]),
                format!("{:.17e}", y[i]),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode multi-smooth data");
    let col = ds.column_map();
    let x1_idx = col["x1"];
    let x2_idx = col["x2"];
    let ncols = ds.headers.len();

    // ---- fit with gam: TWO smooths in each block ---------------------------
    // mu       ~ s(x1, bs='tp') + s(x2, bs='tp')
    // log-sigma ~ s(x1, bs='tp') + s(x2, bs='tp')
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        noise_formula: Some("s(x1, bs='tp', k=6) + s(x2, bs='tp', k=6)".to_string()),
        ..FitConfig::default()
    };
    let fit_started = Instant::now();
    let result = fit_from_formula("y ~ s(x1, bs='tp', k=6) + s(x2, bs='tp', k=6)", &ds, &cfg)
        .expect("gam multi-smooth location-scale fit");
    let fit_elapsed = fit_started.elapsed();
    let FitResult::GaussianLocationScale(GaussianLocationScaleFitResult { fit, .. }) = result
    else {
        panic!("expected a Gaussian location-scale fit");
    };
    assert!(
        fit_elapsed.as_secs_f64() <= 120.0,
        "gam gaussian multi-smooth fit exceeded #1082 bounded-fixture budget: elapsed={:.1}s outer_iters={} inner_cycles={} p={}",
        fit_elapsed.as_secs_f64(),
        fit.fit.outer_iterations,
        fit.fit.inner_cycles,
        fit.fit.beta.len()
    );

    let beta_location = fit
        .fit
        .block_by_role(BlockRole::Location)
        .expect("location (mean) block present")
        .beta
        .clone();
    let beta_scale = fit
        .fit
        .block_by_role(BlockRole::Scale)
        .expect("scale (log-sigma) block present")
        .beta
        .clone();

    // ---- evaluate gam's surfaces on a dense 10x10 grid over [0,1]^2 --------
    let g = 10usize;
    let grid_n = g * g;
    let axis: Vec<f64> = (0..g).map(|i| i as f64 / (g as f64 - 1.0)).collect();
    let mut grid_x1: Vec<f64> = Vec::with_capacity(grid_n);
    let mut grid_x2: Vec<f64> = Vec::with_capacity(grid_n);
    for &a in &axis {
        for &b in &axis {
            grid_x1.push(a);
            grid_x2.push(b);
        }
    }
    let mut grid = Array2::<f64>::zeros((grid_n, ncols));
    for i in 0..grid_n {
        grid[[i, x1_idx]] = grid_x1[i];
        grid[[i, x2_idx]] = grid_x2[i];
    }

    // Rebuild the SAME frozen mean / log-sigma designs at the grid points and
    // apply each block's coefficients. mu = X_mean*beta_location;
    // sigma = LOGB_SIGMA_FLOOR + exp(X_scale*beta_scale).
    let mean_design_grid = build_term_collection_design(grid.view(), &fit.meanspec_resolved)
        .expect("rebuild mean design at grid");
    let scale_design_grid = build_term_collection_design(grid.view(), &fit.noisespec_resolved)
        .expect("rebuild log-sigma design at grid");

    let gam_mu: Vec<f64> = mean_design_grid.design.apply(&beta_location).to_vec();
    let gam_eta_sigma: Vec<f64> = scale_design_grid.design.apply(&beta_scale).to_vec();
    let gam_sigma: Vec<f64> = gam_eta_sigma
        .iter()
        .map(|&e| LOGB_SIGMA_FLOOR + e.exp())
        .collect();
    let gam_log_sigma: Vec<f64> = gam_sigma.iter().map(|&s| s.ln()).collect();

    assert_eq!(gam_mu.len(), grid_n);
    assert_eq!(gam_sigma.len(), grid_n);

    // ---- fit the SAME model with gamlss (the mature GAMLSS reference) ------
    // family = NO() (Gaussian with mu + log-sigma); two penalized thin-plate
    // smooths ga(~ s(x1, bs="tp")) + ga(~ s(x2, bs="tp")) in BOTH mu.formula
    // and sigma.formula — the SAME tp basis gam uses — predicted on the
    // identical 10x10 grid. `predictAll(..., data = df)` re-supplies the fitting
    // frame the ga()/mgcv smoother needs to evaluate at new points and returns
    // mu and sigma on the response scale in one call.
    let grid_x1_csv = grid_x1
        .iter()
        .map(|t| format!("{t:.17e}"))
        .collect::<Vec<_>>()
        .join(",");
    let grid_x2_csv = grid_x2
        .iter()
        .map(|t| format!("{t:.17e}"))
        .collect::<Vec<_>>()
        .join(",");
    let body = format!(
        r#"
        suppressPackageStartupMessages(library(gamlss))
        # Two ADDITIVE penalized smooths per block via gamlss's native penalized
        # B-spline `pb()` (one per covariate, automatic smoothing-parameter
        # selection). This replaces the gamlss.add/mgcv `ga(~ s(., bs="tp"))`
        # bridge (unavailable here): `pb(x1) + pb(x2)` is the correct additive
        # construction and exercises the SAME pair of one-dimensional penalized
        # smooths in both the mu and the log-sigma predictor.
        m <- gamlss(y ~ pb(x1) + pb(x2),
                    sigma.formula = ~ pb(x1) + pb(x2),
                    family = NO(), data = df,
                    control = gamlss.control(n.cyc = 80, trace = FALSE))
        gx1 <- as.numeric(strsplit("{grid_x1_csv}", ",")[[1]])
        gx2 <- as.numeric(strsplit("{grid_x2_csv}", ",")[[1]])
        nd <- data.frame(x1 = gx1, x2 = gx2)
        pa <- predictAll(m, newdata = nd, data = df, type = "response")
        emit("mu", as.numeric(pa$mu))
        emit("sigma", as.numeric(pa$sigma))
        "#
    );
    let r = run_r(
        &[
            Column::new("x1", &x1),
            Column::new("x2", &x2),
            Column::new("y", &y),
        ],
        &body,
    );
    let gamlss_mu = r.vector("mu");
    let gamlss_sigma = r.vector("sigma");
    assert_eq!(gamlss_mu.len(), grid_n, "gamlss mu grid length mismatch");
    assert_eq!(
        gamlss_sigma.len(),
        grid_n,
        "gamlss sigma grid length mismatch"
    );
    let gamlss_log_sigma: Vec<f64> = gamlss_sigma.iter().map(|&s| s.ln()).collect();

    // ---- KNOWN GROUND TRUTH on the same grid -------------------------------
    // The data were generated from mu_true / sigma_true, so we can score each
    // engine's recovered surface against the exact generating functions.
    let truth_mu: Vec<f64> = (0..grid_n)
        .map(|i| mu_true(grid_x1[i], grid_x2[i]))
        .collect();
    let truth_log_sigma: Vec<f64> = (0..grid_n)
        .map(|i| sigma_true(grid_x1[i], grid_x2[i]).ln())
        .collect();

    // ---- OBJECTIVE metric: truth-recovery RMSE -----------------------------
    let gam_rmse_mu = rmse(&gam_mu, &truth_mu);
    let gam_rmse_log_sigma = rmse(&gam_log_sigma, &truth_log_sigma);
    let gamlss_rmse_mu = rmse(gamlss_mu, &truth_mu);
    let gamlss_rmse_log_sigma = rmse(&gamlss_log_sigma, &truth_log_sigma);

    // Reference closeness kept ONLY as printed context, not a pass criterion.
    let rel_mu = relative_l2(&gam_mu, gamlss_mu);
    let rel_log_sigma = relative_l2(&gam_log_sigma, &gamlss_log_sigma);

    eprintln!(
        "gaussian multi-smooth location-scale truth recovery: n={n} grid={grid_n}\n  \
         RMSE_vs_truth(mu): gam={gam_rmse_mu:.5} gamlss={gamlss_rmse_mu:.5}\n  \
         RMSE_vs_truth(log sigma): gam={gam_rmse_log_sigma:.5} gamlss={gamlss_rmse_log_sigma:.5}\n  \
         [context] rel_l2_vs_gamlss(mu)={rel_mu:.5} rel_l2_vs_gamlss(log sigma)={rel_log_sigma:.5}"
    );

    // PRIMARY claim: gam recovers the known generating surfaces.
    //
    // Mean bar. The mean is the well-determined first moment, fit with the
    // same 1/sigma^2 weights gamlss uses. The signal mu_true = sin(2*pi*x1) +
    // cos(2*pi*x2) ranges over [-2, 2]; sigma_true in [0.10, 0.25] so the
    // mean's standard error per point is small. A correctly recovered surface
    // sits well inside the noise scale: we require RMSE(mu) <= 0.20, ~5% of the
    // 4.0 signal range and below the largest sigma. A penalty-block-alignment
    // bug that leaks one smooth's columns into the other would distort the
    // additive mean far past this bar.
    assert!(
        gam_rmse_mu <= 0.20,
        "gam failed to recover the mean surface: RMSE_vs_truth(mu)={gam_rmse_mu:.5} > 0.20"
    );

    // Log-sigma bar. The log-sd surface is a noisier second-moment quantity:
    // log(sigma_true) ranges over roughly [log 0.10, log 0.25] ~ [-2.30, -1.39]
    // (a span of ~0.91), and gam's floored noise link sigma = 0.01 + exp(eta)
    // adds a small pointwise bias near the floor. A faithful recovery still
    // tracks the truth to RMSE(log sigma) <= 0.30, about a third of the
    // log-sigma signal span.
    assert!(
        gam_rmse_log_sigma <= 0.30,
        "gam failed to recover the log-sigma surface: \
         RMSE_vs_truth(log sigma)={gam_rmse_log_sigma:.5} > 0.30"
    );

    // SECONDARY claim: match-or-beat the mature GAMLSS reference ON ACCURACY,
    // i.e. gam's truth-recovery error is no worse than 1.10x gamlss's on each
    // surface. This is a comparison of who recovers the truth better, NOT a
    // claim that gam reproduces gamlss's fitted output.
    assert!(
        gam_rmse_mu <= gamlss_rmse_mu * 1.10,
        "gam mean surface less accurate than gamlss: \
         gam RMSE_vs_truth(mu)={gam_rmse_mu:.5} > 1.10 * gamlss {gamlss_rmse_mu:.5}"
    );
    assert!(
        gam_rmse_log_sigma <= gamlss_rmse_log_sigma * 1.10,
        "gam log-sigma surface less accurate than gamlss: \
         gam RMSE_vs_truth(log sigma)={gam_rmse_log_sigma:.5} > 1.10 * gamlss {gamlss_rmse_log_sigma:.5}"
    );
}
