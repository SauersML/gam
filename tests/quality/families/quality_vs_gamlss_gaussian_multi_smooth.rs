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
//! than 1.10x gamlss's truth-recovery RMSE on each surface, and not RESOLVED
//! worse draw by draw. So gam must both recover the truth in absolute terms AND
//! be at least as accurate as the mature GAMLSS reference.
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
//! PAIRED over draws, not one draw (#2395). Either engine's RMSE against the
//! truth depends on which sample of (x1, x2) and noise it fit, so one draw's ratio
//! conflates the draw with the engine. The single draw this test used read mu
//! gam 0.11604 against gamlss 0.10340, and log sigma gam 0.2584 against gamlss
//! 0.4170 (MSI census 1106132 at 5a5a9c34f): the mean ratio 1.122 failed the 1.10
//! ceiling while the log-sigma surface was a wide gam win. Both engines now fit the
//! SAME `K_SEEDS` draws, every pair is emitted before any assertion runs, and each
//! surface is decided by `assert_paired_match_or_beat`, which keeps the 1.10
//! ceiling on the draw average and adds the resolved-deficit clause. The first draw
//! is the seed the single-draw version used, drawn from the stream in the same
//! order.
//!
//! Each engine's default smoother (#1561). gamlss fits its default `pb()`
//! (penalized cubic B-splines on 20 intervals) in both predictors, so gam fits
//! its default `s()`. An earlier version forced gam to `s(x, bs='tp', k=6)`, a
//! basis about a quarter the size of the reference's 23, while describing the
//! reference as the same thin-plate basis through `ga()`, which it is not. Measured over
//! the 23 draws gamlss fits (MSI jobs 1219883 and 1255785, per-draw gamlss
//! errors from job 1206812), mean paired log(gam/gamlss) RMSE:
//! - `bs='tp', k=6`: mu +0.114 (gam resolved worse), log sigma -0.302;
//! - `bs='tp'` at its default size: mu +0.030 (still resolved worse), log sigma -0.129;
//! - `bs='ps', k=23`, the size of `pb()`'s basis: mu -0.068, log sigma -0.139;
//! - default `s()`: mu -0.075, log sigma -0.202 (gam resolved better on both).
//!
//! Mean-block degrees of freedom: gamlss `pb()` 16.6 to 19.4 per draw,
//! including its intercept (23 draws, job 1251165). gam, averaged over 25
//! draws: 9.3 under the forced `k=6`, 17.4 with the default-size thin-plate
//! basis and 14.1 with its default `s()`. So the forced basis capped gam's
//! mean; its smoothing selection was not undersmoothing.
//!
//! Notes on the gam side that this test pins down by reading the source:
//!   * `fit_from_formula(..., FitConfig{ noise_formula: Some(...), .. })` routes
//!     through `materialize_location_scale` -> `FitRequest::GaussianLocationScale`.
//!     The Gaussian location-scale model fits on `y / response_scale` and maps the
//!     coefficients back to raw response units, so mu = X_mean*beta_location is
//!     already raw.
//!   * gam's noise (sigma) link in raw units is
//!     `sigma = response_scale*LOGB_SIGMA_FLOOR + exp(eta_scale)`
//!     (`families::sigma_link::LOGB_SIGMA_FLOOR`). sigma is read through the
//!     production `GaussianLocationScalePredictor`, so the floor is the one
//!     prediction uses. The location block carries role `BlockRole::Location`,
//!     the log-sigma block role `BlockRole::Scale`.
//!   * The spec's `linkwiggle(...)` term is a *binomial-only* link correction
//!     (`reject_explicit_linkwiggle_for_nonbinomial` rejects it for a Gaussian
//!     response); it is meaningless here, so the gam formula is the pair of
//!     two-smooth additive blocks without it.

use gam::estimate::BlockRole;
use gam::families::sigma_link::LOGB_SIGMA_FLOOR;
use gam::gamlss::GaussianLocationScaleFitResult;
use gam::matrix::LinearOperator;
use gam::predict::gaussian_location_scale::GaussianLocationScalePredictor;
use gam::predict::{PredictInput, PredictableModel};
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{
    Column, PairedFoldComparison, QualityPair, assert_paired_match_or_beat, relative_l2, rmse,
    run_r,
};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::{Array1, Array2};
use std::f64::consts::PI;
use std::time::Instant;

/// Rows per draw.
const N_ROWS: usize = 120;
/// Evaluation grid points per axis.
const GRID_SIDE: usize = 10;
/// Paired draws. See the "PAIRED over draws" note above.
const K_SEEDS: usize = 25;
/// Seed of the first draw: the single draw the unpaired version of this test used.
const FIRST_SEED: u64 = 999;

/// The KNOWN mean surface every fit is judged against.
fn mu_true(a: f64, b: f64) -> f64 {
    (2.0 * PI * a).sin() + (2.0 * PI * b).cos()
}

/// The KNOWN standard-deviation surface every fit is judged against.
fn sigma_true(a: f64, b: f64) -> f64 {
    0.1 + 0.1 * (PI * a).sin() + 0.05 * b
}

/// One draw, fed IDENTICALLY to both engines: x1 ~ Uniform(0,1), x2 ~ Uniform(0,1)
/// (n=120), y ~ N(mu, sigma^2). A deterministic seeded LCG draws the uniforms and
/// then the Box-Muller standard normals from ONE stream, so the exact same data is
/// reproducible in pure Rust and sent verbatim to gamlss.
fn multi_smooth_draw(seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut state = seed;
    let mut next_unit = || -> f64 {
        // Numerical Recipes LCG; take the high bits for a uniform in [0,1).
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };

    let x1: Vec<f64> = (0..N_ROWS).map(|_| next_unit()).collect();
    let x2: Vec<f64> = (0..N_ROWS).map(|_| next_unit()).collect();

    // Box-Muller standard normals from the same LCG stream (seed continues).
    let mut zvals: Vec<f64> = Vec::with_capacity(N_ROWS);
    while zvals.len() < N_ROWS {
        let u1 = next_unit().max(1e-300);
        let u2 = next_unit();
        let r = (-2.0 * u1.ln()).sqrt();
        zvals.push(r * (2.0 * PI * u2).cos());
        if zvals.len() < N_ROWS {
            zvals.push(r * (2.0 * PI * u2).sin());
        }
    }

    let y: Vec<f64> = (0..N_ROWS)
        .map(|i| mu_true(x1[i], x2[i]) + sigma_true(x1[i], x2[i]) * zvals[i])
        .collect();
    (x1, x2, y)
}

/// gam's two-smooth-per-block Gaussian location-scale fit on one draw: the fitted
/// mean and log standard deviation on the grid.
fn gam_surfaces_on_grid(
    seed: u64,
    x1: &[f64],
    x2: &[f64],
    y: &[f64],
    grid_x1: &[f64],
    grid_x2: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let headers: Vec<String> = vec!["x1".to_string(), "x2".to_string(), "y".to_string()];
    let rows: Vec<csv::StringRecord> = (0..N_ROWS)
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

    // mu       ~ s(x1) + s(x2)
    // log-sigma ~ s(x1) + s(x2)
    // gam's default smooth, as the reference fits gamlss's default `pb()`. See
    // "Each engine's default smoother" above.
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        noise_formula: Some("s(x1) + s(x2)".to_string()),
        ..FitConfig::default()
    };
    let fit_started = Instant::now();
    let result = fit_from_formula("y ~ s(x1) + s(x2)", &ds, &cfg)
        .unwrap_or_else(|e| panic!("gam multi-smooth location-scale fit (seed={seed}) failed: {e:?}"));
    let fit_elapsed = fit_started.elapsed();
    let FitResult::GaussianLocationScale(GaussianLocationScaleFitResult {
        fit,
        response_scale,
        ..
    }) = result
    else {
        panic!("expected a Gaussian location-scale fit");
    };
    assert!(
        fit_elapsed.as_secs_f64() <= 120.0,
        "gam gaussian multi-smooth fit exceeded #1082 bounded-fixture budget: seed={seed} \
         elapsed={:.1}s outer_iters={} inner_cycles={} p={}",
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

    let grid_n = grid_x1.len();
    let mut grid = Array2::<f64>::zeros((grid_n, ncols));
    for i in 0..grid_n {
        grid[[i, x1_idx]] = grid_x1[i];
        grid[[i, x2_idx]] = grid_x2[i];
    }

    // Rebuild the SAME frozen mean / log-sigma designs at the grid points.
    // mu = X_mean*beta_location; sigma comes from the production Gaussian
    // location-scale predictor, sigma = response_scale*LOGB_SIGMA_FLOOR +
    // exp(X_scale*beta_scale).
    let mean_design_grid = build_term_collection_design(grid.view(), &fit.meanspec_resolved)
        .expect("rebuild mean design at grid");
    let scale_design_grid = build_term_collection_design(grid.view(), &fit.noisespec_resolved)
        .expect("rebuild log-sigma design at grid");

    let gam_mu: Vec<f64> = mean_design_grid.design.apply(&beta_location).to_vec();
    let gam_sigma: Vec<f64> = GaussianLocationScalePredictor {
        beta_mu: beta_location.clone(),
        beta_noise: beta_scale.clone(),
        sigma_floor: LOGB_SIGMA_FLOOR,
        response_scale,
        covariance: None,
        link_wiggle: None,
    }
    .predict_noise_scale(&PredictInput {
        design: mean_design_grid.design,
        offset: Array1::zeros(grid_n),
        design_noise: Some(scale_design_grid.design),
        offset_noise: None,
        auxiliary_scalar: None,
        auxiliary_matrix: None,
    })
    .expect("production Gaussian location-scale sigma at grid")
    .expect("Gaussian location-scale predictor exposes a noise scale")
    .to_vec();

    assert_eq!(gam_mu.len(), grid_n);
    assert_eq!(gam_sigma.len(), grid_n);
    (gam_mu, gam_sigma.iter().map(|&s| s.ln()).collect())
}

#[test]
fn gam_gaussian_multi_smooth_matches_gamlss() {
    init_parallelism();

    // ---- dense 10x10 evaluation grid over [0,1]^2, and the truth on it -------
    let grid_n = GRID_SIDE * GRID_SIDE;
    let axis: Vec<f64> = (0..GRID_SIDE)
        .map(|i| i as f64 / (GRID_SIDE as f64 - 1.0))
        .collect();
    let mut grid_x1: Vec<f64> = Vec::with_capacity(grid_n);
    let mut grid_x2: Vec<f64> = Vec::with_capacity(grid_n);
    for &a in &axis {
        for &b in &axis {
            grid_x1.push(a);
            grid_x2.push(b);
        }
    }
    let truth_mu: Vec<f64> = (0..grid_n)
        .map(|i| mu_true(grid_x1[i], grid_x2[i]))
        .collect();
    let truth_log_sigma: Vec<f64> = (0..grid_n)
        .map(|i| sigma_true(grid_x1[i], grid_x2[i]).ln())
        .collect();

    // ---- gam on every draw, and the long-format data the reference replays ---
    let mut gam_mu_rmses = Vec::with_capacity(K_SEEDS);
    let mut gam_ls_rmses = Vec::with_capacity(K_SEEDS);
    let mut gam_mu_grids = Vec::with_capacity(K_SEEDS);
    let mut gam_ls_grids = Vec::with_capacity(K_SEEDS);
    let mut long_seed = Vec::with_capacity(K_SEEDS * N_ROWS);
    let mut long_x1 = Vec::with_capacity(K_SEEDS * N_ROWS);
    let mut long_x2 = Vec::with_capacity(K_SEEDS * N_ROWS);
    let mut long_y = Vec::with_capacity(K_SEEDS * N_ROWS);
    for k in 0..K_SEEDS {
        let seed = FIRST_SEED + k as u64;
        let (x1, x2, y) = multi_smooth_draw(seed);
        let (gam_mu, gam_log_sigma) = gam_surfaces_on_grid(seed, &x1, &x2, &y, &grid_x1, &grid_x2);
        gam_mu_rmses.push(rmse(&gam_mu, &truth_mu));
        gam_ls_rmses.push(rmse(&gam_log_sigma, &truth_log_sigma));
        gam_mu_grids.push(gam_mu);
        gam_ls_grids.push(gam_log_sigma);
        long_seed.resize(long_seed.len() + N_ROWS, seed as f64);
        long_x1.extend_from_slice(&x1);
        long_x2.extend_from_slice(&x2);
        long_y.extend_from_slice(&y);
    }

    // ---- the SAME K draws through gamlss, in ONE R session -------------------
    // family = NO() (Gaussian with mu + log-sigma); two additive penalized smooths
    // via gamlss's native penalized B-spline `pb()` (one per covariate, automatic
    // smoothing-parameter selection) in BOTH mu.formula and sigma.formula. This
    // replaces the gamlss.add/mgcv `ga(~ s(., bs="tp"))` bridge (unavailable here):
    // `pb(x1) + pb(x2)` is the correct additive construction and exercises the SAME
    // pair of one-dimensional penalized smooths in both predictors.
    // `predictAll(..., data = d)` re-supplies the draw's fitting frame the smoother
    // needs to evaluate at new points and returns mu and sigma on the response
    // scale in one call.
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
        gx1 <- as.numeric(strsplit("{grid_x1_csv}", ",")[[1]])
        gx2 <- as.numeric(strsplit("{grid_x2_csv}", ",")[[1]])
        nd <- data.frame(x1 = gx1, x2 = gx2)
        mu_all <- c(); sigma_all <- c(); fitted_all <- c()
        for (s in sort(unique(df$seed))) {{
            d <- df[df$seed == s, c("x1", "x2", "y")]
            # pb()'s local smoothing-parameter search can stop on a draw with an R
            # error (MSI job 1206812: "missing value where TRUE/FALSE needed" on
            # draw 1002, "object 'fit' not found" on draw 1011). Such a draw has no
            # reference fit to compare against, so it is recorded as unfitted and
            # carries NaN surfaces; every other draw keeps the reference unchanged.
            pa <- tryCatch({{
                m <- gamlss(y ~ pb(x1) + pb(x2),
                            sigma.formula = ~ pb(x1) + pb(x2),
                            family = NO(), data = d,
                            control = gamlss.control(n.cyc = 80, trace = FALSE))
                predictAll(m, newdata = nd, data = d, type = "response")
            }}, error = function(e) NULL)
            if (is.null(pa)) {{
                fitted_all <- c(fitted_all, 0)
                mu_all <- c(mu_all, rep(NaN, nrow(nd)))
                sigma_all <- c(sigma_all, rep(NaN, nrow(nd)))
            }} else {{
                fitted_all <- c(fitted_all, 1)
                mu_all <- c(mu_all, as.numeric(pa$mu))
                sigma_all <- c(sigma_all, as.numeric(pa$sigma))
            }}
        }}
        emit("fitted", fitted_all)
        emit("mu", mu_all)
        emit("sigma", sigma_all)
        "#
    );
    let r = run_r(
        &[
            Column::new("seed", &long_seed),
            Column::new("x1", &long_x1),
            Column::new("x2", &long_x2),
            Column::new("y", &long_y),
        ],
        &body,
    );
    let gamlss_fitted = r.vector("fitted");
    let gamlss_mu_flat = r.vector("mu");
    let gamlss_sigma_flat = r.vector("sigma");
    assert_eq!(
        gamlss_fitted.len(),
        K_SEEDS,
        "gamlss fitted-draw indicator length mismatch"
    );
    assert_eq!(
        gamlss_mu_flat.len(),
        K_SEEDS * grid_n,
        "gamlss mu panel length mismatch"
    );
    assert_eq!(
        gamlss_sigma_flat.len(),
        K_SEEDS * grid_n,
        "gamlss sigma panel length mismatch"
    );

    // ---- OBJECTIVE metric per draw: truth-recovery RMSE ----------------------
    // Every gam error is printed on every draw. The pairs are the draws where the
    // reference produced a fit: a draw gamlss could not fit has nothing to pair
    // with, and which draws those are depends on the reference alone.
    for k in 0..K_SEEDS {
        eprintln!(
            "multi_smooth draw seed={} gam_mu_rmse={:.5} gam_log_sigma_rmse={:.5} gamlss_fitted={}",
            FIRST_SEED + k as u64,
            gam_mu_rmses[k],
            gam_ls_rmses[k],
            gamlss_fitted[k] == 1.0
        );
    }
    let fitted_draws: Vec<usize> = (0..K_SEEDS).filter(|&k| gamlss_fitted[k] == 1.0).collect();
    let unfitted_seeds: Vec<u64> = (0..K_SEEDS)
        .filter(|&k| gamlss_fitted[k] != 1.0)
        .map(|k| FIRST_SEED + k as u64)
        .collect();
    let mut paired_gam_mu = Vec::with_capacity(fitted_draws.len());
    let mut paired_gam_ls = Vec::with_capacity(fitted_draws.len());
    let mut gamlss_mu_rmses = Vec::with_capacity(fitted_draws.len());
    let mut gamlss_ls_rmses = Vec::with_capacity(fitted_draws.len());
    let mut rel_mu_total = 0.0;
    let mut rel_log_sigma_total = 0.0;
    for &k in &fitted_draws {
        let lo = k * grid_n;
        let hi = lo + grid_n;
        let gamlss_mu = &gamlss_mu_flat[lo..hi];
        let gamlss_log_sigma: Vec<f64> = gamlss_sigma_flat[lo..hi].iter().map(|&s| s.ln()).collect();
        paired_gam_mu.push(gam_mu_rmses[k]);
        paired_gam_ls.push(gam_ls_rmses[k]);
        gamlss_mu_rmses.push(rmse(gamlss_mu, &truth_mu));
        gamlss_ls_rmses.push(rmse(&gamlss_log_sigma, &truth_log_sigma));
        // Reference closeness kept ONLY as printed context, not a pass criterion.
        rel_mu_total += relative_l2(&gam_mu_grids[k], gamlss_mu);
        rel_log_sigma_total += relative_l2(&gam_ls_grids[k], &gamlss_log_sigma);
    }

    // ---- paired panels: same draw, same bytes, draw by draw ------------------
    let mu_panel = PairedFoldComparison::new(&paired_gam_mu, &gamlss_mu_rmses, true);
    let ls_panel = PairedFoldComparison::new(&paired_gam_ls, &gamlss_ls_rmses, true);

    eprintln!(
        "gaussian multi-smooth location-scale truth recovery: n={N_ROWS} grid={grid_n} \
         K={K_SEEDS} draws, {} paired (gamlss did not fit seeds {unfitted_seeds:?})\n  \
         RMSE_vs_truth(mu): gam={:.5} gamlss={:.5}\n  \
         RMSE_vs_truth(log sigma): gam={:.5} gamlss={:.5}\n  \
         [context] mean rel_l2_vs_gamlss(mu)={:.5} mean rel_l2_vs_gamlss(log sigma)={:.5}",
        fitted_draws.len(),
        mu_panel.gam_mean,
        mu_panel.reference_mean,
        ls_panel.gam_mean,
        ls_panel.reference_mean,
        rel_mu_total / fitted_draws.len() as f64,
        rel_log_sigma_total / fitted_draws.len() as f64,
    );
    eprintln!("{}", mu_panel.report("gaussian_multi_smooth::mu"));
    eprintln!("{}", ls_panel.report("gaussian_multi_smooth::log_sigma"));
    eprintln!(
        "{}",
        QualityPair::paired(
            "families",
            "quality_vs_gamlss_gaussian_multi_smooth::mu",
            "mu_rmse_to_truth",
            "gamlss",
            &mu_panel,
        )
        .line()
    );
    eprintln!(
        "{}",
        QualityPair::paired(
            "families",
            "quality_vs_gamlss_gaussian_multi_smooth::log_sigma",
            "log_sigma_rmse_to_truth",
            "gamlss",
            &ls_panel,
        )
        .line()
    );

    // PRIMARY claim: gam recovers the known generating surfaces, on the draw average.
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
        mu_panel.gam_mean <= 0.20,
        "gam failed to recover the mean surface: draw-mean RMSE_vs_truth(mu)={:.5} > 0.20",
        mu_panel.gam_mean
    );

    // Log-sigma bar. The log-sd surface is a noisier second-moment quantity:
    // log(sigma_true) ranges over roughly [log 0.10, log 0.25] ~ [-2.30, -1.39]
    // (a span of ~0.91), and gam's floored noise link sigma = 0.01 + exp(eta)
    // adds a small pointwise bias near the floor. A faithful recovery still
    // tracks the truth to RMSE(log sigma) <= 0.30, about a third of the
    // log-sigma signal span.
    assert!(
        ls_panel.gam_mean <= 0.30,
        "gam failed to recover the log-sigma surface: \
         draw-mean RMSE_vs_truth(log sigma)={:.5} > 0.30",
        ls_panel.gam_mean
    );

    // SECONDARY claim: match-or-beat the mature GAMLSS reference ON ACCURACY,
    // paired across the K shared draws: gam's truth-recovery error may not exceed
    // gamlss's draw average by more than 10%, nor be resolved worse draw by draw.
    // This is a comparison of who recovers the truth better, NOT a claim that gam
    // reproduces gamlss's fitted output.
    assert_paired_match_or_beat("gaussian_multi_smooth::mu", &mu_panel, 1.10);
    assert_paired_match_or_beat("gaussian_multi_smooth::log_sigma", &ls_panel, 1.10);
}
