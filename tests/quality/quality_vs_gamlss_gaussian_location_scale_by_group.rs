//! End-to-end quality: gam's Gaussian location-scale fit with a **by-group**
//! smooth in *both* the mean and the log-σ model must RECOVER THE TRUTH — the
//! known data-generating mean and σ functions of each group — and do so at
//! least as accurately as the standard GAMLSS engine.
//!
//! Objective metric (truth recovery)
//! ----------------------------------
//! The data is synthetic: every (x, y) is drawn from a fully specified Gaussian
//! location-scale law with a KNOWN per-group mean function μ*(x) and KNOWN
//! per-group σ function σ*(x). Quality therefore is not "does gam look like
//! gamlss" — both could overfit the same noise — but "how close is gam's fitted
//! curve to the function that actually generated the data". On a 50-point grid
//! spanning each group's observed x-range we evaluate the true μ*(x) and σ*(x)
//! analytically and assert, PER GROUP:
//!   * RMSE(gam μ̂, μ*) ≤ MU_RMSE_BAR — the fitted mean tracks the true mean to
//!     well inside the measurement noise (the bar is a fraction of the local
//!     noise sigma, NOT a fraction of a reference fit).
//!   * RMSE(gam log σ̂, log σ*) ≤ LOGSIG_RMSE_BAR — the fitted log-σ tracks the
//!     true log-σ (σ is a second, noisier estimand built from squared
//!     residuals, hence its own looser-but-principled bar).
//!
//! Match-or-beat baseline (gamlss on accuracy, not on identity)
//! ------------------------------------------------------------
//! `gamlss::gamlss(family=NO())` is fit to the SAME data, stratified per group
//! (one independent fit per stratum with a `tp` smooth via `ga(~ s(x, bs="tp"))`
//! in both `mu` and `sigma`) — the textbook GAMLSS way to obtain group-specific
//! mean and σ smooths, since gamlss has no `by=` machinery. We then additionally
//! require gam's truth-recovery error to be no worse than gamlss's by more than
//! 10% (RMSE_gam ≤ 1.10 · RMSE_gamlss), for both μ and log σ in both groups.
//! gamlss is thus a YARDSTICK ON ACCURACY-VS-TRUTH, never a target to imitate:
//! matching gamlss's noisy fit proves nothing; beating (or tying) it on distance
//! to the real generating function does.
//!
//! Data (seed 321, n = 200, 100 per group), fed IDENTICALLY to both engines:
//!   group A: x ~ U(0,1), y ~ N(sin(2πx),  (0.10 + 0.10 sin(πx))^2)
//!   group B: x ~ U(0,1), y ~ N(0.5 + 0.3 sin(3πx), (0.12 + 0.08 x)^2)
//! Group A has a smooth heteroscedastic σ (a hump); group B a near-linear σ
//! ramp. The two groups differ in BOTH mean shape and σ shape, so a by=/log-σ
//! cross-block leak would pull a fitted curve away from its own group's truth
//! and blow the truth-recovery RMSE. The relative-L2-to-gamlss numbers are still
//! computed and printed for context, but the PASS/FAIL gate is truth recovery.
//! The bars are NOT loosened to whitelist a divergence; a genuine gap is a real
//! finding.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, relative_l2, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

const N_PER_GROUP: usize = 100;
const GRID_POINTS: usize = 50;

/// Truth-recovery bar on the fitted mean (μ): RMSE between gam μ̂ and the true
/// μ*(x) on the grid. The per-group noise sigma is ~0.10-0.20; a penalized tp
/// smooth on 100 points recovers a smooth mean to well inside that, so 0.07 is a
/// principled bar (a fraction of the local noise, not of a reference fit).
const MU_RMSE_BAR: f64 = 0.07;
/// Truth-recovery bar on the fitted log σ: RMSE between gam log σ̂ and log σ*(x).
/// σ is estimated from squared residuals and is intrinsically noisier than the
/// mean; log σ here ranges roughly over [ln 0.10, ln 0.20] ≈ [-2.30, -1.61], and
/// 0.25 nats keeps the fit firmly tracking the true variance structure.
const LOGSIG_RMSE_BAR: f64 = 0.25;
/// Match-or-beat slack: gam's truth-recovery RMSE must not exceed gamlss's by
/// more than this factor, for either estimand in either group.
const BEAT_FACTOR: f64 = 1.10;

/// True mean for group A.
fn mean_a(x: f64) -> f64 {
    (2.0 * std::f64::consts::PI * x).sin()
}
/// True sigma for group A (smooth heteroscedastic hump).
fn sigma_a(x: f64) -> f64 {
    0.10 + 0.10 * (std::f64::consts::PI * x).sin()
}
/// True mean for group B.
fn mean_b(x: f64) -> f64 {
    0.5 + 0.3 * (3.0 * std::f64::consts::PI * x).sin()
}
/// True sigma for group B (near-linear ramp).
fn sigma_b(x: f64) -> f64 {
    0.12 + 0.08 * x
}

/// `seq(a, b, length.out = GRID_POINTS)` replicated EXACTLY so the gam grid and
/// the R `seq(...)` grid are bit-for-bit identical given identical (a, b).
fn linspace(a: f64, b: f64) -> Vec<f64> {
    (0..GRID_POINTS)
        .map(|i| a + (b - a) * (i as f64) / ((GRID_POINTS - 1) as f64))
        .collect()
}

#[test]
fn gam_location_scale_by_group_matches_gamlss() {
    init_parallelism();

    // ---- synthesize the two-group location-scale data (seed 321) ----------
    // Group A rows first, then group B, so the inferred categorical levels are
    // ["A", "B"] -> codes [0.0, 1.0] in first-seen insertion order.
    let mut rng = StdRng::seed_from_u64(321);
    let ux = Uniform::new(0.0_f64, 1.0_f64).expect("uniform x");
    let std_normal = Normal::new(0.0_f64, 1.0_f64).expect("standard normal");

    let headers = vec!["y".to_string(), "x".to_string(), "group".to_string()];
    let mut rows: Vec<StringRecord> = Vec::with_capacity(2 * N_PER_GROUP);

    // We also keep the raw (x, y) per group so the R side can subset by group
    // and so we can build the per-group prediction grids over the SAME x-range.
    let mut x_a = Vec::with_capacity(N_PER_GROUP);
    let mut x_b = Vec::with_capacity(N_PER_GROUP);

    for _ in 0..N_PER_GROUP {
        let x = ux.sample(&mut rng);
        let y = mean_a(x) + sigma_a(x) * std_normal.sample(&mut rng);
        x_a.push(x);
        rows.push(StringRecord::from(vec![
            y.to_string(),
            x.to_string(),
            "A".to_string(),
        ]));
    }
    for _ in 0..N_PER_GROUP {
        let x = ux.sample(&mut rng);
        let y = mean_b(x) + sigma_b(x) * std_normal.sample(&mut rng);
        x_b.push(x);
        rows.push(StringRecord::from(vec![
            y.to_string(),
            x.to_string(),
            "B".to_string(),
        ]));
    }

    // Columns shipped to R verbatim (same numbers gam encodes). group is sent
    // as a 0/1 code matching gam's first-seen encoding (A=0, B=1); R turns it
    // back into a factor for stratification.
    let mut y_all = Vec::with_capacity(2 * N_PER_GROUP);
    let mut x_all = Vec::with_capacity(2 * N_PER_GROUP);
    let mut g_all = Vec::with_capacity(2 * N_PER_GROUP);
    for r in &rows {
        y_all.push(r.get(0).expect("y field").parse::<f64>().expect("parse y"));
        x_all.push(r.get(1).expect("x field").parse::<f64>().expect("parse x"));
        g_all.push(if r.get(2) == Some("A") { 0.0 } else { 1.0 });
    }

    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode by-group data");
    let col = data.column_map();
    let x_idx = col["x"];
    let group_idx = col["group"];

    // ---- fit gam: Gaussian location-scale, by-group tp smooth on BOTH μ and
    //      log σ ----------------------------------------------------------
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        noise_formula: Some("s(x, bs='tp', by=group)".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x, bs='tp', by=group)", &data, &cfg)
        .expect("gam Gaussian location-scale by-group fit");
    let FitResult::GaussianLocationScale(fit) = result else {
        panic!("expected a GaussianLocationScale fit result for a noise_formula model");
    };

    // Mean (μ) coefficient block: identity link => μ = X_mean · β_mean.
    let beta_mean = &fit
        .fit
        .fit
        .block_by_role(gam::solver::estimate::BlockRole::Location)
        .expect("location-scale fit must carry a Location (mean) block")
        .beta;
    // log-σ (Scale) coefficient block: log σ = X_noise · β_scale (log link).
    let beta_scale = &fit
        .fit
        .fit
        .block_by_role(gam::solver::estimate::BlockRole::Scale)
        .expect("location-scale fit must carry a Scale (log-σ) block")
        .beta;

    // Build the per-group prediction grids over each group's observed x-range.
    let (a_lo, a_hi) = x_a
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    let (b_lo, b_hi) = x_b
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    let grid_a = linspace(a_lo, a_hi);
    let grid_b = linspace(b_lo, b_hi);

    let ncols = data.headers.len();

    // gam μ and σ on each per-group grid: set the group code so the by= term
    // activates only that group's block, and read off X·β through the frozen
    // resolved specs (mean uses meanspec_resolved, σ uses noisespec_resolved).
    let predict_group = |grid: &[f64], code: f64| -> (Vec<f64>, Vec<f64>) {
        let mut design_pts = Array2::<f64>::zeros((grid.len(), ncols));
        for (i, &xv) in grid.iter().enumerate() {
            design_pts[[i, x_idx]] = xv;
            design_pts[[i, group_idx]] = code;
        }
        let mean_design =
            build_term_collection_design(design_pts.view(), &fit.fit.meanspec_resolved)
                .expect("rebuild mean design at grid");
        let noise_design =
            build_term_collection_design(design_pts.view(), &fit.fit.noisespec_resolved)
                .expect("rebuild noise design at grid");
        let mu = mean_design.design.apply(beta_mean).to_vec();
        let log_sigma = noise_design.design.apply(beta_scale).to_vec();
        let sigma: Vec<f64> = log_sigma.iter().map(|v| v.exp()).collect();
        (mu, sigma)
    };

    let (gam_mu_a, gam_sigma_a) = predict_group(&grid_a, 0.0);
    let (gam_mu_b, gam_sigma_b) = predict_group(&grid_b, 1.0);

    // ---- fit the SAME data with gamlss, stratified by group (the mature
    //      GAMLSS reference). One independent gamlss per group, tp smooth via
    //      ga(~ s(x, bs="tp")) in BOTH the mu and sigma models, NO() family.
    //      predictAll returns μ and σ on the response scale. ------------------
    let grid_concat: Vec<f64> = grid_a.iter().chain(grid_b.iter()).copied().collect();
    let r = run_r(
        &[
            Column::new("y", &y_all),
            Column::new("x", &x_all),
            Column::new("g", &g_all),
            // Two extra columns of equal length (200) carrying the 100+100
            // grid points (padded) so we can ship the exact prediction grid.
            Column::new("gridx", &pad_to(&grid_concat, 2 * N_PER_GROUP)),
        ],
        r#"
        suppressPackageStartupMessages(library(gamlss))
        df$g <- factor(ifelse(df$g < 0.5, "A", "B"), levels = c("A", "B"))
        gridx <- df$gridx[1:100]
        gA <- gridx[1:50]
        gB <- gridx[51:100]

        # Smooth mean AND smooth log-sigma via gamlss's native penalized
        # B-spline `pb()` (REML-selected lambda). This is the canonical GAMLSS
        # smoother and lives in base `gamlss`; the earlier `ga(~ s(x))` form
        # required the separate `gamlss.add` package (unavailable here) and
        # produced the "cannot coerce class 'function' to a data.frame" error.
        fit_one <- function(sub, grid) {
          m <- gamlss(y ~ pb(x),
                      sigma.formula = ~ pb(x),
                      family = NO(), data = sub,
                      control = gamlss.control(trace = FALSE))
          nd <- data.frame(x = grid)
          pa <- predictAll(m, newdata = nd, data = sub, type = "response")
          list(mu = as.numeric(pa$mu), sigma = as.numeric(pa$sigma))
        }

        ra <- fit_one(df[df$g == "A", ], gA)
        rb <- fit_one(df[df$g == "B", ], gB)
        emit("mu_a", ra$mu)
        emit("sigma_a", ra$sigma)
        emit("mu_b", rb$mu)
        emit("sigma_b", rb$sigma)
        "#,
    );

    let ref_mu_a = r.vector("mu_a");
    let ref_sigma_a = r.vector("sigma_a");
    let ref_mu_b = r.vector("mu_b");
    let ref_sigma_b = r.vector("sigma_b");
    assert_eq!(ref_mu_a.len(), GRID_POINTS, "gamlss group-A μ grid length");
    assert_eq!(ref_mu_b.len(), GRID_POINTS, "gamlss group-B μ grid length");

    let log_vec = |v: &[f64]| -> Vec<f64> { v.iter().map(|s| s.ln()).collect() };

    // ---- GROUND TRUTH on each grid: the functions that actually generated the
    //      data. These, not the reference fit, are what gam must recover. ------
    let truth_mu_a: Vec<f64> = grid_a.iter().map(|&x| mean_a(x)).collect();
    let truth_mu_b: Vec<f64> = grid_b.iter().map(|&x| mean_b(x)).collect();
    let truth_logsig_a: Vec<f64> = grid_a.iter().map(|&x| sigma_a(x).ln()).collect();
    let truth_logsig_b: Vec<f64> = grid_b.iter().map(|&x| sigma_b(x).ln()).collect();

    // gam truth-recovery error (the PRIMARY claim).
    let gam_mu_a_err = rmse(&gam_mu_a, &truth_mu_a);
    let gam_mu_b_err = rmse(&gam_mu_b, &truth_mu_b);
    let gam_logsig_a_err = rmse(&log_vec(&gam_sigma_a), &truth_logsig_a);
    let gam_logsig_b_err = rmse(&log_vec(&gam_sigma_b), &truth_logsig_b);

    // gamlss truth-recovery error on the SAME truth (the match-or-beat yardstick).
    let ref_mu_a_err = rmse(ref_mu_a, &truth_mu_a);
    let ref_mu_b_err = rmse(ref_mu_b, &truth_mu_b);
    let ref_logsig_a_err = rmse(&log_vec(ref_sigma_a), &truth_logsig_a);
    let ref_logsig_b_err = rmse(&log_vec(ref_sigma_b), &truth_logsig_b);

    // Closeness-to-gamlss, computed and printed for CONTEXT only (not a gate).
    let mu_a_rel = relative_l2(&gam_mu_a, ref_mu_a);
    let mu_b_rel = relative_l2(&gam_mu_b, ref_mu_b);
    let logsig_a_rel = relative_l2(&log_vec(&gam_sigma_a), &log_vec(ref_sigma_a));
    let logsig_b_rel = relative_l2(&log_vec(&gam_sigma_b), &log_vec(ref_sigma_b));

    eprintln!(
        "by-group location-scale truth recovery (RMSE vs TRUTH): \
         A mu gam={gam_mu_a_err:.4} gamlss={ref_mu_a_err:.4} | \
         A logsig gam={gam_logsig_a_err:.4} gamlss={ref_logsig_a_err:.4} | \
         B mu gam={gam_mu_b_err:.4} gamlss={ref_mu_b_err:.4} | \
         B logsig gam={gam_logsig_b_err:.4} gamlss={ref_logsig_b_err:.4}"
    );
    eprintln!(
        "context (rel_l2 gam-vs-gamlss, NOT a gate): \
         A mu_rel={mu_a_rel:.4} logsig_rel={logsig_a_rel:.4} | \
         B mu_rel={mu_b_rel:.4} logsig_rel={logsig_b_rel:.4}"
    );

    // ---- PRIMARY: gam recovers the true per-group mean and log-σ functions. --
    assert!(
        gam_mu_a_err <= MU_RMSE_BAR,
        "group-A mean does not recover truth: RMSE(μ̂, μ*)={gam_mu_a_err:.4} > {MU_RMSE_BAR}"
    );
    assert!(
        gam_mu_b_err <= MU_RMSE_BAR,
        "group-B mean does not recover truth: RMSE(μ̂, μ*)={gam_mu_b_err:.4} > {MU_RMSE_BAR}"
    );
    assert!(
        gam_logsig_a_err <= LOGSIG_RMSE_BAR,
        "group-A log-sigma does not recover truth: RMSE(log σ̂, log σ*)={gam_logsig_a_err:.4} > {LOGSIG_RMSE_BAR}"
    );
    assert!(
        gam_logsig_b_err <= LOGSIG_RMSE_BAR,
        "group-B log-sigma does not recover truth: RMSE(log σ̂, log σ*)={gam_logsig_b_err:.4} > {LOGSIG_RMSE_BAR}"
    );

    // ---- MATCH-OR-BEAT: gam's truth error is within 10% of gamlss's. ---------
    assert!(
        gam_mu_a_err <= ref_mu_a_err * BEAT_FACTOR,
        "group-A mean worse than gamlss on truth: gam={gam_mu_a_err:.4} > {BEAT_FACTOR}*gamlss={ref_mu_a_err:.4}"
    );
    assert!(
        gam_mu_b_err <= ref_mu_b_err * BEAT_FACTOR,
        "group-B mean worse than gamlss on truth: gam={gam_mu_b_err:.4} > {BEAT_FACTOR}*gamlss={ref_mu_b_err:.4}"
    );
    assert!(
        gam_logsig_a_err <= ref_logsig_a_err * BEAT_FACTOR,
        "group-A log-sigma worse than gamlss on truth: gam={gam_logsig_a_err:.4} > {BEAT_FACTOR}*gamlss={ref_logsig_a_err:.4}"
    );
    assert!(
        gam_logsig_b_err <= ref_logsig_b_err * BEAT_FACTOR,
        "group-B log-sigma worse than gamlss on truth: gam={gam_logsig_b_err:.4} > {BEAT_FACTOR}*gamlss={ref_logsig_b_err:.4}"
    );
}

/// Pad `v` with trailing zeros to `len` so it can ride as a fixed-width column.
fn pad_to(v: &[f64], len: usize) -> Vec<f64> {
    assert!(v.len() <= len, "pad_to: vector longer than target length");
    let mut out = v.to_vec();
    out.resize(len, 0.0);
    out
}
