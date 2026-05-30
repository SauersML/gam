//! End-to-end quality: gam's *survival* location-scale family (a smooth
//! location AND a smooth log-scale, with a flexible monotone time baseline and
//! a Gaussian residual) must recover the same two x-dependent surfaces that
//! `gamlss` — the de-facto standard R package for distributional /
//! location-scale regression — recovers on the *identical* right-censored data.
//!
//! Why gamlss (family = NO) is the right reference. gam's survival
//! location-scale model with a Gaussian residual is, structurally, an
//! accelerated-failure-time (AFT) model: reading the predictor assembly in
//! `families::survival_location_scale` (see `survival_location_scale_response_
//! from_predictors`), the standardized survival index is
//!     z(t, x) = h(t) - eta_t(x) * exp(-eta_ls(x)),   S(t|x) = 1 - Phi(z),
//! i.e. a *location* channel `eta_t(x)` (role `BlockRole::Threshold`) and a
//! *log-scale* channel `eta_ls(x)` (role `BlockRole::Scale`, link
//! `sigma = exp(eta_ls)` — the pure `exp_sigma` link, NOT the floored
//! `logb_sigma` link), composed with a learned monotone transform `h(t)` of the
//! survival-time axis. The Gaussian-residual log-survival is exactly a normal
//! AFT on the `h(t)`-warped clock. gamlss with `family = NO()` fits the
//! analogous normal AFT on `log(t)` with a smooth mean `mu(x)` and a smooth
//! `log sigma(x)`; right-censoring is handled with `gamlss.cens` (`gen.cens(NO,
//! type = "right")` → family `NOrc`, response `Surv(log t, event)`).
//!
//! The cross-feature combination this pins down (which single-channel survival
//! tests never exercise): survival x smooth-covariate-in-location x
//! smooth-covariate-in-scale, fit jointly. The data has *strong x-dependence in
//! both channels* (the Weibull scale, hence the AFT location, moves with
//! sin(2*pi*x); we also induce a smooth scale signal). We feed the byte-identical
//! (x, exit, event) rows to both engines and compare the recovered smooth shapes
//! on a 20-point x-grid.
//!
//! What is and is NOT directly comparable. gam learns the time transform `h(t)`
//! flexibly while gamlss fixes it at `log(t)`; the two baselines therefore
//! differ by an unknown smooth (approximately affine over the bulk) reparametri-
//! zation of the time axis. The *shape* of the x-dependence — how the location
//! and the log-scale move with x — is the engine-agnostic invariant, so we
//! compare the mean-centered location surface and the mean-centered log-scale
//! surface (the baseline-induced additive offset removed) by relative L2 and by
//! Pearson correlation. This is the honest quantity: it measures whether gam's
//! two-channel smooth recovery and its `noise_formula` (log-scale) smoothing
//! converge to the same functional dependence gamlss finds, without pretending
//! the two engines share a time-axis gauge they do not.
//!
//! gam-side specifics verified against the source:
//!   * The in-Rust survival location-scale path is selected by
//!     `FitConfig{ survival_likelihood: "location-scale", survival_distribution:
//!     "gaussian", noise_formula: Some(...) }` with a `Surv(entry, exit, event)`
//!     response; `materialize_survival` routes the main RHS to the threshold
//!     (location) `thresholdspec` and the `noise_formula` to the log-sigma
//!     `log_sigmaspec` (workflow.rs `build_location_scale_request`).
//!   * The fit returns `FitResult::SurvivalLocationScale`; the converged
//!     `UnifiedFitResult` exposes the location coefficients via
//!     `beta_threshold()` (role Threshold) and the log-scale coefficients via
//!     `beta_log_sigma()` (role Scale). The frozen designs are rebuildable at
//!     arbitrary x from `resolved_thresholdspec` / `resolved_log_sigmaspec`
//!     through `build_term_collection_design`, exactly as the Gaussian
//!     location-scale quality test rebuilds its mean / log-sigma designs.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, relative_l2, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;

#[test]
fn gam_gaussian_survival_location_scale_matches_gamlss() {
    init_parallelism();

    // ---- synthetic right-censored recipe (fed IDENTICALLY to both engines) ----
    // Mirrors the spec: n = 200, x ~ Uniform(-2, 2), entry = 0,
    // exit ~ Weibull(shape = 1.5, scale = exp(-0.5 + 0.3*sin(2*pi*x))),
    // event ~ Bernoulli(0.7). Strong x-dependence in the time scale gives strong
    // x-dependence in the AFT location; an additional smooth multiplier on the
    // residual induces an x-dependent scale signal so the log-sigma channel is
    // genuinely exercised. A fixed-seed LCG draws everything so the exact same
    // (x, exit, event) reach gam (in-Rust) and gamlss (via Rscript).
    let n = 200usize;
    let two_pi = 2.0 * std::f64::consts::PI;

    // Numerical Recipes 64-bit LCG; deterministic uniforms in [0,1).
    let mut state: u64 = 1234;
    let mut next_unit = || -> f64 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };

    let shape = 1.5_f64;
    let mut x: Vec<f64> = Vec::with_capacity(n);
    let mut exit: Vec<f64> = Vec::with_capacity(n);
    let mut event: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..n {
        let xi = -2.0 + 4.0 * next_unit(); // Uniform(-2, 2)
        // Weibull scale with strong smooth x-dependence (drives the AFT location).
        let scale = (-0.5 + 0.3 * (two_pi * xi).sin()).exp();
        // A smooth scale envelope so the log-sigma channel carries real signal.
        let scale_envelope = 1.0 + 0.4 * (two_pi * xi).cos();
        // Inverse-CDF Weibull draw: T = scale * (-ln U)^(1/shape), then warp the
        // dispersion by the envelope around the subject's own median so the
        // location stays driven by `scale` while the spread varies with x.
        let u = next_unit().max(1e-300);
        let base = scale * (-u.ln()).powf(1.0 / shape);
        let median = scale * (std::f64::consts::LN_2).powf(1.0 / shape);
        let t = (median + (base - median) * scale_envelope).max(1e-6);
        // Bernoulli(0.7) event indicator (1 = event observed, 0 = right-censored).
        let ev = if next_unit() < 0.7 { 1.0 } else { 0.0 };
        x.push(xi);
        exit.push(t);
        event.push(ev);
    }
    let entry: Vec<f64> = vec![0.0; n];

    // ---- build the gam dataset (columns: entry, exit, event, x) ------------
    let headers: Vec<String> = ["entry", "exit", "event", "x"]
        .into_iter()
        .map(str::to_string)
        .collect();
    let rows: Vec<csv::StringRecord> = (0..n)
        .map(|i| {
            csv::StringRecord::from(vec![
                format!("{:.17e}", entry[i]),
                format!("{:.17e}", exit[i]),
                format!("{:.17e}", event[i]),
                format!("{:.17e}", x[i]),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows)
        .expect("encode survival location-scale data");
    let col = ds.column_map();
    let x_idx = col["x"];
    let ncols = ds.headers.len();

    // ---- fit with gam: location ~ s(x, k=6), log-sigma ~ s(x, k=4) ---------
    // Gaussian-residual survival location-scale via the in-Rust workflow.
    let cfg = FitConfig {
        survival_likelihood: "location-scale".to_string(),
        survival_distribution: "gaussian".to_string(),
        noise_formula: Some("s(x, k=4)".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("Surv(entry, exit, event) ~ s(x, k=6)", &ds, &cfg)
        .expect("gam survival location-scale fit");
    let FitResult::SurvivalLocationScale(fit) = result else {
        panic!("expected a survival location-scale fit result");
    };
    let unified = &fit.fit.fit;
    assert!(
        unified.outer_converged,
        "gam survival location-scale outer optimizer did not converge: iters={} grad_norm={:?}",
        unified.outer_iterations, unified.outer_gradient_norm
    );

    let beta_location = unified.beta_threshold();
    let beta_log_sigma = unified.beta_log_sigma();
    assert!(
        beta_location
            .iter()
            .chain(beta_log_sigma.iter())
            .all(|v| v.is_finite()),
        "non-finite gam location / log-sigma coefficients"
    );

    // ---- evaluate gam's location & log-scale smooths on a 20-point grid ----
    // Stay strictly inside the observed x-range so neither engine extrapolates.
    let grid_n = 20usize;
    let (x_lo, x_hi) = (-1.9_f64, 1.9_f64);
    let grid_x: Vec<f64> = (0..grid_n)
        .map(|i| x_lo + (x_hi - x_lo) * i as f64 / (grid_n as f64 - 1.0))
        .collect();
    let mut grid = Array2::<f64>::zeros((grid_n, ncols));
    for (i, &t) in grid_x.iter().enumerate() {
        grid[[i, x_idx]] = t;
    }

    // Rebuild the SAME frozen location / log-sigma designs at the grid and apply
    // each channel's converged coefficients. eta_t(x) is the AFT location;
    // sigma(x) = exp(eta_ls(x)) is the AFT scale (pure exp_sigma link).
    let loc_design = build_term_collection_design(grid.view(), &fit.fit.resolved_thresholdspec)
        .expect("rebuild location (threshold) design at grid");
    let ls_design = build_term_collection_design(grid.view(), &fit.fit.resolved_log_sigmaspec)
        .expect("rebuild log-sigma design at grid");
    let gam_location: Vec<f64> = loc_design.design.apply(&beta_location).to_vec();
    let gam_log_sigma: Vec<f64> = ls_design.design.apply(&beta_log_sigma).to_vec();
    assert_eq!(gam_location.len(), grid_n);
    assert_eq!(gam_log_sigma.len(), grid_n);

    // ---- fit the SAME data with gamlss (the mature GAMLSS reference) -------
    // Normal AFT on log(time) with smooth mean mu(x) and smooth log sigma(x);
    // right-censoring via gamlss.cens (gen.cens(NO, type="right") -> NOrc),
    // response Surv(log t, event). pb() penalized B-splines mirror gam's smooths;
    // predicted on the identical x-grid. mu is the AFT location; sigma the scale.
    let grid_csv = grid_x
        .iter()
        .map(|t| format!("{t:.17e}"))
        .collect::<Vec<_>>()
        .join(",");
    let body = format!(
        r#"
        suppressPackageStartupMessages(library(gamlss))
        suppressPackageStartupMessages(library(gamlss.cens))
        suppressPackageStartupMessages(library(survival))
        gen.cens(NO, type = "right")
        df$logt <- log(df$exit)
        df$surv <- Surv(df$logt, df$event)
        m <- gamlss(surv ~ pb(x), sigma.formula = ~ pb(x), family = NOrc,
                    data = df, control = gamlss.control(trace = FALSE, n.cyc = 200))
        gx <- as.numeric(strsplit("{grid_csv}", ",")[[1]])
        nd <- data.frame(x = gx)
        mu <- predict(m, what = "mu", newdata = nd, type = "response")
        sigma <- predict(m, what = "sigma", newdata = nd, type = "response")
        emit("mu", as.numeric(mu))
        emit("sigma", as.numeric(sigma))
        "#
    );
    let r = run_r(
        &[
            Column::new("exit", &exit),
            Column::new("event", &event),
            Column::new("x", &x),
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

    // ---- compare the recovered x-dependence (gauge-free: mean-centered) ----
    // gam learns the time transform h(t); gamlss fixes it at log(t). The two
    // baselines differ by an additive offset on each channel, so center both
    // surfaces before comparing the x-dependent shape.
    let center = |v: &[f64]| -> Vec<f64> {
        let m = v.iter().sum::<f64>() / v.len() as f64;
        v.iter().map(|&z| z - m).collect()
    };
    let gam_loc_c = center(&gam_location);
    let gam_lsig_c = center(&gam_log_sigma);
    let ref_loc_c = center(gamlss_mu);
    let ref_lsig_c = center(&gamlss_log_sigma);

    let rel_loc = relative_l2(&gam_loc_c, &ref_loc_c);
    let rel_lsig = relative_l2(&gam_lsig_c, &ref_lsig_c);
    let corr_loc = pearson(&gam_loc_c, &ref_loc_c);
    let corr_lsig = pearson(&gam_lsig_c, &ref_lsig_c);

    eprintln!(
        "survival location-scale vs gamlss NOrc: n={n} grid={grid_n} \
         rel_l2(loc)={rel_loc:.4} rel_l2(log sigma)={rel_lsig:.4} \
         pearson(loc)={corr_loc:.5} pearson(log sigma)={corr_lsig:.5}"
    );

    // Guard against a vacuous pass: the data injects a strong sin(2*pi*x) signal
    // into the AFT location and a weaker cos(2*pi*x) signal into the log-scale, so
    // BOTH centered reference surfaces must carry real x-dependence. If gamlss
    // returned a near-flat surface the correlation/rel_l2 below would compare
    // noise and assert nothing; an RMS amplitude floor makes the test bite.
    let rms = |v: &[f64]| -> f64 { (v.iter().map(|z| z * z).sum::<f64>() / v.len() as f64).sqrt() };
    assert!(
        rms(&ref_loc_c) > 0.05,
        "gamlss recovered a near-flat location surface (rms={:.4}); test would be vacuous",
        rms(&ref_loc_c)
    );
    assert!(
        rms(&ref_lsig_c) > 0.02,
        "gamlss recovered a near-flat log-scale surface (rms={:.4}); test would be vacuous",
        rms(&ref_lsig_c)
    );

    // Both engines recover the same Gaussian-AFT location and log-scale
    // x-dependence; only the time-axis gauge differs (gam's flexible h(t) vs
    // gamlss's fixed log t), which the mean-centering removes.
    //
    // The two channels are NOT equally identifiable, so they get distinct,
    // separately-justified bounds (a single shared bound would be unjustified):
    //
    //   * LOCATION carries the dominant first-moment signal (0.3*sin(2*pi*x) in
    //     the Weibull scale -> the AFT mean), which both engines estimate sharply.
    //     We require near-identical shape, pearson(loc) >= 0.99, and a tight
    //     amplitude match, rel_l2(loc) <= 0.05. The residual gap above an ideal
    //     ~0.005 single-smooth mgcv bound is the genuine model difference: gam's
    //     learned monotone h(t) vs gamlss's fixed log t bends the location only
    //     mildly (the warp is ~affine over the bulk), not a tolerance artifact.
    //
    //   * LOG-SCALE is a second-moment effect (0.4*cos(2*pi*x) dispersion) seen
    //     only through the spread of n=200 *right-censored* draws, the hardest
    //     thing in distributional regression to pin down. The shape must still
    //     track, pearson(log sigma) >= 0.95, but the amplitude is allowed a wider
    //     rel_l2(log sigma) <= 0.20 because gam's Gaussian-residual-on-h(t)
    //     variance and gamlss's NO-on-log-t variance legitimately differ in scale
    //     while agreeing on where dispersion rises and falls in x.
    //
    // Exceeding any of these is a real divergence of gam's two-channel survival
    // solver / noise-formula smoothing from the GAMLSS standard, not slack.
    assert!(
        corr_loc >= 0.99,
        "location surface shape diverges from gamlss: pearson(loc)={corr_loc:.5}"
    );
    assert!(
        rel_loc <= 0.05,
        "location surface diverges from gamlss: rel_l2(loc)={rel_loc:.4}"
    );
    assert!(
        corr_lsig >= 0.95,
        "log-scale surface shape diverges from gamlss: pearson(log sigma)={corr_lsig:.5}"
    );
    assert!(
        rel_lsig <= 0.20,
        "log-scale surface diverges from gamlss: rel_l2(log sigma)={rel_lsig:.4}"
    );
}
