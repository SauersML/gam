//! End-to-end quality: gam's *survival* location-scale family (a smooth
//! location AND a smooth log-scale, with a flexible monotone time baseline and
//! a Gaussian residual) must RECOVER THE TRUE two x-dependent surfaces baked
//! into the synthetic right-censored data — not merely reproduce whatever
//! `gamlss` happens to fit.
//!
//! OBJECTIVE METRIC ASSERTED (truth recovery). The data is generated from a
//! known recipe, so both x-dependent channels have a closed-form truth (in the
//! gauge-free, mean-centered space that the unknown time-axis baseline leaves
//! invariant):
//!   * LOCATION truth: the AFT location tracks the Weibull log-scale, whose
//!     x-dependent part is `0.3 * sin(2*pi*x)`. Centered over the grid this is
//!     the location signal gam must recover.
//!   * LOG-SCALE truth: the residual spread is multiplied by
//!     `scale_envelope(x) = 1 + 0.4*cos(2*pi*x)`, so the x-dependent part of the
//!     log-sigma channel is `log(1 + 0.4*cos(2*pi*x))`, centered over the grid.
//! The PRIMARY claim is `RMSE(gam_centered_fit, centered_truth)` is below a
//! principled bar on each channel. gamlss is DEMOTED to a baseline-to-beat:
//! gam's truth-recovery error must be no worse than gamlss's error * 1.10 on the
//! SAME truth (match-or-beat on accuracy), never "gam reproduces gamlss".
//!
//! Why gamlss (family = NO) is the right baseline. gam's survival
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
//! type = "right")` → family `NOrc`, response `Surv(log t, event)`). It is the
//! mature distributional-regression standard, so it is the natural yardstick for
//! "is gam at least as accurate as the field's best tool?"
//!
//! The cross-feature combination this pins down (which single-channel survival
//! tests never exercise): survival x smooth-covariate-in-location x
//! smooth-covariate-in-scale, fit jointly. The data has *strong x-dependence in
//! both channels* (the Weibull scale, hence the AFT location, moves with
//! sin(2*pi*x); a cos(2*pi*x) envelope drives the dispersion). We feed the
//! byte-identical (x, exit, event) rows to both engines and compare each one's
//! recovered smooth, on a 20-point x-grid, against the KNOWN generating signal.
//!
//! Gauge handling. gam learns the time transform `h(t)` flexibly while gamlss
//! fixes it at `log(t)`; the two baselines differ by an additive offset per
//! channel. The closed-form truth is itself only identified up to that additive
//! offset, so we mean-center fit, baseline, and truth on the grid before scoring
//! — the comparison is then a clean accuracy test against the generating signal,
//! not a shared-gauge artifact.
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
use gam::test_support::reference::{Column, relative_l2, rmse, run_r};
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

    // ---- KNOWN generating truth on the grid (gauge-free: mean-centered) ----
    // The recipe fixes both x-dependent surfaces in closed form:
    //   * AFT location  <- Weibull log-scale x-part:        0.3*sin(2*pi*x)
    //   * AFT log-sigma <- dispersion envelope x-part:  log(1 + 0.4*cos(2*pi*x))
    // gam's flexible h(t) and gamlss's fixed log(t) each add an unknown per-
    // channel offset, and the truth itself is only identified up to that offset,
    // so we mean-center everything on the grid before scoring against the truth.
    let center = |v: &[f64]| -> Vec<f64> {
        let m = v.iter().sum::<f64>() / v.len() as f64;
        v.iter().map(|&z| z - m).collect()
    };
    let truth_location: Vec<f64> = grid_x.iter().map(|&xi| 0.3 * (two_pi * xi).sin()).collect();
    let truth_log_sigma: Vec<f64> = grid_x
        .iter()
        .map(|&xi| (1.0 + 0.4 * (two_pi * xi).cos()).ln())
        .collect();

    let gam_loc_c = center(&gam_location);
    let gam_lsig_c = center(&gam_log_sigma);
    let ref_loc_c = center(gamlss_mu);
    let ref_lsig_c = center(&gamlss_log_sigma);
    let truth_loc_c = center(&truth_location);
    let truth_lsig_c = center(&truth_log_sigma);

    // Truth-recovery error (PRIMARY metric) for gam and for the gamlss baseline.
    let gam_err_loc = rmse(&gam_loc_c, &truth_loc_c);
    let gam_err_lsig = rmse(&gam_lsig_c, &truth_lsig_c);
    let ref_err_loc = rmse(&ref_loc_c, &truth_loc_c);
    let ref_err_lsig = rmse(&ref_lsig_c, &truth_lsig_c);

    // Context only (NOT a pass criterion): how close gam sits to the gamlss fit.
    let rel_loc_vs_ref = relative_l2(&gam_loc_c, &ref_loc_c);
    let rel_lsig_vs_ref = relative_l2(&gam_lsig_c, &ref_lsig_c);

    eprintln!(
        "survival location-scale truth recovery: n={n} grid={grid_n} \
         rmse_loc(gam)={gam_err_loc:.4} rmse_loc(gamlss)={ref_err_loc:.4} \
         rmse_logsig(gam)={gam_err_lsig:.4} rmse_logsig(gamlss)={ref_err_lsig:.4} \
         [context rel_l2(loc vs gamlss)={rel_loc_vs_ref:.4} \
         rel_l2(log sigma vs gamlss)={rel_lsig_vs_ref:.4}]"
    );

    // ---- PRIMARY assertion: gam recovers the true x-dependence -------------
    // The two channels are NOT equally identifiable, so the absolute bars differ
    // and are tied to the generating signal's own amplitude on the grid:
    //
    //   * LOCATION is the dominant first-moment signal: 0.3*sin(2*pi*x), whose
    //     centered RMS amplitude over the grid is ~0.21. Estimated sharply by a
    //     k=6 smooth from n=200 (70% observed) draws, so we require the recovery
    //     error to be a small fraction of that amplitude: rmse(loc) <= 0.06
    //     (~28% of signal RMS). Bigger than that means gam failed to recover the
    //     true location curve, not a gauge artifact.
    //
    //   * LOG-SCALE is a second-moment effect: log(1 + 0.4*cos(2*pi*x)), centered
    //     RMS amplitude ~0.29, seen only through the spread of right-censored
    //     draws — the hardest thing in distributional regression to pin down. A
    //     k=4 smooth on n=200 censored points recovers the shape but with more
    //     variance, so a wider absolute bar rmse(log sigma) <= 0.15 (~half the
    //     signal RMS) is the principled accuracy floor here.
    let loc_bar = 0.06;
    let lsig_bar = 0.15;
    assert!(
        gam_err_loc <= loc_bar,
        "gam failed to recover the true AFT location 0.3*sin(2*pi*x): \
         rmse(loc)={gam_err_loc:.4} > {loc_bar}"
    );
    assert!(
        gam_err_lsig <= lsig_bar,
        "gam failed to recover the true log-scale log(1+0.4*cos(2*pi*x)): \
         rmse(log sigma)={gam_err_lsig:.4} > {lsig_bar}"
    );

    // ---- SECONDARY assertion: match-or-beat the gamlss baseline ------------
    // On the IDENTICAL data and the SAME known truth, gam's recovery error must
    // be no worse than the mature GAMLSS tool's, within a 10% tolerance. This
    // demotes gamlss from "the thing gam must reproduce" to "the accuracy bar gam
    // must match or beat" — a real quality claim, not a same-as-peer claim.
    assert!(
        gam_err_loc <= ref_err_loc * 1.10,
        "gam location recovery worse than gamlss baseline: \
         rmse(gam)={gam_err_loc:.4} > 1.10 * rmse(gamlss)={ref_err_loc:.4}"
    );
    assert!(
        gam_err_lsig <= ref_err_lsig * 1.10,
        "gam log-scale recovery worse than gamlss baseline: \
         rmse(gam)={gam_err_lsig:.4} > 1.10 * rmse(gamlss)={ref_err_lsig:.4}"
    );
}
