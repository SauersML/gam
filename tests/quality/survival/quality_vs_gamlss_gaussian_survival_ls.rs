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
//! `families::survival::location_scale` (see `survival_location_scale_response_
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
use gam::test_support::reference::{Column, QualityPair, pad_to, relative_l2, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
    load_csvwith_inferred_schema,
};
use ndarray::Array2;
use std::path::Path;

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
        survival_likelihood: Some("location-scale".to_string()),
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
    // Fit existence is the sealed convergence proof (SPEC 20).

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
    eprintln!(
        "{}",
        QualityPair::error(
            "survival",
            "quality_vs_gamlss_gaussian_survival_ls::loc",
            "rmse_to_truth",
            gam_err_loc,
            "gamlss",
            ref_err_loc,
        )
        .line()
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "survival",
            "quality_vs_gamlss_gaussian_survival_ls::log_sigma",
            "rmse_to_truth",
            gam_err_lsig,
            "gamlss",
            ref_err_lsig,
        )
        .line()
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

/// Harrell's concordance index (C-index) for a *higher-is-longer-survival*
/// risk score against right-censored survival outcomes. For every comparable
/// pair (one subject's event time is known to be strictly shorter than the
/// other's — i.e. the shorter one had an observed event), the pair is
/// concordant when the shorter-surviving subject received the lower score.
/// Ties in the score count as half. C = 0.5 is random ordering; 1.0 is a
/// perfect ranking of survival times. This is the standard objective accuracy
/// metric for survival models and needs no reference tool to compute.
fn concordance(score: &[f64], time: &[f64], event: &[f64]) -> f64 {
    assert_eq!(score.len(), time.len(), "concordance length mismatch");
    assert_eq!(score.len(), event.len(), "concordance length mismatch");
    let n = score.len();
    let mut concordant = 0.0_f64;
    let mut comparable = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            // i is the subject whose survival is known to be the shorter of the
            // pair: i had an observed event at time[i] < time[j]. (When time[j]
            // is censored we still know j outlived i; when time[j] is an event
            // strictly later, likewise.) Only such ordered pairs are comparable.
            if event[i] > 0.5 && time[i] < time[j] {
                comparable += 1.0;
                // Higher score == longer predicted survival, so the
                // shorter-surviving i should score below the longer-surviving j.
                if score[i] < score[j] {
                    concordant += 1.0;
                } else if (score[i] - score[j]).abs() == 0.0 {
                    concordant += 0.5;
                }
            }
        }
    }
    assert!(comparable > 0.0, "no comparable survival pairs");
    concordant / comparable
}

/// Real-data arm of the same survival location-scale (Gaussian-residual AFT)
/// capability. SOURCE: UCI "Heart Failure Clinical Records" dataset
/// (Chicco & Jurman, BMC Med. Inform. Decis. Mak. 2020;20:16), 299 patients,
/// `bench/datasets/heart_failure_clinical_records_dataset.csv`. `time` is the
/// follow-up period in days and `DEATH_EVENT` the death indicator
/// (1 = observed death, 0 = right-censored at last follow-up); the remaining
/// columns are clinical covariates.
///
/// Truth is unknown on real data, so we assert OBJECTIVE held-out predictive
/// accuracy, not truth recovery: a deterministic train/test split (every 4th
/// row held out), fit gam's survival location-scale AFT on the training rows,
/// predict the held-out rows' AFT *location* `eta_t(x)` (which is monotone in
/// predicted survival), and score it with Harrell's concordance index against
/// the held-out (time, event) outcomes.
///
///   PRIMARY (objective, tool-free): held-out C-index >= 0.62 — gam ranks
///     held-out survival meaningfully better than chance (0.5).
///   BASELINE (match-or-beat): gamlss NOrc (the same mature Gaussian-AFT
///     reference, fit on the IDENTICAL training rows and scored on the IDENTICAL
///     held-out rows) — gam's held-out C-index must be no worse than
///     `gamlss_cindex - 0.03`. gamlss is the accuracy bar, never the target.
#[test]
fn gam_gaussian_survival_location_scale_matches_gamlss_on_real_data() {
    init_parallelism();

    const HF_CSV: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/bench/datasets/heart_failure_clinical_records_dataset.csv"
    );
    let ds = load_csvwith_inferred_schema(Path::new(HF_CSV)).expect("load heart failure csv");
    let col = ds.column_map();
    let time_idx = col["time"];
    let event_idx = col["DEATH_EVENT"];
    let age_idx = col["age"];
    let ef_idx = col["ejection_fraction"];
    let screat_idx = col["serum_creatinine"];
    let ssod_idx = col["serum_sodium"];

    let time_all: Vec<f64> = ds.values.column(time_idx).to_vec();
    let event_all: Vec<f64> = ds.values.column(event_idx).to_vec();
    let age_all: Vec<f64> = ds.values.column(age_idx).to_vec();
    let ef_all: Vec<f64> = ds.values.column(ef_idx).to_vec();
    let screat_all: Vec<f64> = ds.values.column(screat_idx).to_vec();
    let ssod_all: Vec<f64> = ds.values.column(ssod_idx).to_vec();
    let n = time_all.len();
    assert!(n > 250, "heart failure should have ~299 rows, got {n}");

    // ---- deterministic train/test split: every 4th row held out -----------
    let is_test = |i: usize| i % 4 == 0;
    let train_rows: Vec<usize> = (0..n).filter(|&i| !is_test(i)).collect();
    let test_rows: Vec<usize> = (0..n).filter(|&i| is_test(i)).collect();
    assert!(
        train_rows.len() > 200 && test_rows.len() > 60,
        "split sizes: train={} test={}",
        train_rows.len(),
        test_rows.len()
    );

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

    // ---- fit gam on TRAIN: Gaussian-residual AFT location-scale ------------
    // Location  eta_t   ~ smooth age + smooth ejection_fraction + smooth
    //                     serum_creatinine + smooth serum_sodium.
    // Log-scale eta_ls  ~ smooth ejection_fraction (dispersion most plausibly
    //                     varies with cardiac output).
    let cfg = FitConfig {
        survival_likelihood: Some("location-scale".to_string()),
        survival_distribution: "gaussian".to_string(),
        noise_formula: Some("s(ejection_fraction, k=4)".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(
        "Surv(time, DEATH_EVENT) ~ s(age, k=5) + s(ejection_fraction, k=5) \
         + s(serum_creatinine, k=5) + s(serum_sodium, k=5)",
        &train_ds,
        &cfg,
    )
    .expect("gam survival location-scale fit on heart failure train");
    let FitResult::SurvivalLocationScale(fit) = result else {
        panic!("expected a survival location-scale fit result");
    };
    let unified = &fit.fit.fit;
    // Fit existence is the sealed convergence proof (SPEC 20).
    let beta_location = unified.beta_threshold();
    assert!(
        beta_location.iter().all(|v| v.is_finite()),
        "non-finite gam location coefficients on real data"
    );

    // ---- gam's AFT location at the held-out rows ---------------------------
    // eta_t(x) is the AFT location (monotone in predicted survival), so it is a
    // higher-is-longer-survival risk score. Rebuild the frozen location design
    // at the test covariates and apply the converged location coefficients.
    let mut test_grid = Array2::<f64>::zeros((test_rows.len(), p));
    for (out_row, &src_row) in test_rows.iter().enumerate() {
        for c in 0..p {
            test_grid[[out_row, c]] = ds.values[[src_row, c]];
        }
    }
    let loc_design =
        build_term_collection_design(test_grid.view(), &fit.fit.resolved_thresholdspec)
            .expect("rebuild location (threshold) design at held-out rows");
    let gam_test_score: Vec<f64> = loc_design.design.apply(&beta_location).to_vec();
    assert_eq!(gam_test_score.len(), test_rows.len());

    // Held-out outcomes (identical rows/order for gam, gamlss, and scoring).
    let test_time: Vec<f64> = test_rows.iter().map(|&i| time_all[i]).collect();
    let test_event: Vec<f64> = test_rows.iter().map(|&i| event_all[i]).collect();

    // ---- fit the SAME model on TRAIN with gamlss NOrc, predict the SAME TEST -
    // Normal AFT on log(time) with smooth covariates in mu (location) and a
    // smooth in log sigma; right-censoring via gamlss.cens (gen.cens(NO,
    // type="right") -> NOrc, response Surv(log t, event)). predict mu on the
    // held-out rows -> the AFT location, a higher-is-longer-survival score
    // directly comparable to gam's. Train- and test-length vectors must not be
    // mixed in one call, so the test covariates ride along as padded columns
    // and only their first `test_n` entries are read back.
    let m_train = train_rows.len();
    let train_time: Vec<f64> = train_rows.iter().map(|&i| time_all[i]).collect();
    let train_event: Vec<f64> = train_rows.iter().map(|&i| event_all[i]).collect();
    let train_age: Vec<f64> = train_rows.iter().map(|&i| age_all[i]).collect();
    let train_ef: Vec<f64> = train_rows.iter().map(|&i| ef_all[i]).collect();
    let train_screat: Vec<f64> = train_rows.iter().map(|&i| screat_all[i]).collect();
    let train_ssod: Vec<f64> = train_rows.iter().map(|&i| ssod_all[i]).collect();
    let test_age: Vec<f64> = test_rows.iter().map(|&i| age_all[i]).collect();
    let test_ef: Vec<f64> = test_rows.iter().map(|&i| ef_all[i]).collect();
    let test_screat: Vec<f64> = test_rows.iter().map(|&i| screat_all[i]).collect();
    let test_ssod: Vec<f64> = test_rows.iter().map(|&i| ssod_all[i]).collect();

    let r = run_r(
        &[
            Column::new("time", &train_time),
            Column::new("event", &train_event),
            Column::new("age", &train_age),
            Column::new("ejection_fraction", &train_ef),
            Column::new("serum_creatinine", &train_screat),
            Column::new("serum_sodium", &train_ssod),
            Column::new("test_age", &pad_to(&test_age, m_train)),
            Column::new("test_ef", &pad_to(&test_ef, m_train)),
            Column::new("test_screat", &pad_to(&test_screat, m_train)),
            Column::new("test_ssod", &pad_to(&test_ssod, m_train)),
            Column::new("test_n", &vec![test_rows.len() as f64; m_train]),
        ],
        r#"
        suppressPackageStartupMessages(library(gamlss))
        suppressPackageStartupMessages(library(gamlss.cens))
        suppressPackageStartupMessages(library(survival))
        gen.cens(NO, type = "right")
        df$logt <- log(df$time)
        df$surv <- Surv(df$logt, df$event)
        m <- gamlss(surv ~ pb(age) + pb(ejection_fraction) + pb(serum_creatinine) + pb(serum_sodium),
                    sigma.formula = ~ pb(ejection_fraction), family = NOrc,
                    data = df, control = gamlss.control(trace = FALSE, n.cyc = 200))
        k <- df$test_n[1]
        nd <- data.frame(age = df$test_age[1:k],
                         ejection_fraction = df$test_ef[1:k],
                         serum_creatinine = df$test_screat[1:k],
                         serum_sodium = df$test_ssod[1:k])
        mu <- predict(m, what = "mu", newdata = nd, type = "response")
        emit("mu", as.numeric(mu))
        "#,
    );
    let gamlss_score = r.vector("mu");
    assert_eq!(
        gamlss_score.len(),
        test_rows.len(),
        "gamlss held-out mu length mismatch"
    );

    // ---- OBJECTIVE held-out accuracy: concordance on identical test rows ----
    let gam_cindex = concordance(&gam_test_score, &test_time, &test_event);
    let gamlss_cindex = concordance(gamlss_score, &test_time, &test_event);

    // Context only (NOT a pass criterion): closeness of the two score rankings.
    let rel_vs_gamlss = relative_l2(
        &{
            let mu = gam_test_score.iter().sum::<f64>() / gam_test_score.len() as f64;
            gam_test_score.iter().map(|&z| z - mu).collect::<Vec<_>>()
        },
        &{
            let mu = gamlss_score.iter().sum::<f64>() / gamlss_score.len() as f64;
            gamlss_score.iter().map(|&z| z - mu).collect::<Vec<_>>()
        },
    );

    eprintln!(
        "heart-failure survival LS held-out: n_train={m_train} n_test={} \
         gam_cindex={gam_cindex:.4} gamlss_cindex={gamlss_cindex:.4} \
         (context: centered-score rel_l2 vs gamlss={rel_vs_gamlss:.4})",
        test_rows.len()
    );

    // ---- PRIMARY objective assertion: gam ranks held-out survival ----------
    // 0.5 is chance ordering; clinical covariates carry real prognostic signal,
    // so a competent AFT must clear 0.62 on the held-out fold.
    assert!(
        gam_cindex >= 0.62,
        "gam's held-out concordance too low: {gam_cindex:.4} (< 0.62)"
    );

    // ---- BASELINE (match-or-beat): no worse than gamlss on the SAME C-index -
    assert!(
        gam_cindex >= gamlss_cindex - 0.03,
        "gam held-out concordance {gam_cindex:.4} worse than gamlss \
         {gamlss_cindex:.4} - 0.03"
    );
}
