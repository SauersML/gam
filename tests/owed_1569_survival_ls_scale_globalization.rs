//! gam#1569 regression: the coupled smooth-scale survival location-scale
//! joint-Newton globalization must converge a HARD heteroscedastic configuration
//! — sharper log-σ variation, with a region of genuinely small σ so the time
//! block's `exp(−η_σ)` factor is large — WITHOUT stalling / burning an excessive
//! number of inner joint-Newton cycles, and it must RECOVER the true two
//! x-dependent surfaces baked into the synthetic right-censored data.
//!
//! Background (commit e3da155e4, "fix(survival-LS): stop residualizing the log-σ
//! design against the location design"): the log-σ design is now kept raw so the
//! heteroscedastic signal is preserved. That fix deferred the DEEPER problem to
//! this issue: in the coupled smooth-scale solve the scale predictor η_σ is free,
//! so `exp(−η_σ)` can inflate the time-block gradient/curvature and degrade the
//! joint-Newton globalization (trust-region step quality) for HARDER
//! heteroscedastic regimes than the existing gamlss-oracle gate
//! (`gam_gaussian_survival_location_scale_matches_gamlss`) exercises.
//!
//! This is an R-FREE, oracle-free truth-recovery reproduction (so it runs to
//! completion anywhere): the data is generated from a known recipe, so each
//! x-dependent channel has a closed-form truth in the gauge-free, mean-centered
//! space that the unknown time-axis baseline leaves invariant. We assert BOTH:
//!   * the inner joint-Newton does not stall (bounded `inner_cycles`), AND
//!   * gam recovers the true location and log-σ surfaces within a principled bar.
//!
//! BEFORE the globalization fix this regime makes the time-block gradient blow up
//! on the small-σ rows; the metric-whitened trust-region step is mis-scaled on
//! the time block, the gain ratio never justifies growing the radius, and the
//! inner solve grinds (excessive `inner_cycles`) and/or misses the true surface.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;

#[test]
fn gam_survival_location_scale_hard_heteroscedastic_globalization_converges() {
    init_parallelism();

    // ---- synthetic right-censored recipe with STRONG heteroscedasticity ----
    // n = 240, x ~ Uniform(-2, 2), entry = 0.
    // Location truth (AFT location, drives the Weibull scale):
    //     loc(x) = 0.5 * sin(2*pi*x)        (sharper than the gate's 0.3)
    // Log-scale truth (dispersion envelope):
    //     log_sigma(x) = 0.9 * cos(pi*x/2)  (amplitude 0.9 ⇒ σ ∈ [~0.41, ~2.46];
    //     the small-σ region is exactly where exp(−η_σ) is large and inflates the
    //     time-block gradient — the globalization stressor this issue targets).
    // event ~ Bernoulli(0.75). A fixed-seed LCG draws everything deterministically.
    let n = 240usize;
    let two_pi = 2.0 * std::f64::consts::PI;
    let half_pi = 0.5 * std::f64::consts::PI;

    // Numerical Recipes 64-bit LCG; deterministic uniforms in [0,1).
    let mut state: u64 = 0x1569_2026_0627_0001;
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
        let scale = (-0.4 + 0.5 * (two_pi * xi).sin()).exp();
        // STRONG smooth scale envelope: multiplicative dispersion exp(log_sigma(x)),
        // with log_sigma amplitude 0.9 (the small-σ region is the stressor).
        let log_sigma = 0.9 * (half_pi * xi).cos();
        let disp = log_sigma.exp();
        // Inverse-CDF Weibull draw, then warp the dispersion around the subject's
        // own median so the location stays driven by `scale` while the spread
        // varies sharply with x.
        let u = next_unit().max(1e-300);
        let base = scale * (-u.ln()).powf(1.0 / shape);
        let median = scale * (std::f64::consts::LN_2).powf(1.0 / shape);
        let t = (median + (base - median) * disp).max(1e-6);
        let ev = if next_unit() < 0.75 { 1.0 } else { 0.0 };
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

    // ---- fit: location ~ s(x, k=8), log-sigma ~ s(x, k=6) (richer than the
    // gate so the harder surfaces are representable) ------------------------
    let cfg = FitConfig {
        survival_likelihood: "location-scale".to_string(),
        survival_distribution: "gaussian".to_string(),
        noise_formula: Some("s(x, k=6)".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("Surv(entry, exit, event) ~ s(x, k=8)", &ds, &cfg)
        .expect("gam survival location-scale fit (hard heteroscedastic)");
    let FitResult::SurvivalLocationScale(fit) = result else {
        panic!("expected a survival location-scale fit result");
    };
    let unified = &fit.fit.fit;

    eprintln!(
        "[#1569 repro] outer_converged={} outer_iterations={} inner_cycles={} grad_norm={:?}",
        unified.outer_converged,
        unified.outer_iterations,
        unified.inner_cycles,
        unified.outer_gradient_norm,
    );

    // ---- GLOBALIZATION assertion: bounded inner joint-Newton effort --------
    // The healthy coupled smooth-scale solve certifies stationarity in a modest
    // number of inner joint-Newton cycles. With `exp(−η_σ)` inflating the
    // time-block gradient on the small-σ rows, the pre-fix globalization grinds:
    // the trust-region gain ratio never justifies radius growth and the inner
    // loop burns hundreds of cycles (or fails to converge outright). The
    // observed-healthy count after the fix is well under this ceiling; the bar is
    // a generous multiple so it only trips on a genuine stall, never on benign
    // iteration-count drift.
    const INNER_CYCLE_CEILING: usize = 250;
    assert!(
        unified.outer_converged,
        "gam hard-heteroscedastic survival LS did not converge: \
         inner_cycles={} outer_iterations={} grad_norm={:?}",
        unified.inner_cycles, unified.outer_iterations, unified.outer_gradient_norm
    );
    assert!(
        unified.inner_cycles <= INNER_CYCLE_CEILING,
        "gam hard-heteroscedastic survival LS joint-Newton stalled: \
         inner_cycles={} > {INNER_CYCLE_CEILING} (the exp(−η_σ) time-block \
         gradient inflation degraded the trust-region step quality)",
        unified.inner_cycles
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

    // ---- evaluate gam's location & log-scale smooths on a 24-point grid ----
    let grid_n = 24usize;
    let (x_lo, x_hi) = (-1.9_f64, 1.9_f64);
    let grid_x: Vec<f64> = (0..grid_n)
        .map(|i| x_lo + (x_hi - x_lo) * i as f64 / (grid_n as f64 - 1.0))
        .collect();
    let mut grid = Array2::<f64>::zeros((grid_n, ncols));
    for (i, &t) in grid_x.iter().enumerate() {
        grid[[i, x_idx]] = t;
    }
    let loc_design = build_term_collection_design(grid.view(), &fit.fit.resolved_thresholdspec)
        .expect("rebuild location (threshold) design at grid");
    let ls_design = build_term_collection_design(grid.view(), &fit.fit.resolved_log_sigmaspec)
        .expect("rebuild log-sigma design at grid");
    let gam_location: Vec<f64> = loc_design.design.apply(&beta_location).to_vec();
    let gam_log_sigma: Vec<f64> = ls_design.design.apply(&beta_log_sigma).to_vec();
    assert_eq!(gam_location.len(), grid_n);
    assert_eq!(gam_log_sigma.len(), grid_n);

    // ---- KNOWN generating truth on the grid (gauge-free: mean-centered) ----
    let center = |v: &[f64]| -> Vec<f64> {
        let m = v.iter().sum::<f64>() / v.len() as f64;
        v.iter().map(|&z| z - m).collect()
    };
    let rmse = |a: &[f64], b: &[f64]| -> f64 {
        let s: f64 = a.iter().zip(b).map(|(p, q)| (p - q) * (p - q)).sum();
        (s / a.len() as f64).sqrt()
    };
    let truth_location: Vec<f64> = grid_x.iter().map(|&xi| 0.5 * (two_pi * xi).sin()).collect();
    let truth_log_sigma: Vec<f64> = grid_x.iter().map(|&xi| 0.9 * (half_pi * xi).cos()).collect();

    let gam_loc_c = center(&gam_location);
    let gam_lsig_c = center(&gam_log_sigma);
    let truth_loc_c = center(&truth_location);
    let truth_lsig_c = center(&truth_log_sigma);

    let gam_err_loc = rmse(&gam_loc_c, &truth_loc_c);
    let gam_err_lsig = rmse(&gam_lsig_c, &truth_lsig_c);

    eprintln!(
        "[#1569 repro] truth recovery: rmse_loc={gam_err_loc:.4} rmse_logsig={gam_err_lsig:.4}"
    );

    // ---- truth-recovery assertion (principled bars tied to signal RMS) -----
    // LOCATION truth 0.5*sin(2*pi*x): centered RMS amplitude ~0.35 on the grid.
    // Estimated sharply by a k=8 smooth from n=240 (75% observed); require the
    // recovery error to be a small fraction of that amplitude.
    //   bar 0.10 ≈ 29% of signal RMS.
    // LOG-SCALE truth 0.9*cos(pi*x/2): centered RMS amplitude ~0.40, a
    // second-moment effect seen only through the spread of right-censored draws —
    // the hardest channel; a k=6 smooth recovers the shape with more variance.
    //   bar 0.22 ≈ 55% of signal RMS.
    let loc_bar = 0.10;
    let lsig_bar = 0.22;
    assert!(
        gam_err_loc <= loc_bar,
        "gam failed to recover the true AFT location 0.5*sin(2*pi*x): \
         rmse(loc)={gam_err_loc:.4} > {loc_bar}"
    );
    assert!(
        gam_err_lsig <= lsig_bar,
        "gam failed to recover the true log-scale 0.9*cos(pi*x/2): \
         rmse(log sigma)={gam_err_lsig:.4} > {lsig_bar}"
    );
}
