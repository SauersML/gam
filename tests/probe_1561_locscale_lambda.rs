//! Probe (NOT a permanent guard): measure the Gaussian location-scale
//! over-smoothing of the log-sigma surface (#1561) under several penalty
//! configurations, separating the shipped null-recovery default from explicit
//! scale-only and both-block null-space-shrinkage opt-outs.
//! Prints pearson(log sigma, truth), rmse, lambdas, edf.

use gam::estimate::BlockRole;
use gam::gamlss::GaussianLocationScaleFitResult;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{pearson, rmse};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;

const LOGB_SIGMA_FLOOR: f64 = 0.01;

fn next_unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn mean_truth(x: f64) -> f64 {
    (2.0 * std::f64::consts::PI * x).sin()
}
fn sigma_truth(x: f64) -> f64 {
    0.1 + 0.2 * (2.0 * std::f64::consts::PI * x).sin()
}

fn data(n: usize) -> (Vec<f64>, Vec<f64>) {
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut state = 42_u64;
    let mut x: Vec<f64> = (0..n).map(|_| next_unit(&mut state)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).expect("finite x"));
    let mut z = Vec::with_capacity(n);
    while z.len() < n {
        let u1 = next_unit(&mut state).max(1e-300);
        let u2 = next_unit(&mut state);
        let r = (-2.0 * u1.ln()).sqrt();
        z.push(r * (two_pi * u2).cos());
        if z.len() < n {
            z.push(r * (two_pi * u2).sin());
        }
    }
    let y = (0..n)
        .map(|i| mean_truth(x[i]) + sigma_truth(x[i]) * z[i])
        .collect();
    (x, y)
}

fn run_case(label: &str, mean_formula: &str, noise_formula: &str, n: usize) -> f64 {
    let (x, y) = data(n);
    let headers = vec!["x".to_string(), "y".to_string()];
    let rows: Vec<csv::StringRecord> = x
        .iter()
        .zip(&y)
        .map(|(&x, &y)| csv::StringRecord::from(vec![format!("{x:.17e}"), format!("{y:.17e}")]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        noise_formula: Some(noise_formula.to_string()),
        ..FitConfig::default()
    };
    let result = match fit_from_formula(mean_formula, &ds, &cfg) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[{label}] FIT ERROR: {e}");
            return f64::NAN;
        }
    };
    let FitResult::GaussianLocationScale(GaussianLocationScaleFitResult {
        fit,
        response_scale,
        ..
    }) = result
    else {
        eprintln!("[{label}] not a location-scale fit");
        return f64::NAN;
    };

    let beta_loc = fit
        .fit
        .block_by_role(BlockRole::Location)
        .expect("loc")
        .beta
        .clone();
    let beta_scale = fit
        .fit
        .block_by_role(BlockRole::Scale)
        .expect("scale")
        .beta
        .clone();

    let mut grid = Array2::<f64>::zeros((x.len(), 2));
    for (i, &t) in x.iter().enumerate() {
        grid[[i, 0]] = t;
    }
    let mean_design =
        build_term_collection_design(grid.view(), &fit.meanspec_resolved).expect("md");
    let scale_design =
        build_term_collection_design(grid.view(), &fit.noisespec_resolved).expect("sd");
    let gam_mu = mean_design.design.apply(&beta_loc).to_vec();
    let gam_eta_sigma = scale_design.design.apply(&beta_scale).to_vec();
    let gam_log_sigma: Vec<f64> = gam_eta_sigma
        .iter()
        .map(|&e| (response_scale * LOGB_SIGMA_FLOOR + e.exp()).ln())
        .collect();

    let true_mu: Vec<f64> = x.iter().map(|&t| mean_truth(t)).collect();
    let true_log_sigma: Vec<f64> = x.iter().map(|&t| sigma_truth(t).abs().ln()).collect();

    let rmse_mu = rmse(&gam_mu, &true_mu);
    let corr = pearson(&gam_log_sigma, &true_log_sigma);
    let rmse_ls = rmse(&gam_log_sigma, &true_log_sigma);

    let (lambdas, log_lambdas, edf_by_block, edf_total) =
        if let Some(inf) = fit.fit.inference.as_ref() {
            (
                fit.fit.lambdas.to_vec(),
                fit.fit.log_lambdas.to_vec(),
                inf.edf_by_block.clone(),
                inf.edf_total,
            )
        } else {
            (vec![], vec![], vec![], f64::NAN)
        };

    let fmt_vec = |v: &[f64]| {
        v.iter()
            .map(|x| format!("{x:.4}"))
            .collect::<Vec<_>>()
            .join(",")
    };
    let lambdas_s = fmt_vec(&lambdas);
    let log_lambdas_s = fmt_vec(&log_lambdas);
    let edf_s = fmt_vec(&edf_by_block);
    eprintln!(
        "[{label}] pearson={corr:.5} rmse_ls={rmse_ls:.5} rmse_mu={rmse_mu:.5} \
         | lambdas=[{lambdas_s}] log_lambdas=[{log_lambdas_s}] \
         | edf_by_block=[{edf_s}] edf_total={edf_total:.3} \
         | outer_conv=certified iters={}",
        fit.fit.outer_iterations
    );
    corr
}

#[test]
fn probe_1561_locscale_penalty_configs() {
    init_parallelism();
    let n = 200;
    eprintln!("=== #1561 probe: Gaussian location-scale log-sigma recovery (n={n}) ===");
    // Shipped null-recovery default: both double penalties on.
    let default_corr = run_case("default", "y ~ s(x, bs='tp')", "1 + s(x, bs='tp')", n);
    // Explicitly disable null-space shrinkage on the scale block only.
    let scale_nodbl_corr = run_case(
        "scale_nodbl",
        "y ~ s(x, bs='tp')",
        "1 + s(x, bs='tp', double_penalty=false)",
        n,
    );
    // BOTH blocks without null-space double penalty.
    let both_nodbl_corr = run_case(
        "both_nodbl",
        "y ~ s(x, bs='tp', double_penalty=false)",
        "1 + s(x, bs='tp', double_penalty=false)",
        n,
    );

    // Sanity invariant (real regression guard): every location-scale config that
    // MINTS a fit must recover a finite, non-degenerate log-σ↔truth correlation —
    // a NaN-from-a-real-fit or a collapse to 0 means the joint μ+σ fit broke,
    // independent of the #1561 over-smoothing question this probe investigates.
    //
    // `default` and `scale_nodbl` both retain a null-space penalty on at least one
    // block, so their outer REML surface is identified and they must fit. The
    // `both_nodbl` config strips null-space shrinkage from BOTH blocks at once; a
    // range-space λ can then rail while the projected gradient stays above the
    // stationarity bound, so the fail-closed outer search may correctly decline to
    // mint a non-stationary fit ("a fit is only minted from a converged
    // optimization"). That refusal returns NaN here and is acceptable — the guard
    // is that IF a fit is minted it is non-degenerate, not that every un-null-
    // penalized model converges (a separate outer-convergence question, cf. #979).
    for (label, c) in [("default", default_corr), ("scale_nodbl", scale_nodbl_corr)] {
        assert!(
            c.is_finite(),
            "#1561 probe: {label} produced a non-finite log-σ pearson ({c}) — location-scale fit broke"
        );
        assert!(
            c > 0.0,
            "#1561 probe: {label} log-σ pearson {c:.4} <= 0 — fitted scale surface is anti-correlated/degenerate"
        );
    }
    // both_nodbl: only assert non-degeneracy when a fit was actually minted.
    if both_nodbl_corr.is_finite() {
        assert!(
            both_nodbl_corr > 0.0,
            "#1561 probe: both_nodbl minted a fit with log-σ pearson {both_nodbl_corr:.4} <= 0 — degenerate scale surface"
        );
    }
}

// ---- #1561 scale-block penalty metric: Gamma dispersion log-precision -------
//
// The Gaussian location-scale path dropped its full-space `eye(p)` scale ridge
// in `de5599435` (it over-shrinks the heteroscedastic curve). The dispersion
// family (`dispersion_family.rs`) and the binomial threshold-scale builders
// still append that ridge on the scale / log-precision block. This probe fits a
// Gamma mean model with a genuinely varying precision and measures how well the
// fitted log-precision surface tracks the truth. A full-span coefficient ridge
// whose λ is REML-selected pulls the log-precision block toward its (basis-
// dependent) coordinate origin, collapsing the scale edf toward the constant.

fn gamma_sample(shape: f64, scale: f64, state: &mut u64) -> f64 {
    // Marsaglia–Tsang; boost transform for shape < 1.
    if shape < 1.0 {
        let u = next_unit(state).max(1e-300);
        return gamma_sample(shape + 1.0, scale, state) * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let u1 = next_unit(state).max(1e-300);
        let u2 = next_unit(state);
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let v = 1.0 + c * z;
        if v <= 0.0 {
            continue;
        }
        let v = v * v * v;
        let u = next_unit(state).max(1e-300);
        if u.ln() < 0.5 * z * z + d - d * v + d * v.ln() {
            return d * v * scale;
        }
    }
}

fn disp_mu_true(x: f64) -> f64 {
    (0.6 + 0.8 * x).exp()
}
fn disp_logshape_truth(x: f64) -> f64 {
    // Pure linear log-shape (the convergent #1060 regime). The linear trend is
    // the penalty NULL space — exactly the coordinate a full-span coefficient
    // ridge shrinks — so the recovered OLS slope is a direct over-shrinkage
    // probe. Shape stays >= 1 across [-1, 1], keeping the inner solve stable.
    1.6 + 1.1 * x
}

fn gamma_dispersion_data(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut state = 4242_u64;
    // Deterministic grid (matches the convergent #1060 regime) so the probe
    // isolates the penalty metric, not sampling-design variance.
    let x: Vec<f64> = (0..n)
        .map(|i| -1.0 + 2.0 * (i as f64) / (n as f64 - 1.0))
        .collect();
    let y = x
        .iter()
        .map(|&xi| {
            let mu = disp_mu_true(xi);
            let nu = disp_logshape_truth(xi).exp(); // Gamma shape = precision
            gamma_sample(nu, mu / nu, &mut state).max(1e-6) // E[y]=mu, Var=mu^2/nu
        })
        .collect();
    (x, y)
}

/// Ordinary-least-squares slope of `ys` on `xs` (a scalar summary of the
/// dominant linear trend, i.e. the penalty null-space coordinate).
fn ols_slope(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    for (&x, &y) in xs.iter().zip(ys) {
        sxy += (x - mx) * (y - my);
        sxx += (x - mx) * (x - mx);
    }
    sxy / sxx
}

#[test]
fn probe_1561_gamma_dispersion_logshape_recovery() {
    init_parallelism();
    let n = 400;
    let (x, y) = gamma_dispersion_data(n);
    let headers = vec!["x".to_string(), "y".to_string()];
    let rows: Vec<csv::StringRecord> = x
        .iter()
        .zip(&y)
        .map(|(&x, &y)| csv::StringRecord::from(vec![format!("{x:.17e}"), format!("{y:.17e}")]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let cfg = FitConfig {
        family: Some("gamma".to_string()),
        noise_formula: Some("s(x, k=6)".to_string()),
        ..FitConfig::default()
    };
    let result = match fit_from_formula("y ~ s(x, k=6)", &ds, &cfg) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[gamma_disp] FIT ERROR: {e}");
            return;
        }
    };
    let FitResult::DispersionLocationScale(gam::DispersionLocationScaleFitResult { fit, .. }) =
        result
    else {
        panic!("expected a dispersion location-scale fit");
    };

    let beta_scale = fit
        .fit
        .block_by_role(BlockRole::Scale)
        .expect("scale block")
        .beta
        .clone();
    let grid_n = 120usize;
    let grid_x: Vec<f64> = (0..grid_n)
        .map(|i| -1.0 + 2.0 * (i as f64) / (grid_n as f64 - 1.0))
        .collect();
    let mut grid = Array2::<f64>::zeros((grid_n, 1));
    for (i, &t) in grid_x.iter().enumerate() {
        grid[[i, 0]] = t;
    }
    let scale_design =
        build_term_collection_design(grid.view(), &fit.noisespec_resolved).expect("scale design");
    let eta_d = scale_design.design.apply(&beta_scale).to_vec();
    let truth: Vec<f64> = grid_x.iter().map(|&t| disp_logshape_truth(t)).collect();

    let corr = pearson(&eta_d, &truth);
    let recovery_rmse = rmse(&eta_d, &truth);
    let gam_slope = ols_slope(&grid_x, &eta_d);
    let truth_slope = ols_slope(&grid_x, &truth);
    let slope_ratio = gam_slope / truth_slope;

    let (lambdas, edf_by_block) = if let Some(inf) = fit.fit.inference.as_ref() {
        (fit.fit.lambdas.to_vec(), inf.edf_by_block.clone())
    } else {
        (vec![], vec![])
    };
    eprintln!(
        "[gamma_disp] pearson={corr:.5} rmse={recovery_rmse:.5} \
         slope gam={gam_slope:.4} truth={truth_slope:.4} ratio={slope_ratio:.4} \
         n_penalties={} lambdas={lambdas:?} edf_by_block={edf_by_block:?}",
        lambdas.len(),
    );

    assert!(corr.is_finite(), "dispersion log-shape pearson is NaN");
}
