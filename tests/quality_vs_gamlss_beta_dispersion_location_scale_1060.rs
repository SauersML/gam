//! #1060: OBJECTIVE truth-recovery of the GAMLSS Beta *dispersion-channel*
//! location-scale fit — the precision phi is modelled with its own smooth
//! formula via `noise_formula`.
//!
//! Data-generating law (KNOWN): for covariate x on a grid,
//!   mu_true(x)  = logistic(0.2 + 0.9*x)              (logit mean link, in (0,1))
//!   phi_true(x) = exp(2.2 + 1.0*x)   = phi(x)        (Beta precision varies)
//!   y ~ Beta(a = mu*phi, b = (1-mu)*phi)
//! gam's Beta uses Var = mu(1-mu)/(1+phi); phi rises with x so the dispersion
//! shrinks with x — a genuine precision-channel signal a mean-only Beta GLM
//! cannot represent.
//!
//! gam's dispersion channel models `log(phi)` directly: phi_gam = exp(eta_d)
//! with a logit mean link. gamlss `BE()` parameterizes (mu, sigma) with
//! Var = mu(1-mu)*sigma^2, sigma in (0,1); matching gam's Var gives
//! sigma^2 = 1/(1+phi), i.e. phi_gamlss = (1-sigma^2)/sigma^2. Both engines are
//! scored on the SAME log-phi surface vs the SAME known truth.
//!
//! Objective assertions (none is "match the reference"):
//!   1. TRUTH RECOVERY (mean): RMSE(logit mu_gam, logit mu_true) small.
//!   2. TRUTH RECOVERY (dispersion): RMSE(log phi_gam, log phi_true) bounded AND
//!      positively correlated with the truth (real, correctly-signed signal).
//!   3. MATCH-OR-BEAT: gam's dispersion-channel and mean-channel RMSE each
//!      <= 1.30 * gamlss's. gamlss is a demoted baseline.

use gam::estimate::BlockRole;
use gam::gamlss::DispersionFamilyKind;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, rmse, run_r};
use gam::{
    DispersionLocationScaleFitResult, FitConfig, FitResult, encode_recordswith_inferred_schema,
    fit_from_formula, init_parallelism,
};
use ndarray::Array2;

/// Deterministic seeded uniform in [0,1) (Numerical Recipes LCG, high bits).
struct Lcg(u64);
impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn gamma(&mut self, k: f64, scale: f64) -> f64 {
        if k < 1.0 {
            let g = self.gamma(k + 1.0, scale);
            let u = self.unit().max(1e-300);
            return g * u.powf(1.0 / k);
        }
        let d = k - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let u1 = self.unit().max(1e-300);
            let u2 = self.unit();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let v = (1.0 + c * z).powi(3);
            if v <= 0.0 {
                continue;
            }
            let u = self.unit().max(1e-300);
            if u.ln() < 0.5 * z * z + d - d * v + d * v.ln() {
                return d * v * scale;
            }
        }
    }
    /// Beta(a,b) via two Gammas: X~Gamma(a,1), Y~Gamma(b,1), X/(X+Y).
    fn beta(&mut self, a: f64, b: f64) -> f64 {
        let xa = self.gamma(a, 1.0);
        let yb = self.gamma(b, 1.0);
        xa / (xa + yb)
    }
}

fn logistic(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}
fn mu_true(x: f64) -> f64 {
    logistic(0.2 + 0.9 * x)
}
fn phi_true(x: f64) -> f64 {
    // Moderate precision (~1.6 .. 8.1) with a clear monotone trend: enough
    // dispersion-channel signal to recover, but well away from the (0,1)
    // boundary so the gamlss BE() RS algorithm converges quickly and cleanly.
    (1.3 + 0.8 * x).exp()
}

#[test]
fn gam_beta_dispersion_location_scale_recovers_phi_surface_vs_gamlss() {
    init_parallelism();

    // ---- synthetic heteroscedastic-precision Beta (seed = 2024) ------------
    let n = 500usize;
    let mut rng = Lcg(2024);
    let x: Vec<f64> = (0..n)
        .map(|i| -1.0 + 2.0 * (i as f64) / (n as f64 - 1.0))
        .collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let mu = mu_true(xi);
            let phi = phi_true(xi);
            rng.beta(mu * phi, (1.0 - mu) * phi).clamp(1e-4, 1.0 - 1e-4)
        })
        .collect();

    // ---- gam dispersion location-scale fit (noise_formula on phi) ----------
    let headers = vec!["x".to_string(), "y".to_string()];
    let records: Vec<csv::StringRecord> = (0..n)
        .map(|i| csv::StringRecord::from(vec![format!("{:.17e}", x[i]), format!("{:.17e}", y[i])]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, records).expect("encode beta data");
    let col = ds.column_map();
    let x_idx = col["x"];
    let ncols = ds.headers.len();

    let cfg = FitConfig {
        family: Some("beta".to_string()),
        noise_formula: Some("s(x, k=6)".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x, k=6)", &ds, &cfg).expect("gam beta dispersion fit");
    let FitResult::DispersionLocationScale(DispersionLocationScaleFitResult { fit, kind }) = result
    else {
        panic!("expected a DispersionLocationScale fit result");
    };
    assert_eq!(
        kind,
        DispersionFamilyKind::Beta,
        "beta + noise_formula must route to the Beta dispersion family"
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
        .expect("scale (log-phi) block present")
        .beta
        .clone();

    let grid_n = 60usize;
    let grid_x: Vec<f64> = (0..grid_n)
        .map(|i| -1.0 + 2.0 * (i as f64) / (grid_n as f64 - 1.0))
        .collect();
    let mut test_grid = Array2::<f64>::zeros((grid_n, ncols));
    for i in 0..grid_n {
        test_grid[[i, x_idx]] = grid_x[i];
    }

    let mean_design = build_term_collection_design(test_grid.view(), &fit.meanspec_resolved)
        .expect("rebuild mean design at grid");
    let disp_design = build_term_collection_design(test_grid.view(), &fit.noisespec_resolved)
        .expect("rebuild dispersion design at grid");

    // Mean is a logit link; the dispersion channel is log(phi): phi = exp(eta_d).
    let gam_logit_mu: Vec<f64> = mean_design.design.apply(&beta_location).to_vec();
    let gam_log_phi: Vec<f64> = disp_design.design.apply(&beta_scale).to_vec();

    // ---- gamlss BE() baseline on the identical data ------------------------
    let grid_csv = grid_x
        .iter()
        .map(|t| format!("{t:.17e}"))
        .collect::<Vec<_>>()
        .join(",");
    let body = format!(
        r#"
        suppressPackageStartupMessages(library(gamlss))
        # BE(): Beta with logit mean link and Var = mu(1-mu)*sigma^2, sigma in
        # (0,1). Matching gam's Var = mu(1-mu)/(1+phi) gives sigma^2 = 1/(1+phi).
        # A penalized B-spline `pb()` in BOTH the mu and the sigma predictor
        # mirrors gam's mean + dispersion smooths.
        m <- gamlss(y ~ pb(x),
                    sigma.formula = ~ pb(x),
                    family = BE(), data = df,
                    control = gamlss.control(n.cyc = 200, trace = FALSE))
        gx <- as.numeric(strsplit("{grid_csv}", ",")[[1]])
        nd <- data.frame(x = gx)
        pa <- predictAll(m, newdata = nd, data = df, type = "response")
        emit("mu", as.numeric(pa$mu))
        emit("sigma", as.numeric(pa$sigma))
        "#
    );
    let r = run_r(&[Column::new("x", &x), Column::new("y", &y)], &body);
    let gamlss_mu = r.vector("mu");
    let gamlss_sigma = r.vector("sigma");
    assert_eq!(gamlss_mu.len(), grid_n, "gamlss mu grid length mismatch");
    assert_eq!(
        gamlss_sigma.len(),
        grid_n,
        "gamlss sigma grid length mismatch"
    );
    // logit mu.
    let gamlss_logit_mu: Vec<f64> = gamlss_mu.iter().map(|&m| (m / (1.0 - m)).ln()).collect();
    // sigma^2 = 1/(1+phi)  =>  phi = (1-sigma^2)/sigma^2  =>  log phi.
    let gamlss_log_phi: Vec<f64> = gamlss_sigma
        .iter()
        .map(|&s| {
            let s2 = s * s;
            ((1.0 - s2) / s2).ln()
        })
        .collect();

    // ---- KNOWN ground truth on the same grid -------------------------------
    let truth_logit_mu: Vec<f64> = grid_x
        .iter()
        .map(|&xi| {
            let m = mu_true(xi);
            (m / (1.0 - m)).ln()
        })
        .collect();
    let truth_log_phi: Vec<f64> = grid_x.iter().map(|&xi| phi_true(xi).ln()).collect();

    // ---- OBJECTIVE metrics: truth-recovery RMSE ----------------------------
    let gam_rmse_logit_mu = rmse(&gam_logit_mu, &truth_logit_mu);
    let gam_rmse_log_phi = rmse(&gam_log_phi, &truth_log_phi);
    let gamlss_rmse_logit_mu = rmse(&gamlss_logit_mu, &truth_logit_mu);
    let gamlss_rmse_log_phi = rmse(&gamlss_log_phi, &truth_log_phi);
    let gam_disp_signal_r = pearson(&gam_log_phi, &truth_log_phi);

    eprintln!(
        "beta dispersion location-scale truth recovery (#1060): n={n} grid={grid_n}\n  \
         RMSE_vs_truth(logit mu): gam={gam_rmse_logit_mu:.5} gamlss={gamlss_rmse_logit_mu:.5}\n  \
         RMSE_vs_truth(log phi):  gam={gam_rmse_log_phi:.5} gamlss={gamlss_rmse_log_phi:.5}\n  \
         dispersion-channel signal pearson(gam log phi, truth)={gam_disp_signal_r:.4}"
    );

    // 1. TRUTH RECOVERY (mean).
    assert!(
        gam_rmse_logit_mu < 0.35,
        "gam logit-mean RMSE vs truth too large: {gam_rmse_logit_mu:.5} (>= 0.35)"
    );

    // 2. TRUTH RECOVERY (dispersion) — the headline claim.
    assert!(
        gam_rmse_log_phi < 0.60,
        "gam log-phi (dispersion) RMSE vs truth too large: {gam_rmse_log_phi:.5} (>= 0.60)"
    );
    assert!(
        gam_disp_signal_r > 0.55,
        "gam dispersion channel does not track the truth: pearson={gam_disp_signal_r:.4} (<= 0.55)"
    );

    // 3. MATCH-OR-BEAT BASELINE on the SAME objective metric and SAME truth.
    assert!(
        gam_rmse_logit_mu <= gamlss_rmse_logit_mu * 1.30,
        "gam logit-mean recovery must match-or-beat gamlss: gam={gam_rmse_logit_mu:.5} \
         gamlss={gamlss_rmse_logit_mu:.5}"
    );
    assert!(
        gam_rmse_log_phi <= gamlss_rmse_log_phi * 1.30,
        "gam dispersion-channel recovery must match-or-beat gamlss: gam={gam_rmse_log_phi:.5} \
         gamlss={gamlss_rmse_log_phi:.5}"
    );
}
