//! #1060: OBJECTIVE truth-recovery of the GAMLSS Gamma *dispersion-channel*
//! location-scale fit — the headline GAMLSS capability where the dispersion
//! (here the Gamma shape) is itself modelled with its own smooth formula.
//!
//! The dispersion families (NB / Gamma / Beta / Tweedie) reach the
//! `DispersionGlmLocationScale` engine (#913) via `noise_formula`; the FFI /
//! magic-routing surface is already pinned by
//! `tests/test_dispersion_location_scale_ffi_surface_913.py`. What was missing
//! (the explicit ask of #1060) is an end-to-end QUALITY gate: does the learned
//! dispersion channel actually RECOVER the true heteroscedastic dispersion
//! surface, matching or beating `gamlss` (R)?
//!
//! Data-generating law (KNOWN): for covariate x on a grid,
//!   mu_true(x)    = exp(0.6 + 0.8*x)                 (log mean link)
//!   shape_true(x) = exp(1.6 + 1.1*x)   = nu(x)       (Gamma shape varies with x)
//!   y ~ Gamma(shape = nu(x), scale = mu(x)/nu(x))    (mean mu, Var = mu^2/nu)
//! The shape (precision) rises with x, so the relative dispersion 1/sqrt(nu)
//! SHRINKS with x — a genuine dispersion-channel signal that a mean-only Gamma
//! GLM cannot represent.
//!
//! gam's dispersion channel models `log(nu)` directly: nu_gam = exp(eta_d) with
//! the mean on a log link. gamlss `GA()` parameterizes (mu, sigma) with
//! Var = sigma^2 * mu^2, i.e. nu_gamlss = 1/sigma_gamlss^2; we map sigma back to
//! a shape so both engines are scored on the SAME log-shape surface against the
//! SAME known truth.
//!
//! Objective assertions (none is "match the reference"):
//!   1. TRUTH RECOVERY (mean): RMSE(log mu_gam, log mu_true) is small.
//!   2. TRUTH RECOVERY (dispersion): RMSE(log nu_gam, log nu_true) is small AND
//!      the recovered log-shape is positively correlated with the truth (the
//!      channel carries real signal, not a flat intercept).
//!   3. MATCH-OR-BEAT: gam's dispersion-channel RMSE <= 1.25 * gamlss's, and the
//!      mean-channel RMSE <= 1.25 * gamlss's. gamlss is a demoted baseline.

use gam::estimate::BlockRole;
use gam::gamlss::DispersionFamilyKind;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, QualityPair, pearson, rmse, run_r};
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
    /// Gamma(shape k>0, scale theta) via Marsaglia–Tsang (k>=1) with the
    /// Ahrens–Dieter boost for k<1. Uses Box–Muller normals off the same LCG so
    /// the exact same y is reproducible in pure Rust and shared (via CSV) with R.
    fn gamma(&mut self, k: f64, theta: f64) -> f64 {
        if k < 1.0 {
            let g = self.gamma(k + 1.0, theta);
            let u = self.unit().max(1e-300);
            return g * u.powf(1.0 / k);
        }
        let d = k - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            // standard normal via Box–Muller
            let u1 = self.unit().max(1e-300);
            let u2 = self.unit();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let v = (1.0 + c * z).powi(3);
            if v <= 0.0 {
                continue;
            }
            let u = self.unit().max(1e-300);
            if u.ln() < 0.5 * z * z + d - d * v + d * v.ln() {
                return d * v * theta;
            }
        }
    }
}

fn mu_true(x: f64) -> f64 {
    (0.6 + 0.8 * x).exp()
}
fn shape_true(x: f64) -> f64 {
    (1.6 + 1.1 * x).exp()
}

#[test]
fn gam_gamma_dispersion_location_scale_recovers_shape_surface_vs_gamlss() {
    init_parallelism();

    // ---- synthetic heteroscedastic-dispersion Gamma (seed = 4242) ----------
    let n = 400usize;
    let mut rng = Lcg(4242);
    let x: Vec<f64> = (0..n)
        .map(|i| -1.0 + 2.0 * (i as f64) / (n as f64 - 1.0))
        .collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let mu = mu_true(xi);
            let nu = shape_true(xi);
            rng.gamma(nu, mu / nu).max(1e-6)
        })
        .collect();

    // ---- gam dispersion location-scale fit (noise_formula on the shape) -----
    let headers = vec!["x".to_string(), "y".to_string()];
    let records: Vec<csv::StringRecord> = (0..n)
        .map(|i| csv::StringRecord::from(vec![format!("{:.17e}", x[i]), format!("{:.17e}", y[i])]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, records).expect("encode gamma data");
    let col = ds.column_map();
    let x_idx = col["x"];
    let ncols = ds.headers.len();

    let cfg = FitConfig {
        family: Some("gamma".to_string()),
        noise_formula: Some("s(x, k=6)".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x, k=6)", &ds, &cfg).expect("gam gamma dispersion fit");
    let FitResult::DispersionLocationScale(DispersionLocationScaleFitResult { fit, kind }) = result
    else {
        panic!("expected a DispersionLocationScale fit result");
    };
    assert_eq!(
        kind,
        DispersionFamilyKind::Gamma,
        "gamma + noise_formula must route to the Gamma dispersion family"
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
        .expect("scale (log-shape) block present")
        .beta
        .clone();

    // Evaluate both channels on a dense test grid. The grid carries the full
    // schema width with the single feature placed at its column index, matching
    // the frozen design's expected column layout.
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

    // Mean is a log link; the dispersion channel is log(shape): nu = exp(eta_d).
    let gam_log_mu: Vec<f64> = mean_design.design.apply(&beta_location).to_vec();
    let gam_log_nu: Vec<f64> = disp_design.design.apply(&beta_scale).to_vec();

    // ---- gamlss GA() baseline on the identical data ------------------------
    let grid_csv = grid_x
        .iter()
        .map(|t| format!("{t:.17e}"))
        .collect::<Vec<_>>()
        .join(",");
    let body = format!(
        r#"
        suppressPackageStartupMessages(library(gamlss))
        # GA(): Gamma with log mean link and Var = sigma^2 * mu^2, so the Gamma
        # shape is nu = 1/sigma^2. A penalized B-spline `pb()` in BOTH the mu and
        # the sigma predictor mirrors gam's mean + dispersion smooths.
        m <- gamlss(y ~ pb(x),
                    sigma.formula = ~ pb(x),
                    family = GA(), data = df,
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
    let gamlss_log_mu: Vec<f64> = gamlss_mu.iter().map(|&m| m.ln()).collect();
    // nu = 1/sigma^2  =>  log nu = -2*log sigma.
    let gamlss_log_nu: Vec<f64> = gamlss_sigma.iter().map(|&s| -2.0 * s.ln()).collect();

    // ---- KNOWN ground truth on the same grid -------------------------------
    let truth_log_mu: Vec<f64> = grid_x.iter().map(|&xi| mu_true(xi).ln()).collect();
    let truth_log_nu: Vec<f64> = grid_x.iter().map(|&xi| shape_true(xi).ln()).collect();

    // ---- OBJECTIVE metrics: truth-recovery RMSE ----------------------------
    let gam_rmse_log_mu = rmse(&gam_log_mu, &truth_log_mu);
    let gam_rmse_log_nu = rmse(&gam_log_nu, &truth_log_nu);
    let gamlss_rmse_log_mu = rmse(&gamlss_log_mu, &truth_log_mu);
    let gamlss_rmse_log_nu = rmse(&gamlss_log_nu, &truth_log_nu);
    let gam_disp_signal_r = pearson(&gam_log_nu, &truth_log_nu);

    eprintln!(
        "gamma dispersion location-scale truth recovery (#1060): n={n} grid={grid_n}\n  \
         RMSE_vs_truth(log mu):    gam={gam_rmse_log_mu:.5} gamlss={gamlss_rmse_log_mu:.5}\n  \
         RMSE_vs_truth(log shape): gam={gam_rmse_log_nu:.5} gamlss={gamlss_rmse_log_nu:.5}\n  \
         dispersion-channel signal pearson(gam log nu, truth)={gam_disp_signal_r:.4}"
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "families",
            "quality_vs_gamlss_gamma_dispersion_location_scale_1060::log_mu",
            "log_mu_rmse_to_truth",
            gam_rmse_log_mu,
            "gamlss",
            gamlss_rmse_log_mu,
        )
        .line()
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "families",
            "quality_vs_gamlss_gamma_dispersion_location_scale_1060::log_nu",
            "log_nu_rmse_to_truth",
            gam_rmse_log_nu,
            "gamlss",
            gamlss_rmse_log_nu,
        )
        .line()
    );

    // 1. TRUTH RECOVERY (mean): the log-mean surface is the well-determined
    //    first moment; gam must recover it tightly.
    assert!(
        gam_rmse_log_mu < 0.20,
        "gam log-mean RMSE vs truth too large: {gam_rmse_log_mu:.5} (>= 0.20)"
    );

    // 2. TRUTH RECOVERY (dispersion): the headline claim — the learned shape
    //    channel tracks the KNOWN heteroscedastic dispersion surface. Dispersion
    //    is a second-moment quantity (noisier than the mean), so the bound is
    //    looser, but it must carry genuine, correctly-signed signal.
    assert!(
        gam_rmse_log_nu < 0.60,
        "gam log-shape (dispersion) RMSE vs truth too large: {gam_rmse_log_nu:.5} (>= 0.60)"
    );
    assert!(
        gam_disp_signal_r > 0.6,
        "gam dispersion channel does not track the truth: pearson={gam_disp_signal_r:.4} (<= 0.6)"
    );

    // 3. MATCH-OR-BEAT BASELINE: gamlss is demoted to a baseline on the SAME
    //    objective metric and the SAME known truth.
    assert!(
        gam_rmse_log_mu <= gamlss_rmse_log_mu * 1.25,
        "gam log-mean recovery must match-or-beat gamlss: gam={gam_rmse_log_mu:.5} \
         gamlss={gamlss_rmse_log_mu:.5}"
    );
    assert!(
        gam_rmse_log_nu <= gamlss_rmse_log_nu * 1.25,
        "gam dispersion-channel recovery must match-or-beat gamlss: gam={gam_rmse_log_nu:.5} \
         gamlss={gamlss_rmse_log_nu:.5}"
    );
}
