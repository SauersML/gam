//! #1060: OBJECTIVE truth-recovery of the GAMLSS Negative-Binomial
//! *dispersion-channel* location-scale fit — the dispersion (here the NB2 size
//! parameter theta) is modelled with its own smooth formula via `noise_formula`.
//!
//! Data-generating law (KNOWN): for covariate x on a grid,
//!   mu_true(x)    = exp(0.7 + 0.7*x)                 (log mean link)
//!   theta_true(x) = exp(1.4 + 1.0*x)   = theta(x)    (NB2 size varies with x)
//!   y ~ NB2(mu(x), theta(x)),  Var = mu + mu^2/theta  (Gamma-Poisson mixture)
//! theta rises with x, so the overdispersion mu^2/theta SHRINKS with x — a
//! genuine dispersion-channel signal a mean-only NB GLM cannot represent.
//!
//! gam's dispersion channel models `log(theta)` directly: theta_gam = exp(eta_d)
//! with a log mean link. gamlss `NBI()` parameterizes (mu, sigma) with
//! Var = mu + sigma*mu^2, i.e. theta_gamlss = 1/sigma; so log theta = -log sigma.
//! Both engines are scored on the SAME log-theta surface vs the SAME known truth.
//!
//! Objective assertions (none is "match the reference"):
//!   1. TRUTH RECOVERY (mean): RMSE(log mu_gam, log mu_true) is small.
//!   2. TRUTH RECOVERY (dispersion): RMSE(log theta_gam, log theta_true) bounded
//!      AND positively correlated with the truth (real, correctly-signed signal).
//!   3. MATCH-OR-BEAT: gam's dispersion-channel and mean-channel RMSE each
//!      <= 1.30 * gamlss's. gamlss is a demoted baseline.

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
    /// Gamma(shape k>0, scale theta) via Marsaglia–Tsang (Ahrens–Dieter boost
    /// for k<1), driven off this LCG.
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
    /// Poisson(lambda) via Knuth's product algorithm (lambda is modest here).
    fn poisson(&mut self, lambda: f64) -> f64 {
        let l = (-lambda).exp();
        let mut k = 0.0;
        let mut p = 1.0;
        loop {
            p *= self.unit();
            if p <= l {
                return k;
            }
            k += 1.0;
        }
    }
    /// NB2 draw via the Gamma–Poisson mixture: lambda ~ Gamma(theta, mu/theta),
    /// y ~ Poisson(lambda) gives E[y]=mu, Var=mu+mu^2/theta.
    fn negbin(&mut self, mu: f64, theta: f64) -> f64 {
        let lambda = self.gamma(theta, mu / theta);
        self.poisson(lambda)
    }
}

fn mu_true(x: f64) -> f64 {
    (0.7 + 0.7 * x).exp()
}
fn theta_true(x: f64) -> f64 {
    (1.4 + 1.0 * x).exp()
}

#[test]
fn gam_negbin_dispersion_location_scale_recovers_theta_surface_vs_gamlss() {
    init_parallelism();

    // ---- synthetic heteroscedastic-dispersion NB2 (seed = 7777) ------------
    let n = 500usize;
    let mut rng = Lcg(7777);
    let x: Vec<f64> = (0..n)
        .map(|i| -1.0 + 2.0 * (i as f64) / (n as f64 - 1.0))
        .collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| rng.negbin(mu_true(xi), theta_true(xi)))
        .collect();

    // ---- gam dispersion location-scale fit (noise_formula on theta) ---------
    let headers = vec!["x".to_string(), "y".to_string()];
    let records: Vec<csv::StringRecord> = (0..n)
        .map(|i| csv::StringRecord::from(vec![format!("{:.17e}", x[i]), format!("{:.17e}", y[i])]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, records).expect("encode negbin data");
    let col = ds.column_map();
    let x_idx = col["x"];
    let ncols = ds.headers.len();

    let cfg = FitConfig {
        family: Some("nb".to_string()),
        noise_formula: Some("s(x, k=6)".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x, k=6)", &ds, &cfg).expect("gam negbin dispersion fit");
    let FitResult::DispersionLocationScale(DispersionLocationScaleFitResult { fit, kind }) = result
    else {
        panic!("expected a DispersionLocationScale fit result");
    };
    assert_eq!(
        kind,
        DispersionFamilyKind::NegativeBinomial,
        "nb + noise_formula must route to the NegativeBinomial dispersion family"
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
        .expect("scale (log-theta) block present")
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

    let gam_log_mu: Vec<f64> = mean_design.design.apply(&beta_location).to_vec();
    let gam_log_theta: Vec<f64> = disp_design.design.apply(&beta_scale).to_vec();

    // ---- gamlss NBI() baseline on the identical data -----------------------
    let grid_csv = grid_x
        .iter()
        .map(|t| format!("{t:.17e}"))
        .collect::<Vec<_>>()
        .join(",");
    let body = format!(
        r#"
        suppressPackageStartupMessages(library(gamlss))
        # NBI(): NB type I with log mean link and Var = mu + sigma*mu^2, so the
        # NB2 size is theta = 1/sigma. A penalized B-spline `pb()` in BOTH the mu
        # and the sigma predictor mirrors gam's mean + dispersion smooths.
        m <- gamlss(y ~ pb(x),
                    sigma.formula = ~ pb(x),
                    family = NBI(), data = df,
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
    // theta = 1/sigma  =>  log theta = -log sigma.
    let gamlss_log_theta: Vec<f64> = gamlss_sigma.iter().map(|&s| -s.ln()).collect();

    // ---- KNOWN ground truth on the same grid -------------------------------
    let truth_log_mu: Vec<f64> = grid_x.iter().map(|&xi| mu_true(xi).ln()).collect();
    let truth_log_theta: Vec<f64> = grid_x.iter().map(|&xi| theta_true(xi).ln()).collect();

    // ---- OBJECTIVE metrics: truth-recovery RMSE ----------------------------
    let gam_rmse_log_mu = rmse(&gam_log_mu, &truth_log_mu);
    let gam_rmse_log_theta = rmse(&gam_log_theta, &truth_log_theta);
    let gamlss_rmse_log_mu = rmse(&gamlss_log_mu, &truth_log_mu);
    let gamlss_rmse_log_theta = rmse(&gamlss_log_theta, &truth_log_theta);
    let gam_disp_signal_r = pearson(&gam_log_theta, &truth_log_theta);

    eprintln!(
        "negbin dispersion location-scale truth recovery (#1060): n={n} grid={grid_n}\n  \
         RMSE_vs_truth(log mu):    gam={gam_rmse_log_mu:.5} gamlss={gamlss_rmse_log_mu:.5}\n  \
         RMSE_vs_truth(log theta): gam={gam_rmse_log_theta:.5} gamlss={gamlss_rmse_log_theta:.5}\n  \
         dispersion-channel signal pearson(gam log theta, truth)={gam_disp_signal_r:.4}"
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "misc",
            "quality_vs_gamlss_negbin_dispersion_location_scale_1060::mu",
            "rmse_log_mu",
            gam_rmse_log_mu,
            "gamlss",
            gamlss_rmse_log_mu,
        )
        .line()
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "misc",
            "quality_vs_gamlss_negbin_dispersion_location_scale_1060::log_theta",
            "rmse_log_theta",
            gam_rmse_log_theta,
            "gamlss",
            gamlss_rmse_log_theta,
        )
        .line()
    );

    // 1. TRUTH RECOVERY (mean).
    assert!(
        gam_rmse_log_mu < 0.20,
        "gam log-mean RMSE vs truth too large: {gam_rmse_log_mu:.5} (>= 0.20)"
    );

    // 2. TRUTH RECOVERY (dispersion). The NB size is a notoriously hard, weakly
    //    identified second-moment quantity, so the absolute bound is generous;
    //    the correctly-signed signal correlation is the substantive claim.
    assert!(
        gam_rmse_log_theta < 0.85,
        "gam log-theta (dispersion) RMSE vs truth too large: {gam_rmse_log_theta:.5} (>= 0.85)"
    );
    assert!(
        gam_disp_signal_r > 0.5,
        "gam dispersion channel does not track the truth: pearson={gam_disp_signal_r:.4} (<= 0.5)"
    );

    // 3. MATCH-OR-BEAT BASELINE on the SAME objective metric and SAME truth.
    assert!(
        gam_rmse_log_mu <= gamlss_rmse_log_mu * 1.30,
        "gam log-mean recovery must match-or-beat gamlss: gam={gam_rmse_log_mu:.5} \
         gamlss={gamlss_rmse_log_mu:.5}"
    );
    assert!(
        gam_rmse_log_theta <= gamlss_rmse_log_theta * 1.30,
        "gam dispersion-channel recovery must match-or-beat gamlss: gam={gam_rmse_log_theta:.5} \
         gamlss={gamlss_rmse_log_theta:.5}"
    );
}
