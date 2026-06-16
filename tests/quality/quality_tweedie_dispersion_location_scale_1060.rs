//! #1060: OBJECTIVE truth-recovery of the GAMLSS Tweedie *dispersion-channel*
//! location-scale fit — the Tweedie dispersion phi is modelled with its own
//! smooth formula via `noise_formula`.
//!
//! Unlike the Gamma / NB / Beta members, neither `gamlss` (base) nor `mgcv`
//! exposes a SMOOTH Tweedie-dispersion predictor (mgcv `tw()` estimates a single
//! scalar phi; `gamlss` has no native Tweedie family). So per the #904
//! reference-as-truth paradigm we score gam against a SELF-CONSTRUCTED known
//! generating law only — no reference tool can serve as a match-or-beat baseline
//! for this channel.
//!
//! Data-generating law (KNOWN): for covariate x on a grid,
//!   mu_true(x)  = exp(0.8 + 0.6*x)                   (log mean link)
//!   phi_true(x) = exp(0.4 - 0.9*x)   = phi(x)        (dispersion varies with x)
//!   y ~ compound Poisson(lambda)–Gamma with p = 1.5, so E[y]=mu, Var=phi*mu^p.
//! phi FALLS with x, so the dispersion shrinks with x — a genuine
//! dispersion-channel signal a mean-only Tweedie GLM cannot represent.
//!
//! gam's dispersion channel models `log(1/phi)` directly: 1/phi = exp(eta_d),
//! i.e. log(1/phi)_gam = eta_d, scored against the KNOWN -log phi_true surface.
//!
//! Compound Poisson–Gamma parameterization for Tweedie(p), 1<p<2 (Jorgensen):
//!   lambda = mu^(2-p) / (phi*(2-p)),   N ~ Poisson(lambda),
//!   gamma_shape = (2-p)/(p-1),  gamma_scale = phi*(p-1)*mu^(p-1),
//!   y = sum_{j=1}^N G_j  with  G_j ~ Gamma(gamma_shape, gamma_scale),  y=0 if N=0.
//! This yields E[y]=mu and Var=phi*mu^p exactly.
//!
//! Objective assertions (truth recovery only):
//!   1. RMSE(log mu_gam, log mu_true) is small (mean is the easy first moment).
//!   2. RMSE(log(1/phi)_gam, log(1/phi)_true) bounded AND positively correlated
//!      with the truth (the channel carries real, correctly-signed signal).

use gam::estimate::BlockRole;
use gam::gamlss::DispersionFamilyKind;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{pearson, rmse};
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
    fn poisson(&mut self, lambda: f64) -> u64 {
        let l = (-lambda).exp();
        let mut k = 0u64;
        let mut p = 1.0;
        loop {
            p *= self.unit();
            if p <= l {
                return k;
            }
            k += 1;
        }
    }
    /// Tweedie(p) compound Poisson–Gamma draw with mean mu, dispersion phi.
    fn tweedie(&mut self, mu: f64, phi: f64, p: f64) -> f64 {
        let lambda = mu.powf(2.0 - p) / (phi * (2.0 - p));
        let n = self.poisson(lambda);
        if n == 0 {
            return 0.0;
        }
        let shape = (2.0 - p) / (p - 1.0);
        let scale = phi * (p - 1.0) * mu.powf(p - 1.0);
        (0..n).map(|_| self.gamma(shape, scale)).sum()
    }
}

const P: f64 = 1.5;

fn mu_true(x: f64) -> f64 {
    (0.8 + 0.6 * x).exp()
}
fn phi_true(x: f64) -> f64 {
    (0.4 - 0.9 * x).exp()
}

#[test]
fn gam_tweedie_dispersion_location_scale_recovers_phi_surface() {
    init_parallelism();

    // ---- synthetic heteroscedastic-dispersion Tweedie (seed = 1357) --------
    let n = 600usize;
    let mut rng = Lcg(1357);
    let x: Vec<f64> = (0..n)
        .map(|i| -1.0 + 2.0 * (i as f64) / (n as f64 - 1.0))
        .collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| rng.tweedie(mu_true(xi), phi_true(xi), P))
        .collect();

    // ---- gam dispersion location-scale fit (noise_formula on 1/phi) --------
    let headers = vec!["x".to_string(), "y".to_string()];
    let records: Vec<csv::StringRecord> = (0..n)
        .map(|i| csv::StringRecord::from(vec![format!("{:.17e}", x[i]), format!("{:.17e}", y[i])]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, records).expect("encode tweedie data");
    let col = ds.column_map();
    let x_idx = col["x"];
    let ncols = ds.headers.len();

    let cfg = FitConfig {
        family: Some("tweedie".to_string()),
        noise_formula: Some("s(x, k=6)".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x, k=6)", &ds, &cfg).expect("gam tweedie dispersion fit");
    let FitResult::DispersionLocationScale(DispersionLocationScaleFitResult { fit, kind }) = result
    else {
        panic!("expected a DispersionLocationScale fit result");
    };
    assert!(
        matches!(kind, DispersionFamilyKind::Tweedie { .. }),
        "tweedie + noise_formula must route to the Tweedie dispersion family"
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
        .expect("scale (log 1/phi) block present")
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

    // Mean is a log link; the dispersion channel is log(1/phi) = eta_d.
    let gam_log_mu: Vec<f64> = mean_design.design.apply(&beta_location).to_vec();
    let gam_log_inv_phi: Vec<f64> = disp_design.design.apply(&beta_scale).to_vec();

    // ---- KNOWN ground truth on the same grid -------------------------------
    let truth_log_mu: Vec<f64> = grid_x.iter().map(|&xi| mu_true(xi).ln()).collect();
    // log(1/phi) = -log phi.
    let truth_log_inv_phi: Vec<f64> = grid_x.iter().map(|&xi| -phi_true(xi).ln()).collect();

    // ---- OBJECTIVE metrics: truth-recovery RMSE ----------------------------
    let gam_rmse_log_mu = rmse(&gam_log_mu, &truth_log_mu);
    let gam_rmse_log_inv_phi = rmse(&gam_log_inv_phi, &truth_log_inv_phi);
    let gam_disp_signal_r = pearson(&gam_log_inv_phi, &truth_log_inv_phi);

    eprintln!(
        "tweedie dispersion location-scale truth recovery (#1060): n={n} grid={grid_n} p={P}\n  \
         RMSE_vs_truth(log mu):      gam={gam_rmse_log_mu:.5}\n  \
         RMSE_vs_truth(log 1/phi):   gam={gam_rmse_log_inv_phi:.5}\n  \
         dispersion-channel signal pearson(gam log 1/phi, truth)={gam_disp_signal_r:.4}"
    );

    // 1. TRUTH RECOVERY (mean): the well-determined first moment.
    assert!(
        gam_rmse_log_mu < 0.25,
        "gam log-mean RMSE vs truth too large: {gam_rmse_log_mu:.5} (>= 0.25)"
    );

    // 2. TRUTH RECOVERY (dispersion) — the headline claim. The Tweedie
    //    dispersion is the hardest second-moment channel here (saddlepoint
    //    likelihood, zero point mass), so the absolute bound is generous; the
    //    correctly-signed signal correlation is the substantive assertion.
    assert!(
        gam_rmse_log_inv_phi < 0.85,
        "gam log(1/phi) (dispersion) RMSE vs truth too large: {gam_rmse_log_inv_phi:.5} (>= 0.85)"
    );
    assert!(
        gam_disp_signal_r > 0.5,
        "gam dispersion channel does not track the truth: pearson={gam_disp_signal_r:.4} (<= 0.5)"
    );
}
