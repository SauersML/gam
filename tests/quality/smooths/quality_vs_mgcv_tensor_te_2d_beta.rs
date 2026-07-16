//! End-to-end quality: gam's tensor-product 2-D smooth `te(x, z)` under the
//! **Beta** family (logit link) must RECOVER THE TRUE mean-proportion surface
//! from bounded (0,1) data — not merely reproduce a peer tool's fit.
//!
//! Why this combination earns its own test. gam's Beta family was previously
//! tested only *additively* (`quality_vs_betareg_beta_logit`: `s(x1)+s(x2)`
//! against `betareg` with fixed natural-spline bases) — never against **mgcv**,
//! and never on a genuine tensor-product INTERACTION surface that an additive
//! model cannot represent. The Beta mean/precision law `Var(y)=mu(1-mu)/(1+phi)`
//! makes the IRLS working weights depend on `mu` through both the logit link and
//! the variance function, so recovering a curved interaction surface is a real
//! test that gam's PIRLS handles the Beta weights correctly. mgcv's native
//! `betar` family (penalised REML) is the canonical peer; it is demoted to a
//! match-or-beat accuracy baseline on the SAME objective, never an output to
//! reproduce.
//!
//! Data (seed=20260603, n=300): x, z ~ U(0,1); truth on the logit scale
//! `eta_true = sin(pi*x)*cos(pi*z)` (range [-1,1] => mu in ~[0.27,0.73], interior
//! so the Beta likelihood is well away from the {0,1} boundary); precision
//! phi=20; y ~ Beta(mu*phi, (1-mu)*phi), clamped strictly inside (0,1).
//!
//! Asserts:
//!   1. RMSE(mu_gam, mu_true) < per-observation Beta noise sd
//!      `sqrt(mean(mu(1-mu)/(1+phi)))` — gam recovers the mean proportion to
//!      better than a single draw's noise.
//!   2. RMSE(eta_gam, eta_true) <= RMSE(eta_mgcv, eta_true) * 1.10 — gam is at
//!      least as accurate at recovering the truth as the mature mgcv fit.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, QualityPair, pearson, relative_l2, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Beta, Distribution, Uniform};
use std::f64::consts::PI;

const N: usize = 300;
const PHI: f64 = 20.0;

#[inline]
fn logit_inv(eta: f64) -> f64 {
    1.0 / (1.0 + (-eta).exp())
}

#[test]
fn gam_tensor_te_2d_beta_matches_mgcv() {
    init_parallelism();

    // ---- synthetic Beta truth on the unit square ---------------------------
    // eta_true = sin(pi*x)*cos(pi*z) (a genuine interaction, not additive) on the
    // logit scale; mu = logit^{-1}(eta) in ~[0.27,0.73]; y ~ Beta(mu*phi,(1-mu)*phi).
    // A fixed seed feeds the SAME draws to gam and mgcv.
    let mut rng = StdRng::seed_from_u64(20260603);
    let u = Uniform::new(0.0_f64, 1.0).expect("uniform [0,1]");

    let mut x = Vec::with_capacity(N);
    let mut z = Vec::with_capacity(N);
    let mut y = Vec::with_capacity(N);
    let mut eta_true = Vec::with_capacity(N);
    let mut mu_true = Vec::with_capacity(N);
    for _ in 0..N {
        let xi = u.sample(&mut rng);
        let zi = u.sample(&mut rng);
        let eta = (PI * xi).sin() * (PI * zi).cos();
        let mu = logit_inv(eta);
        let beta = Beta::new(mu * PHI, (1.0 - mu) * PHI).expect("beta shapes > 0");
        // Keep draws strictly inside (0,1) so the Beta likelihood is finite for
        // both engines (gam and mgcv betar both require open-interval responses).
        let yi = beta.sample(&mut rng).clamp(1e-6, 1.0 - 1e-6);
        x.push(xi);
        z.push(zi);
        y.push(yi);
        eta_true.push(eta);
        mu_true.push(mu);
    }

    // Per-observation Beta noise sd at the truth, averaged over the design — the
    // natural accuracy floor for recovering the mean: Var = mu(1-mu)/(1+phi).
    let noise_sd_bar = (mu_true
        .iter()
        .map(|&m| m * (1.0 - m) / (1.0 + PHI))
        .sum::<f64>()
        / N as f64)
        .sqrt();

    // ---- fit with gam: y ~ te(x, z, k=7), Beta / logit link, REML ----------
    let headers = ["x", "z", "y"].into_iter().map(String::from).collect();
    let rows: Vec<csv::StringRecord> = (0..N)
        .map(|i| {
            csv::StringRecord::from(vec![x[i].to_string(), z[i].to_string(), y[i].to_string()])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode beta dataset");
    let col = ds.column_map();
    let x_idx = col["x"];
    let z_idx = col["z"];

    let cfg = FitConfig {
        family: Some("beta".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ te(x, z, k=7)", &ds, &cfg).expect("gam beta te fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for Beta te(x, z)");
    };

    // gam linear predictor (logit scale) at the training points, then mu.
    let mut grid = Array2::<f64>::zeros((N, ds.headers.len()));
    for i in 0..N {
        grid[[i, x_idx]] = x[i];
        grid[[i, z_idx]] = z[i];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild te design at training points");
    let gam_eta: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    let gam_mu: Vec<f64> = gam_eta.iter().map(|&e| logit_inv(e)).collect();

    // ---- fit the SAME model with mgcv betar (the mature reference) ---------
    let r = run_r(
        &[
            Column::new("x", &x),
            Column::new("z", &z),
            Column::new("y", &y),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        m <- gam(y ~ te(x, z, k = 7), data = df,
                 family = betar(link = "logit"), method = "REML")
        emit("eta", as.numeric(predict(m, type = "link")))
        emit("edf", sum(m$edf))
        "#,
    );
    let mgcv_eta = r.vector("eta");
    let mgcv_edf = r.scalar("edf");
    assert_eq!(mgcv_eta.len(), N, "mgcv linear-predictor length mismatch");

    // ---- OBJECTIVE METRICS -------------------------------------------------
    let gam_eta_err = rmse(&gam_eta, &eta_true);
    let mgcv_eta_err = rmse(mgcv_eta, &eta_true);
    let gam_mu_err = rmse(&gam_mu, &mu_true);

    // Context only (NOT a pass criterion): closeness of the two fitted surfaces.
    let rel_to_mgcv = relative_l2(&gam_eta, mgcv_eta);
    let corr_to_mgcv = pearson(&gam_eta, mgcv_eta);

    eprintln!(
        "te(x,z) Beta/logit: n={N} phi={PHI} mgcv_edf={mgcv_edf:.3} \
         rmse_mu(gam,truth)={gam_mu_err:.4} noise_sd_bar={noise_sd_bar:.4} \
         rmse_eta(gam)={gam_eta_err:.4} rmse_eta(mgcv)={mgcv_eta_err:.4} \
         [context] rel_l2(gam,mgcv)={rel_to_mgcv:.4} pearson(gam,mgcv)={corr_to_mgcv:.5}"
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "smooths",
            "quality_vs_mgcv_tensor_te_2d_beta",
            "eta_err",
            gam_eta_err,
            "mgcv",
            mgcv_eta_err,
        )
        .line()
    );

    // PRIMARY: gam recovers the true mean-proportion surface to better than the
    // per-observation Beta noise. A botched logit gradient or Beta weight makes
    // the recovered surface miss the truth by far more than a single draw's sd.
    assert!(
        gam_mu_err < noise_sd_bar,
        "gam should recover the true Beta mean surface: rmse_mu={gam_mu_err:.4} \
         (bar = per-obs noise sd {noise_sd_bar:.4})"
    );

    // MATCH-OR-BEAT: gam's truth-recovery error on the logit scale is no worse
    // than mgcv's by more than 10%, holding mgcv as an accuracy baseline.
    assert!(
        gam_eta_err <= mgcv_eta_err * 1.10,
        "gam's truth-recovery error must match-or-beat mgcv: \
         rmse_eta(gam)={gam_eta_err:.4} vs mgcv*1.10={:.4}",
        mgcv_eta_err * 1.10
    );
}
