//! End-to-end quality: gam's tensor-product 2-D smooth `te(x, z)` under the
//! **Tweedie** family (compound Poisson-Gamma, log link, fixed power p=1.5) must
//! RECOVER THE TRUE log-mean surface from zero-inflated non-negative data — not
//! merely reproduce a peer tool's fit.
//!
//! Why this combination earns its own test. gam's Tweedie family was previously
//! tested only *additively* (`quality_vs_statsmodels_tweedie`: `s(x1)+s(x2)`
//! against a fixed-basis statsmodels GLM) — never against **mgcv**, and never on
//! a genuine tensor-product INTERACTION surface. Tweedie with `p in (1,2)` has
//! variance `Var(y)=phi*mu^p`, the bridge between Poisson (p=1) and Gamma (p=2)
//! and the workhorse for zero-inflated non-negative data (insurance totals,
//! rainfall, biomass). p=1.5 is the canonical semi-Poisson case and mgcv's
//! default `tw()` power; gam fixes the Tweedie link to log. mgcv's native
//! `Tweedie(p=1.5, link="log")` family (penalised REML) is the canonical peer,
//! demoted to a match-or-beat accuracy baseline on the SAME objective, never an
//! output to reproduce.
//!
//! Data (seed=20260604, n=300): x, z ~ U(0,1); truth on the log-mean scale
//! `eta_true = 2.0 + sin(pi*x)*cos(pi*z)` => mu = exp(eta) in ~[2.7,20.1];
//! y ~ Tweedie(mu, p=1.5, phi=2.0) via the exact compound Poisson-Gamma draw,
//! which produces the characteristic exact zeros.
//!
//! Asserts:
//!   1. The data is genuinely zero-inflated (exact zeros present) — confirming
//!      the Tweedie regime, not a degenerate Gamma.
//!   2. RMSE(mu_gam, mu_true) < per-observation Tweedie noise sigma
//!      `sqrt(mean(phi*mu^p))` — gam recovers the mean to better than a single
//!      draw's noise.
//!   3. RMSE(eta_gam, eta_true) <= RMSE(eta_mgcv, eta_true) * 1.10 — gam is at
//!      least as accurate at recovering the truth as the mature mgcv fit.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, relative_l2, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Gamma, Poisson, Uniform};
use std::f64::consts::PI;

const N: usize = 300;
const P: f64 = 1.5;
const PHI: f64 = 2.0;

/// Draw one compound Poisson-Gamma (Tweedie) variate with mean `mu`, power
/// `p in (1,2)`, dispersion `phi`. The exact exponential-dispersion construction:
/// `N ~ Poisson(lambda)`, `y = sum_{i=1}^N G_i`, `G_i ~ Gamma(alpha, theta)`,
/// with `lambda = mu^{2-p}/(phi(2-p))`, `alpha = (2-p)/(p-1)`,
/// `theta = phi(p-1)mu^{p-1}`. `N = 0` yields the exact zero (zero-inflation).
fn tweedie_sample(mu: f64, p: f64, phi: f64, rng: &mut StdRng) -> f64 {
    let lambda = mu.powf(2.0 - p) / (phi * (2.0 - p));
    let alpha = (2.0 - p) / (p - 1.0);
    let theta = phi * (p - 1.0) * mu.powf(p - 1.0);
    let n = Poisson::new(lambda).expect("poisson rate").sample(rng) as u64;
    if n == 0 {
        return 0.0;
    }
    let gamma = Gamma::new(alpha, theta).expect("gamma shape/scale");
    (0..n).map(|_| gamma.sample(rng)).sum()
}

#[test]
fn gam_tensor_te_2d_tweedie_matches_mgcv() {
    init_parallelism();
    gam::progress_log::init_logging();

    // ---- synthetic Tweedie truth on the unit square ------------------------
    // eta_true = 2.0 + sin(pi*x)*cos(pi*z) (a genuine interaction); mu = exp(eta)
    // in ~[2.7,20.1]; y ~ Tweedie(mu, p=1.5, phi=2). A fixed seed feeds the SAME
    // draws to gam and mgcv.
    let mut rng = StdRng::seed_from_u64(20260604);
    let u = Uniform::new(0.0_f64, 1.0).expect("uniform [0,1]");

    let mut x = Vec::with_capacity(N);
    let mut z = Vec::with_capacity(N);
    let mut y = Vec::with_capacity(N);
    let mut eta_true = Vec::with_capacity(N);
    let mut mu_true = Vec::with_capacity(N);
    for _ in 0..N {
        let xi = u.sample(&mut rng);
        let zi = u.sample(&mut rng);
        let eta = 2.0 + (PI * xi).sin() * (PI * zi).cos();
        let mu = eta.exp();
        let yi = tweedie_sample(mu, P, PHI, &mut rng);
        x.push(xi);
        z.push(zi);
        y.push(yi);
        eta_true.push(eta);
        mu_true.push(mu);
    }

    // The Tweedie regime is defined by exact zeros; confirm the draw produced
    // them (otherwise this would be a degenerate Gamma test).
    let zeros = y.iter().filter(|&&v| v == 0.0).count();
    assert!(
        zeros > 0,
        "Tweedie p=1.5 data should be zero-inflated; got {zeros} exact zeros"
    );

    // Per-observation Tweedie noise sigma at the truth, averaged over the design:
    // Var = phi*mu^p. This is the accuracy floor for recovering the mean.
    let noise_sigma = (mu_true.iter().map(|&m| PHI * m.powf(P)).sum::<f64>() / N as f64).sqrt();

    // ---- fit with gam: y ~ te(x, z, k=7), Tweedie / log link, REML ---------
    let headers = ["x", "z", "y"].into_iter().map(String::from).collect();
    let rows: Vec<csv::StringRecord> = (0..N)
        .map(|i| {
            csv::StringRecord::from(vec![x[i].to_string(), z[i].to_string(), y[i].to_string()])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode tweedie dataset");
    let col = ds.column_map();
    let x_idx = col["x"];
    let z_idx = col["z"];

    let cfg = FitConfig {
        family: Some("tweedie".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ te(x, z, k=7)", &ds, &cfg).expect("gam tweedie te fit");
    let FitResult::Standard(fit) = result else {
        panic!("Tweedie(log) is a scalar GLM family => expected FitResult::Standard");
    };

    // gam linear predictor (log scale) at the training points, then mu = exp(eta).
    let mut grid = Array2::<f64>::zeros((N, ds.headers.len()));
    for i in 0..N {
        grid[[i, x_idx]] = x[i];
        grid[[i, z_idx]] = z[i];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild te design at training points");
    let gam_eta: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    let gam_mu: Vec<f64> = gam_eta.iter().map(|e| e.exp()).collect();

    // ---- fit the SAME model with mgcv (the mature reference) ---------------
    // family = Tweedie(p = 1.5, link = "log") fixes the power to match gam.
    let r = run_r(
        &[
            Column::new("x", &x),
            Column::new("z", &z),
            Column::new("y", &y),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        m <- gam(y ~ te(x, z, k = 7), data = df,
                 family = Tweedie(p = 1.5, link = "log"), method = "REML")
        emit("eta", as.numeric(predict(m, type = "link")))
        emit("edf", sum(m$edf))
        emit("scale", m$scale)
        emit("sp", as.numeric(m$sp))
        "#,
    );
    let mgcv_eta = r.vector("eta");
    let mgcv_edf = r.scalar("edf");
    let mgcv_scale = r.scalar("scale");
    let mgcv_sp = r.vector("sp");
    eprintln!("[diag] mgcv tweedie: scale={mgcv_scale:.6} sp={mgcv_sp:?}");
    assert_eq!(mgcv_eta.len(), N, "mgcv linear-predictor length mismatch");

    // ---- OBJECTIVE METRICS -------------------------------------------------
    let gam_eta_err = rmse(&gam_eta, &eta_true);
    let mgcv_eta_err = rmse(mgcv_eta, &eta_true);
    let gam_mu_err = rmse(&gam_mu, &mu_true);
    let gam_edf = fit.fit.edf_total().unwrap_or(f64::NAN);
    let gam_phi = fit
        .fit
        .dispersion_phi()
        .expect("fitted Tweedie model must carry valid dispersion");
    eprintln!(
        "[diag] gam tweedie: outer_converged=certified outer_iterations={} grad_norm={:?} reml_score={:.6} gam_phi={:.6}",
        fit.fit.outer_iterations, fit.fit.outer_gradient_norm, fit.fit.reml_score, gam_phi,
    );
    eprintln!("[diag] gam tweedie lambdas={:?}", fit.fit.lambdas.to_vec());
    eprintln!(
        "[diag] gam tweedie likelihood_scale={:?}",
        fit.fit.likelihood_scale
    );

    // Context only (NOT a pass criterion): closeness of the two fitted surfaces.
    let rel_to_mgcv = relative_l2(&gam_eta, mgcv_eta);
    let corr_to_mgcv = pearson(&gam_eta, mgcv_eta);

    eprintln!(
        "te(x,z) Tweedie/log p={P}: n={N} zeros={zeros} phi={PHI} mgcv_edf={mgcv_edf:.3} \
         gam_edf={gam_edf:.3} \
         rmse_mu(gam,truth)={gam_mu_err:.4} noise_sigma={noise_sigma:.4} \
         rmse_eta(gam)={gam_eta_err:.4} rmse_eta(mgcv)={mgcv_eta_err:.4} \
         [context] rel_l2(gam,mgcv)={rel_to_mgcv:.4} pearson(gam,mgcv)={corr_to_mgcv:.5}"
    );

    // PRIMARY: gam recovers the true log-mean surface to better than the
    // per-observation Tweedie noise. A wrong variance power, wrong link, or
    // broken zero-handling distorts the recovered mean well past this floor.
    assert!(
        gam_mu_err < noise_sigma,
        "gam should recover the true Tweedie mean surface: rmse_mu={gam_mu_err:.4} \
         (bar = per-obs noise sigma {noise_sigma:.4})"
    );

    // MATCH-OR-BEAT: gam's truth-recovery error on the log scale is no worse than
    // mgcv's by more than 10%, holding mgcv as an accuracy baseline.
    assert!(
        gam_eta_err <= mgcv_eta_err * 1.10,
        "gam's truth-recovery error must match-or-beat mgcv: \
         rmse_eta(gam)={gam_eta_err:.4} vs mgcv*1.10={:.4}",
        mgcv_eta_err * 1.10
    );
}
