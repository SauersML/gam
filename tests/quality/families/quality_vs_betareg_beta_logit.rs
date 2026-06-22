//! End-to-end quality: gam's Beta(logit, phi) regression must RECOVER the known
//! smooth mean function it was simulated from.
//!
//! OBJECTIVE METRIC (truth recovery, not tool agreement): the data are drawn
//! from a known logit-scale truth `eta = mu_eta_truth(x1, x2)` with Beta noise of
//! precision `phi=20`. We assert the PRIMARY claim directly:
//!   * RMSE(gam_mu_hat, mu_truth) on the (0,1) proportion scale is at most the
//!     per-observation Beta noise standard deviation evaluated at the truth
//!     (`sd_bar = sqrt(mean(mu(1-mu)) / (1 + phi))`). A smoother that recovers the
//!     mean must average the noise down well below one noise sd; this bar is a
//!     genuine accuracy floor, not a same-as-reference check.
//!
//! Beta regression is the standard model for bounded continuous outcomes
//! (proportions, rates): the mean `mu` is modelled on the logit scale and a
//! precision `phi` controls the variance `Var(y) = mu (1 - mu) / (1 + phi)`.
//!
//! `betareg` (Cribari-Neto & Zeileis, JSS 2010), the canonical R beta-regression
//! implementation, is fit on the SAME data as a BASELINE TO MATCH-OR-BEAT on the
//! same truth-recovery metric: we additionally require gam's RMSE-to-truth to be
//! no worse than 1.10x betareg's RMSE-to-truth. It is NOT used as ground truth —
//! reproducing its (noisy) fitted output would prove nothing about accuracy.
//! gam's strict-(0,1) boundary handling is exercised by feeding it genuine Beta
//! draws that crowd toward the edges at this precision.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Beta, Distribution, Uniform};

const N: usize = 150;
const SEED: u64 = 234;
const PHI: f64 = 20.0;

#[inline]
fn logit_inv(eta: f64) -> f64 {
    1.0 / (1.0 + (-eta).exp())
}

#[inline]
fn mu_eta_truth(x1: f64, x2: f64) -> f64 {
    use std::f64::consts::PI;
    0.5 + 0.3 * (x1 * PI / 4.0).sin() + 0.2 * (x2 * PI / 4.0).cos()
}

#[test]
fn gam_beta_logit_recovers_smooth_truth() {
    init_parallelism();

    // ---- synthesize identical data for both engines -----------------------
    // x1, x2 ~ U(0,8); truth on logit scale; y ~ Beta(mu*phi, (1-mu)*phi).
    let mut rng = StdRng::seed_from_u64(SEED);
    let ux = Uniform::new(0.0, 8.0).expect("uniform(0,8)");
    let mut x1 = Vec::<f64>::with_capacity(N);
    let mut x2 = Vec::<f64>::with_capacity(N);
    let mut y = Vec::<f64>::with_capacity(N);
    let mut mu_truth = Vec::<f64>::with_capacity(N);
    for _ in 0..N {
        let a = ux.sample(&mut rng);
        let b = ux.sample(&mut rng);
        let mu = logit_inv(mu_eta_truth(a, b));
        // Beta shape parameters from mean/precision parameterisation.
        let beta = Beta::new(mu * PHI, (1.0 - mu) * PHI).expect("beta shapes > 0");
        // Keep draws strictly inside (0,1) so the beta likelihood is finite for
        // both engines (betareg and gam both require open-interval responses).
        let mut yi = beta.sample(&mut rng);
        yi = yi.clamp(1e-6, 1.0 - 1e-6);
        x1.push(a);
        x2.push(b);
        y.push(yi);
        mu_truth.push(mu);
    }

    // Per-observation Beta noise standard deviation at the truth, averaged over
    // the design — the natural accuracy floor for recovering the mean.
    // Var(y_i) = mu_i (1 - mu_i) / (1 + phi); sd_bar = sqrt(mean Var).
    let noise_sd_bar = {
        let mean_var: f64 = mu_truth
            .iter()
            .map(|&m| m * (1.0 - m) / (1.0 + PHI))
            .sum::<f64>()
            / N as f64;
        mean_var.sqrt()
    };

    // ---- fit with gam: y ~ s(x1,k=4) + s(x2,k=4), family = beta (logit) ----
    let headers = vec!["y".to_string(), "x1".to_string(), "x2".to_string()];
    let rows: Vec<csv::StringRecord> = (0..N)
        .map(|i| {
            csv::StringRecord::from(vec![y[i].to_string(), x1[i].to_string(), x2[i].to_string()])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode beta dataset");
    let col = ds.column_map();
    let x1_idx = col["x1"];
    let x2_idx = col["x2"];

    let cfg = FitConfig {
        family: Some("beta".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x1, k=4) + s(x2, k=4)", &ds, &cfg).expect("gam beta fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for the beta family");
    };

    // gam fitted proportions at the training points: rebuild the design from
    // the frozen spec, form eta = design*beta, then mu = logit^{-1}(eta).
    let mut grid = Array2::<f64>::zeros((N, ds.headers.len()));
    for i in 0..N {
        grid[[i, x1_idx]] = x1[i];
        grid[[i, x2_idx]] = x2[i];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild design at training points");
    let gam_mu: Vec<f64> = design
        .design
        .apply(&fit.fit.beta)
        .iter()
        .map(|&e| logit_inv(e))
        .collect();

    // ---- fit the SAME additive model with betareg (the mature reference) ---
    // Natural-spline bases match the rank of each penalised smooth's range
    // space (df = k - 1: s(x1,k=4)->df 3, s(x2,k=4)->df 3). Logit link, shared
    // precision phi. emit the fitted means; we reconstruct eta = logit(mu).
    let r = run_r(
        &[
            Column::new("y", &y),
            Column::new("x1", &x1),
            Column::new("x2", &x2),
        ],
        r#"
        suppressPackageStartupMessages(library(betareg))
        suppressPackageStartupMessages(library(splines))
        m <- betareg(y ~ ns(x1, df = 3) + ns(x2, df = 3), data = df, link = "logit")
        mu <- as.numeric(predict(m, type = "response"))
        emit("mu", mu)
        emit("phi", as.numeric(m$coefficients$precision))
        "#,
    );
    let betareg_mu = r.vector("mu");
    let betareg_phi = r.scalar("phi");
    assert_eq!(betareg_mu.len(), N, "betareg fitted length mismatch");

    // betareg's `precision` coefficient is phi on the natural scale (constant
    // precision => identity phi-link by default), i.e. the SAME mean/precision
    // parameterisation used to draw the data: Var = mu(1-mu)/(1+phi). Confirm it
    // recovers the true PHI within a wide band — this guards against a precision-
    // vs-dispersion mismatch in the comparator (a dispersion 1/phi ~= 0.05 or a
    // variance-style sigma would fall far outside [10, 40]). The band is loose
    // because phi is only weakly identified from n=150 boundary-crowded draws
    // (its sampling CV is O(sqrt(2/n)) ~ 12%, plus a mild downward bias from the
    // residual basis misfit of the smooth truth), but it is a real check that
    // both engines share the mean/precision parameterisation of the data.
    assert!(
        betareg_phi.is_finite() && betareg_phi > 0.5 * PHI && betareg_phi < 2.0 * PHI,
        "betareg precision phi={betareg_phi:.3} is implausible for true phi={PHI} \
         (wrong precision parameterisation?)"
    );

    // ---- OBJECTIVE truth recovery on the (0,1) proportion scale -----------
    // PRIMARY claim: gam recovers the known smooth mean. Compute gam's RMSE to
    // the simulated truth (computed on gam's OWN fitted proportions).
    let gam_rmse_truth = rmse(&gam_mu, &mu_truth);

    // BASELINE TO MATCH-OR-BEAT: betareg's RMSE to the same truth, on its own
    // fitted proportions. Reproducing betareg is NOT the goal; out-accuracy on
    // the truth (within a 10% tolerance) is.
    let betareg_rmse_truth = rmse(betareg_mu, &mu_truth);

    // For context only — shape agreement with the reference, not a pass gate.
    let corr_mu_ref = pearson(&gam_mu, betareg_mu);

    eprintln!(
        "beta-logit truth-recovery: n={N} phi_true={PHI} betareg_phi={betareg_phi:.3} \
         noise_sd_bar={noise_sd_bar:.5} gam_rmse_truth={gam_rmse_truth:.5} \
         betareg_rmse_truth={betareg_rmse_truth:.5} pearson(mu_gam,mu_betareg)={corr_mu_ref:.5}"
    );

    // (1) ACCURACY FLOOR: a smoother that recovers the mean averages the Beta
    // noise down well below one per-observation noise sd. We require gam's
    // RMSE-to-truth to sit under that noise sd — a genuine objective accuracy
    // bar (not a same-as-reference check).
    assert!(
        gam_rmse_truth <= noise_sd_bar,
        "gam did not recover the smooth truth: rmse(mu_hat, mu_truth)={gam_rmse_truth:.5} \
         exceeds the per-observation noise sd bar {noise_sd_bar:.5}"
    );

    // (2) MATCH-OR-BEAT the mature reference on the SAME objective metric:
    // gam's truth-recovery error must be no worse than 1.10x betareg's.
    assert!(
        gam_rmse_truth <= betareg_rmse_truth * 1.10,
        "gam's truth-recovery error rmse(mu)={gam_rmse_truth:.5} is worse than \
         betareg's {betareg_rmse_truth:.5} by more than 10%"
    );
}
