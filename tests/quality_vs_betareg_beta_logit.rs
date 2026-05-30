//! End-to-end quality: gam's Beta(logit, phi) regression on proportions in the
//! open interval (0,1) must agree with R's `betareg` — the canonical, mature
//! reference for beta regression — on the same data.
//!
//! Beta regression is the standard model for bounded continuous outcomes
//! (proportions, rates). The mean `mu` is modelled on the logit scale and a
//! precision parameter `phi` controls the variance
//! `Var(y) = mu (1 - mu) / (1 + phi)`. `betareg` (Cribari-Neto & Zeileis, JSS
//! 2010) is the reference implementation in R; statsmodels has no beta family,
//! so `betareg` is the best-in-class choice here.
//!
//! We fit the additive logit-link beta model `y ~ s(x1, k=4) + s(x2, k=3)` with
//! gam, and the *same* additive structure with `betareg` using natural-spline
//! bases (`splines::ns`, df = k-1 to match the rank of each penalised smooth's
//! range space) and a logit link. Both maximise the *same* beta log-likelihood
//! with a smooth additive logit-mean and a shared precision `phi`, so on data
//! drawn from a smooth truth their fitted mean curves must essentially coincide.
//!
//! We compare on the quantity that matters for a proportions model:
//!   1. the fitted proportions `mu_hat = logit^{-1}(eta_hat)` (Pearson), and
//!   2. the linear-predictor (logit-scale) fits `eta_hat` (RMSE).
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
fn logit(p: f64) -> f64 {
    (p / (1.0 - p)).ln()
}

#[inline]
fn mu_eta_truth(x1: f64, x2: f64) -> f64 {
    use std::f64::consts::PI;
    0.5 + 0.3 * (x1 * PI / 4.0).sin() + 0.2 * (x2 * PI / 4.0).cos()
}

#[test]
fn gam_beta_logit_matches_betareg() {
    init_parallelism();

    // ---- synthesize identical data for both engines -----------------------
    // x1, x2 ~ U(0,8); truth on logit scale; y ~ Beta(mu*phi, (1-mu)*phi).
    let mut rng = StdRng::seed_from_u64(SEED);
    let ux = Uniform::new(0.0, 8.0).expect("uniform(0,8)");
    let mut x1 = Vec::<f64>::with_capacity(N);
    let mut x2 = Vec::<f64>::with_capacity(N);
    let mut y = Vec::<f64>::with_capacity(N);
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
    }

    // ---- fit with gam: y ~ s(x1,k=4) + s(x2,k=3), family = beta (logit) ----
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
    let result = fit_from_formula("y ~ s(x1, k=4) + s(x2, k=3)", &ds, &cfg).expect("gam beta fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for the beta family");
    };

    // gam fitted linear predictor at the training points: rebuild the design
    // from the frozen spec and apply beta (logit link => eta = design*beta).
    let mut grid = Array2::<f64>::zeros((N, ds.headers.len()));
    for i in 0..N {
        grid[[i, x1_idx]] = x1[i];
        grid[[i, x2_idx]] = x2[i];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild design at training points");
    let gam_eta: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    let gam_mu: Vec<f64> = gam_eta.iter().map(|&e| logit_inv(e)).collect();

    // ---- fit the SAME additive model with betareg (the mature reference) ---
    // Natural-spline bases match the rank of each penalised smooth's range
    // space (df = k - 1: s(x1,k=4)->df 3, s(x2,k=3)->df 2). Logit link, shared
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
        m <- betareg(y ~ ns(x1, df = 3) + ns(x2, df = 2), data = df, link = "logit")
        mu <- as.numeric(predict(m, type = "response"))
        emit("mu", mu)
        emit("phi", as.numeric(m$coefficients$precision))
        "#,
    );
    let betareg_mu = r.vector("mu");
    let betareg_phi = r.scalar("phi");
    assert_eq!(betareg_mu.len(), N, "betareg fitted length mismatch");
    let betareg_eta: Vec<f64> = betareg_mu.iter().map(|&m| logit(m)).collect();

    // betareg's `precision` coefficient is phi on the natural scale (constant
    // precision => identity phi-link by default), i.e. the SAME mean/precision
    // parameterisation used to draw the data: Var = mu(1-mu)/(1+phi). Confirm it
    // recovers the true PHI within a wide band — this guards against a precision-
    // vs-dispersion mismatch in the comparator (a dispersion 1/phi ~= 0.05 or a
    // variance-style sigma would fall far outside [10, 40]). The band is loose
    // because phi is only weakly identified from n=150 boundary-crowded draws
    // (its sampling CV is O(sqrt(2/n)) ~ 12%, plus a downward bias from the basis
    // misfit of the full-period x2 cosine under df=2), but it is a real check that
    // both engines share the mean/precision parameterisation of the data.
    assert!(
        betareg_phi.is_finite() && betareg_phi > 0.5 * PHI && betareg_phi < 2.0 * PHI,
        "betareg precision phi={betareg_phi:.3} is implausible for true phi={PHI} \
         (wrong precision parameterisation?)"
    );

    // ---- compare on the proportions and the logit-scale predictions -------
    let corr_mu = pearson(&gam_mu, betareg_mu);
    let rmse_eta = rmse(&gam_eta, &betareg_eta);

    // also surface how well each engine recovers the smooth truth, for context.
    let mu_truth: Vec<f64> = (0..N)
        .map(|i| logit_inv(mu_eta_truth(x1[i], x2[i])))
        .collect();
    let gam_vs_truth = pearson(&gam_mu, &mu_truth);
    let betareg_vs_truth = pearson(betareg_mu, &mu_truth);

    eprintln!(
        "beta-logit: n={N} phi_true={PHI} betareg_phi={betareg_phi:.3} \
         pearson(mu)={corr_mu:.5} rmse(eta)={rmse_eta:.5} \
         gam_vs_truth={gam_vs_truth:.4} betareg_vs_truth={betareg_vs_truth:.4}"
    );

    // Both engines maximise the same beta log-likelihood with a smooth additive
    // logit-mean and a shared precision, so on a smooth truth their fitted
    // proportions must be near-identical and their logit-scale predictions must
    // agree tightly. The differences that remain come only from the basis
    // convention (gam's penalised range space vs betareg's natural-spline df)
    // and gam's REML smoothing penalty (mild shrinkage that betareg lacks);
    // these are small relative to the signal here.
    //
    // pearson(mu) is the scale/level-free shape match: >0.99 means the two fitted
    // proportion curves trace the same shape, robust to the constant offset that
    // penalised vs unpenalised fits can introduce. rmse(eta) is on the logit
    // scale where the truth spans eta in [0.5 - 0.5, 0.5 + 0.5] = [0, 1] (a unit
    // range); 0.08 RMSE is <8% of that span — loose enough to absorb the basis-
    // convention and REML-shrinkage differences, tight enough that any genuine
    // smoother divergence (which would be O(0.2+) at the data extremes) fails.
    assert!(
        corr_mu > 0.99,
        "fitted proportions diverge from betareg: pearson(mu)={corr_mu:.5}"
    );
    assert!(
        rmse_eta < 0.08,
        "logit-scale predictions diverge from betareg: rmse(eta)={rmse_eta:.5}"
    );
}
