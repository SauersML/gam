//! End-to-end quality: gam's 1-D smooth `s(x)` under the Negative-Binomial
//! family with **ESTIMATED overdispersion `theta`** (log link) must RECOVER both
//! the true smooth log-mean function AND the true `theta` from overdispersed
//! counts — judged against mgcv's `nb()` family, which is the canonical peer that
//! ALSO estimates theta jointly with the REML smoothing parameter.
//!
//! ## Why this combination earns its own test (the #1471 coverage gap)
//!
//! Every existing NB-vs-reference quality test pins theta to a FIXED, known value
//! and hands the SAME fixed theta to both gam and the reference:
//!   * `quality_vs_statsmodels_negbin`     — `s(x)+linear(z)`, FIXED theta vs statsmodels.
//!   * `quality_vs_mgcv_tensor_te_2d_negbin`— `te(x,z)`, `negbin(theta=3)` FIXED vs mgcv.
//! gam's NB family was therefore validated against mgcv only for the mean given a
//! known theta, and against statsmodels only for coefficient SEs — **never** for
//! the joint estimation of `theta` against mgcv's `nb()` family. That joint path
//! is the interesting one: theta enters the NB2 variance `Var=mu+mu^2/theta`, so
//! the PIRLS working weight `W=mu*theta/(theta+mu)` depends on theta, which in
//! turn depends on the fitted mean, which depends on the REML-selected smoothing
//! parameter lambda. gam resolves this with an OUTER theta<->lambda alternation
//! (#1448/#1463): freeze theta during the lambda search, ML-refresh theta at the
//! converged eta, and re-run the lambda search if theta drifted past tolerance.
//! mgcv's `nb()` does the analogous nested estimation. A divergence here would be
//! a real, previously-unguarded bug (a #1426-class over/under-smoothing that, for
//! a non-Gaussian family, also corrupts the recovered theta).
//!
//! ## Data
//!
//! seed=20260622, n=400: x ~ U(0,1); truth on the log-mean scale
//! `eta_true = 1.6 + 1.1*sin(2*pi*x)` => mu = exp(eta) in ~[1.65,14.9];
//! y ~ NegBinom(mu, theta_true=4) via the gamma-Poisson mixture. The SAME draws
//! (fixed seed) feed gam and mgcv.
//!
//! ## What is asserted (OBJECTIVE truth recovery + match-or-beat)
//!
//!  1. SMOOTH truth recovery: RMSE(gam_eta, eta_true) < 0.30 (the wiggly part
//!     1.1*sin spans [-1.1,1.1], range 2.2; a correct estimated-theta NB-log fit
//!     stays well inside ~14% of that span at n=400).
//!  2. SMOOTH match-or-beat: gam's RMSE-to-truth <= mgcv's RMSE-to-truth * 1.10.
//!  3. THETA truth recovery: gam's ESTIMATED theta_hat lands within 40% of
//!     theta_true=4 (theta is a notoriously high-variance estimate; the band is
//!     wide on purpose but excludes the frozen seed 1.0 and the Poisson clamp).
//!  4. THETA match-or-beat: gam's |theta_hat - theta_true| no worse than mgcv's
//!     own |theta_mgcv - theta_true| by more than a generous slack — theta is so
//!     high-variance that we require gam land in the SAME ballpark as mgcv, never
//!     reproduce it. (Absolute relative agreement gam-vs-mgcv printed for context.)
//!  5. NB-variance Pearson chi2/n near 1 with the recovered theta.

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
use std::time::Instant;

const N: usize = 400;
const THETA_TRUE: f64 = 4.0;

/// Sample one Negative-Binomial(mu, theta) count via the gamma-Poisson mixture:
/// `lambda ~ Gamma(shape=theta, scale=mu/theta)` so `E[lambda]=mu` and
/// `Var(lambda)=mu^2/theta`, then `y ~ Poisson(lambda)`, giving the NB2 law with
/// `Var(y) = mu + mu^2/theta`.
fn sample_negbin(mu: f64, theta: f64, rng: &mut StdRng) -> f64 {
    let gamma = Gamma::new(theta, mu / theta).expect("gamma params valid");
    let lambda = gamma.sample(rng).max(1e-12);
    Poisson::new(lambda)
        .expect("poisson rate valid")
        .sample(rng)
}

#[test]
fn gam_negbin_estimated_theta_smooth_matches_mgcv_nb() {
    init_parallelism();

    // ---- synthetic overdispersed-count truth on the unit interval ----------
    // eta_true = 1.6 + 1.1*sin(2*pi*x); mu = exp(eta); y ~ NegBinom(mu, theta=4).
    let mut rng = StdRng::seed_from_u64(20260622);
    let u = Uniform::new(0.0_f64, 1.0).expect("uniform [0,1]");

    let mut x = Vec::with_capacity(N);
    let mut y = Vec::with_capacity(N);
    let mut eta_true = Vec::with_capacity(N);
    for _ in 0..N {
        let xi = u.sample(&mut rng);
        let eta = 1.6 + 1.1 * (2.0 * PI * xi).sin();
        let mu = eta.exp();
        x.push(xi);
        y.push(sample_negbin(mu, THETA_TRUE, &mut rng));
        eta_true.push(eta);
    }

    // ---- fit with gam: y ~ s(x, k=12), NB(log), theta ESTIMATED, REML ------
    // CRITICAL: no `negative_binomial_theta` => `theta_fixed = false` => gam
    // ML-estimates theta jointly with the mean and the REML smoothing parameter
    // (the #1448/#1463 outer theta<->lambda alternation). This is the path mgcv's
    // nb() family exercises and the one no prior mgcv test covered.
    let headers = vec!["x".to_string(), "y".to_string()];
    let rows: Vec<csv::StringRecord> = (0..N)
        .map(|i| csv::StringRecord::from(vec![x[i].to_string(), (y[i] as i64).to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode negbin dataset");
    let col = ds.column_map();
    let x_idx = col["x"];

    let cfg = FitConfig {
        family: Some("negative-binomial".to_string()),
        // No negative_binomial_theta: theta is estimated from the data.
        ..FitConfig::default()
    };
    let gam_t0 = Instant::now();
    let result = fit_from_formula("y ~ s(x, k=12)", &ds, &cfg).expect("gam negbin nb() fit");
    let gam_elapsed = gam_t0.elapsed();
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for NB s(x) with estimated theta");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");
    let gam_theta = fit
        .fit
        .likelihood_scale
        .negbin_theta()
        .expect("NB fit with estimated theta must record theta_hat in likelihood_scale");

    // gam linear predictor (log scale) at the training points, then mu = exp(eta).
    let mut grid = Array2::<f64>::zeros((N, ds.headers.len()));
    for i in 0..N {
        grid[[i, x_idx]] = x[i];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild s(x) design at training points");
    let gam_eta: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    let gam_mu: Vec<f64> = gam_eta.iter().map(|e| e.exp()).collect();

    // ---- fit the SAME model with mgcv nb() (the mature reference) ----------
    // family = nb() (NOT negbin(theta=...)): mgcv ESTIMATES theta jointly with the
    // REML smoothing parameter, exactly mirroring gam's estimated-theta path. We
    // read mgcv's converged theta off the fitted family (`getTheta`), its link-
    // scale fit, and its edf.
    let r_t0 = Instant::now();
    let r = run_r(
        &[Column::new("x", &x), Column::new("y", &y)],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        m <- gam(y ~ s(x, k = 12), data = df, family = nb(), method = "REML")
        emit("eta", as.numeric(predict(m, type = "link")))
        emit("theta", as.numeric(m$family$getTheta(TRUE)))
        emit("edf", sum(m$edf))
        "#,
    );
    let r_elapsed = r_t0.elapsed();
    let mgcv_eta = r.vector("eta");
    let mgcv_theta = r.scalar("theta");
    let mgcv_edf = r.scalar("edf");
    assert_eq!(mgcv_eta.len(), N, "mgcv linear-predictor length mismatch");

    // ---- OBJECTIVE METRICS -------------------------------------------------
    let gam_err = rmse(&gam_eta, &eta_true);
    let mgcv_err = rmse(mgcv_eta, &eta_true);

    // theta truth-recovery errors (absolute) and gam-vs-mgcv relative agreement.
    let gam_theta_err = (gam_theta - THETA_TRUE).abs();
    let mgcv_theta_err = (mgcv_theta - THETA_TRUE).abs();
    let theta_rel_to_mgcv = (gam_theta - mgcv_theta).abs() / mgcv_theta.max(1e-12);

    // NB-variance Pearson statistic with the RECOVERED theta: under
    // Var=mu+mu^2/theta_hat, E[(y-mu)^2/Var] = 1, so this lands near 1 when both
    // the mean and the estimated overdispersion are correct.
    let chi2_over_n: f64 = (0..N)
        .map(|i| {
            let mu = gam_mu[i];
            let var = mu + mu * mu / gam_theta;
            let d = y[i] - mu;
            d * d / var.max(1e-12)
        })
        .sum::<f64>()
        / N as f64;

    let rel_to_mgcv = relative_l2(&gam_eta, mgcv_eta);
    let corr_to_mgcv = pearson(&gam_eta, mgcv_eta);

    eprintln!(
        "s(x) NB/log ESTIMATED-theta: n={N} gam_wall={:.2}s r_mgcv_wall={:.2}s \
         theta_true={THETA_TRUE} gam_theta={gam_theta:.4} mgcv_theta={mgcv_theta:.4} \
         gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} \
         rmse_to_truth(gam)={gam_err:.4} rmse_to_truth(mgcv)={mgcv_err:.4} \
         theta_err(gam)={gam_theta_err:.4} theta_err(mgcv)={mgcv_theta_err:.4} \
         chi2/n(NB-var)={chi2_over_n:.3} \
         [context] rel_l2(gam,mgcv)={rel_to_mgcv:.4} pearson(gam,mgcv)={corr_to_mgcv:.5} \
         theta_rel(gam,mgcv)={theta_rel_to_mgcv:.4}",
        gam_elapsed.as_secs_f64(),
        r_elapsed.as_secs_f64(),
    );

    // (1) PRIMARY smooth truth-recovery: gam recovers the true log-mean curve.
    // The wiggly part 1.1*sin(2*pi*x) spans [-1.1,1.1] (range 2.2); with n=400 and
    // NB(theta=4) overdispersion a correct estimated-theta log-link fit stays well
    // inside ~14% of that span.
    assert!(
        gam_err < 0.30,
        "gam should recover the true NB log-mean curve with estimated theta: \
         rmse_to_truth={gam_err:.4} (bar 0.30)"
    );

    // (2) SMOOTH match-or-beat: no worse than mgcv's nb() by more than 10% on the
    // truth-recovery objective.
    assert!(
        gam_err <= mgcv_err * 1.10,
        "gam's smooth truth-recovery must match-or-beat mgcv nb(): \
         rmse_to_truth(gam)={gam_err:.4} vs mgcv*1.10={:.4}",
        mgcv_err * 1.10
    );

    // (3) THETA truth-recovery: gam's ESTIMATED theta lands within 40% of the
    // true theta=4. theta is a high-variance estimate (the band is deliberately
    // wide) but this still excludes the frozen seed 1.0 (75% low) and the Poisson
    // clamp (1e6), so it is a real claim that the alternation converged near truth.
    assert!(
        gam_theta > THETA_TRUE * 0.6 && gam_theta < THETA_TRUE * 1.4,
        "gam estimated theta {gam_theta:.4} must recover theta_true={THETA_TRUE} \
         within 40% (band {:.2}..{:.2}); frozen-seed 1.0 or Poisson-clamp would fail",
        THETA_TRUE * 0.6,
        THETA_TRUE * 1.4
    );

    // (4) THETA match-or-beat mgcv on truth-recovery. theta is high-variance, so
    // the slack is generous (gam's theta error no worse than mgcv's by more than
    // 0.5 of true theta) — this asserts gam lands in the SAME ballpark as mgcv's
    // own nb() theta, never that it reproduces it.
    assert!(
        gam_theta_err <= mgcv_theta_err + 0.5 * THETA_TRUE,
        "gam estimated-theta error {gam_theta_err:.4} must be in mgcv's ballpark: \
         mgcv theta error {mgcv_theta_err:.4} + slack {:.4}",
        0.5 * THETA_TRUE
    );

    // (5) NB-variance Pearson chi2/n sits near 1 with the recovered theta. A model
    // that under-estimates theta (over-states overdispersion) drives this below 1;
    // ignoring theta (plain Poisson variance) drives it well above.
    assert!(
        (0.75..1.25).contains(&chi2_over_n),
        "NB-variance Pearson chi2/n outside (0.75,1.25) with recovered theta: {chi2_over_n:.3}"
    );

    // sane complexity: more than a line, well under the basis dimension k=12.
    assert!(
        gam_edf > 1.5 && gam_edf < 12.0,
        "gam effective complexity out of range: edf={gam_edf:.3} (expected 1.5 < edf < 12)"
    );
}
