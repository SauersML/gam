//! End-to-end quality: gam's Negative-Binomial(log, fixed theta) GLM must
//! RECOVER THE TRUE conditional-mean function from overdispersed count data with
//! a smooth + linear additive structure. The objective metric is accuracy
//! against the known generating truth, not agreement with any reference tool.
//!
//! Capability under test: `family="negative-binomial"` with a fixed
//! overdispersion parameter `theta`, log link, and the key feature combination
//! `y ~ s(x, k=5) + linear(z)` (a penalized smooth term plus a parametric
//! linear term). The negative-binomial variance function is
//! `Var(mu) = mu + mu^2 / theta`, so the IRLS/PIRLS working weights depend on
//! `theta`; recovering the true mean function is exactly the test that gam's
//! PIRLS incorporates the theta-dependent variance function correctly.
//!
//! Data (seed=678, n=220): x~U(0,10), z~U(-2,2);
//! eta = 1.0 + 0.6*sin(x*pi/5) + 0.4*z; y ~ NegBinom(mu=exp(eta), theta=2.0).
//! Because the data is generated from this known eta, the true conditional mean
//! mu_true = exp(eta) is KNOWN at every row, so we can score gam's fitted means
//! directly against ground truth.
//!
//! OBJECTIVE METRIC (primary claim): relative RMSE of gam's fitted means against
//! the true means, rRMSE = RMSE(mu_gam, mu_true) / range(mu_true). On a true
//! smooth+linear signal with NB(theta=2) noise, a correctly-specified penalized
//! NB-log fit recovers the conditional mean to a small fraction of the signal
//! range; we require rRMSE <= 0.15.
//!
//! BASELINE TO MATCH-OR-BEAT (accuracy, not output-agreement): the same NB-log
//! model is fit with statsmodels — `sm.GLM(y, X, NegativeBinomial(alpha=1/theta))`
//! on a 5-df cubic B-spline of x (mirroring s(x,k=5)) + z + intercept, same fixed
//! dispersion. We require gam's RMSE-to-truth <= statsmodels' RMSE-to-truth * 1.10.
//! statsmodels is a yardstick on the SAME objective (truth recovery); we never
//! assert gam reproduces statsmodels' fitted output. The reference rel_l2 between
//! the two fits is printed only as context.
//!
//! Asserts:
//!   1. rRMSE(mu_gam, mu_true) <= 0.15 — gam recovers the true mean function.
//!   2. RMSE(mu_gam, mu_true) <= RMSE(mu_sm, mu_true) * 1.10 — gam is at least as
//!      accurate at recovering the truth as the mature reference.
//!   3. Pearson chi-square / n in (0.8, 1.2). For a correctly-specified NB GLM
//!      the Pearson statistic sum((y-mu)^2 / V(mu)) with the NB variance
//!      V(mu) = mu + mu^2/theta has E[(y-mu)^2 / V(mu)] = 1 per row, so the
//!      statistic ~ n and chi2/n ~ 1. This confirms the theta-dependent variance
//!      the PIRLS weights bake in matches the data's overdispersion; a model
//!      ignoring theta (plain Poisson variance) drives this well above 1.2.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, relative_l2, rmse, run_python};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Gamma, Poisson, Uniform};

const N: usize = 220;
const THETA: f64 = 2.0;
const SEED: u64 = 678;

/// Sample one Negative-Binomial(mu, theta) count via the gamma-Poisson mixture:
/// lambda ~ Gamma(shape=theta, scale=mu/theta) so E[lambda]=mu and
/// Var(lambda)=mu^2/theta, then y ~ Poisson(lambda) giving the NB2 law with
/// Var(y) = mu + mu^2/theta — the same overdispersion gam and statsmodels model.
fn sample_negbin(mu: f64, theta: f64, rng: &mut StdRng) -> f64 {
    let gamma = Gamma::new(theta, mu / theta).expect("gamma params valid");
    let lambda = gamma.sample(rng);
    let pois = Poisson::new(lambda.max(1e-12)).expect("poisson rate valid");
    // rand_distr 0.6 `Poisson<f64>::sample` already returns the count as f64.
    pois.sample(rng)
}

#[test]
fn gam_negbin_matches_statsmodels_overdispersed_counts() {
    init_parallelism();

    // ---- synthesize identical data for both engines ----------------------
    let mut rng = StdRng::seed_from_u64(SEED);
    let ux = Uniform::new(0.0, 10.0).expect("uniform x");
    let uz = Uniform::new(-2.0, 2.0).expect("uniform z");

    let mut x = vec![0.0_f64; N];
    let mut z = vec![0.0_f64; N];
    let mut y = vec![0.0_f64; N];
    let mut mu_true = vec![0.0_f64; N];
    for i in 0..N {
        let xi = ux.sample(&mut rng);
        let zi = uz.sample(&mut rng);
        let eta = 1.0 + 0.6 * (xi * std::f64::consts::PI / 5.0).sin() + 0.4 * zi;
        let mu = eta.exp();
        x[i] = xi;
        z[i] = zi;
        mu_true[i] = mu;
        y[i] = sample_negbin(mu, THETA, &mut rng);
    }

    // ---- fit with gam: y ~ s(x, k=5) + linear(z), NB(log, theta=2) -------
    let headers = vec!["x".to_string(), "z".to_string(), "y".to_string()];
    let rows: Vec<StringRecord> = (0..N)
        .map(|i| StringRecord::from(vec![x[i].to_string(), z[i].to_string(), y[i].to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode negbin dataset");
    let col = ds.column_map();
    let x_idx = col["x"];
    let z_idx = col["z"];

    let cfg = FitConfig {
        family: Some("negative-binomial".to_string()),
        negative_binomial_theta: Some(THETA),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x, k=5) + linear(z)", &ds, &cfg).expect("gam negbin fit");
    let FitResult::Standard(fit) = result else {
        panic!("negative-binomial GLM should produce a Standard fit");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // gam fitted means at the training rows: rebuild the design from the frozen
    // spec, apply beta to get the log-link predictor eta, then exp() to means.
    let mut grid = Array2::<f64>::zeros((N, ds.headers.len()));
    for i in 0..N {
        grid[[i, x_idx]] = x[i];
        grid[[i, z_idx]] = z[i];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild design at training points");
    let gam_eta = design.design.apply(&fit.fit.beta);
    let gam_mu: Vec<f64> = gam_eta.iter().map(|e| e.exp()).collect();

    // ---- fit the SAME structure with statsmodels (the mature reference) ---
    // A 5-df cubic B-spline on x (mirrors s(x, k=5)) + z + intercept, NB-log,
    // dispersion alpha = 1/theta held fixed (statsmodels does not re-estimate it).
    let r = run_python(
        &[
            Column::new("x", &x),
            Column::new("z", &z),
            Column::new("y", &y),
        ],
        r#"
import numpy as np
import statsmodels.api as sm
from patsy import dmatrix

theta = 2.0
xv = np.asarray(df["x"], dtype=float)
zv = np.asarray(df["z"], dtype=float)
yv = np.asarray(df["y"], dtype=float)

# 5 spline columns on x to mirror gam's s(x, k=5); intercept supplied separately.
spl = np.asarray(dmatrix("bs(x, df=5, degree=3, include_intercept=False)",
                         {"x": xv}, return_type="dataframe"))
X = np.column_stack([np.ones_like(xv), spl, zv])

fam = sm.families.NegativeBinomial(alpha=1.0 / theta)  # alpha = 1/theta dispersion
m = sm.GLM(yv, X, family=fam).fit()
mu = np.asarray(m.fittedvalues, dtype=float)
emit("mu", mu)
"#,
    );
    let sm_mu = r.vector("mu");
    assert_eq!(sm_mu.len(), N, "statsmodels fitted-mean length mismatch");

    // ---- score BOTH fits against the KNOWN true conditional means --------
    // Truth recovery on the exp(eta) (fitted-mean) scale: mu_true is exact.
    let rmse_gam_truth = rmse(&gam_mu, &mu_true);
    let rmse_sm_truth = rmse(sm_mu, &mu_true);

    // Relative RMSE normalized by the dynamic range of the true mean function,
    // so the bar is scale-free and signal-appropriate.
    let mu_min = mu_true.iter().copied().fold(f64::INFINITY, f64::min);
    let mu_max = mu_true.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mu_range = (mu_max - mu_min).max(1e-12);
    let rrmse_gam = rmse_gam_truth / mu_range;

    // Pearson chi-square with the NB variance V(mu) = mu + mu^2/theta (this is
    // the statistic whose expectation is n, matching the cited (0.8,1.2) bound).
    let chi2_gam: f64 = (0..N)
        .map(|i| {
            let mu = gam_mu[i];
            let var = mu + mu * mu / THETA;
            let d = y[i] - mu;
            d * d / var.max(1e-12)
        })
        .sum();
    let chi2_over_n = chi2_gam / N as f64;

    // Reference output-agreement is printed for CONTEXT only — never asserted.
    let rel_l2_vs_sm = relative_l2(&gam_mu, sm_mu);
    eprintln!(
        "negbin s(x,k=5)+linear(z): n={N} theta={THETA} gam_edf={gam_edf:.3} \
         rmse(mu_gam,truth)={rmse_gam_truth:.4} rmse(mu_sm,truth)={rmse_sm_truth:.4} \
         rRMSE_gam={rrmse_gam:.4} chi2/n(NB-var)={chi2_over_n:.3} \
         rel_l2(gam,sm)={rel_l2_vs_sm:.4} (context only)"
    );

    // (1) PRIMARY: gam recovers the true conditional mean to a small fraction of
    // the signal range. NB(theta=2) noise on a smooth+linear signal of this size
    // is fit by a correctly-specified penalized NB-log model to rRMSE well under
    // 0.15; failure here means gam mis-recovers the mean, not "differs from a tool".
    assert!(
        rrmse_gam <= 0.15,
        "gam failed to recover the true NB mean: rRMSE={rrmse_gam:.4} (rmse={rmse_gam_truth:.4}, range={mu_range:.4})"
    );

    // (2) MATCH-OR-BEAT (accuracy on the SAME objective): gam is at least as
    // accurate at recovering the truth as the mature statsmodels NB-log fit.
    assert!(
        rmse_gam_truth <= rmse_sm_truth * 1.10,
        "gam less accurate than statsmodels at recovering truth: \
         rmse_gam={rmse_gam_truth:.4} > 1.10 * rmse_sm={rmse_sm_truth:.4}"
    );

    // (3) Dispersion sanity: under the NB variance V(mu)=mu+mu^2/theta,
    // E[(y-mu)^2/V(mu)] = 1, so chi2/n ~ 1; the (0.8,1.2) window confirms the
    // theta-dependent variance the PIRLS weights bake in matches the data's
    // overdispersion. A model ignoring theta drives this well above 1.2.
    assert!(
        (0.8..1.2).contains(&chi2_over_n),
        "Pearson chi2/n (NB variance) outside (0.8,1.2): {chi2_over_n:.3}"
    );
}
