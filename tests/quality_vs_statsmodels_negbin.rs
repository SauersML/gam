//! End-to-end quality: gam's Negative-Binomial(log, fixed theta) GLM must match
//! statsmodels — the mature, standard Python GLM implementation — on
//! overdispersed count data with a smooth + linear additive structure.
//!
//! Capability under test: `family="negative-binomial"` with a fixed
//! overdispersion parameter `theta`, log link, and the key feature combination
//! `y ~ s(x, k=5) + linear(z)` (a penalized smooth term plus a parametric
//! linear term). The negative-binomial variance function is
//! `Var(mu) = mu + mu^2 / theta`, so the IRLS/PIRLS working weights depend on
//! `theta`; getting the fitted means right is exactly the test that gam's PIRLS
//! incorporates the theta-dependent variance function correctly.
//!
//! Reference: `statsmodels.api.GLM(y, X, family=NegativeBinomial(alpha=1/theta))`.
//! statsmodels parameterizes the NB family by the dispersion `alpha = 1/theta`
//! (so theta=2.0 => alpha=0.5), with the SAME log link and the SAME fixed
//! dispersion held constant during fitting that gam uses. We hand statsmodels a
//! design that mirrors the gam formula: a 5-column cubic B-spline basis on `x`
//! (matching `s(x, k=5)`) plus the raw `z` column (`linear(z)`) plus an
//! intercept. statsmodels is unpenalized; gam REML-penalizes the smooth, so the
//! two fitted-mean vectors are not bit-identical, but on a true smooth signal
//! the recovered conditional means must track each other tightly.
//!
//! Data (seed=678, n=220): x~U(0,10), z~U(-2,2);
//! eta = 1.0 + 0.6*sin(x*pi/5) + 0.4*z; y ~ NegBinom(mu=exp(eta), theta=2.0).
//!
//! Asserts:
//!   1. pearson(gam fitted means, statsmodels fitted means) > 0.995 on the
//!      exp(eta) scale — both are MLE/penalized-MLE NB-log fits of the same
//!      smooth+linear structure, so their fitted means must nearly coincide.
//!   2. Pearson chi-square / n in (0.8, 1.2). For a correctly-specified NB GLM
//!      the Pearson statistic sum((y-mu)^2 / V(mu)) with the NB variance
//!      V(mu) = mu + mu^2/theta has E[(y-mu)^2 / V(mu)] = 1 per row, so the
//!      statistic ~ n and chi2/n ~ 1. This is the standard dispersion check:
//!      values near 1 confirm the modeled variance (which BAKES IN theta)
//!      matches the data's overdispersion. The spec writes the chi2 with a
//!      mu_hat denominator, but the (0.8,1.2) bound it cites is the NB
//!      dispersion=1/theta statistic, which is only ~1 under the NB variance;
//!      we therefore use V(mu) so the asserted bound and its stated rationale
//!      agree. A model that ignored theta (e.g. plain Poisson variance) would
//!      drive this far above 1.2.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, run_python};
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism};
use csv::StringRecord;
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
    let result =
        fit_from_formula("y ~ s(x, k=5) + linear(z)", &ds, &cfg).expect("gam negbin fit");
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

    // ---- compare on the exp(eta) (fitted mean) scale ---------------------
    let corr = pearson(&gam_mu, sm_mu);

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

    eprintln!(
        "negbin s(x,k=5)+linear(z): n={N} theta={THETA} gam_edf={gam_edf:.3} \
         pearson(mu_gam,mu_sm)={corr:.5} chi2/n(NB-var)={chi2_over_n:.3}"
    );

    // (1) Both engines fit a NB-log smooth+linear model to identical data with
    // the same fixed theta, so their conditional means must nearly coincide.
    assert!(
        corr > 0.995,
        "gam vs statsmodels NB fitted means diverge: pearson={corr:.5}"
    );

    // (2) Dispersion sanity: under the NB variance V(mu)=mu+mu^2/theta,
    // E[(y-mu)^2/V(mu)] = 1, so chi2/n ~ 1; the (0.8,1.2) window confirms the
    // theta-dependent variance the PIRLS weights bake in matches the data's
    // overdispersion. A model ignoring theta drives this well above 1.2.
    assert!(
        (0.8..1.2).contains(&chi2_over_n),
        "Pearson chi2/n (NB variance) outside (0.8,1.2): {chi2_over_n:.3}"
    );
}
