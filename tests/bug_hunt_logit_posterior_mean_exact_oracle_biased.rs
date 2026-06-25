//! Bug hunt (#1459): `quadrature::logit_posterior_mean_exact(mu, sigma)` is
//! documented as the EXACT oracle for `E[sigmoid(eta)]` with `eta ~ N(mu, sigma^2)`,
//! used to certify the cheap Gauss-Hermite `logit_posterior_mean` path. But it
//! carried a systematic, sigma-INDEPENDENT, mu-odd bias toward 0.5: at `mu = 3`
//! it was off by ~3.71e-5 for EVERY sigma, while the GHQ path matched truth to
//! ~1e-8.
//!
//! Root cause: the oracle summed the Faddeeva-series form
//! `1/2 - (sqrt(2*pi)/sigma) * Sum_{n>=1} Im w(z_n)` directly. After the
//! `sqrt(2*pi)/sigma` weighting, the terms decay only as `O(1/n)`, so the fixed
//! truncation (and a magnitude early-exit the slow tail never triggers) left an
//! `O(1/N)` remainder. That remainder is constant in sigma and proportional to
//! mu -- exactly the observed bias. Sharpening the `w(z)` evaluator does NOT fix
//! it; the defect is the truncated slow tail.
//!
//! This test pins the oracle against an INDEPENDENT dense-trapezoid reference
//! for `integral phi(z) * sigmoid(mu + sigma*z) dz` (standard-normal pdf `phi`,
//! NO Gauss-Hermite, NO Faddeeva) on a fine grid. It must FAIL before the fix
//! (errors ~1e-5 at the hard cases) and PASS after (errors < 1e-7).

use gam::inference::quadrature::{
    QuadratureContext, logit_posterior_mean, logit_posterior_mean_exact,
};

/// Standard-normal pdf.
fn phi(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Numerically stable sigmoid for the reference integrand.
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// INDEPENDENT reference for `E[sigmoid(eta)]`, `eta ~ N(mu, sigma^2)`, via a
/// dense composite-trapezoid rule on the standardized variable `z`, with
/// `eta = mu + sigma*z`. No Gauss-Hermite, no Faddeeva, no special functions:
/// just `phi` and `sigmoid` on a fine grid `z in [-12, 12]`, 200_000 points.
fn dense_trapezoid_reference(mu: f64, sigma: f64) -> f64 {
    let lo = -12.0_f64;
    let hi = 12.0_f64;
    let n: usize = 200_000;
    let h = (hi - lo) / (n as f64);
    let mut sum = 0.0_f64;
    for i in 0..=n {
        let z = lo + (i as f64) * h;
        let f = phi(z) * sigmoid(mu + sigma * z);
        let weight = if i == 0 || i == n { 0.5 } else { 1.0 };
        sum += weight * f;
    }
    sum * h
}

#[test]
fn logit_posterior_mean_exact_matches_dense_trapezoid_reference() {
    // The hard cases from #1459 (mu=3 had the largest ~3.71e-5 pre-fix bias),
    // spanning small / moderate / large sigma and both signs of mu, plus the
    // mu=0 anchor where the integrand is symmetric and the bias vanished.
    let cases = [
        (1.0_f64, 0.02_f64),
        (1.0, 0.5),
        (1.0, 2.0),
        (3.0, 0.05),
        (3.0, 0.5),
        (3.0, 2.0),
        (-2.0, 0.05),
        (-2.0, 2.0),
        (0.0, 0.5),
    ];

    let tol = 1e-7;
    for &(mu, sigma) in &cases {
        let reference = dense_trapezoid_reference(mu, sigma);
        let oracle = logit_posterior_mean_exact(mu, sigma);
        let err = (oracle - reference).abs();
        assert!(
            err < tol,
            "logit_posterior_mean_exact(mu={mu}, sigma={sigma}) = {oracle:.15} but dense-trapezoid reference = {reference:.15}; |error| = {err:.3e} >= {tol:.0e}",
        );
    }
}

#[test]
fn dense_trapezoid_reference_anchored_by_ghq_path() {
    // Sanity: anchor the independent reference against the cheap Gauss-Hermite
    // production path `logit_posterior_mean`, which #1459 reports already matches
    // truth to ~1e-8. This guards against a bug in the reference itself.
    let ctx = QuadratureContext::new();
    let cases = [
        (1.0_f64, 0.5_f64),
        (1.0, 2.0),
        (3.0, 0.5),
        (3.0, 2.0),
        (-2.0, 2.0),
        (0.0, 0.5),
    ];
    let tol = 1e-6;
    for &(mu, sigma) in &cases {
        let reference = dense_trapezoid_reference(mu, sigma);
        let ghq = logit_posterior_mean(&ctx, mu, sigma);
        let err = (ghq - reference).abs();
        assert!(
            err < tol,
            "GHQ path logit_posterior_mean(mu={mu}, sigma={sigma}) = {ghq:.15} disagrees with dense-trapezoid reference = {reference:.15}; |error| = {err:.3e} >= {tol:.0e}",
        );
    }
}
