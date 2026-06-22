//! Bug hunt: `integrated_inverse_link_mean_and_derivative` is inaccurate for
//! the complementary-log-log link at large latent sigma.
//!
//! For eta ~ N(mu, sigma^2) and the cloglog inverse link
//! g^{-1}(x) = 1 - exp(-exp(x)), both E[g^{-1}(eta)] and its mu-derivative
//! E[(g^{-1})'(eta)] (with (g^{-1})'(x) = exp(x - exp(x))) are smooth, bounded
//! integrals. At sigma = 4 the scalar dispatcher (which falls back to a
//! Gauss-Hermite quadrature net here, mode `QuadratureFallback`) is off by
//! ~1e-3 in the mean and ~3.5e-3 in the derivative: the GHQ net is undersampled
//! for the sharply-varying cloglog integrand and its internal drift-check
//! cannot detect the error because the reference it compares against is just as
//! coarse.
//!
//! Reference: dense composite-Simpson quadrature over +/-18 sigma, independently
//! validated by finite-differencing the mean (matches to < 1e-6).

use gam::inference::quadrature::{QuadratureContext, integrated_inverse_link_mean_and_derivative};
use gam::types::LinkFunction;

fn cloglog(x: f64) -> f64 {
    1.0 - (-(x.exp())).exp()
}
fn cloglog_d1(x: f64) -> f64 {
    (x - x.exp()).exp()
}

/// E[f(eta)] for eta ~ N(mu, sigma^2) via composite Simpson over +/-18 sigma.
fn gaussian_expectation(f: impl Fn(f64) -> f64, mu: f64, sigma: f64) -> f64 {
    let lo = mu - 18.0 * sigma;
    let hi = mu + 18.0 * sigma;
    let n = 1_000_000usize;
    let h = (hi - lo) / n as f64;
    let pdf = |x: f64| {
        let z = (x - mu) / sigma;
        (-0.5 * z * z).exp() / (sigma * (2.0 * std::f64::consts::PI).sqrt())
    };
    let mut acc = f(lo) * pdf(lo) + f(hi) * pdf(hi);
    for i in 1..n {
        let x = lo + i as f64 * h;
        let w = if i % 2 == 1 { 4.0 } else { 2.0 };
        acc += w * f(x) * pdf(x);
    }
    acc * h / 3.0
}

#[test]
fn cloglog_integrated_mean_and_derivative_are_accurate_at_large_sigma() {
    let ctx = QuadratureContext::new();
    let cases = [(1.0, 4.0), (0.5, 4.0), (-1.0, 4.0), (0.0, 4.0)];
    let tol = 1e-4;
    let mut worst_m = 0.0f64;
    let mut worst_m_at = (0.0, 0.0, 0.0, 0.0);
    let mut worst_d = 0.0f64;
    let mut worst_d_at = (0.0, 0.0, 0.0, 0.0);
    for (mu, sigma) in cases {
        let got =
            integrated_inverse_link_mean_and_derivative(&ctx, LinkFunction::CLogLog, mu, sigma)
                .unwrap();
        let want_m = gaussian_expectation(cloglog, mu, sigma);
        let want_d = gaussian_expectation(cloglog_d1, mu, sigma);
        let em = (got.mean - want_m).abs();
        let ed = (got.dmean_dmu - want_d).abs();
        if em > worst_m {
            worst_m = em;
            worst_m_at = (mu, sigma, got.mean, want_m);
        }
        if ed > worst_d {
            worst_d = ed;
            worst_d_at = (mu, sigma, got.dmean_dmu, want_d);
        }
    }
    assert!(
        worst_m < tol,
        "CLogLog integrated mean off by {worst_m:.3e} (tol {tol:.0e}) at \
         (mu={}, sigma={}): got {}, want {}",
        worst_m_at.0,
        worst_m_at.1,
        worst_m_at.2,
        worst_m_at.3
    );
    assert!(
        worst_d < tol,
        "CLogLog integrated d/dmu off by {worst_d:.3e} (tol {tol:.0e}) at \
         (mu={}, sigma={}): got {}, want {}",
        worst_d_at.0,
        worst_d_at.1,
        worst_d_at.2,
        worst_d_at.3
    );
}
