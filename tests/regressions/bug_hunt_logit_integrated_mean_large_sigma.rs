//! Bug hunt: `integrated_inverse_link_mean_and_derivative` returns a badly
//! inaccurate posterior mean for the Logit link at large latent sigma.
//!
//! For eta ~ N(mu, sigma^2), E[sigmoid(eta)] is a well-defined logistic-normal
//! integral. The scalar dispatcher routes large-sigma cases through a
//! Monahan-Stefanski probit approximation (mode `ControlledAsymptotic`) whose
//! error is ~1e-1 in absolute terms — far outside any reasonable tolerance for
//! a function used by `predict_gam_posterior_mean` on binomial-logit models.
//!
//! Reference: dense composite-Simpson quadrature over +/-18 sigma. The
//! reference is independently validated (FD of the mean matches the analytic
//! integrand derivative to < 1e-6), so the gap below is a real method error.

use gam::inference::quadrature::{QuadratureContext, integrated_inverse_link_mean_and_derivative};
use gam::types::LinkFunction;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
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
fn logit_integrated_mean_is_accurate_at_large_sigma() {
    let ctx = QuadratureContext::new();
    // Points that route through the large-sigma asymptotic branch.
    let cases = [(3.0, 3.0), (4.0, 4.0), (2.0, 5.0), (5.0, 5.0)];
    let tol = 1e-4;
    let mut worst = 0.0f64;
    let mut worst_at = (0.0, 0.0, 0.0, 0.0);
    for (mu, sigma) in cases {
        let got = integrated_inverse_link_mean_and_derivative(&ctx, LinkFunction::Logit, mu, sigma)
            .unwrap();
        let want = gaussian_expectation(sigmoid, mu, sigma);
        let err = (got.mean - want).abs();
        if err > worst {
            worst = err;
            worst_at = (mu, sigma, got.mean, want);
        }
    }
    assert!(
        worst < tol,
        "Logit integrated mean off by {worst:.3e} (tol {tol:.0e}) at \
         (mu={}, sigma={}): got {}, want {}",
        worst_at.0,
        worst_at.1,
        worst_at.2,
        worst_at.3
    );
}
