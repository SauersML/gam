//! Bug hunt: `integrated_inverse_link_mean_and_derivative` returns a wrong
//! d/dmu E[sigmoid(eta)] for the Logit link on the erfcx "exact" branch.
//!
//! By the location identity, d/dmu E[f(mu + sigma Z)] = E[f'(mu + sigma Z)],
//! so for the logit link d/dmu E[sigmoid(eta)] = E[sigmoid'(eta)] with
//! sigmoid'(x) = sigmoid(x)(1 - sigmoid(x)). At mu = 0 this expectation is a
//! modest positive number (< 0.25, shrinking as sigma grows), yet the scalar
//! dispatcher's `ExactSpecialFunction` (erfcx-series) branch returns ~0.58 at
//! sigma = 0.3 — more than 2x the true value. The mean on this branch is fine;
//! only the derivative formula is wrong.
//!
//! Reference: dense composite-Simpson quadrature over +/-18 sigma, independently
//! validated by finite-differencing the mean (matches to < 1e-6).

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
fn logit_integrated_derivative_matches_expected_integrand() {
    let ctx = QuadratureContext::new();
    // Points on the erfcx exact branch where the derivative is mishandled.
    let cases = [(0.0, 0.3), (0.0, 0.5), (0.0, 0.4)];
    let tol = 1e-4;
    let mut worst = 0.0f64;
    let mut worst_at = (0.0, 0.0, 0.0, 0.0);
    for (mu, sigma) in cases {
        let got = integrated_inverse_link_mean_and_derivative(&ctx, LinkFunction::Logit, mu, sigma)
            .unwrap();
        // E[sigmoid'(eta)] = d/dmu E[sigmoid(eta)].
        let want = gaussian_expectation(|x| sigmoid(x) * (1.0 - sigmoid(x)), mu, sigma);
        let err = (got.dmean_dmu - want).abs();
        if err > worst {
            worst = err;
            worst_at = (mu, sigma, got.dmean_dmu, want);
        }
    }
    assert!(
        worst < tol,
        "Logit integrated d/dmu off by {worst:.3e} (tol {tol:.0e}) at \
         (mu={}, sigma={}): got {}, want {}",
        worst_at.0,
        worst_at.1,
        worst_at.2,
        worst_at.3
    );
}
