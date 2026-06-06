//! Bug hunt: the large-σ closed-form survival approximation is biased low by
//! several percent in the σ ∈ [8, 15] band, where it is engaged.
//!
//! The survival transform / cloglog inverse-link complement
//!
//!     S(mu, sigma) = E[ exp(-exp(eta)) ],   eta ~ N(mu, sigma^2)
//!
//! is reachable through the public lognormal kernel: `K_{0,1}(mu, sigma) = S`,
//! so `log_kernel_term(ctx, 0, 1.0, mu, sigma).0.exp() == S(mu, sigma)`.
//!
//! For `sigma >= CLOGLOG_LARGE_SIGMA_ASYMPTOTIC_MIN` (= 8) the controlled
//! evaluator routes to `cloglog_large_sigma_transition_approx` /
//! `cloglog_survival_extreme_asymptotic`
//! (`src/inference/quadrature.rs`, ~lines 1281–1373), which uses the two-term
//! "sharp transition" split
//!
//!     S ≈ Phi(-mu/sigma) - exp(mu + sigma^2/2) * Phi(-(mu + sigma^2)/sigma).
//!
//! That split replaces the genuine O(1)-wide transition region of
//! `exp(-exp(eta))` around `eta = 0` with a step function, dropping a positive
//! correction of order `phi(mu/sigma)/sigma`. The result is systematically too
//! LOW by ~2–7% across the σ band where the branch is active — well outside any
//! "exact / controlled" claim (the module comment names exactly these points,
//! e.g. (10, 10), as 256-point-GHQ validation targets).
//!
//! Observed (library vs converged reference quadrature):
//!     (0,  8): 0.45088 vs 0.47189  (-4.5%)
//!     (10,10): 0.13684 vs 0.14704  (-6.9%)
//!     (-1, 9): 0.50018 vs 0.51876  (-3.6%)
//!
//! This test compares S against a converged Simpson reference (the reference
//! matches the library to ~1e-15 at small σ, and converges to <1e-9 between
//! step sizes here). When the asymptotic is corrected (transition term added,
//! or the threshold raised so a higher-accuracy branch handles this band), the
//! library value matches the reference and the test passes unchanged.

use gam::families::lognormal_kernel::log_kernel_term;
use gam::inference::probability::normal_cdf;
use gam::quadrature::QuadratureContext;

/// Converged reference for S(mu, sigma) = E[exp(-exp(eta))], eta ~ N(mu,sigma^2).
///
/// Composite Simpson over [-60, 50] (the integrand is 1 to machine precision
/// below -60 and 0 above 50), plus the Gaussian tail mass below -60 where the
/// integrand is exactly 1.
fn survival_reference(mu: f64, sigma: f64) -> f64 {
    let lo = -60.0_f64;
    let hi = 50.0_f64;
    let dh = 5.0e-4_f64;
    let n = (((hi - lo) / dh).ceil() as usize) | 1; // ensure even #intervals
    let h = (hi - lo) / n as f64;
    let two_pi = 2.0 * std::f64::consts::PI;
    let f = |eta: f64| -> f64 {
        let integrand = (-eta.exp()).exp();
        let z = (eta - mu) / sigma;
        let dens = (-0.5 * z * z).exp() / (sigma * two_pi.sqrt());
        integrand * dens
    };
    let mut acc = f(lo) + f(hi);
    for i in 1..n {
        let eta = lo + i as f64 * h;
        acc += if i % 2 == 1 { 4.0 } else { 2.0 } * f(eta);
    }
    acc * h / 3.0 + normal_cdf((lo - mu) / sigma)
}

#[test]
fn cloglog_survival_value_matches_reference_in_large_sigma_band() {
    let ctx = QuadratureContext::new();

    // Points inside the large-σ asymptotic band (sigma >= 8) where the library
    // is currently several percent low.
    let cases = [(0.0_f64, 8.0_f64), (10.0, 10.0), (-1.0, 9.0), (0.0, 12.0)];

    for (mu, sigma) in cases {
        let lib = log_kernel_term(&ctx, 0, 1.0, mu, sigma)
            .expect("kernel term")
            .0
            .exp();
        let reference = survival_reference(mu, sigma);
        let rel = (lib - reference).abs() / reference.abs();
        assert!(
            rel < 0.01,
            "S(mu={mu}, sigma={sigma}): library survival {lib} disagrees with the \
             converged reference {reference} by rel err {rel} (> 1%)",
        );
    }
}
