//! Bug hunt: the integrated cloglog inverse-link slope collapses to exactly
//! zero at large latent σ, contradicting the function's own mean.
//!
//! `latent_cloglog_jet5(ctx, eta, sigma)` returns the integrated cloglog inverse
//! link and its η-derivatives:
//!
//!     mean(eta) = E[ 1 - exp(-exp(eta + sigma Z)) ],   Z ~ N(0, 1)
//!     d1        = d mean / d eta
//!
//! `mean(eta)` is strictly increasing in `eta` for every finite `(eta, sigma)`
//! (the integrand `1 - exp(-exp(·))` is strictly increasing and the expectation
//! preserves monotonicity), so the analytic slope `d1` is strictly positive and
//! must agree with a central finite difference of the *same function's* mean.
//!
//! It does not. For σ ≳ 8 the slope is severely under-estimated, and for σ ≳ 12
//! it is returned as exactly `0.0`, even though `mean(eta + h) != mean(eta - h)`.
//!
//! Root cause (read, not patched):
//!   * `src/inference/quadrature.rs::latent_cloglog_kernel_terms` builds the
//!     kernels `K_{k,1} = exp(k·μ + ½k²σ²) · S(μ + kσ², σ)` where
//!     `S = E[exp(-exp(η))]` is the survival. It computes `S` via
//!     `lognormal_laplace_term_shared` (the controlled survival evaluator) and
//!     then does `if survival <= 0.0 { *out = 0.0; continue; }`.
//!   * `cloglog_survival_term_controlled` returns a *hard* `0.0` in its
//!     saturation branch (`mu - 8σ >= 5`, quadrature.rs ~line 1366). At large σ
//!     the k=1 shifted location `μ + σ²` lands in that branch, so the survival
//!     underflows to 0 in value space — its large-but-finite log-magnitude is
//!     discarded — and `K_{1,1}` is zeroed.
//!   * `cloglog_inverse_link_controlled_values` (quadrature.rs ~line 3408) then
//!     sets `values[1] = k[1].max(0.0)` with NO asymptotic fallback (d2..d5 do
//!     fall back to `integrate_normal_adaptive`, but d1 does not), so the slope
//!     is reported as 0.
//!
//! The true `K_{1,1}` is finite: the value-space underflow of `S` is exactly
//! compensated by the `exp(½σ²)` prefix. Keeping `S` in log space (or routing d1
//! through the same quadrature fallback used for d2..d5) restores the slope.
//!
//! When fixed, `d1` becomes positive and matches the finite-difference slope, so
//! this test passes without edits.

use gam::families::lognormal_kernel::latent_cloglog_jet5;
use gam::quadrature::QuadratureContext;

#[test]
fn integrated_cloglog_slope_matches_its_own_mean_finite_difference() {
    let ctx = QuadratureContext::new();

    // (eta, sigma) points in the large-σ regime. All have a strictly positive,
    // O(1e-2) true slope; the library currently returns 0 (σ≥12) or ~2× too
    // small (σ=8).
    let cases = [(0.0_f64, 8.0_f64), (0.0, 12.0), (10.0, 10.0), (-2.0, 30.0)];

    for (eta, sigma) in cases {
        let jet = latent_cloglog_jet5(&ctx, eta, sigma).expect("jet");

        // Self-consistent reference: central finite difference of the SAME
        // function's mean. The mean (k=0 kernel, no shift) is unaffected by the
        // saturation bug, so its FD is the true slope.
        let h = 1e-3 * sigma;
        let mean_plus = latent_cloglog_jet5(&ctx, eta + h, sigma).unwrap().mean;
        let mean_minus = latent_cloglog_jet5(&ctx, eta - h, sigma).unwrap().mean;
        let fd = (mean_plus - mean_minus) / (2.0 * h);

        // The mean genuinely changes with eta (sanity-check the reference is
        // a real, non-trivial slope, not numerical noise).
        assert!(
            fd > 1e-3,
            "reference FD slope should be a clear positive number at \
             (eta={eta}, sigma={sigma}); got fd={fd}"
        );

        // The analytic slope must be strictly positive (mean is strictly
        // increasing) and must agree with the finite difference.
        assert!(
            jet.d1 > 0.0,
            "integrated cloglog slope d1 must be strictly positive at \
             (eta={eta}, sigma={sigma}) because the mean is strictly increasing, \
             but got d1={} (mean changes from {mean_minus} to {mean_plus})",
            jet.d1
        );
        let rel = (jet.d1 - fd).abs() / fd.abs();
        assert!(
            rel < 0.1,
            "integrated cloglog slope d1={} disagrees with the finite difference \
             fd={fd} of its own mean at (eta={eta}, sigma={sigma}); rel err={rel}",
            jet.d1
        );
    }
}
