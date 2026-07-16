//! Boundary factors for the constrained Laplace normalizer (gam#2306 §4).
//!
//! When the inner mode sits on active inequality faces (the CTN monotonicity
//! cone), the Laplace approximation of `∫_K exp(−J)` factors into the
//! tangent-space determinant `det(ZᵀH̄Z)` (already assembled by the active-face
//! logdet path) times a product of per-face half-line factors
//!
//! ```text
//!   g(μ, h) = ∫₀^∞ exp(−μ u − ½ h u²) du
//!           = √(2π/h) · e^{μ²/2h} · Φ(−μ/√h),
//! ```
//!
//! one for each active normal direction with H̄-Schur curvature `h > 0` and KKT
//! multiplier `μ`. This module provides `g` in the log domain and its analytic
//! `(μ, h)` gradient; the outer criterion adds `−2·Σ_a log g(μ̃_a, h̃_a)` in place
//! of the proportional-ridge placeholder that currently stands in for the
//! normal-direction logdet. Limits (all exact):
//!
//! - `μ = 0`: `g = ½√(2π/h)` — the half-Gaussian, so an activation event with a
//!   zero multiplier is continuous (no `log μ` blow-up).
//! - `μ/√h → +∞`: `g → 1/μ` — the linear-decay tail.
//! - `μ → −∞` (far interior): `g → √(2π/h)·e^{μ²/2h}`, i.e. `log g` recovers the
//!   unrestricted Gaussian normalizer — the criterion reduces to today's LAML.

use gam_math::probability::{erfcx_nonnegative, normal_logcdf};
use std::f64::consts::{LN_2, PI};

/// `ln(2π)`.
const LN_2PI: f64 = 1.837_877_066_409_345_3;

/// `log g(μ, h)` for the half-line boundary factor, evaluated in the log domain
/// so it stays finite across the whole range.
///
/// For `μ ≥ 0` the naive `e^{μ²/2h}·Φ(−μ/√h)` both overflows and underflows, so
/// we use the analytically-equivalent scaled-complementary-error form
/// `log g = ½ln(2π/h) − ln2 + ln erfcx(μ/√(2h))` (`erfcx` stays bounded). For
/// `μ < 0` the direct form `log g = ½ln(2π/h) + μ²/(2h) + logΦ(−μ/√h)` has no
/// cancellation (`log g` grows smoothly toward the unrestricted normalizer).
pub fn log_boundary_g(mu: f64, h: f64) -> f64 {
    assert!(
        h.is_finite() && h > 0.0,
        "boundary-g curvature h must be finite and positive, got {h}"
    );
    assert!(mu.is_finite(), "boundary-g multiplier μ must be finite, got {mu}");
    let half_log = 0.5 * ((2.0 * PI) / h).ln();
    if mu >= 0.0 {
        let u = mu / (2.0 * h).sqrt();
        half_log - LN_2 + erfcx_nonnegative(u).ln()
    } else {
        half_log + mu * mu / (2.0 * h) + normal_logcdf(-mu / h.sqrt())
    }
}

/// Reverse Mills ratio `m(z) = φ(z)/Φ(z)`, computed as `exp(ln φ(z) − ln Φ(z))`
/// so the `z → −∞` tail (where both `φ` and `Φ` underflow) stays finite and
/// approaches `|z|` — the log-domain form the review requires instead of a clamp
/// (`Φ(−x)` underflows near `x ≈ 38`).
fn reverse_mills(z: f64) -> f64 {
    let ln_phi = -0.5 * z * z - 0.5 * LN_2PI;
    (ln_phi - normal_logcdf(z)).exp()
}

/// `(∂ log g/∂μ, ∂ log g/∂h)`, the analytic gradient the outer ρ-derivative path
/// contracts with `(μ̇, ḣ)` from the joint-derivative IFT solves.
///
/// `∂ log g/∂μ = μ/h − m/√h`, `∂ log g/∂h = −1/(2h) − μ²/(2h²) + m·μ/(2h^{3/2})`,
/// with `m = φ(μ/√h)/Φ(−μ/√h)` the reverse Mills ratio at `z = −μ/√h`.
pub fn log_boundary_g_derivatives(mu: f64, h: f64) -> (f64, f64) {
    assert!(
        h.is_finite() && h > 0.0,
        "boundary-g curvature h must be finite and positive, got {h}"
    );
    assert!(mu.is_finite(), "boundary-g multiplier μ must be finite, got {mu}");
    let sqrt_h = h.sqrt();
    let m = reverse_mills(-mu / sqrt_h);
    let d_mu = mu / h - m / sqrt_h;
    let d_h = -0.5 / h - mu * mu / (2.0 * h * h) + m * mu / (2.0 * h * sqrt_h);
    (d_mu, d_h)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Composite-Simpson quadrature of `∫₀^∞ e^{−μu−½hu²} du` on a support wide
    /// enough that the Gaussian tail is negligible — the ground-truth `g`.
    fn integral_g(mu: f64, h: f64) -> f64 {
        // The integrand `e^{−μu−½hu²}` is below e^{-40} of its peak past
        // u* = (−μ + sqrt(μ² + 80 h)) / h; integrate a bit beyond that.
        let u_max = ((-mu + (mu * mu + 80.0 * h).sqrt()) / h).max(1.0) * 1.5;
        let panels = 2_000_000usize;
        let step = u_max / panels as f64;
        let f = |u: f64| (-mu * u - 0.5 * h * u * u).exp();
        let mut acc = f(0.0) + f(u_max);
        for i in 1..panels {
            let u = step * i as f64;
            acc += if i % 2 == 1 { 4.0 } else { 2.0 } * f(u);
        }
        acc * step / 3.0
    }

    #[test]
    fn log_boundary_g_matches_direct_integral() {
        // Moderate (μ, h) where g neither overflows nor underflows.
        for &(mu, h) in &[
            (0.0, 1.0),
            (0.5, 2.0),
            (2.0, 0.7),
            (-1.0, 3.0),
            (-0.3, 0.5),
            (1.5, 4.0),
        ] {
            let got = log_boundary_g(mu, h).exp();
            let want = integral_g(mu, h);
            let rel = (got - want).abs() / want.abs();
            assert!(
                rel < 5e-6,
                "g({mu},{h}) = {got:.10e} but the direct integral is {want:.10e} (rel {rel:.2e})"
            );
        }
    }

    #[test]
    fn log_boundary_g_zero_multiplier_is_half_gaussian() {
        // μ = 0: g = ½√(2π/h) ⇒ log g = ½ln(2π/h) − ln 2.
        for &h in &[0.25_f64, 1.0, 5.0, 40.0] {
            let want = 0.5 * ((2.0 * PI) / h).ln() - LN_2;
            let got = log_boundary_g(0.0, h);
            assert!(
                (got - want).abs() <= 1e-13 * want.abs().max(1.0),
                "half-Gaussian mismatch at h={h}: {got} vs {want}"
            );
        }
    }

    #[test]
    fn log_boundary_g_large_positive_ratio_is_reciprocal() {
        // μ/√h → ∞: g → 1/μ ⇒ log g → −ln μ. Use a ratio past the Φ-underflow
        // point (≈38) to exercise the erfcx path.
        for &(mu, h) in &[(50.0_f64, 1.0), (100.0, 4.0), (400.0, 25.0)] {
            let got = log_boundary_g(mu, h);
            let want = -mu.ln();
            // The next-order term is O(h/μ²); allow a small relative slack.
            assert!(
                (got - want).abs() < 1e-3,
                "reciprocal tail mismatch at μ={mu}, h={h}: log g={got} vs −ln μ={want}"
            );
            assert!(got.is_finite(), "log g must stay finite deep in the tail");
        }
    }

    #[test]
    fn log_boundary_g_far_interior_recovers_gaussian() {
        // μ → −∞: g → √(2π/h)·e^{μ²/2h} ⇒ log g → ½ln(2π/h) + μ²/(2h). This is
        // the unrestricted normalizer — the interior reduction.
        for &(mu, h) in &[(-30.0_f64, 1.0), (-60.0, 4.0), (-10.0, 0.5)] {
            let got = log_boundary_g(mu, h);
            let want = 0.5 * ((2.0 * PI) / h).ln() + mu * mu / (2.0 * h);
            let rel = (got - want).abs() / want.abs();
            assert!(
                rel < 1e-3,
                "interior Gaussian mismatch at μ={mu}, h={h}: {got} vs {want} (rel {rel:.2e})"
            );
        }
    }

    #[test]
    fn log_boundary_g_derivatives_match_central_difference() {
        for &(mu, h) in &[
            (0.0, 1.0),
            (0.7, 2.0),
            (-1.2, 0.6),
            (3.0, 0.9),
            (-0.4, 4.0),
        ] {
            let (d_mu, d_h) = log_boundary_g_derivatives(mu, h);
            let eps = 1e-6;
            let fd_mu =
                (log_boundary_g(mu + eps, h) - log_boundary_g(mu - eps, h)) / (2.0 * eps);
            let fd_h = (log_boundary_g(mu, h + eps) - log_boundary_g(mu, h - eps)) / (2.0 * eps);
            assert!(
                (d_mu - fd_mu).abs() <= 1e-5 * fd_mu.abs().max(1.0),
                "∂logg/∂μ mismatch at ({mu},{h}): analytic {d_mu} vs fd {fd_mu}"
            );
            assert!(
                (d_h - fd_h).abs() <= 1e-5 * fd_h.abs().max(1.0),
                "∂logg/∂h mismatch at ({mu},{h}): analytic {d_h} vs fd {fd_h}"
            );
        }
    }
}
