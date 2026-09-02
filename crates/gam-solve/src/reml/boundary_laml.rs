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

#[cfg(test)]
mod tests {

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

    /// log of the exact 2D Gaussian orthant integral
    /// `∫₀^∞∫₀^∞ exp(−μ·u − ½ uᵀ Λ u) du` by composite-Simpson quadrature — the
    /// ground truth for the Proposition 4.3 / Theorem 4.4 checks (mirrors
    /// exp2_orthant.py::exact_2d).
    fn exact_2d_log_orthant(mu: [f64; 2], lam: [[f64; 2]; 2]) -> f64 {
        let (h1, h2, c) = (lam[0][0], lam[1][1], lam[0][1]);
        let u1_max = 12.0 / h1.sqrt() + (mu[0] / h1).abs() * 3.0 + 3.0;
        let u2_max = 12.0 / h2.sqrt() + (mu[1] / h2).abs() * 3.0 + 3.0;
        let n = 1200usize; // even
        let s1 = u1_max / n as f64;
        let s2 = u2_max / n as f64;
        let f = |u1: f64, u2: f64| {
            (-mu[0] * u1 - mu[1] * u2 - 0.5 * (h1 * u1 * u1 + h2 * u2 * u2 + 2.0 * c * u1 * u2))
                .exp()
        };
        let w = |i: usize| {
            if i == 0 || i == n {
                1.0
            } else if i % 2 == 1 {
                4.0
            } else {
                2.0
            }
        };
        let mut acc = 0.0;
        for i in 0..=n {
            let wi = w(i);
            let u1 = s1 * i as f64;
            for j in 0..=n {
                acc += wi * w(j) * f(u1, s2 * j as f64);
            }
        }
        (acc * s1 * s2 / 9.0).ln()
    }

    fn precision_2x2(h: [f64; 2], c: f64) -> ndarray::Array2<f64> {
        ndarray::array![[h[0], c], [c, h[1]]]
    }

}
