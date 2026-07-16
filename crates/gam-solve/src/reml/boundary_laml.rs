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
use ndarray::{ArrayView1, ArrayView2};
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

/// Log of the interior (inactive-face) boundary factor
/// `g_int(s, h) = √(2π/h) · Φ(s√h)`, the normal-direction contribution of one
/// near-boundary INACTIVE constraint whose constrained mode sits at signed slack
/// `s > 0` inside the feasible half-space (gam#2306 §4). Unlike the active factor
/// [`log_boundary_g`], the mode is the unconstrained peak so there is no linear
/// (multiplier) term — the integral is just the Gaussian tail past the boundary.
///
/// This is what makes the criterion smooth across an activation event: at `s = 0`
/// it equals `½√(2π/h)`, exactly the active factor `g(μ=0, h)`, and as
/// `s√h → +∞` it recovers `√(2π/h)` — the unrestricted Gaussian normalizer, i.e.
/// a far-interior row reduces byte-identically to today's LAML.
///
/// `log g_int = ½ln(2π/h) + logΦ(s√h)`, stable for all `s` via `normal_logcdf`.
pub fn log_interior_boundary_g(s: f64, h: f64) -> f64 {
    assert!(
        h.is_finite() && h > 0.0,
        "interior boundary-g curvature h must be finite and positive, got {h}"
    );
    assert!(s.is_finite(), "interior boundary-g slack s must be finite, got {s}");
    0.5 * ((2.0 * PI) / h).ln() + normal_logcdf(s * h.sqrt())
}

/// `(∂ log g_int/∂s, ∂ log g_int/∂h)`.
///
/// `∂ log g_int/∂s = √h · r`, `∂ log g_int/∂h = −1/(2h) + s·r/(2√h)`, with
/// `r = φ(s√h)/Φ(s√h)` the (forward) Mills ratio — the same log-domain
/// `exp(lnφ − logΦ)` form used by the active factor.
pub fn log_interior_boundary_g_derivatives(s: f64, h: f64) -> (f64, f64) {
    assert!(
        h.is_finite() && h > 0.0,
        "interior boundary-g curvature h must be finite and positive, got {h}"
    );
    assert!(s.is_finite(), "interior boundary-g slack s must be finite, got {s}");
    let sqrt_h = h.sqrt();
    let r = reverse_mills(s * sqrt_h);
    let d_s = sqrt_h * r;
    let d_h = -0.5 / h + s * r / (2.0 * sqrt_h);
    (d_s, d_h)
}

/// `−∂ log g/∂μ = (−μ + √h·r)/h`, the truncated-normal mean `m_a` of the
/// half-line boundary factor (`r = φ(z)/Φ(−z)`, `z = μ/√h`). This is the
/// coefficient the Proposition 4.3 (#2337 §4) first-order correction contracts
/// with the precision off-diagonals.
fn boundary_factor_trunc_mean(mu: f64, h: f64) -> f64 {
    -log_boundary_g_derivatives(mu, h).0
}

/// `log I(μ, Λ)` — the log of the Gaussian orthant integral
/// `∫_{ℝ₊^m} exp(−μᵀu − ½ uᵀ Λ u) du` — the exact boundary factor of the
/// constrained Laplace normalizer over `m` active reduced-representative faces,
/// with `Λ = Σ⁻¹` the PRECISION of the slack coordinates
/// (`Σ = C_A H̄⁻¹ C_Aᵀ`), evaluated by the Proposition 4.3 (#2337 §4)
/// diagonal-anchored expansion:
///
/// ```text
///   log I = Σ_a log g(μ_a, Λ_aa) − Σ_{a<b} Λ_ab · m_a · m_b + O(‖Λ_off‖²),
/// ```
///
/// `m_a = −∂_μ log g(μ_a, Λ_aa)` the truncated-normal mean. Exact through first
/// order in the off-diagonal precision: at diagonal `Λ` the orthant factorizes
/// into the per-face `g`, and `∂_{Λ_ab}` of the integrand is `−u_a u_b`, whose
/// expectation factorizes to `−m_a m_b` at the diagonal reference. The wiring
/// supplies `Λ = Σ⁻¹` (m×m, small — the reduced face size) and, separately, the
/// `det Σ` bookkeeping the criterion also needs (the exact boundary correction
/// to the log-normalizer is `−(m/2)·ln(2π) − ½·ln det Σ + log I`).
pub fn log_gaussian_orthant(mu: ArrayView1<'_, f64>, precision: ArrayView2<'_, f64>) -> f64 {
    let m = mu.len();
    assert_boundary_precision_shape(m, precision);
    let mut trunc = Vec::with_capacity(m);
    let mut acc = 0.0;
    for a in 0..m {
        let laa = precision[[a, a]];
        acc += log_boundary_g(mu[a], laa);
        trunc.push(boundary_factor_trunc_mean(mu[a], laa));
    }
    for a in 0..m {
        for b in (a + 1)..m {
            acc -= precision[[a, b]] * trunc[a] * trunc[b];
        }
    }
    acc
}

/// Deterministic two-sided bracket on `log I(μ, Λ)` (Theorem 4.4, #2337 §4).
///
/// Loewner monotonicity: `diag(Λ) − s·I ⪯ Λ ⪯ diag(Λ) + s·I` for any
/// `s ≥ ‖Λ_off‖₂`, and each per-face `g(μ_a, ·)` is decreasing in its curvature,
/// so
///
/// ```text
///   Σ_a log g(μ_a, Λ_aa + s)  ≤  log I  ≤  Σ_a log g(μ_a, Λ_aa − s).
/// ```
///
/// `s` is taken as the Gershgorin off-diagonal row-sum bound
/// `max_a Σ_{b≠a} |Λ_ab| ≥ ‖Λ_off‖₂` (exact for `m ≤ 2`, conservative and
/// eigensolve-free above), so the bracket is always valid. When any
/// `Λ_aa − s ≤ 0` the upper factor's curvature is non-positive and the bound
/// diverges — returned as `+∞`, the escalation signal (block-partition /
/// tanh-sinh) the criterion uses when the bracket exceeds tolerance.
pub fn log_gaussian_orthant_bracket(
    mu: ArrayView1<'_, f64>,
    precision: ArrayView2<'_, f64>,
) -> (f64, f64) {
    let m = mu.len();
    assert_boundary_precision_shape(m, precision);
    // Gershgorin bound on ‖Λ_off‖₂: the max absolute off-diagonal row sum.
    let mut s = 0.0_f64;
    for a in 0..m {
        let mut row = 0.0_f64;
        for b in 0..m {
            if a != b {
                row += precision[[a, b]].abs();
            }
        }
        s = s.max(row);
    }
    let mut lower = 0.0;
    let mut upper = 0.0;
    for a in 0..m {
        let laa = precision[[a, a]];
        lower += log_boundary_g(mu[a], laa + s);
        if laa - s > 0.0 {
            upper += log_boundary_g(mu[a], laa - s);
        } else {
            upper = f64::INFINITY;
        }
    }
    (lower, upper)
}

#[inline]
fn assert_boundary_precision_shape(m: usize, precision: ArrayView2<'_, f64>) {
    assert!(
        precision.nrows() == m && precision.ncols() == m,
        "boundary orthant precision must be {m}x{m}, got {}x{}",
        precision.nrows(),
        precision.ncols()
    );
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

    #[test]
    fn interior_factor_joins_active_factor_at_the_boundary() {
        // s = 0 (inactive mode exactly on the boundary) must equal μ = 0 (active
        // face with zero multiplier) — the continuity that removes the log2 jump.
        for &h in &[0.3_f64, 1.0, 7.0, 25.0] {
            let interior = log_interior_boundary_g(0.0, h);
            let active = log_boundary_g(0.0, h);
            assert!(
                (interior - active).abs() <= 1e-13 * active.abs().max(1.0),
                "boundary join mismatch at h={h}: interior {interior} vs active {active}"
            );
        }
    }

    #[test]
    fn interior_factor_far_from_boundary_recovers_full_gaussian() {
        // s√h → ∞: g_int → √(2π/h) ⇒ log g_int → ½ln(2π/h). This is the
        // unrestricted normalizer — a far-interior row costs nothing extra, so
        // the criterion reduces byte-identically to today's LAML there. Every
        // case keeps s√h ≥ 9, well past where |logΦ(s√h)| drops below the 1e-6
        // limit tolerance (|logΦ(z)| ≈ φ(z)/z, ~1e-5 at z≈4.2 but ~1e-19 at z≈9).
        for &(s, h) in &[(10.0_f64, 1.0_f64), (30.0, 4.0), (13.0, 0.5)] {
            let z = s * h.sqrt();
            assert!(z >= 9.0, "far-interior test case must keep s√h ≥ 9, got {z}");
            let got = log_interior_boundary_g(s, h);
            let want = 0.5 * ((2.0 * PI) / h).ln();
            assert!(
                (got - want).abs() < 1e-6,
                "far-interior mismatch at s={s}, h={h}: {got} vs {want}"
            );
        }
    }

    #[test]
    fn interior_factor_matches_gaussian_tail_integral() {
        // g_int(s,h) = √(2π/h)·Φ(s√h) = ∫_{−s}^∞ e^{−½ h u²} du.
        for &(s, h) in &[(0.0_f64, 1.0), (0.4, 2.0), (1.0, 0.7), (2.0, 3.0)] {
            let got = log_interior_boundary_g(s, h).exp();
            // Composite-Simpson of the Gaussian tail from the boundary (−s) out.
            let u_lo = -s;
            let u_hi = (80.0 / h).sqrt();
            let panels = 2_000_000usize;
            let step = (u_hi - u_lo) / panels as f64;
            let f = |u: f64| (-0.5 * h * u * u).exp();
            let mut acc = f(u_lo) + f(u_hi);
            for i in 1..panels {
                let u = u_lo + step * i as f64;
                acc += if i % 2 == 1 { 4.0 } else { 2.0 } * f(u);
            }
            let want = acc * step / 3.0;
            let rel = (got - want).abs() / want.abs();
            assert!(
                rel < 5e-6,
                "interior g({s},{h}) = {got:.10e} but the tail integral is {want:.10e} (rel {rel:.2e})"
            );
        }
    }

    #[test]
    fn interior_factor_derivatives_match_central_difference() {
        for &(s, h) in &[(0.0, 1.0), (0.6, 2.0), (1.5, 0.6), (-0.3, 3.0), (2.0, 0.9)] {
            let (d_s, d_h) = log_interior_boundary_g_derivatives(s, h);
            let eps = 1e-6;
            let fd_s = (log_interior_boundary_g(s + eps, h)
                - log_interior_boundary_g(s - eps, h))
                / (2.0 * eps);
            let fd_h = (log_interior_boundary_g(s, h + eps)
                - log_interior_boundary_g(s, h - eps))
                / (2.0 * eps);
            assert!(
                (d_s - fd_s).abs() <= 1e-5 * fd_s.abs().max(1.0),
                "∂logg_int/∂s mismatch at ({s},{h}): analytic {d_s} vs fd {fd_s}"
            );
            assert!(
                (d_h - fd_h).abs() <= 1e-5 * fd_h.abs().max(1.0),
                "∂logg_int/∂h mismatch at ({s},{h}): analytic {d_h} vs fd {fd_h}"
            );
        }
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
            (-mu[0] * u1 - mu[1] * u2 - 0.5 * (h1 * u1 * u1 + h2 * u2 * u2 + 2.0 * c * u1 * u2)).exp()
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

    #[test]
    fn log_gaussian_orthant_diagonal_is_exact_product() {
        // Zero off-diagonal: the orthant factorizes, so log I = Σ log g exactly
        // and matches the 2D quadrature.
        let mu = ndarray::array![0.7_f64, -0.4];
        let h = [2.0_f64, 0.9];
        let lam = precision_2x2(h, 0.0);
        let got = log_gaussian_orthant(mu.view(), lam.view());
        let product = log_boundary_g(mu[0], h[0]) + log_boundary_g(mu[1], h[1]);
        assert!((got - product).abs() <= 1e-14 * product.abs().max(1.0));
        let quad = exact_2d_log_orthant([mu[0], mu[1]], [[h[0], 0.0], [0.0, h[1]]]);
        assert!(
            (got - quad).abs() < 5e-6,
            "diagonal orthant {got} vs quadrature {quad}"
        );
    }

    #[test]
    fn log_gaussian_orthant_first_order_correction_beats_product() {
        // The Prop 4.3 −Λ_ab m_a m_b correction is first-order exact: its error
        // is O(c²), strictly smaller than the product form's O(c) error, and
        // shrinks quadratically as c → 0 (mirrors exp2_orthant.py).
        let mu = ndarray::array![0.7_f64, -0.4];
        let h = [2.0_f64, 0.9];
        let product = log_boundary_g(mu[0], h[0]) + log_boundary_g(mu[1], h[1]);
        // Ratio corrected_err/product_err ~ O(c²)/O(c) = O(c): the correction is
        // strictly better AND its relative advantage grows as c → 0. Both are
        // constant-free (robust) signatures of first-order exactness — no
        // magic-number tolerance on the residual magnitude.
        let mut prev_relative: Option<f64> = None;
        for &c in &[0.4_f64, 0.2, 0.1, 0.05] {
            let lam = precision_2x2(h, c);
            let corrected = log_gaussian_orthant(mu.view(), lam.view());
            let exact = exact_2d_log_orthant([mu[0], mu[1]], [[h[0], c], [c, h[1]]]);
            let product_err = (product - exact).abs();
            let corrected_err = (corrected - exact).abs();
            assert!(
                corrected_err < product_err,
                "c={c}: corrected err {corrected_err:.3e} not < product err {product_err:.3e}"
            );
            let relative = corrected_err / product_err;
            if let Some(prev) = prev_relative {
                assert!(
                    relative < prev,
                    "c={c}: corrected/product err ratio {relative:.3e} did not shrink below {prev:.3e} as c decreased"
                );
            }
            prev_relative = Some(relative);
        }
    }

    #[test]
    fn log_gaussian_orthant_bracket_contains_exact() {
        // Theorem 4.4 Loewner bracket must two-side the exact log I.
        let mu = ndarray::array![0.7_f64, -0.4];
        let h = [2.0_f64, 0.9];
        for &c in &[0.4_f64, 0.2, 0.1] {
            let lam = precision_2x2(h, c);
            let (lo, hi) = log_gaussian_orthant_bracket(mu.view(), lam.view());
            let exact = exact_2d_log_orthant([mu[0], mu[1]], [[h[0], c], [c, h[1]]]);
            assert!(
                lo <= exact && exact <= hi,
                "c={c}: bracket [{lo:.6}, {hi:.6}] excludes exact {exact:.6}"
            );
            // The Prop 4.3 estimate also sits inside its own rigor bracket.
            let est = log_gaussian_orthant(mu.view(), lam.view());
            assert!(lo <= est && est <= hi, "c={c}: estimate {est} outside bracket");
        }
    }

    #[test]
    fn log_gaussian_orthant_bracket_escalates_on_nonpositive_curvature() {
        // A large off-diagonal precision drives Λ_aa − s ≤ 0; the upper bound
        // must report +∞ (escalate), never panic on a non-positive curvature.
        let mu = ndarray::array![0.3_f64, 0.1];
        let lam = precision_2x2([1.0, 1.0], 1.5); // s = 1.5 > diag 1.0
        let (lo, hi) = log_gaussian_orthant_bracket(mu.view(), lam.view());
        assert!(lo.is_finite());
        assert!(hi.is_infinite(), "non-positive upper curvature must escalate to +∞");
    }
}
