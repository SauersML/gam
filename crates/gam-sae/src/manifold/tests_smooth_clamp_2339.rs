//! #2339 — the ARD PSD majorizer's hard clamp `max(V'', 0)` is replaced by the
//! smooth homogeneity-preserving softplus envelope `α·softplus_{τ₀}(cos κt)`, so
//! the streaming `½log|B̃|` criterion becomes a composite-analytic estimand (no
//! kink at the clamp seam). These gates pin the four load-bearing properties:
//!
//!  1. SEAM IDENTITY — `psd_majorizer_hess + negative_hessian_remainder == V''`
//!     (the majorizer + its complementary concave remainder reconstruct the exact
//!     signed prior curvature), and the Euclidean path is a bit-identical no-op.
//!  2. HOMOGENEITY — the majorizer is exactly degree-one in `α = e^{ρ_ard}`, which
//!     the `½log|B|` explicit-ρ θ-adjoint traces rely on (`∂/∂ρ = value`).
//!  3. POSITIVITY / FLOOR — `B ≻ 0` everywhere, and the deviation from the hard
//!     clamp stays at/below the criterion's spectral-deflation floor `α·1e-8`.
//!  4. C¹ AT THE SEAM — the majorizer's coordinate derivative is the continuous
//!     `α·logistic(cos/τ₀)·(−κ sin κt)` (the hard clamp's `1{cos>0}` step becomes
//!     the logistic passing smoothly through `½` at the seam), matched by finite
//!     differences of the value.

use crate::manifold::ArdAxisPrior;
use gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR;
use std::f64::consts::{LN_2, TAU};

/// τ₀, the dimensionless softplus temperature. Mirrors the derived constant so
/// the test fails loudly if the derivation constant is ever silently retuned.
fn tau0() -> f64 {
    ArdAxisPrior::CLAMP_TEMPERATURE
}

/// The maximum admissible deviation of the smooth clamp from the hard clamp, in
/// curvature units: `α·τ₀·ln2 = α·SPECTRAL_DEFLATION_REL_FLOOR` (the deflation
/// floor relative to the axis's own curvature scale `α`).
fn deviation_ceiling(alpha: f64) -> f64 {
    alpha * tau0() * LN_2
}

#[test]
fn tau0_is_the_derived_deflation_floor_constant() {
    // τ₀ = SPECTRAL_DEFLATION_REL_FLOOR / ln2, and hence α·τ₀·ln2 = α·floor.
    assert_eq!(tau0(), SPECTRAL_DEFLATION_REL_FLOOR / LN_2);
    assert!(
        (deviation_ceiling(1.0) - SPECTRAL_DEFLATION_REL_FLOOR).abs()
            <= 4.0 * f64::EPSILON * SPECTRAL_DEFLATION_REL_FLOOR,
        "the per-axis deviation ceiling must equal the relative deflation floor"
    );
}

#[test]
fn euclidean_clamp_is_a_bit_identical_no_op() {
    // Non-periodic axes have constant curvature `V'' = α > 0`; the clamp must be a
    // literal no-op (bit-for-bit), and the remainder exactly zero.
    for &alpha in &[1e-3, 0.5, 1.0, 7.0, 250.0] {
        for &t in &[-3.0, -0.25, 0.0, 0.8, 5.5] {
            let prior = ArdAxisPrior::eval(alpha, t, None);
            assert_eq!(prior.psd_majorizer_hess(), alpha);
            assert_eq!(prior.negative_hessian_remainder(), 0.0);
            assert_eq!(
                prior.psd_majorizer_hess() + prior.negative_hessian_remainder(),
                prior.hess
            );
        }
    }
}

#[test]
fn seam_identity_reconstructs_exact_hess() {
    // `B + R == V''`. Outside the ~1e-8-wide transition band one term underflows to
    // exactly zero (softplus collapses to the hard clamp), so the reconstruction is
    // bit-for-bit exact; the generic offset grid below never lands in that band.
    let period = 1.0;
    for &alpha in &[1e-2, 1.0, 13.0, 400.0] {
        for i in 0..257 {
            let t = 0.001_3 + i as f64 * (0.997 / 257.0);
            let prior = ArdAxisPrior::eval(alpha, t, Some(period));
            let sum = prior.psd_majorizer_hess() + prior.negative_hessian_remainder();
            assert_eq!(
                sum, prior.hess,
                "seam identity must reconstruct V'' exactly at alpha={alpha}, t={t}"
            );
        }
    }
}

#[test]
fn majorizer_is_strictly_positive_everywhere() {
    // `B ≻ 0`: softplus is strictly positive, so the majorized H_tt is PD even on
    // the concave half where the exact curvature `α cos κt` is negative.
    let period = 1.0;
    for &alpha in &[1e-6, 1e-2, 1.0, 50.0, 1e4] {
        for i in 0..512 {
            let t = i as f64 * (period / 512.0);
            let b = ArdAxisPrior::eval(alpha, t, Some(period)).psd_majorizer_hess();
            assert!(b > 0.0, "majorizer must be strictly positive; got {b} at t={t}");
        }
    }
}

#[test]
fn concave_remainder_is_nonpositive() {
    // `R = V'' − B ≤ 0` (since B = α·softplus(cos) ≥ α·max(cos,0) ≥ α cos = V''),
    // i.e. the restored `E = −R ⪰ 0` — the exact `A = B − E` split is preserved.
    let period = 1.0;
    for &alpha in &[1e-2, 1.0, 30.0] {
        for i in 0..512 {
            let t = i as f64 * (period / 512.0);
            let r = ArdAxisPrior::eval(alpha, t, Some(period)).negative_hessian_remainder();
            assert!(r <= 0.0, "remainder must be non-positive; got {r} at t={t}");
        }
    }
}

#[test]
fn deviation_from_hard_clamp_within_floor() {
    // |B − α·max(cos,0)| ≤ α·τ₀·ln2 everywhere, with equality (the kink is filled)
    // at the seam cos κt = 0.
    let period = 1.0;
    let kappa = TAU / period;
    for &alpha in &[1e-2, 1.0, 25.0, 1e3] {
        let ceiling = deviation_ceiling(alpha);
        let mut max_dev = 0.0_f64;
        for i in 0..2048 {
            let t = i as f64 * (period / 2048.0);
            let cos = (kappa * t).cos();
            let hard = alpha * cos.max(0.0);
            let smooth = ArdAxisPrior::eval(alpha, t, Some(period)).psd_majorizer_hess();
            let dev = (smooth - hard).abs();
            assert!(
                dev <= ceiling + 8.0 * f64::EPSILON * alpha,
                "deviation {dev} exceeds floor ceiling {ceiling} at alpha={alpha}, t={t}"
            );
            max_dev = max_dev.max(dev);
        }
        // At the exact seam cos = 0 the kink is filled to the ceiling: the hard
        // clamp gives 0, the smooth clamp gives α·softplus(0) = α·τ₀·ln2.
        let seam = ArdAxisPrior::smooth_clamp(alpha, 0.0);
        assert!(
            (seam - ceiling).abs() <= 1e-12 * ceiling.max(f64::MIN_POSITIVE),
            "seam value {seam} must equal the deviation ceiling {ceiling}"
        );
        assert!(max_dev > 0.5 * ceiling, "the smoothing must actually engage");
    }
}

#[test]
fn homogeneity_degree_one_in_alpha() {
    // ∂/∂ρ_ard[α·softplus(cos)] = α·softplus(cos) requires exact degree-one
    // homogeneity in α (the explicit-ρ log-det traces depend on it). Doubling α is
    // an exact power-of-two scaling, so the majorizer must double bit-for-bit.
    let period = 1.0;
    for &alpha in &[1e-3, 0.7, 3.0, 90.0] {
        for i in 0..129 {
            let t = 0.002 + i as f64 * (0.996 / 129.0);
            let b1 = ArdAxisPrior::eval(alpha, t, Some(period)).psd_majorizer_hess();
            let b2 = ArdAxisPrior::eval(2.0 * alpha, t, Some(period)).psd_majorizer_hess();
            assert_eq!(2.0 * b1, b2, "majorizer must be exactly degree-1 in alpha");
            // A general (non power-of-two) scale to tight relative tolerance.
            let lambda = 3.5;
            let bl = ArdAxisPrior::eval(lambda * alpha, t, Some(period)).psd_majorizer_hess();
            assert!(
                (bl - lambda * b1).abs() <= 8.0 * f64::EPSILON * bl.abs().max(f64::MIN_POSITIVE),
                "degree-1 homogeneity must hold for a general scale"
            );
        }
    }
}

#[test]
fn slope_is_continuous_through_the_seam() {
    // The hard clamp's derivative is a step `1{cos>0}`; the smooth clamp's is the
    // logistic `s'(c) = logistic(c/τ₀)`, which passes continuously through ½ at the
    // seam (no kink). Monotone, symmetric about (0, ½), saturating to {0,1}.
    assert_eq!(ArdAxisPrior::clamp_slope(0.0), 0.5);
    assert_eq!(ArdAxisPrior::clamp_slope(1.0), 1.0); // saturates within f64
    assert_eq!(ArdAxisPrior::clamp_slope(-1.0), 0.0);
    let mut prev = 0.0_f64;
    for i in 0..401 {
        let c = -1.0 + i as f64 * (2.0 / 400.0);
        let s = ArdAxisPrior::clamp_slope(c);
        assert!((0.0..=1.0).contains(&s));
        assert!(s >= prev - 1e-15, "slope must be non-decreasing in cos");
        prev = s;
        // Symmetry logistic(−z) = 1 − logistic(z).
        let s_neg = ArdAxisPrior::clamp_slope(-c);
        assert!((s + s_neg - 1.0).abs() <= 1e-12);
    }
}

#[test]
fn value_slope_consistency_by_finite_difference() {
    // The analytic coordinate derivative `α·s'(cos)·(−κ sin κt)` (the closed form
    // `ard_majorized_hessian_derivative` assembles) must equal a central finite
    // difference of the majorizer value. Away from the seam the function is the
    // (smooth) hard clamp; at the seam a τ₀-scaled step resolves the transition and
    // recovers the ½-slope, demonstrating C¹.
    let period = 1.0;
    let kappa = TAU / period;
    let alpha = 4.0;

    // (a) Away from the seam: h = 1e-6 is fine (function ≈ α·max(cos,0), smooth).
    for &cos_target in &[0.9_f64, 0.5, -0.5, -0.9] {
        // pick t with cos(κt) = cos_target on the falling branch (sin > 0).
        let t = cos_target.acos() / kappa;
        let h = 1e-6;
        let f = |tt: f64| ArdAxisPrior::eval(alpha, tt, Some(period)).psd_majorizer_hess();
        let fd = (f(t + h) - f(t - h)) / (2.0 * h);
        let cos = (kappa * t).cos();
        let sin = (kappa * t).sin();
        let analytic = -alpha * kappa * sin * ArdAxisPrior::clamp_slope(cos);
        assert!(
            (fd - analytic).abs() <= 1e-4 * (1.0 + analytic.abs()),
            "value/slope FD mismatch away from seam at cos={cos_target}: fd={fd}, analytic={analytic}"
        );
    }

    // (b) At the seam, resolve the τ₀-wide band via smooth_clamp directly. Central
    // FD in the dimensionless cosine with a τ₀-scaled step recovers exactly the
    // logistic midpoint slope ½ — the kink is filled, not a step.
    for &k in &[1.0_f64, 2.0, 5.0] {
        let h = k * tau0();
        let fd = (ArdAxisPrior::smooth_clamp(alpha, h) - ArdAxisPrior::smooth_clamp(alpha, -h))
            / (2.0 * h);
        let analytic = alpha * ArdAxisPrior::clamp_slope(0.0); // = 0.5·α
        assert!(
            (fd - analytic).abs() <= 1e-9 * alpha,
            "seam FD slope {fd} must equal the logistic midpoint {analytic}"
        );
    }
}
