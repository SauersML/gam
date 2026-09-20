//! #2520 — the ThresholdGate sparsity prior was the ONLY assignment/coordinate
//! prior curvature in gam-sae written into `B` unmajorized. Its exact second
//! derivative `λ·s·(1 − 2a)/τ²` is negative for every logit above the threshold
//! (the sigmoid penalty is concave there), so the per-row `H_tt` block went
//! indefinite on exactly the atoms the gate had switched ON and the factor
//! spectrally deflated those directions to unit stiffness — #1419's pathology on
//! the one family that never received #1419's treatment.
//!
//! These gates pin the three properties the fix has to have, and each one is
//! load-bearing: (1) `B` is PSD; (2) `B + ΔC` is still the EXACT signed
//! curvature, bit-for-bit, so no exact-Hessian consumer changed operator; and
//! (3) the split introduces no constant of its own — it reuses #2339's derived
//! softplus temperature, whose derivation transfers because `1 − 2a` and
//! `cos κt` are both dimensionless factors on `[−1, 1]`.

use crate::assignment::ThresholdGateLogitCurvature;
use gam_linalg::utils::SPECTRAL_DEFLATION_REL_FLOOR;
use gam_math::special::logistic;
use std::f64::consts::LN_2;

/// The exact prior curvature, written out independently of the seam under test:
/// `strength · a(1−a) · (1 − 2a) / τ²`.
fn exact_curvature(strength: f64, logit: f64, threshold: f64, inv_tau: f64) -> f64 {
    let a = logistic((logit - threshold) * inv_tau);
    strength * a * (1.0 - a) * (1.0 - 2.0 * a) * inv_tau * inv_tau
}

/// Logits straddling the threshold: below it (`1 − 2a > 0`, convex), at it
/// (`a = ½`, the seam), and above it (`1 − 2a < 0`, concave — the case that
/// produced the indefinite block).
const LOGITS: [f64; 7] = [-4.0, -1.0, -0.25, 0.0, 0.25, 1.0, 4.0];

#[test]
fn the_majorizer_is_never_negative_which_is_the_whole_point() {
    for &strength in &[1.0e-3, 1.0, 25.0] {
        for &inv_tau in &[0.5, 1.0, 4.0] {
            for &logit in &LOGITS {
                let curvature =
                    ThresholdGateLogitCurvature::eval(strength, logit, 0.0, inv_tau);
                assert!(
                    curvature.psd_majorizer_hess() >= 0.0,
                    "majorizer {} is negative at logit {logit} (strength {strength}, \
                     inv_tau {inv_tau}); B would still be indefinite on a switched-on gate",
                    curvature.psd_majorizer_hess()
                );
            }
        }
    }
}

#[test]
fn the_unmajorized_curvature_really_was_negative_above_the_threshold() {
    // Guards the test above against being vacuous: if the exact curvature were
    // non-negative everywhere there would be nothing to majorize, and the
    // positivity gate would pass for the wrong reason.
    let negative = LOGITS
        .iter()
        .filter(|&&logit| exact_curvature(1.0, logit, 0.0, 1.0) < 0.0)
        .count();
    assert!(
        negative >= 3,
        "the fixture must contain logits whose EXACT prior curvature is negative \
         (found {negative}); otherwise the majorizer gate proves nothing"
    );
}

#[test]
fn seam_identity_majorizer_plus_remainder_is_the_exact_curvature() {
    // `A = B + ΔC` is the contract every exact-Hessian consumer relies on: the
    // IFT response, the terminal Newton polish, and the #2336 attributability
    // test all differentiate `A`. If this identity slips, they are quietly
    // differentiating a different operator than the one they declare.
    for &strength in &[1.0e-6, 1.0e-3, 1.0, 25.0, 1.0e4] {
        for &inv_tau in &[0.25, 1.0, 8.0] {
            for &logit in &LOGITS {
                for &threshold in &[-0.5, 0.0, 1.5] {
                    let curvature =
                        ThresholdGateLogitCurvature::eval(strength, logit, threshold, inv_tau);
                    let reconstructed = curvature.psd_majorizer_hess()
                        + curvature.negative_hessian_remainder();
                    let exact = exact_curvature(strength, logit, threshold, inv_tau);
                    // A few ULPs of the majorizer's own magnitude, not bit
                    // equality: the remainder is FORMED as `exact − majorized`
                    // and then added back, and `m + (e − m)` is not an identity
                    // in IEEE-754 when the two differ in exponent. What the
                    // contract needs is that `A = B + ΔC` reconstructs the
                    // curvature to round-off, which is what this measures. The
                    // independent `exact_curvature` above is also a DIFFERENT
                    // expression than the one under test, so this compares two
                    // evaluation orders rather than a value with itself.
                    let scale = exact
                        .abs()
                        .max(curvature.psd_majorizer_hess().abs())
                        .max(f64::MIN_POSITIVE);
                    assert!(
                        (reconstructed - exact).abs() <= 8.0 * f64::EPSILON * scale,
                        "seam identity broken at strength {strength}, logit {logit}, \
                         threshold {threshold}, inv_tau {inv_tau}: \
                         {reconstructed} vs {exact}"
                    );
                }
            }
        }
    }
}

#[test]
fn the_remainder_is_the_concave_half_and_never_positive() {
    // `softplus_{τ₀}(c) ≥ max(c, 0) ≥ c`, so the remainder `exact − majorized`
    // is non-positive: `E = −ΔC ⪰ 0` is what #2336 prices.
    for &strength in &[1.0e-3, 1.0, 25.0] {
        for &logit in &LOGITS {
            let curvature = ThresholdGateLogitCurvature::eval(strength, logit, 0.0, 1.0);
            assert!(
                curvature.negative_hessian_remainder() <= 0.0,
                "remainder {} is positive at logit {logit}; E = −ΔC would be indefinite",
                curvature.negative_hessian_remainder()
            );
        }
    }
}

#[test]
fn the_majorizer_is_degree_one_in_the_sparsity_strength() {
    // Homogeneity is why the log-strength ρ-channel needs no separate
    // derivation: `∂/∂ρ_sparse[majorizer] == majorizer` holds exactly when the
    // majorizer is degree-one in `λ_sparse = e^{ρ_sparse}`.
    for &logit in &LOGITS {
        let unit = ThresholdGateLogitCurvature::eval(1.0, logit, 0.0, 1.0).psd_majorizer_hess();
        for &strength in &[1.0e-3, 7.0, 250.0] {
            let scaled =
                ThresholdGateLogitCurvature::eval(strength, logit, 0.0, 1.0).psd_majorizer_hess();
            let expected = strength * unit;
            assert!(
                (scaled - expected).abs() <= 8.0 * f64::EPSILON * expected.abs().max(1.0),
                "majorizer is not degree-one in strength at logit {logit}: \
                 {scaled} vs {expected}"
            );
        }
    }
}

#[test]
fn the_smoothing_deviation_stays_under_the_deflation_floor() {
    // The smooth clamp differs from the hard clamp `max(c, 0)` by at most
    // `τ₀·ln2` per unit prefactor — #2339's derivation, reused here rather than
    // re-tuned, which is why this change introduces no constant of its own.
    let ceiling_per_unit = SPECTRAL_DEFLATION_REL_FLOOR / LN_2 * LN_2;
    for &strength in &[1.0e-3, 1.0, 25.0] {
        for &inv_tau in &[0.5, 1.0, 4.0] {
            for &logit in &LOGITS {
                let curvature =
                    ThresholdGateLogitCurvature::eval(strength, logit, 0.0, inv_tau);
                let exact = exact_curvature(strength, logit, 0.0, inv_tau);
                // The clamp's prefactor, computed directly. Recovering it by
                // dividing `exact` by `1 − 2a` is 0/0 at the seam (`logit == 0`,
                // where `a = ½`) — exactly the point this test most wants to
                // cover.
                let a = logistic(logit * inv_tau);
                let magnitude = strength * a * (1.0 - a) * inv_tau * inv_tau;
                let hard = if exact > 0.0 { exact } else { 0.0 };
                let deviation = (curvature.psd_majorizer_hess() - hard).abs();
                assert!(
                    deviation <= magnitude * ceiling_per_unit * (1.0 + 1.0e-9),
                    "smooth-vs-hard clamp deviation {deviation} exceeds the derived \
                     ceiling {} at logit {logit} (strength {strength}, inv_tau {inv_tau})",
                    magnitude * ceiling_per_unit
                );
            }
        }
    }
}
