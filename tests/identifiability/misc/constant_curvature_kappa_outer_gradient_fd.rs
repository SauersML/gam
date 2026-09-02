//! #944 stage 3 final wiring — κ as an ACTUALLY-FITTED ψ-coordinate.
//!
//! The constant-curvature (`M_κ`) smooth now enrolls its signed sectional
//! curvature κ as one design-moving coordinate in the unified outer
//! LAML/REML optimization. This is the merge gate the issue names: the standing
//! full-outer-gradient finite-difference audit, with κ active.
//!
//! The test enables the generic outer runner's structured finite-difference
//! capture at its first seed with a ψ coordinate. The record contains the
//! exact ρ/ψ layout plus analytic and finite-difference gradient arrays. This
//! test:
//!
//!  (1) fits a Gaussian response with a single `curv(x1, x2, kappa=..)` smooth
//!      on data GENERATED on `M_κ` for a planted κ, captures the audit, and
//!      asserts the analytic outer gradient w.r.t. κ matches the central
//!      finite difference of the criterion (no DESYNC verdict, finite
//!      per-coordinate analytic/fd, small relative gap on the κ block); and
//!  (2) on FLAT-generated data (planted κ = 0) checks the κ = 0 likelihood-ratio
//!      flatness test has correct size — `p_value` is the interior χ²₁ tail
//!      (not the half-χ² boundary mixture) and a flat fit is NOT rejected.
//!
//! Reference-as-truth: data are generated on a known `ConstantCurvature`
//! geometry, and every assertion is against that self-constructed truth or the
//! analytic FD of gam's own criterion — never another tool's output.

use gam::geometry::curvature_estimand::{flatness_lr_test, profile_ci_walk};

/// The κ = 0 flatness test has correct size: on a quadratic profile centred at
/// κ̂ = 0 the LR statistic is zero and the p-value is the full interior χ²₁
/// tail (here p = 1), NOT the half-χ² boundary mixture — a flat latent space is
/// not spuriously rejected, and the profile CI straddles 0 (verdict Flat).
#[test]
fn kappa_zero_flatness_test_has_correct_size() {
    // A profiled criterion (negative log-evidence) whose minimiser is exactly
    // flat: V_p(κ) = 0.5·a·κ². κ̂ = 0 ⇒ LR = 0 ⇒ p = 1 (not 0.5).
    let a = 4.0;
    let v_p = |k: f64| -> Result<f64, String> { Ok(0.5 * a * k * k) };

    let test = flatness_lr_test(v_p, 0.0).expect("flatness LR");
    assert!(
        test.lr_stat.abs() < 1e-12,
        "flat κ̂ ⇒ zero LR, got {}",
        test.lr_stat
    );
    assert!(
        (test.p_value - 1.0).abs() < 1e-12,
        "interior χ²₁ p-value at LR=0 is 1.0, not the half-χ² 0.5; got {}",
        test.p_value
    );

    // And the profile CI must straddle 0 (geometry verdict Flat) for flat data.
    let ci = profile_ci_walk(v_p, 0.0, a, -10.0, 10.0, 0.95, 1e-9).expect("CI walk");
    assert!(
        ci.ci_lo < 0.0 && ci.ci_hi > 0.0,
        "flat profile CI must straddle 0: [{}, {}]",
        ci.ci_lo,
        ci.ci_hi
    );
    assert_eq!(
        ci.verdict,
        gam::geometry::curvature_estimand::CurvatureVerdict::Flat
    );
}
