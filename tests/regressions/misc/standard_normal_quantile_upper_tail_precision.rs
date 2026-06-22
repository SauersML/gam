//! Regression for issue #513: `standard_normal_quantile` (Φ⁻¹) is
//! machine-accurate in the lower tail (`p → 0`) but loses ~6–7 orders of
//! magnitude of accuracy in the upper tail (`p → 1`).
//!
//! Root cause: the two Halley polishing steps in
//! `src/inference/probability.rs` form the residual `F(x) - p` as
//! `normal_cdf(x) - p`. For an upper-tail seed `x > 0`, `normal_cdf(x)`
//! saturates to ~1 (`Φ(6.36) = 1 − 1e-10`); subtracting `p ≈ 1 − 1e-10`
//! annihilates every significant digit of the ~1e-10 residual the refinement
//! is meant to act on, so the loop is a no-op and the result is stuck at the
//! Acklam rational seed's ~1e-9 accuracy.
//!
//! The cancellation-free upper-tail residual is
//! `F(x) − p = (1 − p) − 0.5·erfc(x/√2)`, where both terms are the *small*
//! upper-tail quantities (no near-1 subtraction; `1 − p` is exact by Sterbenz
//! for `p ∈ [½,1)`). The lower tail (`x ≤ 0`) keeps the direct
//! `normal_cdf(x) − p = 0.5·erfc(|x|/√2) − p`, which never cancels.
//!
//! This is the inverse-CDF sibling of the probit/cloglog deep-tail
//! cancellation (#351) and the truncated-Gaussian moment bug (#352).
//! Related: #514 (poincaré arccosh), #515 (sphere acos).

use gam::probability::standard_normal_quantile;

/// By the exact symmetry `Φ⁻¹(p) = −Φ⁻¹(1 − p)`, the upper-tail input and its
/// mirror lower-tail input recover the same magnitude. The lower tail already
/// polishes to ~1e-12; a correct upper tail must match it (and the external
/// scipy `ndtri` reference) to the same precision instead of being stuck at
/// the ~1e-9 rational seed.
#[test]
fn upper_tail_quantile_matches_reference_like_the_lower_tail() {
    // (upper p, scipy ndtri(p)) pairs; |Φ⁻¹| ranges 6.36 .. 7.94.
    let cases = [
        (0.9999999999_f64, 6.361340889697422_f64),
        (0.99999999999_f64, 6.706023143414748_f64),
        (0.999999999999_f64, 7.0344869100478356_f64),
        (0.9999999999999_f64, 7.3487545403000425_f64),
        (0.999999999999999_f64, 7.941444487415979_f64),
    ];

    for (p, reference) in cases {
        let hi = standard_normal_quantile(p).expect("upper-tail quantile");
        let abs_err = (hi - reference).abs();

        // The mirror lower-tail input must hit the same magnitude; this is the
        // accuracy the polish already achieves on the lower side.
        let lo = standard_normal_quantile(1.0 - p).expect("mirror lower-tail quantile");
        let mirror_err = (lo + reference).abs();

        assert!(
            abs_err < 1e-11,
            "Φ⁻¹({p}) = {hi:.16}, scipy ndtri = {reference:.16}, abs err {abs_err:.3e} \
             (>1e-11). The mirror lower input Φ⁻¹(1−p) recovers the same magnitude to \
             abs err {mirror_err:.3e}; the upper tail must polish to the same precision, \
             not stay stuck at the ~1e-9 rational seed."
        );
        // Symmetry must hold to near machine precision once the polish works.
        assert!(
            (hi + lo).abs() < 1e-11,
            "broken symmetry Φ⁻¹(p) = −Φ⁻¹(1−p): Φ⁻¹({p}) = {hi:.16}, \
             Φ⁻¹(1−p) = {lo:.16}, sum {:.3e}",
            (hi + lo).abs()
        );
    }
}

/// Bonferroni simultaneous bands push `p` toward 1 as the row count grows:
/// `p = 0.5 + 0.5·(1 − α/N)`. For a 95% band over `N = 1e6` rows this is
/// `p = 0.9999999749999999`, where the cancelling residual costs ~3e-10 of
/// accuracy versus the cancellation-free form.
#[test]
fn bonferroni_simultaneous_band_quantile_is_accurate() {
    let alpha = 0.05_f64;
    let n = 1_000_000.0_f64;
    let p = 0.5 + 0.5 * (1.0 - alpha / n);
    // scipy ndtri(0.9999999749999999)
    let reference = 5.451310437346854_f64;

    let z = standard_normal_quantile(p).expect("band quantile");
    let abs_err = (z - reference).abs();
    assert!(
        abs_err < 1e-11,
        "N=1e6 simultaneous 95% band quantile Φ⁻¹({p}) = {z:.16}, scipy ndtri = \
         {reference:.16}, abs err {abs_err:.3e} (>1e-11)."
    );
}
