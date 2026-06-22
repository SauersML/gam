//! Deep-tail precision bug in the truncated-Gaussian zeroth moment computed by
//! `affine_anchor_moment_vector` (`src/families/cubic_cell_kernel.rs`).
//!
//! `affine_anchor_moment_vector(α, β, left, right, max_degree)` builds the raw
//! affine-transformed truncated standard-normal moments.  Its degree-0 entry is
//!
//! ```text
//! out[0] = anchor · T_0(a, b),   T_0(a, b) = √(2π)·(Φ(b) − Φ(a)) = ∫_a^b e^{−z²/2} dz
//! ```
//!
//! (`fill_truncated_gaussian_moments`, line 3075).  With α = β = 0 the affine
//! map is the identity (`anchor = 1`, `a = left`, `b = right`), so
//! `out[0] = ∫_left^right e^{−z²/2} dz` exactly.  That integrand is strictly
//! positive everywhere, hence `out[0] > 0` for every non-degenerate finite
//! interval `left < right` — this is a hard mathematical contract, independent
//! of how the implementation evaluates Φ.
//!
//! The implementation evaluates Φ with the cancellation-prone closure
//!
//! ```ignore
//! let cdf = |x: f64| 0.5 * (1.0 + statrs::function::erf::erf(x / std::f64::consts::SQRT_2));
//! ```
//!
//! (line 3072).  `erf(x/√2)` saturates at the IEEE-754 value `-1.0` for
//! x ≲ -8.3, so `cdf(x)` returns exactly `0.0` there.  For a deep-tail interval
//! such as `[-12, -10]` both `cdf(left)` and `cdf(right)` collapse to `0.0`, and
//! `T_0` — a strictly positive integral (≈ 1.9e-23) — is reported as exactly
//! `0.0`.  Even at the boundary (`[-9, -8]`) the surviving endpoint already
//! carries ~2% relative error from the same cancellation.
//!
//! This is the same root cause as the probit / cloglog `apply_inverse_link_vec`
//! deep-tail bugs (#344 and its probit sibling): a cancelling `½(1 + erf)` form
//! where the complementary evaluator `Φ(x) = ½·erfc(−x/√2)` is exact.  Here it
//! manifests as a difference of two CDFs collapsing to zero rather than a single
//! CDF underflowing.  Computing the difference as
//! `½·(erfc(−b/√2) − erfc(−a/√2))` keeps every digit, since `erfc` returns the
//! tiny tail values accurately instead of as `1 − 1`.
//!
//! `affine_anchor_moment_vector` backs the transformation-normal and
//! bernoulli-marginal-slope affine-cell solves (`evaluate_affine_cell_*`), where
//! a vanishing T_0 silently zeroes the entire moment vector for that cell.

use gam::families::cubic_cell_kernel::affine_anchor_moment_vector;

/// Stable T_0(a, b) = √(2π)·(Φ(b) − Φ(a)) via the complementary error function.
/// For a ≤ b ≤ 0 both `erfc` evaluations return tiny *positive* tail values
/// computed without cancelling against 1, so the difference keeps full f64
/// precision down to the underflow boundary.
fn truncated_gaussian_t0_stable(a: f64, b: f64) -> f64 {
    let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
    let phi_b = 0.5 * statrs::function::erf::erfc(-b * inv_sqrt2);
    let phi_a = 0.5 * statrs::function::erf::erfc(-a * inv_sqrt2);
    (2.0 * std::f64::consts::PI).sqrt() * (phi_b - phi_a)
}

#[test]
fn affine_anchor_t0_is_strictly_positive_on_deep_negative_interval() {
    // Identity affine map (α = β = 0) ⇒ out[0] = ∫_{-12}^{-10} e^{−z²/2} dz.
    // The integrand is strictly positive, so the integral over a non-degenerate
    // interval is strictly positive (true value ≈ 1.9e-23, well within the f64
    // normal range).
    let out = affine_anchor_moment_vector(0.0, 0.0, -12.0, -10.0, 2);
    let t0 = out[0];
    assert!(
        t0 > 0.0,
        "affine_anchor_moment_vector(0,0,-12,-10)[0] = {t0:.3e}; the zeroth \
         truncated-Gaussian moment ∫_{{-12}}^{{-10}} e^(-z²/2) dz is strictly \
         positive and must not collapse to exactly 0.0"
    );

    let reference = truncated_gaussian_t0_stable(-12.0, -10.0);
    let rel_err = (t0 - reference).abs() / reference;
    assert!(
        rel_err < 1e-9,
        "affine_anchor_moment_vector(0,0,-12,-10)[0] = {t0:.6e}, stable reference \
         √(2π)(Φ(-10)-Φ(-12)) = {reference:.6e}, relative error = {rel_err:.3e}"
    );
}

#[test]
fn affine_anchor_t0_matches_stable_reference_at_cancellation_boundary() {
    // [-9, -8] still produces a representable nonzero T_0, but the ½(1+erf) form
    // carries ~2% relative error here (Φ(-8) alone is ~2% off), while the
    // erfc-difference form is exact.
    let out = affine_anchor_moment_vector(0.0, 0.0, -9.0, -8.0, 2);
    let t0 = out[0];
    let reference = truncated_gaussian_t0_stable(-9.0, -8.0);
    let rel_err = (t0 - reference).abs() / reference;
    assert!(
        rel_err < 1e-6,
        "affine_anchor_moment_vector(0,0,-9,-8)[0] = {t0:.6e}, stable reference = \
         {reference:.6e}, relative error = {rel_err:.3e}; the cancelling \
         ½(1+erf) CDF loses ~2% here while ½·erfc keeps full precision"
    );
}

#[test]
fn affine_anchor_t0_is_strictly_decreasing_across_adjacent_deep_intervals() {
    // T_0 over unit-width intervals marching into the negative tail is strictly
    // positive and strictly decreasing (each integrates a strictly smaller
    // positive integrand). The naive form returns nonzero values near the
    // boundary then collapses every deep interval to 0.0, so consecutive zeros
    // violate strict monotone decrease.
    let intervals = [(-7.0, -6.0), (-9.0, -8.0), (-11.0, -10.0), (-13.0, -12.0)];
    let t0: Vec<f64> = intervals
        .iter()
        .map(|&(a, b)| affine_anchor_moment_vector(0.0, 0.0, a, b, 2)[0])
        .collect();

    for &v in &t0 {
        assert!(
            v > 0.0,
            "affine_anchor_moment_vector T_0 over adjacent deep intervals must be \
             strictly positive; got sequence {t0:?}"
        );
    }
    for win in t0.windows(2) {
        assert!(
            win[0] > win[1],
            "affine_anchor_moment_vector T_0 is not strictly decreasing across the \
             intervals [-7,-6], [-9,-8], [-11,-10], [-13,-12]: sequence {t0:?}. \
             Each successive interval integrates a strictly smaller positive \
             integrand, so T_0 must strictly decrease"
        );
    }
}
