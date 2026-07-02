//! Regression guard for `affine_anchor_moment_vector`'s zeroth truncated-
//! Gaussian moment (issue #352), covering the angles the committed
//! deep-*negative*-tail test does not: the symmetric **positive** deep tail,
//! zero-straddling intervals, and infinite (semi-infinite cell) endpoints.
//!
//! The bug is catastrophic cancellation in the `½(1 + erf)` CDF difference.
//! The naive form — and a negative-tail-only `erfc(−x/√2)` patch — both still
//! collapse in the *positive* deep tail, where `erf(x/√2)` saturates at
//! `+1.0` (equivalently `erfc(−x/√2)` saturates at `2.0`) and the CDF
//! difference becomes `1 − 1` (resp. `2 − 2`). Since the affine map placing
//! the truncation window `(y_left, y_right)` can land anywhere on ℝ — outer
//! spline cells routinely sit in either tail — a correct fix must hold in
//! *both* tails, not just the one the committed repro exercises.

use gam::families::cubic_cell_kernel::affine_anchor_moment_vector;

/// Stable reference for `T_0(a, b) = ∫_a^b e^(−z²/2) dz = √(2π)(Φ(b) − Φ(a))`,
/// computed in whichever tail keeps full precision.
fn t0_reference(a: f64, b: f64) -> f64 {
    let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
    let za = a * inv_sqrt2;
    let zb = b * inv_sqrt2;
    let erf_diff = if za >= 0.0 {
        libm::erfc(za) - libm::erfc(zb)
    } else if zb <= 0.0 {
        libm::erfc(-zb) - libm::erfc(-za)
    } else if zb <= 0.5 && -za <= 0.5 {
        libm::erf(zb) + libm::erf(-za)
    } else {
        2.0 - libm::erfc(zb) - libm::erfc(-za)
    };
    (std::f64::consts::PI / 2.0).sqrt() * erf_diff
}

#[test]
fn t0_is_strictly_positive_and_accurate_in_deep_positive_tail() {
    // The mirror image of the committed negative-tail test. With α = β = 0 the
    // affine map is the identity, so out[0] = ∫_{10}^{12} e^(−z²/2) dz ≈ 1.9e-23,
    // strictly positive and well inside the f64 normal range. A negative-tail-
    // only fix leaves this collapsing to 0.0.
    let out = affine_anchor_moment_vector(0.0, 0.0, 10.0, 12.0, 2);
    let t0 = out[0];
    assert!(
        t0 > 0.0,
        "affine_anchor_moment_vector(0,0,10,12)[0] = {t0:.3e}; the zeroth \
         truncated-Gaussian moment over a deep positive interval is strictly \
         positive and must not collapse to 0.0 — a both-tails fix is required"
    );
    let reference = t0_reference(10.0, 12.0);
    let rel_err = (t0 - reference).abs() / reference;
    assert!(
        rel_err < 1e-9,
        "affine_anchor_moment_vector(0,0,10,12)[0] = {t0:.6e}, stable reference \
         √(2π)(Φ(12)−Φ(10)) = {reference:.6e}, relative error = {rel_err:.3e}"
    );
}

#[test]
fn t0_symmetric_under_reflection_of_the_interval() {
    // T_0(a, b) = T_0(−b, −a): the standard normal is even, so reflecting the
    // window about 0 must leave the integral unchanged to f64. This pins both
    // tails to the *same* value and would fail loudly if only one tail were fixed.
    let pairs = [
        (2.0, 4.0),
        (6.0, 8.0),
        (8.0, 9.0),
        (10.0, 12.0),
        (11.0, 13.0),
    ];
    for &(a, b) in &pairs {
        let pos = affine_anchor_moment_vector(0.0, 0.0, a, b, 2)[0];
        let neg = affine_anchor_moment_vector(0.0, 0.0, -b, -a, 2)[0];
        assert!(pos > 0.0 && neg > 0.0, "T_0 must be positive in both tails");
        let rel = (pos - neg).abs() / pos.max(f64::MIN_POSITIVE);
        assert!(
            rel < 1e-12,
            "T_0([{a},{b}]) = {pos:.6e} but T_0([{}, {}]) = {neg:.6e} (rel diff \
             {rel:.3e}); the even symmetry of e^(−z²/2) forces these to be equal",
            -b,
            -a
        );
    }
}

#[test]
fn t0_strictly_decreasing_marching_into_positive_tail() {
    // Mirror of the committed negative-tail monotonicity test. Unit-width
    // windows marching out into the positive tail integrate strictly smaller
    // positive integrands, so T_0 must be strictly positive and strictly
    // decreasing — consecutive zeros (the bug) would violate this.
    let intervals = [(6.0, 7.0), (8.0, 9.0), (10.0, 11.0), (12.0, 13.0)];
    let t0: Vec<f64> = intervals
        .iter()
        .map(|&(a, b)| affine_anchor_moment_vector(0.0, 0.0, a, b, 2)[0])
        .collect();
    for &v in &t0 {
        assert!(v > 0.0, "T_0 in the positive tail must be > 0; got {t0:?}");
    }
    for win in t0.windows(2) {
        assert!(
            win[0] > win[1],
            "T_0 not strictly decreasing across positive-tail intervals: {t0:?}"
        );
    }
}

#[test]
fn t0_handles_straddling_and_infinite_endpoints() {
    // Zero-straddling window: exact, no cancellation regime, sanity vs reference.
    let straddle = affine_anchor_moment_vector(0.0, 0.0, -1.0, 2.0, 2)[0];
    let straddle_ref = t0_reference(-1.0, 2.0);
    assert!(
        (straddle - straddle_ref).abs() / straddle_ref < 1e-12,
        "T_0([-1,2]) = {straddle:.10e} vs reference {straddle_ref:.10e}"
    );

    // Whole line: ∫_{−∞}^{∞} e^(−z²/2) dz = √(2π).
    let full = affine_anchor_moment_vector(0.0, 0.0, f64::NEG_INFINITY, f64::INFINITY, 2)[0];
    let two_pi_sqrt = (2.0 * std::f64::consts::PI).sqrt();
    assert!(
        (full - two_pi_sqrt).abs() / two_pi_sqrt < 1e-12,
        "T_0([-inf, inf]) = {full:.10e}, expected √(2π) = {two_pi_sqrt:.10e}"
    );

    // Semi-infinite tail in the deep positive region: ∫_{10}^{∞} e^(−z²/2) dz,
    // strictly positive (≈ 1.5e-23). Naive form returns √(2π)(1 − 1) = 0.0.
    let upper = affine_anchor_moment_vector(0.0, 0.0, 10.0, f64::INFINITY, 2)[0];
    let upper_ref = t0_reference(10.0, f64::INFINITY);
    assert!(
        upper > 0.0 && (upper - upper_ref).abs() / upper_ref < 1e-9,
        "T_0([10, inf]) = {upper:.10e} vs reference {upper_ref:.10e}; semi-infinite \
         positive tail must stay strictly positive"
    );

    // Semi-infinite tail in the deep negative region: ∫_{−∞}^{−10}, mirror value.
    let lower = affine_anchor_moment_vector(0.0, 0.0, f64::NEG_INFINITY, -10.0, 2)[0];
    assert!(
        (lower - upper).abs() / upper < 1e-12,
        "T_0([-inf,-10]) = {lower:.10e} must equal T_0([10,inf]) = {upper:.10e} by symmetry"
    );
}

#[test]
fn t0_handles_tiny_interval_straddling_affine_anchor() {
    // A straddling interval may be much narrower than machine epsilon around
    // the O(1) erfc terms. Computing it as `2 - erfc(b) - erfc(-a)` rounds both
    // erfc terms to 1 and loses the cell mass entirely; the anchor-centered
    // `erf(b) + erf(-a)` form keeps the small positive integral.
    let eps = 1.0e-300;
    let got = affine_anchor_moment_vector(0.0, 0.0, -eps, eps, 2)[0];
    let reference = 2.0 * eps;
    let rel = (got - reference).abs() / reference;
    assert!(
        got > 0.0 && rel < 1e-12,
        "T_0([-{eps:e},{eps:e}]) = {got:.10e} vs local reference {reference:.10e}; \
         tiny anchor-straddling cells must not cancel to zero"
    );
}
