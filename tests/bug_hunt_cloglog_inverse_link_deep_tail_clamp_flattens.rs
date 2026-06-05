//! Regression test for #756: the cloglog branch of `apply_inverse_link_vec`
//! hard-clamped its input to `[-50, 50]` before evaluating the inverse link,
//! so it returned the *constant* value `exp(-50) ≈ 1.93e-22` for every
//! `η ≤ -50`. The cloglog inverse link
//!
//! ```text
//! μ(η) = 1 − exp(−exp(η))
//! ```
//!
//! is strictly increasing on the whole real line, so the clamp made the FFI
//! posterior-band dispatcher both **non-monotone** (flat below −50) and **wrong
//! by orders of magnitude** in the deep tail (`μ(-100)` was ~22 orders of
//! magnitude too large).
//!
//! Issue #746 fixed the *cancellation* in this branch by switching to the
//! stable `-expm1(-exp(η))` form. That form is already finite and accurate all
//! the way to the exp-underflow boundary (η ≈ −745) and the exp-overflow
//! boundary (η ≈ +709.8), so the `[-50, 50]` clamp was unnecessary — and on the
//! negative side it actively threw away the precision #746 restored. The fix is
//! to drop the clamp entirely.
//!
//! These tests pin the invariants the clamp broke:
//!   1. strict monotonicity of μ(η) across the old −50 clamp boundary,
//!   2. the leading-order deep-tail asymptotic μ(η) ~ exp(η) *below* −50,
//!   3. agreement with the crate's own stable evaluator over a wide grid,
//!   4. positive-tail overflow safety (μ → 1, never NaN/inf, never > 1).

use gam::families::inverse_link::apply_inverse_link_vec;

/// Stable cloglog inverse link via `expm1`, exact to all f64 digits across the
/// entire representable η range. Mirrors the crate's own
/// `src/inference/quadrature.rs::cloglog_negative_tail_mean`.
fn cloglog_inv_link_stable(eta: f64) -> f64 {
    let ex = eta.exp();
    -((-ex).exp_m1())
}

#[test]
fn cloglog_inverse_link_is_strictly_monotone_across_the_minus_50_clamp_boundary() {
    // A strictly decreasing η-grid that crosses the old −50 clamp boundary and
    // descends well past it. μ(η) is strictly increasing, so the response on a
    // strictly decreasing η-grid must be strictly decreasing. The old clamp
    // froze every entry at or below −50 to the same exp(-50), violating this.
    let eta = vec![-45.0_f64, -50.0, -55.0, -60.0, -80.0, -100.0];
    let mu =
        apply_inverse_link_vec(&eta, "cloglog").expect("cloglog dispatch succeeds on deep tail");
    assert_eq!(mu.len(), eta.len());

    for win in mu.windows(2) {
        let (prev, next) = (win[0], win[1]);
        assert!(
            prev > next,
            "apply_inverse_link_vec('cloglog') is not strictly monotone across the \
             η = {eta:?} grid: full response sequence {mu:?}. μ(η) = 1 - exp(-exp(η)) \
             is strictly increasing, so a strictly decreasing η-grid must map to a \
             strictly decreasing μ-sequence — the [-50,50] clamp froze every entry \
             ≤ -50 to exp(-50) ≈ 1.93e-22 and broke this."
        );
    }
}

#[test]
fn cloglog_inverse_link_tracks_exp_eta_asymptotic_below_the_clamp_bound() {
    // For η → −∞, μ(η) = exp(η) − ½exp(2η) + O(exp(3η)), so μ(η)/exp(η) → 1.
    // These η all sit *below* the old −50 clamp, exactly where the clamp
    // returned the constant exp(-50) instead of the true ~exp(η).
    for &e in &[-55.0_f64, -60.0, -80.0, -100.0, -200.0] {
        let mu = apply_inverse_link_vec(&[e], "cloglog").expect("cloglog dispatch succeeds")[0];

        // Strictly positive on every finite η — exactly 0.0 would cascade to
        // -inf through any downstream ln(μ).
        assert!(
            mu > 0.0,
            "apply_inverse_link_vec('cloglog')[{e}] = {mu:.3e}; the cloglog inverse \
             link is strictly positive on every finite η"
        );

        let ratio = mu / e.exp();
        assert!(
            (ratio - 1.0).abs() < 1e-12,
            "apply_inverse_link_vec('cloglog')[{e}] = {mu:.6e}; expected μ/exp(η) → 1 \
             in the deep tail but got ratio {ratio:.6e}. The old clamp returned a \
             constant exp(-50) here, giving ratios many orders of magnitude off."
        );
    }
}

#[test]
fn cloglog_inverse_link_matches_stable_reference_over_wide_grid() {
    // Agreement with the stable expm1 reference across the whole representable
    // span, including the deep negative tail the clamp corrupted and the
    // positive tail where μ saturates to 1.
    let eta: Vec<f64> = vec![
        -300.0, -100.0, -60.0, -55.0, -50.0, -45.0, -20.0, -5.0, -1.0, 0.0, 1.0, 5.0, 20.0, 50.0,
        100.0,
    ];
    let mu = apply_inverse_link_vec(&eta, "cloglog").expect("cloglog dispatch succeeds");

    for (&e, &got) in eta.iter().zip(mu.iter()) {
        let reference = cloglog_inv_link_stable(e);
        // Compare in absolute terms near the μ→1 saturation (reference == 1.0)
        // and relative terms elsewhere.
        if reference >= 1.0 {
            assert!(
                (got - 1.0).abs() <= f64::EPSILON,
                "apply_inverse_link_vec('cloglog')[{e}] = {got:.17e}; expected μ = 1 \
                 (stable reference saturates to 1.0)"
            );
        } else {
            let rel_err = (got - reference).abs() / reference;
            assert!(
                rel_err < 1e-12,
                "apply_inverse_link_vec('cloglog')[{e}] = {got:.6e}, stable reference \
                 = {reference:.6e}, relative error = {rel_err:.3e}; the FFI cloglog \
                 helper must agree with the crate's own stable evaluator everywhere."
            );
        }
        // μ is a probability: it must land in [0, 1] and never be NaN/inf.
        assert!(
            got.is_finite() && (0.0..=1.0).contains(&got),
            "apply_inverse_link_vec('cloglog')[{e}] = {got:?} is not a finite \
             probability in [0, 1]"
        );
    }
}

#[test]
fn cloglog_inverse_link_positive_tail_saturates_to_one_without_overflow() {
    // The positive bound of the old clamp was redundant: the -expm1(-exp(η))
    // form saturates to exactly 1.0 once exp(η) overflows to +∞ (η ≳ 709.8),
    // with no NaN/inf leaking out. μ must be non-decreasing into the far tail
    // and never exceed 1.
    let eta = vec![
        10.0_f64,
        50.0,
        100.0,
        500.0,
        709.0,
        710.0,
        1000.0,
        f64::INFINITY,
    ];
    let mu = apply_inverse_link_vec(&eta, "cloglog").expect("cloglog dispatch succeeds");

    for (&e, &got) in eta.iter().zip(mu.iter()) {
        assert!(
            got.is_finite(),
            "apply_inverse_link_vec('cloglog')[{e}] = {got:?}; positive tail must not \
             produce NaN/inf"
        );
        assert!(
            (0.0..=1.0).contains(&got),
            "apply_inverse_link_vec('cloglog')[{e}] = {got}; μ must stay in [0, 1]"
        );
    }
    // Non-decreasing into the tail.
    for win in mu.windows(2) {
        assert!(
            win[1] >= win[0],
            "apply_inverse_link_vec('cloglog') positive tail is not non-decreasing: {mu:?}"
        );
    }
    // Far enough out, μ is indistinguishable from 1.
    let far = *mu.last().unwrap();
    assert!(
        (far - 1.0).abs() <= f64::EPSILON,
        "apply_inverse_link_vec('cloglog')[+inf] = {far}; expected μ = 1 at +∞"
    );
}

#[test]
fn cloglog_inverse_link_underflow_boundary_returns_zero_not_nan() {
    // Below the exp-underflow boundary (η ≲ -745.1) exp(η) underflows to 0,
    // and -expm1(-0) = 0 is the correct limit — a clean 0.0, not NaN.
    let eta = vec![-745.0_f64, -750.0, -1000.0, f64::NEG_INFINITY];
    let mu = apply_inverse_link_vec(&eta, "cloglog").expect("cloglog dispatch succeeds");
    for (&e, &got) in eta.iter().zip(mu.iter()) {
        assert!(
            got.is_finite() && got >= 0.0,
            "apply_inverse_link_vec('cloglog')[{e}] = {got:?}; deep underflow tail \
             must return a finite, non-negative μ (0.0 in the limit), not NaN"
        );
    }
}
