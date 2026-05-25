//! Contract test for [`gam::families::cubic_cell_kernel::transformed_link_cubic`].
//!
//! The kernel returns the **polynomial coefficients** of
//! `f(z) = link_span.evaluate(a + b*z)` in `z`, i.e.
//!
//! ```text
//!     f(z) = d0 + d1·z + d2·z² + d3·z³.
//! ```
//!
//! This is not the same as the Taylor *derivatives* of `f` at `z = 0` (those
//! would be `f^(k)(0) = k!·d_k`); confusing the two would silently scale `d2`
//! and `d3` by factors of 2 and 6, respectively, and corrupt every downstream
//! caller in the de-nested cubic transport kernel (cell coefficients, link
//! basis coefficients, moment-recurrence inputs, ...).  This test pins the
//! polynomial-coefficient contract by reconstructing `f(z)` from
//! `(d0, d1, d2, d3)` at several `z` values to bit-precision.

use gam::families::cubic_cell_kernel::{LocalSpanCubic, transformed_link_cubic};

#[test]
fn cubic_cell_kernel_transformed_link_returns_polynomial_coefficients_in_z() {
    let span = LocalSpanCubic {
        left: -0.7,
        right: 2.0,
        c0: 0.21,
        c1: -0.43,
        c2: 0.37,
        c3: -0.19,
    };
    let a = 0.38;
    let b = -1.17;

    let (d0, d1, d2, d3) = transformed_link_cubic(span, a, b);

    // `d0 = f(0)` exactly (no rounding aside from one evaluate call).
    let f0 = span.evaluate(a);
    assert!((d0 - f0).abs() < 1e-14, "d0={d0} should equal f(0)={f0}");

    // Reconstruct `f(z)` from the polynomial coefficients at a handful of
    // points and compare to the direct cubic evaluation. Any factor-of-`k!`
    // confusion between coefficients and derivatives would break this.
    for &z in &[-0.9_f64, -0.3, 0.0, 0.25, 0.7, 1.4] {
        let f = span.evaluate(a + b * z);
        let poly = d0 + d1 * z + d2 * z * z + d3 * z * z * z;
        assert!(
            (poly - f).abs() < 1e-12,
            "z={z}: poly={poly} should equal f(z)={f} (d0={d0}, d1={d1}, d2={d2}, d3={d3})"
        );
    }

    // Spell out the closed-form coefficients to nail the contract: the
    // kernel returns the *polynomial* coefficients, so `d2 = b²·(c2 + 3·c3·s)`
    // and `d3 = c3·b³` (not the second/third derivatives of `f`).
    let shift = a - span.left;
    let expected_d1 = b * (span.c1 + 2.0 * span.c2 * shift + 3.0 * span.c3 * shift * shift);
    let expected_d2 = b * b * (span.c2 + 3.0 * span.c3 * shift);
    let expected_d3 = span.c3 * b * b * b;
    assert!(
        (d1 - expected_d1).abs() < 1e-14,
        "d1={d1} expected {expected_d1}"
    );
    assert!(
        (d2 - expected_d2).abs() < 1e-14,
        "d2={d2} expected {expected_d2}"
    );
    assert!(
        (d3 - expected_d3).abs() < 1e-14,
        "d3={d3} expected {expected_d3}"
    );
}
