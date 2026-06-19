//! EXAMPLE micro-repro (template for the convention; mirrors gam#1348).
//!
//! This is the small-n, fast counterpart of the basis_smooth gate test
//! `nonperiodic_bspline_first_derivative_matches_central_difference`. It runs
//! in well under a second and exercises the same observable: on a uniform OPEN
//! knot vector the analytic first derivative of the B-spline basis must match a
//! central difference of the basis value everywhere, including boundary spans.
//!
//! Copy this file's shape for a new issue: keep n small, keep the assert on the
//! real observable, and keep the tolerance honest (no loosening just to pass).

use gam::terms::basis::{BasisOptions, Dense, KnotSource, create_basis};
use ndarray::Array1;

#[test]
fn micro_open_knot_bspline_derivative_matches_central_difference() {
    let degree = 3usize;
    // Uniform OPEN knots; small but enough spans to cover both boundaries.
    let knots = Array1::linspace(-2.0, 3.0, 10);
    let n = 41usize; // small-n: the gate test uses 121
    let lo = knots[0];
    let hi = knots[knots.len() - 1];
    let tt = Array1::from_iter((0..n).map(|i| lo + (hi - lo) * (i as f64) / ((n - 1) as f64)));

    let h = 1e-6;
    let value_at = |shift: f64| {
        create_basis::<Dense>(
            tt.mapv(|v| v + shift).view(),
            KnotSource::Provided(knots.view()),
            degree,
            BasisOptions::value(),
        )
        .expect("value basis")
        .0
    };
    let v_plus = value_at(h);
    let v_minus = value_at(-h);
    let (d1, _) = create_basis::<Dense>(
        tt.view(),
        KnotSource::Provided(knots.view()),
        degree,
        BasisOptions::first_derivative(),
    )
    .expect("first-derivative basis");

    let mut max_err = 0.0_f64;
    for ((a, p), m) in d1.iter().zip(v_plus.iter()).zip(v_minus.iter()) {
        let fd = (p - m) / (2.0 * h);
        max_err = max_err.max((a - fd).abs());
    }
    assert!(
        max_err < 1e-5,
        "open-knot first derivative disagrees with central difference: max|d1 - fd| = {max_err:e}"
    );
}
