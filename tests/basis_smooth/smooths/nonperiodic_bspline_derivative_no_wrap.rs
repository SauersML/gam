use gam::terms::basis::{BasisOptions, Dense, KnotSource, create_basis};
use ndarray::Array1;

/// gam#1348: on a uniform OPEN (unclamped) knot vector the first derivative of
/// the B-spline basis must equal a central difference of the basis VALUE
/// everywhere, including the two boundary spans. The bug ran the eval point
/// through a geometric periodic wrap that fired for any uniform open knot
/// vector, moving a boundary-span point forward by one period onto unrelated
/// interior columns (max|d1 - fd| ~ 1.7). This is the Rust-side mirror of the
/// public `gamfit.bspline_basis_derivative(..., periodic=False)` repro.
#[test]
fn nonperiodic_bspline_first_derivative_matches_central_difference() {
    let degree = 3usize;
    // Uniform OPEN knots over [-2, 3]; no repeated boundary knots.
    let knots = Array1::linspace(-2.0, 3.0, 14);
    let n = 121usize;
    let lo = knots[0];
    let hi = knots[knots.len() - 1];
    let tt = Array1::from_iter((0..n).map(|i| lo + (hi - lo) * (i as f64) / ((n - 1) as f64)));

    let h = 1e-6;
    let (v_plus, _) = create_basis::<Dense>(
        tt.mapv(|v| v + h).view(),
        KnotSource::Provided(knots.view()),
        degree,
        BasisOptions::value(),
    )
    .expect("value basis at +h");
    let (v_minus, _) = create_basis::<Dense>(
        tt.mapv(|v| v - h).view(),
        KnotSource::Provided(knots.view()),
        degree,
        BasisOptions::value(),
    )
    .expect("value basis at -h");
    let (d1, _) = create_basis::<Dense>(
        tt.view(),
        KnotSource::Provided(knots.view()),
        degree,
        BasisOptions::first_derivative(),
    )
    .expect("first derivative");

    // Support check at a point in the FIRST knot span: the derivative must be
    // supported on a subset of the value's columns (no forward periodic wrap).
    let t_span = Array1::from_elem(1, knots[1] - 0.25 * (knots[1] - knots[0]));
    let (val_span, _) = create_basis::<Dense>(
        t_span.view(),
        KnotSource::Provided(knots.view()),
        degree,
        BasisOptions::value(),
    )
    .expect("value at first span");
    let (der_span, _) = create_basis::<Dense>(
        t_span.view(),
        KnotSource::Provided(knots.view()),
        degree,
        BasisOptions::first_derivative(),
    )
    .expect("derivative at first span");
    for j in 0..der_span.ncols() {
        if der_span[[0, j]].abs() > 1e-9 {
            assert!(
                val_span[[0, j]].abs() > 1e-9,
                "derivative wrapped: nonzero at col {j} where the value basis is zero"
            );
        }
    }

    let mut max_abs = 0.0_f64;
    let mut any_nonzero = false;
    for i in 0..tt.len() {
        for j in 0..d1.ncols() {
            let fd = (v_plus[[i, j]] - v_minus[[i, j]]) / (2.0 * h);
            max_abs = max_abs.max((d1[[i, j]] - fd).abs());
            if d1[[i, j]].abs() > 1e-9 {
                any_nonzero = true;
            }
        }
    }
    assert!(
        any_nonzero,
        "degenerate: open-knot first derivative is identically zero"
    );
    assert!(
        max_abs < 1e-5,
        "open-knot B-spline 1st derivative disagrees with central diff of value \
         (max|d1 - fd| = {max_abs}); a periodic wrap is corrupting boundary spans"
    );
}
