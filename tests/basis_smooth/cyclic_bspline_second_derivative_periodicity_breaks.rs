use gam::terms::basis::{BasisOptions, Dense, KnotSource, create_basis};
use ndarray::Array1;

/// Regression for gam#1348: on a UNIFORM OPEN (unclamped) knot vector the
/// analytic second derivative of the B-spline basis must equal a central
/// difference of the basis VALUE everywhere — including the boundary spans.
///
/// Previously the open-knot derivative path ran the eval point through a
/// geometric periodic wrap (`periodic_unclamped_derivative_eval_point`) that
/// fired for any uniform open knot vector, moving boundary-span points onto
/// unrelated interior columns. That broke value/derivative agreement (and was
/// only ever needed by a hand-rolled open-knot "cyclic fold-back" test, never by
/// a production periodic path — the real periodic derivative has its own closed
/// form). The wrap is gone; the open second derivative must now match FD.
#[test]
fn cyclic_bspline_second_derivative_periodicity_breaks() {
    let degree = 3usize;
    let start = 0.0_f64;
    let end = 1.0_f64;
    let num_basis = 8usize;
    let h_knot = (end - start) / num_basis as f64;
    let total_knots = num_basis + 2 * degree + 1;
    let knots =
        Array1::from_iter((0..total_knots).map(|i| start + (i as f64 - degree as f64) * h_knot));

    // Interior modeling domain of the open basis: [knots[degree], knots[num_basis]].
    let left = knots[degree];
    let right = knots[num_basis];
    let n = 121usize;
    let tt = Array1::from_iter(
        (0..n).map(|i| left + (right - left) * (i as f64) / ((n - 1) as f64)),
    );

    let fd_h = 1e-5;
    let tt_plus = tt.mapv(|v| v + fd_h);
    let tt_minus = tt.mapv(|v| v - fd_h);

    let (v_plus, _) = create_basis::<Dense>(
        tt_plus.view(),
        KnotSource::Provided(knots.view()),
        degree,
        BasisOptions::first_derivative(),
    )
    .expect("first derivative at +h");
    let (v_minus, _) = create_basis::<Dense>(
        tt_minus.view(),
        KnotSource::Provided(knots.view()),
        degree,
        BasisOptions::first_derivative(),
    )
    .expect("first derivative at -h");
    let (d2, _) = create_basis::<Dense>(
        tt.view(),
        KnotSource::Provided(knots.view()),
        degree,
        BasisOptions::second_derivative(),
    )
    .expect("second derivative");

    assert_eq!(d2.ncols(), num_basis + degree, "extended basis column count");

    let mut max_abs = 0.0_f64;
    let mut any_nonzero = false;
    for i in 0..tt.len() {
        for j in 0..d2.ncols() {
            let fd = (v_plus[[i, j]] - v_minus[[i, j]]) / (2.0 * fd_h);
            let diff = (d2[[i, j]] - fd).abs();
            if diff > max_abs {
                max_abs = diff;
            }
            if d2[[i, j]].abs() > 1e-9 {
                any_nonzero = true;
            }
        }
    }
    assert!(
        any_nonzero,
        "degenerate fix: open-knot second derivative is identically zero"
    );
    assert!(
        max_abs < 1e-4,
        "open-knot B-spline 2nd derivative disagrees with central diff of 1st \
         derivative (max|d2 - fd| = {max_abs}); a periodic wrap is corrupting the \
         non-periodic boundary spans"
    );
}
