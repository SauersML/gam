use gam::terms::basis::{
    SplineScratch, evaluate_bspline_basis_scalar, evaluate_ispline_scalar, evaluate_mspline_scalar,
};
use ndarray::array;

#[test]
fn bug_mspline_and_ispline_scalar_evaluators_match_scaled_bspline_and_integral_identity() {
    let knots = array![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0];
    let degree = 2usize;
    let n_b = knots.len() - degree - 1;
    let n_i = knots.len() - (degree + 1) - 2;
    let x = 1.3;

    let mut b = vec![0.0; n_b];
    let mut m = vec![0.0; n_b];
    evaluate_bspline_basis_scalar(
        x,
        knots.view(),
        degree,
        &mut b,
        &mut SplineScratch::new(degree),
    )
    .unwrap();
    evaluate_mspline_scalar(
        x,
        knots.view(),
        degree,
        &mut m,
        &mut SplineScratch::new(degree),
    )
    .unwrap();
    for i in 0..n_b {
        let span = knots[i + degree + 1] - knots[i];
        let expected = if span > 0.0 {
            ((degree + 1) as f64 / span) * b[i]
        } else {
            0.0
        };
        assert!(
            (m[i] - expected).abs() < 1e-12,
            "M-spline scale mismatch at i={i}"
        );
    }

    let mut i_left = vec![0.0; n_i];
    let mut i_x = vec![0.0; n_i];
    evaluate_ispline_scalar(knots[degree + 1], knots.view(), degree, &mut i_left).unwrap();
    evaluate_ispline_scalar(x, knots.view(), degree, &mut i_x).unwrap();
    for j in 0..n_i {
        assert!(
            i_left[j].abs() < 1e-13,
            "I-spline should anchor to zero at left boundary j={j}"
        );
        assert!(
            i_x[j] >= -1e-13,
            "I-spline should be nonnegative at x j={j}"
        );
    }
}
