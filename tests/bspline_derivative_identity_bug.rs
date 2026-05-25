use gam::terms::basis::{
    SplineScratch, evaluate_bspline_basis_scalar, evaluate_bspline_derivative_scalar,
};
use ndarray::array;

#[test]
fn bug_first_derivative_matches_analytic_difference_identity_including_multiplicity_points() {
    let knots = array![0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.5, 2.0, 2.0, 2.0];
    let degree = 2usize;
    let n = knots.len() - degree - 1;
    let xs = [0.25, 0.5, 0.75, 1.25];

    for &x in &xs {
        let mut b_km1 = vec![0.0; knots.len() - (degree - 1) - 1];
        let mut db = vec![0.0; n];
        evaluate_bspline_basis_scalar(
            x,
            knots.view(),
            degree - 1,
            &mut b_km1,
            &mut SplineScratch::new(degree - 1),
        )
        .expect("B_{k-1}");
        evaluate_bspline_derivative_scalar(x, knots.view(), degree, &mut db).expect("dB");

        for i in 0..n {
            let term1 = if knots[i + degree] > knots[i] {
                degree as f64 / (knots[i + degree] - knots[i]) * b_km1[i]
            } else {
                0.0
            };
            let term2 = if i + 1 < b_km1.len() && knots[i + degree + 1] > knots[i + 1] {
                degree as f64 / (knots[i + degree + 1] - knots[i + 1]) * b_km1[i + 1]
            } else {
                0.0
            };
            let rhs = term1 - term2;
            assert!(
                (db[i] - rhs).abs() < 2e-11,
                "identity mismatch at x={x}, i={i}: dB={} rhs={rhs}",
                db[i]
            );
        }
    }
}
