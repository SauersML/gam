use gam::terms::basis::{BasisOptions, Dense, KnotSource, create_basis};
use ndarray::array;

#[test]
fn cyclic_bspline_second_derivative_periodicity_breaks() {
    let degree = 3usize;
    let start = 0.0;
    let end = 1.0;
    let num_basis = 8usize;
    let period = end - start;
    let h = period / num_basis as f64;
    let total_knots = num_basis + 2 * degree + 1;
    let knots = ndarray::Array1::from_iter(
        (0..total_knots).map(|i| start + (i as f64 - degree as f64) * h),
    );

    let x = array![0.22222];
    let x_shifted = x.mapv(|v| v + period);

    let (b0, _) = create_basis::<Dense>(
        x.view(),
        KnotSource::Provided(knots.view()),
        degree,
        BasisOptions::second_derivative(),
    )
    .expect("basis should build at x");
    let (b1, _) = create_basis::<Dense>(
        x_shifted.view(),
        KnotSource::Provided(knots.view()),
        degree,
        BasisOptions::second_derivative(),
    )
    .expect("basis should build at x+period");

    for j in 0..num_basis {
        let folded0 = b0[[0, j]] + b0[[0, j + num_basis]];
        let folded1 = b1[[0, j]] + b1[[0, j + num_basis]];
        assert!(
            (folded0 - folded1).abs() < 1e-10,
            "bug: cyclic B-spline second derivative is not periodic after fold-back"
        );
    }
}
