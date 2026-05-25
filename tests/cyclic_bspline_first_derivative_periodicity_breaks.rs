use gam::terms::basis::{BasisOptions, Dense, KnotSource, create_basis};
use ndarray::array;

#[test]
fn cyclic_bspline_first_derivative_periodicity_breaks() {
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

    let x = array![0.123456789, 0.876543211];
    let x_shifted = x.mapv(|v| v + period);

    let (b0, _) = create_basis::<Dense>(
        x.view(),
        KnotSource::Provided(knots.view()),
        degree,
        BasisOptions::first_derivative(),
    )
    .expect("basis should build at x");
    let (b1, _) = create_basis::<Dense>(
        x_shifted.view(),
        KnotSource::Provided(knots.view()),
        degree,
        BasisOptions::first_derivative(),
    )
    .expect("basis should build at x+period");

    for i in 0..b0.nrows() {
        for j in 0..num_basis {
            let folded0 = b0[[i, j]] + b0[[i, j + num_basis]];
            let folded1 = b1[[i, j]] + b1[[i, j + num_basis]];
            assert!(
                (folded0 - folded1).abs() < 1e-12,
                "bug: cyclic B-spline first derivative is not periodic after fold-back"
            );
        }
    }
}
