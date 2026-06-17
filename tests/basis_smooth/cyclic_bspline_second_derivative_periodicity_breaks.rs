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

    // The extended (open-knot) basis has `total_knots - degree - 1` columns,
    // which for a cyclic layout is `num_basis + degree` — NOT `2 * num_basis`.
    // Fold exactly as the production cyclic evaluator does
    // (`cyclic[j % num_basis] += extended[j]`, see
    // `evaluate_bspline_basis_chunk` in src/terms/basis/bspline_eval.rs), then
    // compare the folded second-derivative rows at `x` and `x + period`.
    let ncols = b0.ncols();
    assert_eq!(ncols, num_basis + degree, "extended basis column count");
    let mut folded0 = vec![0.0_f64; num_basis];
    let mut folded1 = vec![0.0_f64; num_basis];
    for j in 0..ncols {
        folded0[j % num_basis] += b0[[0, j]];
        folded1[j % num_basis] += b1[[0, j]];
    }
    for j in 0..num_basis {
        assert!(
            (folded0[j] - folded1[j]).abs() < 1e-10,
            "bug: cyclic B-spline second derivative is not periodic after fold-back"
        );
    }
}
