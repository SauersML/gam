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

    // Keep both `x` and `x + period` inside the degree-padded knot support
    // `[-degree*h, (num_basis + degree)*h]` so the raw open-basis derivative is a
    // genuine translate of itself across the period (the property the modular
    // fold below relies on). With max support `(num_basis + degree)*h`, `x +
    // period` stays in support iff `x < degree * h` = 0.375 here.
    let x = array![0.123456789, 0.311111111];
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

    // The extended (open-knot) basis has `total_knots - degree - 1` columns,
    // which for a cyclic layout is `num_basis + degree` — NOT `2 * num_basis`.
    // Fold exactly as the production cyclic evaluator does
    // (`cyclic[j % num_basis] += extended[j]`, see
    // `evaluate_bspline_basis_chunk` in src/terms/basis/bspline_eval.rs), then
    // compare the folded derivative rows at `x` and `x + period`. Periodicity of
    // the folded first derivative follows from translation invariance of the
    // uniform B-spline derivative basis under the modular wrap.
    let ncols = b0.ncols();
    assert_eq!(ncols, num_basis + degree, "extended basis column count");
    for i in 0..b0.nrows() {
        let mut folded0 = vec![0.0_f64; num_basis];
        let mut folded1 = vec![0.0_f64; num_basis];
        for j in 0..ncols {
            folded0[j % num_basis] += b0[[i, j]];
            folded1[j % num_basis] += b1[[i, j]];
        }
        for j in 0..num_basis {
            assert!(
                (folded0[j] - folded1[j]).abs() < 1e-12,
                "bug: cyclic B-spline first derivative is not periodic after fold-back"
            );
        }
    }
}
