use gam::terms::basis::periodic_bspline_first_derivative_nd;
use ndarray::array;

/// The production cyclic/periodic B-spline first derivative
/// (`periodic_bspline_first_derivative_nd`) must be exactly periodic: evaluating
/// at `x` and at `x + period` must yield the same derivative row.
///
/// This is the real cyclic-derivative path used in production (the periodic
/// closed form that `basis_with_jet` / `PeriodicSplineCurve::evaluate_derivative`
/// consume). It wraps its own input into the base period internally
/// (`wrap_periodic_phase`), so periodicity is a property of the basis itself —
/// it does NOT depend on the open-basis derivative evaluator wrapping its eval
/// point (that geometric wrap incorrectly fired for non-periodic open knots and
/// corrupted boundary spans, gam#1348; it was removed).
#[test]
fn cyclic_bspline_first_derivative_periodicity_breaks() {
    let degree = 3usize;
    let start = 0.0;
    let end = 1.0;
    let num_basis = 8usize;
    let period = end - start;

    // Span the circle, including points in the seam spans where a non-periodic
    // open basis would differ from its period-shifted self.
    for &x in &[0.0, 0.05, 0.123_456_789, 0.311_111_111, 0.5, 0.77, 0.999] {
        let t0 = array![[x]];
        let t1 = array![[x + period]];
        let d0 = periodic_bspline_first_derivative_nd(t0.view(), (start, end), degree, num_basis)
            .expect("periodic derivative at x");
        let d1 = periodic_bspline_first_derivative_nd(t1.view(), (start, end), degree, num_basis)
            .expect("periodic derivative at x+period");
        for j in 0..num_basis {
            assert!(
                (d0[[0, j, 0]] - d1[[0, j, 0]]).abs() < 1e-10,
                "bug: cyclic B-spline first derivative is not periodic at x={x}, col {j}: \
                 {} vs {}",
                d0[[0, j, 0]],
                d1[[0, j, 0]]
            );
        }
    }
}
