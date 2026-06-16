use gam::terms::basis::{
    PeriodicBSplineBasisSpec, build_periodic_bspline_basis_1d, periodic_bspline_first_derivative_nd,
};
use ndarray::{Axis, array};

#[test]
fn bug_periodic_bspline_boundary_value_and_derivative_wrap_continuity() {
    let spec = PeriodicBSplineBasisSpec::new(3, 16, std::f64::consts::TAU, -1.2, 2);
    let u = array![
        -1.2,
        -1.2 + std::f64::consts::TAU,
        -1.2 - 1e-10,
        -1.2 + std::f64::consts::TAU + 1e-10
    ];
    let basis = build_periodic_bspline_basis_1d(u.view(), &spec).expect("periodic basis");

    for j in 0..spec.num_basis {
        assert!(
            (basis[[0, j]] - basis[[1, j]]).abs() < 1e-12,
            "value seam mismatch col={j}"
        );
        assert!(
            (basis[[2, j]] - basis[[3, j]]).abs() < 1e-9,
            "near-seam value mismatch col={j}"
        );
    }

    let t = u.clone().insert_axis(Axis(1));
    let d = periodic_bspline_first_derivative_nd(
        t.view(),
        (spec.origin, spec.origin + spec.period),
        spec.degree,
        spec.num_basis,
    )
    .expect("periodic derivative")
    .index_axis(Axis(2), 0)
    .to_owned();

    for j in 0..spec.num_basis {
        assert!(
            (d[[0, j]] - d[[1, j]]).abs() < 1e-10,
            "derivative seam mismatch col={j}"
        );
        assert!(
            (d[[2, j]] - d[[3, j]]).abs() < 1e-7,
            "near-seam derivative mismatch col={j}"
        );
    }
}
