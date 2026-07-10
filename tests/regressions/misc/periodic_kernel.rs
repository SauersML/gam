//! RED test for issue #225 — supporting Rust-kernel coverage.
//!
//! The pyffi binding `periodic_spline_curve_basis` is missing (see
//! `tests/test_issue_225_periodic_spline_curve_basis.py`). The underlying
//! Rust kernels `build_periodic_bspline_basis_1d` +
//! `cyclic_bspline_derivative_penalty_matrix` are what the pyfunction
//! must call. These asserts pin the contract the Python wrapper expects:
//! correct shape, seam periodicity, partition of unity, and cyclic
//! exact derivative roughness with constant nullspace.

use gam::terms::basis::{
    PeriodicBSplineBasisSpec, build_periodic_bspline_basis_1d,
    cyclic_bspline_derivative_penalty_matrix,
};
use ndarray::{Array1, Array2};

#[test]
fn issue_225_periodic_basis_shape_partition_and_seam() {
    let n_knots = 12usize;
    let degree = 3usize;
    let penalty_order = 2usize;
    let spec = PeriodicBSplineBasisSpec::new(degree, n_knots, 1.0, 0.0, penalty_order);

    let t = Array1::from(vec![0.0, 0.07, 0.25, 0.5, 0.999_999, 1.0, 1.07, -0.93]);
    let basis = build_periodic_bspline_basis_1d(t.view(), &spec).expect("periodic basis");
    assert_eq!(basis.shape(), &[t.len(), n_knots]);

    for row in basis.rows() {
        let s: f64 = row.iter().sum();
        assert!((s - 1.0).abs() < 1e-12, "partition of unity violated: {s}");
        assert!(row.iter().all(|v| *v >= -1e-14));
    }
    for j in 0..n_knots {
        assert!(
            (basis[[0, j]] - basis[[5, j]]).abs() < 1e-12,
            "seam mismatch at col {j} between t=0 and t=1"
        );
    }
}

#[test]
fn issue_225_cyclic_penalty_rank_and_nullspace() {
    let k = 12usize;
    let s = cyclic_bspline_derivative_penalty_matrix(3, k, 1.0, 2)
        .expect("cyclic derivative penalty");
    assert_eq!(s.shape(), &[k, k]);
    let ones = Array2::from_elem((k, 1), 1.0);
    let p = s.dot(&ones);
    assert!(
        p.iter().all(|v| v.abs() < 1e-9),
        "constant vector must lie in nullspace"
    );
    for i in 0..k {
        for j in 0..k {
            assert!((s[[i, j]] - s[[j, i]]).abs() < 1e-12, "symmetry");
        }
    }
}
