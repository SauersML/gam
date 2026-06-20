//! Regression for gam#1348 on the *per-point* sparse, scalar, and recurrence
//! B-spline derivative evaluators — the paths the dense-only exterior fix left
//! behind.
//!
//! The original #1348 fix zeroed the open-knot exterior in
//! `apply_dense_bspline_extrapolation` (the dense matrix post-processor) and the
//! recurrence/scalar clamp. But on an *open* (unclamped / Eilers–Marx P-spline)
//! knot vector with points outside the modeling interval:
//!
//! * `SparseStorage::build` does NOT take the clamped-extrapolation fallback
//!   (it is gated on `has_clamped_bspline_boundaries`), so the per-point sparse
//!   evaluator was used directly and returned the raw, nonzero mathematical
//!   derivative `B'(x)` in the boundary spans — disagreeing with the (now zero)
//!   dense path and with the constant-extended value basis; and
//! * the scalar `evaluate_bspline_derivative_scalar*` path clamped the eval point
//!   and returned the nonzero *boundary slope* there, which is the right answer
//!   for a clamped (linearly extended) basis but wrong for an open
//!   (constant-extended) one.
//!
//! The boundary treatment must follow the value basis per knot geometry:
//! open ⇒ constant extension ⇒ zero exterior derivative; clamped ⇒ linear
//! extension ⇒ nonzero boundary-slope exterior derivative. These tests pin both,
//! across the sparse, dense, and scalar paths.

use gam::terms::basis::{
    BasisOptions, Dense, KnotSource, Sparse, SplineScratch, create_basis,
    evaluate_bspline_basis_scalar, evaluate_bspline_derivative_scalar,
    evaluate_bsplinesecond_derivative_scalar,
};
use ndarray::{Array1, Array2};

fn open_knots(a: f64, b: f64, m: usize) -> Array1<f64> {
    Array1::from_iter((0..m).map(|i| a + (b - a) * i as f64 / (m as f64 - 1.0)))
}

fn clamped_knots(degree: usize, internal: &[f64], span: f64) -> Array1<f64> {
    let mut k = vec![0.0; degree + 1];
    k.extend_from_slice(internal);
    k.extend(std::iter::repeat(span).take(degree + 1));
    Array1::from(k)
}

fn dense_derivative(
    t: &Array1<f64>,
    knots: &Array1<f64>,
    degree: usize,
    order: usize,
) -> Array2<f64> {
    let options = if order == 1 {
        BasisOptions::first_derivative()
    } else {
        BasisOptions::second_derivative()
    };
    let (b, _) = create_basis::<Dense>(
        t.view(),
        KnotSource::Provided(knots.view()),
        degree,
        options,
    )
    .expect("dense derivative");
    (*b).clone()
}

fn sparse_derivative_dense(
    t: &Array1<f64>,
    knots: &Array1<f64>,
    degree: usize,
    order: usize,
) -> Array2<f64> {
    let options = if order == 1 {
        BasisOptions::first_derivative()
    } else {
        BasisOptions::second_derivative()
    };
    let (sparse, _) = create_basis::<Sparse>(
        t.view(),
        KnotSource::Provided(knots.view()),
        degree,
        options,
    )
    .expect("sparse derivative");
    let mut dense = Array2::<f64>::zeros((sparse.nrows(), sparse.ncols()));
    let (symbolic, values) = sparse.parts();
    let col_ptr = symbolic.col_ptr();
    let row_idx = symbolic.row_idx();
    for col in 0..sparse.ncols() {
        for idx in col_ptr[col]..col_ptr[col + 1] {
            dense[[row_idx[idx], col]] += values[idx];
        }
    }
    dense
}

/// The sparse derivative design must equal the dense one bit-for-bit on an open
/// knot vector — including the exterior boundary spans, where both must be zero.
/// Before the per-point fix the sparse path left the raw nonzero derivative
/// there while the dense path zeroed it.
#[test]
fn sparse_open_knot_derivative_matches_dense_including_exterior() {
    for degree in [2usize, 3, 4] {
        let knots = open_knots(-2.0, 3.0, degree + 9);
        let num_basis = knots.len() - degree - 1;
        let left = knots[degree];
        let right = knots[num_basis];

        // Points across the FULL knot range: interior + both exterior spans.
        let t = Array1::from_iter(
            (0..97).map(|i| knots[0] + (knots[knots.len() - 1] - knots[0]) * i as f64 / 96.0),
        );

        for order in [1usize, 2] {
            let dense = dense_derivative(&t, &knots, degree, order);
            let sparse = sparse_derivative_dense(&t, &knots, degree, order);
            assert_eq!(dense.dim(), sparse.dim());
            let mut max_diff = 0.0_f64;
            let mut exterior_checked = false;
            for i in 0..dense.nrows() {
                let x = t[i];
                for j in 0..dense.ncols() {
                    max_diff = max_diff.max((dense[[i, j]] - sparse[[i, j]]).abs());
                }
                if x < left || x > right {
                    exterior_checked = true;
                    // Both paths must be exactly zero in the exterior.
                    for j in 0..sparse.ncols() {
                        assert_eq!(
                            sparse[[i, j]],
                            0.0,
                            "degree={degree} order={order}: sparse derivative nonzero at exterior x={x}, col {j}"
                        );
                    }
                }
            }
            assert!(
                exterior_checked,
                "degree={degree}: no exterior sample points"
            );
            assert!(
                max_diff < 1e-12,
                "degree={degree} order={order}: sparse and dense open-knot derivative disagree by {max_diff}"
            );
        }
    }
}

/// The public scalar derivative evaluators (the marginal-slope / boundary-pin
/// path) must zero the open-knot exterior and match a value finite difference in
/// the interior, for orders 1 and 2 (order 2 exercises the recurrence).
#[test]
fn scalar_open_knot_derivative_zero_exterior_fd_interior() {
    let degree = 3usize;
    let knots = open_knots(-1.0, 2.0, 15);
    let num_basis = knots.len() - degree - 1;
    let left = knots[degree];
    let right = knots[num_basis];

    let value_col = |x: f64, i: usize| -> f64 {
        let mut out = vec![0.0; num_basis];
        let mut scratch = SplineScratch::new(degree);
        evaluate_bspline_basis_scalar(x, knots.view(), degree, &mut out, &mut scratch)
            .expect("value");
        out[i]
    };

    let mut d1 = vec![0.0; num_basis];
    let mut d2 = vec![0.0; num_basis];

    // Exterior: every order is exactly zero.
    for &x in &[
        knots[0],
        knots[1],
        0.5 * (knots[0] + left),
        right + 0.3,
        knots[knots.len() - 1],
    ] {
        if x >= left && x <= right {
            continue;
        }
        evaluate_bspline_derivative_scalar(x, knots.view(), degree, &mut d1).expect("d1");
        evaluate_bsplinesecond_derivative_scalar(x, knots.view(), degree, &mut d2).expect("d2");
        assert!(
            d1.iter().all(|&v| v == 0.0),
            "scalar d1 nonzero at exterior x={x}: {d1:?}"
        );
        assert!(
            d2.iter().all(|&v| v == 0.0),
            "scalar d2 nonzero at exterior x={x}: {d2:?}"
        );
    }

    // Interior: first derivative matches the value's central difference.
    let h = 1e-6;
    let mut any_nonzero = false;
    for &x in &[left + 0.4, 0.5 * (left + right), right - 0.4, 0.137, 1.21] {
        if x <= left || x >= right {
            continue;
        }
        evaluate_bspline_derivative_scalar(x, knots.view(), degree, &mut d1).expect("d1");
        for i in 0..num_basis {
            let fd = (value_col(x + h, i) - value_col(x - h, i)) / (2.0 * h);
            assert!(
                (d1[i] - fd).abs() < 1e-6,
                "scalar interior d1 mismatch at x={x}, i={i}: {} vs {fd}",
                d1[i]
            );
            any_nonzero |= d1[i].abs() > 1e-6;
        }
    }
    assert!(any_nonzero, "scalar interior derivative degenerately zero");
}

/// Guard the *clamped* geometry: a clamped knot vector extends the value
/// linearly, so its exterior derivative must be the NONZERO boundary slope —
/// the open-knot zeroing must not bleed into the clamped path. The exterior
/// derivative equals the derivative evaluated at the clamped boundary (constant
/// continuation).
#[test]
fn clamped_knot_exterior_derivative_is_boundary_slope_not_zero() {
    let degree = 3usize;
    let knots = clamped_knots(degree, &[0.4, 0.9, 1.5], 2.0);
    let num_basis = knots.len() - degree - 1;
    let right = knots[num_basis];
    let left = knots[degree];

    let mut at_boundary = vec![0.0; num_basis];
    let mut at_exterior = vec![0.0; num_basis];

    // Right boundary slope: derivative just inside vs. an exterior point both
    // resolve to the boundary slope under linear extension.
    evaluate_bspline_derivative_scalar(right, knots.view(), degree, &mut at_boundary)
        .expect("d@right");
    evaluate_bspline_derivative_scalar(right + 0.7, knots.view(), degree, &mut at_exterior)
        .expect("d@exterior");

    let boundary_norm: f64 = at_boundary.iter().map(|v| v.abs()).sum();
    assert!(
        boundary_norm > 1e-6,
        "clamped boundary slope should be nonzero; got {at_boundary:?}"
    );
    for i in 0..num_basis {
        assert!(
            (at_boundary[i] - at_exterior[i]).abs() < 1e-12,
            "clamped exterior derivative must equal the boundary slope (linear extension); \
             col {i}: {} vs {}",
            at_boundary[i],
            at_exterior[i]
        );
    }

    // Same on the left side.
    evaluate_bspline_derivative_scalar(left, knots.view(), degree, &mut at_boundary)
        .expect("d@left");
    evaluate_bspline_derivative_scalar(left - 0.7, knots.view(), degree, &mut at_exterior)
        .expect("d@left-exterior");
    for i in 0..num_basis {
        assert!(
            (at_boundary[i] - at_exterior[i]).abs() < 1e-12,
            "clamped left exterior derivative must equal the boundary slope; col {i}"
        );
    }
}
