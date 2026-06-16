//! Finite-difference oracle for the raw B-spline derivative recurrences.
//!
//! The sibling `bspline_derivative_identity_bug.rs` only checks the *first*
//! derivative against the de-Boor recurrence written out by hand — a circular
//! check (recurrence vs. recurrence) that cannot catch a sign/index/order bug
//! in the recurrence itself, and never exercises orders 2/3/4.
//!
//! This test instead pins every analytic derivative order (1..=4) against a
//! central finite-difference stencil of the *value* evaluator
//! (`evaluate_bspline_basis_scalar`), which shares no code with the derivative
//! path. A bug in `evaluate_bspline_derivative_recurrence_into` (the body that
//! powers orders 2/3/4) or in `evaluate_bspline_derivative_scalar` (order 1)
//! shows up as an FD mismatch.
//!
//! The basis is evaluated column-by-column: each column `i` is one scalar
//! function `B_{i,k}(x)`, so we can FD it independently and compare to row `i`
//! of the analytic derivative vector.

use gam::terms::basis::{
    SplineScratch, evaluate_bspline_basis_scalar, evaluate_bspline_derivative_scalar,
    evaluate_bsplinesecond_derivative_scalar, evaluate_bsplinethird_derivative_scalar,
    evaluate_bspline_fourth_derivative_scalar,
};
use ndarray::{Array1, ArrayView1};

/// Value of basis column `i` at scalar `x`.
fn value_col(x: f64, knots: ArrayView1<f64>, degree: usize, i: usize) -> f64 {
    let n = knots.len() - degree - 1;
    let mut out = vec![0.0; n];
    let mut scratch = SplineScratch::new(degree);
    evaluate_bspline_basis_scalar(x, knots, degree, &mut out, &mut scratch).expect("value eval");
    out[i]
}

/// Central finite-difference of derivative `order` of basis column `i` at `x`.
///
/// Uses the standard central stencils:
///   f'   ≈ [f(x+h) − f(x−h)] / (2h)
///   f''  ≈ [f(x+h) − 2f(x) + f(x−h)] / h²
///   f''' ≈ [f(x+2h) − 2f(x+h) + 2f(x−h) − f(x−2h)] / (2h³)
///   f⁴   ≈ [f(x+2h) − 4f(x+h) + 6f(x) − 4f(x−h) + f(x−2h)] / h⁴
fn fd_col(x: f64, knots: ArrayView1<f64>, degree: usize, i: usize, order: usize, h: f64) -> f64 {
    let v = |dx: f64| value_col(x + dx, knots, degree, i);
    match order {
        1 => (v(h) - v(-h)) / (2.0 * h),
        2 => (v(h) - 2.0 * v(0.0) + v(-h)) / (h * h),
        3 => (v(2.0 * h) - 2.0 * v(h) + 2.0 * v(-h) - v(-2.0 * h)) / (2.0 * h * h * h),
        4 => {
            (v(2.0 * h) - 4.0 * v(h) + 6.0 * v(0.0) - 4.0 * v(-h) + v(-2.0 * h)) / (h * h * h * h)
        }
        _ => unreachable!("order out of range"),
    }
}

fn analytic(x: f64, knots: ArrayView1<f64>, degree: usize, order: usize) -> Array1<f64> {
    let n = knots.len() - degree - 1;
    let mut out = vec![0.0; n];
    match order {
        1 => evaluate_bspline_derivative_scalar(x, knots, degree, &mut out).expect("d1"),
        2 => evaluate_bsplinesecond_derivative_scalar(x, knots, degree, &mut out).expect("d2"),
        3 => evaluate_bsplinethird_derivative_scalar(x, knots, degree, &mut out).expect("d3"),
        4 => evaluate_bspline_fourth_derivative_scalar(x, knots, degree, &mut out).expect("d4"),
        _ => unreachable!(),
    }
    Array1::from(out)
}

/// A clamped, non-uniform knot vector of the given `degree` over `[0, span]`
/// with `n_internal` interior knots. Clamped ends (degree+1 repeats) make the
/// basis a partition of unity on the full interior so each column is a smooth
/// piecewise polynomial we can finite-difference.
fn clamped_knots(degree: usize, internal: &[f64], span: f64) -> Array1<f64> {
    let mut k = Vec::new();
    for _ in 0..=degree {
        k.push(0.0);
    }
    k.extend_from_slice(internal);
    for _ in 0..=degree {
        k.push(span);
    }
    Array1::from(k)
}

#[test]
fn bspline_derivatives_1_through_4_match_central_finite_differences() {
    // Non-uniform interior knots (unequal spacing) to exercise the
    // (t_{i+k}-t_i) denominators that a sign/index bug would corrupt.
    let cases: &[(usize, Vec<f64>)] = &[
        (1, vec![0.3, 0.55, 0.8]),
        (2, vec![0.2, 0.5, 0.9, 1.3]),
        (3, vec![0.25, 0.6, 1.1, 1.6]),
        (4, vec![0.4, 0.9, 1.5]),
    ];
    let span = 2.0;

    for (degree, internal) in cases {
        let degree = *degree;
        let knots = clamped_knots(degree, internal, span);
        let n = knots.len() - degree - 1;

        // Interior sample points away from knots (where high derivatives jump):
        // pick midpoints of polynomial pieces and a few generic points, staying
        // a safe margin from the boundary so the central stencil stays inside
        // the support and away from the clamped ends.
        let xs = [0.13, 0.37, 0.62, 0.88, 1.05, 1.27, 1.44, 1.71, 1.86];

        for order in 1..=degree.min(4) {
            // Step size: small enough for accuracy, large enough that the
            // O(h²) truncation error of the stencils dominates round-off. The
            // tolerance scales with the stencil's truncation order and the
            // magnitude of the high derivative.
            let h = 1e-3;
            for &x in &xs {
                // Skip points too close to an interior knot for this order: the
                // (order)-th derivative is discontinuous across a knot of the
                // relevant multiplicity, so a central stencil straddling it is
                // not a valid oracle. We require the whole stencil width
                // (±2h for orders 3/4) to lie strictly within one knot span.
                let half_width = 2.0 * h;
                let near_knot = knots
                    .iter()
                    .any(|&t| (x - t).abs() <= half_width + 1e-9 && t > 0.0 && t < span);
                if near_knot {
                    continue;
                }

                let an = analytic(x, knots.view(), degree, order);
                for i in 0..n {
                    let fd = fd_col(x, knots.view(), degree, i, order, h);
                    let diff = (an[i] - fd).abs();
                    // Truncation error of an O(h²) central stencil is
                    // ~ C·h²·|f^{(order+2)}|. With h=1e-3 that is ~1e-6 times a
                    // bounded high derivative; allow a generous absolute floor
                    // plus a relative term on the analytic magnitude.
                    let tol = 1e-4 + 1e-3 * an[i].abs();
                    assert!(
                        diff <= tol,
                        "degree={degree} order={order} i={i} x={x}: \
                         analytic={} fd={} |diff|={diff} tol={tol}",
                        an[i],
                        fd,
                    );
                }
            }
        }
    }
}

#[test]
fn bspline_derivative_partition_of_unity_sums_to_zero() {
    // The B-spline columns sum to 1 everywhere on the clamped interior, so
    // every derivative order must sum to exactly 0 across columns. This is an
    // exact (round-off-only) algebraic invariant that catches a per-column
    // sign/scale bug the FD test could only catch statistically.
    let degree = 3usize;
    let knots = clamped_knots(degree, &[0.4, 0.9, 1.5], 2.0);
    let n = knots.len() - degree - 1;
    let xs = [0.2, 0.55, 0.7, 1.1, 1.35, 1.8];

    for &x in &xs {
        for order in 1..=degree {
            let an = analytic(x, knots.view(), degree, order);
            assert_eq!(an.len(), n);
            let s: f64 = an.iter().sum();
            assert!(
                s.abs() < 1e-9,
                "sum of order-{order} derivatives at x={x} must be 0 (partition of unity); got {s}"
            );
        }
    }
}
