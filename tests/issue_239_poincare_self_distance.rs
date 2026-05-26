//! RED tests for issue #239: `poincare_distance` must return exactly 0
//! for identical points. Currently `src/geometry/poincare.rs:172` clamps
//! the arccosh argument to `1.0 + ORIGIN_EPS`, producing
//! `acosh(1 + 1e-15) ≈ 4.47e-8` for self-distance, even though
//! `acosh(1.0) == 0.0` exactly in IEEE 754.
//!
//! These tests assert the principled contract (exact zero, symmetry,
//! triangle inequality, near-origin scale) and will fail until the
//! +ORIGIN_EPS clamp is removed (or self-distance is special-cased).

use gam::geometry::poincare::poincare_distance;
use ndarray::{Array1, array};

#[test]
fn self_distance_is_exact_zero_curvature_minus_one() {
    let a = array![0.1_f64, -0.2, 0.05];
    let d = poincare_distance(a.view(), a.view(), -1.0).expect("distance");
    assert_eq!(d, 0.0, "self-distance must be exactly 0.0, got {d:e}");
}

#[test]
fn self_distance_is_exact_zero_at_origin() {
    let a: Array1<f64> = Array1::zeros(4);
    let d = poincare_distance(a.view(), a.view(), -1.0).expect("distance");
    assert_eq!(d, 0.0, "self-distance at origin must be 0.0, got {d:e}");
}

#[test]
fn self_distance_is_exact_zero_high_curvature() {
    let a = array![0.05_f64, 0.05, 0.05];
    let d = poincare_distance(a.view(), a.view(), -8.0).expect("distance");
    assert_eq!(d, 0.0, "self-distance under c=-8 must be 0.0, got {d:e}");
}

#[test]
fn distance_is_symmetric_to_high_precision() {
    let a = array![0.3_f64, 0.1];
    let b = array![-0.2_f64, 0.4];
    let dab = poincare_distance(a.view(), b.view(), -1.0).expect("d(a,b)");
    let dba = poincare_distance(b.view(), a.view(), -1.0).expect("d(b,a)");
    assert!(
        (dab - dba).abs() < 1.0e-15,
        "symmetry violated: d(a,b)={dab}, d(b,a)={dba}"
    );
}

#[test]
fn distance_satisfies_triangle_inequality() {
    let a = array![0.1_f64, 0.2];
    let b = array![-0.15_f64, 0.05];
    let c = array![0.25_f64, -0.1];
    let dab = poincare_distance(a.view(), b.view(), -1.0).expect("d(a,b)");
    let dbc = poincare_distance(b.view(), c.view(), -1.0).expect("d(b,c)");
    let dac = poincare_distance(a.view(), c.view(), -1.0).expect("d(a,c)");
    assert!(
        dac <= dab + dbc + 1.0e-12,
        "triangle inequality violated: d(a,c)={dac} > d(a,b)+d(b,c)={}",
        dab + dbc
    );
}

#[test]
fn near_origin_distance_approaches_euclidean_scale() {
    // For small a, b near origin under c=-1, d(a,b) ≈ 2 * ||a - b||.
    // The current ORIGIN_EPS clamp adds ~4.5e-8 noise floor that is
    // larger than the true distance for very small separations.
    let a = array![1.0e-6_f64, 0.0];
    let b = array![2.0e-6_f64, 0.0];
    let d = poincare_distance(a.view(), b.view(), -1.0).expect("distance");
    let expected = 2.0e-6_f64; // 2 * ||a - b|| to leading order
    let rel = (d - expected).abs() / expected;
    assert!(
        rel < 1.0e-3,
        "near-origin distance {d} differs from expected ~{expected} by rel {rel}"
    );
}
