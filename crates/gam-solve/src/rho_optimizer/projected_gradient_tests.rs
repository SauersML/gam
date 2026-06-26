//! KKT-convention regression for [`super::projected_gradient_norm`] (#1074).
//!
//! The bound-projected gradient norm is the stationarity residual the outer
//! cost-stall guard consults to decide whether a box-constrained ρ optimum is
//! converged. At an *active* bound the projection must zero only the infeasible
//! KKT-multiplier pull and KEEP the feasible interior descent — otherwise a
//! coordinate that still has a real descent off its bound is reported as
//! stationary and the optimizer rails (the #1074 quakes-trend null-space coords
//! that sat at ρ = upper while the REML cost was strictly lower at an interior
//! ρ ≈ 2). A prior version inverted both branches: it kept the infeasible pull
//! and dropped the feasible descent. These tests pin the correct convention and
//! fail loudly on that inversion.

use super::projected_gradient_norm;
use ndarray::array;

const EPS: f64 = 1e-12;

#[test]
fn no_bounds_is_plain_euclidean_norm() {
    let x = array![0.0, 0.0, 0.0];
    let g = array![3.0, -4.0, 0.0];
    let n = projected_gradient_norm(&x, &g, None);
    assert!((n - 5.0).abs() < EPS, "expected ‖g‖₂ = 5, got {n}");
}

#[test]
fn interior_point_keeps_full_gradient() {
    // x strictly inside (lower, upper): no projection, full norm retained.
    let lower = array![-10.0, -10.0];
    let upper = array![10.0, 10.0];
    let x = array![0.0, 0.0];
    let g = array![3.0, 4.0];
    let n = projected_gradient_norm(&x, &g, Some(&(lower, upper)));
    assert!((n - 5.0).abs() < EPS, "interior must keep ‖g‖, got {n}");
}

#[test]
fn lower_bound_drops_infeasible_multiplier_keeps_feasible_descent() {
    let lower = array![0.0, 0.0];
    let upper = array![10.0, 10.0];
    let x = array![0.0, 0.0]; // both sitting on the lower bound

    // g_i > 0 at the lower bound: the descent step -g_i exits the box → it is a
    // KKT multiplier and must be ZEROED. min(f(x)=x) on [0,10] is at x=0, g=+1:
    // a genuine constrained optimum ⇒ projected norm 0.
    let g_multiplier = array![1.0, 2.0];
    let n0 = projected_gradient_norm(&x, &g_multiplier, Some(&(lower.clone(), upper.clone())));
    assert!(
        n0 < EPS,
        "g>0 at lower bound is an infeasible multiplier and must project to 0; got {n0}"
    );

    // g_i < 0 at the lower bound: -g_i moves up, into the box → a feasible
    // descent that must be RETAINED. The inverted code zeroed this (max(g,0)=0)
    // and would falsely report stationarity.
    let g_descent = array![-3.0, -4.0];
    let n1 = projected_gradient_norm(&x, &g_descent, Some(&(lower, upper)));
    assert!(
        (n1 - 5.0).abs() < EPS,
        "g<0 at lower bound is feasible descent and must be retained (‖g‖=5); got {n1}"
    );
}

#[test]
fn upper_bound_drops_infeasible_multiplier_keeps_feasible_descent() {
    let lower = array![-10.0, -10.0];
    let upper = array![0.0, 0.0];
    let x = array![0.0, 0.0]; // both sitting on the upper bound

    // g_i < 0 at the upper bound: descent step -g_i > 0 exits the box → infeasible
    // multiplier, must be ZEROED. min(f(x)=-x) on [-10,0] is at x=0, g=-1.
    let g_multiplier = array![-1.0, -2.0];
    let n0 = projected_gradient_norm(&x, &g_multiplier, Some(&(lower.clone(), upper.clone())));
    assert!(
        n0 < EPS,
        "g<0 at upper bound is an infeasible multiplier and must project to 0; got {n0}"
    );

    // g_i > 0 at the upper bound: -g_i moves down, into the box → feasible descent
    // that must be RETAINED. This is the exact #1074 case: a ρ railed at the
    // upper bound with ∂cost/∂ρ > 0 (cost strictly lower at smaller ρ). The
    // inverted code zeroed this (min(g,0)=0) ⇒ the guard certified the rail as
    // converged.
    let g_descent = array![3.0, 4.0];
    let n1 = projected_gradient_norm(&x, &g_descent, Some(&(lower, upper)));
    assert!(
        (n1 - 5.0).abs() < EPS,
        "g>0 at upper bound is feasible descent and must be retained (‖g‖=5); got {n1}"
    );
}

#[test]
fn mixed_active_set_matches_kkt_residual() {
    // Coord 0 railed low with feasible descent (retained), coord 1 railed low with
    // infeasible multiplier (dropped), coord 2 interior (retained), coord 3 railed
    // high with feasible descent (retained).
    let lower = array![0.0, 0.0, -10.0, -10.0];
    let upper = array![10.0, 10.0, 10.0, 0.0];
    let x = array![0.0, 0.0, 5.0, 0.0];
    let g = array![-3.0, 7.0, 4.0, 12.0];
    // retained = (-3)² + 0 + 4² + 12² = 9 + 16 + 144 = 169 → 13.
    let n = projected_gradient_norm(&x, &g, Some(&(lower, upper)));
    assert!(
        (n - 13.0).abs() < EPS,
        "mixed active set KKT residual should be 13, got {n}"
    );
}
