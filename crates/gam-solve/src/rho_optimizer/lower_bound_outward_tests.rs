//! Sign-convention regression for [`super::lower_bound_outward_active_count`]
//! (#1074 / #1082 railing class).
//!
//! This is the exact mirror of the [`super::projected_gradient_norm`] KKT bug
//! fixed in a14b712. The optimizer MINIMIZES the cost and `gradient` is
//! ∂cost/∂ρ, so at an *active lower bound* the descent step `-g_i` exits the box
//! exactly when `g_i > 0`: a POSITIVE gradient is the infeasible outward /
//! separation pull this function is meant to count. A NEGATIVE `g_i` is feasible
//! interior descent (the part `projected_gradient_norm` KEEPS via `g_i.min(0.0)`)
//! and is NOT outward.
//!
//! The prior code filtered on `gradient[i] < -outward_floor`, counting the
//! feasible-interior-descent axes instead of the outward ones — the opposite —
//! so the separation-stationary fast-path mis-fired on axes that still had real
//! descent and certified a railed/under-fit optimum as converged. These tests
//! pin the correct sign and fail loudly on that inversion.

use super::{lower_bound_outward_active_count, LOWER_BOUND_SEPARATION_ACTIVE_MIN};
use ndarray::{array, Array1};

// `outward_floor = grad_threshold.max(COST_STALL_PROJECTED_GRAD_FLOOR)` with
// `COST_STALL_PROJECTED_GRAD_FLOOR = 1e-3`, so this pins the floor at 1e-3.
const GRAD_THRESHOLD: f64 = 1.0e-3;

fn bounds(lower: Array1<f64>, upper: Array1<f64>) -> (Array1<f64>, Array1<f64>) {
    (lower, upper)
}

#[test]
fn no_bounds_counts_nothing() {
    let x = array![0.0, 0.0];
    let g = array![5.0, 5.0];
    assert_eq!(
        lower_bound_outward_active_count(&x, &g, None, GRAD_THRESHOLD),
        0,
        "with no bounds no axis can be bound-active",
    );
}

#[test]
fn positive_gradient_at_lower_bound_is_counted() {
    // (a) Axis pinned at the lower bound with a strong POSITIVE gradient: the
    // descent step -g_i exits below the box → genuine outward/separation pull →
    // MUST be counted. The OLD code (g < -floor) would NOT count this.
    let lower = array![0.0];
    let upper = array![10.0];
    let x = array![0.0]; // on the lower bound
    let g = array![1.0]; // strong positive (≫ 1e-3 floor)
    let n = lower_bound_outward_active_count(&x, &g, Some(&bounds(lower, upper)), GRAD_THRESHOLD);
    assert_eq!(
        n, 1,
        "a strong POSITIVE gradient at the lower bound is the outward pull and must be counted",
    );
}

#[test]
fn negative_gradient_at_lower_bound_is_not_counted() {
    // (b) Axis pinned at the lower bound with a strong NEGATIVE gradient: the
    // descent step -g_i moves UP into the box → feasible interior descent → must
    // NOT be counted. The OLD code (g < -floor) would WRONGLY count this.
    let lower = array![0.0];
    let upper = array![10.0];
    let x = array![0.0]; // on the lower bound
    let g = array![-1.0]; // strong negative: feasible interior descent
    let n = lower_bound_outward_active_count(&x, &g, Some(&bounds(lower, upper)), GRAD_THRESHOLD);
    assert_eq!(
        n, 0,
        "a strong NEGATIVE gradient at the lower bound is feasible descent, not outward",
    );
}

#[test]
fn interior_axis_is_never_counted() {
    // (c) An interior axis (well off the lower bound) is never counted, no matter
    // how positive its gradient is.
    let lower = array![0.0];
    let upper = array![10.0];
    let x = array![5.0]; // strictly interior
    let g = array![100.0]; // huge positive, but the axis is not bound-active
    let n = lower_bound_outward_active_count(&x, &g, Some(&bounds(lower, upper)), GRAD_THRESHOLD);
    assert_eq!(n, 0, "an interior (non-bound) axis must never be counted");
}

#[test]
fn weak_outward_gradient_below_floor_is_not_counted() {
    // The pull must clear the floor: a tiny positive gradient at the bound is
    // numerical noise, not a separation probe.
    let lower = array![0.0];
    let upper = array![10.0];
    let x = array![0.0];
    let g = array![1.0e-6]; // positive but ≪ floor (1e-3)
    let n = lower_bound_outward_active_count(&x, &g, Some(&bounds(lower, upper)), GRAD_THRESHOLD);
    assert_eq!(n, 0, "a sub-floor outward gradient must not be counted");
}

#[test]
fn separation_active_min_threshold_semantics() {
    // (d) The consumer compares the count against LOWER_BOUND_SEPARATION_ACTIVE_MIN
    // (== 2): one outward-railed axis is below the activation threshold; two trip
    // it. Mix in feasible-descent and interior axes that must NOT contribute.
    //
    //   idx 0: lower-bound + positive gradient  → outward, counted
    //   idx 1: lower-bound + positive gradient  → outward, counted
    //   idx 2: lower-bound + negative gradient  → feasible descent, NOT counted
    //   idx 3: interior    + positive gradient  → not bound-active, NOT counted
    let lower = array![0.0, 0.0, 0.0, 0.0];
    let upper = array![10.0, 10.0, 10.0, 10.0];
    let x = array![0.0, 0.0, 0.0, 5.0];
    let g = array![1.0, 2.0, -3.0, 4.0];
    let n = lower_bound_outward_active_count(
        &x,
        &g,
        Some(&bounds(lower.clone(), upper.clone())),
        GRAD_THRESHOLD,
    );
    assert_eq!(
        n, 2,
        "only the two outward lower-bound axes count (idx 0,1); feasible-descent and interior excluded",
    );
    assert!(
        n >= LOWER_BOUND_SEPARATION_ACTIVE_MIN,
        "two outward axes must meet the separation-active activation threshold",
    );

    // A single outward axis (drop idx 1's pull below the floor) is below threshold.
    let g_one = array![1.0, -2.0, -3.0, 4.0];
    let n_one =
        lower_bound_outward_active_count(&x, &g_one, Some(&bounds(lower, upper)), GRAD_THRESHOLD);
    assert_eq!(n_one, 1, "only one axis is outward-railed here");
    assert!(
        n_one < LOWER_BOUND_SEPARATION_ACTIVE_MIN,
        "a single outward axis must stay below the separation-active threshold",
    );
}
