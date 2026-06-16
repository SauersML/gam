//! Regression: the log-barrier objective must diverge to `+∞` at and past a
//! constraint boundary (continuous extension of `−τ Σ log Δ_j`), never return
//! a finite value or NaN. Generic outer/line-search code compares scalar
//! objectives; the only way "the infeasible step is rejected" is for the
//! evaluator's scalar to be larger than every feasible neighbour. That is
//! exactly the limit `−τ ln(0⁺) = +∞`, so we encode that contract literally.

use gam::solver::BarrierConfig;
use ndarray::array;

/// `barrier_cost` at the exact constraint boundary `s β = b` must return
/// `+∞`, not a finite number and not NaN.
#[test]
fn barrier_term_diverges_at_constraint_boundary() {
    // Single coordinate constraint β₀ ≥ 0 with τ = 1.
    let cfg = BarrierConfig {
        tau: 1.0,
        constrained_indices: vec![0],
        lower_bounds: vec![0.0],
        bound_signs: vec![1.0],
    };

    // -- Feasible interior: must be finite real. ---------------------------
    let beta_interior = array![0.5];
    let cost_interior = cfg.barrier_cost(&beta_interior);
    assert!(
        cost_interior.is_finite(),
        "barrier cost on the feasible interior must be finite; got {cost_interior}"
    );

    // -- Approach to the boundary: monotone increase to +∞. ----------------
    let mut last = f64::NEG_INFINITY;
    for &delta in &[1e-1_f64, 1e-3, 1e-6, 1e-12] {
        let beta = array![delta];
        let c = cfg.barrier_cost(&beta);
        assert!(
            c.is_finite(),
            "barrier cost must remain finite for Δ > 0 (Δ = {delta}); got {c}"
        );
        assert!(
            c > last,
            "barrier cost must increase as Δ → 0⁺ (Δ = {delta}); got {c} ≤ previous {last}"
        );
        last = c;
    }

    // -- Exactly on the boundary: must be +∞ (continuous extension), not
    //    a finite number, not NaN.
    let beta_boundary = array![0.0];
    let cost_boundary = cfg.barrier_cost(&beta_boundary);
    assert!(
        !cost_boundary.is_nan(),
        "barrier cost at the boundary must not be NaN; got NaN"
    );
    assert_eq!(
        cost_boundary,
        f64::INFINITY,
        "barrier cost at the exact boundary (Δ = 0) must be +∞; got {cost_boundary}"
    );

    // -- Past the boundary (infeasible): must be +∞, never NaN or finite. -
    let beta_violation = array![-1e-9];
    let cost_violation = cfg.barrier_cost(&beta_violation);
    assert!(
        !cost_violation.is_nan(),
        "barrier cost past the boundary must not be NaN; got NaN"
    );
    assert_eq!(
        cost_violation,
        f64::INFINITY,
        "barrier cost past the boundary (Δ < 0) must be +∞; got {cost_violation}"
    );

    // -- Upper-bound flavor (s = −1, β ≤ b): symmetric behaviour. ---------
    let cfg_upper = BarrierConfig {
        tau: 1.0,
        constrained_indices: vec![0],
        lower_bounds: vec![0.0], // constraint: −β ≥ 0  ⇔  β ≤ 0
        bound_signs: vec![-1.0],
    };
    let cost_upper_boundary = cfg_upper.barrier_cost(&array![0.0]);
    assert_eq!(
        cost_upper_boundary,
        f64::INFINITY,
        "upper-bound barrier at the boundary must be +∞; got {cost_upper_boundary}"
    );
    let cost_upper_violation = cfg_upper.barrier_cost(&array![1e-9]);
    assert_eq!(
        cost_upper_violation,
        f64::INFINITY,
        "upper-bound barrier past the boundary must be +∞; got {cost_upper_violation}"
    );
}
