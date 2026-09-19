//! #2953: a trust-region run that stops without a convergence claim leaves the iterate it
//! stopped at as a non-converged checkpoint, whichever stop it was.
//!
//! A dense ARC run whose cubic regularisation saturated next to the prior mode ended in
//! `RemlOptimizationFailed` and lost the iterate `opt` handed back with its
//! `TrustRegionRejectFloor`, 7e4 below the point the fit then returned. The reject floor and
//! the budget exit now map through one function.

use super::*;
use ndarray::array;

fn stopped_solution(point: f64, value: f64) -> Solution {
    Solution {
        final_point: array![point],
        final_value: value,
        final_gradient: Some(array![8.259e-2]),
        final_hessian: None,
        final_gradient_norm: Some(8.259e-2),
        final_step_norm: None,
        stationarity_kind: opt::StationarityKind::ProjectedGradient,
        iterations: 7,
        func_evals: 48,
        grad_evals: 48,
        hess_evals: 48,
        termination: TerminationReason::TrustRegionRejectFloor {
            radius: 1.0e-12,
            floor: 1.0e-12,
            consecutive_rejections: 40,
            grad_norm: 8.259e-2,
        },
    }
}

fn arc_plan() -> OuterPlan {
    OuterPlan {
        solver: Solver::Arc,
        hessian_source: HessianSource::Analytic,
    }
}

#[test]
fn a_reject_floor_stop_keeps_the_iterate_it_stopped_at_2953() {
    let result = stopped_run_checkpoint(
        "reject floor #2953",
        "ARC reject-floor",
        stopped_solution(6.907748424675003, -5.907172032664134e5),
        None,
        arc_plan(),
    );
    assert!(
        !result.solver_claimed_convergence(),
        "a reject-floor stop makes no convergence claim"
    );
    assert_eq!(result.rho[0].to_bits(), 6.907748424675003_f64.to_bits());
    assert_eq!(result.final_value.to_bits(), (-5.907172032664134e5_f64).to_bits());
    assert_eq!(result.iterations, 7);
    assert!(
        matches!(
            result.solver_termination,
            Some(TerminationReason::TrustRegionRejectFloor { .. })
        ),
        "the checkpoint must say how the run stopped: {:?}",
        result.solver_termination
    );
    assert!(
        matches!(result.origin, OuterResultOrigin::Solver),
        "the iterate is the solver's own: {:?}",
        result.origin
    );
}

#[test]
fn a_stopped_run_takes_the_guards_lower_feasible_iterate_2953() {
    let best = CostStallExit {
        rho: array![6.5],
        value: -5.95e5,
        grad_norm: 3.0e-1,
        iterations: 5,
        converged: false,
        probe_scale: None,
        rank_boundary: None,
    };
    let result = stopped_run_checkpoint(
        "reject floor #2953",
        "ARC reject-floor",
        stopped_solution(6.907748424675003, -5.907172032664134e5),
        Some(best),
        arc_plan(),
    );
    assert!(!result.solver_claimed_convergence());
    assert_eq!(result.rho[0].to_bits(), 6.5_f64.to_bits());
    assert_eq!(result.final_value.to_bits(), (-5.95e5_f64).to_bits());
    assert_eq!(
        result.iterations, 7,
        "the run spent its whole count, not the index of the substituted iterate"
    );
    assert!(
        matches!(result.origin, OuterResultOrigin::ArcBestIterateSubstitution),
        "{:?}",
        result.origin
    );

    let higher = CostStallExit {
        rho: array![6.0],
        value: -5.0e5,
        grad_norm: 1.0,
        iterations: 3,
        converged: false,
        probe_scale: None,
        rank_boundary: None,
    };
    let kept = stopped_run_checkpoint(
        "reject floor #2953",
        "ARC reject-floor",
        stopped_solution(6.907748424675003, -5.907172032664134e5),
        Some(higher),
        arc_plan(),
    );
    assert_eq!(
        kept.rho[0].to_bits(),
        6.907748424675003_f64.to_bits(),
        "a higher feasible iterate never replaces the stopped one"
    );
}
