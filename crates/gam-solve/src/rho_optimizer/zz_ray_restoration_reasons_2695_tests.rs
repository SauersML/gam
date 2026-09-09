//! gam#2695 — a ray the inner solve was descending is a property of the
//! ACCEPTED STEP, not of the exit that noticed it, so the outer's restoration
//! must read one from every terminal reason that can carry one.
//!
//! Before this, only `SlowGeometricRate` was matched. On the 1569 pair four of
//! five seeds leave through the residual-stall / divergence guard instead —
//! every step accepted at a good model ratio, the objective still falling and
//! `‖β‖∞` still growing — and their rays were dropped on the floor, so
//! `eval_seed_restoring_rays` refused each of those seeds "as evaluated" and
//! the next seed started cold.

use super::*;

fn ray() -> gam_problem::RayRestoration {
    gam_problem::RayRestoration {
        block: 2,
        rho_first: 3,
        rho_count: 1,
        log_strength_ratio: 5.75,
        likelihood_slope: -4.473e-3,
        penalty_slope: 1.317e-5,
        block_step_inf: 7.5e-2,
    }
}

fn refusal(reason: gam_problem::JointNewtonTerminalReason) -> EstimationError {
    EstimationError::CustomFamily(gam_problem::CustomFamilyError::InnerSolveNotConverged {
        cycles: 51,
        terminal: Some(gam_problem::InnerConvergenceTerminalState::JointNewton {
            cycle: 51,
            stationarity_residual: 7.443,
            residual_tol: 2.791e-3,
            stationarity_scale: 7.3e9,
            step_inf: 9.31,
            step_tol: 1e-6,
            resolvable_negative_curvature: false,
            best_stationarity_residual: 7.443,
            cycles_since_best_residual: 4,
            termination_reason: reason,
        }),
        kkt_residual: Some(7.443),
        kkt_tol: Some(2.791e-3),
        theta_dim: 8,
        rho_dim: 5,
        psi_dim: 0,
    })
}

#[test]
fn a_stalled_exit_carries_its_ray_to_the_restoration_2695() {
    let stalled = refusal(
        gam_problem::JointNewtonTerminalReason::StalledOnDescendingRay {
            residual: 7.443,
            residual_tol: 2.791e-3,
            cycles: 51,
            ray: ray(),
        },
    );
    let read = ray_restoration_in(&stalled).expect(
        "the residual-stall / divergence exit's ray must reach the outer restoration (gam#2695)",
    );
    assert_eq!(read.block, 2);
    assert_eq!(read.rho_indices().collect::<Vec<_>>(), vec![3]);
    assert_eq!(read.log_strength_ratio, 5.75);

    // The slow-rate exit still reads, on both of its arms.
    let slow_with = refusal(gam_problem::JointNewtonTerminalReason::SlowGeometricRate {
        rate_per_cycle: 0.99,
        window_cycles: 8,
        projected_cycles_to_tolerance: 4000,
        residual: 7.443,
        residual_tol: 2.791e-3,
        ray: Some(ray()),
    });
    assert!(ray_restoration_in(&slow_with).is_some());
    let slow_without = refusal(gam_problem::JointNewtonTerminalReason::SlowGeometricRate {
        rate_per_cycle: 0.99,
        window_cycles: 8,
        projected_cycles_to_tolerance: 4000,
        residual: 7.443,
        residual_tol: 2.791e-3,
        ray: None,
    });
    assert!(
        ray_restoration_in(&slow_without).is_none(),
        "an unpenalized ray is still no restoration"
    );

    // A reason that carries no ray reads as none — the negative control that
    // keeps the match above from being a blanket `Some`.
    let budget = refusal(gam_problem::JointNewtonTerminalReason::CycleBudget);
    assert!(ray_restoration_in(&budget).is_none());

    // And the new reason says what it is, so a log reader can tell it from a
    // slow rate.
    let text = format!(
        "{}",
        gam_problem::JointNewtonTerminalReason::StalledOnDescendingRay {
            residual: 7.443,
            residual_tol: 2.791e-3,
            cycles: 51,
            ray: ray(),
        }
    );
    assert!(
        text.contains("stalled or grew") && text.contains("under-penalized"),
        "the stalled reason must read as a stall, not as a rate: {text}"
    );
}
