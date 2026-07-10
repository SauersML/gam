//! #2235 — a fit object exists only from a CONVERGED optimization. The local
//! ledger is accounting/checkpoint telemetry only; convergence is decided by
//! the shared outer optimizer's analytic certificate, never by a wall-clock or
//! evaluation-count deadline.

use super::outer_objective::OuterTerminationLedger;

#[test]
fn ledger_tracks_material_objective_improvement_without_deciding_convergence() {
    let mut ledger = OuterTerminationLedger::new();
    assert!(ledger.record(50.0), "the first finite objective is banked");
    assert!(!ledger.record(f64::NAN), "non-finite values are never improvements");
    assert!(
        !ledger.record(50.0 - 1.0e-12),
        "roundoff below the material objective tolerance is not improvement"
    );
    assert!(ledger.record(40.0), "a material objective descent is banked");
    let (evals, last_improvement, best) = ledger.checkpoint_counters();
    assert_eq!(evals, 4);
    assert_eq!(last_improvement, 4);
    assert_eq!(best, Some(40.0));
}

/// #2230 regression: the real bridge must return only with a first-order
/// certificate for the same objective/state that is minted into the fit.
#[test]
fn planted_circle_fit_returns_with_analytic_certificate() {
    use super::tests::{global_ev, planted_circle_embedded};
    use super::tests_startup_validation_1782::{Topo, objective_and_seed};

    let z = planted_circle_embedded(48, 6, 0.03);
    let (mut objective, seed) = objective_and_seed(
        z.view(),
        2,
        Topo::Circle,
        crate::assignment::AssignmentMode::softmax(1.0),
    );
    let n_params = seed.len();
    let result = gam_solve::rho_optimizer::OuterProblem::new(n_params)
        .with_initial_rho(seed)
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..Default::default()
        })
        .run(&mut objective, "SAE manifold")
        .expect("the healthy planted-circle fit must converge through the bridge");
    // #2235/#2241 — a certified conclusion always names WHICH certificate
    // concluded it (gradient-stationary / criterion-flat / recurrent-
    // incumbent); `certify_outer_optimality` stamps it on every converged
    // result the engine returns.
    assert!(result.converged, "run() only returns certified results");
    assert!(
        result.converged_via.is_some(),
        "a converged OuterResult must carry its converged-via certificate verdict"
    );
    let certificate = result
        .criterion_certificate
        .as_ref()
        .expect("a converged result carries the analytic objective certificate");
    assert!(
        certificate.projected_grad_norm <= certificate.stationarity_bound,
        "projected outer gradient {} exceeds certified bound {}",
        certificate.projected_grad_norm,
        certificate.stationarity_bound
    );
    let fitted = objective.into_fitted().expect("outer fit was evaluated");
    let ev = global_ev(z.view(), fitted.term.fitted().view());
    eprintln!(
        "[#2235] converged fit: evals={} since_improvement={} wall={:.2?} ev={ev:.4}",
        fitted.termination.evals,
        fitted.termination.evals_since_improvement,
        fitted.termination.wall
    );
    assert!(
        ev.is_finite() && ev > 0.3,
        "converged fit must explain the circle"
    );
    assert!(
        fitted.termination.evals >= 1,
        "the accounting must have ticked"
    );
}
