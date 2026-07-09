//! #2235 — OuterTerminationLedger semantics under the FORCING-FUNCTION
//! contract: a fit object exists only from a CONVERGED optimization. The
//! ledger is pure accounting plus one tripwire — a stationarity DEFECT that
//! must surface as a typed error (never a frozen "fit"). There is no wall
//! budget and no criterion freeze: wall survival belongs to checkpoint/resume,
//! and non-convergence gets fixed, not returned.

use super::outer_objective::OuterTerminationLedger;

fn window() -> u64 {
    // Mirror of the derivation (2 line-search directions × 70 probes); the
    // ledger exposes it only through behavior, so pin the derived value here —
    // if the derivation changes, this test documents that it was deliberate.
    140
}

#[test]
fn stationarity_defect_fires_within_window_and_never_while_improving() {
    let mut ledger = OuterTerminationLedger::new();
    // Improving phase: the defect must never trip while material improvement
    // keeps landing.
    let mut cost = 100.0;
    for _ in 0..window() * 2 {
        assert!(
            ledger.stationarity_defect().is_none(),
            "improving walk must never trip the defect"
        );
        ledger.record(cost);
        cost -= 1.0;
    }
    // Stationary phase: a full window with no material improvement is the
    // defect. The message must carry the evidence (window size, eval count).
    let mut tripped_at = None;
    for i in 0..window() + 2 {
        if let Some(msg) = ledger.stationarity_defect() {
            assert!(
                msg.contains("SOLVER DEFECT"),
                "defect message must be unambiguous: {msg}"
            );
            tripped_at = Some(i);
            break;
        }
        ledger.record(cost + 0.5); // worse than best: no improvement
    }
    let tripped_at = tripped_at.expect("stationary walk must trip the defect");
    assert!(
        tripped_at <= window() + 1,
        "defect must fire within the stationarity window (fired at {tripped_at})"
    );
}

#[test]
fn defect_never_fires_before_anything_is_banked() {
    let ledger = OuterTerminationLedger::new();
    // With no banked incumbent there is no evidence of wandering — the very
    // first evaluations can never trip the defect.
    assert!(ledger.stationarity_defect().is_none());
}

#[test]
fn reset_stationarity_gives_a_fresh_seed_a_fresh_window() {
    let mut ledger = OuterTerminationLedger::new();
    ledger.record(10.0);
    for _ in 0..window() + 1 {
        ledger.record(10.0);
    }
    assert!(ledger.stationarity_defect().is_some(), "defect armed");
    // A multi-start reset starts a NEW walk: fresh evidence window.
    ledger.reset_stationarity();
    assert!(
        ledger.stationarity_defect().is_none(),
        "fresh seed must get a fresh stationarity window"
    );
}

#[test]
fn non_finite_and_immaterial_costs_do_not_count_as_improvement() {
    let mut ledger = OuterTerminationLedger::new();
    ledger.record(50.0);
    for i in 0..window() + 1 {
        // Alternate NaN and sub-tolerance wiggles: none is material.
        if i % 2 == 0 {
            ledger.record(f64::NAN);
        } else {
            ledger.record(50.0 - 1.0e-12);
        }
    }
    assert!(
        ledger.stationarity_defect().is_some(),
        "noise below the material tolerance must not reset the window"
    );
}

/// #2230 regression, forcing-function form. The historical hang re-entered the
/// inner driver for hours with no criterion improvement and no conclusion.
/// Under the new contract a fit has exactly two possible endings: the bridge
/// CONVERGES (a fit object with bounded eval accounting — the healthy path this
/// planted-circle fit must take), or the stationarity defect RAISES a typed
/// error. An unbounded loop is no longer expressible. This drives the real
/// outer bridge with no budget and asserts the healthy ending + the bound.
#[test]
fn unbudgeted_fit_terminates_bounded_never_hangs() {
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
    gam_solve::rho_optimizer::OuterProblem::new(n_params)
        .with_initial_rho(seed)
        .with_max_iter(8)
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..Default::default()
        })
        .run(&mut objective, "SAE manifold")
        .expect("the healthy planted-circle fit must converge through the bridge");
    let fitted = objective.into_fitted();
    let ev = global_ev(z.view(), fitted.term.fitted().view());
    eprintln!(
        "[#2235] converged fit: evals={} since_improvement={} wall={:.2?} ev={ev:.4}",
        fitted.termination.evals,
        fitted.termination.evals_since_improvement,
        fitted.termination.wall
    );
    assert!(ev.is_finite() && ev > 0.3, "converged fit must explain the circle");
    assert!(
        fitted.termination.evals >= 1,
        "the accounting must have ticked"
    );
    // The #2230 bound: the whole 8-outer-iteration fit must conclude within a
    // few stationarity windows of evaluations — the unbounded churn regime
    // would blow far past this.
    assert!(
        fitted.termination.evals < 10 * window(),
        "outer search used {} evals — churn regression (#2230)",
        fitted.termination.evals
    );
}
