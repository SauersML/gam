//! #2235 — OuterTerminationLedger unit semantics: the pure state machine that
//! freezes the outer criterion. The fit-level property (a frozen criterion
//! converges the bridge onto the banked incumbent) is exercised end-to-end by
//! the zoo-micro harness; these pin the ledger's own contract.

use super::outer_objective::{OuterTerminationLedger, SaeOuterTerminationVerdict};

fn window() -> u64 {
    // Mirror of the derivation (2 line-search directions × 70 probes); the
    // ledger exposes it only through behavior, so pin the derived value here —
    // if the derivation changes, this test documents that it was deliberate.
    140
}

#[test]
fn ledger_freezes_on_incumbent_stationarity_and_reports_verdict() {
    let mut ledger = OuterTerminationLedger::new(None);
    // Improving phase: never freezes while material improvement keeps landing.
    let mut cost = 100.0;
    for _ in 0..window() * 2 {
        assert!(ledger.frozen_cost().is_none(), "improving walk must stay live");
        ledger.record(cost);
        cost -= 1.0; // material at any scale vs the tolerance
    }
    // Stationary phase: identical cost for a full window ⇒ freeze at the best.
    let best = cost + 1.0;
    let mut froze_at = None;
    for i in 0..window() + 2 {
        if let Some(frozen) = ledger.frozen_cost() {
            froze_at = Some((i, frozen));
            break;
        }
        ledger.record(best + 0.5); // worse than best, no improvement
    }
    let (evals_to_freeze, frozen) = froze_at.expect("stationary walk must freeze");
    assert!(
        evals_to_freeze <= window() + 1,
        "freeze must fire within the stationarity window (fired at {evals_to_freeze})"
    );
    assert_eq!(frozen, best, "frozen cost must be the banked best");
    let report = ledger.report();
    assert_eq!(report.verdict, SaeOuterTerminationVerdict::IncumbentStationary);
    // Frozen is latched: further consults return the same incumbent.
    assert_eq!(ledger.frozen_cost(), Some(best));
}

#[test]
fn ledger_never_freezes_before_anything_is_banked() {
    let mut ledger = OuterTerminationLedger::new(Some(std::time::Duration::ZERO));
    // Even with an already-expired wall budget there is no incumbent to
    // converge onto, so the very first consult must keep the fit live.
    assert!(ledger.frozen_cost().is_none());
    ledger.record(42.0);
    // Now the budget verdict can fire, onto the banked value.
    assert_eq!(ledger.frozen_cost(), Some(42.0));
    assert_eq!(
        ledger.report().verdict,
        SaeOuterTerminationVerdict::BudgetExhausted
    );
}

#[test]
fn reset_stationarity_reopens_a_stationary_freeze_but_not_a_budget_one() {
    let mut ledger = OuterTerminationLedger::new(None);
    ledger.record(10.0);
    for _ in 0..window() + 1 {
        ledger.record(10.0);
    }
    assert!(ledger.frozen_cost().is_some(), "stationary freeze armed");
    // A multi-start reset gives the new walk a fresh window.
    ledger.reset_stationarity();
    assert!(
        ledger.frozen_cost().is_none(),
        "fresh seed must get a live criterion"
    );

    let mut budget = OuterTerminationLedger::new(Some(std::time::Duration::ZERO));
    budget.record(1.0);
    assert!(budget.frozen_cost().is_some(), "budget freeze armed");
    budget.reset_stationarity();
    assert!(
        budget.frozen_cost().is_some(),
        "the wall budget is fit-global; reset must not reopen it"
    );
}

/// Fit-level deadline honesty: a full production cascade under an
/// adversarially tiny wall budget must RETURN — a fitted term with finite
/// reconstruction and a `BudgetExhausted` (or engine-converged) verdict —
/// never hang and never error. This is #2235's T4 property at the smallest
/// real shape (the planted-circle fixture the 1782 harness uses).
#[test]
fn tiny_wall_budget_fit_returns_certified_incumbent_not_error() {
    use super::tests::{global_ev, planted_circle_embedded};
    use super::tests_startup_validation_1782::{Topo, objective_and_seed};

    let z = planted_circle_embedded(48, 6, 0.03);
    let (mut objective, seed) = objective_and_seed(
        z.view(),
        2,
        Topo::Circle,
        crate::assignment::AssignmentMode::softmax(1.0),
    );
    // A 1ms budget expires before the second evaluation; the ledger freezes
    // as soon as the first cost is banked.
    objective.set_outer_wall_budget(std::time::Duration::from_millis(1));
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
        .expect("a budget-frozen fit must complete through the bridge, not error");
    let fitted = objective.into_fitted();
    let ev = global_ev(z.view(), fitted.term.fitted().view());
    eprintln!(
        "[#2235 tiny-budget] verdict={:?} evals={} ev={ev:.4}",
        fitted.termination.verdict, fitted.termination.evals
    );
    assert!(ev.is_finite(), "budget-frozen fit must return a real reconstruction");
    assert!(
        fitted.termination.evals >= 1,
        "at least one evaluation must have banked an incumbent"
    );
    // With a 1ms budget the verdict is BudgetExhausted unless the engine
    // legitimately finished inside the first evaluation (EngineStopped) —
    // both are verdicts; a hang or an Err is the only failure.
    assert!(
        matches!(
            fitted.termination.verdict,
            super::outer_objective::SaeOuterTerminationVerdict::BudgetExhausted
                | super::outer_objective::SaeOuterTerminationVerdict::EngineStopped
        ),
        "unexpected verdict {:?}",
        fitted.termination.verdict
    );
}

#[test]
fn non_finite_and_immaterial_costs_do_not_count_as_improvement() {
    let mut ledger = OuterTerminationLedger::new(None);
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
        ledger.frozen_cost().is_some(),
        "noise below the material tolerance must not reset the window"
    );
}
