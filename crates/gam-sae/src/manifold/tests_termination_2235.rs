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
