//! Regression guard for the useless-test gate's helper-delegation resolution
//! (`build.rs`: `scan_for_useless_tests` → `test_body_reaches_assertion`).
//!
//! The `#[test]` below delegates ALL of its assertions to a local helper whose
//! name — `check_rmse_invariants` — does NOT match the gate's
//! `assert_*`/`expect_*`/`require_*`/`ensure_*` prefix convention. It can
//! therefore only satisfy the gate if the gate follows the call into the helper
//! body and finds the real `assert!`s there (semantic resolution), rather than
//! trusting or rejecting the helper by its name. If the gate ever regresses to
//! a pure name-prefix allowlist, the whole crate stops building on this file —
//! exactly the failure mode reported in issue #503, reproduced here from a
//! different angle (a generic `check_*` name instead of `report_and_check`).
//!
//! The asserted content is genuine: it pins the behaviour of the shared
//! `reference::rmse` helper that the reference-quality tests depend on.

use gam::test_support::reference::rmse;

/// Non-prefix-named assertion helper. The useless-test gate must resolve the
/// delegating `#[test]` into this body to see that it asserts anything.
fn check_rmse_invariants() {
    // Identical vectors → exactly zero error.
    assert_eq!(rmse(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]), 0.0);
    // A pure constant offset of c collapses to RMSE == |c|.
    let got = rmse(&[0.0, 1.0, 2.0, 3.0], &[2.0, 3.0, 4.0, 5.0]);
    assert!((got - 2.0).abs() < 1e-12, "constant-offset rmse: {got}");
    // Mixed diffs [3, 4] → sqrt((9 + 16) / 2) = sqrt(12.5).
    let mixed = rmse(&[0.0, 0.0], &[3.0, 4.0]);
    assert!(
        (mixed - 12.5_f64.sqrt()).abs() < 1e-12,
        "mixed-diff rmse: {mixed}"
    );
}

#[test]
fn rmse_invariants_via_delegated_helper() {
    check_rmse_invariants();
}
