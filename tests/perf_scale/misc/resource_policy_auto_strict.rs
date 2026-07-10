// Auto-derived `ResourcePolicy::for_problem` selection and the downstream
// dense-materialization guard.
//
// The policy no longer carries a row/column cliff: shape alone never flips the
// mode (that anti-pattern was replaced by the process-wide `MemoryGovernor`
// byte ledger, which decides dense-vs-chunked-vs-matrix-free from the checked
// predicted live bytes of each operation). Strict mode is reached only through
// a *structural* signal — the marginal-slope large-scale path, which is
// operator-only by construction — or the explicit `analytic_operator_required`
// preset. These integration tests pin that contract plus the
// `try_to_dense`-guard behavior on a known-dense fallback.

use gam::resource::{DerivativeStorageMode, ProblemHints, ResourcePolicy};

#[test]
fn small_problem_selects_default_library() {
    let p = ResourcePolicy::for_problem(ProblemHints::default());
    assert!(matches!(
        p.derivative_storage_mode,
        DerivativeStorageMode::MaterializeIfSmall
    ));
}

#[test]
fn shape_never_flips_the_mode() {
    // The old `STRICT_POLICY_NROWS_THRESHOLD` cliff is gone: a nominally
    // large-scale problem with no structural hint stays permissive, and the
    // per-operation governor — not the row count — makes the dense decision.
    let default_hints = ResourcePolicy::for_problem(ProblemHints::default());
    let large_shape_hints = ResourcePolicy::for_problem(ProblemHints {
        marginal_slope_large_scale_active: false,
    });
    assert!(matches!(
        default_hints.derivative_storage_mode,
        DerivativeStorageMode::MaterializeIfSmall
    ));
    assert!(matches!(
        large_shape_hints.derivative_storage_mode,
        DerivativeStorageMode::MaterializeIfSmall
    ));
}

#[test]
fn marginal_slope_hint_forces_strict() {
    let p = ResourcePolicy::for_problem(ProblemHints {
        marginal_slope_large_scale_active: true,
    });
    assert!(matches!(
        p.derivative_storage_mode,
        DerivativeStorageMode::AnalyticOperatorRequired
    ));
}

#[test]
fn strict_policy_rejects_dense_materialization_with_helpful_message() {
    use gam::matrix::panic_or_error_if_large_scale_mode_and_to_dense_called_with_policy;
    // Strict mode comes from the structural preset, not a row count.
    let policy = ResourcePolicy::analytic_operator_required();
    let err = panic_or_error_if_large_scale_mode_and_to_dense_called_with_policy(
        "TestOperator::to_dense",
        200_000,
        50,
        &policy,
    )
    .unwrap_err();
    // The error must name the operator (so users can find the missing analytic
    // path) and the policy mode (so the cause is unambiguous).
    assert!(
        err.contains("TestOperator::to_dense"),
        "expected operator name in error, got: {err}"
    );
    assert!(
        err.contains("AnalyticOperatorRequired"),
        "expected policy name in error, got: {err}"
    );
}

#[test]
fn default_library_admits_small_dense_materialization() {
    use gam::matrix::panic_or_error_if_large_scale_mode_and_to_dense_called_with_policy;
    let policy = ResourcePolicy::for_problem(ProblemHints::default());
    // Small dense block (1k rows * 50 cols * 8 B = 400 KiB) is well under the
    // single-materialization cap and the policy is permissive, so the guard
    // returns Ok.
    panic_or_error_if_large_scale_mode_and_to_dense_called_with_policy(
        "TestOperator::to_dense",
        1_000,
        50,
        &policy,
    )
    .unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "small-data densification should be permitted", e
        )
    });
}
