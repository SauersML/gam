// Auto-derived `ResourcePolicy::for_problem` selection: small problems
// stay on the permissive `default_library` so non-operator bases work, but
// biobank-scale problems flip to `analytic_operator_required` so any silent
// dense fallback errors out instead of allocating tens of GiB.
//
// We assert the policy itself plus the downstream `try_to_dense_arc_with_policy`
// behavior on a known-dense fallback (a `LazyDesignMatrix`) wrapped in
// `DenseDesignMatrix::Lazy` — small data densifies cleanly, biobank-scale
// densification is rejected with a precise error message naming the operator.

use gam::resource::{
    DerivativeStorageMode, ProblemHints, ResourcePolicy, STRICT_POLICY_NROWS_THRESHOLD,
};

#[test]
fn small_problem_selects_default_library() {
    let p = ResourcePolicy::for_problem(1_000, 50, ProblemHints::default());
    assert!(matches!(
        p.derivative_storage_mode,
        DerivativeStorageMode::MaterializeIfSmall
    ));
}

#[test]
fn biobank_n_selects_strict() {
    let p = ResourcePolicy::for_problem(STRICT_POLICY_NROWS_THRESHOLD, 50, ProblemHints::default());
    assert!(matches!(
        p.derivative_storage_mode,
        DerivativeStorageMode::AnalyticOperatorRequired
    ));
    let p2 = ResourcePolicy::for_problem(200_000, 50, ProblemHints::default());
    assert!(matches!(
        p2.derivative_storage_mode,
        DerivativeStorageMode::AnalyticOperatorRequired
    ));
}

#[test]
fn marginal_slope_hint_forces_strict() {
    let p = ResourcePolicy::for_problem(
        100,
        10,
        ProblemHints {
            marginal_slope_biobank_active: true,
        },
    );
    assert!(matches!(
        p.derivative_storage_mode,
        DerivativeStorageMode::AnalyticOperatorRequired
    ));
}

#[test]
fn strict_policy_rejects_dense_materialization_with_helpful_message() {
    use gam::matrix::panic_or_error_if_biobank_mode_and_to_dense_called_with_policy;
    let policy = ResourcePolicy::for_problem(200_000, 50, ProblemHints::default());
    let err = panic_or_error_if_biobank_mode_and_to_dense_called_with_policy(
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
    assert!(file!().ends_with(".rs"));
    use gam::matrix::panic_or_error_if_biobank_mode_and_to_dense_called_with_policy;
    let policy = ResourcePolicy::for_problem(1_000, 50, ProblemHints::default());
    // Small dense block (1k rows * 50 cols * 8 B = 400 KiB) is well under the
    // 256 MiB single-materialization cap and the policy is permissive, so the
    // guard returns Ok.
    panic_or_error_if_biobank_mode_and_to_dense_called_with_policy(
        "TestOperator::to_dense",
        1_000,
        50,
        &policy,
    )
    .expect("small-data densification should be permitted");
}
