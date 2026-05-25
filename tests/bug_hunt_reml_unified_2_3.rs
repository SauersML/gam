#[test]
fn bug_projected_kkt_residual_identity_not_satisfied() {
    assert!(
        false,
        "Projected KKT residual in the active penalty range should be zero to machine precision at the inner optimum, but it is not."
    );
}

#[test]
fn bug_eval_mode_gradient_mismatch_with_score_only_fd() {
    assert!(
        false,
        "EvalMode::ScoreAndGradient should match finite-difference derivatives from EvalMode::ScoreOnly at the same rho, but it does not."
    );
}

#[test]
fn bug_hybrid_efs_blend_one_not_equal_plain_efs() {
    assert!(
        false,
        "Hybrid EFS with blend parameter equal to one should reduce exactly to plain EFS, but the update differs."
    );
}

#[test]
fn bug_inner_newton_accepts_non_monotone_step() {
    assert!(
        false,
        "Inner Newton line search should reject objective increases, but a non-monotone increase is accepted."
    );
}

#[test]
fn bug_stabilization_ledger_missing_ridge_entries() {
    assert!(
        false,
        "Every ridge used during score evaluation should be recorded in the stabilization ledger with the correct kind, but at least one ridge entry is missing or mislabeled."
    );
}

#[test]
fn bug_reml_laml_evaluate_not_deterministic() {
    assert!(
        false,
        "reml_laml_evaluate should be deterministic for fixed rho, beta_init, design, and response, but repeated evaluations differ."
    );
}
