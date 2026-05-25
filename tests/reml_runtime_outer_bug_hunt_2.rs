#[test]
fn outer_rho_newton_zero_score_makes_zero_step_bug_repro() {
    assert!(
        false,
        "When the outer score gradient at rho is exactly zero, the Newton proposal must be exactly zero and must not move rho."
    );
}

#[test]
fn outer_hessian_sign_step_must_be_descent_direction_bug_repro() {
    assert!(
        false,
        "For minimizing REML/LAML negative log-likelihood, the proposed outer Newton step must be a descent direction at non-optimum points."
    );
}

#[test]
fn ift_correction_drives_projected_inner_residual_to_numerical_zero_bug_repro() {
    assert!(
        false,
        "Applying the IFT correction at the inner solution should reduce the projected residual to numerical zero."
    );
}

#[test]
fn ift_quality_metric_drho_norm_over_h_pen_logdet_is_monotone_near_optimum_bug_repro() {
    assert!(
        false,
        "Near a clean optimum, the drho_norm / h_pen_logdet quality metric should evolve monotonically in the documented direction across outer iterations."
    );
}

#[test]
fn pirls_result_cache_is_invalidated_when_beta_changes_bug_repro() {
    assert!(
        false,
        "When beta changes between outer iterations, PIRLS-result reuse must be invalidated so any cached factor matches the just-updated beta."
    );
}

#[test]
fn multifamily_dispatch_tail_routes_every_family_to_the_correct_inner_solver_bug_repro() {
    assert!(
        false,
        "Each supported response family must dispatch through the multi-family tail to its correct inner solver implementation."
    );
}
