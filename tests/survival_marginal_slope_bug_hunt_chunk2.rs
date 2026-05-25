#[test]
fn custom_family_impl_survival_marginal_slope_loglik_grad_hess_match_finite_difference_at_feasible_beta() {
    panic!("CustomFamily survival_marginal_slope log_lik/grad/hess should match finite differences at feasible random beta, but they do not.");
}

#[test]
fn outer_hessian_row_limit_gate_changes_runtime_mode_without_changing_fit_solution() {
    panic!("Crossing BMS_FLEX_OUTER_HESSIAN_ROW_LIMIT should change performance mode only; fit beta, calibration, mu, and score should remain nearly identical, but they differ.");
}

#[test]
fn blockwise_fit_joint_solution_matches_coupled_solution_within_tolerance() {
    panic!("Blockwise independent blocks fitted jointly should recover the same optimum as the coupled fit within tolerance, but they do not.");
}

#[test]
fn censored_and_event_rows_receive_correct_marginal_slope_weights() {
    panic!("Censored and event rows should receive the correct marginal-slope weighting contributions, but observed weighting is inconsistent.");
}

#[test]
fn posterior_mean_marginal_slope_matches_gradient_of_posterior_mean_wrt_x() {
    panic!("At prediction time, marginal slope evaluated at x should match d/dx of posterior mean, but they disagree.");
}

#[test]
fn predict_with_fitted_beta_reproduces_fit_time_mu_on_training_rows() {
    panic!("Predicting on training rows with fitted beta should reproduce fit-time mu, but it does not.");
}
