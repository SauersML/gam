#[test]
fn fails_leapfrog_energy_drift_scales_quadratic_in_step_size_over_50_steps() {
    assert!(
        false,
        "Leapfrog energy drift should scale as O(epsilon^2) over 50 steps, not O(1), when epsilon is varied."
    );
}

#[test]
fn fails_leapfrog_reversibility_forward_then_backward_returns_initial_state() {
    assert!(
        false,
        "Forward k leapfrog steps followed by backward k steps should return to original (q, p) within 1e-10."
    );
}

#[test]
fn fails_nuts_uturn_is_only_detected_when_documented_condition_holds() {
    assert!(
        false,
        "NUTS U-turn must be detected exactly when the documented U-turn condition holds, and never earlier."
    );
}

#[test]
fn fails_whitening_momentum_draw_has_identity_covariance_after_1000_draws() {
    assert!(
        false,
        "Cov-orthogonalized momentum draws should have identity covariance to tolerance across 1000 draws."
    );
}

#[test]
fn fails_joint_beta_rho_posterior_matches_unnormalized_exp_log_density_identity() {
    assert!(
        false,
        "At random (beta, rho), joint posterior must match unnormalized exp(L(beta, rho)) consistent with REML/LAML assembly."
    );
}

#[test]
fn fails_validate_firth_likelihood_support_returns_err_for_unsupported_families_without_panic() {
    assert!(
        false,
        "validate_firth_likelihood_support should return Err for unsupported families and must not panic."
    );
}

#[test]
fn fails_per_family_logp_and_grad_consistency_for_every_response_link_tuple() {
    assert!(
        false,
        "Per-family logp and gradient should be consistent for every supported (response, link) tuple."
    );
}

#[test]
fn fails_posterior_mean_recovery_for_known_gaussian_posterior_within_3sigma() {
    assert!(
        false,
        "HMC draws from a known Gaussian posterior should recover the posterior mean within 3 sigma."
    );
}
