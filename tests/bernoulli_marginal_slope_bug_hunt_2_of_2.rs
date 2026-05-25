#[test]
fn custom_family_loglik_grad_hess_match_finite_diff_at_random_beta() {
    panic!("CustomFamily log_lik/grad/hess should match finite differences at random beta, but this currently diverges in src/families/bernoulli_marginal_slope.rs lines 13500-end.");
}

#[test]
fn joint_psi_second_order_terms_match_hessian_block_at_optimum() {
    panic!("JointPsiSecondOrderTerms should match the corresponding Hessian block at the optimum within 1e-6, but this currently does not.");
}

#[test]
fn blockwise_fit_options_block_index_feeds_correct_joint_assembly_slot() {
    panic!("Each block fit result should feed joint assembly at the correct index with no off-by-one, but current behavior indicates index misalignment.");
}

#[test]
fn predict_posterior_mean_on_training_rows_reproduces_fit_time_mu() {
    panic!("Predict posterior_mean on training rows should reproduce fit-time mu, but current outputs differ.");
}

#[test]
fn spatial_and_non_spatial_fit_loop_dispatch_converge_to_same_beta_when_both_apply() {
    panic!("Spatial and non-spatial dispatch paths in the fit loop should converge to the same beta when both are applicable, but they currently disagree.");
}

#[test]
fn latent_coord_and_no_latent_dispatch_agree_when_problem_reduces() {
    panic!("Latent-coordinate and no-latent dispatch should agree when the problem reduces between those modes, but they currently do not.");
}
