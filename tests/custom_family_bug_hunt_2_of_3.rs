#[test]
fn exact_newton_joint_psi_terms_rows_sum_without_block_boundary_double_counting() {
    panic!("ExactNewtonJointPsiTerms should assemble per-row contributions that exactly equal the joint Psi gradient and Hessian at fit time, with no block-boundary double counting.");
}

#[test]
fn custom_family_warm_start_roundtrip_refit_preserves_beta() {
    panic!("CustomFamilyWarmStart serialize -> deserialize -> re-fit should produce identical beta values for a fixed seed.");
}

#[test]
fn block_working_set_deactivation_zeroes_contribution_and_reactivation_restores_it() {
    panic!("BlockWorkingSet activation flags should zero deactivated block contributions in the working set and restore them after reactivation.");
}

#[test]
fn embedded_implicit_and_dense_psi_derivative_operators_match_matvec() {
    panic!("EmbeddedImplicitPsiDerivativeOperator and EmbeddedDensePsiDerivativeOperator should produce identical matvec outputs for the same random input vector.");
}

#[test]
fn materialize_then_apply_matches_apply_then_materialize_for_psi_operator() {
    panic!("MaterializablePsiDerivativeOperator contract violated: M = op.materialize() must satisfy Mv == op.apply(v).");
}

#[test]
fn custom_family_psi_linear_map_ref_borrow_must_not_outlive_source() {
    panic!("CustomFamilyPsiLinearMapRef lifetime contract should prevent borrows from outliving their source linear map.");
}

#[test]
fn per_block_outer_hessian_coupling_populates_off_diagonal_cross_derivatives() {
    panic!("Per-block outer-Hessian assembly should populate off-diagonal entries with cross-block second derivatives when blocks are coupled.");
}

#[test]
fn warm_start_reapplying_same_data_has_zero_beta_drift() {
    panic!("Warm-start drift detected: applying the same warm start on unchanged data should produce zero beta change.");
}
