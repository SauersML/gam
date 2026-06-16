//! Grouped integration-test crate root for arrow_gpu tests (issue #1146).
//!
//! Former top-level crates included as modules so they link as ONE binary.

#[path = "arrow_gpu/arrow_schur_block_solve_matches_dense_identity.rs"]
mod arrow_schur_block_solve_matches_dense_identity;
#[path = "arrow_gpu/arrow_schur_complement_matches_dense_reference.rs"]
mod arrow_schur_complement_matches_dense_reference;
#[path = "arrow_gpu/arrow_schur_cross_row.rs"]
mod arrow_schur_cross_row;
#[path = "arrow_gpu/arrow_schur_gpu_v100_validation.rs"]
mod arrow_schur_gpu_v100_validation;
#[path = "arrow_gpu/arrow_schur_inertia_matches_sylvester_law.rs"]
mod arrow_schur_inertia_matches_sylvester_law;
#[path = "arrow_gpu/arrow_schur_mixed_precision.rs"]
mod arrow_schur_mixed_precision;
#[path = "arrow_gpu/gpu_backend_status_and_policy_dispatch_are_consistent.rs"]
mod gpu_backend_status_and_policy_dispatch_are_consistent;
#[path = "arrow_gpu/gpu_matvec_matmul_xtdiagx_match_cpu_small_inputs.rs"]
mod gpu_matvec_matmul_xtdiagx_match_cpu_small_inputs;
#[path = "arrow_gpu/gpu_numerical_stability.rs"]
mod gpu_numerical_stability;
#[path = "arrow_gpu/gpu_pirls_gating.rs"]
mod gpu_pirls_gating;
#[path = "arrow_gpu/gpu_required_tests_did_not_skip.rs"]
mod gpu_required_tests_did_not_skip;
#[path = "arrow_gpu/gpu_runtime_global_init_is_deterministic_and_idempotent.rs"]
mod gpu_runtime_global_init_is_deterministic_and_idempotent;
#[path = "arrow_gpu/gpu_solver_dimension_mismatch_returns_error.rs"]
mod gpu_solver_dimension_mismatch_returns_error;
#[path = "arrow_gpu/gpu_solver_matches_cpu_on_small_spd_system.rs"]
mod gpu_solver_matches_cpu_on_small_spd_system;
#[path = "arrow_gpu/gpu_spectral_leverage_diagonal_matches_cpu.rs"]
mod gpu_spectral_leverage_diagonal_matches_cpu;
