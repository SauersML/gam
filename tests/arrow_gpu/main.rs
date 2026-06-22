//! Grouped integration-test crate root for arrow_gpu tests (issue #1146).
//!
//! Former top-level crates included as modules so they link as ONE binary.

mod gpu_gate;

mod arrow_schur_block_solve_matches_dense_identity;
mod arrow_schur_complement_matches_dense_reference;
mod arrow_schur_cross_row;
mod arrow_schur_gpu_v100_validation;
mod arrow_schur_inertia_matches_sylvester_law;
mod arrow_schur_mixed_precision;
mod gpu_backend_status_and_policy_dispatch_are_consistent;
mod gpu_matvec_matmul_xtdiagx_match_cpu_small_inputs;
mod gpu_numerical_stability;
mod gpu_pirls_gating;
mod gpu_required_tests_did_not_skip;
mod gpu_runtime_global_init_is_deterministic_and_idempotent;
mod gpu_solver_dimension_mismatch_returns_error;
mod gpu_solver_matches_cpu_on_small_spd_system;
mod gpu_spectral_leverage_diagonal_matches_cpu;
