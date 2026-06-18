//! Grouped integration-test crate root for pyffi tests (issue #1146).
//!
//! Former top-level crates included as modules so they link as ONE binary.

#[path = "pyffi/audit_priority_perm_invariance.rs"]
mod audit_priority_perm_invariance;
#[path = "pyffi/ban_gate_delegated_assertion.rs"]
mod ban_gate_delegated_assertion;
#[path = "pyffi/canonicalize_routes_channel_aware_for_multi_channel.rs"]
mod canonicalize_routes_channel_aware_for_multi_channel;
#[path = "pyffi/cli_parse_link_choice_unknown_returns_err.rs"]
mod cli_parse_link_choice_unknown_returns_err;
#[path = "pyffi/once_lock_get_or_init_not_inside_parallel_regions.rs"]
mod once_lock_get_or_init_not_inside_parallel_regions;
#[path = "pyffi/pyffi_registration_coverage_bug.rs"]
mod pyffi_registration_coverage_bug;
#[path = "pyffi/python_rust_ffi_parity_gaussian_linear.rs"]
mod python_rust_ffi_parity_gaussian_linear;
#[path = "pyffi/stateful_saved_model_sync.rs"]
mod stateful_saved_model_sync;
#[path = "pyffi/warm_start_invariance_contract.rs"]
mod warm_start_invariance_contract;
