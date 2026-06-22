//! Grouped integration-test crate root for pyffi tests (issue #1146).
//!
//! Former top-level crates included as modules so they link as ONE binary.

mod audit_priority_perm_invariance;
mod ban_gate_delegated_assertion;
mod canonicalize_routes_channel_aware_for_multi_channel;
mod cli_parse_link_choice_unknown_returns_err;
mod once_lock_get_or_init_not_inside_parallel_regions;
mod pyffi_registration_coverage_bug;
mod python_rust_ffi_parity_gaussian_linear;
mod stateful_saved_model_sync;
mod warm_start_invariance_contract;
