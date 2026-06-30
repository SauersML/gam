//! Unit / integration tests for the `smooth` term-collection module.
//!
//! The test bodies live in the shared `tests/src_modules/` fixtures and are
//! pulled in verbatim; every item they reference resolves through the parent
//! module's re-exports (`use super::*`).

// include!("../../../../tests/src_modules/smooths/smooth_design_assembly_constraint_tests.rs");
// include!("../../../../tests/src_modules/smooths/smooth_adaptive_bounded_duchon_tests.rs");
// #1274: the Matérn n-free re-key topology tests were re-homed into
// `gam-models` (`fit_orchestration/drivers/matern_nfree_rekey_topology_tests.rs`)
// — they need the gam-models-private `FrozenTermCollectionIncrementalRealizer`
// that gam-terms cannot see, so they never compiled here. Reference removed.
