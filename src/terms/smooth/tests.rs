//! Unit / integration tests for the `smooth` term-collection module.
//!
//! The test bodies live in the shared `tests/src_modules/` fixtures and are
//! pulled in verbatim; every item they reference resolves through the parent
//! module's re-exports (`use super::*`).

include!("../../../tests/src_modules/smooth_design_assembly_constraint_tests.rs");
include!("../../../tests/src_modules/smooth_adaptive_bounded_duchon_tests.rs");
