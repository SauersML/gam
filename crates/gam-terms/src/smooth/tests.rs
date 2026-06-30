//! Unit / integration tests for the `smooth` term-collection module.
//!
//! #1601 RESOLVED: the three smooth-term test fixtures that previously lived in
//! `tests/src_modules/smooths/` and were `include!`'d here all referenced
//! `crate::solver::*` / `crate::estimate::*` paths that only exist once you are
//! ABOVE gam-terms in the dependency graph. gam-solve and gam-models depend on
//! gam-terms, so those bodies could never compile in `gam-terms --lib` tests —
//! which is why #1601 (commit 28bab3753) commented the `include!`s out to unbreak
//! the build. The "preserved for relocation" promise was never kept: the parked
//! `tests/src_modules/` tree was `mod`'d into no test binary, so the guards ran
//! NOWHERE.
//!
//! They have now been re-homed into the gam-models drivers test tree, where their
//! `ExternalJointHyperEvaluator` / design-build / freeze dependencies resolve
//! after the #1521 crate carve, and the stale source copies were deleted:
//!   - smooth_design_assembly_constraint_tests   -> gam-models …/drivers/design_assembly_constraint_tests.rs
//!   - smooth_adaptive_bounded_duchon_tests       -> gam-models …/drivers/adaptive_bounded_duchon_tests.rs
//!   - smooth_matern_nfree_rekey_topology_tests   -> gam-models …/drivers/matern_nfree_rekey_topology_tests.rs
//!
//! The genuinely gam-terms-local smooth unit tests continue to live alongside the
//! code they exercise (e.g. `smooth/term_design.rs`, `smooth/term_specs.rs`).
