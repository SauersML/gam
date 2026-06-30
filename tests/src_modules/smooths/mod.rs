// NOTE: this mod.rs is not itself `mod`'d into any binary; the two basis
// fixtures below are pulled in directly via `include!` from
// `gam-terms/src/basis/tests.rs`. The three `smooth_*` fixtures that used to be
// listed here were re-homed under #1601 into the gam-models drivers test tree
// (`design_assembly_constraint_tests.rs`, `adaptive_bounded_duchon_tests.rs`,
// `matern_nfree_rekey_topology_tests.rs`) where their cross-crate deps resolve,
// and the dead copies were deleted.
mod basis_duchon_matern_jet_derivative_tests;
mod basis_radial_periodic_thinplate_tests;
