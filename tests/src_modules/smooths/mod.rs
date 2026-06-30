mod basis_duchon_matern_jet_derivative_tests;
mod basis_radial_periodic_thinplate_tests;
mod smooth_adaptive_bounded_duchon_tests;
mod smooth_design_assembly_constraint_tests;
// #1274: `smooth_matern_nfree_rekey_topology_tests` re-homed into
// `crates/gam-models/src/fit_orchestration/drivers/matern_nfree_rekey_topology_tests.rs`
// (the gam-models-private `FrozenTermCollectionIncrementalRealizer` lives there
// post-#1521 carve; this `mod.rs` is itself not compiled anywhere). Removed here.
