//! Reference-tool, CLI, and calibration test harnesses.
//!
//! These utilities consume design geometry and scalar diagnostics directly;
//! they do not depend on the model implementations they help test. This keeps
//! a model unit test from compiling the model crate a second time through a
//! test-support dependency cycle. Leaf fixtures remain with their owning
//! crates and are re-exported below for this harness's consumers.

pub mod calibration;
pub mod cli_harness;
pub mod reference;
pub mod synthetic;

// `no_densify_design` (and the operator-backed fixture behind it) is a
// linear-algebra fixture; it lives in `gam-linalg` alongside the operator traits
// it exercises and is re-exported here so this crate's consumers keep their
// familiar path. Single source of truth — the previous duplicate copy drifted
// out of the crate that owns the types.
pub use gam_linalg::test_support::no_densify_design;

// The stderr backend for production's `log::info!` diagnostics is `log` in,
// stderr out — it owns no model-layer type, so by the rule above it lives in
// `gam-runtime` (which already owns `span`/`process_monitor`/`loop_progress` and
// already depends on `log`) and is re-exported here. Without a backend installed
// the `log` facade DROPS every record, which is why the BMS intercept counters,
// the GL-ladder histogram and the certificate-bound discriminator have all been
// emitting into nothing in every test binary.
// `diagnostic_write_failures` travels with it: a backend that silently failed to
// write is exactly the "instrument that never ran produces evidence of absence"
// shape, and a caller that reads a run for its instrumentation needs to be able
// to assert the records reached the stream. Re-exporting the installer without
// the counter leaves that assertion one crate away from every test that needs it.
pub use gam_runtime::test_support::{diagnostic_write_failures, install_diagnostic_logger};

// Finite-difference derivative checking is `ndarray` in, `ndarray` out: it owns
// no model-layer type, so it lives in `gam-linalg`.
pub use gam_linalg::test_support::fd_checker;
pub use gam_linalg::test_support::fd_checker::{
    FdDerivative, RiddersConfig, assert_matrix_derivativefd, assert_matrix_derivativefd_rel,
    ridders_derivative, ridders_partial_derivative,
};

// `ParameterBlockSpec` fixtures live in `gam-problem`, the crate that owns the
// spec type.
pub use gam_problem::test_support::{
    BinomialLocationScaleBaseFixture, binomial_location_scale_base_fixture, spec_from_dense,
    spec_from_dense_with_priority,
};

// The central-difference macro is `ndarray`-only and expands `approx` at the
// call site, so it lives in `gam-linalg` with the rest of the FD harness.
pub use gam_linalg::assert_central_difference_array;
