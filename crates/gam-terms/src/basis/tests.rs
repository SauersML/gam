//! Unit / integration tests for the `basis` module.
//!
//! The test bodies live in the shared `tests/src_modules/` fixtures and are
//! pulled in verbatim; every item they reference resolves through the parent
//! module's re-exports (`use super::*`).

include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/src_modules/smooths/basis_radial_periodic_thinplate_tests.rs"
));
include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/src_modules/smooths/basis_duchon_matern_jet_derivative_tests.rs"
));
