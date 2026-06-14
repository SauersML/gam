# gam-pyffi Test Modules

Rust tests in this directory are compiled inside `crates/gam-pyffi` with
`#[cfg(test)]` and `#[path = ...]`, currently from
`manifold_and_posterior_ffi.rs`.

`lib_tests.rs` covers crate-internal FFI behavior that needs private helpers,
including shared-tangent Gaussian REML scale/lambda/frame-equivariance
regressions and saved-model payload version rejection.

Use this location only for tests that need access to private `gam-pyffi`
module state. Public API tests belong under the top-level `tests/` tree.
