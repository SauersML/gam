//! Shared test fixtures for `gam_mpd` (#2951).
//!
#![cfg(test)]

use gam_linalg::faer_ndarray::FaerQr;
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

// The planted known-answer toys.
pub mod planted_toys;

// Hand-built networks with a known decomposition: induction, modular addition, residual MLPs.
pub mod known_answer_toys;

/// A random orthonormal basis: Householder QR of a seeded uniform draw.
pub fn hidden_basis(dimension: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let draw =
        Array2::<f64>::from_shape_fn((dimension, dimension), |_| rng.random_range(-1.0..1.0));
    let (q, _) = draw.qr().expect("Householder QR of a uniform draw");
    q
}

/// The ledger a test's kernels reserve on when the test does not assert on
/// reservations: a private governor, so no test draws on or reads the process-wide
/// ledger. Its budget, `2^34` bytes, is far above any fixture's footprint and far
/// below what a planted refusal requests; a test that asserts on its reservations
/// builds its own `MemoryGovernor::with_budget_bytes` instead.
pub fn test_governor() -> &'static gam_runtime::resource::MemoryGovernor {
    static GOVERNOR: std::sync::OnceLock<gam_runtime::resource::MemoryGovernor> = std::sync::OnceLock::new();
    GOVERNOR.get_or_init(|| gam_runtime::resource::MemoryGovernor::with_budget_bytes(1 << 34))
}
