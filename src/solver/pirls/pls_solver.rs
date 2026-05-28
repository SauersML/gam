//! Penalized least-squares solver and Gaussian fast paths.
//!
//! Conceptual home of:
//! - `solve_penalized_least_squares_implicit` — identity/Gaussian implicit PLS,
//!   dense and sparse-native paths.
//! - `GaussianFixedCache` — `XᵀWX`/`XᵀW(y−offset)` cache for the Gaussian-Identity
//!   short-circuit that the REML outer loop reuses across smoothing-parameter
//!   candidates.
//! - `SparseXtwxPrecomputed` — the sparse-pattern-aligned twin of the above for
//!   designs that take the sparse-native PIRLS path.
//!
//! These items remain defined in [`super`] (the `pirls` parent module) while the
//! file split is being introduced incrementally. This stub establishes the
//! directory entry; subsequent commits move the bodies here without altering the
//! public API: callers continue to reach `GaussianFixedCache` /
//! `SparseXtwxPrecomputed` via `crate::solver::pirls::*` exactly as before.

pub(super) use super::{
    GaussianFixedCache, SparseXtwxPrecomputed, solve_penalized_least_squares_implicit,
};
