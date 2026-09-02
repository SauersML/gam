//! Gate for the deflation-aware β-Schur selected inverse
//! (`ArrowFactorCache::schur_inverse_apply_deflated` /
//! `schur_inverse_block_deflated`) — the shared primitive the λ→0 REML EDF fix
//! and the Path-C log-det HVP both consume.
//!
//! Two arms:
//!   (a) INTERIOR — a well-conditioned S_β: no eigen-direction sits below the
//!       canonical rank floor, so zero deflation fires and the deflated selected
//!       inverse reproduces the plain one to round-off (no silent bias).
//!   (b) BOUNDARY — an S_β with one doubly-null (data-null AND penalty-null)
//!       curvature direction: the PLAIN selected inverse divides by the ~zero
//!       pivot and blows up (the exact λ→0 EDF divergence); the DEFLATED one
//!       drops that direction and stays finite and bounded, equal to the
//!       pseudo-inverse over the kept subspace.
//!
//! Lives as a standalone integration test (only the gam-solve public API +
//! this one binary) so it is immune to unrelated tears in the crate's `#[cfg(test)]`
//! unit-test modules.

