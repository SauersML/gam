//! Test oracle for the cubic-cell derivative-moment substrate.
//!
//! This module is compiled only by the parent module's test configuration. It
//! produces the same row-major `[n_cells, max_degree+1]` moment layout as the
//! device kernel, without creating a production CPU fallback for a selected
//! device route.
//!
//! The host path defers all heavy math to the existing CPU evaluator
//! `crate::cubic_cell_kernel::evaluate_cell_derivative_moments_uncached`.
//! This module's only jobs are:
//!
//! 1. validate the supported degree,
//! 2. classify each cell through the same canonical predicate as production,
//! 3. pack moments into the GPU-shaped output buffer with the agreed stride,
//! 4. record one status per cell so the caller can react to per-cell
//!    failures without having to re-run the CPU classifier.
//!
//! The moment-emitting path below is the numerical parity oracle.

