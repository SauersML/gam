//! Caller-facing thin wrapper around `crate::gpu::arrow_schur`.
//!
//! The entire dense per-row factor + Schur reduce + back-sub pipeline lives
//! device-side; this module only translates the device failure enum into the
//! `ArrowSchurError` variant the PIRLS outer loop already understands, so
//! call-sites do not need to learn the device-specific reason codes.
//!
//! ## Dispatch logic for matrix-free systems
//!
//! When `solve_arrow_newton_step` returns `GpuRequiresDenseSystem`, the GPU
//! dense-Schur path is structurally incompatible with the supplied operators.
//! This wrapper routes such systems to CPU `InexactPCG` — the mode that was
//! designed precisely for SAE-manifold scale callers that cannot materialise
//! a dense `K × K` block. No information is lost: `GpuRequiresDenseSystem`
//! is not a numerical failure, just a capability mismatch, so the CPU solver
//! receives the full system without escalating any ridge.

use crate::gpu::arrow_schur::{ArrowSchurGpuFailure, solve_arrow_newton_step};
use crate::solver::arrow_schur::{
    ArrowSchurError, ArrowSchurSystem, ArrowSolveOptions, ArrowSolverMode,
};
use ndarray::Array1;

pub fn solve_arrow_newton_step_gpu(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
) -> Result<(Array1<f64>, Array1<f64>), ArrowSchurError> {
    match solve_arrow_newton_step(sys, ridge_t, ridge_beta) {
        Ok(solution) => Ok((solution.delta_t, solution.delta_beta)),
        Err(ArrowSchurGpuFailure::Unavailable) => {
            // Mirror the CPU path's failure variant so the outer loop falls
            // through to its existing recovery logic.
            sys.solve_with_options(
                ridge_t,
                ridge_beta,
                &ArrowSolveOptions::automatic(sys.k),
            )
            .map(|(dt, db, _diag)| (dt, db))
        }
        Err(ArrowSchurGpuFailure::GpuRequiresDenseSystem { .. }) => {
            // Matrix-free H_ββ or H_tβ operators present — the dense GPU
            // Schur path cannot consume them. Route to CPU InexactPCG which
            // was built for exactly this SAE-manifold scale use case. The GPU
            // PCG path (Part B of issue #288) will lift this at K ≥ 5000
            // once the row-procedural H_tβ kernel is wired.
            let mut opts = ArrowSolveOptions::automatic(sys.k);
            opts.mode = ArrowSolverMode::InexactPCG;
            sys.solve_with_options(ridge_t, ridge_beta, &opts)
                .map(|(dt, db, _diag)| (dt, db))
        }
        Err(ArrowSchurGpuFailure::RidgeBumpRequired { row, bump }) => {
            Err(ArrowSchurError::PerRowFactorFailed {
                row,
                reason: format!("GPU Cholesky factor failed; suggested ridge bump {bump:.3e}"),
            })
        }
        Err(ArrowSchurGpuFailure::SchurFactorFailed { reason }) => {
            Err(ArrowSchurError::SchurFactorFailed { reason })
        }
    }
}
