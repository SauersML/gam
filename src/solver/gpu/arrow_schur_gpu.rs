//! Caller-facing thin wrapper around `crate::gpu::arrow_schur`.
//!
//! The entire dense per-row factor + Schur reduce + back-sub pipeline lives
//! device-side; this module only translates the device failure enum into the
//! `ArrowSchurError` variant the PIRLS outer loop already understands, so
//! call-sites do not need to learn the device-specific reason codes.

use crate::gpu::arrow_schur::{
    ArrowSchurGpuFailure, solve_arrow_newton_step,
};
use crate::solver::arrow_schur::{ArrowSchurError, ArrowSchurSystem};
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
                &crate::solver::arrow_schur::ArrowSolveOptions::automatic(sys.k),
            )
        }
        Err(ArrowSchurGpuFailure::RidgeBumpRequired { row, bump }) => {
            Err(ArrowSchurError::PerRowFactorFailed {
                row,
                reason: format!(
                    "GPU Cholesky factor failed; suggested ridge bump {bump:.3e}"
                ),
            })
        }
        Err(ArrowSchurGpuFailure::SchurFactorFailed { reason }) => {
            Err(ArrowSchurError::SchurFactorFailed { reason })
        }
    }
}
