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

use crate::gpu::kernels::arrow_schur::{
    ArrowSchurGpuFailure, gpu_schur_matvec_backend, solve_arrow_newton_step,
};
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
            sys.solve_with_options(ridge_t, ridge_beta, &ArrowSolveOptions::automatic(sys.k))
                .map(|(dt, db, _diag)| (dt, db))
        }
        Err(ArrowSchurGpuFailure::GpuRequiresDenseSystem { .. }) => {
            // Matrix-free H_ββ or H_tβ operators present — the dense GPU Schur
            // path cannot consume them, but the reduced K-system PCG can.
            // Build the GPU-backed reduced Schur matvec (row-procedural sparse
            // Kronecker apply over active atoms; per-row latent eliminated via
            // cached factors) and run `InexactPCG` against it. Only when the
            // device matvec is genuinely `Unavailable` do we fall back to the
            // pure-CPU `InexactPCG` matvec.
            let mut opts = ArrowSolveOptions::automatic(sys.k);
            opts.mode = ArrowSolverMode::InexactPCG;
            match gpu_schur_matvec_backend(sys, ridge_t, ridge_beta) {
                Ok(gpu_matvec) => {
                    opts.gpu_matvec = Some(gpu_matvec);
                }
                Err(ArrowSchurGpuFailure::Unavailable) => {
                    // No device matvec available; CPU InexactPCG owns the solve.
                }
                Err(ArrowSchurGpuFailure::RidgeBumpRequired { row, bump }) => {
                    return Err(ArrowSchurError::PerRowFactorFailed {
                        row,
                        reason: format!(
                            "GPU row-procedural factor failed; suggested ridge bump {bump:.3e}"
                        ),
                    });
                }
                Err(ArrowSchurGpuFailure::GpuRequiresDenseSystem { .. }) => {
                    // The matvec builder cannot lift this system either; CPU
                    // InexactPCG matvec handles the reduction.
                }
                Err(ArrowSchurGpuFailure::SchurFactorFailed { reason }) => {
                    return Err(ArrowSchurError::SchurFactorFailed { reason });
                }
            }
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
