//! Caller-facing thin wrapper around `crate::gpu_kernels::arrow_schur`.
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

use crate::arrow_schur::{ArrowSchurError, ArrowSchurSystem, ArrowSolveOptions, ArrowSolverMode};
use crate::gpu_kernels::arrow_schur::{
    ArrowSchurGpuFailure, gpu_schur_matvec_backend, solve_arrow_newton_step,
};
use ndarray::Array1;

