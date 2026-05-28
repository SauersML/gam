//! Stage 3.3 GPU dispatch glue that lives on the PIRLS side of the boundary.
//!
//! Conceptual home of:
//! - The Gaussian-Identity exact PLS GPU dispatch shim (`try_gpu_gaussian_pls_dispatch`
//!   call site) wired into `fit_model_for_fixed_rho_with_adaptive_kkt` for
//!   Gaussian-Identity fits with a populated `GaussianFixedCache`.
//! - The general PIRLS-loop GPU dispatch shim (`try_gpu_pirls_loop_admit` /
//!   `try_gpu_pirls_loop_dispatch` call site) for dense designs on the CPU LM
//!   path that admit the device-resident loop.
//! - Local Linux-gated wrappers that decide whether to ship the row-kernel work
//!   to CUDA before paying for host LM iteration.
//!
//! The GPU **kernel** bodies live in `crate::solver::gpu::pirls_gpu` /
//! `crate::solver::gpu::pirls_dispatch_wire`; this file only owns the host-side
//! dispatch glue that decides when to call them.
//!
//! The current implementations remain inline in [`super::fit_model_for_fixed_rho_with_adaptive_kkt`]
//! while the file split is being introduced incrementally. This stub establishes
//! the directory entry; subsequent commits factor the two `#[cfg(target_os = "linux")]`
//! dispatch blocks out into private helpers re-exported below, with no change to
//! the public API.
