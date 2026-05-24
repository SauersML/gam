//! Placeholder module for the CUDA backend phase described in the GPU HAL.
//!
//! The public HAL is present so solver call sites can be routed without CUDA
//! types. Concrete cudarc kernels are intentionally isolated here in follow-up
//! implementations and unavailable backends fall back to CPU numerics.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendStatus {
    CpuFallback,
    CudaUnavailable,
    CudaReady,
}

pub fn backend_status() -> BackendStatus {
    BackendStatus::CpuFallback
}
