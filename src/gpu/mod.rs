//! GPU acceleration facade.
//!
//! The public fitting API remains CPU-first and unchanged.  This module owns
//! all CUDA probing, policy, profiling, and dispatch plumbing so hot paths can
//! ask for an operation-specific acceleration decision without depending on a
//! concrete CUDA installation.  CPU-only builds compile without CUDA libraries;
//! `--features cuda` enables the optional CUDA dependency boundary for future
//! device kernels while preserving the same safe fallback behaviour.

pub mod device;
pub mod kernels;
pub mod linalg;
pub mod memory;
pub mod policy;
pub mod profile;
pub mod runtime;
pub mod solver;
pub mod sparse;
pub mod stream;

pub use device::{GpuCapability, GpuDeviceInfo};
pub use policy::{AccelPolicy, GpuDispatchPolicy, Operation};
pub use profile::{KernelStat, ProfileSnapshot};
pub use runtime::{GpuRuntime, selected_gpu_info};

/// Returns true when GPU execution is both requested and a CUDA-capable device
/// has been discovered.
#[inline]
pub fn gpu_available() -> bool {
    runtime::global_runtime().is_gpu_enabled()
}

/// Returns true when profiling is enabled through `GAM_GPU_PROFILE=1`.
#[inline]
pub fn profiling_enabled() -> bool {
    runtime::global_runtime().profile_enabled()
}
