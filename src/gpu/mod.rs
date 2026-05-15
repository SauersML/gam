//! GPU acceleration backend facade.
//!
//! The CPU implementation remains the reference path. This module provides the
//! runtime probe, dispatch policy, device-memory descriptors, and profiling
//! hooks that hot kernels use to decide when a CUDA implementation may be used.
//! When the `cuda` feature is disabled, or CUDA cannot be loaded at runtime,
//! every operation deterministically falls back to the existing CPU kernels.

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
pub use linalg::{GpuDenseKernel, try_dispatch_dense};
pub use policy::{AccelPolicy, GpuDispatchDecision, GpuDispatchPolicy, GpuOperation};
pub use profile::{KernelStat, record_cpu_kernel};
pub use runtime::{ExecutionTarget, GpuContext, GpuRuntime, gpu_available, selected_gpu_info};
