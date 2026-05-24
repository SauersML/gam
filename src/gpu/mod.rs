//! GPU acceleration backend entry points.
//!
//! This module is deliberately available in CPU-only builds. CUDA discovery is
//! dynamic and policy-gated, so the public fitting API can keep using the
//! existing CPU implementation when CUDA is unavailable, disabled, too small
//! for a kernel, or numerically unsuitable.

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

pub use device::{GpuCapability, GpuDeviceInfo, GpuSelection};
pub use policy::{AccelPolicy, GpuDispatchPolicy, Operation};
pub use runtime::{GpuRuntime, gpu_available, selected_gpu_info};
