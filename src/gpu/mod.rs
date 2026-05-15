//! GPU acceleration facade.
//!
//! The module is intentionally always compiled so CPU-only builds can expose a
//! stable API and deterministic fallback behavior. CUDA-specific crates remain
//! behind the optional `cuda` Cargo feature and runtime policy defaults to CPU
//! unless a usable device is detected and the operation is large enough.

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
pub use policy::{AccelPolicy, GpuDispatchPolicy, Operation, OperationDecision};
pub use runtime::{ExecutionTarget, GpuContext, GpuRuntime, gpu_available, selected_gpu_info};
