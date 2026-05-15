//! GPU acceleration HAL for `gam`.
//!
//! The module is intentionally backend-contained: solver and linalg code talk
//! to the traits and policy types exported here, while cudarc-backed execution
//! remains behind the `cuda` Cargo feature.  In the initial implementation all
//! operations are routed through the CPU fallback unless a future backend marks
//! the operation as supported by the active [`GpuRuntime`].

pub mod blas;
pub mod cpu_traits;
pub mod device;
pub mod graph;
pub mod kernels;
pub mod memory;
pub mod policy;
pub mod profile;
pub mod rand;
pub mod runtime;
pub mod solver;
pub mod sparse;
pub mod stream;
pub mod validate;

pub use cpu_traits::{
    DeviceBlas, DeviceDesignOperator, DeviceSolver, ExecutionTarget, MatrixLocation,
};
pub use device::{GpuCapability, GpuDeviceInfo};
pub use memory::{DeviceBuffer, DeviceCsrMatrix, DeviceMatrix, DeviceVector, GpuFitSession};
pub use policy::{GpuBackendDecision, GpuDispatchPolicy, GpuEnv, GpuOpKind};
pub use profile::{GpuProfile, KernelStat, record_cpu_kernel};
pub use runtime::{GpuRuntime, GpuRuntimeStatus};
