//! GPU acceleration hardware-abstraction layer.
//!
//! The module is intentionally callable from CPU-only builds: all public entry
//! points are available without CUDA, and the runtime reports an unavailable
//! backend instead of changing numerical results.  CUDA-specific code is kept
//! behind the `cuda` Cargo feature and does not leak into solver modules.

pub mod blas;
pub mod cpu_traits;
pub mod device;
pub mod graph;
pub mod memory;
pub mod policy;
pub mod profile;
pub mod rand;
pub mod runtime;
pub mod solver;
pub mod sparse;
pub mod stream;
pub mod validate;

pub mod kernels {
    pub mod cell_moments;
    pub mod fused_xtwx;
    pub mod hutchpp;
    pub mod irls_link;
    pub mod reductions;
    pub mod row_scale;
    pub mod spatial;
}

pub use cpu_traits::{
    DeviceBlas, DeviceDesignOperator, DeviceSolver, ExecutionTarget, MatrixLocation,
};
pub use device::{GpuCapability, GpuDeviceInfo};
pub use memory::{DeviceBuffer, DeviceCsrMatrix, DeviceMatrix, DeviceVector};
pub use policy::{GpuDispatchPolicy, MixedPrecisionPolicy};
pub use profile::{KernelStat, KernelStatsSnapshot};
pub use runtime::{GpuProbeError, GpuRuntime};
