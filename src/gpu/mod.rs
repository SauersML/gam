//! GPU acceleration hardware-abstraction layer.
//!
//! The module is intentionally a thin, CPU-safe façade.  The `cuda` Cargo
//! feature pulls in cudarc with dynamic loading, but all public entry points are
//! allowed to return [`GpuUnavailable`] and fall back to the existing CPU path.
//! This keeps default builds CUDA-free while giving solver/linalg call sites a
//! stable routing contract for the device-resident pipeline.

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

pub use cpu_traits::{ExecutionTarget, MatrixLocation};
pub use device::{GpuCapability, GpuDeviceInfo};
pub use memory::{DeviceBuffer, DeviceCsrMatrix, DeviceMatrix, DeviceVector, GpuFitSession};
pub use policy::{GpuDispatchPolicy, GpuEnv, MixedPrecisionMode, ValidationMode};
pub use profile::{KernelStat, OperationKind, record_cpu_fallback};
pub use runtime::{GpuProbeStatus, GpuRuntime};
