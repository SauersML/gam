//! GPU acceleration scaffolding.
//!
//! This module is the crate's GPU boundary. It is split into a small set of
//! concerns that future device backends (cuBLAS / cuSPARSE / cuSOLVER and
//! peers) can implement without changing solver call sites:
//!
//! * [`runtime`] — one-shot, env-free autodetect of an installed CUDA driver
//!   via dynamic loading (`libloading`). Builds without a driver continue on
//!   the CPU path silently.
//! * [`device`] — host-visible description of the selected GPU.
//! * [`policy`] — workload-size thresholds derived from device capability.
//! * [`memory`] — host-shadow device buffer / matrix / sparse matrix types
//!   used to express device-resident state in trait contracts.
//! * [`traits`] — `DeviceBlas` / `DeviceSolver` / `DeviceDesignOperator`
//!   surfaces the future CUDA backend must implement.
//! * [`profile`] — `cpu_scope` wraps hot CPU kernels with non-invasive
//!   timing so policy decisions and benchmarks can read the actual cost.
//! * [`kernels`] — bit-checkable host reference implementations of the
//!   numerical contracts a device backend must reproduce.
//!
//! The public crate API has no GPU configuration: detection is automatic, the
//! CPU execution is unconditional on hosts without CUDA, and the user-visible behavior is
//! unchanged on hosts without a usable NVIDIA driver.

mod blas;
pub mod device;
pub mod dispatch;
pub mod kernels;
pub mod memory;
pub mod policy;
pub mod profile;
pub mod runtime;
pub mod traits;

pub use device::{GpuCapability, GpuDeviceInfo};
pub use dispatch::{
    try_fast_ab, try_fast_atb, try_fast_atv, try_fast_av, try_fast_xt_diag_x, try_fast_xt_diag_y,
};
pub use memory::{DeviceBuffer, DeviceCsrMatrix, DeviceMatrix, DeviceVector, MatrixLocation};
pub use policy::DispatchPolicy;
pub use profile::{KernelStat, KernelStatsSnapshot, cpu_scope};
pub use runtime::{GpuProbeError, GpuRuntime, gpu_available, selected_gpu_info};
pub use traits::{DeviceBlas, DeviceDesignOperator, DeviceSolver, ExecutionTarget};
