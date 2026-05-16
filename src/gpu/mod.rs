//! GPU acceleration for hot dense kernels.
//!
//! This module is the crate's GPU boundary:
//!
//! * [`runtime`] — one-shot, env-free autodetect of an installed CUDA driver
//!   via dynamic loading (`libloading`). Builds without a driver continue on
//!   the CPU path silently.
//! * [`device`] — host-visible description of the selected GPU.
//! * [`policy`] — workload-size thresholds derived from device capability.
//! * [`dispatch`] — public hooks used by CPU linalg call sites.
//! * [`solver`] — cuSOLVER routing for large dense symmetric eigensystems.
//! * [`kernels`] — bit-checkable host reference implementations of the
//!   numerical contracts a device backend must reproduce.
//!
//! The public crate API has no GPU configuration: detection is automatic, the
//! CPU execution is unconditional on hosts without CUDA, and the user-visible behavior is
//! unchanged on hosts without a usable NVIDIA driver.

mod blas;
pub mod device;
pub mod dispatch;
mod driver;
pub mod kernels;
pub mod policy;
pub mod runtime;
pub mod solver;
pub mod sparse;

pub use device::GpuDeviceInfo;
pub use dispatch::{
    try_fast_ab, try_fast_atb, try_fast_atv, try_fast_av, try_fast_xt_diag_x, try_fast_xt_diag_y,
};
pub use policy::DispatchPolicy;
pub use runtime::{GpuProbeError, GpuRuntime, gpu_available, selected_gpu_info};
pub use solver::{try_chol_solve_inplace, try_syevd_inplace};
pub use sparse::{try_csr_spmv_usize, try_csr_t_spmv_usize};
