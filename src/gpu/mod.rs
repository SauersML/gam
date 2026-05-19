//! GPU acceleration for hot dense kernels.
//!
//! This module is the crate's GPU boundary:
//!
//! * [`runtime`] — one-shot, env-free autodetect of an installed CUDA driver
//!   via dynamic loading (`libloading`). Builds without a driver continue on
//!   the CPU path silently.
//! * [`device`] — host-visible description of the selected GPU, including
//!   measured SM count and FP64 throughput.
//! * [`policy`] — workload-size thresholds derived from that throughput plus
//!   the PCIe transfer cost.
//! * [`dispatch`] — public `try_fast_*` entry points used by CPU linalg
//!   call sites; route to `blas` / `solver` / `sparse` when the policy
//!   approves and fall through to CPU otherwise.
//!
//! The public crate API has no GPU configuration: detection is automatic
//! and behavior on hosts without a usable NVIDIA driver is unchanged.

mod blas;
pub mod calibration;
pub mod device;
pub mod diagnostics;
pub mod dispatch;
mod driver;
pub mod policy;
pub mod runtime;
pub mod session;
pub mod solver;
pub mod sparse;

pub use diagnostics::{flush_gpu_activity_summary, gpu_activity_summary};
pub use session::{DeviceXSession, try_fast_xt_diag_x_arc};

pub use device::GpuDeviceInfo;
pub use dispatch::{
    try_fast_a_broadcast_bt_batched, try_fast_ab, try_fast_ab_broadcast_b_batched,
    try_fast_ab_strided_batched, try_fast_abt_strided_batched, try_fast_atb,
    try_fast_atb_strided_batched, try_fast_atv, try_fast_av, try_fast_xt_diag_x,
    try_fast_xt_diag_y, try_solve_lower_triangular_matrix, try_solve_upper_triangular_matrix,
};
pub use policy::DispatchPolicy;
pub use runtime::{GpuProbeError, GpuRuntime, gpu_available, selected_gpu_info, warm};
pub use solver::{
    describe_chol_solve_route, try_chol_solve_inplace, try_cholesky_batched_lower_inplace,
    try_cholesky_lower_inplace, try_syevd_inplace, will_attempt_chol_solve,
};
pub use sparse::{try_csr_spmv_usize, try_csr_t_spmv_usize};
