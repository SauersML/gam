//! Device-resident dispatcher for the cubic-cell derivative-moment substrate.
//!
//! Every cell is classified exactly once by the canonical CPU predicate, then
//! the all-branch NVRTC kernel evaluates affine, non-affine finite, and affine
//! tail cells into one device-resident `[n_cells, max_degree+1]` buffer. There
//! is no selected-device-to-host fallback. Invalid cells receive zeroed rows and
//! a typed [`super::CubicCellMomentStatus`].

#[cfg(target_os = "linux")]
use crate::gpu_kernels::cubic_cell::{
    CubicCellDerivativeMomentHostView, CubicCellDerivativeMomentOutput, CubicCellMomentStatus,
    GpuCellBranchTag, branch::classify_cell_for_gpu,
};
#[cfg(target_os = "linux")]
use gam_gpu::gpu_err;
#[cfg(target_os = "linux")]
use gam_gpu::gpu_error::GpuError;
#[cfg(target_os = "linux")]
use gam_gpu::gpu_error::GpuResultExt;

#[cfg(target_os = "linux")]
use std::sync::{Arc, Mutex, OnceLock};

#[cfg(target_os = "linux")]
use cudarc::driver::{CudaContext, CudaModule, CudaStream};

/// Linux-only: launch the device dispatcher and return the moments buffer in
/// device memory (`CudaSlice<f64>`). The caller reaches this boundary only
/// after selecting device execution, so backend absence and driver/NVRTC/shape
/// failures are all typed errors; this function never substitutes host work.
#[cfg(target_os = "linux")]
pub(crate) fn build_device_moments_resident(
    view: &CubicCellDerivativeMomentHostView<'_>,
) -> Result<CubicCellDerivativeMomentOutput, GpuError> {
    let backend = CubicCellGpuBackend::probe()?;
    backend.dispatch_device_resident(view)
}

/// Process-wide cubic-cell GPU backend. Mirrors the
/// `BmsFlexGpuBackend` / `SurvivalFlexGpuBackend` shape so future
/// device-residency residencies can swap in without churn. Linux-only:
/// non-Linux builds skip [`build_device_moments_resident`] at the call
/// site (`super::try_build_cubic_cell_derivative_moments`) via
/// `#[cfg(target_os = "linux")]`, so this backend is never referenced.
#[cfg(target_os = "linux")]
#[must_use]
pub(crate) struct CubicCellGpuBackend {
    inner: CubicCellGpuContextLinux,
}

#[cfg(target_os = "linux")]
struct CubicCellGpuContextLinux {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    /// NVRTC-compiled module per `max_degree` specialization. Keyed by
    /// `MOMENT_STRIDE = max_degree + 1` so a single integer suffices.
    modules: Mutex<std::collections::HashMap<usize, Arc<CudaModule>>>,
}

#[cfg(target_os = "linux")]
impl CubicCellGpuBackend {
    /// Lazily initialise the process-wide backend. First-call NVRTC-compile
    /// of the kernel module is deferred to dispatch (each `max_degree`
    /// specialization compiles on first use, cached forever).
    pub(crate) fn probe() -> Result<&'static Self, GpuError> {
        static BACKEND: OnceLock<Result<CubicCellGpuBackend, GpuError>> = OnceLock::new();
        BACKEND
            .get_or_init(Self::probe_linux)
            .as_ref()
            .map_err(GpuError::clone)
    }

    #[cfg(target_os = "linux")]
    fn probe_linux() -> Result<Self, GpuError> {
        let parts = gam_gpu::backend_probe::probe_cuda_backend("cubic_cell")?;
        Ok(CubicCellGpuBackend {
            inner: CubicCellGpuContextLinux {
                ctx: parts.ctx,
                stream: parts.stream,
                modules: Mutex::new(std::collections::HashMap::new()),
            },
        })
    }

    /// NVRTC-compile and load (or fetch from cache) the kernel module for
    /// `max_degree`.
    #[cfg(target_os = "linux")]
    fn module_for_degree(&self, max_degree: usize) -> Result<Arc<CudaModule>, GpuError> {
        let key = max_degree;
        {
            let guard = self
                .inner
                .modules
                .lock()
                .gpu_ctx("cubic_cell module cache mutex poisoned")?;
            if let Some(module) = guard.get(&key) {
                return Ok(Arc::clone(module));
            }
        }
        let source =
            crate::gpu_kernels::cubic_cell::kernel_src::build_cubic_deriv_moments_kernel_source(
                max_degree,
            );
        // Route through the shared arch-aware NVRTC compile (#1551), NOT the bare
        // `cudarc::nvrtc::compile_ptx`. That sets the device-keyed `--gpu-architecture`
        // pin AND the NVRTC include search paths (`/usr/local/cuda/include`, …).
        // The bare path supplies no `-I`, so this kernel's `#include <stdint.h>`
        // failed with "catastrophic error: could not open source file stdint.h"
        // and the device path silently fell back to the CPU on every GPU box.
        let ptx = gam_gpu::device_cache::compile_ptx_arch(&source).gpu_ctx_with(|err| {
            format!("cubic_cell NVRTC compile (degree={max_degree}) failed: {err}")
        })?;
        let module = self.inner.ctx.load_module(ptx).gpu_ctx_with(|err| {
            format!("cubic_cell module load (degree={max_degree}) failed: {err}")
        })?;
        let mut guard = self
            .inner
            .modules
            .lock()
            .gpu_ctx("cubic_cell module cache mutex poisoned")?;
        let entry = guard.entry(key).or_insert(module);
        Ok(Arc::clone(entry))
    }

    /// Device-resident dispatcher: leaves the moments + status buffers on
    /// the GPU. Stage-4 strategy: route **all three**
    /// branches through the single NVRTC kernel (which already covers
    /// Affine, NonAffineFinite, and AffineTail in closed form) so the
    /// output is naturally `[n_cells, stride]` indexed by original cell
    /// index — no host-side scatter required.
    ///
    /// The host CPU classifier still runs to assign branch codes and reject
    /// cells the kernel can't handle (`InvalidInterval`, `NonAffineInfiniteInterval`,
    /// `NonFiniteCoefficient`); rejected cells get host status codes and the
    /// kernel is fed a placeholder cell whose row stays zero.
    ///
    /// The returned `CudaSlice<f64>` is allocated on the cubic-cell
    /// backend's default stream — which (because `gam_gpu::device_runtime::cuda_context_for`
    /// caches one `CudaContext` per device ordinal) is the same default
    /// stream every other gam GPU backend uses on the same device, so
    /// downstream kernels can consume the slice without any cross-context
    /// copying.
    #[cfg(target_os = "linux")]
    fn dispatch_device_resident(
        &self,
        view: &CubicCellDerivativeMomentHostView<'_>,
    ) -> Result<CubicCellDerivativeMomentOutput, GpuError> {
        use cudarc::driver::{LaunchConfig, PushKernelArg};

        let n_cells = view.cells.len();
        let stride = view.max_degree + 1;
        assert!(n_cells > 0, "caller must guard empty views");

        // ---- Run the host classifier so cells the kernel can't handle
        //      (genuinely degenerate intervals, non-finite coefficients,
        //      non-affine infinite intervals) get host status codes and a
        //      placeholder branch the kernel will reject. This mirrors the
        //      `dispatch` host-resident path's classifier behavior.
        let mut status_host = vec![CubicCellMomentStatus::Ok; n_cells];
        // Branch code per cell for the kernel:
        // BRANCH_AFFINE = 0, BRANCH_NONAFFINE_FIN = 1, BRANCH_AFFINE_TAIL = 2.
        // 255 marks "classifier-rejected" — the kernel's lane-0 validator
        // falls into the trailing `else { local_status = STATUS_INVALID; }`
        // branch on any unrecognized code, which zeros the row + writes
        // STATUS_INVALID; we then overwrite with the real classifier code
        // below.
        let mut branch_code = vec![255_u8; n_cells];
        let mut left = vec![0.0_f64; n_cells];
        let mut right = vec![0.0_f64; n_cells];
        let mut c0 = vec![0.0_f64; n_cells];
        let mut c1 = vec![0.0_f64; n_cells];
        let mut c2 = vec![0.0_f64; n_cells];
        let mut c3 = vec![0.0_f64; n_cells];
        for (i, &gpu_cell) in view.cells.iter().enumerate() {
            left[i] = gpu_cell.left;
            right[i] = gpu_cell.right;
            c0[i] = gpu_cell.c0;
            c1[i] = gpu_cell.c1;
            c2[i] = gpu_cell.c2;
            c3[i] = gpu_cell.c3;
            match classify_cell_for_gpu(gpu_cell) {
                Ok(host_tag) => {
                    branch_code[i] = match host_tag {
                        GpuCellBranchTag::Affine => 0,
                        GpuCellBranchTag::NonAffineFinite => 1,
                        GpuCellBranchTag::AffineTail => 2,
                    };
                }
                Err(code) => {
                    status_host[i] = code;
                }
            }
        }

        // ---- Allocate device buffers + launch the kernel against the full
        //      `[n_cells, stride]` layout indexed by original cell index.
        let max_degree = view.max_degree;
        let module = self.module_for_degree(max_degree)?;
        let kernel_name = format!("cubic_deriv_moments_d{max_degree}");
        let func = module
            .load_function(&kernel_name)
            .gpu_ctx_with(|err| format!("cubic_cell load_function {kernel_name}: {err}"))?;

        let stream = &self.inner.stream;
        let d_left = stream
            .clone_htod(&left)
            .gpu_ctx("cubic_cell device-resident memcpy left")?;
        let d_right = stream
            .clone_htod(&right)
            .gpu_ctx("cubic_cell device-resident memcpy right")?;
        let d_c0 = stream
            .clone_htod(&c0)
            .gpu_ctx("cubic_cell device-resident memcpy c0")?;
        let d_c1 = stream
            .clone_htod(&c1)
            .gpu_ctx("cubic_cell device-resident memcpy c1")?;
        let d_c2 = stream
            .clone_htod(&c2)
            .gpu_ctx("cubic_cell device-resident memcpy c2")?;
        let d_c3 = stream
            .clone_htod(&c3)
            .gpu_ctx("cubic_cell device-resident memcpy c3")?;
        let d_branch = stream
            .clone_htod(&branch_code)
            .gpu_ctx("cubic_cell device-resident memcpy branch")?;
        let mut d_moments = stream
            .alloc_zeros::<f64>(n_cells * stride)
            .map_err(|err| gpu_err!("cubic_cell device-resident alloc moments: {err}"))?;
        let mut d_status = stream
            .alloc_zeros::<u8>(n_cells)
            .gpu_ctx("cubic_cell device-resident alloc status")?;

        let warps_per_block: u32 = 4;
        let block: u32 = 32 * warps_per_block;
        let n_u32: u32 = u32::try_from(n_cells)
            .map_err(|_| gpu_err!("cubic_cell n_cells={n_cells} overflows u32"))?;
        let grid: u32 = n_u32.div_ceil(warps_per_block).max(1);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_cells_u32 = n_u32;
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&d_left)
            .arg(&d_right)
            .arg(&d_c0)
            .arg(&d_c1)
            .arg(&d_c2)
            .arg(&d_c3)
            .arg(&d_branch)
            .arg(&mut d_moments)
            .arg(&mut d_status)
            .arg(&n_cells_u32);
        // SAFETY: every kernel argument is a typed device pointer / scalar
        // matching the kernel signature above; the grid covers exactly
        // `n_cells` warps; out-of-range warps early-return. The kernel's
        // lane-0 validator rejects unrecognized branch codes (255 sentinel)
        // by zeroing the row and writing STATUS_INVALID, so
        // classifier-rejected slots are safe.
        unsafe { builder.launch(cfg) }.gpu_ctx("cubic_cell device-resident kernel launch")?;

        // Read back per-cell statuses so the host can:
        //   (a) merge with classifier-rejected entries it already knows
        //       (those use the classifier's specific status code, not
        //        the kernel's catch-all STATUS_INVALID),
        //   (b) decode the kernel ABI bytes exactly once into typed statuses.
        let kernel_status = stream
            .clone_dtoh(&d_status)
            .gpu_ctx("cubic_cell device-resident DtoH status")?;
        stream
            .synchronize()
            .gpu_ctx("cubic_cell device-resident sync after kernel")?;

        // Merge: if the classifier already rejected a cell, its specific
        // code wins (the kernel's row for that cell was zeroed by the
        // STATUS_INVALID path so the device buffer is already correct).
        // Otherwise take the kernel's status verbatim.
        for i in 0..n_cells {
            if status_host[i] == CubicCellMomentStatus::Ok {
                status_host[i] = CubicCellMomentStatus::from_device_code(kernel_status[i])?;
            }
        }
        drop(d_status);
        Ok(CubicCellDerivativeMomentOutput {
            d_moments,
            status: status_host,
            stride,
            n_cells,
        })
    }
}

