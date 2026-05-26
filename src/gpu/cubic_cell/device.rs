//! Device-resident dispatcher for the cubic-cell derivative-moment substrate.
//!
//! Stage-1 scope:
//!
//! * **NonAffineFinite** cells are evaluated on the GPU by NVRTC-compiling the
//!   384-point Gauss–Legendre kernel emitted by [`super::kernel_src`] (the
//!   `cubic_deriv_moments_d{degree}` specialization). One warp processes one
//!   cell; results land in the same row-major `[n_cells, max_degree+1]` host
//!   buffer the host substrate produces.
//! * **Affine** and **AffineTail** cells stay on CPU for Stage-1: the device
//!   kernel already contains closed-form branches for them, but Stage-1 keeps
//!   the dispatcher conservative and bit-equal to the CPU parity reference for
//!   those buckets. The closed-form device port lands with Stage-2.
//!
//! No silent fallback: cells are pre-bucketed on the host. The GPU launch is
//! issued only on the NonAffineFinite bucket; CPU evaluation is issued only on
//! the Affine / AffineTail buckets. Both contribute to the same host output
//! buffer. Cells with non-OK classifier status receive zeroed rows and the
//! corresponding [`super::CubicCellMomentStatus`] code, exactly like the host
//! substrate.

use crate::gpu::cubic_cell::host_substrate::HostMomentBatch;
#[cfg(target_os = "linux")]
use crate::gpu::cubic_cell::{
    CubicCellDerivativeMomentOutput, CubicCellMomentStatus, GpuCellBranchTag,
    branch::classify_cell_for_gpu,
};
use crate::gpu::cubic_cell::CubicCellDerivativeMomentHostView;
use crate::gpu::error::GpuError;


#[cfg(target_os = "linux")]
use std::sync::{Arc, Mutex, OnceLock};

#[cfg(target_os = "linux")]
use cudarc::driver::{CudaContext, CudaModule, CudaStream};

/// Try to dispatch the substrate through the GPU.
///
/// On Linux+CUDA with a usable runtime this returns `Ok(Some(_))` after
/// launching the NonAffineFinite kernel and CPU-evaluating the affine
/// buckets. Returns `Ok(None)` when no GPU runtime is available (caller
/// should fall back to the host substrate). Returns `Err` on a genuine
/// driver / NVRTC / shape failure that the caller must surface.
#[cfg(target_os = "linux")]
pub(crate) fn try_device_moments(
    view: &CubicCellDerivativeMomentHostView<'_>,
) -> Result<Option<HostMomentBatch>, GpuError> {
    let backend = match CubicCellGpuBackend::probe() {
        Ok(b) => b,
        Err(GpuError::DriverLibraryUnavailable { .. }) => return Ok(None),
        Err(other) => return Err(other),
    };
    backend.dispatch(view).map(Some)
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn try_device_moments(
    _view: &CubicCellDerivativeMomentHostView<'_>,
) -> Result<Option<HostMomentBatch>, GpuError> {
    Ok(None)
}

/// Linux-only: launch the same Stage-1 dispatcher but return the moments
/// buffer in device memory (`CudaSlice<f64>`) instead of downloading to host.
/// Caller may pass the returned slice directly to any kernel launch on the
/// same default stream (e.g. `bms_flex_row_kernel`). Returns `Ok(None)` if no
/// CUDA runtime is available, just like [`try_device_moments`].
#[cfg(target_os = "linux")]
pub(crate) fn try_device_moments_resident(
    view: &CubicCellDerivativeMomentHostView<'_>,
) -> Result<Option<CubicCellDerivativeMomentOutput>, GpuError> {
    let backend = match CubicCellGpuBackend::probe() {
        Ok(b) => b,
        Err(GpuError::DriverLibraryUnavailable { .. }) => return Ok(None),
        Err(other) => return Err(other),
    };
    backend.dispatch_device_resident(view).map(Some)
}

/// Process-wide cubic-cell GPU backend. Mirrors the
/// `BmsFlexGpuBackend` / `SurvivalFlexGpuBackend` shape so future
/// device-residency residencies can swap in without churn. Linux-only:
/// non-Linux builds compile a `try_device_moments` stub that never
/// constructs this backend.
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
        let runtime = crate::gpu::runtime::GpuRuntime::global().ok_or_else(|| {
            GpuError::DriverLibraryUnavailable {
                reason: "cubic_cell backend: no CUDA runtime available".to_string(),
            }
        })?;
        let ctx = crate::gpu::runtime::cuda_context_for(runtime.selected_device().ordinal)
            .ok_or_else(|| GpuError::DriverCallFailed {
                reason: format!(
                    "cubic_cell backend: failed to create CUDA context for device {}",
                    runtime.selected_device().ordinal
                ),
            })?;
        let stream = ctx.default_stream();
        Ok(CubicCellGpuBackend {
            inner: CubicCellGpuContextLinux {
                ctx,
                stream,
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
            let guard =
                self.inner
                    .modules
                    .lock()
                    .map_err(|err| GpuError::DriverCallFailed {
                        reason: format!("cubic_cell module cache mutex poisoned: {err}"),
                    })?;
            if let Some(module) = guard.get(&key) {
                return Ok(Arc::clone(module));
            }
        }
        let source =
            crate::gpu::cubic_cell::kernel_src::build_cubic_deriv_moments_kernel_source(max_degree);
        let ptx =
            cudarc::nvrtc::compile_ptx(&source).map_err(|err| GpuError::DriverCallFailed {
                reason: format!("cubic_cell NVRTC compile (degree={max_degree}) failed: {err}"),
            })?;
        let module =
            self.inner
                .ctx
                .load_module(ptx)
                .map_err(|err| GpuError::DriverCallFailed {
                    reason: format!(
                        "cubic_cell module load (degree={max_degree}) failed: {err}"
                    ),
                })?;
        let mut guard = self
            .inner
            .modules
            .lock()
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("cubic_cell module cache mutex poisoned: {err}"),
            })?;
        let entry = guard.entry(key).or_insert(module);
        Ok(Arc::clone(entry))
    }

    /// Stage-1 dispatcher: pre-bucket cells by branch, run NonAffineFinite
    /// on the device, evaluate Affine / AffineTail on the CPU using the
    /// host substrate's parity reference, and assemble both back into a
    /// single row-major output buffer.
    #[cfg(target_os = "linux")]
    fn dispatch(
        &self,
        view: &CubicCellDerivativeMomentHostView<'_>,
    ) -> Result<HostMomentBatch, GpuError> {
        let n_cells = view.cells.len();
        let stride = view.max_degree + 1;
        let mut moments = vec![0.0_f64; n_cells * stride];
        let mut status = vec![CubicCellMomentStatus::Ok as u8; n_cells];

        // First pass: classifier check + caller-tag agreement. Indices
        // surviving this pass are eligible for compute (GPU or CPU).
        let mut eligible_branches: Vec<GpuCellBranchTag> = Vec::with_capacity(n_cells);
        let mut eligible_idx: Vec<usize> = Vec::with_capacity(n_cells);
        for (i, &gpu_cell) in view.cells.iter().enumerate() {
            match classify_cell_for_gpu(gpu_cell) {
                Ok(host_tag) => {
                    if host_tag != view.branches[i] {
                        status[i] = CubicCellMomentStatus::InvalidInterval as u8;
                        continue;
                    }
                    eligible_branches.push(host_tag);
                    eligible_idx.push(i);
                }
                Err(code) => {
                    status[i] = code as u8;
                }
            }
        }

        // Bucket NonAffineFinite for the device launch; Affine /
        // AffineTail go through the CPU parity reference.
        let mut nonaffine_idx: Vec<usize> = Vec::new();
        let mut cpu_idx: Vec<usize> = Vec::new();
        for (pos, &tag) in eligible_branches.iter().enumerate() {
            let cell_idx = eligible_idx[pos];
            match tag {
                GpuCellBranchTag::NonAffineFinite => nonaffine_idx.push(cell_idx),
                GpuCellBranchTag::Affine | GpuCellBranchTag::AffineTail => cpu_idx.push(cell_idx),
            }
        }

        // CPU buckets: route through the existing per-cell evaluator so
        // Affine / AffineTail produce the exact CPU parity values.
        if !cpu_idx.is_empty() {
            self.populate_cpu_buckets(
                view,
                &cpu_idx,
                stride,
                moments.as_mut_slice(),
                status.as_mut_slice(),
            );
        }

        // GPU bucket: pre-assemble SoA inputs, copy, launch, gather.
        if !nonaffine_idx.is_empty() {
            self.launch_nonaffine_bucket(
                view,
                &nonaffine_idx,
                stride,
                moments.as_mut_slice(),
                status.as_mut_slice(),
            )?;
        }

        Ok(HostMomentBatch {
            moments,
            status,
            stride,
        })
    }

    #[cfg(target_os = "linux")]
    fn populate_cpu_buckets(
        &self,
        view: &CubicCellDerivativeMomentHostView<'_>,
        cpu_idx: &[usize],
        stride: usize,
        moments: &mut [f64],
        status: &mut [u8],
    ) {
        use crate::families::cubic_cell_kernel::{
            DenestedCubicCell, evaluate_cell_derivative_moments_uncached,
        };
        for &cell_idx in cpu_idx {
            let gpu_cell = view.cells[cell_idx];
            let cpu_cell = DenestedCubicCell {
                left: gpu_cell.left,
                right: gpu_cell.right,
                c0: gpu_cell.c0,
                c1: gpu_cell.c1,
                c2: gpu_cell.c2,
                c3: gpu_cell.c3,
            };
            let row = &mut moments[cell_idx * stride..(cell_idx + 1) * stride];
            match evaluate_cell_derivative_moments_uncached(cpu_cell, view.max_degree) {
                Ok(state) => {
                    let copy_len = state.moments.len().min(stride);
                    row[..copy_len].copy_from_slice(&state.moments[..copy_len]);
                    if row.iter().any(|x| !x.is_finite()) {
                        for slot in row.iter_mut() {
                            *slot = 0.0;
                        }
                        status[cell_idx] = CubicCellMomentStatus::NonFiniteEvaluation as u8;
                    }
                }
                Err(_) => {
                    for slot in row.iter_mut() {
                        *slot = 0.0;
                    }
                    status[cell_idx] = match view.branches[cell_idx] {
                        GpuCellBranchTag::AffineTail => {
                            CubicCellMomentStatus::NonAffineInfiniteInterval as u8
                        }
                        _ => CubicCellMomentStatus::InvalidInterval as u8,
                    };
                }
            }
        }
    }

    #[cfg(target_os = "linux")]
    fn launch_nonaffine_bucket(
        &self,
        view: &CubicCellDerivativeMomentHostView<'_>,
        nonaffine_idx: &[usize],
        stride: usize,
        moments: &mut [f64],
        status: &mut [u8],
    ) -> Result<(), GpuError> {
        use cudarc::driver::{LaunchConfig, PushKernelArg};

        let m = nonaffine_idx.len();
        // Pack the SoA layout the kernel signature expects.
        let mut left = Vec::with_capacity(m);
        let mut right = Vec::with_capacity(m);
        let mut c0 = Vec::with_capacity(m);
        let mut c1 = Vec::with_capacity(m);
        let mut c2 = Vec::with_capacity(m);
        let mut c3 = Vec::with_capacity(m);
        let mut branch_code = Vec::with_capacity(m);
        for &i in nonaffine_idx {
            let c = view.cells[i];
            left.push(c.left);
            right.push(c.right);
            c0.push(c.c0);
            c1.push(c.c1);
            c2.push(c.c2);
            c3.push(c.c3);
            // Mirrors `BRANCH_NONAFFINE_FIN = 1` from kernel_src.
            branch_code.push(1u8);
        }

        let max_degree = view.max_degree;
        let module = self.module_for_degree(max_degree)?;
        let kernel_name = format!("cubic_deriv_moments_d{max_degree}");
        let func =
            module
                .load_function(&kernel_name)
                .map_err(|err| GpuError::DriverCallFailed {
                    reason: format!("cubic_cell load_function {kernel_name}: {err}"),
                })?;

        let stream = &self.inner.stream;
        let d_left = stream
            .clone_htod(&left)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("cubic_cell memcpy_stod left: {err}"),
            })?;
        let d_right = stream
            .clone_htod(&right)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("cubic_cell memcpy_stod right: {err}"),
            })?;
        let d_c0 = stream
            .clone_htod(&c0)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("cubic_cell memcpy_stod c0: {err}"),
            })?;
        let d_c1 = stream
            .clone_htod(&c1)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("cubic_cell memcpy_stod c1: {err}"),
            })?;
        let d_c2 = stream
            .clone_htod(&c2)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("cubic_cell memcpy_stod c2: {err}"),
            })?;
        let d_c3 = stream
            .clone_htod(&c3)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("cubic_cell memcpy_stod c3: {err}"),
            })?;
        let d_branch = stream
            .clone_htod(&branch_code)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("cubic_cell memcpy_stod branch_code: {err}"),
            })?;
        let mut d_moments = stream.alloc_zeros::<f64>(m * stride).map_err(|err| {
            GpuError::DriverCallFailed {
                reason: format!("cubic_cell alloc_zeros moments: {err}"),
            }
        })?;
        let mut d_status = stream
            .alloc_zeros::<u8>(m)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("cubic_cell alloc_zeros status: {err}"),
            })?;

        // One warp per cell.  Block = 4 warps (128 threads).
        let warps_per_block: u32 = 4;
        let block: u32 = 32 * warps_per_block;
        let m_u32: u32 = u32::try_from(m).map_err(|_| GpuError::DriverCallFailed {
            reason: format!("cubic_cell n_cells={m} overflows u32"),
        })?;
        let grid: u32 = m_u32.div_ceil(warps_per_block).max(1);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_cells_u32 = m_u32;
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
        // SAFETY: every argument is a typed device pointer / scalar
        // matching the kernel signature above; grid covers exactly `m`
        // warps; out-of-range warps early-return.
        unsafe { builder.launch(cfg) }.map_err(|err| GpuError::DriverCallFailed {
            reason: format!("cubic_cell kernel launch: {err}"),
        })?;

        let host_moments = stream
            .clone_dtoh(&d_moments)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("cubic_cell memcpy_dtov moments: {err}"),
            })?;
        let host_status = stream
            .clone_dtoh(&d_status)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("cubic_cell memcpy_dtov status: {err}"),
            })?;
        stream
            .synchronize()
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("cubic_cell synchronize: {err}"),
            })?;

        // Scatter the GPU bucket back into the output buffer at original
        // indices.
        for (pos, &cell_idx) in nonaffine_idx.iter().enumerate() {
            let dst = &mut moments[cell_idx * stride..(cell_idx + 1) * stride];
            let src = &host_moments[pos * stride..(pos + 1) * stride];
            dst.copy_from_slice(src);
            status[cell_idx] = host_status[pos];
            // Zero rows whose kernel-side status is non-OK to keep the
            // host substrate's contract.
            if host_status[pos] != CubicCellMomentStatus::Ok as u8 {
                for slot in dst.iter_mut() {
                    *slot = 0.0;
                }
            }
        }
        Ok(())
    }

    /// Device-resident sibling of [`Self::dispatch`]: leaves the moments +
    /// status buffers on the GPU. Stage-4 strategy: route **all three**
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
    /// backend's default stream — which (because `crate::gpu::runtime::cuda_context_for`
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
        let mut status_host = vec![CubicCellMomentStatus::Ok as u8; n_cells];
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
                    if host_tag != view.branches[i] {
                        status_host[i] = CubicCellMomentStatus::InvalidInterval as u8;
                        continue;
                    }
                    branch_code[i] = match host_tag {
                        GpuCellBranchTag::Affine => 0,
                        GpuCellBranchTag::NonAffineFinite => 1,
                        GpuCellBranchTag::AffineTail => 2,
                    };
                }
                Err(code) => {
                    status_host[i] = code as u8;
                }
            }
        }

        // ---- Allocate device buffers + launch the kernel against the full
        //      `[n_cells, stride]` layout indexed by original cell index.
        let max_degree = view.max_degree;
        let module = self.module_for_degree(max_degree)?;
        let kernel_name = format!("cubic_deriv_moments_d{max_degree}");
        let func =
            module
                .load_function(&kernel_name)
                .map_err(|err| GpuError::DriverCallFailed {
                    reason: format!("cubic_cell load_function {kernel_name}: {err}"),
                })?;

        let stream = &self.inner.stream;
        let d_left = stream
            .clone_htod(&left)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("cubic_cell device-resident memcpy left: {err}"),
            })?;
        let d_right = stream
            .clone_htod(&right)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("cubic_cell device-resident memcpy right: {err}"),
            })?;
        let d_c0 = stream
            .clone_htod(&c0)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("cubic_cell device-resident memcpy c0: {err}"),
            })?;
        let d_c1 = stream
            .clone_htod(&c1)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("cubic_cell device-resident memcpy c1: {err}"),
            })?;
        let d_c2 = stream
            .clone_htod(&c2)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("cubic_cell device-resident memcpy c2: {err}"),
            })?;
        let d_c3 = stream
            .clone_htod(&c3)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("cubic_cell device-resident memcpy c3: {err}"),
            })?;
        let d_branch = stream
            .clone_htod(&branch_code)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("cubic_cell device-resident memcpy branch: {err}"),
            })?;
        let mut d_moments = stream.alloc_zeros::<f64>(n_cells * stride).map_err(|err| {
            GpuError::DriverCallFailed {
                reason: format!("cubic_cell device-resident alloc moments: {err}"),
            }
        })?;
        let mut d_status = stream
            .alloc_zeros::<u8>(n_cells)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("cubic_cell device-resident alloc status: {err}"),
            })?;

        let warps_per_block: u32 = 4;
        let block: u32 = 32 * warps_per_block;
        let n_u32: u32 = u32::try_from(n_cells).map_err(|_| GpuError::DriverCallFailed {
            reason: format!("cubic_cell n_cells={n_cells} overflows u32"),
        })?;
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
        // SAFETY: layout contract identical to `launch_nonaffine_bucket`,
        // and the kernel's lane-0 validator rejects unrecognized branch
        // codes (255 sentinel) by zeroing the row and writing
        // STATUS_INVALID, so classifier-rejected slots are safe.
        unsafe { builder.launch(cfg) }.map_err(|err| GpuError::DriverCallFailed {
            reason: format!("cubic_cell device-resident kernel launch: {err}"),
        })?;

        // Read back per-cell statuses so the host can:
        //   (a) merge with classifier-rejected entries it already knows
        //       (those use the classifier's specific status code, not
        //        the kernel's catch-all STATUS_INVALID),
        //   (b) hand callers a ready-made `status: Vec<u8>` they can
        //       branch on without a second DtoH.
        let kernel_status =
            stream
                .clone_dtoh(&d_status)
                .map_err(|err| GpuError::DriverCallFailed {
                    reason: format!("cubic_cell device-resident DtoH status: {err}"),
                })?;
        stream
            .synchronize()
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("cubic_cell device-resident sync after kernel: {err}"),
            })?;

        // Merge: if the classifier already rejected a cell, its specific
        // code wins (the kernel's row for that cell was zeroed by the
        // STATUS_INVALID path so the device buffer is already correct).
        // Otherwise take the kernel's status verbatim.
        for i in 0..n_cells {
            if status_host[i] == CubicCellMomentStatus::Ok as u8 {
                status_host[i] = kernel_status[i];
            }
        }
        // Re-upload merged statuses so device and host views agree.
        let merged_status = status_host.clone();
        d_status = stream
            .clone_htod(&merged_status)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("cubic_cell device-resident HtoD merged status: {err}"),
            })?;
        stream
            .synchronize()
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("cubic_cell device-resident synchronize: {err}"),
            })?;

        Ok(CubicCellDerivativeMomentOutput::Device {
            d_moments,
            d_status,
            status: status_host,
            stride,
            n_cells,
        })
    }

}

/// Test-only DtoH helpers for cubic-cell device residency parity tests.
/// `#[cfg(test)]` on the `mod` declaration (with the `test_support` name
/// the ban scanner's allow-list explicitly accepts) keeps every item inside
/// invisible to production builds; the dead-pub scanner skips
/// `#[cfg(test)]`-region defs, so the inner `pub(crate)` is sound even
/// when the only consumers live in sibling `mod tests` blocks.
#[cfg(test)]
pub(crate) mod test_support {
    use super::CubicCellGpuBackend;
    use crate::gpu::error::GpuError;

    pub(crate) fn download_moments(
        backend: &CubicCellGpuBackend,
        d_moments: &cudarc::driver::CudaSlice<f64>,
    ) -> Result<Vec<f64>, GpuError> {
        let stream = &backend.inner.stream;
        let host = stream
            .clone_dtoh(d_moments)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("cubic_cell test_support::download_moments DtoH: {err}"),
            })?;
        stream
            .synchronize()
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("cubic_cell test_support::download_moments sync: {err}"),
            })?;
        Ok(host)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::cubic_cell::{
        CubicCellDerivativeMomentHostView, CubicCellMomentResidency, CubicCellMomentStatus,
        GpuCellBranchTag, GpuDenestedCubicCell,
    };
    use crate::gpu::runtime::GpuRuntime;

    fn make_nonaffine_cells() -> Vec<GpuDenestedCubicCell> {
        vec![
            GpuDenestedCubicCell {
                left: -1.25,
                right: -0.2,
                c0: -0.35,
                c1: 0.85,
                c2: 0.4,
                c3: 0.0,
            },
            GpuDenestedCubicCell {
                left: -0.5,
                right: 1.7,
                c0: 0.2,
                c1: -0.6,
                c2: 0.25,
                c3: 0.18,
            },
            GpuDenestedCubicCell {
                left: 0.1,
                right: 0.9,
                c0: 0.05,
                c1: 0.0,
                c2: -0.3,
                c3: 0.12,
            },
            GpuDenestedCubicCell {
                left: -2.0,
                right: 2.0,
                c0: 0.0,
                c1: 0.0,
                c2: 0.5,
                c3: 0.0,
            },
            GpuDenestedCubicCell {
                left: -0.8,
                right: 0.4,
                c0: 0.1,
                c1: -0.25,
                c2: 0.05,
                c3: -0.07,
            },
        ]
    }

    /// V100-only: end-to-end CPU↔GPU parity on a NonAffineFinite batch.
    /// Skipped on hosts without a usable CUDA runtime so this still passes
    /// on the Mac builder.
    #[test]
    fn cubic_cell_gpu_nonaffine_matches_cpu_within_tol() {
        let Some(runtime) = GpuRuntime::global() else {
            eprintln!(
                "[cubic_cell_gpu test] no CUDA runtime — skipping NonAffineFinite parity test"
            );
            return;
        };
        eprintln!(
            "[cubic_cell_gpu test] runtime selected device ordinal={}",
            runtime.selected_device().ordinal
        );

        let cells = make_nonaffine_cells();
        let branches = vec![GpuCellBranchTag::NonAffineFinite; cells.len()];
        let max_degree = 9;
        let view = CubicCellDerivativeMomentHostView {
            cells: &cells,
            branches: &branches,
            max_degree,
            residency: CubicCellMomentResidency::Host,
        };

        // GPU path.
        let gpu_batch = try_device_moments(&view)
            .expect("device dispatch must succeed on a host with CUDA")
            .expect("Some(_) from device dispatch when GPU is present");

        // CPU parity reference.
        let cpu_batch = crate::gpu::cubic_cell::host_substrate::build_host_moments(&view)
            .expect("host substrate produces parity reference");

        assert_eq!(gpu_batch.stride, cpu_batch.stride);
        assert_eq!(gpu_batch.status, cpu_batch.status);
        let stride = gpu_batch.stride;
        for cell_idx in 0..cells.len() {
            assert_eq!(
                gpu_batch.status[cell_idx],
                CubicCellMomentStatus::Ok as u8,
                "cell {cell_idx} must classify Ok"
            );
            let gpu_row = &gpu_batch.moments[cell_idx * stride..(cell_idx + 1) * stride];
            let cpu_row = &cpu_batch.moments[cell_idx * stride..(cell_idx + 1) * stride];
            for (k, (&got, &want)) in gpu_row.iter().zip(cpu_row.iter()).enumerate() {
                let denom = want.abs().max(1.0);
                let rel = (got - want).abs() / denom;
                assert!(
                    rel <= 1e-8,
                    "cell={cell_idx} k={k} gpu={got:.17e} cpu={want:.17e} rel={rel:.3e}"
                );
            }
        }
    }

    /// Static (no-device) smoke test: when no CUDA runtime is present
    /// `try_device_moments` must return `Ok(None)` so the caller knows to
    /// fall back to the host substrate.
    #[test]
    fn cubic_cell_gpu_returns_none_when_runtime_absent() {
        if GpuRuntime::global().is_some() {
            eprintln!(
                "[cubic_cell_gpu test] CUDA runtime present — skipping the absent-runtime case"
            );
            return;
        }
        let cells = make_nonaffine_cells();
        let branches = vec![GpuCellBranchTag::NonAffineFinite; cells.len()];
        let view = CubicCellDerivativeMomentHostView {
            cells: &cells,
            branches: &branches,
            max_degree: 9,
            residency: CubicCellMomentResidency::Host,
        };
        let out = try_device_moments(&view).expect("clean Ok on hosts without CUDA");
        assert!(
            out.is_none(),
            "expected Ok(None) on a host without a usable CUDA runtime"
        );
    }
}
