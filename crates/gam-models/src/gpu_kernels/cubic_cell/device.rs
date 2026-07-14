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

/// Linux-only: launch the Stage-1 dispatcher and return the moments
/// buffer in device memory (`CudaSlice<f64>`). Caller may pass the
/// returned slice directly to any kernel launch on the same default
/// stream (e.g. `bms_flex_row_kernel`). Returns `Ok(None)` when no CUDA
/// runtime is available (caller should fall back to the host substrate).
/// Returns `Err` on a genuine driver / NVRTC / shape failure that the
/// caller must surface.
#[cfg(target_os = "linux")]
pub(crate) fn try_device_moments_resident(
    view: &CubicCellDerivativeMomentHostView<'_>,
) -> Result<Option<CubicCellDerivativeMomentOutput>, GpuError> {
    let Some(_) = gam_gpu::device_runtime::GpuRuntime::resolve(gam_gpu::global_policy())? else {
        return Ok(None);
    };
    let backend = CubicCellGpuBackend::probe()?;
    backend.dispatch_device_resident(view).map(Some)
}

/// Process-wide cubic-cell GPU backend. Mirrors the
/// `BmsFlexGpuBackend` / `SurvivalFlexGpuBackend` shape so future
/// device-residency residencies can swap in without churn. Linux-only:
/// non-Linux builds skip [`try_device_moments_resident`] at the call
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
        //   (b) hand callers a ready-made `status: Vec<u8>` they can
        //       branch on without a second DtoH.
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
            if status_host[i] == CubicCellMomentStatus::Ok as u8 {
                status_host[i] = kernel_status[i];
            }
        }
        drop(d_status);
        Ok(CubicCellDerivativeMomentOutput::Device {
            d_moments,
            status: status_host,
            stride,
            n_cells,
        })
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use crate::gpu_kernels::cubic_cell::{
        CubicCellDerivativeMomentHostView, CubicCellDerivativeMomentOutput,
        CubicCellMomentResidency, CubicCellMomentStatus, GpuCellBranchTag, GpuDenestedCubicCell,
        try_build_cubic_cell_derivative_moments,
    };
    use gam_gpu::device_runtime::GpuRuntime;
    use gam_gpu::gpu_error::GpuError;
    use gam_gpu::gpu_error::GpuResultExt;

    /// Test-only DtoH helper for cubic-cell device residency parity tests.
    fn download_moments(
        backend: &CubicCellGpuBackend,
        d_moments: &cudarc::driver::CudaSlice<f64>,
    ) -> Result<Vec<f64>, GpuError> {
        let stream = &backend.inner.stream;
        let host = stream
            .clone_dtoh(d_moments)
            .gpu_ctx("cubic_cell tests::download_moments DtoH")?;
        stream
            .synchronize()
            .gpu_ctx("cubic_cell tests::download_moments sync")?;
        Ok(host)
    }

    /// Phase 4 parity test: device-resident moments must match the CPU
    /// `evaluate_cell_derivative_moments_uncached` reference across all
    /// three branches (`Affine`, `NonAffineFinite`, `AffineTail`) at the
    /// production high-water-mark degrees (9, 15, 21).
    ///
    /// Skipped silently on hosts without a usable CUDA runtime so the test
    /// passes on the Mac builder. On V100 it runs the device pipeline,
    /// downloads the moments for verification, and compares elementwise against
    /// the CPU evaluator at `abs <= 1e-12 OR rel <= 1e-12·10^(k/3)` — a per-order
    /// relative band that tracks the affine moment recurrence's condition growth
    /// (see the loop body for the #1175 derivation and measured V100 drift).
    #[cfg(target_os = "linux")]
    #[test]
    fn cubic_cell_device_residency_matches_cpu_all_branches() {
        use crate::cubic_cell_kernel::{
            DenestedCubicCell, evaluate_cell_derivative_moments_uncached,
        };
        match GpuRuntime::resolve(gam_gpu::GpuPolicy::Auto) {
            Ok(Some(_)) => {}
            Ok(None) => {
                eprintln!("[cubic_cell device-residency parity] no CUDA device — skipping");
                return;
            }
            Err(error) => {
                panic!("[cubic_cell device-residency parity] CUDA probe failed: {error}")
            }
        }
        // One cell per branch, plus a sextic NonAffineFinite stressor.
        let cpu_cells = vec![
            // Pure Affine.
            DenestedCubicCell {
                left: -1.0,
                right: 1.0,
                c0: 0.2,
                c1: 0.7,
                c2: 0.0,
                c3: 0.0,
            },
            // Quartic NonAffineFinite.
            DenestedCubicCell {
                left: -1.25,
                right: -0.2,
                c0: -0.35,
                c1: 0.85,
                c2: 0.4,
                c3: 0.0,
            },
            // Sextic NonAffineFinite.
            DenestedCubicCell {
                left: -0.5,
                right: 1.7,
                c0: 0.2,
                c1: -0.6,
                c2: 0.25,
                c3: 0.18,
            },
            // AffineTail (left-infinite).
            DenestedCubicCell {
                left: f64::NEG_INFINITY,
                right: -0.7,
                c0: 0.1,
                c1: 0.5,
                c2: 0.0,
                c3: 0.0,
            },
            // AffineTail (right-infinite).
            DenestedCubicCell {
                left: 1.2,
                right: f64::INFINITY,
                c0: -0.05,
                c1: 0.3,
                c2: 0.0,
                c3: 0.0,
            },
            // Whole-line affine.
            DenestedCubicCell {
                left: f64::NEG_INFINITY,
                right: f64::INFINITY,
                c0: 0.0,
                c1: 0.0,
                c2: 0.0,
                c3: 0.0,
            },
        ];
        let cells_gpu: Vec<GpuDenestedCubicCell> = cpu_cells
            .iter()
            .map(|c| GpuDenestedCubicCell {
                left: c.left,
                right: c.right,
                c0: c.c0,
                c1: c.c1,
                c2: c.c2,
                c3: c.c3,
            })
            .collect();
        let branches: Vec<GpuCellBranchTag> = cpu_cells
            .iter()
            .map(|c| {
                if !c.left.is_finite() || !c.right.is_finite() {
                    GpuCellBranchTag::AffineTail
                } else if c.c2 == 0.0 && c.c3 == 0.0 {
                    GpuCellBranchTag::Affine
                } else {
                    GpuCellBranchTag::NonAffineFinite
                }
            })
            .collect();

        for &max_degree in &[9_usize, 15, 21] {
            let view = CubicCellDerivativeMomentHostView {
                cells: &cells_gpu,
                branches: &branches,
                max_degree,
                residency: CubicCellMomentResidency::Device,
            };
            let out = try_build_cubic_cell_derivative_moments(view)
                .expect("device-residency dispatch must succeed with CUDA")
                .expect("non-empty input must yield output");
            let (d_moments, status, stride, n_cells) = match out {
                CubicCellDerivativeMomentOutput::Device {
                    d_moments,
                    status,
                    stride,
                    n_cells,
                } => (d_moments, status, stride, n_cells),
                CubicCellDerivativeMomentOutput::Host { .. } => panic!(
                    "device residency must produce CubicCellDerivativeMomentOutput::Device on a CUDA host"
                ),
            };
            assert_eq!(stride, max_degree + 1);
            assert_eq!(n_cells, cpu_cells.len());
            assert_eq!(status.len(), cpu_cells.len());
            // Download for verification using the in-mod helper.
            let backend = CubicCellGpuBackend::probe().expect("backend probe");
            let host_moments =
                download_moments(backend, &d_moments).expect("DtoH download for parity check");
            for (i, &cpu_cell) in cpu_cells.iter().enumerate() {
                assert_eq!(
                    status[i],
                    CubicCellMomentStatus::Ok as u8,
                    "cell {i} must classify Ok (status={})",
                    status[i]
                );
                let row = &host_moments[i * stride..(i + 1) * stride];
                let cpu_state = evaluate_cell_derivative_moments_uncached(cpu_cell, max_degree)
                    .expect("cpu reference");
                for (k, (&got, &want)) in row.iter().zip(cpu_state.moments.iter()).enumerate() {
                    let abs = (got - want).abs();
                    // Per-order relative parity band (#1175). The Affine /
                    // AffineTail branches run the forward moment recurrence
                    // `M_{n+1} = (n·M_{n-1} − d0·M_n − B_n)/d1` (kernel_src.rs).
                    // That recurrence is mildly ill-conditioned: each step
                    // amplifies the prior rounding error, so the relative gap
                    // between the CPU's serial evaluation and the device's
                    // evaluation of the SAME recurrence compounds geometrically
                    // with the moment order k. Measured on a Tesla V100 (with
                    // NVRTC `--fmad=false`, i.e. #1686's FMA-contraction fix
                    // ACTIVE — this kernel compiles through `compile_ptx_arch`):
                    // Affine ~5e-13 at k=9, ~5e-10 at k=15, ~1.48e-6 at k=21 —
                    // about ×10³ per +6 orders, i.e. the recurrence's own
                    // condition growth. Notably this is essentially UNCHANGED
                    // from the pre-#1686 fmad=true measurement (1.5e-6 at k=21),
                    // which proves the drift is NOT FMA contraction but pure
                    // round-off order: CPU-serial vs device evaluation of an
                    // ill-conditioned recurrence. (Confirmation: the
                    // NonAffineFinite GL-quadrature branch, which has no such
                    // recurrence, agrees to ≤2.4e-15 at every degree; the
                    // AffineTail branch's closed-form path holds ≤8e-16.) A flat
                    // `rel ≤ 1e-11` therefore wrongly fails the high-order
                    // affine moments. Gate each order against a band that
                    // tracks the condition growth with ~10× headroom:
                    //   rel_tol(k) = 1e-12 · 10^(k/3)
                    // (k=21 → 1e-5, well above the worst 1.48e-6; k=0 → 1e-12).
                    // A real bug perturbs a moment by O(value) → rel ~1, which
                    // blows through this band at every order.
                    let rel = abs / want.abs().max(1e-300);
                    let rel_tol = 1e-12 * 10f64.powf(k as f64 / 3.0);
                    assert!(
                        abs <= 1e-12 || rel <= rel_tol,
                        "device parity drift at branch={:?} degree={max_degree} cell={i} k={k} \
                         gpu={got:.17e} cpu={want:.17e} abs={abs:.3e} rel={rel:.3e} > rel_tol={rel_tol:.3e}",
                        branches[i]
                    );
                }
            }
        }
    }
}
