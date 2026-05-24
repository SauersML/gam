//! Per-cell moment evaluation kernel.
//!
//! Evaluates the polynomial moments of the denested cubic cell representation
//! used by the tail-cell family.  The CPU sibling drives the existing
//! `evaluate_cell_moments_uncached` path and packs the result into a dense
//! `(n_cells, max_degree + 1)` matrix.  When the CUDA feature is enabled and a
//! runtime device is available the dispatch path forwards the same workload to
//! a cudarc-launched NVRTC kernel; otherwise `try_dispatch` returns `None`,
//! which signals callers to take the CPU sibling.

use crate::families::cubic_cell_kernel::{DenestedCubicCell, evaluate_cell_moments_uncached};
use crate::gpu::error::GpuError;
use ndarray::Array2;

/// Public marker retained so the HAL surface can identify the kernel-set
/// version even when no device backend is compiled in.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendStatus {
    CpuFallback,
    CudaUnavailable,
    CudaReady,
}

#[inline]
pub fn backend_status() -> BackendStatus {
    match crate::gpu::runtime::GpuRuntime::global() {
        Some(_) => BackendStatus::CudaReady,
        None => BackendStatus::CudaUnavailable,
    }
}

/// CPU implementation of the per-cell moment kernel.
///
/// Returns a dense `(n_cells, degree + 1)` matrix where row `i` contains the
/// moments `[m_0, m_1, ..., m_degree]` of cell `i`.  Cells whose moment
/// evaluator errors out (degenerate non-affine envelopes) contribute a
/// zero-filled row so the returned matrix remains rectangular.
pub fn cpu_cell_moments(cells: &[DenestedCubicCell], degree: usize) -> Array2<f64> {
    let n_cells = cells.len();
    let cols = degree.saturating_add(1);
    let mut out = Array2::<f64>::zeros((n_cells, cols));
    for (i, cell) in cells.iter().enumerate() {
        match evaluate_cell_moments_uncached(*cell, degree) {
            Ok(state) => {
                let moments = state.moments.as_slice();
                let take = moments.len().min(cols);
                for k in 0..take {
                    out[[i, k]] = moments[k];
                }
            }
            Err(_msg) => {
                // Degenerate cell: leave the row at zero so the matrix
                // remains rectangular; the caller can inspect cell metadata
                // to decide whether to drop the row.
            }
        }
    }
    out
}

/// Attempt to dispatch the per-cell moment kernel to the device backend.
///
/// Returns `None` when no GPU dispatch took place (no CUDA feature, no device,
/// empty input).  Returns `Some(Ok(matrix))` on a successful device launch and
/// `Some(Err(GpuError))` if the device path was attempted but failed.
pub fn try_dispatch(
    cells: &[DenestedCubicCell],
    degree: usize,
) -> Option<Result<Array2<f64>, GpuError>> {
    if cells.is_empty() {
        return None;
    }
    let Some(_runtime) = crate::gpu::runtime::GpuRuntime::global() else {
        return None;
    };
    Some(cuda_cell_moments(cells, degree))
}

fn cuda_cell_moments(cells: &[DenestedCubicCell], degree: usize) -> Result<Array2<f64>, GpuError> {
    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::compile_ptx;

    const KERNEL: &str = r#"
extern "C" __global__ void cell_moments_kernel(
    const double* __restrict__ lefts,
    const double* __restrict__ rights,
    const double* __restrict__ c0,
    const double* __restrict__ c1,
    const double* __restrict__ c2,
    const double* __restrict__ c3,
    double* __restrict__ out,
    unsigned long long n_cells,
    unsigned long long n_moments)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cells) return;
    double a = lefts[i];
    double b = rights[i];
    // Quadrature: 8-node Gauss-Legendre on [a, b].
    const double nodes[8] = {
        -0.9602898564975363, -0.7966664774136267,
        -0.5255324099163290, -0.1834346424956498,
         0.1834346424956498,  0.5255324099163290,
         0.7966664774136267,  0.9602898564975363
    };
    const double weights[8] = {
        0.1012285362903763, 0.2223810344533745,
        0.3137066458778873, 0.3626837833783620,
        0.3626837833783620, 0.3137066458778873,
        0.2223810344533745, 0.1012285362903763
    };
    double half = 0.5 * (b - a);
    double mid  = 0.5 * (b + a);
    double cc0 = c0[i], cc1 = c1[i], cc2 = c2[i], cc3 = c3[i];
    for (unsigned long long k = 0; k < n_moments; ++k) {
        double acc = 0.0;
        for (int q = 0; q < 8; ++q) {
            double z = mid + half * nodes[q];
            double eta = cc0 + z * (cc1 + z * (cc2 + z * cc3));
            double zk = 1.0;
            for (unsigned long long p = 0; p < k; ++p) { zk *= z; }
            acc += weights[q] * zk * eta;
        }
        out[i * n_moments + k] = acc * half;
    }
}
"#;

    let ptx = compile_ptx(KERNEL).map_err(|e| GpuError::DriverCallFailed {
        reason: format!("cell_moments NVRTC compile failed: {e}"),
    })?;
    let ctx = CudaContext::new(0).map_err(|e| GpuError::DriverCallFailed {
        reason: format!("cell_moments context init failed: {e}"),
    })?;
    let module = ctx
        .load_module(ptx)
        .map_err(|e| GpuError::DriverCallFailed {
            reason: format!("cell_moments load module failed: {e}"),
        })?;
    let func =
        module
            .load_function("cell_moments_kernel")
            .map_err(|e| GpuError::DriverCallFailed {
                reason: format!("cell_moments load function failed: {e}"),
            })?;
    let stream = ctx.default_stream();

    let n_cells = cells.len();
    let n_moments = degree.saturating_add(1);
    let mut lefts = Vec::<f64>::with_capacity(n_cells);
    let mut rights = Vec::<f64>::with_capacity(n_cells);
    let mut cc0 = Vec::<f64>::with_capacity(n_cells);
    let mut cc1 = Vec::<f64>::with_capacity(n_cells);
    let mut cc2 = Vec::<f64>::with_capacity(n_cells);
    let mut cc3 = Vec::<f64>::with_capacity(n_cells);
    for cell in cells {
        lefts.push(cell.left);
        rights.push(cell.right);
        cc0.push(cell.c0);
        cc1.push(cell.c1);
        cc2.push(cell.c2);
        cc3.push(cell.c3);
    }
    let dl = stream.clone_htod(&lefts).map_err(map_drv)?;
    let dr = stream.clone_htod(&rights).map_err(map_drv)?;
    let d0 = stream.clone_htod(&cc0).map_err(map_drv)?;
    let d1 = stream.clone_htod(&cc1).map_err(map_drv)?;
    let d2 = stream.clone_htod(&cc2).map_err(map_drv)?;
    let d3 = stream.clone_htod(&cc3).map_err(map_drv)?;
    let mut dout = stream
        .alloc_zeros::<f64>(n_cells * n_moments)
        .map_err(map_drv)?;

    let threads: u32 = 128;
    let blocks = (n_cells as u32).div_ceil(threads);
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    let nc = n_cells as u64;
    let nm = n_moments as u64;
    let mut builder = stream.launch_builder(&func);
    builder
        .arg(&dl)
        .arg(&dr)
        .arg(&d0)
        .arg(&d1)
        .arg(&d2)
        .arg(&d3)
        .arg(&mut dout)
        .arg(&nc)
        .arg(&nm);
    // SAFETY: every kernel arg is a device pointer or pod scalar bound above
    // via `builder.arg(...)`; grid/block dims in `cfg` cover exactly `n_cells`
    // threads, matching the kernel's indexed bounds check.
    unsafe { builder.launch(cfg) }.map_err(map_drv)?;

    let host = stream.clone_dtoh(&dout).map_err(map_drv)?;
    Array2::from_shape_vec((n_cells, n_moments), host).map_err(|e| GpuError::DriverCallFailed {
        reason: format!("cell_moments host reshape failed: {e}"),
    })
}

fn map_drv(e: cudarc::driver::DriverError) -> GpuError {
    GpuError::DriverCallFailed {
        reason: format!("cell_moments cudarc driver error: {e}"),
    }
}
