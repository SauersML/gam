//! In-place row scaling kernel: `X_i,: ← w_i · X_i,:` for each row.
//!
//! The CPU sibling rescales each row in place using the dense ndarray view.
//! The CUDA dispatch path uploads, launches an NVRTC kernel that performs the
//! per-row multiply, then downloads back into the caller's buffer.

use crate::gpu::error::GpuError;
use ndarray::{Array2, ArrayView1};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendStatus {
    CpuFallback,
    CudaUnavailable,
    CudaReady,
}

#[inline]
pub fn backend_status() -> BackendStatus {
    #[cfg(feature = "cuda")]
    {
        match crate::gpu::runtime::GpuRuntime::global() {
            Some(_) => BackendStatus::CudaReady,
            None => BackendStatus::CudaUnavailable,
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        BackendStatus::CpuFallback
    }
}

/// CPU sibling: in-place row scaling, negative weights are clamped to zero.
pub fn cpu_row_scale(x: &mut Array2<f64>, w: ArrayView1<'_, f64>) {
    let (n, p) = x.dim();
    let m = n.min(w.len());
    for i in 0..m {
        let ww = if w[i] < 0.0 { 0.0 } else { w[i] };
        for j in 0..p {
            x[[i, j]] *= ww;
        }
    }
}

/// Attempt to dispatch the row-scale kernel to the device backend.
pub fn try_dispatch(
    x: &mut Array2<f64>,
    w: ArrayView1<'_, f64>,
) -> Option<Result<(), GpuError>> {
    let (n, _p) = x.dim();
    if n == 0 || n != w.len() {
        return None;
    }
    #[cfg(feature = "cuda")]
    {
        let Some(_runtime) = crate::gpu::runtime::GpuRuntime::global() else {
            return None;
        };
        Some(cuda_row_scale(x, w))
    }
    #[cfg(not(feature = "cuda"))]
    {
        drop((x, w));
        None
    }
}

#[cfg(feature = "cuda")]
fn cuda_row_scale(x: &mut Array2<f64>, w: ArrayView1<'_, f64>) -> Result<(), GpuError> {
    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::compile_ptx;

    const KERNEL: &str = r#"
extern "C" __global__ void row_scale_inplace_kernel(
    double* __restrict__ x,
    const double* __restrict__ w,
    unsigned long long n,
    unsigned long long p)
{
    unsigned long long row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned long long col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= p) return;
    double ww = w[row];
    if (ww < 0.0) ww = 0.0;
    x[row * p + col] *= ww;
}
"#;

    let (n, p) = x.dim();
    let ctx = CudaContext::new(0).map_err(|e| GpuError::DriverCallFailed {
        reason: format!("row_scale context init failed: {e}"),
    })?;
    let stream = ctx.default_stream();
    let ptx = compile_ptx(KERNEL).map_err(|e| GpuError::DriverCallFailed {
        reason: format!("row_scale NVRTC compile failed: {e}"),
    })?;
    let module = ctx.load_module(ptx).map_err(|e| GpuError::DriverCallFailed {
        reason: format!("row_scale load module failed: {e}"),
    })?;
    let func = module
        .load_function("row_scale_inplace_kernel")
        .map_err(|e| GpuError::DriverCallFailed {
            reason: format!("row_scale load function failed: {e}"),
        })?;

    // Pack into a row-major Vec because the caller's ndarray may not be
    // contiguous in a slice-friendly way.
    let mut host = Vec::<f64>::with_capacity(n * p);
    for i in 0..n {
        for j in 0..p {
            host.push(x[[i, j]]);
        }
    }
    let w_slice = w.as_slice().ok_or_else(|| GpuError::DriverCallFailed {
        reason: "row_scale weight slice not contiguous".to_string(),
    })?;
    let mut dx = stream.memcpy_stod(&host).map_err(map_drv)?;
    let dw = stream.memcpy_stod(w_slice).map_err(map_drv)?;

    let tx: u32 = 16;
    let ty: u32 = 16;
    let bx = ((p as u32) + tx - 1) / tx;
    let by = ((n as u32) + ty - 1) / ty;
    let cfg = LaunchConfig {
        grid_dim: (bx, by, 1),
        block_dim: (tx, ty, 1),
        shared_mem_bytes: 0,
    };
    let nn = n as u64;
    let pp = p as u64;
    let mut builder = stream.launch_builder(&func);
    builder.arg(&mut dx).arg(&dw).arg(&nn).arg(&pp);
    unsafe { builder.launch(cfg) }.map_err(map_drv)?;

    let host_back = stream.memcpy_dtov(&dx).map_err(map_drv)?;
    for i in 0..n {
        for j in 0..p {
            x[[i, j]] = host_back[i * p + j];
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn map_drv(e: cudarc::driver::DriverError) -> GpuError {
    GpuError::DriverCallFailed {
        reason: format!("row_scale cudarc driver error: {e}"),
    }
}
