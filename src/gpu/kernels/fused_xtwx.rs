//! Fused symmetric `XᵀWX` kernel.
//!
//! The CPU sibling forwards to the faer-backed implementation already wired
//! into the codebase.  The CUDA path materializes the diagonal-weight gram via
//! a cublasDgemm call on `Xᵀ · (W · X)`, where the row-scaling stage is a
//! single NVRTC kernel that fuses the per-row multiply with the gemm input.

use crate::gpu::error::GpuError;
use crate::linalg::faer_ndarray::fast_xt_diag_x;
use ndarray::{Array2, ArrayView1, ArrayView2};

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

/// CPU sibling: forwards to the faer-backed `Xᵀ diag(w) X` implementation.
pub fn cpu_xtwx(x: ArrayView2<'_, f64>, w: ArrayView1<'_, f64>) -> Array2<f64> {
    fast_xt_diag_x(&x, &w)
}

/// Attempt to dispatch the symmetric `XᵀWX` build to the device backend.
pub fn try_dispatch(
    x: ArrayView2<'_, f64>,
    w: ArrayView1<'_, f64>,
) -> Option<Result<Array2<f64>, GpuError>> {
    if x.nrows() != w.len() || x.nrows() == 0 || x.ncols() == 0 {
        return None;
    }
    let Some(_runtime) = crate::gpu::runtime::GpuRuntime::global() else {
        return None;
    };
    Some(cuda_xtwx(x, w))
}

fn cuda_xtwx(x: ArrayView2<'_, f64>, w: ArrayView1<'_, f64>) -> Result<Array2<f64>, GpuError> {
    use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::compile_ptx;

    const ROW_SCALE: &str = r#"
extern "C" __global__ void row_scale_kernel(
    const double* __restrict__ src,
    const double* __restrict__ w,
    double* __restrict__ dst,
    unsigned long long n,
    unsigned long long p)
{
    unsigned long long row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned long long col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= p) return;
    double ww = w[row];
    if (ww < 0.0) ww = 0.0;
    dst[col * n + row] = ww * src[col * n + row];
}
"#;

    let n = x.nrows();
    let p = x.ncols();
    let ctx = CudaContext::new(0).map_err(|e| GpuError::DriverCallFailed {
        reason: format!("xtwx context init failed: {e}"),
    })?;
    let stream = ctx.default_stream();
    let blas = CudaBlas::new(stream.clone()).map_err(|e| GpuError::DriverCallFailed {
        reason: format!("xtwx cublas init failed: {e}"),
    })?;

    // Move X in column-major order to match cuBLAS expectations.
    let mut col_major = Vec::<f64>::with_capacity(n * p);
    for j in 0..p {
        for i in 0..n {
            col_major.push(x[[i, j]]);
        }
    }
    let dx = stream.memcpy_stod(&col_major).map_err(map_drv)?;
    let dw = stream
        .memcpy_stod(w.as_slice().ok_or_else(|| GpuError::DriverCallFailed {
            reason: "xtwx weight slice not contiguous".to_string(),
        })?)
        .map_err(map_drv)?;
    let mut dwx = stream.alloc_zeros::<f64>(n * p).map_err(map_drv)?;

    let ptx = compile_ptx(ROW_SCALE).map_err(|e| GpuError::DriverCallFailed {
        reason: format!("xtwx NVRTC compile failed: {e}"),
    })?;
    let module = ctx.load_module(ptx).map_err(|e| GpuError::DriverCallFailed {
        reason: format!("xtwx load module failed: {e}"),
    })?;
    let scale_fn = module
        .load_function("row_scale_kernel")
        .map_err(|e| GpuError::DriverCallFailed {
            reason: format!("xtwx load function failed: {e}"),
        })?;

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
    let mut builder = stream.launch_builder(&scale_fn);
    builder.arg(&dx).arg(&dw).arg(&mut dwx).arg(&nn).arg(&pp);
    // SAFETY: scale_fn was just loaded from a freshly NVRTC-compiled module
    // for this context; all args bound to live device buffers (dx, dw, dwx)
    // and pod u64 scalars (nn, pp); grid/block dims cover n*p threads.
    unsafe { builder.launch(cfg) }.map_err(map_drv)?;

    let mut dout = stream.alloc_zeros::<f64>(p * p).map_err(map_drv)?;
    // Compute (Xᵀ) * (W X) → p×p.  Both operands are column-major n×p.
    let gemm = GemmConfig {
        transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
        transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        m: p as i32,
        n: p as i32,
        k: n as i32,
        alpha: 1.0,
        lda: n as i32,
        ldb: n as i32,
        beta: 0.0,
        ldc: p as i32,
    };
    // SAFETY: cuBLAS gemm requires column-major device buffers of declared
    // dimensions; dx is n×p column-major (caller contract), dwx is the n×p
    // intermediate just written by scale_fn, dout is the p×p zero-initialized
    // target. trans flags + leading-dim values match exactly.
    unsafe { blas.gemm(gemm, &dx, &dwx, &mut dout) }.map_err(|e| GpuError::DriverCallFailed {
        reason: format!("xtwx cublas gemm failed: {e}"),
    })?;

    let host_cm = stream.memcpy_dtov(&dout).map_err(map_drv)?;
    let mut out = Array2::<f64>::zeros((p, p));
    for j in 0..p {
        for i in 0..p {
            out[[i, j]] = host_cm[j * p + i];
        }
    }
    Ok(out)
}

fn map_drv(e: cudarc::driver::DriverError) -> GpuError {
    GpuError::DriverCallFailed {
        reason: format!("xtwx cudarc driver error: {e}"),
    }
}
