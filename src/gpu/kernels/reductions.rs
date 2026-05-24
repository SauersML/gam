//! Generic reduction kernel (sum / dot-with-self / max / abs-max / l2-norm).
//!
//! The CPU sibling implements each reduction directly; the CUDA dispatch path
//! launches a two-pass shared-memory reduction.  The `ReductionKind` enum is
//! defined locally so the kernel module remains self-contained.

use crate::gpu::error::GpuError;
use ndarray::ArrayView1;

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

/// Supported reduction operations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReductionKind {
    Sum,
    DotSelf,
    Max,
    AbsMax,
    L2Norm,
}

/// CPU sibling: scalar reduction over a contiguous vector view.
pub fn cpu_reduce(values: ArrayView1<'_, f64>, kind: ReductionKind) -> f64 {
    if values.is_empty() {
        return match kind {
            ReductionKind::Max => f64::NEG_INFINITY,
            ReductionKind::AbsMax => 0.0,
            _ => 0.0,
        };
    }
    match kind {
        ReductionKind::Sum => values.iter().copied().sum(),
        ReductionKind::DotSelf => values.iter().map(|v| v * v).sum(),
        ReductionKind::Max => values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        ReductionKind::AbsMax => values.iter().map(|v| v.abs()).fold(0.0, f64::max),
        ReductionKind::L2Norm => values.iter().map(|v| v * v).sum::<f64>().sqrt(),
    }
}

/// Attempt to dispatch the reduction to the device backend.
pub fn try_dispatch(
    values: ArrayView1<'_, f64>,
    kind: ReductionKind,
) -> Option<Result<f64, GpuError>> {
    if values.is_empty() {
        return None;
    }
    let Some(_runtime) = crate::gpu::runtime::GpuRuntime::global() else {
        return None;
    };
    Some(cuda_reduce(values, kind))
}

fn cuda_reduce(values: ArrayView1<'_, f64>, kind: ReductionKind) -> Result<f64, GpuError> {
    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::compile_ptx;

    // op_code: 0=sum, 1=dot-self, 2=max, 3=abs-max, 4=l2norm (sum-of-squares;
    // host takes the sqrt).
    const KERNEL: &str = r#"
extern "C" __global__ void reduce_kernel(
    const double* __restrict__ x,
    double* __restrict__ partial,
    unsigned long long n,
    int op_code)
{
    extern __shared__ double sdata[];
    unsigned long long tid = threadIdx.x;
    unsigned long long i = blockIdx.x * blockDim.x + tid;
    double v;
    if (i < n) {
        double xi = x[i];
        if (op_code == 0) v = xi;
        else if (op_code == 1) v = xi * xi;
        else if (op_code == 2) v = xi;
        else if (op_code == 3) v = fabs(xi);
        else v = xi * xi;
    } else {
        if (op_code == 2) v = -1.0e308;
        else v = 0.0;
    }
    sdata[tid] = v;
    __syncthreads();
    for (unsigned long long s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            double a = sdata[tid];
            double b = sdata[tid + s];
            if (op_code == 2 || op_code == 3) sdata[tid] = (a > b) ? a : b;
            else sdata[tid] = a + b;
        }
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = sdata[0];
}
"#;

    let n = values.len();
    let ctx = CudaContext::new(0).map_err(|e| GpuError::DriverCallFailed {
        reason: format!("reductions context init failed: {e}"),
    })?;
    let stream = ctx.default_stream();
    let ptx = compile_ptx(KERNEL).map_err(|e| GpuError::DriverCallFailed {
        reason: format!("reductions NVRTC compile failed: {e}"),
    })?;
    let module = ctx.load_module(ptx).map_err(|e| GpuError::DriverCallFailed {
        reason: format!("reductions load module failed: {e}"),
    })?;
    let func = module
        .load_function("reduce_kernel")
        .map_err(|e| GpuError::DriverCallFailed {
            reason: format!("reductions load function failed: {e}"),
        })?;

    let slice = values.as_slice().ok_or_else(|| GpuError::DriverCallFailed {
        reason: "reductions input slice not contiguous".to_string(),
    })?;
    let dx = stream.memcpy_stod(slice).map_err(map_drv)?;

    let threads: u32 = 256;
    let blocks = ((n as u32) + threads - 1) / threads;
    let mut dpart = stream
        .alloc_zeros::<f64>(blocks as usize)
        .map_err(map_drv)?;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: (threads as u32) * (std::mem::size_of::<f64>() as u32),
    };
    let op_code: i32 = match kind {
        ReductionKind::Sum => 0,
        ReductionKind::DotSelf => 1,
        ReductionKind::Max => 2,
        ReductionKind::AbsMax => 3,
        ReductionKind::L2Norm => 4,
    };
    let nn = n as u64;
    let mut builder = stream.launch_builder(&func);
    builder
        .arg(&dx)
        .arg(&mut dpart)
        .arg(&nn)
        .arg(&op_code);
    // NVRTC func fresh; dx len-n; dpart per-block partial sums.
    // Grid covers exactly the kernel's bounded tid range.
    // SAFETY: all kernel-arg lifetimes + bounds checked above.
    unsafe { builder.launch(cfg) }.map_err(map_drv)?;

    let host_part = stream.memcpy_dtov(&dpart).map_err(map_drv)?;
    let result = match kind {
        ReductionKind::Sum | ReductionKind::DotSelf => host_part.iter().copied().sum::<f64>(),
        ReductionKind::Max => host_part.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        ReductionKind::AbsMax => host_part.iter().copied().fold(0.0, f64::max),
        ReductionKind::L2Norm => host_part.iter().copied().sum::<f64>().sqrt(),
    };
    Ok(result)
}

fn map_drv(e: cudarc::driver::DriverError) -> GpuError {
    GpuError::DriverCallFailed {
        reason: format!("reductions cudarc driver error: {e}"),
    }
}
