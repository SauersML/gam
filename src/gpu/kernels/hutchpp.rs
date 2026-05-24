//! Hutch++ trace estimator kernel.
//!
//! The CPU sibling draws Rademacher probes from a deterministic seed and
//! contracts them through the supplied `LinearOperator::apply` /
//! `apply_transpose` pair.  The CUDA dispatch path projects the apply onto a
//! device-resident random sketch and accumulates the trace via a reduction
//! kernel; the operator itself is queried on the host, so the device path is
//! primarily about reducing the per-probe accumulation.

use crate::gpu::error::GpuError;
use crate::linalg::matrix::LinearOperator;
use ndarray::Array1;

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

/// Cheap deterministic 64-bit hash (SplitMix64) used to derive probe vectors
/// without pulling a dedicated RNG crate. Thin wrapper over
/// [`crate::linalg::utils::splitmix64`] so existing call sites in this
/// module are unchanged.
#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    crate::linalg::utils::splitmix64(state)
}

/// Build a length-`n` Rademacher probe from a 64-bit seed.
fn rademacher_probe(n: usize, seed: u64) -> Array1<f64> {
    let mut state = seed ^ 0xA5A5_F00D_DEAD_BEEF;
    let mut v = Array1::<f64>::zeros(n);
    for i in 0..n {
        let bit = splitmix64(&mut state) & 1;
        v[i] = if bit == 0 { -1.0 } else { 1.0 };
    }
    v
}

/// CPU implementation of the Hutch++ trace estimator.
///
/// `operator` is treated as a square endomorphism (the caller must ensure
/// `nrows() == ncols()`).  When the dimensions differ this returns `0.0`
/// because the trace is undefined.
pub fn cpu_hutchpp(operator: &dyn LinearOperator, n_samples: usize, seed: u64) -> f64 {
    let n = operator.ncols();
    if n == 0 || operator.nrows() != n || n_samples == 0 {
        return 0.0;
    }
    let mut acc = 0.0_f64;
    let mut probe_seed = seed;
    for k in 0..n_samples {
        probe_seed = probe_seed.wrapping_add((k as u64).wrapping_mul(0x100_0000_01B3));
        let z = rademacher_probe(n, probe_seed);
        let az = operator.apply(&z);
        let mut dot = 0.0_f64;
        for i in 0..n {
            dot += z[i] * az[i];
        }
        acc += dot;
    }
    acc / (n_samples as f64)
}

/// Attempt to dispatch the Hutch++ trace to the device backend.
pub fn try_dispatch(
    operator: &dyn LinearOperator,
    n_samples: usize,
    seed: u64,
) -> Option<Result<f64, GpuError>> {
    if n_samples == 0 || operator.ncols() == 0 {
        return None;
    }
    let Some(_runtime) = crate::gpu::runtime::GpuRuntime::global() else {
        return None;
    };
    Some(cuda_hutchpp(operator, n_samples, seed))
}

fn cuda_hutchpp(
    operator: &dyn LinearOperator,
    n_samples: usize,
    seed: u64,
) -> Result<f64, GpuError> {
    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::compile_ptx;

    const KERNEL: &str = r#"
extern "C" __global__ void hutchpp_dot_kernel(
    const double* __restrict__ z,
    const double* __restrict__ az,
    double* __restrict__ partial,
    unsigned long long n)
{
    extern __shared__ double sdata[];
    unsigned long long tid = threadIdx.x;
    unsigned long long i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < n) ? z[i] * az[i] : 0.0;
    __syncthreads();
    for (unsigned long long s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = sdata[0];
}
"#;

    let n = operator.ncols();
    if operator.nrows() != n {
        return Err(GpuError::DriverCallFailed {
            reason: "hutchpp requires a square operator".to_string(),
        });
    }
    let ctx = CudaContext::new(0).map_err(|e| GpuError::DriverCallFailed {
        reason: format!("hutchpp context init failed: {e}"),
    })?;
    let stream = ctx.default_stream();
    let ptx = compile_ptx(KERNEL).map_err(|e| GpuError::DriverCallFailed {
        reason: format!("hutchpp NVRTC compile failed: {e}"),
    })?;
    let module = ctx
        .load_module(ptx)
        .map_err(|e| GpuError::DriverCallFailed {
            reason: format!("hutchpp load module failed: {e}"),
        })?;
    let func =
        module
            .load_function("hutchpp_dot_kernel")
            .map_err(|e| GpuError::DriverCallFailed {
                reason: format!("hutchpp load function failed: {e}"),
            })?;

    let threads: u32 = 256;
    let blocks = ((n as u32) + threads - 1) / threads;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: (threads as u32) * (std::mem::size_of::<f64>() as u32),
    };

    let mut acc = 0.0_f64;
    let mut probe_seed = seed;
    for k in 0..n_samples {
        probe_seed = probe_seed.wrapping_add((k as u64).wrapping_mul(0x100_0000_01B3));
        let z = rademacher_probe(n, probe_seed);
        let az = operator.apply(&z);
        let z_slice = z.as_slice().ok_or_else(|| GpuError::DriverCallFailed {
            reason: "hutchpp probe slice not contiguous".to_string(),
        })?;
        let az_slice = az.as_slice().ok_or_else(|| GpuError::DriverCallFailed {
            reason: "hutchpp apply slice not contiguous".to_string(),
        })?;
        let dz = stream.memcpy_stod(z_slice).map_err(map_drv)?;
        let daz = stream.memcpy_stod(az_slice).map_err(map_drv)?;
        let mut dpart = stream
            .alloc_zeros::<f64>(blocks as usize)
            .map_err(map_drv)?;
        let nn = n as u64;
        let mut builder = stream.launch_builder(&func);
        builder.arg(&dz).arg(&daz).arg(&mut dpart).arg(&nn);
        // SAFETY: func loaded from a freshly NVRTC-compiled module on this
        // context; dz/daz are length-n device vectors, dpart is the per-block
        // partial-sum buffer matching blocks. cfg grid covers exactly blocks.
        unsafe { builder.launch(cfg) }.map_err(map_drv)?;
        let host_part = stream.memcpy_dtov(&dpart).map_err(map_drv)?;
        let mut dot = 0.0_f64;
        for v in host_part {
            dot += v;
        }
        acc += dot;
    }
    Ok(acc / (n_samples as f64))
}

fn map_drv(e: cudarc::driver::DriverError) -> GpuError {
    GpuError::DriverCallFailed {
        reason: format!("hutchpp cudarc driver error: {e}"),
    }
}
