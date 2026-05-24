//! Pairwise spatial kernel matrix construction (`K[i, j] = k(a_i, b_j)`).
//!
//! `SpatialKernel` is defined locally so this module remains self-contained.
//! The CPU sibling materializes the dense `(n_a, n_b)` Gram via a triple loop.
//! The CUDA dispatch path executes the same evaluation through an NVRTC
//! kernel parameterized on the same enum codes.

use crate::gpu::error::GpuError;
use ndarray::{Array2, ArrayView2};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendStatus {
    CpuFallback,
    CudaUnavailable,
    CudaReady,
}

#[inline]
pub fn backend_status() -> BackendStatus {
    if crate::gpu::runtime::GpuRuntime::global().is_some() {
        BackendStatus::CudaReady
    } else {
        BackendStatus::CudaUnavailable
    }
}

/// Supported spatial kernel forms.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SpatialKernel {
    /// `exp(-‖a − b‖² / (2 σ²))`
    SquaredExponential { sigma: f64 },
    /// `exp(-‖a − b‖ / ℓ)`
    Exponential { length_scale: f64 },
    /// Matérn ν = 3/2: `(1 + √3 r / ℓ) · exp(-√3 r / ℓ)`
    Matern32 { length_scale: f64 },
    /// Plain inner product `aᵀ b`.
    InnerProduct,
}

fn squared_distance(a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>, i: usize, j: usize) -> f64 {
    let d = a.ncols().min(b.ncols());
    let mut s = 0.0_f64;
    for k in 0..d {
        let diff = a[[i, k]] - b[[j, k]];
        s += diff * diff;
    }
    s
}

fn inner_product(a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>, i: usize, j: usize) -> f64 {
    let d = a.ncols().min(b.ncols());
    let mut s = 0.0_f64;
    for k in 0..d {
        s += a[[i, k]] * b[[j, k]];
    }
    s
}

/// CPU sibling: build the pairwise kernel matrix.
pub fn cpu_spatial(
    kernel: SpatialKernel,
    points_a: ArrayView2<'_, f64>,
    points_b: ArrayView2<'_, f64>,
) -> Array2<f64> {
    let na = points_a.nrows();
    let nb = points_b.nrows();
    let mut out = Array2::<f64>::zeros((na, nb));
    for i in 0..na {
        for j in 0..nb {
            out[[i, j]] = match kernel {
                SpatialKernel::SquaredExponential { sigma } => {
                    let denom = 2.0 * sigma * sigma;
                    if denom <= 0.0 {
                        0.0
                    } else {
                        (-squared_distance(points_a, points_b, i, j) / denom).exp()
                    }
                }
                SpatialKernel::Exponential { length_scale } => {
                    if length_scale <= 0.0 {
                        0.0
                    } else {
                        let r = squared_distance(points_a, points_b, i, j).sqrt();
                        (-r / length_scale).exp()
                    }
                }
                SpatialKernel::Matern32 { length_scale } => {
                    if length_scale <= 0.0 {
                        0.0
                    } else {
                        let r = squared_distance(points_a, points_b, i, j).sqrt();
                        let s = 3.0_f64.sqrt() * r / length_scale;
                        (1.0 + s) * (-s).exp()
                    }
                }
                SpatialKernel::InnerProduct => inner_product(points_a, points_b, i, j),
            };
        }
    }
    out
}

/// Attempt to dispatch the spatial-kernel build to the device backend.
pub fn try_dispatch(
    kernel: SpatialKernel,
    points_a: ArrayView2<'_, f64>,
    points_b: ArrayView2<'_, f64>,
) -> Option<Result<Array2<f64>, GpuError>> {
    if points_a.nrows() == 0 || points_b.nrows() == 0 || points_a.ncols() != points_b.ncols() {
        return None;
    }
    if crate::gpu::runtime::GpuRuntime::global().is_none() {
        return None;
    }
    Some(cuda_spatial(kernel, points_a, points_b))
}

fn cuda_spatial(
    kernel: SpatialKernel,
    points_a: ArrayView2<'_, f64>,
    points_b: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, GpuError> {
    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::compile_ptx;

    const KERNEL: &str = r#"
extern "C" __global__ void spatial_kernel(
    const double* __restrict__ a,
    const double* __restrict__ b,
    double* __restrict__ out,
    unsigned long long na,
    unsigned long long nb,
    unsigned long long d,
    int kind,
    double p1)
{
    unsigned long long i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned long long j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= na || j >= nb) return;
    double s_dot = 0.0;
    double s_sqd = 0.0;
    for (unsigned long long k = 0; k < d; ++k) {
        double av = a[i * d + k];
        double bv = b[j * d + k];
        double diff = av - bv;
        s_sqd += diff * diff;
        s_dot += av * bv;
    }
    double val = 0.0;
    if (kind == 0) {
        double denom = 2.0 * p1 * p1;
        val = (denom > 0.0) ? exp(-s_sqd / denom) : 0.0;
    } else if (kind == 1) {
        val = (p1 > 0.0) ? exp(-sqrt(s_sqd) / p1) : 0.0;
    } else if (kind == 2) {
        if (p1 > 0.0) {
            double r = sqrt(s_sqd);
            double s = sqrt(3.0) * r / p1;
            val = (1.0 + s) * exp(-s);
        } else {
            val = 0.0;
        }
    } else {
        val = s_dot;
    }
    out[i * nb + j] = val;
}
"#;

    let na = points_a.nrows();
    let nb = points_b.nrows();
    let d = points_a.ncols();
    let ctx = CudaContext::new(0).map_err(|e| GpuError::DriverCallFailed {
        reason: format!("spatial context init failed: {e}"),
    })?;
    let stream = ctx.default_stream();
    let ptx = compile_ptx(KERNEL).map_err(|e| GpuError::DriverCallFailed {
        reason: format!("spatial NVRTC compile failed: {e}"),
    })?;
    let module = ctx
        .load_module(ptx)
        .map_err(|e| GpuError::DriverCallFailed {
            reason: format!("spatial load module failed: {e}"),
        })?;
    let func = module
        .load_function("spatial_kernel")
        .map_err(|e| GpuError::DriverCallFailed {
            reason: format!("spatial load function failed: {e}"),
        })?;

    let mut a_host = Vec::<f64>::with_capacity(na * d);
    let mut b_host = Vec::<f64>::with_capacity(nb * d);
    for i in 0..na {
        for k in 0..d {
            a_host.push(points_a[[i, k]]);
        }
    }
    for j in 0..nb {
        for k in 0..d {
            b_host.push(points_b[[j, k]]);
        }
    }
    let da = stream.memcpy_stod(&a_host).map_err(map_drv)?;
    let db = stream.memcpy_stod(&b_host).map_err(map_drv)?;
    let mut dout = stream.alloc_zeros::<f64>(na * nb).map_err(map_drv)?;

    let (kind_code, p1): (i32, f64) = match kernel {
        SpatialKernel::SquaredExponential { sigma } => (0, sigma),
        SpatialKernel::Exponential { length_scale } => (1, length_scale),
        SpatialKernel::Matern32 { length_scale } => (2, length_scale),
        SpatialKernel::InnerProduct => (3, 0.0),
    };

    let tx: u32 = 16;
    let ty: u32 = 16;
    let bx = ((nb as u32) + tx - 1) / tx;
    let by = ((na as u32) + ty - 1) / ty;
    let cfg = LaunchConfig {
        grid_dim: (bx, by, 1),
        block_dim: (tx, ty, 1),
        shared_mem_bytes: 0,
    };
    let nna = na as u64;
    let nnb = nb as u64;
    let dd = d as u64;
    let mut builder = stream.launch_builder(&func);
    builder
        .arg(&da)
        .arg(&db)
        .arg(&mut dout)
        .arg(&nna)
        .arg(&nnb)
        .arg(&dd)
        .arg(&kind_code)
        .arg(&p1);
    // NVRTC func fresh; da/db/dout sized na*d/nb*d/na*nb; pods match.
    // Grid covers na*nb threads with kernel's bounds check.
    // SAFETY: all kernel-arg lifetimes + bounds checked above.
    unsafe { builder.launch(cfg) }.map_err(map_drv)?;

    let host = stream.memcpy_dtov(&dout).map_err(map_drv)?;
    Array2::from_shape_vec((na, nb), host).map_err(|e| GpuError::DriverCallFailed {
        reason: format!("spatial host reshape failed: {e}"),
    })
}

fn map_drv(e: cudarc::driver::DriverError) -> GpuError {
    GpuError::DriverCallFailed {
        reason: format!("spatial cudarc driver error: {e}"),
    }
}
