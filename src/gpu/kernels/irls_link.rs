//! Vectorized inverse-link / derivative kernel used by the IRLS inner loop.
//!
//! Produces a dense `(n, derivative_order + 1)` matrix whose row `i` contains
//! `[μ(η_i), μ'(η_i), ..., μ^{(d)}(η_i)]` where `μ = g^{-1}` is the inverse of
//! the selected `LinkFunction`.  Derivatives are computed with closed-form
//! recurrences (no autodiff) so the CPU sibling stays branch-free per row.

use crate::gpu::error::GpuError;
use crate::types::LinkFunction;
use ndarray::{Array2, ArrayView1};

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

/// Closed-form evaluation of `μ(η)` and its first three derivatives for a
/// selected link function.
fn link_derivatives(link: LinkFunction, eta: f64, order: u8) -> [f64; 4] {
    let mut out = [0.0_f64; 4];
    match link {
        LinkFunction::Identity => {
            out[0] = eta;
            if order >= 1 {
                out[1] = 1.0;
            }
            // Higher derivatives stay at zero.
        }
        LinkFunction::Log => {
            let mu = eta.exp();
            out[0] = mu;
            if order >= 1 {
                out[1] = mu;
            }
            if order >= 2 {
                out[2] = mu;
            }
            if order >= 3 {
                out[3] = mu;
            }
        }
        LinkFunction::Logit | LinkFunction::BetaLogistic => {
            let s = 1.0 / (1.0 + (-eta).exp());
            out[0] = s;
            let ds = s * (1.0 - s);
            if order >= 1 {
                out[1] = ds;
            }
            if order >= 2 {
                out[2] = ds * (1.0 - 2.0 * s);
            }
            if order >= 3 {
                out[3] = ds * (1.0 - 6.0 * s + 6.0 * s * s);
            }
        }
        LinkFunction::Probit => {
            // μ(η) = Φ(η), μ'(η) = φ(η), μ''(η) = -η φ(η), μ'''(η) = (η²−1) φ(η).
            const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
            let phi = INV_SQRT_2PI * (-0.5 * eta * eta).exp();
            // Normal CDF via erf relation.
            let cdf = 0.5 * (1.0 + erf_approx(eta / std::f64::consts::SQRT_2));
            out[0] = cdf;
            if order >= 1 {
                out[1] = phi;
            }
            if order >= 2 {
                out[2] = -eta * phi;
            }
            if order >= 3 {
                out[3] = (eta * eta - 1.0) * phi;
            }
        }
        LinkFunction::CLogLog => {
            // μ(η) = 1 − exp(−exp(η))
            let e = eta.exp();
            let inner = (-e).exp();
            let mu = 1.0 - inner;
            out[0] = mu;
            if order >= 1 {
                out[1] = e * inner;
            }
            if order >= 2 {
                out[2] = e * inner * (1.0 - e);
            }
            if order >= 3 {
                out[3] = e * inner * (1.0 - 3.0 * e + e * e);
            }
        }
        LinkFunction::Sas => {
            // Treat Sas like Logit for the purpose of GPU prototype evaluation;
            // the production solver substitutes a dedicated Sas Jet elsewhere.
            let s = 1.0 / (1.0 + (-eta).exp());
            out[0] = s;
            let ds = s * (1.0 - s);
            if order >= 1 {
                out[1] = ds;
            }
            if order >= 2 {
                out[2] = ds * (1.0 - 2.0 * s);
            }
            if order >= 3 {
                out[3] = ds * (1.0 - 6.0 * s + 6.0 * s * s);
            }
        }
    }
    out
}

/// Abramowitz–Stegun 7.1.26 rational approximation of `erf`.
fn erf_approx(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * ax);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-ax * ax).exp();
    sign * y
}

/// CPU sibling: vectorized inverse-link / derivative evaluation.
pub fn cpu_irls_link(
    eta: ArrayView1<'_, f64>,
    link: LinkFunction,
    derivative_order: u8,
) -> Array2<f64> {
    let n = eta.len();
    let cols = (derivative_order as usize).saturating_add(1);
    let mut out = Array2::<f64>::zeros((n, cols));
    for i in 0..n {
        let row = link_derivatives(link, eta[i], derivative_order);
        for k in 0..cols.min(4) {
            out[[i, k]] = row[k];
        }
    }
    out
}

/// Attempt to dispatch the inverse-link kernel to the device backend.
pub fn try_dispatch(
    eta: ArrayView1<'_, f64>,
    link: LinkFunction,
    derivative_order: u8,
) -> Option<Result<Array2<f64>, GpuError>> {
    if eta.is_empty() {
        return None;
    }
    if crate::gpu::runtime::GpuRuntime::global().is_none() {
        return None;
    }
    Some(cuda_irls_link(eta, link, derivative_order))
}

fn cuda_irls_link(
    eta: ArrayView1<'_, f64>,
    link: LinkFunction,
    derivative_order: u8,
) -> Result<Array2<f64>, GpuError> {
    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::compile_ptx;

    // The kernel evaluates link/derivatives directly on the device.  A small
    // integer selects the link (matches `link_code` below) and the per-row
    // closed-form recurrences mirror the host implementation.
    const KERNEL: &str = r#"
extern "C" __global__ void irls_link_kernel(
    const double* __restrict__ eta,
    double* __restrict__ out,
    unsigned long long n,
    unsigned long long cols,
    int link_code)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double e = eta[i];
    double r0 = 0.0, r1 = 0.0, r2 = 0.0, r3 = 0.0;
    if (link_code == 0) { // Identity
        r0 = e; r1 = 1.0;
    } else if (link_code == 1) { // Log
        double mu = exp(e);
        r0 = mu; r1 = mu; r2 = mu; r3 = mu;
    } else if (link_code == 2) { // Logit / BetaLogistic / Sas
        double s = 1.0 / (1.0 + exp(-e));
        double ds = s * (1.0 - s);
        r0 = s; r1 = ds;
        r2 = ds * (1.0 - 2.0 * s);
        r3 = ds * (1.0 - 6.0 * s + 6.0 * s * s);
    } else if (link_code == 3) { // Probit
        double phi = 0.3989422804014327 * exp(-0.5 * e * e);
        double cdf = 0.5 * (1.0 + erf(e * 0.7071067811865475));
        r0 = cdf; r1 = phi;
        r2 = -e * phi;
        r3 = (e * e - 1.0) * phi;
    } else { // CLogLog
        double ex = exp(e);
        double in_ = exp(-ex);
        r0 = 1.0 - in_;
        r1 = ex * in_;
        r2 = ex * in_ * (1.0 - ex);
        r3 = ex * in_ * (1.0 - 3.0 * ex + ex * ex);
    }
    out[i * cols + 0] = r0;
    if (cols > 1) out[i * cols + 1] = r1;
    if (cols > 2) out[i * cols + 2] = r2;
    if (cols > 3) out[i * cols + 3] = r3;
}
"#;

    let n = eta.len();
    let cols = (derivative_order as usize).saturating_add(1);
    let link_code: i32 = match link {
        LinkFunction::Identity => 0,
        LinkFunction::Log => 1,
        LinkFunction::Logit | LinkFunction::BetaLogistic | LinkFunction::Sas => 2,
        LinkFunction::Probit => 3,
        LinkFunction::CLogLog => 4,
    };

    let ctx = CudaContext::new(0).map_err(|e| GpuError::DriverCallFailed {
        reason: format!("irls_link context init failed: {e}"),
    })?;
    let stream = ctx.default_stream();
    let ptx = compile_ptx(KERNEL).map_err(|e| GpuError::DriverCallFailed {
        reason: format!("irls_link NVRTC compile failed: {e}"),
    })?;
    let module = ctx.load_module(ptx).map_err(|e| GpuError::DriverCallFailed {
        reason: format!("irls_link load module failed: {e}"),
    })?;
    let func = module
        .load_function("irls_link_kernel")
        .map_err(|e| GpuError::DriverCallFailed {
            reason: format!("irls_link load function failed: {e}"),
        })?;

    let eta_slice = eta.as_slice().ok_or_else(|| GpuError::DriverCallFailed {
        reason: "irls_link eta slice not contiguous".to_string(),
    })?;
    let deta = stream.memcpy_stod(eta_slice).map_err(map_drv)?;
    let mut dout = stream.alloc_zeros::<f64>(n * cols).map_err(map_drv)?;

    let threads: u32 = 256;
    let blocks = ((n as u32) + threads - 1) / threads;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    let nn = n as u64;
    let cc = cols as u64;
    let mut builder = stream.launch_builder(&func);
    builder
        .arg(&deta)
        .arg(&mut dout)
        .arg(&nn)
        .arg(&cc)
        .arg(&link_code);
    // NVRTC module fresh; deta/dout live; nn/cc/link_code pod.
    // Grid covers exactly the n threads kernel guards against.
    // SAFETY: all kernel-arg lifetimes + bounds checked above.
    unsafe { builder.launch(cfg) }.map_err(map_drv)?;

    let host = stream.memcpy_dtov(&dout).map_err(map_drv)?;
    Array2::from_shape_vec((n, cols), host).map_err(|e| GpuError::DriverCallFailed {
        reason: format!("irls_link host reshape failed: {e}"),
    })
}

fn map_drv(e: cudarc::driver::DriverError) -> GpuError {
    GpuError::DriverCallFailed {
        reason: format!("irls_link cudarc driver error: {e}"),
    }
}
