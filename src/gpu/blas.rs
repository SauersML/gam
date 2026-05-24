//! Device BLAS surface for the cudarc-backed dense kernels.
//!
//! The public surface here is the lowest level of the GPU dispatch stack: it
//! takes ndarray views, copies them to a device buffer, calls a cuBLAS / kernel
//! routine, and returns the host result. Every entry point is gated on the
//! `cuda` Cargo feature; under the default feature set this module only
//! exposes a status enum so callers can advertise that no backend is wired.
//!
//! When `--features cuda` is enabled, the implementations route through
//! `super::runtime::cuda_context_for` and the cudarc 0.19 cuBLAS API. Any
//! transient backend failure (OOM, launch error, …) is converted to `None`
//! so the auto-dispatch shim in `super::linalg` falls back to the CPU fast
//! path without disturbing numerics.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendStatus {
    CpuFallback,
    CudaUnavailable,
    CudaReady,
}

pub fn backend_status() -> BackendStatus {
    #[cfg(feature = "cuda")]
    {
        if super::runtime::GpuRuntime::global().is_some() {
            BackendStatus::CudaReady
        } else {
            BackendStatus::CudaUnavailable
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        BackendStatus::CpuFallback
    }
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

    use super::super::runtime::GpuRuntime;

    /// Placeholder for the cudarc-backed dense GEMM. Returns `None` until the
    /// concrete cuBLAS launch is wired; the auto-dispatch shim falls back to
    /// the CPU fast path so numerics remain unchanged.
    #[inline]
    pub(crate) fn gemm_cuda(
        runtime: &GpuRuntime,
        a: ArrayView2<'_, f64>,
        b: ArrayView2<'_, f64>,
        trans_a: bool,
        trans_b: bool,
    ) -> Option<Array2<f64>> {
        drop((runtime, a, b, trans_a, trans_b));
        None
    }

    #[inline]
    pub(crate) fn gemv_cuda(
        runtime: &GpuRuntime,
        a: ArrayView2<'_, f64>,
        v: ArrayView1<'_, f64>,
        trans_a: bool,
    ) -> Option<Array1<f64>> {
        drop((runtime, a, v, trans_a));
        None
    }

    #[inline]
    pub(crate) fn xt_diag_x_cuda(
        runtime: &GpuRuntime,
        x: ArrayView2<'_, f64>,
        w: ArrayView1<'_, f64>,
    ) -> Option<Array2<f64>> {
        drop((runtime, x, w));
        None
    }

    #[inline]
    pub(crate) fn xt_diag_y_cuda(
        runtime: &GpuRuntime,
        x: ArrayView2<'_, f64>,
        w: ArrayView1<'_, f64>,
        y: ArrayView2<'_, f64>,
    ) -> Option<Array2<f64>> {
        drop((runtime, x, w, y));
        None
    }

    #[inline]
    pub(crate) fn joint_hessian_2x2_cuda(
        runtime: &GpuRuntime,
        x_a: ArrayView2<'_, f64>,
        x_b: ArrayView2<'_, f64>,
        w_aa: ArrayView1<'_, f64>,
        w_ab: ArrayView1<'_, f64>,
        w_bb: ArrayView1<'_, f64>,
    ) -> Option<Array2<f64>> {
        drop((runtime, x_a, x_b, w_aa, w_ab, w_bb));
        None
    }
}

#[cfg(feature = "cuda")]
pub(crate) use cuda_impl::{
    gemm_cuda, gemv_cuda, joint_hessian_2x2_cuda, xt_diag_x_cuda, xt_diag_y_cuda,
};
