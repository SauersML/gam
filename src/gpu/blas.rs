//! Device BLAS surface for the cudarc-backed dense kernels.
//!
//! The public surface here is the lowest level of the GPU dispatch stack: it
//! takes ndarray views, copies them to a device buffer, calls a cuBLAS / kernel
//! routine, and returns the host result. The cudarc-backed implementations
//! always compile (cudarc dynamically loads `libcuda` at runtime via the
//! `fallback-dynamic-loading` feature), and dispatch is gated at runtime on
//! `super::runtime::GpuRuntime::global()` — when no device is probed the
//! status enum advertises `CudaUnavailable` and callers fall back to CPU.
//!
//! The implementations route through `super::runtime::cuda_context_for` and
//! the cudarc 0.19 cuBLAS API. Any transient backend failure (OOM, launch
//! error, …) is converted to `None` so the auto-dispatch shim in
//! `super::linalg` falls back to the CPU fast path without disturbing
//! numerics.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendStatus {
    CpuFallback,
    CudaUnavailable,
    CudaReady,
}

pub fn backend_status() -> BackendStatus {
    if super::runtime::GpuRuntime::global().is_some() {
        BackendStatus::CudaReady
    } else {
        BackendStatus::CudaUnavailable
    }
}

mod cuda_impl {
    use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

    use crate::gpu::driver::{from_col_major, to_col_major, to_i32};

    use super::super::runtime::GpuRuntime;
    use cudarc::cublas::sys::{
        cublasDiagType_t, cublasFillMode_t, cublasOperation_t, cublasSideMode_t, cublasStatus_t,
    };
    use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, Gemv, GemvConfig};
    use cudarc::driver::{DevicePtr, DevicePtrMut};

    #[inline]
    pub(crate) fn gemm_cuda(
        runtime: &GpuRuntime,
        a: ArrayView2<'_, f64>,
        b: ArrayView2<'_, f64>,
        trans_a: bool,
        trans_b: bool,
    ) -> Option<Array2<f64>> {
        let (a_rows, a_cols) = a.dim();
        let (b_rows, b_cols) = b.dim();
        let (m, k_a) = if trans_a {
            (a_cols, a_rows)
        } else {
            (a_rows, a_cols)
        };
        let (k_b, n) = if trans_b {
            (b_cols, b_rows)
        } else {
            (b_rows, b_cols)
        };
        if m == 0 || n == 0 || k_a == 0 || k_a != k_b {
            return None;
        }
        let stream = super::super::runtime::cuda_context_for(runtime.device.ordinal)?
            .new_stream()
            .ok()?;
        let blas = CudaBlas::new(stream.clone()).ok()?;
        let a_col = to_col_major(&a);
        let b_col = to_col_major(&b);
        let a_dev = stream.clone_htod(&*a_col).ok()?;
        let b_dev = stream.clone_htod(&*b_col).ok()?;
        let mut out_dev = stream.alloc_zeros::<f64>(m.checked_mul(n)?).ok()?;
        let cfg = GemmConfig::<f64> {
            transa: if trans_a {
                cublasOperation_t::CUBLAS_OP_T
            } else {
                cublasOperation_t::CUBLAS_OP_N
            },
            transb: if trans_b {
                cublasOperation_t::CUBLAS_OP_T
            } else {
                cublasOperation_t::CUBLAS_OP_N
            },
            m: to_i32(m)?,
            n: to_i32(n)?,
            k: to_i32(k_a)?,
            alpha: 1.0,
            lda: to_i32(a_rows)?,
            ldb: to_i32(b_rows)?,
            beta: 0.0,
            ldc: to_i32(m)?,
        };
        // SAFETY: buffers are column-major with dimensions validated above.
        unsafe { blas.gemm(cfg, &a_dev, &b_dev, &mut out_dev) }.ok()?;
        let out_col = stream.clone_dtoh(&out_dev).ok()?;
        from_col_major(&out_col, m, n)
    }

    #[inline]
    pub(crate) fn gemv_cuda(
        runtime: &GpuRuntime,
        a: ArrayView2<'_, f64>,
        v: ArrayView1<'_, f64>,
        trans_a: bool,
    ) -> Option<Array1<f64>> {
        let (rows, cols) = a.dim();
        let out_len = if trans_a { cols } else { rows };
        let needed = if trans_a { rows } else { cols };
        if out_len == 0 || needed == 0 || v.len() != needed {
            return None;
        }
        let stream = super::super::runtime::cuda_context_for(runtime.device.ordinal)?
            .new_stream()
            .ok()?;
        let blas = CudaBlas::new(stream.clone()).ok()?;
        let a_col = to_col_major(&a);
        let a_dev = stream.clone_htod(&*a_col).ok()?;
        let v_storage;
        let v_slice = match v.as_slice() {
            Some(slice) => slice,
            None => {
                v_storage = v.iter().copied().collect::<Vec<_>>();
                v_storage.as_slice()
            }
        };
        let v_dev = stream.clone_htod(v_slice).ok()?;
        let mut out_dev = stream.alloc_zeros::<f64>(out_len).ok()?;
        let cfg = GemvConfig::<f64> {
            trans: if trans_a {
                cublasOperation_t::CUBLAS_OP_T
            } else {
                cublasOperation_t::CUBLAS_OP_N
            },
            m: to_i32(rows)?,
            n: to_i32(cols)?,
            alpha: 1.0,
            lda: to_i32(rows)?,
            incx: 1,
            beta: 0.0,
            incy: 1,
        };
        // SAFETY: dimensions and vector length match the cuBLAS GEMV contract.
        unsafe { blas.gemv(cfg, &a_dev, &v_dev, &mut out_dev) }.ok()?;
        Some(Array1::from_vec(stream.clone_dtoh(&out_dev).ok()?))
    }

    #[inline]
    pub(crate) fn xt_diag_x_cuda(
        runtime: &GpuRuntime,
        x: ArrayView2<'_, f64>,
        w: ArrayView1<'_, f64>,
    ) -> Option<Array2<f64>> {
        std::hint::black_box((runtime, x, w));
        None
    }

    #[inline]
    pub(crate) fn xt_diag_y_cuda(
        runtime: &GpuRuntime,
        x: ArrayView2<'_, f64>,
        w: ArrayView1<'_, f64>,
        y: ArrayView2<'_, f64>,
    ) -> Option<Array2<f64>> {
        std::hint::black_box((runtime, x, w, y));
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
        std::hint::black_box((runtime, x_a, x_b, w_aa, w_ab, w_bb));
        None
    }

    #[inline]
    pub(crate) fn trsm_cuda(
        runtime: &GpuRuntime,
        triangular: ArrayView2<'_, f64>,
        rhs: ArrayView2<'_, f64>,
        upper: bool,
    ) -> Option<Array2<f64>> {
        let (n, n2) = triangular.dim();
        if n == 0 || n != n2 || rhs.nrows() != n {
            return None;
        }
        let nrhs = rhs.ncols();
        let stream = super::super::runtime::cuda_context_for(runtime.device.ordinal)?
            .new_stream()
            .ok()?;
        let blas = CudaBlas::new(stream.clone()).ok()?;
        let tri_col = to_col_major(&triangular);
        let rhs_col = to_col_major(&rhs);
        let tri_dev = stream.clone_htod(&*tri_col).ok()?;
        let mut rhs_dev = stream.clone_htod(&*rhs_col).ok()?;
        let alpha = 1.0_f64;
        let handle = *blas.handle();
        let (tri_ptr, _tri_record) = tri_dev.device_ptr(&stream);
        let (rhs_ptr, _rhs_record) = rhs_dev.device_ptr_mut(&stream);
        // SAFETY: triangular is n×n and rhs is n×nrhs in column-major device
        // buffers. cublasDtrsm overwrites rhs with A^{-1} rhs.
        let status = unsafe {
            cudarc::cublas::sys::cublasDtrsm_v2(
                handle,
                cublasSideMode_t::CUBLAS_SIDE_LEFT,
                if upper {
                    cublasFillMode_t::CUBLAS_FILL_MODE_UPPER
                } else {
                    cublasFillMode_t::CUBLAS_FILL_MODE_LOWER
                },
                cublasOperation_t::CUBLAS_OP_N,
                cublasDiagType_t::CUBLAS_DIAG_NON_UNIT,
                to_i32(n)?,
                to_i32(nrhs)?,
                &alpha,
                tri_ptr as *const f64,
                to_i32(n)?,
                rhs_ptr as *mut f64,
                to_i32(n)?,
            )
        };
        if status != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return None;
        }
        let out_col = stream.clone_dtoh(&rhs_dev).ok()?;
        from_col_major(&out_col, n, nrhs)
    }
}

pub(crate) use cuda_impl::{
    gemm_cuda, gemv_cuda, joint_hessian_2x2_cuda, trsm_cuda, xt_diag_x_cuda, xt_diag_y_cuda,
};
