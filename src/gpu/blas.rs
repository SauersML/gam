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
    use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
    use std::sync::Arc;

    #[inline]
    fn stream_and_blas(runtime: &GpuRuntime) -> Option<(Arc<CudaStream>, CudaBlas)> {
        let stream = super::super::runtime::cuda_context_for(runtime.device.ordinal)?
            .new_stream()
            .ok()?;
        let blas = CudaBlas::new(stream.clone()).ok()?;
        Some((stream, blas))
    }

    #[inline]
    fn vector_values(v: ArrayView1<'_, f64>) -> Vec<f64> {
        v.iter().copied().collect()
    }

    #[inline]
    fn row_scale_device(
        blas: &CudaBlas,
        stream: &Arc<CudaStream>,
        matrix_dev: &CudaSlice<f64>,
        weights_dev: &CudaSlice<f64>,
        scaled_dev: &mut CudaSlice<f64>,
        rows: usize,
        cols: usize,
    ) -> Option<()> {
        let rows_i = to_i32(rows)?;
        let cols_i = to_i32(cols)?;
        let handle = *blas.handle();
        let (matrix_ptr, _matrix_record) = matrix_dev.device_ptr(stream);
        let (weights_ptr, _weights_record) = weights_dev.device_ptr(stream);
        let (scaled_ptr, _scaled_record) = scaled_dev.device_ptr_mut(stream);
        // SAFETY: all device slices are on this stream/context. `matrix_dev`
        // and `scaled_dev` are rows×cols column-major matrices with lda/ldc
        // equal to rows; `weights_dev` has one contiguous value per row.
        let status = unsafe {
            cudarc::cublas::sys::cublasDdgmm(
                handle,
                cublasSideMode_t::CUBLAS_SIDE_LEFT,
                rows_i,
                cols_i,
                matrix_ptr as *const f64,
                rows_i,
                weights_ptr as *const f64,
                1,
                scaled_ptr as *mut f64,
                rows_i,
            )
        };
        if status == cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            Some(())
        } else {
            None
        }
    }

    #[inline]
    fn weighted_crossprod(
        runtime: &GpuRuntime,
        left: ArrayView2<'_, f64>,
        weights: ArrayView1<'_, f64>,
        right: ArrayView2<'_, f64>,
    ) -> Option<Array2<f64>> {
        let (rows, left_cols) = left.dim();
        let (right_rows, right_cols) = right.dim();
        if rows == 0
            || left_cols == 0
            || right_cols == 0
            || rows != right_rows
            || rows != weights.len()
        {
            return None;
        }

        let (stream, blas) = stream_and_blas(runtime)?;
        let left_col = to_col_major(&left);
        let right_col = to_col_major(&right);
        let weights_host = vector_values(weights);
        let left_dev = stream.clone_htod(&*left_col).ok()?;
        let right_dev = stream.clone_htod(&*right_col).ok()?;
        let weights_dev = stream.clone_htod(&weights_host).ok()?;
        let mut weighted_right_dev = stream
            .alloc_zeros::<f64>(rows.checked_mul(right_cols)?)
            .ok()?;
        row_scale_device(
            &blas,
            &stream,
            &right_dev,
            &weights_dev,
            &mut weighted_right_dev,
            rows,
            right_cols,
        )?;

        let mut out_dev = stream
            .alloc_zeros::<f64>(left_cols.checked_mul(right_cols)?)
            .ok()?;
        let cfg = GemmConfig::<f64> {
            transa: cublasOperation_t::CUBLAS_OP_T,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: to_i32(left_cols)?,
            n: to_i32(right_cols)?,
            k: to_i32(rows)?,
            alpha: 1.0,
            lda: to_i32(rows)?,
            ldb: to_i32(rows)?,
            beta: 0.0,
            ldc: to_i32(left_cols)?,
        };
        // SAFETY: cfg computes leftᵀ (left_cols×rows) times weighted_right
        // (rows×right_cols) into a left_cols×right_cols column-major output.
        unsafe { blas.gemm(cfg, &left_dev, &weighted_right_dev, &mut out_dev) }.ok()?;
        let out_col = stream.clone_dtoh(&out_dev).ok()?;
        from_col_major(&out_col, left_cols, right_cols)
    }

    #[inline]
    fn assign_block(
        out: &mut Array2<f64>,
        row_offset: usize,
        col_offset: usize,
        block: &Array2<f64>,
    ) {
        let (rows, cols) = block.dim();
        for col in 0..cols {
            for row in 0..rows {
                out[[row_offset + row, col_offset + col]] = block[[row, col]];
            }
        }
    }

    #[inline]
    fn mirror_upper_to_lower(out: &mut Array2<f64>) {
        let n = out.nrows();
        for row in 0..n {
            for col in 0..row {
                out[[row, col]] = out[[col, row]];
            }
        }
    }

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
        let (stream, blas) = stream_and_blas(runtime)?;
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
        let (stream, blas) = stream_and_blas(runtime)?;
        let a_col = to_col_major(&a);
        let a_dev = stream.clone_htod(&*a_col).ok()?;
        let v_host = vector_values(v);
        let v_dev = stream.clone_htod(&v_host).ok()?;
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
        let (rows, cols) = x.dim();
        if rows == 0 || cols == 0 || rows != w.len() {
            return None;
        }
        weighted_crossprod(runtime, x, w, x)
    }

    #[inline]
    pub(crate) fn xt_diag_y_cuda(
        runtime: &GpuRuntime,
        x: ArrayView2<'_, f64>,
        w: ArrayView1<'_, f64>,
        y: ArrayView2<'_, f64>,
    ) -> Option<Array2<f64>> {
        weighted_crossprod(runtime, x, w, y)
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
        let (rows, pa) = x_a.dim();
        let (rows_b, pb) = x_b.dim();
        let total = pa.checked_add(pb)?;
        if rows == 0
            || total == 0
            || rows != rows_b
            || rows != w_aa.len()
            || rows != w_ab.len()
            || rows != w_bb.len()
        {
            return None;
        }

        let mut out = Array2::<f64>::zeros((total, total));
        if pa > 0 {
            let aa = weighted_crossprod(runtime, x_a, w_aa, x_a)?;
            assign_block(&mut out, 0, 0, &aa);
        }
        if pa > 0 && pb > 0 {
            let ab = weighted_crossprod(runtime, x_a, w_ab, x_b)?;
            assign_block(&mut out, 0, pa, &ab);
        }
        if pb > 0 {
            let bb = weighted_crossprod(runtime, x_b, w_bb, x_b)?;
            assign_block(&mut out, pa, pa, &bb);
        }
        mirror_upper_to_lower(&mut out);
        Some(out)
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
        let (stream, blas) = stream_and_blas(runtime)?;
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
