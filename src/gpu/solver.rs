//! cuSOLVER-backed dense solver kernels for the GPU HAL.
//!
//! This module owns CUDA solver functionality that is shared by GPU linear
//! algebra dispatch and higher-level solver code. CPU solves do not live behind
//! these entry points: unavailable CUDA support is reported as an error.

use ndarray::{Array2, ArrayView2};

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

#[cfg(feature = "cuda")]
mod cuda {
    use crate::gpu::driver::{from_col_major, to_col_major};
    use cudarc::cusolver::{DnHandle, sys as cusolver_sys};
    use cudarc::driver::{CudaContext, CudaSlice, DevicePtr, DevicePtrMut};
    use ndarray::{Array2, ArrayView2};

    pub(super) fn cholesky_solve(
        hessian: ArrayView2<'_, f64>,
        rhs: ArrayView2<'_, f64>,
    ) -> Result<(Array2<f64>, f64), String> {
        let (ctx, stream) = context_and_stream()?;
        let (p, p2) = hessian.dim();
        if p == 0 || p != p2 || rhs.nrows() != p {
            return Err("Cholesky solve dimension mismatch".to_string());
        }
        let nrhs = rhs.ncols();
        let solver = DnHandle::new(stream.clone()).map_err(|e| format!("cusolver init: {e}"))?;
        let h_col = to_col_major(&hessian);
        let rhs_col = to_col_major(&rhs);
        let mut h_dev = pinned_htod(&ctx, &stream, &h_col)?;
        let mut rhs_dev = pinned_htod(&ctx, &stream, &rhs_col)?;
        potrf_in_place(&solver, &stream, p, &mut h_dev)?;
        potrs_in_place(&solver, &stream, p, nrhs, &h_dev, &mut rhs_dev)?;
        let factor_col = stream
            .clone_dtoh(&h_dev)
            .map_err(|e| format!("download Cholesky factor: {e}"))?;
        let out_col = stream
            .clone_dtoh(&rhs_dev)
            .map_err(|e| format!("download solution: {e}"))?;
        let solved =
            from_col_major(&out_col, p, nrhs).ok_or("solution layout conversion failed")?;
        Ok((solved, cholesky_logdet_from_col_major(&factor_col, p)))
    }

    pub(super) fn cholesky_lower(hessian: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
        let (ctx, stream) = context_and_stream()?;
        let (p, p2) = hessian.dim();
        if p == 0 || p != p2 {
            return Err("Cholesky factorization dimension mismatch".to_string());
        }
        let solver = DnHandle::new(stream.clone()).map_err(|e| format!("cusolver init: {e}"))?;
        let h_col = to_col_major(&hessian);
        let mut h_dev = pinned_htod(&ctx, &stream, &h_col)?;
        potrf_in_place(&solver, &stream, p, &mut h_dev)?;
        let factor_col = stream
            .clone_dtoh(&h_dev)
            .map_err(|e| format!("download Cholesky factor: {e}"))?;
        let mut lower =
            from_col_major(&factor_col, p, p).ok_or("factor layout conversion failed")?;
        for row in 0..p {
            for col in (row + 1)..p {
                lower[[row, col]] = 0.0;
            }
        }
        Ok(lower)
    }

    pub(crate) fn context_and_stream() -> Result<
        (
            std::sync::Arc<CudaContext>,
            std::sync::Arc<cudarc::driver::CudaStream>,
        ),
        String,
    > {
        let ctx = CudaContext::new(0).map_err(|e| format!("cuda context: {e}"))?;
        let stream = ctx.new_stream().map_err(|e| format!("cuda stream: {e}"))?;
        Ok((ctx, stream))
    }

    pub(crate) fn pinned_htod<
        T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Copy,
    >(
        ctx: &std::sync::Arc<CudaContext>,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        src: &[T],
    ) -> Result<CudaSlice<T>, String> {
        // SAFETY: alloc_pinned reserves uninitialized pinned host memory of src.len() T elements;
        // we immediately copy_from_slice over the entire region below before any device read.
        let mut pinned = unsafe { ctx.alloc_pinned::<T>(src.len()) }
            .map_err(|e| format!("pinned host alloc: {e}"))?;
        pinned
            .as_mut_slice()
            .map_err(|e| format!("pinned host slice: {e}"))?
            .copy_from_slice(src);
        stream
            .clone_htod(&pinned)
            .map_err(|e| format!("cuda H2D: {e}"))
    }

    pub(crate) fn potrf_in_place(
        solver: &DnHandle,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        p: usize,
        h: &mut CudaSlice<f64>,
    ) -> Result<(), String> {
        let p_i = to_i32(p)?;
        let uplo = cusolver_sys::cublasFillMode_t::CUBLAS_FILL_MODE_LOWER;
        let mut lwork = 0_i32;
        {
            let (h_ptr, _h_record) = h.device_ptr_mut(stream);
            // SAFETY: cuSOLVER buffer-size query; h_ptr is a live p*p device buffer, lwork is a
            // valid mutable host i32, solver handle is initialized.
            let status = unsafe {
                cusolver_sys::cusolverDnDpotrf_bufferSize(
                    solver.cu(),
                    uplo,
                    p_i,
                    h_ptr as *mut f64,
                    p_i,
                    &mut lwork,
                )
            };
            check_cusolver(status, "cusolverDnDpotrf_bufferSize")?;
        }
        let mut workspace = stream
            .alloc_zeros::<f64>(usize::try_from(lwork).map_err(|_| "negative potrf workspace")?)
            .map_err(|e| format!("cuda alloc potrf workspace: {e}"))?;
        let mut info = stream
            .alloc_zeros::<i32>(1)
            .map_err(|e| format!("cuda alloc potrf info: {e}"))?;
        {
            let (h_ptr, _h_record) = h.device_ptr_mut(stream);
            let (work_ptr, _work_record) = workspace.device_ptr_mut(stream);
            let (info_ptr, _info_record) = info.device_ptr_mut(stream);
            // SAFETY: cuSOLVER potrf factorization; h is p*p, workspace was allocated with the
            // lwork size reported by the buffer-size query above, info is a 1-element device i32
            // buffer.
            let status = unsafe {
                cusolver_sys::cusolverDnDpotrf(
                    solver.cu(),
                    uplo,
                    p_i,
                    h_ptr as *mut f64,
                    p_i,
                    work_ptr as *mut f64,
                    lwork,
                    info_ptr as *mut i32,
                )
            };
            check_cusolver(status, "cusolverDnDpotrf")?;
        }
        let info_host = stream
            .clone_dtoh(&info)
            .map_err(|e| format!("download potrf info: {e}"))?;
        if info_host[0] == 0 {
            Ok(())
        } else {
            Err(format!("cusolverDnDpotrf returned info={}", info_host[0]))
        }
    }

    pub(crate) fn potrs_in_place(
        solver: &DnHandle,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        p: usize,
        nrhs: usize,
        h: &CudaSlice<f64>,
        rhs: &mut CudaSlice<f64>,
    ) -> Result<(), String> {
        let p_i = to_i32(p)?;
        let nrhs_i = to_i32(nrhs)?;
        let uplo = cusolver_sys::cublasFillMode_t::CUBLAS_FILL_MODE_LOWER;
        let mut info = stream
            .alloc_zeros::<i32>(1)
            .map_err(|e| format!("cuda alloc potrs info: {e}"))?;
        {
            let (h_ptr, _h_record) = h.device_ptr(stream);
            let (rhs_ptr, _rhs_record) = rhs.device_ptr_mut(stream);
            let (info_ptr, _info_record) = info.device_ptr_mut(stream);
            // SAFETY: cuSOLVER potrs solve; h is a p*p Cholesky factor from potrf above, rhs is
            // p*nrhs, info is a 1-element device i32 buffer, leading dims match column-major p_i.
            let status = unsafe {
                cusolver_sys::cusolverDnDpotrs(
                    solver.cu(),
                    uplo,
                    p_i,
                    nrhs_i,
                    h_ptr as *const f64,
                    p_i,
                    rhs_ptr as *mut f64,
                    p_i,
                    info_ptr as *mut i32,
                )
            };
            check_cusolver(status, "cusolverDnDpotrs")?;
        }
        let info_host = stream
            .clone_dtoh(&info)
            .map_err(|e| format!("download potrs info: {e}"))?;
        if info_host[0] == 0 {
            Ok(())
        } else {
            Err(format!("cusolverDnDpotrs returned info={}", info_host[0]))
        }
    }

    pub(crate) fn cholesky_logdet_from_col_major(factor: &[f64], p: usize) -> f64 {
        let mut acc = 0.0_f64;
        for i in 0..p {
            acc += factor[i + i * p].abs().ln();
        }
        2.0 * acc
    }

    fn check_cusolver(
        status: cusolver_sys::cusolverStatus_t,
        label: &'static str,
    ) -> Result<(), String> {
        if status == cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
            Ok(())
        } else {
            Err(format!("{label} failed with {status:?}"))
        }
    }

    fn to_i32(value: usize) -> Result<i32, String> {
        i32::try_from(value).map_err(|_| format!("CUDA dimension {value} exceeds i32"))
    }
}

#[cfg(feature = "cuda")]
pub(crate) use cuda::{
    cholesky_logdet_from_col_major, context_and_stream, pinned_htod, potrf_in_place, potrs_in_place,
};

#[cfg(feature = "cuda")]
pub fn cholesky_solve_gpu(
    hessian: ArrayView2<'_, f64>,
    rhs: ArrayView2<'_, f64>,
) -> Result<(Array2<f64>, f64), String> {
    cuda::cholesky_solve(hessian, rhs)
}

#[cfg(not(feature = "cuda"))]
pub fn cholesky_solve_gpu(
    hessian: ArrayView2<'_, f64>,
    rhs: ArrayView2<'_, f64>,
) -> Result<(Array2<f64>, f64), String> {
    std::hint::black_box((hessian, rhs));
    Err("cuda feature is not enabled".to_string())
}

#[cfg(feature = "cuda")]
pub fn cholesky_lower_gpu(hessian: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
    cuda::cholesky_lower(hessian)
}

#[cfg(not(feature = "cuda"))]
pub fn cholesky_lower_gpu(hessian: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
    std::hint::black_box(hessian);
    Err("cuda feature is not enabled".to_string())
}
