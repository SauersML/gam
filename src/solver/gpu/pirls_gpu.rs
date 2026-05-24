use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

#[derive(Clone, Debug)]
pub struct PirlsGpuInput<'a> {
    pub x: ArrayView2<'a, f64>,
    pub weights: ArrayView1<'a, f64>,
    pub penalty_hessian: ArrayView2<'a, f64>,
    pub gradient: ArrayView1<'a, f64>,
    pub lm_ridge: f64,
}

#[derive(Clone, Debug)]
pub struct PirlsGpuStep {
    pub penalized_hessian: Array2<f64>,
    pub direction: Array1<f64>,
    pub logdet: f64,
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::{PirlsGpuInput, PirlsGpuStep};
    use crate::gpu::driver::{from_col_major, to_col_major};
    use cudarc::cublas::sys::{
        cublasDdgmm, cublasDgeam, cublasOperation_t, cublasSideMode_t, cublasStatus_t,
    };
    use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
    use cudarc::cusolver::{DnHandle, sys as cusolver_sys};
    use cudarc::driver::{CudaContext, CudaSlice, DevicePtr, DevicePtrMut};
    use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

    pub(super) fn weighted_crossprod(
        x: ArrayView2<'_, f64>,
        weights: ArrayView1<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        let (ctx, stream) = context_and_stream()?;
        let (n, p) = validate_design(x, weights)?;
        let blas = CudaBlas::new(stream.clone()).map_err(|e| format!("cublas init: {e}"))?;
        let x_col = to_col_major(&x);
        let x_dev = pinned_htod(&ctx, &stream, &x_col)?;
        let mut w_dev = pinned_htod(
            &ctx,
            &stream,
            weights.as_slice().ok_or("weights must be contiguous")?,
        )?;
        let mut wx_dev = stream
            .alloc_zeros::<f64>(n.checked_mul(p).ok_or("X size overflow")?)
            .map_err(|e| format!("cuda alloc WX: {e}"))?;
        left_scale_rows(&blas, &stream, n, p, &x_dev, &mut w_dev, &mut wx_dev)?;
        let mut h_dev = stream
            .alloc_zeros::<f64>(p.checked_mul(p).ok_or("H size overflow")?)
            .map_err(|e| format!("cuda alloc H: {e}"))?;
        let n_i = to_i32(n)?;
        let p_i = to_i32(p)?;
        let cfg = GemmConfig::<f64> {
            transa: cublasOperation_t::CUBLAS_OP_T,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: p_i,
            n: p_i,
            k: n_i,
            alpha: 1.0,
            lda: n_i,
            ldb: n_i,
            beta: 0.0,
            ldc: p_i,
        };
        // SAFETY: cuBLAS dgemm with validated i32 dimensions; x_dev/wx_dev are n*p f64 device
        // buffers and h_dev is the p*p output, all allocated above with matching sizes.
        unsafe { blas.gemm(cfg, &x_dev, &wx_dev, &mut h_dev) }
            .map_err(|e| format!("cublas dgemm XtWX: {e}"))?;
        let h_col = stream
            .clone_dtoh(&h_dev)
            .map_err(|e| format!("download H: {e}"))?;
        from_col_major(&h_col, p, p).ok_or_else(|| "H layout conversion failed".to_string())
    }

    pub(super) fn solve_step(input: PirlsGpuInput<'_>) -> Result<PirlsGpuStep, String> {
        let (ctx, stream) = context_and_stream()?;
        let (n, p) = validate_design(input.x, input.weights)?;
        if input.penalty_hessian.dim() != (p, p) {
            return Err(format!(
                "penalty Hessian shape {:?} does not match p={p}",
                input.penalty_hessian.dim()
            ));
        }
        if input.gradient.len() != p {
            return Err(format!(
                "gradient length {} does not match p={p}",
                input.gradient.len()
            ));
        }
        let blas = CudaBlas::new(stream.clone()).map_err(|e| format!("cublas init: {e}"))?;
        let solver = DnHandle::new(stream.clone()).map_err(|e| format!("cusolver init: {e}"))?;
        let x_col = to_col_major(&input.x);
        let x_dev = pinned_htod(&ctx, &stream, &x_col)?;
        let mut w_dev = pinned_htod(
            &ctx,
            &stream,
            input
                .weights
                .as_slice()
                .ok_or("weights must be contiguous")?,
        )?;
        let mut wx_dev = stream
            .alloc_zeros::<f64>(n.checked_mul(p).ok_or("X size overflow")?)
            .map_err(|e| format!("cuda alloc WX: {e}"))?;
        left_scale_rows(&blas, &stream, n, p, &x_dev, &mut w_dev, &mut wx_dev)?;
        let mut xtwx_dev = stream
            .alloc_zeros::<f64>(p.checked_mul(p).ok_or("H size overflow")?)
            .map_err(|e| format!("cuda alloc XtWX: {e}"))?;
        let n_i = to_i32(n)?;
        let p_i = to_i32(p)?;
        let cfg = GemmConfig::<f64> {
            transa: cublasOperation_t::CUBLAS_OP_T,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: p_i,
            n: p_i,
            k: n_i,
            alpha: 1.0,
            lda: n_i,
            ldb: n_i,
            beta: 0.0,
            ldc: p_i,
        };
        // SAFETY: cuBLAS dgemm with validated i32 dimensions; x_dev/wx_dev are n*p f64 device
        // buffers and xtwx_dev is the p*p output, all allocated above with matching sizes.
        unsafe { blas.gemm(cfg, &x_dev, &wx_dev, &mut xtwx_dev) }
            .map_err(|e| format!("cublas dgemm XtWX: {e}"))?;

        let penalty = penalty_with_ridge(input.penalty_hessian, input.lm_ridge);
        let penalty_view = penalty.view();
        let penalty_col = to_col_major(&penalty_view);
        let penalty_dev = pinned_htod(&ctx, &stream, &penalty_col)?;
        let mut h_dev = stream
            .alloc_zeros::<f64>(p.checked_mul(p).ok_or("H size overflow")?)
            .map_err(|e| format!("cuda alloc H total: {e}"))?;
        geam_add(&blas, &stream, p, &xtwx_dev, &penalty_dev, &mut h_dev)?;

        let mut rhs_col = vec![0.0_f64; p];
        rhs_col.copy_from_slice(
            input
                .gradient
                .as_slice()
                .ok_or("gradient must be contiguous")?,
        );
        let mut rhs_dev = pinned_htod(&ctx, &stream, &rhs_col)?;
        let h_total_col = stream
            .clone_dtoh(&h_dev)
            .map_err(|e| format!("download penalized Hessian: {e}"))?;
        let penalized_hessian =
            from_col_major(&h_total_col, p, p).ok_or("H layout conversion failed")?;
        potrf_in_place(&solver, &stream, p, &mut h_dev)?;
        potrs_in_place(&solver, &stream, p, 1, &h_dev, &mut rhs_dev)?;
        let mut direction = Array1::from_vec(
            stream
                .clone_dtoh(&rhs_dev)
                .map_err(|e| format!("download direction: {e}"))?,
        );
        direction.mapv_inplace(|v| -v);
        let h_factor_col = stream
            .clone_dtoh(&h_dev)
            .map_err(|e| format!("download Cholesky factor: {e}"))?;
        let logdet = cholesky_logdet_from_col_major(&h_factor_col, p);
        Ok(PirlsGpuStep {
            penalized_hessian,
            direction,
            logdet,
        })
    }

    pub(super) fn cholesky_solve(
        hessian: ArrayView2<'_, f64>,
        rhs: ArrayView2<'_, f64>,
    ) -> Result<(Array2<f64>, f64), String> {
        let (ctx, stream) = context_and_stream()?;
        let (p, p2) = hessian.dim();
        if p != p2 || rhs.nrows() != p {
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

    fn context_and_stream() -> Result<
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

    fn validate_design(
        x: ArrayView2<'_, f64>,
        weights: ArrayView1<'_, f64>,
    ) -> Result<(usize, usize), String> {
        let (n, p) = x.dim();
        if weights.len() != n {
            return Err(format!(
                "weights length {} does not match rows {n}",
                weights.len()
            ));
        }
        if n == 0 || p == 0 {
            return Err("empty design cannot be solved on CUDA".to_string());
        }
        Ok((n, p))
    }

    fn pinned_htod<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Copy>(
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

    fn left_scale_rows(
        blas: &CudaBlas,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        n: usize,
        p: usize,
        x_dev: &CudaSlice<f64>,
        w_dev: &mut CudaSlice<f64>,
        wx_dev: &mut CudaSlice<f64>,
    ) -> Result<(), String> {
        let n_i = to_i32(n)?;
        let p_i = to_i32(p)?;
        let handle = *blas.handle();
        let (x_ptr, _x_record) = x_dev.device_ptr(stream);
        let (w_ptr, _w_record) = w_dev.device_ptr(stream);
        let (wx_ptr, _wx_record) = wx_dev.device_ptr_mut(stream);
        // SAFETY: FFI call into cuBLAS; pointers come from live CudaSlice device buffers sized
        // n*p (x, wx) and n (w), leading dims match column-major layout, handle is valid.
        let status = unsafe {
            cublasDdgmm(
                handle,
                cublasSideMode_t::CUBLAS_SIDE_LEFT,
                n_i,
                p_i,
                x_ptr as *const f64,
                n_i,
                w_ptr as *const f64,
                1,
                wx_ptr as *mut f64,
                n_i,
            )
        };
        if status == cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            Ok(())
        } else {
            Err(format!("cublasDdgmm failed with {status:?}"))
        }
    }

    fn geam_add(
        blas: &CudaBlas,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        p: usize,
        a: &CudaSlice<f64>,
        b: &CudaSlice<f64>,
        out: &mut CudaSlice<f64>,
    ) -> Result<(), String> {
        let p_i = to_i32(p)?;
        let alpha = 1.0_f64;
        let beta = 1.0_f64;
        let handle = *blas.handle();
        let (a_ptr, _a_record) = a.device_ptr(stream);
        let (b_ptr, _b_record) = b.device_ptr(stream);
        let (out_ptr, _out_record) = out.device_ptr_mut(stream);
        // SAFETY: FFI call into cuBLAS geam; a, b, out are live p*p device buffers in column-major
        // with leading dim p_i, scalars live on host stack, handle is valid.
        let status = unsafe {
            cublasDgeam(
                handle,
                cublasOperation_t::CUBLAS_OP_N,
                cublasOperation_t::CUBLAS_OP_N,
                p_i,
                p_i,
                &alpha,
                a_ptr as *const f64,
                p_i,
                &beta,
                b_ptr as *const f64,
                p_i,
                out_ptr as *mut f64,
                p_i,
            )
        };
        if status == cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            Ok(())
        } else {
            Err(format!("cublasDgeam failed with {status:?}"))
        }
    }

    fn potrf_in_place(
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

    fn potrs_in_place(
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

    fn penalty_with_ridge(penalty: ArrayView2<'_, f64>, ridge: f64) -> Array2<f64> {
        let mut out = penalty.to_owned();
        if ridge != 0.0 {
            for i in 0..out.nrows().min(out.ncols()) {
                out[[i, i]] += ridge;
            }
        }
        out
    }

    fn cholesky_logdet_from_col_major(factor: &[f64], p: usize) -> f64 {
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
pub fn weighted_crossprod_gpu(
    x: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
) -> Result<Array2<f64>, String> {
    cuda::weighted_crossprod(x, weights)
}

#[cfg(not(feature = "cuda"))]
pub fn weighted_crossprod_gpu(
    x: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
) -> Result<Array2<f64>, String> {
    std::hint::black_box((x, weights));
    Err("cuda feature is not enabled".to_string())
}

#[cfg(feature = "cuda")]
pub fn solve_pirls_step_gpu(input: PirlsGpuInput<'_>) -> Result<PirlsGpuStep, String> {
    cuda::solve_step(input)
}

#[cfg(not(feature = "cuda"))]
pub fn solve_pirls_step_gpu(input: PirlsGpuInput<'_>) -> Result<PirlsGpuStep, String> {
    drop(input);
    Err("cuda feature is not enabled".to_string())
}

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
