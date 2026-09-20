//! cuSOLVER-backed dense solver kernels for the GPU HAL.
//!
//! This module owns CUDA solver functionality that is shared by GPU linear
//! algebra dispatch and higher-level solver code. CPU solves do not live behind
//! these entry points: unavailable CUDA support is reported as an error.

use ndarray::{Array2, ArrayView2};

#[cfg(target_os = "linux")]
mod cuda {
    use crate::driver::{from_col_major, to_col_major};
    use cudarc::cublas::sys as cublas_sys;
    use cudarc::cublas::{CudaBlas, Gemv, GemvConfig};
    use cudarc::cusolver::{DnHandle, sys as cusolver_sys};
    use cudarc::driver::{CudaContext, CudaSlice, DevicePtr, DevicePtrMut};
    use faer::MatRef;
    use gam_linalg::faer_ndarray::cholesky_factor_logdet;
    use ndarray::{Array2, ArrayView2};

    pub(super) fn cholesky_solve(
        hessian: ArrayView2<'_, f64>,
        rhs: ArrayView2<'_, f64>,
    ) -> Result<(Array2<f64>, f64), String> {
        let (_, stream) = context_and_stream()?;
        let (p, p2) = hessian.dim();
        if p == 0 || p != p2 || rhs.nrows() != p {
            return Err("Cholesky solve dimension mismatch".to_string());
        }
        let nrhs = rhs.ncols();
        let solver = DnHandle::new(stream.clone()).map_err(|e| format!("cusolver init: {e}"))?;
        let h_col = to_col_major(&hessian);
        let rhs_col = to_col_major(&rhs);
        let mut h_dev = pinned_htod(&stream, &h_col)?;
        let mut rhs_dev = pinned_htod(&stream, &rhs_col)?;
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

    /// fp64 log-determinant of an SPD matrix via POTRF only.
    ///
    /// This is [`cholesky_solve`] stripped of the triangular solve (POTRS) and
    /// the solution download/layout conversion: the log-determinant depends
    /// solely on the Cholesky factor's diagonal, so when a caller already holds
    /// the solution (e.g. from fp32 + iterative refinement) and needs *only* an
    /// accurate fp64 logdet, doing a full solve here would burn an O(p²·nrhs)
    /// POTRS plus a host round-trip on a solution that is immediately discarded.
    pub(super) fn cholesky_logdet(hessian: ArrayView2<'_, f64>) -> Result<f64, String> {
        let (_, stream) = context_and_stream()?;
        let (p, p2) = hessian.dim();
        if p == 0 || p != p2 {
            return Err("Cholesky logdet dimension mismatch".to_string());
        }
        let solver = DnHandle::new(stream.clone()).map_err(|e| format!("cusolver init: {e}"))?;
        let h_col = to_col_major(&hessian);
        let mut h_dev = pinned_htod(&stream, &h_col)?;
        potrf_in_place(&solver, &stream, p, &mut h_dev)?;
        let factor_col = stream
            .clone_dtoh(&h_dev)
            .map_err(|e| format!("download Cholesky factor: {e}"))?;
        Ok(cholesky_logdet_from_col_major(&factor_col, p))
    }

    pub(super) fn cholesky_lower_on_ordinal(
        ordinal: usize,
        hessian: ArrayView2<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        let (_, stream) = context_and_stream_for(ordinal)?;
        cholesky_lower_on_stream(hessian, &stream)
    }

    fn cholesky_lower_on_stream(
        hessian: ArrayView2<'_, f64>,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    ) -> Result<Array2<f64>, String> {
        let (p, p2) = hessian.dim();
        if p == 0 || p != p2 {
            return Err("Cholesky factorization dimension mismatch".to_string());
        }
        let solver = DnHandle::new(stream.clone()).map_err(|e| format!("cusolver init: {e}"))?;
        let h_col = to_col_major(&hessian);
        let mut h_dev = pinned_htod(&stream, &h_col)?;
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

    // -----------------------------------------------------------------------
    // Precision-generic Cholesky scaffold
    //
    // POTRF / POTRS host scaffolds are identical across single and double
    // precision apart from the cuSOLVER symbol called and the device pointer
    // type. `CholScalar` selects those per-precision pieces so the host-side
    // allocation / info-handling / error-formatting logic lives once. The
    // `Dpotr*` (f64) and `Spotr*` (f32) entry points below are thin wrappers
    // over the generic helpers, preserving their public signatures byte for
    // byte.
    // -----------------------------------------------------------------------

    /// cuSOLVER scalar abstraction: selects the precision-specific POTRF/POTRS
    /// symbols and the precision tag used in deferred-info error messages.
    ///
    /// The FFI into cuSOLVER lives inside the trait methods' bodies (in `unsafe`
    /// blocks), so the trait and its methods are safe to call: each impl wires
    /// its method bodies to the cuSOLVER entry points whose pointer arguments
    /// match `Self` (e.g. `cusolverDnDpotrf` for `f64`). Implementors must keep
    /// that pairing consistent — the device pointer passed in is typed `*mut
    /// Self` / `*const Self`, so a mismatched symbol would hand cuSOLVER a
    /// wrongly-typed buffer.
    pub(crate) trait CholScalar:
        cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Copy
    {
        /// cuSOLVER `*potrf_bufferSize`: `(handle, uplo, n, A, lda, *lwork)`.
        ///
        /// `a` is a live `n*n` column-major device buffer of type `Self`,
        /// `lwork` is a host out-param. The unsafe FFI call is contained in the
        /// method body.
        fn potrf_buffer_size(
            handle: cusolver_sys::cusolverDnHandle_t,
            uplo: cusolver_sys::cublasFillMode_t,
            n: i32,
            a: *mut Self,
            lda: i32,
            lwork: *mut i32,
        ) -> cusolver_sys::cusolverStatus_t;
        /// cuSOLVER `*potrf`: `(handle, uplo, n, A, lda, work, lwork, info)`.
        ///
        /// Pointer args must reference live device buffers of the documented
        /// shape; the unsafe FFI call is contained in the method body.
        fn potrf(
            handle: cusolver_sys::cusolverDnHandle_t,
            uplo: cusolver_sys::cublasFillMode_t,
            n: i32,
            a: *mut Self,
            lda: i32,
            work: *mut Self,
            lwork: i32,
            info: *mut i32,
        ) -> cusolver_sys::cusolverStatus_t;
        /// cuSOLVER `*potrs`: `(handle, uplo, n, nrhs, A, lda, B, ldb, info)`.
        ///
        /// Pointer args must reference live device buffers of the documented
        /// shape; the unsafe FFI call is contained in the method body.
        fn potrs(
            handle: cusolver_sys::cusolverDnHandle_t,
            uplo: cusolver_sys::cublasFillMode_t,
            n: i32,
            nrhs: i32,
            a: *const Self,
            lda: i32,
            b: *mut Self,
            ldb: i32,
            info: *mut i32,
        ) -> cusolver_sys::cusolverStatus_t;
        /// Symbol name fragment for error messages (e.g. `"Dpotrf"`).
        const POTRF_NAME: &'static str;
        const POTRS_NAME: &'static str;
        /// Trailing clause appended to a POTRF "not SPD" error (e.g.
        /// `" (matrix not SPD at f32)"`); empty for f64.
        const POTRF_FAIL_SUFFIX: &'static str;
    }

    impl CholScalar for f64 {
        fn potrf_buffer_size(
            handle: cusolver_sys::cusolverDnHandle_t,
            uplo: cusolver_sys::cublasFillMode_t,
            n: i32,
            a: *mut f64,
            lda: i32,
            lwork: *mut i32,
        ) -> cusolver_sys::cusolverStatus_t {
            // SAFETY: caller guarantees `a` is a live n*n column-major f64 device
            // buffer and `lwork` is a valid host out-param; symbol matches f64.
            unsafe { cusolver_sys::cusolverDnDpotrf_bufferSize(handle, uplo, n, a, lda, lwork) }
        }
        fn potrf(
            handle: cusolver_sys::cusolverDnHandle_t,
            uplo: cusolver_sys::cublasFillMode_t,
            n: i32,
            a: *mut f64,
            lda: i32,
            work: *mut f64,
            lwork: i32,
            info: *mut i32,
        ) -> cusolver_sys::cusolverStatus_t {
            // SAFETY: caller guarantees `a` is a live n*n column-major f64 buffer,
            // `work` was sized by potrf_buffer_size, `info` is a 1-element i32
            // device buffer; symbol matches f64.
            unsafe { cusolver_sys::cusolverDnDpotrf(handle, uplo, n, a, lda, work, lwork, info) }
        }
        fn potrs(
            handle: cusolver_sys::cusolverDnHandle_t,
            uplo: cusolver_sys::cublasFillMode_t,
            n: i32,
            nrhs: i32,
            a: *const f64,
            lda: i32,
            b: *mut f64,
            ldb: i32,
            info: *mut i32,
        ) -> cusolver_sys::cusolverStatus_t {
            // SAFETY: caller guarantees `a` is a live n*n f64 Cholesky factor,
            // `b` is n*nrhs column-major f64, `info` is a 1-element i32 device
            // buffer; symbol matches f64.
            unsafe { cusolver_sys::cusolverDnDpotrs(handle, uplo, n, nrhs, a, lda, b, ldb, info) }
        }
        const POTRF_NAME: &'static str = "Dpotrf";
        const POTRS_NAME: &'static str = "Dpotrs";
        const POTRF_FAIL_SUFFIX: &'static str = "";
    }

    impl CholScalar for f32 {
        fn potrf_buffer_size(
            handle: cusolver_sys::cusolverDnHandle_t,
            uplo: cusolver_sys::cublasFillMode_t,
            n: i32,
            a: *mut f32,
            lda: i32,
            lwork: *mut i32,
        ) -> cusolver_sys::cusolverStatus_t {
            // SAFETY: caller guarantees `a` is a live n*n column-major f32 device
            // buffer and `lwork` is a valid host out-param; symbol matches f32.
            unsafe { cusolver_sys::cusolverDnSpotrf_bufferSize(handle, uplo, n, a, lda, lwork) }
        }
        fn potrf(
            handle: cusolver_sys::cusolverDnHandle_t,
            uplo: cusolver_sys::cublasFillMode_t,
            n: i32,
            a: *mut f32,
            lda: i32,
            work: *mut f32,
            lwork: i32,
            info: *mut i32,
        ) -> cusolver_sys::cusolverStatus_t {
            // SAFETY: caller guarantees `a` is a live n*n column-major f32 buffer,
            // `work` was sized by potrf_buffer_size, `info` is a 1-element i32
            // device buffer; symbol matches f32.
            unsafe { cusolver_sys::cusolverDnSpotrf(handle, uplo, n, a, lda, work, lwork, info) }
        }
        fn potrs(
            handle: cusolver_sys::cusolverDnHandle_t,
            uplo: cusolver_sys::cublasFillMode_t,
            n: i32,
            nrhs: i32,
            a: *const f32,
            lda: i32,
            b: *mut f32,
            ldb: i32,
            info: *mut i32,
        ) -> cusolver_sys::cusolverStatus_t {
            // SAFETY: caller guarantees `a` is a live n*n f32 Cholesky factor,
            // `b` is n*nrhs column-major f32, `info` is a 1-element i32 device
            // buffer; symbol matches f32.
            unsafe { cusolver_sys::cusolverDnSpotrs(handle, uplo, n, nrhs, a, lda, b, ldb, info) }
        }
        const POTRF_NAME: &'static str = "Spotrf";
        const POTRS_NAME: &'static str = "Spotrs";
        const POTRF_FAIL_SUFFIX: &'static str = " (matrix not SPD at f32)";
    }

    /// Query the cuSOLVER POTRF workspace size (element count) for a p×p
    /// matrix at precision `T`. Allocates a temporary p×p dummy buffer for the
    /// query.
    fn potrf_bufsize_generic<T: CholScalar>(
        solver: &DnHandle,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        p: usize,
    ) -> Result<usize, String> {
        let p_i = to_i32(p)?;
        let uplo = cusolver_sys::cublasFillMode_t::CUBLAS_FILL_MODE_LOWER;
        let mut lwork = 0_i32;
        let mut dummy = stream
            .alloc_zeros::<T>(p.checked_mul(p).ok_or("p² overflow in lwork query")?)
            .map_err(|e| format!("cuda alloc dummy for lwork query: {e}"))?;
        {
            let (ptr, _rec) = dummy.device_ptr_mut(stream);
            // dummy is a live p*p device buffer of type T, lwork is a host i32;
            // the unsafe cuSOLVER FFI is contained in T::potrf_buffer_size.
            let status =
                T::potrf_buffer_size(solver.cu(), uplo, p_i, ptr as *mut T, p_i, &mut lwork);
            check_cusolver(status, "cusolverDn*potrf_bufferSize")?;
        }
        usize::try_from(lwork).map_err(|_| "negative potrf lwork".to_string())
    }

    /// Factor a p×p SPD device buffer in-place (lower-triangular Cholesky) at
    /// precision `T`, querying and allocating its own workspace. Returns `Err`
    /// if the matrix is singular/indefinite at precision `T`.
    ///
    /// This is the single-matrix POTRF core shared across the GPU layer:
    /// `solver.rs`'s `potrf_in_place`/`spotrf_in_place` and `linalg.rs`'s
    /// `potrf_lower_in_place` all route through it (the latter mapping the
    /// `Result` to its `Option` contract at the boundary). The batched POTRF
    /// (`cusolverDnDpotrfBatched`) in `linalg.rs` is intentionally separate.
    pub(crate) fn potrf_in_place_generic<T: CholScalar>(
        solver: &DnHandle,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        p: usize,
        a: &mut CudaSlice<T>,
    ) -> Result<(), String> {
        let p_i = to_i32(p)?;
        let lwork = potrf_bufsize_generic::<T>(solver, stream, p)?;
        let lwork_i = i32::try_from(lwork).map_err(|_| "negative potrf workspace".to_string())?;
        let mut workspace = stream
            .alloc_zeros::<T>(lwork.max(1))
            .map_err(|e| format!("cuda alloc potrf workspace: {e}"))?;
        let mut info = stream
            .alloc_zeros::<i32>(1)
            .map_err(|e| format!("cuda alloc potrf info: {e}"))?;
        let uplo = cusolver_sys::cublasFillMode_t::CUBLAS_FILL_MODE_LOWER;
        {
            let (a_ptr, _a_rec) = a.device_ptr_mut(stream);
            let (work_ptr, _work_rec) = workspace.device_ptr_mut(stream);
            let (info_ptr, _info_rec) = info.device_ptr_mut(stream);
            // a is p*p col-major T, workspace was sized by T::potrf_buffer_size,
            // info is a 1-element i32 device buffer; the unsafe cuSOLVER FFI is
            // contained in T::potrf.
            let status = T::potrf(
                solver.cu(),
                uplo,
                p_i,
                a_ptr as *mut T,
                p_i,
                work_ptr as *mut T,
                lwork_i,
                info_ptr as *mut i32,
            );
            check_cusolver(status, "cusolverDn*potrf")?;
        }
        let info_host = stream
            .clone_dtoh(&info)
            .map_err(|e| format!("download potrf info: {e}"))?;
        if info_host[0] == 0 {
            Ok(())
        } else {
            Err(format!(
                "cusolverDn{} returned info={}{}",
                T::POTRF_NAME,
                info_host[0],
                T::POTRF_FAIL_SUFFIX
            ))
        }
    }

    /// Triangular solve using a pre-factored Cholesky lower-triangle at
    /// precision `T`. Solves `A · x = rhs` in-place into `rhs` (column-major,
    /// p × nrhs), allocating and downloading its own info scalar.
    fn potrs_in_place_generic<T: CholScalar>(
        solver: &DnHandle,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        p: usize,
        nrhs: usize,
        factor: &CudaSlice<T>,
        rhs: &mut CudaSlice<T>,
    ) -> Result<(), String> {
        let p_i = to_i32(p)?;
        let nrhs_i = to_i32(nrhs)?;
        let uplo = cusolver_sys::cublasFillMode_t::CUBLAS_FILL_MODE_LOWER;
        let mut info = stream
            .alloc_zeros::<i32>(1)
            .map_err(|e| format!("cuda alloc potrs info: {e}"))?;
        {
            let (f_ptr, _f_rec) = factor.device_ptr(stream);
            let (r_ptr, _r_rec) = rhs.device_ptr_mut(stream);
            let (info_ptr, _info_rec) = info.device_ptr_mut(stream);
            // factor is a p*p lower-triangular T from potrf, rhs is p*nrhs
            // col-major T, info is a 1-element i32 device buffer; leading dims
            // match column-major p_i. The unsafe cuSOLVER FFI is contained in
            // T::potrs.
            let status = T::potrs(
                solver.cu(),
                uplo,
                p_i,
                nrhs_i,
                f_ptr as *const T,
                p_i,
                r_ptr as *mut T,
                p_i,
                info_ptr as *mut i32,
            );
            check_cusolver(status, "cusolverDn*potrs")?;
        }
        let info_host = stream
            .clone_dtoh(&info)
            .map_err(|e| format!("download potrs info: {e}"))?;
        if info_host[0] == 0 {
            Ok(())
        } else {
            Err(format!(
                "cusolverDn{} returned info={}",
                T::POTRS_NAME,
                info_host[0]
            ))
        }
    }

    // -----------------------------------------------------------------------
    // fp32 entry points (thin wrappers over the precision-generic scaffold)
    // -----------------------------------------------------------------------

    /// Factor a p×p symmetric positive-definite f32 device buffer in-place
    /// (lower-triangular Cholesky). Returns `Err` if the matrix is
    /// singular/indefinite.
    fn spotrf_in_place(
        solver: &DnHandle,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        p: usize,
        a: &mut CudaSlice<f32>,
    ) -> Result<(), String> {
        potrf_in_place_generic::<f32>(solver, stream, p, a)
    }

    /// Triangular solve using a pre-factored fp32 Cholesky lower-triangle.
    /// Solves `A · x = rhs` in-place into `rhs` (column-major, p × nrhs).
    fn spotrs_in_place(
        solver: &DnHandle,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        p: usize,
        nrhs: usize,
        factor: &CudaSlice<f32>,
        rhs: &mut CudaSlice<f32>,
    ) -> Result<(), String> {
        potrs_in_place_generic::<f32>(solver, stream, p, nrhs, factor, rhs)
    }

    // -----------------------------------------------------------------------
    // fp64 DGEMV residual: r = b − A·x in double precision
    // -----------------------------------------------------------------------

    /// Compute `r = b − A·x` in fp64 where A is p×p and x, b, r are length p.
    ///
    /// Overwrites the output buffer `r_dev` with the residual. Uses
    /// `cublasDgemv` (CUBLAS_OP_N): `r = 1·A·x + 0·0 = A·x`, then the host
    /// subtracts from b. Because p is small here (the policy gates on p ≥ 64
    /// and the Newton system is p×p), downloading the p-vector for the host
    /// subtract is cheap relative to the GEMV.
    fn residual_norm_and_vec(
        blas: &CudaBlas,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        p: usize,
        a_dev: &CudaSlice<f64>,
        x_dev: &CudaSlice<f64>,
        b_host: &[f64],
    ) -> Result<(Vec<f64>, f64), String> {
        let p_i = to_i32(p)?;
        // ax_dev = A · x
        let mut ax_dev = stream
            .alloc_zeros::<f64>(p)
            .map_err(|e| format!("alloc ax: {e}"))?;
        {
            let cfg = GemvConfig::<f64> {
                trans: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                m: p_i,
                n: p_i,
                alpha: 1.0_f64,
                lda: p_i,
                incx: 1,
                beta: 0.0_f64,
                incy: 1,
            };
            // SAFETY: cuBLAS Dgemv; a_dev is p*p col-major f64, x_dev is
            // length-p f64, ax_dev is length-p output; all on the same stream.
            unsafe { blas.gemv(cfg, a_dev, x_dev, &mut ax_dev) }
                .map_err(|e| format!("cublasDgemv for residual: {e}"))?;
        }
        let ax_host = stream
            .clone_dtoh(&ax_dev)
            .map_err(|e| format!("download A·x: {e}"))?;
        // r = b − A·x  (host subtract; p is small)
        let r: Vec<f64> = b_host
            .iter()
            .zip(ax_host.iter())
            .map(|(bi, axi)| bi - axi)
            .collect();
        let norm_r = r.iter().map(|v| v * v).sum::<f64>().sqrt();
        Ok((r, norm_r))
    }

    // -----------------------------------------------------------------------
    // Iterative refinement: fp32 factor → fp32 solve → fp64 residual loop
    // -----------------------------------------------------------------------

    /// Solve `A x = b` using an fp32 Cholesky factorization with up to
    /// `max_steps` fp64-residual iterative refinement corrections.
    ///
    /// # Algorithm
    ///
    /// 1. Cast `A` (f64) to f32 on device. Factor in fp32 (POTRF).
    /// 2. Cast `b` (f64) to f32. Solve `A x = b` in fp32 (POTRS). Lift `x`
    ///    to f64.
    /// 3. `r = b − A·x` accumulated in fp64 (cuBLAS Dgemv), then
    ///    `refine_to_certificate`: up to `max_steps` corrections
    ///    (cast `r` to f32, solve `A e = r` in fp32, `x += e` in f64, recompute
    ///    the fp64 residual) until `‖r‖ ≤ γ_{p+1}·(‖A‖_F‖x‖ + ‖b‖)`, the
    ///    residual's own rounding band.
    /// 4. Return `x` only when that certificate holds.
    ///
    /// Returns `Err` when the fp32 POTRF fails (not SPD at f32), when the
    /// residual does not decrease monotonically (κ(A)·u_f32 ≥ 1 regime), or
    /// when `max_steps` corrections end with the residual still above its
    /// rounding band (the fp32 factor contracts too slowly to certify this
    /// system within the step budget). `Ok` therefore always carries a solution
    /// whose fp64 residual is inside its own rounding. Callers use fp64 POTRF
    /// on `Err`.
    pub(super) fn iterative_refinement_solve_impl(
        hessian: ArrayView2<'_, f64>,
        rhs: &[f64],
    ) -> Result<ndarray::Array1<f64>, String> {
        use crate::policy::GpuDispatchPolicy;
        let (p, p2) = hessian.dim();
        if p == 0 || p != p2 || rhs.len() != p {
            return Err("iterative_refinement_solve: dimension mismatch".to_string());
        }
        let max_steps = GpuDispatchPolicy::REFINEMENT_MAX_STEPS;

        let (_, stream) = context_and_stream()?;
        let solver = DnHandle::new(stream.clone()).map_err(|e| format!("cusolver init: {e}"))?;
        let blas = CudaBlas::new(stream.clone()).map_err(|e| format!("cublas init: {e}"))?;

        // Upload fp64 hessian for residual GEMV.
        let h_col_f64 = to_col_major(&hessian);
        let a_dev_f64 = pinned_htod(&stream, &h_col_f64)?;

        // Cast A to f32 and upload.
        let h_col_f32: Vec<f32> = h_col_f64.iter().map(|&v| v as f32).collect();
        let mut a_dev_f32 =
            pinned_htod(&stream, &h_col_f32).map_err(|e| format!("upload f32 A: {e}"))?;

        // fp32 POTRF — returns Err if A is not SPD at f32 precision.
        spotrf_in_place(&solver, &stream, p, &mut a_dev_f32)?;

        // Cast b to f32 and upload; solve in fp32.
        let b_f32: Vec<f32> = rhs.iter().map(|&v| v as f32).collect();
        let mut x_dev_f32 =
            pinned_htod(&stream, &b_f32).map_err(|e| format!("upload f32 rhs: {e}"))?;
        spotrs_in_place(&solver, &stream, p, 1, &a_dev_f32, &mut x_dev_f32)?;

        // Lift x to f64.
        let x_f32 = stream
            .clone_dtoh(&x_dev_f32)
            .map_err(|e| format!("download f32 x: {e}"))?;
        let mut x: Vec<f64> = x_f32.iter().map(|&v| v as f64).collect();

        let norm_b = rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
        // Refinement has converged once the fp64 residual `b − A·x` is inside its own
        // rounding: each component is a `p`-term inner product and a subtraction, so it
        // rounds by at most `γ_{p+1}·(|A||x| + |b|)`, whose 2-norm is at most
        // `γ_{p+1}·(‖A‖_F‖x‖₂ + ‖b‖₂)`. No correction resolves a smaller residual.
        let hessian_frobenius = hessian.iter().map(|v| v * v).sum::<f64>().sqrt();
        let growth = gam_linalg::roundoff::accumulation_growth(p + 1);
        let attainable = |x: &[f64]| {
            growth * (hessian_frobenius * x.iter().map(|v| v * v).sum::<f64>().sqrt() + norm_b)
        };

        let x_dev_f64 = pinned_htod(&stream, &x).map_err(|e| format!("upload f64 x: {e}"))?;
        let (r0, norm_r0) = residual_norm_and_vec(&blas, &stream, p, &a_dev_f64, &x_dev_f64, rhs)?;

        refine_to_certificate(
            &mut x,
            r0,
            norm_r0,
            max_steps,
            attainable,
            |r| {
                // Cast residual to f32, solve A e = r with the fp32 factor, lift e to f64.
                let r_f32: Vec<f32> = r.iter().map(|&v| v as f32).collect();
                let mut e_dev_f32 = pinned_htod(&stream, &r_f32)
                    .map_err(|e| format!("upload f32 residual: {e}"))?;
                spotrs_in_place(&solver, &stream, p, 1, &a_dev_f32, &mut e_dev_f32)?;
                let e_f32 = stream
                    .clone_dtoh(&e_dev_f32)
                    .map_err(|e| format!("download f32 e: {e}"))?;
                Ok(e_f32.iter().map(|&v| v as f64).collect())
            },
            |x| {
                let x_dev =
                    pinned_htod(&stream, x).map_err(|e| format!("upload refined x: {e}"))?;
                residual_norm_and_vec(&blas, &stream, p, &a_dev_f64, &x_dev, rhs)
            },
        )?;

        Ok(ndarray::Array1::from_vec(x))
    }

    /// Drive fp32-factor iterative refinement of `x` until its fp64 residual is
    /// certified, or report why it cannot be.
    ///
    /// `r`/`norm_r` are the fp64 residual `b − A·x` of the incoming `x` and its
    /// 2-norm. `attainable(x)` is the residual's own rounding band
    /// `γ_{p+1}·(‖A‖_F‖x‖ + ‖b‖)`; `correct(r)` solves `A e = r` with the fp32
    /// factor; `residual(x)` recomputes `b − A·x` in fp64.
    ///
    /// With a fixed fp32 factor the refinement contracts the error LINEARLY, by
    /// about `κ(A)·u_f32` per correction, so reaching the band may need more
    /// corrections than `max_steps` allows. Exhausting the budget is therefore
    /// an `Err`, never an uncertified `Ok`: `Ok(())` means the final residual is
    /// inside its rounding band. A correction that does not reduce the residual
    /// (the `κ(A)·u_f32 ≥ 1` regime, where the fp32 factor cannot contract) is
    /// also an `Err`.
    fn refine_to_certificate(
        x: &mut [f64],
        mut r: Vec<f64>,
        mut norm_r: f64,
        max_steps: usize,
        attainable: impl Fn(&[f64]) -> f64,
        mut correct: impl FnMut(&[f64]) -> Result<Vec<f64>, String>,
        mut residual: impl FnMut(&[f64]) -> Result<(Vec<f64>, f64), String>,
    ) -> Result<(), String> {
        let mut band = attainable(&*x);
        if norm_r <= band {
            return Ok(());
        }
        for _ in 0..max_steps {
            let e = correct(r.as_slice())?;
            for (xi, ei) in x.iter_mut().zip(e.iter()) {
                *xi += *ei;
            }
            let (r_new, norm_r_new) = residual(&*x)?;
            if norm_r_new >= norm_r {
                return Err(format!(
                    "iterative refinement: residual not decreasing ({norm_r_new:.3e} ≥ {norm_r:.3e}); \
                     κ(A)·u_f32 ≥ 1, cannot refine"
                ));
            }
            r = r_new;
            norm_r = norm_r_new;
            band = attainable(&*x);
            if norm_r <= band {
                return Ok(());
            }
        }
        Err(format!(
            "iterative refinement: residual {norm_r:.3e} still above its rounding band {band:.3e} \
             after {max_steps} corrections; the fp32 factor does not certify this solve"
        ))
    }

    #[cfg(test)]
    mod refine_to_certificate_tests {
        use super::refine_to_certificate;

        // A = diag(1, 2, 3, 4), b = 1. The residual is evaluated exactly as
        // b − A∘x, and the "fp32 factor" is modelled as a solve whose error
        // contracts by a fixed ratio ρ per correction: e = (1 − ρ)·r / a.
        const A: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
        const B: [f64; 4] = [1.0, 1.0, 1.0, 1.0];

        fn residual(x: &[f64]) -> Result<(Vec<f64>, f64), String> {
            let r: Vec<f64> = B
                .iter()
                .zip(A.iter())
                .zip(x)
                .map(|((b, a), x)| b - a * x)
                .collect();
            let n = r.iter().map(|v| v * v).sum::<f64>().sqrt();
            Ok((r, n))
        }

        fn attainable(x: &[f64]) -> f64 {
            let p = A.len();
            let a_frob = A.iter().map(|v| v * v).sum::<f64>().sqrt();
            let norm_x = x.iter().map(|v| v * v).sum::<f64>().sqrt();
            let norm_b = B.iter().map(|v| v * v).sum::<f64>().sqrt();
            gam_linalg::roundoff::accumulation_growth(p + 1) * (a_frob * norm_x + norm_b)
        }

        fn run(rho: f64, max_steps: usize) -> (Vec<f64>, Result<(), String>) {
            let mut x = vec![0.0; A.len()];
            let (r0, n0) = residual(&x).unwrap();
            let out = refine_to_certificate(
                &mut x,
                r0,
                n0,
                max_steps,
                attainable,
                |r| {
                    Ok(r.iter()
                        .zip(A.iter())
                        .map(|(ri, ai)| (1.0 - rho) * ri / ai)
                        .collect())
                },
                residual,
            );
            (x, out)
        }

        /// Regression for #3547: a slowly contracting factor (ρ = 0.5, i.e. the
        /// large-κ regime) leaves the residual far above its rounding band after
        /// the step budget. That must be an `Err`, not an uncertified `Ok`.
        #[test]
        fn refine_exhausted_budget_is_an_error_not_an_uncertified_solution() {
            let (x, out) = run(0.5, 3);
            let (_, norm_r) = residual(&x).unwrap();
            assert!(
                norm_r > attainable(&x),
                "precondition: band not reached ({norm_r:e})"
            );
            let err = out.expect_err("uncertified refinement must not return Ok");
            assert!(err.contains("rounding band"), "unexpected error: {err}");
        }

        /// A fast-contracting factor certifies within the budget and the
        /// returned `x` then solves the system to rounding.
        #[test]
        fn refine_certifies_when_contraction_is_fast() {
            let (x, out) = run(1e-8, 3);
            out.expect("fast contraction must certify");
            let (_, norm_r) = residual(&x).unwrap();
            assert!(norm_r <= attainable(&x));
            for (xi, (b, a)) in x.iter().zip(B.iter().zip(A.iter())) {
                assert!((xi - b / a).abs() <= 1e-14, "x = {xi}, want {}", b / a);
            }
        }

        /// A correction that over-shoots (ρ = −2, so ‖r‖ doubles) is the
        /// κ(A)·u_f32 ≥ 1 regime and must be reported, not iterated.
        #[test]
        fn refine_non_decreasing_residual_is_an_error() {
            let (_, out) = run(-2.0, 3);
            let err = out.expect_err("non-contracting refinement must not return Ok");
            assert!(err.contains("not decreasing"), "unexpected error: {err}");
        }
    }

    /// Bind a specific device ordinal's cached context on the calling thread and
    /// open a fresh stream on it. This is the per-ordinal entry point used by
    /// multi-GPU fan-out (`crate::pool::scatter_batched` workers) so a
    /// Cholesky / TRSM can target the device the worker thread owns. The
    /// primary-device convenience wrapper [`context_and_stream`] calls this with
    /// the probe-selected ordinal.
    pub(crate) fn context_and_stream_for(
        ordinal: usize,
    ) -> Result<
        (
            std::sync::Arc<CudaContext>,
            std::sync::Arc<cudarc::driver::CudaStream>,
        ),
        String,
    > {
        let ctx = super::super::device_runtime::cuda_context_for(ordinal)
            .ok_or_else(|| format!("cuda context for ordinal {ordinal} unavailable"))?;
        ctx.bind_to_thread()
            .map_err(|e| format!("cuda context bind_to_thread: {e}"))?;
        let stream = ctx.new_stream().map_err(|e| format!("cuda stream: {e}"))?;
        Ok((ctx, stream))
    }

    pub fn context_and_stream() -> Result<
        (
            std::sync::Arc<CudaContext>,
            std::sync::Arc<cudarc::driver::CudaStream>,
        ),
        String,
    > {
        // Route through the runtime's cached primary context for the selected
        // device so every CUDA client in the process (calibration, session,
        // cuSolver) shares one CUcontext per ordinal. Falling back to
        // `CudaContext::new(0)` here would fragment driver state across
        // distinct contexts, defeat memory-pool sharing, and pin work to
        // ordinal 0 even when the runtime probe chose a different device.
        let runtime = super::super::device_runtime::GpuRuntime::require()
            .map_err(|error| format!("cuda runtime unavailable: {error}"))?;
        context_and_stream_for(runtime.selected_device().ordinal)
    }

    pub fn pinned_htod<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Copy>(
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        src: &[T],
    ) -> Result<CudaSlice<T>, String> {
        // Originally this routine round-tripped the upload through a
        // `CU_MEMHOSTALLOC_WRITECOMBINED` pinned staging buffer
        // (`ctx.alloc_pinned`) to enable async DMA. In cudarc 0.19 the
        // `PinnedHostSlice` returned from `alloc_pinned` carries an event that
        // its `Drop` impl unconditionally `event.synchronize()`s before freeing
        // the host mapping — see cudarc-0.19.7 `core.rs::PinnedHostSlice::drop`.
        // Because the staging buffer goes out of scope at the end of this
        // function, the host thread blocks here until the H2D copy completes,
        // immediately defeating the "async" of pinned DMA. The net cost is two
        // extra driver calls per upload (`cuMemHostAlloc_WC` + `cuMemFreeHost`)
        // plus a forced stream synchronization, and the workspace ends up
        // strictly slower than a plain pageable H2D — the driver already
        // stages pageable copies internally via its own pinned pool, and that
        // path does not block the issuing host thread for unrelated stream
        // work. Issue a direct async H2D from the pageable buffer instead.
        stream.clone_htod(src).map_err(|e| format!("cuda H2D: {e}"))
    }

    pub fn potrf_in_place(
        solver: &DnHandle,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        p: usize,
        h: &mut CudaSlice<f64>,
    ) -> Result<(), String> {
        potrf_in_place_generic::<f64>(solver, stream, p, h)
    }

    pub fn potrs_in_place(
        solver: &DnHandle,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        p: usize,
        nrhs: usize,
        h: &CudaSlice<f64>,
        rhs: &mut CudaSlice<f64>,
    ) -> Result<(), String> {
        potrs_in_place_generic::<f64>(solver, stream, p, nrhs, h, rhs)
    }

    /// Query the cuSOLVER POTRF workspace size for a p×p matrix.
    ///
    /// Called once at workspace construction to size the persistent workspace
    /// buffer. Returns the number of f64 elements required.
    pub fn potrf_query_lwork(
        solver: &DnHandle,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        p: usize,
    ) -> Result<usize, String> {
        potrf_bufsize_generic::<f64>(solver, stream, p)
    }

    /// POTRF factorization using pre-allocated workspace and info buffers.
    ///
    /// Does not allocate, does not download `info`. The caller is responsible
    /// for calling [`check_deferred_potrf_info`] at end-of-fit to confirm no
    /// factorization failed.
    ///
    /// `workspace` must have been allocated with at least `lwork` elements
    /// (as reported by [`potrf_query_lwork`] at workspace construction).
    /// `info_dev` is a 1-element device i32 buffer; after a failed
    /// factorization it holds a positive integer but stays device-resident.
    pub fn potrf_in_place_reuse(
        solver: &DnHandle,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        p: usize,
        lwork: i32,
        h: &mut CudaSlice<f64>,
        workspace: &mut CudaSlice<f64>,
        info_dev: &mut CudaSlice<i32>,
    ) -> Result<(), String> {
        let p_i = to_i32(p)?;
        let uplo = cusolver_sys::cublasFillMode_t::CUBLAS_FILL_MODE_LOWER;
        {
            let (h_ptr, _h_record) = h.device_ptr_mut(stream);
            let (work_ptr, _work_record) = workspace.device_ptr_mut(stream);
            let (info_ptr, _info_record) = info_dev.device_ptr_mut(stream);
            // SAFETY: cuSOLVER potrf; h is p*p col-major, workspace was sized
            // by potrf_query_lwork, info_dev is a pre-allocated 1-element i32
            // device buffer. All buffers are live on the same stream.
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
        Ok(())
    }

    /// POTRS triangular solve using a pre-allocated info buffer.
    ///
    /// Does not allocate, does not download `info`. The caller is responsible
    /// for calling [`check_deferred_potrs_info`] at end-of-fit.
    pub fn potrs_in_place_reuse(
        solver: &DnHandle,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        p: usize,
        nrhs: usize,
        h: &CudaSlice<f64>,
        rhs: &mut CudaSlice<f64>,
        info_dev: &mut CudaSlice<i32>,
    ) -> Result<(), String> {
        let p_i = to_i32(p)?;
        let nrhs_i = to_i32(nrhs)?;
        let uplo = cusolver_sys::cublasFillMode_t::CUBLAS_FILL_MODE_LOWER;
        {
            let (h_ptr, _h_record) = h.device_ptr(stream);
            let (rhs_ptr, _rhs_record) = rhs.device_ptr_mut(stream);
            let (info_ptr, _info_record) = info_dev.device_ptr_mut(stream);
            // SAFETY: cuSOLVER potrs; h is a p*p Cholesky factor, rhs is p*nrhs,
            // info_dev is a pre-allocated 1-element i32 device buffer.
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
        Ok(())
    }

    /// Download the POTRF deferred info scalar and return an error if non-zero.
    ///
    /// Called once at end-of-fit (or whenever the convergence loop exits) to
    /// surface any factorization failure that was deferred device-side by
    /// [`potrf_in_place_reuse`].
    pub fn check_deferred_potrf_info(
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        info_dev: &CudaSlice<i32>,
    ) -> Result<(), String> {
        let info_host = stream
            .clone_dtoh(info_dev)
            .map_err(|e| format!("download deferred potrf info: {e}"))?;
        if info_host[0] == 0 {
            Ok(())
        } else {
            Err(format!(
                "cusolverDnDpotrf returned info={} (detected at end-of-fit)",
                info_host[0]
            ))
        }
    }

    /// Download the POTRS deferred info scalar and return an error if non-zero.
    ///
    /// Mirrors [`check_deferred_potrf_info`] for the triangular-solve step.
    pub fn check_deferred_potrs_info(
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        info_dev: &CudaSlice<i32>,
    ) -> Result<(), String> {
        let info_host = stream
            .clone_dtoh(info_dev)
            .map_err(|e| format!("download deferred potrs info: {e}"))?;
        if info_host[0] == 0 {
            Ok(())
        } else {
            Err(format!(
                "cusolverDnDpotrs returned info={} (detected at end-of-fit)",
                info_host[0]
            ))
        }
    }

    pub fn cholesky_logdet_from_col_major(factor: &[f64], p: usize) -> f64 {
        let factor = MatRef::from_column_major_slice(factor, p, p);
        cholesky_factor_logdet(factor)
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

// These solver entry points are consumed by sibling crates (`gam-solve`'s
// pirls/reml GPU paths, `gam-models`, ...) via `gam_gpu::solver::*`, so they
// are part of gam-gpu's public surface. `potrf_in_place_generic` is the
// only one with no cross-crate consumer; it stays crate-private and is
// reached internally through `crate::solver::potrf_in_place_generic`.
#[cfg(target_os = "linux")]
pub(crate) use cuda::potrf_in_place_generic;
#[cfg(target_os = "linux")]
pub use cuda::{
    check_deferred_potrf_info, check_deferred_potrs_info, cholesky_logdet_from_col_major,
    context_and_stream, pinned_htod, potrf_in_place, potrf_in_place_reuse, potrf_query_lwork,
    potrs_in_place, potrs_in_place_reuse,
};

/// Solve `A x = b` with fp32 Cholesky factorization + fp64-residual iterative
/// refinement, automatically falling back to fp64 when the policy rejects the
/// attempt or when the fp32 path fails / diverges.
///
/// The `p` threshold and maximum step count come from `GpuDispatchPolicy`
/// constants — there is no user-facing knob. The decision path is:
///
/// 1. `policy.iterative_refinement_should_attempt(p)` → `false` or
///    multi-column RHS: skip to the fp64 Cholesky path.
/// 2. Attempt fp32 POTRF + up to `REFINEMENT_MAX_STEPS` residual-correction
///    steps. Falls back to fp64 on:
///    - fp32 POTRF info ≠ 0 (A is not SPD at f32 precision),
///    - non-monotone residual (κ(A)·u_fp32 ≥ 1 regime),
///    - the step budget ending with the fp64 residual still above its rounding
///      band (refinement contracts linearly, by ≈ κ(A)·u_fp32 per step, so a
///      large-κ system can need more steps than the budget allows).
/// 3. On fp32 success the logdet is computed from the fp64 Cholesky factor —
///    BUT only when `need_logdet` is true. The fp64 POTRF is an O(p³)
///    factorization that fully negates the mixed-precision speedup (the whole
///    point is to do the expensive factor in fp32), so a caller that only needs
///    the *solution* (e.g. the PIRLS Newton direction solve, which discards the
///    logdet) passes `need_logdet = false` and the redundant fp64 POTRF is
///    skipped entirely — the returned logdet is `NaN` in that case. A solution
///    from the fp32 path is returned only when its fp64 residual is certified
///    inside its own rounding band; otherwise the fp64 path produces it.
///
/// Returns `(solution, logdet)`. When `need_logdet` is false and the fp32 path
/// succeeds, the logdet is `NaN`.
pub(crate) fn iterative_refinement_cholesky_solve(
    hessian: ArrayView2<'_, f64>,
    rhs: ArrayView2<'_, f64>,
    need_logdet: bool,
) -> Result<(Array2<f64>, f64), String> {
    #[cfg(not(target_os = "linux"))]
    {
        let (rows, cols) = hessian.dim();
        return Err(format!(
            "CUDA support not compiled; hessian={rows}x{cols}, rhs={}x{}, need_logdet={need_logdet}",
            rhs.nrows(),
            rhs.ncols()
        ));
    }

    #[cfg(target_os = "linux")]
    {
        let runtime = super::device_runtime::GpuRuntime::require().map_err(|error| {
            let (rows, cols) = hessian.dim();
            format!(
                "CUDA runtime unavailable; hessian={rows}x{cols}, rhs={}x{}: {error}",
                rhs.nrows(),
                rhs.ncols()
            )
        })?;
        let p = hessian.nrows();

        // Attempt fp32 + refinement only for single-column RHS with p large
        // enough that the fp64 GEMV residual cost is amortised.
        if rhs.ncols() == 1 && runtime.policy.iterative_refinement_should_attempt(p) {
            let rhs_col = rhs.column(0);
            let rhs_slice: Vec<f64> = rhs_col.iter().copied().collect();
            if let Ok(solution) = cuda::iterative_refinement_solve_impl(hessian, &rhs_slice) {
                // fp32 + refinement succeeded: the refined solution's fp64
                // residual is certified inside its rounding band. The logdet,
                // however, needs the fp64 Cholesky factor (the fp32 diagonal
                // is only fp32-accurate, and the logdet feeds the REML
                // criterion / EDF). Run the fp64 POTRF
                // ONLY when the caller actually consumes the logdet: otherwise
                // that O(p³) factorization is pure overhead that cancels the
                // mixed-precision win (the expensive factor would then run in
                // BOTH precisions). A solution-only caller (PIRLS Newton
                // direction, which discards the logdet) gets the genuine
                // fp32-factor speedup; logdet is reported as NaN.
                let mut sol = Array2::<f64>::zeros((p, 1));
                sol.column_mut(0).assign(&solution);
                if !need_logdet {
                    return Ok((sol, f64::NAN));
                }
                // A failed fp64 POTRF here is the same factorization the fp64
                // path below would repeat, so it is reported, not retried.
                return cuda::cholesky_logdet(hessian).map(|logdet| (sol, logdet));
            }
            // fp32 path did not certify (not SPD at f32, residual non-monotone,
            // or step budget exhausted above the rounding band) → fp64.
        }

        cuda::cholesky_solve(hessian, rhs)
    }
}

pub fn cholesky_solve_gpu(
    hessian: ArrayView2<'_, f64>,
    rhs: ArrayView2<'_, f64>,
) -> Result<(Array2<f64>, f64), String> {
    // Route through iterative refinement. The function falls back to fp64
    // internally, so callers always get a valid result. This wrapper returns
    // the logdet, so it must request it (`need_logdet`).
    iterative_refinement_cholesky_solve(hessian, rhs, /*need_logdet=*/ true)
}

/// Solution-only mixed-precision solve: like [`cholesky_solve_gpu`] but skips
/// the redundant fp64 POTRF when the fp32 + refinement path succeeds, since the
/// caller does not consume the log-determinant. This is the path that delivers
/// the full mixed-precision speedup (expensive O(p³) factor stays fp32) for the
/// PIRLS Newton direction solve, where the logdet is discarded. The solution's
/// fp64 residual is certified inside its rounding band (fp32 path) or it comes
/// from the fp64 factorization.
pub fn cholesky_solve_only_gpu(
    hessian: ArrayView2<'_, f64>,
    rhs: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, String> {
    let result = iterative_refinement_cholesky_solve(hessian, rhs, /*need_logdet=*/ false)?;
    Ok(result.0)
}

#[cfg(target_os = "linux")]
pub(crate) fn cholesky_lower_on_ordinal_gpu(
    ordinal: usize,
    hessian: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, String> {
    cuda::cholesky_lower_on_ordinal(ordinal, hessian)
}
