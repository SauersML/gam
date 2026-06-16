//! cuSOLVER-backed dense solver kernels for the GPU HAL.
//!
//! This module owns CUDA solver functionality that is shared by GPU linear
//! algebra dispatch and higher-level solver code. CPU solves do not live behind
//! these entry points: unavailable CUDA support is reported as an error.

use ndarray::{Array2, ArrayView2};

pub fn backend_status() -> super::BackendStatus {
    if super::runtime::GpuRuntime::global().is_some() {
        super::BackendStatus::CudaReady
    } else {
        super::BackendStatus::CudaUnavailable
    }
}

/// Outcome reported by [`iterative_refinement_cholesky_solve`].
#[derive(Clone, Debug)]
pub struct RefinementOutcome {
    /// Solution vector `x` satisfying `A x ≈ b`.
    pub solution: ndarray::Array1<f64>,
    /// `‖r‖ / ‖b‖` where `r = b − A x` after the last refinement step
    /// (or after the initial fp32 solve when no steps were taken).
    pub relative_residual: f64,
    /// Precision path used for the factorization.
    pub used_fp32_factor: bool,
    /// Number of refinement steps taken (0 means only the initial solve ran).
    pub refinement_steps: usize,
}

#[cfg(target_os = "linux")]
mod cuda {
    use crate::gpu::driver::{from_col_major, to_col_major};
    use crate::linalg::faer_ndarray::cholesky_factor_logdet;
    use cudarc::cublas::sys as cublas_sys;
    use cudarc::cublas::{CudaBlas, Gemv, GemvConfig};
    use cudarc::cusolver::{DnHandle, sys as cusolver_sys};
    use cudarc::driver::{CudaContext, CudaSlice, DevicePtr, DevicePtrMut};
    use faer::MatRef;
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

    pub(super) fn cholesky_lower(hessian: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
        let (_, stream) = context_and_stream()?;
        cholesky_lower_on_stream(hessian, &stream)
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
    /// 3. Loop up to `max_steps`:
    ///    a. `r = b − A·x` accumulated in fp64 (cuBLAS Dgemv).
    ///    b. `‖r‖ / ‖b‖ ≤ tol` → converged, break.
    ///    c. Residual did not drop below previous step → bail, return `Err`.
    ///    d. Cast `r` to f32. Solve `A e = r` in fp32. `x += e` (f64).
    /// 4. Return `(x, ‖r‖/‖b‖, refinement_steps)`.
    ///
    /// Returns `Err` when the fp32 POTRF fails (not SPD at f32) or when the
    /// residual does not decrease monotonically (κ(A)·u_f32 ≥ 1 regime).
    /// Callers should fall back to fp64 POTRF on `Err`.
    pub(super) fn iterative_refinement_solve_impl(
        hessian: ArrayView2<'_, f64>,
        rhs: &[f64],
    ) -> Result<super::RefinementOutcome, String> {
        use crate::gpu::policy::GpuDispatchPolicy;
        let (p, p2) = hessian.dim();
        if p == 0 || p != p2 || rhs.len() != p {
            return Err("iterative_refinement_solve: dimension mismatch".to_string());
        }
        let max_steps = GpuDispatchPolicy::REFINEMENT_MAX_STEPS;
        let tol = GpuDispatchPolicy::REFINEMENT_TOL;

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

        // Compute ‖b‖ for relative residual.
        let norm_b = rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
        let norm_b_safe = if norm_b > 0.0 { norm_b } else { 1.0 };

        let mut x_dev_f64 = pinned_htod(&stream, &x).map_err(|e| format!("upload f64 x: {e}"))?;
        let (r0, norm_r0) = residual_norm_and_vec(&blas, &stream, p, &a_dev_f64, &x_dev_f64, rhs)?;
        let mut rel_residual = norm_r0 / norm_b_safe;

        // Early exit: already converged after initial solve.
        if rel_residual <= tol {
            return Ok(super::RefinementOutcome {
                solution: ndarray::Array1::from_vec(x),
                relative_residual: rel_residual,
                used_fp32_factor: true,
                refinement_steps: 0,
            });
        }

        let mut r = r0;
        let mut prev_norm_r = norm_r0;
        let mut steps_taken = 0_usize;

        for _ in 0..max_steps {
            // Cast residual to f32, solve A e = r in fp32.
            let r_f32: Vec<f32> = r.iter().map(|&v| v as f32).collect();
            let mut e_dev_f32 =
                pinned_htod(&stream, &r_f32).map_err(|e| format!("upload f32 residual: {e}"))?;
            spotrs_in_place(&solver, &stream, p, 1, &a_dev_f32, &mut e_dev_f32)?;

            // x += e in f64.
            let e_f32 = stream
                .clone_dtoh(&e_dev_f32)
                .map_err(|e| format!("download f32 e: {e}"))?;
            for (xi, ei) in x.iter_mut().zip(e_f32.iter()) {
                *xi += *ei as f64;
            }
            steps_taken += 1;

            // Reupload x_dev_f64 and compute new residual.
            x_dev_f64 = pinned_htod(&stream, &x).map_err(|e| format!("upload refined x: {e}"))?;
            let (r_new, norm_r_new) =
                residual_norm_and_vec(&blas, &stream, p, &a_dev_f64, &x_dev_f64, rhs)?;
            rel_residual = norm_r_new / norm_b_safe;

            // Check monotone decrease. Non-monotone → κ(A)·u ≥ 1.
            if norm_r_new >= prev_norm_r {
                return Err(format!(
                    "iterative refinement: residual not decreasing ({norm_r_new:.3e} ≥ {prev_norm_r:.3e}); \
                     κ(A)·u_f32 ≥ 1, cannot refine"
                ));
            }
            prev_norm_r = norm_r_new;
            r = r_new;

            if rel_residual <= tol {
                break;
            }
        }

        Ok(super::RefinementOutcome {
            solution: ndarray::Array1::from_vec(x),
            relative_residual: rel_residual,
            used_fp32_factor: true,
            refinement_steps: steps_taken,
        })
    }

    /// Bind a specific device ordinal's cached context on the calling thread and
    /// open a fresh stream on it. This is the per-ordinal entry point used by
    /// multi-GPU fan-out (`crate::gpu::pool::scatter_batched` workers) so a
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
        let ctx = super::super::runtime::cuda_context_for(ordinal)
            .ok_or_else(|| format!("cuda context for ordinal {ordinal} unavailable"))?;
        ctx.bind_to_thread()
            .map_err(|e| format!("cuda context bind_to_thread: {e}"))?;
        let stream = ctx.new_stream().map_err(|e| format!("cuda stream: {e}"))?;
        Ok((ctx, stream))
    }

    pub(crate) fn context_and_stream() -> Result<
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
        let runtime = super::super::runtime::GpuRuntime::global()
            .ok_or_else(|| "cuda runtime unavailable".to_string())?;
        context_and_stream_for(runtime.selected_device().ordinal)
    }

    pub(crate) fn pinned_htod<
        T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Copy,
    >(
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

    pub(crate) fn potrf_in_place(
        solver: &DnHandle,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        p: usize,
        h: &mut CudaSlice<f64>,
    ) -> Result<(), String> {
        potrf_in_place_generic::<f64>(solver, stream, p, h)
    }

    pub(crate) fn potrs_in_place(
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
    pub(crate) fn potrf_query_lwork(
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
    pub(crate) fn potrf_in_place_reuse(
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
    pub(crate) fn potrs_in_place_reuse(
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
    pub(crate) fn check_deferred_potrf_info(
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
    pub(crate) fn check_deferred_potrs_info(
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

    pub(crate) fn cholesky_logdet_from_col_major(factor: &[f64], p: usize) -> f64 {
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

#[cfg(target_os = "linux")]
pub(crate) use cuda::{
    check_deferred_potrf_info, check_deferred_potrs_info, cholesky_logdet_from_col_major,
    context_and_stream, pinned_htod, potrf_in_place, potrf_in_place_generic, potrf_in_place_reuse,
    potrf_query_lwork, potrs_in_place, potrs_in_place_reuse,
};

/// Solve `A x = b` with fp32 Cholesky factorization + fp64-residual iterative
/// refinement, automatically falling back to fp64 when the policy rejects the
/// attempt or when the fp32 path fails / diverges.
///
/// The `p` threshold and maximum step count come from [`GpuDispatchPolicy`]
/// constants — there is no user-facing knob. The decision path is:
///
/// 1. `policy.iterative_refinement_should_attempt(p)` → `false` or
///    multi-column RHS: skip to the fp64 Cholesky path.
/// 2. Attempt fp32 POTRF + up to `REFINEMENT_MAX_STEPS` residual-correction
///    steps. Falls back to fp64 on:
///    - fp32 POTRF info ≠ 0 (A is not SPD at f32 precision),
///    - non-monotone residual (κ(A)·u_fp32 ≥ 1 regime).
/// 3. On fp32 success the logdet is computed from the fp64 Cholesky factor
///    (fp64 POTRF is always run for logdet accuracy). The solution comes from
///    the refined fp32 path.
///
/// Returns `(solution, logdet, Some(RefinementOutcome))` when the fp32 path
/// succeeded, or `(solution, logdet, None)` on the fp64 fallback.
pub fn iterative_refinement_cholesky_solve(
    hessian: ArrayView2<'_, f64>,
    rhs: ArrayView2<'_, f64>,
) -> Result<(Array2<f64>, f64, Option<RefinementOutcome>), String> {
    #[cfg(not(target_os = "linux"))]
    {
        let (rows, cols) = hessian.dim();
        return Err(format!(
            "CUDA support not compiled; hessian={rows}x{cols}, rhs={}x{}",
            rhs.nrows(),
            rhs.ncols()
        ));
    }

    #[cfg(target_os = "linux")]
    {
        let runtime = super::runtime::GpuRuntime::global().ok_or_else(|| {
            let (rows, cols) = hessian.dim();
            format!(
                "CUDA runtime unavailable; hessian={rows}x{cols}, rhs={}x{}",
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
            if let Ok(outcome) = cuda::iterative_refinement_solve_impl(hessian, &rhs_slice) {
                // fp32 + refinement succeeded. Still run fp64 POTRF for logdet
                // accuracy (the fp64 factor is always authoritative for EDF /
                // REML curvature). Solution comes from the refined fp32 path.
                if let Ok(fp64_result) = cuda::cholesky_solve(hessian, rhs) {
                    let logdet = fp64_result.1;
                    let mut sol = Array2::<f64>::zeros((p, 1));
                    sol.column_mut(0).assign(&outcome.solution);
                    return Ok((sol, logdet, Some(outcome)));
                }
                // fp64 logdet failed (theoretically impossible for SPD A);
                // fall through to plain fp64 path.
            }
            // fp32 path failed (not SPD at f32, or residual non-monotone) →
            // fall through to fp64.
        }

        let (sol, logdet) = cuda::cholesky_solve(hessian, rhs)?;
        Ok((sol, logdet, None))
    }
}

pub fn cholesky_solve_gpu(
    hessian: ArrayView2<'_, f64>,
    rhs: ArrayView2<'_, f64>,
) -> Result<(Array2<f64>, f64), String> {
    // Route through iterative refinement. The function falls back to fp64
    // internally, so callers always get a valid result; the refinement
    // outcome metadata is intentionally not surfaced by this thin wrapper.
    let result = iterative_refinement_cholesky_solve(hessian, rhs)?;
    Ok((result.0, result.1))
}

pub fn cholesky_lower_gpu(hessian: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
    #[cfg(not(target_os = "linux"))]
    {
        let (rows, cols) = hessian.dim();
        return Err(format!(
            "CUDA support not compiled for Cholesky factorization; hessian={rows}x{cols}"
        ));
    }

    #[cfg(target_os = "linux")]
    {
        if super::runtime::GpuRuntime::global().is_none() {
            let (rows, cols) = hessian.dim();
            return Err(format!(
                "CUDA runtime unavailable for Cholesky factorization; hessian={rows}x{cols}"
            ));
        }
        cuda::cholesky_lower(hessian)
    }
}

#[cfg(target_os = "linux")]
pub(crate) fn cholesky_lower_on_ordinal_gpu(
    ordinal: usize,
    hessian: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, String> {
    cuda::cholesky_lower_on_ordinal(ordinal, hessian)
}
