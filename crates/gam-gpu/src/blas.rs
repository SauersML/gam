//! Device BLAS surface for the cudarc-backed dense kernels.
//!
//! The public surface here is the lowest level of the GPU dispatch stack: it
//! takes ndarray views, copies them to a device buffer, calls a cuBLAS / kernel
//! routine, and returns the host result. The cudarc-backed implementations
//! always compile (cudarc dynamically loads `libcuda` at runtime via the
//! `fallback-dynamic-loading` feature), and dispatch is gated at runtime on
//! `super::device_runtime::GpuRuntime::global()` — when no device is probed the
//! status enum advertises `CudaUnavailable` and callers fall back to CPU.
//!
//! The implementations route through `super::device_runtime::cuda_context_for` and
//! the cudarc 0.19 cuBLAS API. Any transient backend failure (OOM, launch
//! error, …) is converted to `None` so the auto-dispatch shim in
//! `super::linalg` falls back to the CPU fast path without disturbing
//! numerics.

pub fn blas_backend_status() -> super::CudaBackendStatus {
    super::cuda_backend_status()
}

#[cfg(target_os = "linux")]
mod cuda_impl {
    use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis};

    use crate::driver::{from_col_major, to_col_major, to_i32};

    use super::super::device_runtime::GpuRuntime;
    use cudarc::cublas::sys::{
        cublasDiagType_t, cublasFillMode_t, cublasOperation_t, cublasSideMode_t, cublasStatus_t,
    };
    use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, Gemv, GemvConfig, StridedBatchedConfig};
    use cudarc::cusolver::{DnHandle, sys as cusolver_sys};
    use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
    use std::sync::Arc;

    /// Create a fresh stream + cuBLAS handle bound to a specific device
    /// ordinal. This is the per-ordinal entry point used by multi-GPU fan-out
    /// (`super::super::pool::scatter_batched` workers): the worker thread has
    /// already bound that ordinal's context, and the stream/handle created here
    /// target the same device. The single-device helper below is the
    /// primary-ordinal specialization.
    #[inline]
    pub(crate) fn stream_and_blas_for(ordinal: usize) -> Option<(Arc<CudaStream>, CudaBlas)> {
        let stream = super::super::device_runtime::cuda_context_for(ordinal)?
            .new_stream()
            .ok()?;
        let blas = CudaBlas::new(stream.clone()).ok()?;
        Some((stream, blas))
    }

    #[inline]
    fn stream_and_blas(runtime: &GpuRuntime) -> Option<(Arc<CudaStream>, CudaBlas)> {
        stream_and_blas_for(runtime.device.ordinal)
    }

    #[inline]
    fn vector_values(v: ArrayView1<'_, f64>) -> Vec<f64> {
        v.iter().copied().collect()
    }

    #[inline]
    fn to_col_major_batch(batch: ArrayView3<'_, f64>) -> Vec<f64> {
        let (batch_len, rows, cols) = batch.dim();
        let mut out = Vec::with_capacity(batch_len.saturating_mul(rows).saturating_mul(cols));
        for matrix in batch.axis_iter(Axis(0)) {
            out.extend(to_col_major(&matrix).iter().copied());
        }
        out
    }

    #[inline]
    fn from_col_major_batch(
        data: &[f64],
        batch: usize,
        rows: usize,
        cols: usize,
    ) -> Option<Array3<f64>> {
        if data.len() != batch.checked_mul(rows)?.checked_mul(cols)? {
            return None;
        }
        let mut out = Array3::<f64>::zeros((batch, rows, cols));
        let matrix_len = rows.checked_mul(cols)?;
        for batch_idx in 0..batch {
            let base = batch_idx.checked_mul(matrix_len)?;
            for col in 0..cols {
                for row in 0..rows {
                    out[[batch_idx, row, col]] = data[base + col * rows + row];
                }
            }
        }
        Some(out)
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
        weighted_crossprod_for(runtime.device.ordinal, left, weights, right)
    }

    #[inline]
    fn weighted_crossprod_for(
        ordinal: usize,
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

        let (stream, blas) = stream_and_blas_for(ordinal)?;
        // #1412: the symmetric Gram `Xᵀ·diag(w)·X` (xt_diag_x) passes the SAME
        // array as `left` and `right`. Detect that (identical data pointer +
        // shape) and stage `X` ONCE instead of column-majoring and H2D-uploading
        // two byte-identical n×p copies — halving the dominant H2D for the Gram.
        // The GEMM operands are unchanged (`left_dev` doubles as the row-scale
        // source), so the result is bit-identical to the two-upload path.
        let same_operand = std::ptr::eq(left.as_ptr(), right.as_ptr())
            && left.dim() == right.dim()
            && left.strides() == right.strides();
        let left_col = to_col_major(&left);
        let weights_host = vector_values(weights);
        let left_dev = stream.clone_htod(&*left_col).ok()?;
        // Symmetric Gram: `right` IS `left`, so row-scale directly from the
        // single resident `left_dev` and never upload a second n×p copy. The
        // asymmetric path uploads `right` as before.
        let right_dev = if same_operand {
            None
        } else {
            let right_col = to_col_major(&right);
            Some(stream.clone_htod(&*right_col).ok()?)
        };
        let weights_dev = stream.clone_htod(&weights_host).ok()?;
        let mut weighted_right_dev = stream
            .alloc_zeros::<f64>(rows.checked_mul(right_cols)?)
            .ok()?;
        row_scale_device(
            &blas,
            &stream,
            right_dev.as_ref().unwrap_or(&left_dev),
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

    /// #1017 Phase 3: a device-resident design matrix `X` whose `n×p` values are
    /// uploaded to the device ONCE and reused across many `Xᵀ·diag(w)·X` Gram
    /// evaluations.
    ///
    /// The per-call [`xt_diag_x_cuda`] path re-uploads the full `n×p` `X` (and a
    /// second copy as the `right` operand) on EVERY call. For the SAE / IRLS
    /// inner loop — where `X` is frozen across weight updates and the Gram is
    /// rebuilt once per Newton/PIRLS step — that H2D staging dominates the wall
    /// clock (measured #1412: the `XtWX` GEMM is ~98% of the pipeline at <20% GPU
    /// utilisation, i.e. the device is starved by the per-call upload, not the
    /// arithmetic). Uploading `X` once and crossing only the `n`-vector `w` (and
    /// the `p×p` result) per call removes that ping-pong: the resident `X` is
    /// `n·p` doubles vs the per-call `w` of `n` doubles, so the amortised
    /// transfer per Gram drops by a factor of `p`.
    pub(crate) struct ResidentWeightedGram {
        stream: Arc<CudaStream>,
        blas: CudaBlas,
        x_dev: CudaSlice<f64>,
        rows: usize,
        cols: usize,
    }

    impl ResidentWeightedGram {
        /// Upload `x` (`n×p`) to `ordinal` once, column-major, and keep it
        /// resident. Returns `None` on a degenerate shape or any device failure
        /// (the caller falls back to the per-call CPU/GPU path).
        pub(crate) fn new(ordinal: usize, x: ArrayView2<'_, f64>) -> Option<Self> {
            let (rows, cols) = x.dim();
            if rows == 0 || cols == 0 {
                return None;
            }
            let (stream, blas) = stream_and_blas_for(ordinal)?;
            let x_col = to_col_major(&x);
            let x_dev = stream.clone_htod(&*x_col).ok()?;
            Some(Self {
                stream,
                blas,
                x_dev,
                rows,
                cols,
            })
        }

        #[inline]
        pub(crate) fn dims(&self) -> (usize, usize) {
            (self.rows, self.cols)
        }

        /// Compute `Xᵀ·diag(w)·X` reusing the resident `X`. Only `w` (`n`
        /// doubles) crosses H2D and only the `p×p` Gram crosses D2H. The
        /// arithmetic is bit-identical to [`xt_diag_x_cuda`] on the same device
        /// (same `cublasDdgmm` row-scale + same `gemm` reduction order).
        pub(crate) fn gram(&self, w: ArrayView1<'_, f64>) -> Option<Array2<f64>> {
            if w.len() != self.rows {
                return None;
            }
            let weights_host = vector_values(w);
            let weights_dev = self.stream.clone_htod(&weights_host).ok()?;
            let mut weighted_dev = self
                .stream
                .alloc_zeros::<f64>(self.rows.checked_mul(self.cols)?)
                .ok()?;
            row_scale_device(
                &self.blas,
                &self.stream,
                &self.x_dev,
                &weights_dev,
                &mut weighted_dev,
                self.rows,
                self.cols,
            )?;
            let mut out_dev = self
                .stream
                .alloc_zeros::<f64>(self.cols.checked_mul(self.cols)?)
                .ok()?;
            let cfg = GemmConfig::<f64> {
                transa: cublasOperation_t::CUBLAS_OP_T,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: to_i32(self.cols)?,
                n: to_i32(self.cols)?,
                k: to_i32(self.rows)?,
                alpha: 1.0,
                lda: to_i32(self.rows)?,
                ldb: to_i32(self.rows)?,
                beta: 0.0,
                ldc: to_i32(self.cols)?,
            };
            // SAFETY: `x_dev` is the resident n×p column-major design; cfg forms
            // Xᵀ (p×n) · weighted (n×p) → a p×p column-major Gram.
            unsafe {
                self.blas
                    .gemm(cfg, &self.x_dev, &weighted_dev, &mut out_dev)
            }
            .ok()?;
            let out_col = self.stream.clone_dtoh(&out_dev).ok()?;
            from_col_major(&out_col, self.cols, self.cols)
        }

        /// Compute the resident weighted Gram `G = Xᵀ·diag(w)·X + ridge·I`,
        /// factor it (cuSOLVER POTRF), and solve `G·β = rhs` — keeping `G`, its
        /// Cholesky factor, and the RHS all DEVICE-RESIDENT. Only `w` (`n`),
        /// `rhs` (`p`), and the result `β` (`p`) cross the PCIe boundary; the
        /// `p×p` Gram is NEVER downloaded.
        ///
        /// This is the #1017 Phase-3 ceiling fix for the normal-equations solve:
        /// the per-call [`gram`] still pays a `p×p` D2H (134 MB at p=4096 — the
        /// next bottleneck once `X` is resident), whereas the SAE/IRLS inner step
        /// only needs the `p`-vector `β = (XᵀWX+λ)⁻¹ XᵀWz`. Chaining
        /// row-scale→GEMM→POTRF→TRSM on-device and returning only `β` removes the
        /// Gram transfer entirely.
        ///
        /// `ridge` (e.g. the penalty diagonal `λ` or a Tikhonov floor) is seeded
        /// as `ridge·I` on the device and the Gram is GEMM-accumulated onto it
        /// (`beta = 1`), so the diagonal bump never costs a Gram round-trip.
        /// Returns `None` on shape mismatch, a non-PD factorisation, or any
        /// device failure (the caller falls back to the CPU solve).
        pub(crate) fn solve_psd_normal_equations(
            &self,
            w: ArrayView1<'_, f64>,
            rhs: ArrayView1<'_, f64>,
            ridge: f64,
        ) -> Option<Array1<f64>> {
            if w.len() != self.rows || rhs.len() != self.cols {
                return None;
            }
            let p = self.cols;

            // weighted = diag(w) · X  (resident X row-scaled).
            let weights_dev = self.stream.clone_htod(&vector_values(w)).ok()?;
            let mut weighted_dev = self
                .stream
                .alloc_zeros::<f64>(self.rows.checked_mul(p)?)
                .ok()?;
            row_scale_device(
                &self.blas,
                &self.stream,
                &self.x_dev,
                &weights_dev,
                &mut weighted_dev,
                self.rows,
                p,
            )?;

            // Pre-seed G with `ridge·I` on the device, then GEMM-accumulate
            // `XᵀW X` onto it with `beta = 1.0`. The Gram is formed and stays
            // device-resident: `ridge·I` is a one-time H2D upload (the only way
            // to set a diagonal without an NVRTC kernel), and crucially the p×p
            // Gram is NEVER read back — only `β` returns. `ridge·I` upload is
            // bandwidth-trivial vs the avoided per-solve Gram download.
            let mut ridge_init = vec![0.0_f64; p.checked_mul(p)?];
            for i in 0..p {
                ridge_init[i * p + i] = ridge;
            }
            let mut g_dev = self.stream.clone_htod(&ridge_init).ok()?;
            let cfg = GemmConfig::<f64> {
                transa: cublasOperation_t::CUBLAS_OP_T,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: to_i32(p)?,
                n: to_i32(p)?,
                k: to_i32(self.rows)?,
                alpha: 1.0,
                lda: to_i32(self.rows)?,
                ldb: to_i32(self.rows)?,
                // Accumulate onto the resident ridge·I seed.
                beta: 1.0,
                ldc: to_i32(p)?,
            };
            // SAFETY: resident n×p X and the n×p weighted buffer form a p×p Gram;
            // beta=1 accumulates Xᵀ(WX) onto the resident ridge·I in g_dev.
            unsafe { self.blas.gemm(cfg, &self.x_dev, &weighted_dev, &mut g_dev) }.ok()?;

            // POTRF(G) → lower factor L, resident in g_dev.
            let solver = DnHandle::new(self.stream.clone()).ok()?;
            let info = potrf_single_dev(&solver, &self.stream, p, &mut g_dev)?;
            if info != 0 {
                // Not positive-definite at pivot `info`; caller falls back.
                return None;
            }

            // Solve L Lᵀ β = rhs via two triangular solves, β resident in rhs_dev.
            let mut rhs_dev = self.stream.clone_htod(&vector_values(rhs)).ok()?;
            trsm_single_vec(&self.blas, &self.stream, p, &g_dev, &mut rhs_dev, false)?; // L y = rhs
            trsm_single_vec(&self.blas, &self.stream, p, &g_dev, &mut rhs_dev, true)?; // Lᵀ β = y

            // Download ONLY the p-vector solution.
            let beta_host = self.stream.clone_dtoh(&rhs_dev).ok()?;
            Some(Array1::from_vec(beta_host))
        }
    }

    /// Single cuSOLVER `DPOTRF` (lower) of a resident `p×p` column-major matrix,
    /// factored in place. Returns the cuSOLVER `info` (0 = success, k>0 = the
    /// leading minor of order k is not PD). Mirrors the arrow-Schur frame POTRF.
    fn potrf_single_dev(
        solver: &DnHandle,
        stream: &Arc<CudaStream>,
        p: usize,
        matrix: &mut CudaSlice<f64>,
    ) -> Option<i32> {
        let p_i = to_i32(p)?;
        let uplo = cusolver_sys::cublasFillMode_t::CUBLAS_FILL_MODE_LOWER;
        let mut lwork = 0_i32;
        {
            let (mat_ptr, _rec) = matrix.device_ptr_mut(stream);
            // SAFETY: buffer-size query against a live p×p column-major matrix.
            let status = unsafe {
                cusolver_sys::cusolverDnDpotrf_bufferSize(
                    solver.cu(),
                    uplo,
                    p_i,
                    mat_ptr as *mut f64,
                    p_i,
                    &mut lwork,
                )
            };
            if status != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
                return None;
            }
        }
        let mut workspace = stream.alloc_zeros::<f64>(lwork.max(1) as usize).ok()?;
        let mut info_dev = stream.alloc_zeros::<i32>(1).ok()?;
        {
            let (mat_ptr, _rec) = matrix.device_ptr_mut(stream);
            let (work_ptr, _wrec) = workspace.device_ptr_mut(stream);
            let (info_ptr, _irec) = info_dev.device_ptr_mut(stream);
            // SAFETY: all buffers live on this stream; matrix is p×p column-major.
            let status = unsafe {
                cusolver_sys::cusolverDnDpotrf(
                    solver.cu(),
                    uplo,
                    p_i,
                    mat_ptr as *mut f64,
                    p_i,
                    work_ptr as *mut f64,
                    lwork,
                    info_ptr as *mut i32,
                )
            };
            if status != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
                return None;
            }
        }
        let info_host = stream.clone_dtoh(&info_dev).ok()?;
        info_host.first().copied()
    }

    /// Triangular solve `op(L)·x = b` for a single `p`-vector RHS against a
    /// resident lower Cholesky factor `L` (`p×p` column-major), in place over
    /// `rhs`. `transposed` selects `Lᵀ` (the second back-substitution).
    fn trsm_single_vec(
        blas: &CudaBlas,
        stream: &Arc<CudaStream>,
        p: usize,
        l: &CudaSlice<f64>,
        rhs: &mut CudaSlice<f64>,
        transposed: bool,
    ) -> Option<()> {
        let alpha = 1.0_f64;
        let p_i = to_i32(p)?;
        let handle = *blas.handle();
        let (l_ptr, _l_rec) = l.device_ptr(stream);
        let (rhs_ptr, _rhs_rec) = rhs.device_ptr_mut(stream);
        // SAFETY: p×p lower factor and a single p-vector RHS, both resident.
        let status = unsafe {
            cudarc::cublas::sys::cublasDtrsm_v2(
                handle,
                cublasSideMode_t::CUBLAS_SIDE_LEFT,
                cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
                if transposed {
                    cublasOperation_t::CUBLAS_OP_T
                } else {
                    cublasOperation_t::CUBLAS_OP_N
                },
                cublasDiagType_t::CUBLAS_DIAG_NON_UNIT,
                p_i,
                1,
                &alpha,
                l_ptr as *const f64,
                p_i,
                rhs_ptr as *mut f64,
                p_i,
            )
        };
        if status == cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            Some(())
        } else {
            None
        }
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
        gemm_on_ordinal_cuda(runtime.device.ordinal, a, b, trans_a, trans_b)
    }

    /// Dense GEMM (optionally transposing either operand) on a specific device
    /// ordinal. The ordinal's context is expected to be bound on the calling
    /// thread (pool-tiled callers via `super::super::pool::scatter_batched`, or
    /// the single-device dispatcher through [`gemm_cuda`]). Semantics are
    /// identical to [`gemm_cuda`]; only the target device differs.
    #[inline]
    pub(crate) fn gemm_on_ordinal_cuda(
        ordinal: usize,
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
        let (stream, blas) = stream_and_blas_for(ordinal)?;
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

    /// Broadcast-B batched GEMM on a specific device ordinal. The caller
    /// (`super::super::pool::scatter_batched` worker, or the single-device
    /// dispatcher) supplies the ordinal whose context is already bound on this
    /// thread; the stream/handle are created on that same device.
    #[inline]
    pub(crate) fn gemm_broadcast_b_batched_cuda(
        ordinal: usize,
        a: ArrayView3<'_, f64>,
        b: ArrayView2<'_, f64>,
    ) -> Option<Array3<f64>> {
        let (batch, m, k) = a.dim();
        let (b_rows, n) = b.dim();
        if batch == 0 || m == 0 || n == 0 || k == 0 || b_rows != k {
            return None;
        }
        let (stream, blas) = stream_and_blas_for(ordinal)?;
        let a_col = to_col_major_batch(a);
        let b_col = to_col_major(&b);
        let a_dev = stream.clone_htod(&a_col).ok()?;
        let b_dev = stream.clone_htod(&*b_col).ok()?;
        let mut out_dev = stream
            .alloc_zeros::<f64>(batch.checked_mul(m)?.checked_mul(n)?)
            .ok()?;
        let cfg = StridedBatchedConfig::<f64> {
            gemm: GemmConfig::<f64> {
                transa: cublasOperation_t::CUBLAS_OP_N,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: to_i32(m)?,
                n: to_i32(n)?,
                k: to_i32(k)?,
                alpha: 1.0,
                lda: to_i32(m)?,
                ldb: to_i32(k)?,
                beta: 0.0,
                ldc: to_i32(m)?,
            },
            batch_size: to_i32(batch)?,
            stride_a: i64::try_from(m.checked_mul(k)?).ok()?,
            stride_b: 0,
            stride_c: i64::try_from(m.checked_mul(n)?).ok()?,
        };
        // SAFETY: `a_dev` is a stack of batch column-major m×k matrices,
        // `b_dev` is one shared column-major k×n matrix with zero batch stride,
        // and `out_dev` is a stack of batch column-major m×n outputs.
        unsafe { blas.gemm_strided_batched(cfg, &a_dev, &b_dev, &mut out_dev) }.ok()?;
        let out_col = stream.clone_dtoh(&out_dev).ok()?;
        from_col_major_batch(&out_col, batch, m, n)
    }

    /// A·Bᵀ strided-batched GEMM on a specific device ordinal. As with the
    /// broadcast variant, the ordinal's context is expected to be bound on the
    /// calling thread (multi-GPU worker or single-device dispatcher).
    #[inline]
    pub(crate) fn gemm_abt_strided_batched_cuda(
        ordinal: usize,
        a: ArrayView3<'_, f64>,
        b: ArrayView3<'_, f64>,
    ) -> Option<Array3<f64>> {
        let (batch, m, k) = a.dim();
        let (batch_b, n, k_b) = b.dim();
        if batch == 0 || m == 0 || n == 0 || k == 0 || batch != batch_b || k != k_b {
            return None;
        }
        let (stream, blas) = stream_and_blas_for(ordinal)?;
        let a_col = to_col_major_batch(a);
        let b_col = to_col_major_batch(b);
        let a_dev = stream.clone_htod(&a_col).ok()?;
        let b_dev = stream.clone_htod(&b_col).ok()?;
        let mut out_dev = stream
            .alloc_zeros::<f64>(batch.checked_mul(m)?.checked_mul(n)?)
            .ok()?;
        let cfg = StridedBatchedConfig::<f64> {
            gemm: GemmConfig::<f64> {
                transa: cublasOperation_t::CUBLAS_OP_N,
                transb: cublasOperation_t::CUBLAS_OP_T,
                m: to_i32(m)?,
                n: to_i32(n)?,
                k: to_i32(k)?,
                alpha: 1.0,
                lda: to_i32(m)?,
                ldb: to_i32(n)?,
                beta: 0.0,
                ldc: to_i32(m)?,
            },
            batch_size: to_i32(batch)?,
            stride_a: i64::try_from(m.checked_mul(k)?).ok()?,
            stride_b: i64::try_from(n.checked_mul(k)?).ok()?,
            stride_c: i64::try_from(m.checked_mul(n)?).ok()?,
        };
        // SAFETY: each batch item is column-major. The B batch stores n×k
        // matrices and cuBLAS transposes each to k×n before multiplication.
        unsafe { blas.gemm_strided_batched(cfg, &a_dev, &b_dev, &mut out_dev) }.ok()?;
        let out_col = stream.clone_dtoh(&out_dev).ok()?;
        from_col_major_batch(&out_col, batch, m, n)
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
    pub(crate) fn xt_diag_x_on_ordinal_cuda(
        ordinal: usize,
        x: ArrayView2<'_, f64>,
        w: ArrayView1<'_, f64>,
    ) -> Option<Array2<f64>> {
        let (rows, cols) = x.dim();
        if rows == 0 || cols == 0 || rows != w.len() {
            return None;
        }
        weighted_crossprod_for(ordinal, x, w, x)
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
        {
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
        };
        let out_col = stream.clone_dtoh(&rhs_dev).ok()?;
        from_col_major(&out_col, n, nrhs)
    }
}

#[cfg(target_os = "linux")]
pub(crate) use cuda_impl::{
    ResidentWeightedGram, gemm_abt_strided_batched_cuda, gemm_broadcast_b_batched_cuda, gemm_cuda,
    gemm_on_ordinal_cuda, gemv_cuda, joint_hessian_2x2_cuda, trsm_cuda, xt_diag_x_cuda,
    xt_diag_x_on_ordinal_cuda, xt_diag_y_cuda,
};
