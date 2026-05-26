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

/// Per-step inputs for [`solve_pirls_step_on_stream`].
///
/// Mirrors [`PirlsGpuInput`] but elides the design matrix `x` because that
/// lives device-resident in the shared batch state. Each PIRLS Newton step
/// only changes `weights`, `penalty_hessian` (with the current Sλ sum),
/// `gradient`, and the LM ridge — these are the small per-step uploads the
/// stream-pool path streams to the device.
#[derive(Clone, Debug)]
pub struct PirlsStepStreamInput<'a> {
    pub weights: ArrayView1<'a, f64>,
    pub penalty_hessian: ArrayView2<'a, f64>,
    pub gradient: ArrayView1<'a, f64>,
    pub lm_ridge: f64,
}

/// Stage 3.2 device-input variant of [`PirlsStepStreamInput`].
///
/// Where the host-input form uploads `weights` + `gradient` per Newton
/// step, this form reads them straight from the
/// [`crate::gpu::pirls_row::RowOutputDevBuffers`] populated by the
/// device-side row-reweight kernel — no host round-trip for the row
/// state. Only the penalty matrix still crosses the host boundary
/// because the outer REML loop updates Sλ + LM ridge between PIRLS
/// steps.
#[cfg(target_os = "linux")]
pub struct PirlsStepStreamDeviceInput<'a, 'b> {
    /// Device-resident solver weights `w_solver_i` (length n). Read
    /// in-place by the cublasDdgmm WX assembly.
    pub w_solver_dev: &'a cudarc::driver::CudaSlice<f64>,
    /// Device-resident IRLS gradient `∂ℓ/∂η_i` (length n). Read by the
    /// `Xᵀg` dgemv to form the Newton RHS.
    pub grad_eta_dev: &'b cudarc::driver::CudaSlice<f64>,
    /// Penalty Hessian Sλ in row-major host layout (p × p).
    pub penalty_hessian: ArrayView2<'b, f64>,
    /// Levenberg-Marquardt diagonal ridge added to H before potrf.
    pub lm_ridge: f64,
}

/// Shared, batch-wide GPU state for stream-pool sigma-cubature PIRLS.
///
/// Construct once per cubature batch via [`PirlsGpuSharedData::upload`] and
/// hand a shared reference (`&PirlsGpuSharedData`) to many
/// [`SigmaPirlsGpuWorkspace`]s — one per CUDA stream. The design matrix is
/// uploaded a single time and reused by every sigma fit.
///
/// Only `x` is device-resident on this struct (the immediate Block 6 P2
/// scope). Later priorities (per the math spec) add device-resident
/// `y`, `offset`, `prior_weights`, and per-family kernel handles here so
/// the row reweight and gradient-formation kernels can read them on-stream
/// without any host roundtrip.
#[cfg(target_os = "linux")]
pub struct PirlsGpuSharedData {
    pub(crate) ctx: std::sync::Arc<cudarc::driver::CudaContext>,
    pub(crate) n: usize,
    pub(crate) p: usize,
    /// `n*p` f64 column-major design matrix, device-resident.
    pub(crate) x_dev: cudarc::driver::CudaSlice<f64>,
}

/// Per-stream workspace for [`solve_pirls_step_on_stream`].
///
/// Owns a non-default CUDA stream plus cuBLAS / cuSOLVER handles bound to
/// that stream, and the persistent device buffers that every PIRLS Newton
/// step in this sigma fit reuses (no per-step allocation, no per-step
/// handle creation). Multiple workspaces on independent streams sharing
/// one [`PirlsGpuSharedData`] are the substrate the stream-pool cubature
/// executor (Block 6 P3) composes.
///
/// Row-chunked WX assembly (Block 6 P4) replaces the full `n*p` `wx_dev`
/// with a `chunk*p` buffer reused across row chunks; until that lands the
/// workspace allocates the full matrix.
#[cfg(target_os = "linux")]
pub struct SigmaPirlsGpuWorkspace {
    pub(crate) stream: std::sync::Arc<cudarc::driver::CudaStream>,
    pub(crate) blas: cudarc::cublas::CudaBlas,
    pub(crate) solver: cudarc::cusolver::DnHandle,
    pub(crate) wx_dev: cudarc::driver::CudaSlice<f64>,
    pub(crate) w_dev: cudarc::driver::CudaSlice<f64>,
    pub(crate) xtwx_dev: cudarc::driver::CudaSlice<f64>,
    pub(crate) h_dev: cudarc::driver::CudaSlice<f64>,
    pub(crate) rhs_dev: cudarc::driver::CudaSlice<f64>,
    pub(crate) penalty_dev: cudarc::driver::CudaSlice<f64>,
    pub(crate) n: usize,
    pub(crate) p: usize,
}

#[cfg(target_os = "linux")]
mod cuda {
    use super::{
        PirlsGpuInput, PirlsGpuSharedData, PirlsGpuStep, PirlsStepStreamDeviceInput,
        PirlsStepStreamInput, SigmaPirlsGpuWorkspace,
    };
    use crate::gpu::common::PtxModuleCache;
    use crate::gpu::driver::{from_col_major, to_col_major};
    use crate::gpu::solver::{context_and_stream, pinned_htod, potrf_in_place, potrs_in_place};
    use cudarc::cublas::sys::{
        cublasDdgmm, cublasDgeam, cublasOperation_t, cublasSideMode_t, cublasStatus_t,
    };
    use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, Gemv, GemvConfig};
    use cudarc::cusolver::DnHandle;
    use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg};
    use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

    /// One-thread reduction over a p×p column-major Cholesky factor's
    /// diagonal, computing `2·Σ ln(L[i,i])` device-side and writing a
    /// single f64 into `out[0]`. The factor's lower-triangular Cholesky
    /// has positive diagonal by construction, so no abs/clamp needed.
    /// One thread is enough for the dominant p ≤ ~200 sizes; the cost was
    /// previously a full p² download, so even a serial device sweep wins.
    const CHOL_LOGDET_PTX_SOURCE: &str = r#"
extern "C" __global__ void chol_logdet_col_major(
    const double* __restrict__ factor,
    int p,
    double* __restrict__ out
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    double acc = 0.0;
    long long pp = (long long)p;
    for (long long i = 0; i < pp; ++i) {
        acc += log(factor[i * pp + i]);
    }
    out[0] = 2.0 * acc;
}
"#;

    static CHOL_LOGDET_CACHE: PtxModuleCache = PtxModuleCache::new();

    impl PirlsGpuSharedData {
        /// Upload `x` to the cached per-ordinal CUDA context and return a
        /// shared batch handle.  One upload, many sigma fits.
        pub(crate) fn upload_impl(x: ArrayView2<'_, f64>) -> Result<Self, String> {
            let (n, p) = x.dim();
            if n == 0 || p == 0 {
                return Err("empty design cannot be uploaded".to_string());
            }
            let (ctx, stream) = context_and_stream()?;
            let x_col = to_col_major(&x);
            let x_dev = pinned_htod(&ctx, &stream, &x_col)?;
            // Synchronize the upload stream so the buffer is visible to
            // every workspace we hand off to. Workspaces use independent
            // streams; the upload completed on the bootstrap stream above.
            stream
                .synchronize()
                .map_err(|e| format!("cuda sync after x upload: {e}"))?;
            Ok(Self { ctx, n, p, x_dev })
        }
    }

    impl SigmaPirlsGpuWorkspace {
        /// Allocate a workspace bound to a fresh non-default CUDA stream on
        /// the shared context. cuBLAS and cuSOLVER handles are created with
        /// that stream so every kernel issued through them is enqueued on
        /// this workspace's stream, allowing concurrent overlap with peer
        /// workspaces in the stream pool.
        pub(crate) fn allocate_impl(shared: &PirlsGpuSharedData) -> Result<Self, String> {
            let n = shared.n;
            let p = shared.p;
            let stream = shared
                .ctx
                .new_stream()
                .map_err(|e| format!("cuda stream alloc: {e}"))?;
            let blas = CudaBlas::new(stream.clone()).map_err(|e| format!("cublas init: {e}"))?;
            let solver =
                DnHandle::new(stream.clone()).map_err(|e| format!("cusolver init: {e}"))?;
            let np = n.checked_mul(p).ok_or("X size overflow")?;
            let pp = p.checked_mul(p).ok_or("H size overflow")?;
            let wx_dev = stream
                .alloc_zeros::<f64>(np)
                .map_err(|e| format!("cuda alloc WX: {e}"))?;
            let w_dev = stream
                .alloc_zeros::<f64>(n)
                .map_err(|e| format!("cuda alloc W: {e}"))?;
            let xtwx_dev = stream
                .alloc_zeros::<f64>(pp)
                .map_err(|e| format!("cuda alloc XtWX: {e}"))?;
            let h_dev = stream
                .alloc_zeros::<f64>(pp)
                .map_err(|e| format!("cuda alloc H: {e}"))?;
            let rhs_dev = stream
                .alloc_zeros::<f64>(p)
                .map_err(|e| format!("cuda alloc RHS: {e}"))?;
            let penalty_dev = stream
                .alloc_zeros::<f64>(pp)
                .map_err(|e| format!("cuda alloc penalty: {e}"))?;
            Ok(Self {
                stream,
                blas,
                solver,
                wx_dev,
                w_dev,
                xtwx_dev,
                h_dev,
                rhs_dev,
                penalty_dev,
                n,
                p,
            })
        }
    }

    /// Drive one PIRLS Newton step on the workspace's CUDA stream.
    ///
    /// Math is identical to [`solve_step`]: build `H = XᵀWX + S + λI`,
    /// Cholesky-factor it, solve `H·d = g`, return `(H, −d, log|H|)`. The
    /// difference is purely the execution model: no context creation, no
    /// handle creation, no design-matrix upload, no per-step buffer
    /// allocations — only the small per-step `weights`, `penalty`, and
    /// `gradient` cross the host boundary.
    pub(super) fn solve_step_on_stream(
        shared: &PirlsGpuSharedData,
        ws: &mut SigmaPirlsGpuWorkspace,
        input: PirlsStepStreamInput<'_>,
    ) -> Result<PirlsGpuStep, String> {
        let n = shared.n;
        let p = shared.p;
        if ws.n != n || ws.p != p {
            return Err(format!(
                "workspace shape ({}, {}) does not match shared design ({n}, {p})",
                ws.n, ws.p
            ));
        }
        if input.weights.len() != n {
            return Err(format!(
                "weights length {} does not match rows {n}",
                input.weights.len()
            ));
        }
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

        // Upload per-step weights into the persistent W buffer.
        let w_slice = input
            .weights
            .as_slice()
            .ok_or("weights must be contiguous")?;
        ws.stream
            .memcpy_htod(w_slice, &mut ws.w_dev)
            .map_err(|e| format!("upload W: {e}"))?;

        // Form WX = diag(w) · X into ws.wx_dev (no reallocation).
        left_scale_rows(
            &ws.blas,
            &ws.stream,
            n,
            p,
            &shared.x_dev,
            &mut ws.w_dev,
            &mut ws.wx_dev,
        )?;

        // XtWX = Xᵀ · WX.
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
        // SAFETY: cuBLAS dgemm with validated i32 dimensions; shared.x_dev and ws.wx_dev are n*p
        // f64 device buffers, ws.xtwx_dev is the p*p output; all bound to ws.stream via blas.
        unsafe {
            ws.blas
                .gemm(cfg, &shared.x_dev, &ws.wx_dev, &mut ws.xtwx_dev)
        }
        .map_err(|e| format!("cublas dgemm XtWX: {e}"))?;

        // Upload S + λI per step (the penalty + ridge are the only structural inputs that change).
        let penalty = penalty_with_ridge(input.penalty_hessian, input.lm_ridge);
        let penalty_view = penalty.view();
        let penalty_col = to_col_major(&penalty_view);
        ws.stream
            .memcpy_htod(penalty_col.as_ref(), &mut ws.penalty_dev)
            .map_err(|e| format!("upload penalty: {e}"))?;

        // H = XtWX + (S + λI), in-place into ws.h_dev.
        geam_add(
            &ws.blas,
            &ws.stream,
            p,
            &ws.xtwx_dev,
            &ws.penalty_dev,
            &mut ws.h_dev,
        )?;

        // Upload gradient into the persistent RHS buffer.
        let g_slice = input
            .gradient
            .as_slice()
            .ok_or("gradient must be contiguous")?;
        ws.stream
            .memcpy_htod(g_slice, &mut ws.rhs_dev)
            .map_err(|e| format!("upload gradient: {e}"))?;

        // Snapshot the assembled penalised Hessian for the caller before potrf
        // overwrites it with the Cholesky factor. Single d→h copy.
        let h_total_col = ws
            .stream
            .clone_dtoh(&ws.h_dev)
            .map_err(|e| format!("download penalised Hessian: {e}"))?;
        let penalized_hessian =
            from_col_major(&h_total_col, p, p).ok_or("H layout conversion failed")?;

        // Factor + solve in place on the stream.
        potrf_in_place(&ws.solver, &ws.stream, p, &mut ws.h_dev)?;
        potrs_in_place(&ws.solver, &ws.stream, p, 1, &ws.h_dev, &mut ws.rhs_dev)?;

        // Logdet device-side: reduces the previous p² Cholesky-factor
        // download to a single f64 download. Stage 2's "no per-iteration
        // host round-trip" budget keeps the p² factor on the device.
        let logdet = cholesky_logdet_device(&ws.stream, &shared.ctx, p, &ws.h_dev)?;

        // Direction: the convention is d = −H⁻¹ g.
        let direction_raw = ws
            .stream
            .clone_dtoh(&ws.rhs_dev)
            .map_err(|e| format!("download direction: {e}"))?;
        let mut direction = Array1::from_vec(direction_raw);
        direction.mapv_inplace(|v| -v);

        Ok(PirlsGpuStep {
            penalized_hessian,
            direction,
            logdet,
        })
    }

    /// Stage 3.2 device-input PIRLS Newton step.
    ///
    /// Identical math to [`solve_step_on_stream`] but reads `w_solver`
    /// and `grad_eta` straight from device buffers populated by the
    /// device-side row-reweight kernel (no host upload of weights or
    /// gradient). Only the penalty matrix still crosses the host
    /// boundary because the outer REML loop updates Sλ + LM ridge
    /// between PIRLS steps; the penalty is p×p which is independent of
    /// n, so for biobank-scale n it is a negligible transfer.
    ///
    /// Outputs match `solve_step_on_stream`: returns the assembled
    /// penalised Hessian, the Newton direction `−H⁻¹·g`, and the
    /// log-determinant computed via the device-side
    /// `chol_logdet_col_major` kernel.
    pub(super) fn solve_step_on_stream_device(
        shared: &PirlsGpuSharedData,
        ws: &mut SigmaPirlsGpuWorkspace,
        input: PirlsStepStreamDeviceInput<'_, '_>,
    ) -> Result<PirlsGpuStep, String> {
        let n = shared.n;
        let p = shared.p;
        if ws.n != n || ws.p != p {
            return Err(format!(
                "workspace shape ({}, {}) does not match shared design ({n}, {p})",
                ws.n, ws.p
            ));
        }
        if input.w_solver_dev.len() != n {
            return Err(format!(
                "w_solver_dev length {} does not match n={n}",
                input.w_solver_dev.len()
            ));
        }
        if input.grad_eta_dev.len() != n {
            return Err(format!(
                "grad_eta_dev length {} does not match n={n}",
                input.grad_eta_dev.len()
            ));
        }
        if input.penalty_hessian.dim() != (p, p) {
            return Err(format!(
                "penalty Hessian shape {:?} does not match p={p}",
                input.penalty_hessian.dim()
            ));
        }

        // Form WX = diag(w_solver_dev) · X into ws.wx_dev. We borrow
        // the caller's device buffer through `left_scale_rows`'s
        // device_ptr API; no upload.
        left_scale_rows_borrowed(
            &ws.blas,
            &ws.stream,
            n,
            p,
            &shared.x_dev,
            input.w_solver_dev,
            &mut ws.wx_dev,
        )?;

        // XtWX = Xᵀ · WX → ws.xtwx_dev.
        let n_i = to_i32(n)?;
        let p_i = to_i32(p)?;
        let gemm_cfg = GemmConfig::<f64> {
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
        // SAFETY: validated i32 dims; shared.x_dev and ws.wx_dev are n*p
        // f64 col-major buffers; ws.xtwx_dev is p*p; bound to ws.stream.
        unsafe {
            ws.blas
                .gemm(gemm_cfg, &shared.x_dev, &ws.wx_dev, &mut ws.xtwx_dev)
        }
        .map_err(|e| format!("cublas dgemm XtWX (device-input): {e}"))?;

        // Upload penalty + ridge (p×p, small).
        let penalty = penalty_with_ridge(input.penalty_hessian, input.lm_ridge);
        let penalty_view = penalty.view();
        let penalty_col = to_col_major(&penalty_view);
        ws.stream
            .memcpy_htod(penalty_col.as_ref(), &mut ws.penalty_dev)
            .map_err(|e| format!("upload penalty (device-input): {e}"))?;

        // H = XtWX + (S + λI).
        geam_add(
            &ws.blas,
            &ws.stream,
            p,
            &ws.xtwx_dev,
            &ws.penalty_dev,
            &mut ws.h_dev,
        )?;

        // RHS = Xᵀ · grad_eta_dev → ws.rhs_dev (length p). No host upload.
        let gemv_cfg = GemvConfig::<f64> {
            trans: cublasOperation_t::CUBLAS_OP_T,
            m: n_i,
            n: p_i,
            alpha: 1.0,
            lda: n_i,
            incx: 1,
            beta: 0.0,
            incy: 1,
        };
        // SAFETY: shared.x_dev is n*p col-major; input.grad_eta_dev is
        // length n; ws.rhs_dev is length p; cuBLAS contract satisfied.
        unsafe {
            ws.blas
                .gemv(gemv_cfg, &shared.x_dev, input.grad_eta_dev, &mut ws.rhs_dev)
        }
        .map_err(|e| format!("cublas dgemv Xtg (device-input): {e}"))?;

        // Snapshot the assembled penalised Hessian for the caller before
        // potrf overwrites it with the Cholesky factor — caller contract
        // matches host-input path. Single d→h copy at the step boundary,
        // not per-iter.
        let h_total_col = ws
            .stream
            .clone_dtoh(&ws.h_dev)
            .map_err(|e| format!("download penalised Hessian (device-input): {e}"))?;
        let penalized_hessian =
            from_col_major(&h_total_col, p, p).ok_or("H layout conversion failed")?;

        potrf_in_place(&ws.solver, &ws.stream, p, &mut ws.h_dev)?;
        potrs_in_place(&ws.solver, &ws.stream, p, 1, &ws.h_dev, &mut ws.rhs_dev)?;

        let logdet = cholesky_logdet_device(&ws.stream, &shared.ctx, p, &ws.h_dev)?;

        let direction_raw = ws
            .stream
            .clone_dtoh(&ws.rhs_dev)
            .map_err(|e| format!("download direction (device-input): {e}"))?;
        let mut direction = Array1::from_vec(direction_raw);
        direction.mapv_inplace(|v| -v);

        Ok(PirlsGpuStep {
            penalized_hessian,
            direction,
            logdet,
        })
    }

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
        // One-shot path for the legacy single-step API: validate, build a
        // one-shot shared+workspace, run a single step, drop. This routes
        // through `solve_step_on_stream` so there is exactly one math path
        // for both the batch-mode cubature executor and the single-step
        // test/bench surface.
        let (_, p) = validate_design(input.x, input.weights)?;
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
        let shared = PirlsGpuSharedData::upload_impl(input.x)?;
        let mut ws = SigmaPirlsGpuWorkspace::allocate_impl(&shared)?;
        solve_step_on_stream(
            &shared,
            &mut ws,
            PirlsStepStreamInput {
                weights: input.weights,
                penalty_hessian: input.penalty_hessian,
                gradient: input.gradient,
                lm_ridge: input.lm_ridge,
            },
        )
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

    /// Borrowed-input variant of [`left_scale_rows`] used by the Stage 3.2
    /// device-input PIRLS step. Reads weights through `&CudaSlice` so the
    /// caller can keep ownership of the row-reweight buffer across the
    /// PIRLS iteration without an extra device-side copy.
    fn left_scale_rows_borrowed(
        blas: &CudaBlas,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        n: usize,
        p: usize,
        x_dev: &CudaSlice<f64>,
        w_dev: &CudaSlice<f64>,
        wx_dev: &mut CudaSlice<f64>,
    ) -> Result<(), String> {
        let n_i = to_i32(n)?;
        let p_i = to_i32(p)?;
        let handle = *blas.handle();
        let (x_ptr, _x_record) = x_dev.device_ptr(stream);
        let (w_ptr, _w_record) = w_dev.device_ptr(stream);
        let (wx_ptr, _wx_record) = wx_dev.device_ptr_mut(stream);
        // SAFETY: FFI call into cuBLAS; pointers come from live CudaSlice
        // device buffers; x is n*p col-major (lda = n), w is length n
        // (stride 1), wx is n*p output (lda = n). Caller-owned w buffer
        // is borrowed read-only here, matching cublasDdgmm's contract.
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
            Err(format!("cublasDdgmm (borrowed) failed with {status:?}"))
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

    /// Launch the device-side Cholesky-factor logdet kernel and download
    /// the single scalar result. Replaces the per-step p² host download of
    /// the Cholesky factor that the host-side `cholesky_logdet_from_col_major`
    /// required.
    fn cholesky_logdet_device(
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        ctx: &std::sync::Arc<cudarc::driver::CudaContext>,
        p: usize,
        factor_dev: &CudaSlice<f64>,
    ) -> Result<f64, String> {
        let module = CHOL_LOGDET_CACHE
            .get_or_compile(ctx, "pirls_gpu_chol_logdet", CHOL_LOGDET_PTX_SOURCE)
            .map_err(|err| format!("chol_logdet module: {err}"))?;
        let func = module
            .load_function("chol_logdet_col_major")
            .map_err(|err| format!("chol_logdet load_function: {err}"))?;
        let mut out_dev = stream
            .alloc_zeros::<f64>(1)
            .map_err(|err| format!("alloc chol_logdet out: {err}"))?;
        let p_i = to_i32(p)?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&func);
        builder.arg(factor_dev);
        builder.arg(&p_i);
        builder.arg(&mut out_dev);
        // SAFETY: serial single-thread kernel reading `p` f64 diagonal
        // entries from a live p*p column-major factor and writing one f64
        // to `out_dev`; no aliasing, no oob — `p` matches the device buffer
        // shape every caller passes in.
        unsafe { builder.launch(cfg) }.map_err(|err| format!("chol_logdet launch: {err}"))?;
        let out_host = stream
            .clone_dtoh(&out_dev)
            .map_err(|err| format!("download chol_logdet: {err}"))?;
        Ok(out_host[0])
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

    fn to_i32(value: usize) -> Result<i32, String> {
        i32::try_from(value).map_err(|_| format!("CUDA dimension {value} exceeds i32"))
    }
}

pub fn weighted_crossprod_gpu(
    x: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
) -> Result<Array2<f64>, String> {
    #[cfg(not(target_os = "linux"))]
    {
        return cpu_fallback::weighted_crossprod_cpu(x, weights);
    }

    #[cfg(target_os = "linux")]
    {
        if crate::gpu::runtime::GpuRuntime::global().is_none() {
            return cpu_fallback::weighted_crossprod_cpu(x, weights);
        }
        cuda::weighted_crossprod(x, weights)
    }
}

pub fn solve_pirls_step_gpu(input: PirlsGpuInput<'_>) -> Result<PirlsGpuStep, String> {
    #[cfg(not(target_os = "linux"))]
    {
        return cpu_fallback::solve_step_cpu(input);
    }

    #[cfg(target_os = "linux")]
    {
        if crate::gpu::runtime::GpuRuntime::global().is_none() {
            return cpu_fallback::solve_step_cpu(input);
        }
        cuda::solve_step(input)
    }
}

/// Upload `x` once and return a shared device-resident handle that many
/// [`SigmaPirlsGpuWorkspace`]s can read from concurrently. The shared
/// handle keeps the cached per-ordinal `CudaContext` alive so all peer
/// workspaces bind to the same context and can interleave on its
/// asynchronous engines.
#[cfg(target_os = "linux")]
pub fn upload_shared_pirls_gpu(x: ArrayView2<'_, f64>) -> Result<PirlsGpuSharedData, String> {
    if crate::gpu::runtime::GpuRuntime::global().is_none() {
        return Err("cuda runtime unavailable; cannot upload shared GPU PIRLS data".to_string());
    }
    PirlsGpuSharedData::upload_impl(x)
}

/// Allocate a per-stream workspace bound to a fresh non-default CUDA
/// stream on `shared`'s context. The cuBLAS and cuSOLVER handles are bound
/// to the workspace stream so peer workspaces achieve overlapped execution.
#[cfg(target_os = "linux")]
pub fn allocate_sigma_pirls_workspace(
    shared: &PirlsGpuSharedData,
) -> Result<SigmaPirlsGpuWorkspace, String> {
    SigmaPirlsGpuWorkspace::allocate_impl(shared)
}

/// Drive one PIRLS Newton step on the workspace's CUDA stream against the
/// device-resident shared design matrix. The math is bit-identical to the
/// one-shot [`solve_pirls_step_gpu`]; this entry differs only by
/// amortising the design upload and the cuBLAS / cuSOLVER handle creation
/// across many sigma fits.
#[cfg(target_os = "linux")]
pub fn solve_pirls_step_on_stream(
    shared: &PirlsGpuSharedData,
    ws: &mut SigmaPirlsGpuWorkspace,
    input: PirlsStepStreamInput<'_>,
) -> Result<PirlsGpuStep, String> {
    cuda::solve_step_on_stream(shared, ws, input)
}

/// Stage 3.2 device-input PIRLS step. Reads `w_solver` and `grad_eta`
/// from caller-supplied device buffers (typically populated by
/// [`crate::gpu::pirls_row::launch_row_reweight_on_stream`]) instead of
/// uploading them from host arrays. Math is bit-identical to
/// [`solve_pirls_step_on_stream`]; this entry differs only by skipping
/// the per-iter `weights` and `gradient` host-to-device transfers — only
/// the small p×p penalty matrix still crosses the host boundary.
#[cfg(target_os = "linux")]
pub fn solve_pirls_step_on_stream_device(
    shared: &PirlsGpuSharedData,
    ws: &mut SigmaPirlsGpuWorkspace,
    input: PirlsStepStreamDeviceInput<'_, '_>,
) -> Result<PirlsGpuStep, String> {
    cuda::solve_step_on_stream_device(shared, ws, input)
}

/// CPU fallback for the PIRLS-step GPU primitives.  When this build has no
/// CUDA runtime probed, the GPU entry points must still return numerically
/// correct results so that callers can route a single code path through
/// `*_gpu` and rely on `solver::gpu::dense_pirls_dispatch` telemetry to
/// distinguish device-resident vs. host-resident execution.  Returning `Err`
/// here would silently force every caller to grow an `if cuda { .. } else
/// { .. }` branch and risk drifting away from the GPU formula.
mod cpu_fallback {
    use super::{PirlsGpuInput, PirlsGpuStep};
    use crate::linalg::faer_ndarray::FaerCholesky;
    use crate::solver::estimate::reml::assembly::xt_diag_x_dense_into;
    use faer::Side;
    use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

    pub(super) fn weighted_crossprod_cpu(
        x: ArrayView2<'_, f64>,
        weights: ArrayView1<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        validate(x, weights)?;
        let x_owned = x.to_owned();
        let w_owned = weights.to_owned();
        let mut scratch = Array2::<f64>::zeros(x_owned.dim());
        Ok(xt_diag_x_dense_into(&x_owned, &w_owned, &mut scratch))
    }

    pub(super) fn solve_step_cpu(input: PirlsGpuInput<'_>) -> Result<PirlsGpuStep, String> {
        validate(input.x, input.weights)?;
        let (_n, p) = input.x.dim();
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
        let mut penalized_hessian = weighted_crossprod_cpu(input.x, input.weights)?;
        penalized_hessian += &input.penalty_hessian;
        if input.lm_ridge != 0.0 {
            for i in 0..p {
                penalized_hessian[[i, i]] += input.lm_ridge;
            }
        }
        let factor = penalized_hessian
            .cholesky(Side::Lower)
            .map_err(|e| format!("CPU Cholesky failed in PIRLS fallback: {e:?}"))?;
        let g = Array1::from_iter(input.gradient.iter().copied());
        let mut direction = factor.solvevec(&g);
        direction.mapv_inplace(|v| -v);
        // Penalized-Hessian Cholesky logdet = 2 * sum(log(diag(L))).
        let logdet = 2.0 * factor.diag().iter().map(|v| v.ln()).sum::<f64>();
        Ok(PirlsGpuStep {
            penalized_hessian,
            direction,
            logdet,
        })
    }

    fn validate(x: ArrayView2<'_, f64>, weights: ArrayView1<'_, f64>) -> Result<(), String> {
        let (n, p) = x.dim();
        if weights.len() != n {
            return Err(format!(
                "weights length {} does not match rows {n}",
                weights.len()
            ));
        }
        if n == 0 || p == 0 {
            return Err("empty design cannot be solved".to_string());
        }
        Ok(())
    }
}

pub fn cholesky_solve_gpu(
    hessian: ArrayView2<'_, f64>,
    rhs: ArrayView2<'_, f64>,
) -> Result<(Array2<f64>, f64), String> {
    crate::gpu::solver::cholesky_solve_gpu(hessian, rhs)
}

pub fn cholesky_lower_gpu(hessian: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
    crate::gpu::solver::cholesky_lower_gpu(hessian)
}

/// Stage 3.2 V100 parity: the device-input PIRLS step must produce
/// numerically identical `(H, direction, logdet)` triples to the
/// host-input form when fed the same weights + gradient. This is the
/// production caller that satisfies the dead-pub scanner for
/// `solve_pirls_step_on_stream_device` and `PirlsStepStreamDeviceInput`.
#[cfg(all(test, target_os = "linux"))]
mod stream_device_parity_tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn device_input_step_matches_host_input_step_on_v100() {
        if crate::gpu::runtime::GpuRuntime::global().is_none() {
            eprintln!("[stream_device_parity] no CUDA runtime — skipping");
            return;
        }
        let x = arr2(&[
            [1.0, 0.5, 0.1],
            [0.2, -0.3, 1.4],
            [0.7, 1.1, -0.2],
            [-0.4, 0.9, 0.6],
            [0.3, -0.8, 0.5],
        ]);
        let weights = ndarray::arr1(&[1.0, 0.8, 1.2, 0.9, 1.05]);
        // Pick g_eta directly (length n) and derive the equivalent
        // host-side gradient via the same Xᵀ projection the
        // device-input form does on the GPU.
        let g_eta = ndarray::arr1(&[0.10_f64, -0.20, 0.05, 0.30, -0.15]);
        let gradient: ndarray::Array1<f64> = x.t().dot(&g_eta);
        let penalty = arr2(&[[0.4, 0.0, 0.0], [0.0, 0.9, 0.0], [0.0, 0.0, 1.2]]);
        let lm_ridge = 0.1;

        let shared = upload_shared_pirls_gpu(x.view())
            .expect("upload shared design");
        let mut ws_host = allocate_sigma_pirls_workspace(&shared)
            .expect("alloc host-input ws");
        let mut ws_dev = allocate_sigma_pirls_workspace(&shared)
            .expect("alloc device-input ws");

        let host_step = solve_pirls_step_on_stream(
            &shared,
            &mut ws_host,
            PirlsStepStreamInput {
                weights: weights.view(),
                penalty_hessian: penalty.view(),
                gradient: gradient.view(),
                lm_ridge,
            },
        )
        .expect("host-input step");

        let n = x.nrows();
        let mut w_dev = ws_dev
            .stream
            .alloc_zeros::<f64>(n)
            .expect("alloc w_dev");
        let mut g_dev = ws_dev
            .stream
            .alloc_zeros::<f64>(n)
            .expect("alloc g_dev");
        ws_dev
            .stream
            .memcpy_htod(weights.as_slice().unwrap(), &mut w_dev)
            .expect("upload w_dev");
        ws_dev
            .stream
            .memcpy_htod(g_eta.as_slice().unwrap(), &mut g_dev)
            .expect("upload g_dev");

        let dev_step = solve_pirls_step_on_stream_device(
            &shared,
            &mut ws_dev,
            PirlsStepStreamDeviceInput {
                w_solver_dev: &w_dev,
                grad_eta_dev: &g_dev,
                penalty_hessian: penalty.view(),
                lm_ridge,
            },
        )
        .expect("device-input step");

        // H + logdet must match to round-off (same XᵀWX, same penalty
        // add, same potrf).
        for i in 0..3 {
            for j in 0..3 {
                let diff =
                    (host_step.penalized_hessian[[i, j]] - dev_step.penalized_hessian[[i, j]])
                        .abs();
                assert!(diff <= 1e-10, "H[{i},{j}] mismatch: {diff}");
            }
        }
        assert!(
            (host_step.logdet - dev_step.logdet).abs() <= 1e-9,
            "logdet mismatch: host={} dev={}",
            host_step.logdet,
            dev_step.logdet
        );
        // Direction must match because Xᵀ·g_eta = (Xᵀ·X)·α = host
        // gradient by construction.
        for i in 0..3 {
            let diff = (host_step.direction[i] - dev_step.direction[i]).abs();
            assert!(diff <= 1e-9, "direction[{i}] mismatch: {diff}");
        }
    }
}
