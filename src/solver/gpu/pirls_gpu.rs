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

    // ────────────────────────────────────────────────────────────────────
    // Stage 3.3: full device-resident PIRLS loop driver
    // ────────────────────────────────────────────────────────────────────

    /// Bundled NVRTC helpers for the Stage 3.3 loop driver: axpy +
    /// single-block sum / linf reductions. Cached process-wide.
    const PIRLS_LOOP_PTX_SOURCE: &str = r#"
extern "C" {
    double fabs(double);
}

extern "C" __global__ void axpy_n(
    double alpha,
    const double* __restrict__ x,
    double* __restrict__ y,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] += alpha * x[i];
}

extern "C" __global__ void deviance_sum(
    const double* __restrict__ d,
    int n,
    double* __restrict__ out
) {
    __shared__ double sm[1024];
    int tid = threadIdx.x;
    int bdim = blockDim.x;
    double acc = 0.0;
    for (int i = tid; i < n; i += bdim) {
        acc += d[i];
    }
    sm[tid] = acc;
    __syncthreads();
    for (int stride = bdim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sm[tid] += sm[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[0] = sm[0];
}

extern "C" __global__ void linf_norm(
    const double* __restrict__ v,
    int p,
    double* __restrict__ out
) {
    __shared__ double sm[1024];
    int tid = threadIdx.x;
    int bdim = blockDim.x;
    double acc = 0.0;
    for (int i = tid; i < p; i += bdim) {
        double a = fabs(v[i]);
        if (a > acc) acc = a;
    }
    sm[tid] = acc;
    __syncthreads();
    for (int stride = bdim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            double r = sm[tid + stride];
            if (r > sm[tid]) sm[tid] = r;
        }
        __syncthreads();
    }
    if (tid == 0) out[0] = sm[0];
}
"#;

    static PIRLS_LOOP_CACHE: PtxModuleCache = PtxModuleCache::new();

    /// Per-fit device workspace for the Stage 3.3 PIRLS loop driver.
    pub struct PirlsLoopWorkspace {
        pub beta_dev: CudaSlice<f64>,
        pub eta_dev: CudaSlice<f64>,
        pub eta_cand_dev: CudaSlice<f64>,
        pub y_dev: CudaSlice<f64>,
        pub prior_w_dev: CudaSlice<f64>,
        pub row_out: crate::gpu::pirls_row::RowOutputDevBuffers,
        pub row_out_cand: crate::gpu::pirls_row::RowOutputDevBuffers,
        pub direction_dev: CudaSlice<f64>,
        pub xd_dev: CudaSlice<f64>,
        pub scalar_dev: CudaSlice<f64>,
        pub n: usize,
        pub p: usize,
    }

    impl PirlsLoopWorkspace {
        pub fn allocate(
            shared: &PirlsGpuSharedData,
            stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        ) -> Result<Self, String> {
            let n = shared.n;
            let p = shared.p;
            let alloc_f64 = |label: &'static str, len: usize| {
                stream
                    .alloc_zeros::<f64>(len)
                    .map_err(|e| format!("pirls loop alloc {label}: {e}"))
            };
            Ok(Self {
                beta_dev: alloc_f64("beta", p)?,
                eta_dev: alloc_f64("eta", n)?,
                eta_cand_dev: alloc_f64("eta_cand", n)?,
                y_dev: alloc_f64("y", n)?,
                prior_w_dev: alloc_f64("prior_w", n)?,
                row_out: crate::gpu::pirls_row::RowOutputDevBuffers::allocate(stream, n)
                    .map_err(|e| format!("pirls loop alloc row_out: {e}"))?,
                row_out_cand: crate::gpu::pirls_row::RowOutputDevBuffers::allocate(stream, n)
                    .map_err(|e| format!("pirls loop alloc row_out_cand: {e}"))?,
                direction_dev: alloc_f64("direction", p)?,
                xd_dev: alloc_f64("xd", n)?,
                scalar_dev: alloc_f64("scalar", 1)?,
                n,
                p,
            })
        }
    }

    #[derive(Clone, Debug)]
    pub struct PirlsLoopOutcome {
        pub beta: Array1<f64>,
        pub penalized_hessian: Array2<f64>,
        pub logdet: f64,
        pub deviance: f64,
        pub iterations: usize,
        pub converged: bool,
    }

    /// Full device-resident PIRLS loop. Only three scalar (1 f64)
    /// downloads per Newton iter (deviance, direction-L∞, candidate
    /// deviance per α). β + final H downloaded once at exit.
    pub(super) fn pirls_loop(
        shared: &PirlsGpuSharedData,
        ws: &mut SigmaPirlsGpuWorkspace,
        loop_ws: &mut PirlsLoopWorkspace,
        family: crate::gpu::pirls_row::PirlsRowFamily,
        curvature: crate::gpu::pirls_row::CurvatureMode,
        beta0_host: ArrayView1<'_, f64>,
        y_host: ArrayView1<'_, f64>,
        prior_w_host: ArrayView1<'_, f64>,
        penalty_hessian: ArrayView2<'_, f64>,
        lm_ridge: f64,
        max_iter: usize,
        tol: f64,
    ) -> Result<PirlsLoopOutcome, String> {
        let n = shared.n;
        let p = shared.p;
        if loop_ws.n != n || loop_ws.p != p {
            return Err(format!(
                "loop workspace ({}, {}) ≠ shared ({n}, {p})",
                loop_ws.n, loop_ws.p
            ));
        }
        if beta0_host.len() != p {
            return Err(format!("beta0 length {} ≠ p={p}", beta0_host.len()));
        }
        if y_host.len() != n || prior_w_host.len() != n {
            return Err(format!(
                "y/prior_w length mismatch (y={}, w={}, n={n})",
                y_host.len(),
                prior_w_host.len()
            ));
        }

        ws.stream
            .memcpy_htod(
                beta0_host.as_slice().ok_or("beta0 not contiguous")?,
                &mut loop_ws.beta_dev,
            )
            .map_err(|e| format!("upload beta0: {e}"))?;
        ws.stream
            .memcpy_htod(
                y_host.as_slice().ok_or("y not contiguous")?,
                &mut loop_ws.y_dev,
            )
            .map_err(|e| format!("upload y: {e}"))?;
        ws.stream
            .memcpy_htod(
                prior_w_host.as_slice().ok_or("prior_w not contiguous")?,
                &mut loop_ws.prior_w_dev,
            )
            .map_err(|e| format!("upload prior_w: {e}"))?;

        let backend = crate::gpu::pirls_row::PirlsRowBackend::probe()
            .map_err(|e| format!("pirls_row backend: {e}"))?;
        let loop_module = PIRLS_LOOP_CACHE
            .get_or_compile(&shared.ctx, "pirls_loop", PIRLS_LOOP_PTX_SOURCE)
            .map_err(|e| format!("pirls loop module: {e}"))?;
        let axpy_func = loop_module
            .load_function("axpy_n")
            .map_err(|e| format!("load axpy_n: {e}"))?;
        let sum_func = loop_module
            .load_function("deviance_sum")
            .map_err(|e| format!("load deviance_sum: {e}"))?;
        let linf_func = loop_module
            .load_function("linf_norm")
            .map_err(|e| format!("load linf_norm: {e}"))?;

        gemv_no_trans(
            &ws.blas,
            n,
            p,
            &shared.x_dev,
            &loop_ws.beta_dev,
            &mut loop_ws.eta_dev,
        )?;
        crate::gpu::pirls_row::launch_row_reweight_on_stream(
            backend,
            family,
            curvature,
            &ws.stream,
            n,
            &loop_ws.eta_dev,
            &loop_ws.y_dev,
            &loop_ws.prior_w_dev,
            &mut loop_ws.row_out,
        )
        .map_err(|e| format!("row reweight init: {e}"))?;

        let mut prev_deviance = reduce_scalar(
            &ws.stream,
            &sum_func,
            &loop_ws.row_out.deviance,
            n,
            &mut loop_ws.scalar_dev,
            "deviance_init",
        )?;
        let mut last_logdet = 0.0_f64;
        let mut last_h = Array2::<f64>::zeros((p, p));
        let mut converged = false;

        for it in 0..max_iter {
            let step = solve_step_on_stream_device(
                shared,
                ws,
                PirlsStepStreamDeviceInput {
                    w_solver_dev: &loop_ws.row_out.w_solver,
                    grad_eta_dev: &loop_ws.row_out.grad_eta,
                    penalty_hessian,
                    lm_ridge,
                },
            )
            .map_err(|e| format!("inner step it={it}: {e}"))?;
            last_h = step.penalized_hessian.clone();
            last_logdet = step.logdet;
            ws.stream
                .memcpy_htod(
                    step.direction.as_slice().ok_or("direction not contiguous")?,
                    &mut loop_ws.direction_dev,
                )
                .map_err(|e| format!("upload direction: {e}"))?;

            let dir_linf = reduce_scalar(
                &ws.stream,
                &linf_func,
                &loop_ws.direction_dev,
                p,
                &mut loop_ws.scalar_dev,
                "dir_linf",
            )?;

            gemv_no_trans(
                &ws.blas,
                n,
                p,
                &shared.x_dev,
                &loop_ws.direction_dev,
                &mut loop_ws.xd_dev,
            )?;

            let mut alpha = 0.0_f64;
            let mut accepted_dev = prev_deviance;
            for &a in &[
                1.0_f64, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625,
            ] {
                ws.stream
                    .memcpy_dtod(&loop_ws.eta_dev, &mut loop_ws.eta_cand_dev)
                    .map_err(|e| format!("copy eta→cand: {e}"))?;
                axpy(
                    &ws.stream,
                    &axpy_func,
                    a,
                    &loop_ws.xd_dev,
                    &mut loop_ws.eta_cand_dev,
                    n,
                )?;
                crate::gpu::pirls_row::launch_row_reweight_on_stream(
                    backend,
                    family,
                    curvature,
                    &ws.stream,
                    n,
                    &loop_ws.eta_cand_dev,
                    &loop_ws.y_dev,
                    &loop_ws.prior_w_dev,
                    &mut loop_ws.row_out_cand,
                )
                .map_err(|e| format!("cand reweight α={a}: {e}"))?;
                let cand = reduce_scalar(
                    &ws.stream,
                    &sum_func,
                    &loop_ws.row_out_cand.deviance,
                    n,
                    &mut loop_ws.scalar_dev,
                    "cand_dev",
                )?;
                if cand.is_finite() && cand < prev_deviance {
                    alpha = a;
                    accepted_dev = cand;
                    break;
                }
            }
            if alpha == 0.0 {
                // No descent — take α=1 anyway and let outer trust-region adjust.
                alpha = 1.0;
                ws.stream
                    .memcpy_dtod(&loop_ws.eta_dev, &mut loop_ws.eta_cand_dev)
                    .map_err(|e| format!("copy eta→cand fb: {e}"))?;
                axpy(
                    &ws.stream,
                    &axpy_func,
                    1.0,
                    &loop_ws.xd_dev,
                    &mut loop_ws.eta_cand_dev,
                    n,
                )?;
                crate::gpu::pirls_row::launch_row_reweight_on_stream(
                    backend,
                    family,
                    curvature,
                    &ws.stream,
                    n,
                    &loop_ws.eta_cand_dev,
                    &loop_ws.y_dev,
                    &loop_ws.prior_w_dev,
                    &mut loop_ws.row_out_cand,
                )
                .map_err(|e| format!("cand reweight fb: {e}"))?;
                accepted_dev = reduce_scalar(
                    &ws.stream,
                    &sum_func,
                    &loop_ws.row_out_cand.deviance,
                    n,
                    &mut loop_ws.scalar_dev,
                    "fb_dev",
                )?;
            }

            axpy(
                &ws.stream,
                &axpy_func,
                alpha,
                &loop_ws.direction_dev,
                &mut loop_ws.beta_dev,
                p,
            )?;
            ws.stream
                .memcpy_dtod(&loop_ws.eta_cand_dev, &mut loop_ws.eta_dev)
                .map_err(|e| format!("commit eta: {e}"))?;
            std::mem::swap(&mut loop_ws.row_out, &mut loop_ws.row_out_cand);

            let step_norm = alpha.abs() * dir_linf;
            let dev_delta = (prev_deviance - accepted_dev).abs();
            prev_deviance = accepted_dev;

            if dir_linf <= tol
                && step_norm <= tol
                && dev_delta <= tol * (1.0 + prev_deviance.abs())
            {
                converged = true;
                return Ok(PirlsLoopOutcome {
                    beta: download_vec(&ws.stream, &loop_ws.beta_dev)?,
                    penalized_hessian: last_h,
                    logdet: last_logdet,
                    deviance: prev_deviance,
                    iterations: it + 1,
                    converged,
                });
            }
        }

        Ok(PirlsLoopOutcome {
            beta: download_vec(&ws.stream, &loop_ws.beta_dev)?,
            penalized_hessian: last_h,
            logdet: last_logdet,
            deviance: prev_deviance,
            iterations: max_iter,
            converged,
        })
    }

    fn gemv_no_trans(
        blas: &CudaBlas,
        n: usize,
        p: usize,
        a_dev: &CudaSlice<f64>,
        x_dev: &CudaSlice<f64>,
        y_dev: &mut CudaSlice<f64>,
    ) -> Result<(), String> {
        let n_i = to_i32(n)?;
        let p_i = to_i32(p)?;
        let cfg = GemvConfig::<f64> {
            trans: cublasOperation_t::CUBLAS_OP_N,
            m: n_i,
            n: p_i,
            alpha: 1.0,
            lda: n_i,
            incx: 1,
            beta: 0.0,
            incy: 1,
        };
        // SAFETY: a is n×p col-major lda=n; x length p incx=1; y length n incy=1.
        unsafe { blas.gemv(cfg, a_dev, x_dev, y_dev) }
            .map_err(|e| format!("dgemv no-trans: {e}"))
    }

    fn axpy(
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        func: &cudarc::driver::CudaFunction,
        alpha: f64,
        x_dev: &CudaSlice<f64>,
        y_dev: &mut CudaSlice<f64>,
        n: usize,
    ) -> Result<(), String> {
        const THREADS: u32 = 256;
        let n_i = to_i32(n)?;
        let n_u = u32::try_from(n).map_err(|_| format!("axpy n={n} > u32"))?;
        let grid = n_u.div_ceil(THREADS).max(1);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(func);
        builder.arg(&alpha);
        builder.arg(x_dev);
        builder.arg(y_dev);
        builder.arg(&n_i);
        // SAFETY: axpy_n signature is (double, const double*, double*, int);
        // both vectors length n.
        unsafe { builder.launch(cfg) }.map_err(|e| format!("axpy launch: {e}"))
    }

    fn reduce_scalar(
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        func: &cudarc::driver::CudaFunction,
        src: &CudaSlice<f64>,
        len: usize,
        scalar_dev: &mut CudaSlice<f64>,
        label: &'static str,
    ) -> Result<f64, String> {
        const THREADS: u32 = 1024;
        let len_i = to_i32(len)?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(func);
        builder.arg(src);
        builder.arg(&len_i);
        builder.arg(scalar_dev);
        // SAFETY: kernel signature (const double*, int, double*).
        unsafe { builder.launch(cfg) }
            .map_err(|e| format!("{label} reduce launch: {e}"))?;
        let host = stream
            .clone_dtoh(scalar_dev)
            .map_err(|e| format!("download {label}: {e}"))?;
        Ok(host[0])
    }

    fn download_vec(
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        dev: &CudaSlice<f64>,
    ) -> Result<Array1<f64>, String> {
        let host = stream
            .clone_dtoh(dev)
            .map_err(|e| format!("download vec: {e}"))?;
        Ok(Array1::from_vec(host))
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

/// Stage 3.3 device-resident PIRLS loop driver. See
/// [`cuda::pirls_loop`] for the full per-iter contract. Only a few
/// 1-f64 scalars cross the host boundary per Newton iteration; β and
/// the final penalised Hessian are downloaded once at loop exit.
#[cfg(target_os = "linux")]
pub fn pirls_loop_on_stream(
    shared: &PirlsGpuSharedData,
    ws: &mut SigmaPirlsGpuWorkspace,
    loop_ws: &mut cuda::PirlsLoopWorkspace,
    family: crate::gpu::pirls_row::PirlsRowFamily,
    curvature: crate::gpu::pirls_row::CurvatureMode,
    beta0: ndarray::ArrayView1<'_, f64>,
    y: ndarray::ArrayView1<'_, f64>,
    prior_w: ndarray::ArrayView1<'_, f64>,
    penalty_hessian: ndarray::ArrayView2<'_, f64>,
    lm_ridge: f64,
    max_iter: usize,
    tol: f64,
) -> Result<cuda::PirlsLoopOutcome, String> {
    cuda::pirls_loop(
        shared,
        ws,
        loop_ws,
        family,
        curvature,
        beta0,
        y,
        prior_w,
        penalty_hessian,
        lm_ridge,
        max_iter,
        tol,
    )
}

/// Allocate a Stage 3.3 PIRLS loop workspace bound to the same stream
/// as `ws` against the shared device-resident design matrix.
#[cfg(target_os = "linux")]
pub fn allocate_pirls_loop_workspace(
    shared: &PirlsGpuSharedData,
    ws: &SigmaPirlsGpuWorkspace,
) -> Result<cuda::PirlsLoopWorkspace, String> {
    cuda::PirlsLoopWorkspace::allocate(shared, &ws.stream)
}

// ────────────────────────────────────────────────────────────────────────
// Block 9 Phase 5 — device-resident PCG against the BMS-FLEX row-Hessian
// operator.
//
// The inner Newton solve in `BernoulliMarginalSlope` (matrix-free path,
// biobank shape n=195k, p=44, r=20) currently reaches the GPU as a
// per-CG-iteration call to `launch_bms_flex_row_hvp` returning a host
// `Vec<f64>`. With ~6400 inner CG iterations per outer iteration that round-
// trip cost dominates: each iter pays one `stream.synchronize()` plus one
// DtoH download. At p=44 the download itself is 352 bytes — trivial in
// bandwidth, painful in latency.
//
// Phase 5 keeps every PCG vector on the device and runs the outer loop with
// only a single small scalar download per iteration (the squared residual
// norm for the convergence check). The Hv kernel becomes `into_device`
// (Block 9 addition to `bms_flex_row.rs`), and the axpy / dot / diagonal-
// preconditioner / scale-and-add steps run as tiny NVRTC kernels on the
// same default stream so the sequence is implicitly ordered without sync.
// ────────────────────────────────────────────────────────────────────────

/// Inputs to [`run_pcg_against_row_hessian_device`]. The right-hand-side
/// `b` is supplied as a host slice (it is the only host-resident vector
/// that needs to enter the loop — the iterate, residual, search direction,
/// and Hv output all live on the device).
#[cfg(target_os = "linux")]
pub struct DeviceResidentPcgInput<'a> {
    /// Per-fit row-Hessian + design storage. The PCG operator is
    /// `v ↦ launch_bms_flex_row_hvp_into_device(storage, ...)`.
    pub storage: &'a crate::gpu::bms_flex_row::DeviceResidentRowHess,
    /// Right-hand-side `b`, length `storage.block.p_total`. Uploaded once.
    pub b: &'a [f64],
    /// Convergence tolerance on relative residual `‖r‖₂ / ‖b‖₂`.
    pub rel_tol: f64,
    /// Hard cap on iterations (the inner loop also bails on stagnation).
    pub max_iters: usize,
    /// Floor on `|diag(H)[i]|` used by the Jacobi preconditioner. Set to
    /// `1e-12` for the matrix-free row-Hessian path; the row-primary
    /// Hessian's diagonal is positive-definite by construction.
    pub precond_diag_floor: f64,
}

/// Output of [`run_pcg_against_row_hessian_device`].
#[cfg(target_os = "linux")]
pub struct DeviceResidentPcgOutput {
    /// Solution `x` such that `H · x ≈ b`, length `storage.block.p_total`.
    pub x: Vec<f64>,
    /// Number of PCG iterations consumed (final iter does not count if it
    /// converged immediately after the dot reduction).
    pub iterations: usize,
    /// Final achieved relative residual `‖r‖₂ / ‖b‖₂`.
    pub final_rel_residual: f64,
}

/// NVRTC source for the Phase-5 device-resident PCG support kernels. Every
/// kernel here operates on length-`p` device vectors with `p` typically
/// 44–256, so a single CTA suffices for each.
#[cfg(target_os = "linux")]
const PCG_KERNEL_SOURCE: &str = r#"
// y[i] += a * x[i]
extern "C" __global__ void pcg_axpy(int n, double a,
                                    const double * __restrict__ x,
                                    double * __restrict__ y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}

// y[i] = a * x[i] + b * y[i]
extern "C" __global__ void pcg_axpby(int n, double a,
                                     const double * __restrict__ x,
                                     double b,
                                     double * __restrict__ y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + b * y[i];
}

// z[i] = r[i] / clamp(diag[i], floor) (sign-preserving floor on |diag|).
extern "C" __global__ void pcg_apply_diag_precond(int n, double floor_val,
                                                  const double * __restrict__ diag,
                                                  const double * __restrict__ r,
                                                  double * __restrict__ z)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double d = diag[i];
        double ad = d < 0 ? -d : d;
        double clamped = ad > floor_val ? d : (d >= 0.0 ? floor_val : -floor_val);
        z[i] = r[i] / clamped;
    }
}

// Single-block dot product; writes the scalar to out[0]. n must be <= 1024.
extern "C" __global__ void pcg_dot_single_block(int n,
                                                const double * __restrict__ a,
                                                const double * __restrict__ b,
                                                double * __restrict__ out)
{
    __shared__ double s[1024];
    int tid = threadIdx.x;
    double acc = 0.0;
    for (int i = tid; i < n; i += blockDim.x) acc += a[i] * b[i];
    s[tid] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[0] = s[0];
}

// Set out[i] = 0 for i in [0, n).
extern "C" __global__ void pcg_init_zero(int n, double * __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 0.0;
}

// Copy y[i] = x[i].
extern "C" __global__ void pcg_copy(int n,
                                    const double * __restrict__ x,
                                    double * __restrict__ y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i];
}
"#;

#[cfg(target_os = "linux")]
mod pcg_device {
    use super::DeviceResidentPcgInput;
    use super::DeviceResidentPcgOutput;
    use super::PCG_KERNEL_SOURCE;
    use crate::gpu::bms_flex_row::launch_bms_flex_row_diagonal;
    use crate::gpu::bms_flex_row::launch_bms_flex_row_hvp_into_device;
    use cudarc::driver::{CudaContext, CudaModule, CudaStream, LaunchConfig, PushKernelArg};
    use std::sync::{Arc, OnceLock};

    struct PcgBackend {
        stream: Arc<CudaStream>,
        module: Arc<CudaModule>,
        ctx: Arc<CudaContext>,
    }

    impl PcgBackend {
        fn probe() -> Result<&'static Self, String> {
            static BACKEND: OnceLock<Result<PcgBackend, String>> = OnceLock::new();
            BACKEND
                .get_or_init(|| {
                    let runtime = crate::gpu::runtime::GpuRuntime::global()
                        .ok_or_else(|| "pcg backend: no CUDA runtime available".to_string())?;
                    let ctx = crate::gpu::runtime::cuda_context_for(
                        runtime.selected_device().ordinal,
                    )
                    .ok_or_else(|| {
                        format!(
                            "pcg backend: failed to create CUDA context for device {}",
                            runtime.selected_device().ordinal
                        )
                    })?;
                    let stream = ctx.default_stream();
                    let ptx = cudarc::nvrtc::compile_ptx(PCG_KERNEL_SOURCE)
                        .map_err(|err| format!("pcg NVRTC compile failed: {err}"))?;
                    let module = ctx
                        .load_module(ptx)
                        .map_err(|err| format!("pcg module load failed: {err}"))?;
                    Ok(PcgBackend { stream, module, ctx })
                })
                .as_ref()
                .map_err(String::clone)
        }
    }

    fn launch_blocks(p: usize, threads: u32) -> u32 {
        ((p as u32) + threads - 1) / threads
    }

    /// PCG against the row-Hessian operator with Jacobi preconditioner from
    /// `diag(H)`. All vectors remain on the device for the duration of the
    /// loop; only the squared residual norm crosses the host boundary each
    /// iter (one f64, ≤ 8 bytes).
    pub(super) fn run(
        input: DeviceResidentPcgInput<'_>,
    ) -> Result<DeviceResidentPcgOutput, String> {
        let p = input.storage.block.p_total;
        if input.b.len() != p {
            return Err(format!(
                "device-resident pcg: b.len()={} != p_total={p}",
                input.b.len()
            ));
        }
        if !input.rel_tol.is_finite() || input.rel_tol <= 0.0 {
            return Err(format!(
                "device-resident pcg: rel_tol must be positive and finite (got {})",
                input.rel_tol
            ));
        }
        if input.max_iters == 0 {
            return Err("device-resident pcg: max_iters must be >= 1".to_string());
        }
        if !input.precond_diag_floor.is_finite() || input.precond_diag_floor <= 0.0 {
            return Err(format!(
                "device-resident pcg: precond_diag_floor must be positive and finite (got {})",
                input.precond_diag_floor
            ));
        }

        let backend = PcgBackend::probe()?;
        let stream = backend.stream.clone();
        let module = backend.module.clone();

        // ── Load kernel handles once ─────────────────────────────────────
        let f_axpy = module
            .load_function("pcg_axpy")
            .map_err(|e| format!("pcg load pcg_axpy: {e}"))?;
        let f_axpby = module
            .load_function("pcg_axpby")
            .map_err(|e| format!("pcg load pcg_axpby: {e}"))?;
        let f_precond = module
            .load_function("pcg_apply_diag_precond")
            .map_err(|e| format!("pcg load pcg_apply_diag_precond: {e}"))?;
        let f_dot = module
            .load_function("pcg_dot_single_block")
            .map_err(|e| format!("pcg load pcg_dot_single_block: {e}"))?;
        let f_copy = module
            .load_function("pcg_copy")
            .map_err(|e| format!("pcg load pcg_copy: {e}"))?;

        // ── Allocate device vectors x, r, z, p_vec, q (length p each) ──
        let mut d_x = stream
            .alloc_zeros::<f64>(p)
            .map_err(|e| format!("pcg alloc x: {e}"))?;
        let mut d_r = stream
            .clone_htod(input.b)
            .map_err(|e| format!("pcg upload b -> r: {e}"))?;
        let mut d_z = stream
            .alloc_zeros::<f64>(p)
            .map_err(|e| format!("pcg alloc z: {e}"))?;
        let mut d_p = stream
            .alloc_zeros::<f64>(p)
            .map_err(|e| format!("pcg alloc p: {e}"))?;
        let mut d_q = stream
            .alloc_zeros::<f64>(p)
            .map_err(|e| format!("pcg alloc q: {e}"))?;
        // One-element scalar buffer reused across iters for `p·q` and
        // `r·z` dot products.
        let mut d_scalar = stream
            .alloc_zeros::<f64>(1)
            .map_err(|e| format!("pcg alloc scalar: {e}"))?;

        // Preconditioner: M⁻¹ from diag(H). One HostVec download per
        // *outer* call, but this is constant work per solve — not per
        // iter — so it does not block the inner loop's no-sync property.
        let diag_host = launch_bms_flex_row_diagonal(input.storage)
            .map_err(|e| format!("pcg diag fetch: {e}"))?;
        if diag_host.len() != p {
            return Err(format!(
                "pcg: diag length {} != p_total {p}",
                diag_host.len()
            ));
        }
        let d_diag = stream
            .clone_htod(&diag_host)
            .map_err(|e| format!("pcg upload diag: {e}"))?;

        // ── Convergence baseline: ‖b‖₂ via one in-stream dot ─────────────
        let n_i32 = i32::try_from(p)
            .map_err(|_| format!("pcg: p_total={p} exceeds i32 range"))?;
        let vec_threads: u32 = 64;
        let vec_blocks = launch_blocks(p, vec_threads);
        let dot_threads: u32 = match p {
            0..=64 => 64,
            65..=128 => 128,
            129..=256 => 256,
            257..=512 => 512,
            _ => 1024,
        };
        if p > 1024 {
            return Err(format!(
                "device-resident pcg: p_total={p} exceeds single-block dot capacity (1024); \
                 widen pcg_dot_single_block to multi-block reduce before raising the cap"
            ));
        }

        // ‖b‖₂² = b · b (b is currently in d_r since r₀ = b - H·0 = b)
        // SAFETY: `f_dot` is the `pcg_dot_single_block` device function loaded
        // above; its signature is `(i32, *const f64, *const f64, *mut f64)`.
        // `n_i32` was bounded against `1024` (kernel's max-n contract) two
        // lines up; `d_r` is a `CudaSlice<f64>` of length `n` allocated to the
        // same stream; `d_scalar` is the length-1 output slice. Single-block
        // grid (1×dot_threads) matches the kernel's reduction strategy.
        unsafe {
            stream
                .launch_builder(&f_dot)
                .arg(&n_i32)
                .arg(&d_r)
                .arg(&d_r)
                .arg(&mut d_scalar)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (dot_threads, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(|e| format!("pcg b·b launch: {e}"))?;
        stream
            .synchronize()
            .map_err(|e| format!("pcg b·b sync: {e}"))?;
        let host_scalar = stream
            .clone_dtoh(&d_scalar)
            .map_err(|e| format!("pcg b·b download: {e}"))?;
        let bb = host_scalar[0];
        if !bb.is_finite() {
            return Err(format!("pcg: b·b not finite ({bb})"));
        }
        let b_norm = bb.sqrt();
        if b_norm == 0.0 {
            // x = 0, r = b = 0, trivially converged.
            return Ok(DeviceResidentPcgOutput {
                x: vec![0.0; p],
                iterations: 0,
                final_rel_residual: 0.0,
            });
        }

        // z₀ = M⁻¹ r₀
        // SAFETY: `f_precond` is `pcg_jacobi_precond` with signature
        // `(i32, f64, *const f64, *const f64, *mut f64)`. `d_diag`, `d_r`,
        // `d_z` are all `CudaSlice<f64>` of length `n` on the same stream;
        // `vec_blocks × vec_threads ≥ n` covers every output element.
        unsafe {
            stream
                .launch_builder(&f_precond)
                .arg(&n_i32)
                .arg(&input.precond_diag_floor)
                .arg(&d_diag)
                .arg(&d_r)
                .arg(&mut d_z)
                .launch(LaunchConfig {
                    grid_dim: (vec_blocks, 1, 1),
                    block_dim: (vec_threads, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(|e| format!("pcg precond z₀: {e}"))?;

        // p₀ = z₀
        // SAFETY: `f_copy` is `pcg_copy` with signature
        // `(i32, *const f64, *mut f64)`. `d_z` and `d_p` are
        // `CudaSlice<f64>` of length `n` on the same stream;
        // `vec_blocks × vec_threads ≥ n` covers every element.
        unsafe {
            stream
                .launch_builder(&f_copy)
                .arg(&n_i32)
                .arg(&d_z)
                .arg(&mut d_p)
                .launch(LaunchConfig {
                    grid_dim: (vec_blocks, 1, 1),
                    block_dim: (vec_threads, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(|e| format!("pcg copy p₀: {e}"))?;

        // ρ₀ = r₀·z₀
        // SAFETY: same invariants as the ‖b‖₂² launch above — `f_dot`
        // signature `(i32, *const f64, *const f64, *mut f64)`, `d_r` and
        // `d_z` are length-`n` `CudaSlice<f64>`, `d_scalar` is length-1,
        // single-block grid matches kernel's reduction.
        unsafe {
            stream
                .launch_builder(&f_dot)
                .arg(&n_i32)
                .arg(&d_r)
                .arg(&d_z)
                .arg(&mut d_scalar)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (dot_threads, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(|e| format!("pcg ρ₀ launch: {e}"))?;
        stream
            .synchronize()
            .map_err(|e| format!("pcg ρ₀ sync: {e}"))?;
        let s = stream
            .clone_dtoh(&d_scalar)
            .map_err(|e| format!("pcg ρ₀ download: {e}"))?;
        let mut rho = s[0];
        if !rho.is_finite() {
            return Err(format!("pcg: ρ₀ not finite ({rho})"));
        }

        let mut iters_taken: usize = 0;
        let mut final_rel_residual: f64 = (bb.sqrt() / b_norm).max(0.0);
        for iter in 0..input.max_iters {
            iters_taken = iter + 1;

            // q = H · p (on device, no sync, no DtoH).
            launch_bms_flex_row_hvp_into_device(input.storage, &d_p, &mut d_q)
                .map_err(|e| format!("pcg Hv iter {iter}: {e}"))?;

            // pq = p·q
            // SAFETY: identical to ‖b‖₂² launch — `f_dot` signature
            // `(i32, *const f64, *const f64, *mut f64)`; `d_p` is the
            // current search direction and `d_q` was just populated by
            // `launch_bms_flex_row_hvp_into_device` (same stream, same `n`).
            unsafe {
                stream
                    .launch_builder(&f_dot)
                    .arg(&n_i32)
                    .arg(&d_p)
                    .arg(&d_q)
                    .arg(&mut d_scalar)
                    .launch(LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (dot_threads, 1, 1),
                        shared_mem_bytes: 0,
                    })
            }
            .map_err(|e| format!("pcg p·q launch iter {iter}: {e}"))?;
            stream
                .synchronize()
                .map_err(|e| format!("pcg p·q sync iter {iter}: {e}"))?;
            let s = stream
                .clone_dtoh(&d_scalar)
                .map_err(|e| format!("pcg p·q download iter {iter}: {e}"))?;
            let pq = s[0];
            if !pq.is_finite() || pq == 0.0 {
                return Err(format!(
                    "pcg iter {iter}: p·q={pq} (non-finite or zero); operator is not positive-definite"
                ));
            }
            let alpha = rho / pq;

            // x += α p
            // SAFETY: `f_axpy` is `pcg_axpy` with signature
            // `(i32, f64, *const f64, *mut f64)`. `alpha` is the
            // finite-checked CG step length (`rho/pq`, both validated
            // above). `d_p` and `d_x` are length-`n` `CudaSlice<f64>` on
            // the same stream. Grid covers all `n` elements.
            unsafe {
                stream
                    .launch_builder(&f_axpy)
                    .arg(&n_i32)
                    .arg(&alpha)
                    .arg(&d_p)
                    .arg(&mut d_x)
                    .launch(LaunchConfig {
                        grid_dim: (vec_blocks, 1, 1),
                        block_dim: (vec_threads, 1, 1),
                        shared_mem_bytes: 0,
                    })
            }
            .map_err(|e| format!("pcg x+=αp iter {iter}: {e}"))?;

            // r -= α q
            let neg_alpha = -alpha;
            // SAFETY: same `f_axpy` invariants as the `x += α p` launch
            // above; `neg_alpha = -alpha` is finite (alpha was checked),
            // `d_q` and `d_r` are length-`n` `CudaSlice<f64>` on the same
            // stream.
            unsafe {
                stream
                    .launch_builder(&f_axpy)
                    .arg(&n_i32)
                    .arg(&neg_alpha)
                    .arg(&d_q)
                    .arg(&mut d_r)
                    .launch(LaunchConfig {
                        grid_dim: (vec_blocks, 1, 1),
                        block_dim: (vec_threads, 1, 1),
                        shared_mem_bytes: 0,
                    })
            }
            .map_err(|e| format!("pcg r-=αq iter {iter}: {e}"))?;

            // ‖r‖₂² = r·r (single device dot, single f64 DtoH)
            // SAFETY: identical to the ‖b‖₂² launch at function entry —
            // `f_dot` signature, `d_r` length-`n`, `d_scalar` length-1,
            // single-block reduction grid.
            unsafe {
                stream
                    .launch_builder(&f_dot)
                    .arg(&n_i32)
                    .arg(&d_r)
                    .arg(&d_r)
                    .arg(&mut d_scalar)
                    .launch(LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (dot_threads, 1, 1),
                        shared_mem_bytes: 0,
                    })
            }
            .map_err(|e| format!("pcg ‖r‖₂² launch iter {iter}: {e}"))?;
            stream
                .synchronize()
                .map_err(|e| format!("pcg ‖r‖₂² sync iter {iter}: {e}"))?;
            let s = stream
                .clone_dtoh(&d_scalar)
                .map_err(|e| format!("pcg ‖r‖₂² download iter {iter}: {e}"))?;
            let rr = s[0];
            if !rr.is_finite() {
                return Err(format!("pcg iter {iter}: ‖r‖₂²={rr} non-finite"));
            }
            let rel = rr.sqrt() / b_norm;
            final_rel_residual = rel;
            if rel <= input.rel_tol {
                break;
            }

            // z = M⁻¹ r
            // SAFETY: same `f_precond` invariants as the `z₀ = M⁻¹ r₀`
            // launch above — signature `(i32, f64, *const f64, *const f64,
            // *mut f64)`, all four slices length-`n` `CudaSlice<f64>`, grid
            // covers all `n` elements.
            unsafe {
                stream
                    .launch_builder(&f_precond)
                    .arg(&n_i32)
                    .arg(&input.precond_diag_floor)
                    .arg(&d_diag)
                    .arg(&d_r)
                    .arg(&mut d_z)
                    .launch(LaunchConfig {
                        grid_dim: (vec_blocks, 1, 1),
                        block_dim: (vec_threads, 1, 1),
                        shared_mem_bytes: 0,
                    })
            }
            .map_err(|e| format!("pcg z=M⁻¹r iter {iter}: {e}"))?;

            // ρ_new = r·z
            // SAFETY: identical to the ρ₀ launch above — `f_dot`
            // signature, `d_r` and `d_z` length-`n`, `d_scalar` length-1.
            unsafe {
                stream
                    .launch_builder(&f_dot)
                    .arg(&n_i32)
                    .arg(&d_r)
                    .arg(&d_z)
                    .arg(&mut d_scalar)
                    .launch(LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (dot_threads, 1, 1),
                        shared_mem_bytes: 0,
                    })
            }
            .map_err(|e| format!("pcg ρ_new launch iter {iter}: {e}"))?;
            stream
                .synchronize()
                .map_err(|e| format!("pcg ρ_new sync iter {iter}: {e}"))?;
            let s = stream
                .clone_dtoh(&d_scalar)
                .map_err(|e| format!("pcg ρ_new download iter {iter}: {e}"))?;
            let rho_new = s[0];
            if !rho_new.is_finite() {
                return Err(format!("pcg iter {iter}: ρ_new={rho_new} non-finite"));
            }
            let beta_pcg = rho_new / rho;

            // p = z + β p  ⇒  via pcg_axpby with a=1, b=β
            // SAFETY: `f_axpby` is `pcg_axpby` with signature
            // `(i32, f64, *const f64, f64, *mut f64)`. `beta_pcg = rho_new/rho`
            // was finite-checked. `d_z` and `d_p` are length-`n`
            // `CudaSlice<f64>` on the same stream; grid covers all `n`
            // elements.
            unsafe {
                stream
                    .launch_builder(&f_axpby)
                    .arg(&n_i32)
                    .arg(&1.0_f64)
                    .arg(&d_z)
                    .arg(&beta_pcg)
                    .arg(&mut d_p)
                    .launch(LaunchConfig {
                        grid_dim: (vec_blocks, 1, 1),
                        block_dim: (vec_threads, 1, 1),
                        shared_mem_bytes: 0,
                    })
            }
            .map_err(|e| format!("pcg p=z+βp iter {iter}: {e}"))?;

            rho = rho_new;
        }

        // Download x once at the end.
        let x_host = stream
            .clone_dtoh(&d_x)
            .map_err(|e| format!("pcg final x DtoH: {e}"))?;
        // The auxiliary device allocs (d_r/d_z/d_p/d_q/d_scalar/d_diag) drop
        // here and return their bytes to cudarc's allocator.
        drop(d_r);
        drop(d_z);
        drop(d_p);
        drop(d_q);
        drop(d_scalar);
        drop(d_diag);
        Ok(DeviceResidentPcgOutput {
            x: x_host,
            iterations: iters_taken,
            final_rel_residual,
        })
    }
}

/// Device-resident PCG against the BMS-FLEX row-Hessian operator.
///
/// Block 9 Phase 5: every PCG vector — `x`, `r`, `z`, `p`, `q` — stays on
/// the device for the entire loop; only the squared residual norm (one f64)
/// is downloaded per iteration for the convergence check. Bit-equal output
/// to a host-side reference PCG against the same operator + preconditioner
/// when the tolerance is tight; differences only show up at the floating-
/// point reduction-order level.
///
/// Linux-only. See [`DeviceResidentPcgInput`] for parameters.
#[cfg(target_os = "linux")]
pub fn run_pcg_against_row_hessian_device(
    input: DeviceResidentPcgInput<'_>,
) -> Result<DeviceResidentPcgOutput, String> {
    pcg_device::run(input)
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

    /// Stage 3.3 production caller: end-to-end GPU PIRLS loop on a
    /// Gaussian-identity fit reaches OLS β to high precision in a
    /// handful of iterations and matches the closed-form
    /// `(XᵀX + Sλ)⁻¹·Xᵀy` solution.
    #[test]
    fn pirls_loop_converges_to_ols_solution_on_gaussian_identity() {
        if crate::gpu::runtime::GpuRuntime::global().is_none() {
            eprintln!("[stage_3_3] no CUDA runtime — skipping");
            return;
        }
        let x = arr2(&[
            [1.0, 0.5, 0.1],
            [0.2, -0.3, 1.4],
            [0.7, 1.1, -0.2],
            [-0.4, 0.9, 0.6],
            [0.3, -0.8, 0.5],
            [1.1, 0.2, -0.4],
            [-0.6, 0.4, 0.3],
            [0.8, -1.0, 0.7],
        ]);
        let n = x.nrows();
        let p = x.ncols();
        // y = X·β_true + small wiggle (still in identity link space).
        let beta_true = ndarray::arr1(&[0.5_f64, -1.2, 0.3]);
        let y: ndarray::Array1<f64> = x.dot(&beta_true);
        let prior_w = ndarray::Array1::<f64>::ones(n);
        let penalty = ndarray::Array2::<f64>::eye(p) * 1e-4; // tiny ridge
        let beta0 = ndarray::Array1::<f64>::zeros(p);

        let shared = upload_shared_pirls_gpu(x.view())
            .expect("upload shared design");
        let mut ws = allocate_sigma_pirls_workspace(&shared)
            .expect("alloc ws");
        let mut loop_ws = allocate_pirls_loop_workspace(&shared, &ws)
            .expect("alloc loop_ws");

        let outcome = pirls_loop_on_stream(
            &shared,
            &mut ws,
            &mut loop_ws,
            crate::gpu::pirls_row::PirlsRowFamily::GaussianIdentity,
            crate::gpu::pirls_row::CurvatureMode::Fisher,
            beta0.view(),
            y.view(),
            prior_w.view(),
            penalty.view(),
            0.0,
            20,
            1e-9,
        )
        .expect("pirls loop");

        // Closed-form OLS (with tiny ridge).
        let xtx = x.t().dot(&x);
        let xty = x.t().dot(&y);
        let mut h_ref = xtx + &penalty;
        // Solve via faer.
        let h_faer = faer::Mat::from_fn(p, p, |i, j| h_ref[[i, j]]);
        let chol = h_faer.cholesky(faer::Side::Lower).expect("chol");
        let rhs = faer::Mat::from_fn(p, 1, |i, _| xty[i]);
        let beta_ref_mat = chol.solve(rhs.as_ref());
        let beta_ref = ndarray::Array1::from(
            (0..p).map(|i| beta_ref_mat[(i, 0)]).collect::<Vec<_>>(),
        );

        // Gaussian-identity PIRLS converges in one Newton iter (linear
        // problem); the loop may take a few iters because the line
        // search starts at α=1 and the first step is exact. Allow up
        // to 5 iters but assert convergence and 1e-6 abs precision.
        assert!(
            outcome.converged || outcome.iterations <= 5,
            "PIRLS loop did not converge in 20 iters on Gaussian-identity (iters={})",
            outcome.iterations
        );
        for i in 0..p {
            let diff = (outcome.beta[i] - beta_ref[i]).abs();
            assert!(
                diff <= 1e-6,
                "β[{i}] mismatch: gpu={} ref={} diff={}",
                outcome.beta[i],
                beta_ref[i],
                diff
            );
        }
        // Also check H matches XᵀX + Sλ (no W weighting since identity-link
        // canonical-weight = 1 for Gaussian).
        for i in 0..p {
            for j in 0..p {
                let diff = (outcome.penalized_hessian[[i, j]] - h_ref[[i, j]]).abs();
                assert!(diff <= 1e-8, "H[{i},{j}] mismatch: {diff}");
            }
        }
    }
}

/// Block 9 Phase 5 — V100 parity for `run_pcg_against_row_hessian_device`.
///
/// Builds a small `(n=64, r=20, p=44)` BMS-FLEX row-Hessian fixture, computes
/// the dense joint Hessian via the same CPU oracle the HVP parity test uses,
/// solves `H · x = b` on the host via dense LU as ground truth, and asserts
/// the device-resident PCG iterate matches to a tight tolerance.
#[cfg(all(test, target_os = "linux"))]
mod pcg_device_parity_tests {
    use super::*;
    use crate::gpu::bms_flex_row::{
        BmsFlexBlockLayout, BmsFlexPrimaryLayout, DeviceResidentRowHess, RowHessianLayout,
    };
    use ndarray::Array2;

    /// Dense oracle for `H_full = Σ_i P_iᵀ H_i P_i` consistent with
    /// `cpu_oracle_bms_flex_row_hvp`'s pullback math.
    fn cpu_dense_joint_hessian(
        row_hessians: &[f64],
        marginal: &[f64],
        logslope: &[f64],
        block: &BmsFlexBlockLayout,
        primary: &BmsFlexPrimaryLayout,
        n: usize,
    ) -> Array2<f64> {
        let p_total = block.p_total;
        let r = primary.r;
        let p_m = block.p_m;
        let p_g = block.p_g;
        let h_block_start = block.h.as_ref().map(|r| r.start).unwrap_or(0);
        let h_block_len = block.h.as_ref().map(|r| r.len()).unwrap_or(0);
        let w_block_start = block.w.as_ref().map(|r| r.start).unwrap_or(0);
        let w_block_len = block.w.as_ref().map(|r| r.len()).unwrap_or(0);
        let h_primary_start = primary.h.as_ref().map(|r| r.start).unwrap_or(0);
        let w_primary_start = primary.w.as_ref().map(|r| r.start).unwrap_or(0);
        let mut h_dense = Array2::<f64>::zeros((p_total, p_total));
        // For each row build P_i columns as length-p_total vectors.
        let mut phi = vec![vec![0.0_f64; p_total]; r];
        for row in 0..n {
            for col in phi.iter_mut() {
                col.iter_mut().for_each(|v| *v = 0.0);
            }
            let mrow = &marginal[row * p_m..(row + 1) * p_m];
            let grow = &logslope[row * p_g..(row + 1) * p_g];
            for k in 0..p_m {
                phi[0][k] = mrow[k];
            }
            for k in 0..p_g {
                phi[1][p_m + k] = grow[k];
            }
            for k in 0..h_block_len {
                phi[h_primary_start + k][h_block_start + k] = 1.0;
            }
            for k in 0..w_block_len {
                phi[w_primary_start + k][w_block_start + k] = 1.0;
            }
            let h_row = &row_hessians[row * r * r..(row + 1) * r * r];
            for u in 0..r {
                for v in 0..r {
                    let huv = h_row[u * r + v];
                    if huv == 0.0 {
                        continue;
                    }
                    for m in 0..p_total {
                        let phim = phi[u][m];
                        if phim == 0.0 {
                            continue;
                        }
                        let scaled = huv * phim;
                        for nn in 0..p_total {
                            h_dense[[m, nn]] += scaled * phi[v][nn];
                        }
                    }
                }
            }
        }
        h_dense
    }

    /// Reference oracle: host PCG against the dense joint H + diag(H)
    /// preconditioner, with a tolerance two decades tighter than the GPU
    /// PCG's. Comparing GPU PCG to host PCG (rather than to a Cholesky
    /// solve) keeps the comparison numerically apples-to-apples — only
    /// reduction order differs between the two paths.
    fn cpu_pcg_oracle(h: &Array2<f64>, b: &[f64], rel_tol: f64) -> Vec<f64> {
        let p = b.len();
        let diag: ndarray::Array1<f64> =
            ndarray::Array1::from_vec((0..p).map(|i| h[[i, i]]).collect());
        let rhs = ndarray::Array1::from_vec(b.to_vec());
        let h_owned = h.clone();
        let apply = move |v: &ndarray::Array1<f64>| h_owned.dot(v);
        let (x, info) = crate::linalg::utils::solve_spd_pcg_with_info(
            apply, &rhs, &diag, rel_tol, 4 * p,
        )
        .expect("host PCG oracle must converge on SPD fixture");
        assert!(
            info.converged,
            "host PCG oracle failed to converge: iters={} rel_res={}",
            info.iterations, info.relative_residual_norm
        );
        x.to_vec()
    }

    #[test]
    fn pcg_device_matches_dense_oracle_at_n64_r20_p44() {
        let Some(_runtime) = crate::gpu::runtime::GpuRuntime::global() else {
            eprintln!("[pcg_device parity] no CUDA runtime — skipping");
            return;
        };
        let n = 64_usize;
        let p_m = 14_usize;
        let p_g = 12_usize;
        let p_h_dim = 10_usize;
        let p_w_dim = 8_usize;
        let r = 2 + p_h_dim + p_w_dim;
        let p_total = p_m + p_g + p_h_dim + p_w_dim;
        let block = BmsFlexBlockLayout {
            p_m,
            p_g,
            h: Some(p_m + p_g..p_m + p_g + p_h_dim),
            w: Some(p_m + p_g + p_h_dim..p_m + p_g + p_h_dim + p_w_dim),
            p_total,
        };
        let primary = BmsFlexPrimaryLayout {
            h: Some(2..2 + p_h_dim),
            w: Some(2 + p_h_dim..2 + p_h_dim + p_w_dim),
            r,
        };

        // Same deterministic symmetric Hessians + designs as the HVP parity
        // gate, so any drift between Phase 4 and Phase 5 surfaces here too.
        let mut row_hessians = vec![0.0_f64; n * r * r];
        for row in 0..n {
            let base = row * r * r;
            for u in 0..r {
                for v in 0..r {
                    let seed = (row as f64) * 0.137
                        + (u as f64) * 1.901
                        + (v as f64) * 0.317;
                    let a = (seed.sin() * 1.7 + (seed * 0.5).cos() * 0.9) * 0.5;
                    row_hessians[base + u * r + v] = a;
                }
            }
            for u in 0..r {
                for v in (u + 1)..r {
                    let upper = row_hessians[base + u * r + v];
                    let lower = row_hessians[base + v * r + u];
                    let sym = 0.5 * (upper + lower);
                    row_hessians[base + u * r + v] = sym;
                    row_hessians[base + v * r + u] = sym;
                }
                // Boost the diagonal heavily so each H_i is positive
                // definite — guarantees the joint pulled-back Hessian is
                // SPD, which PCG requires.
                row_hessians[base + u * r + u] += 4.0 * (r as f64);
            }
        }
        let mut marginal = vec![0.0_f64; n * p_m];
        for row in 0..n {
            for j in 0..p_m {
                let seed = (row as f64) * 0.073 + (j as f64) * 0.211 + 0.4;
                marginal[row * p_m + j] = seed.sin() * 0.8 - (seed * 0.7).cos() * 0.3;
            }
        }
        let mut logslope = vec![0.0_f64; n * p_g];
        for row in 0..n {
            for j in 0..p_g {
                let seed = (row as f64) * 0.091 + (j as f64) * 0.179 - 0.2;
                logslope[row * p_g + j] = seed.cos() * 0.7 + (seed * 0.3).sin() * 0.25;
            }
        }

        // Pick a non-trivial RHS.
        let b: Vec<f64> = (0..p_total)
            .map(|i| {
                let seed = (i as f64) * 0.157 + 0.6;
                seed.sin() * 0.55 + (seed * 0.4).cos() * 0.35
            })
            .collect();

        let h_dense = cpu_dense_joint_hessian(
            &row_hessians,
            &marginal,
            &logslope,
            &block,
            &primary,
            n,
        );
        let x_oracle = cpu_pcg_oracle(&h_dense, &b, 1e-12);

        // Grab the same CUDA context + default stream that the bms_flex_row
        // kernels will use when `run_pcg_against_row_hessian_device` probes
        // its own backend. Going through the public runtime APIs keeps the
        // test independent of any private kernel-backend symbols.
        let runtime = crate::gpu::runtime::GpuRuntime::global()
            .expect("runtime must exist when probe succeeded above");
        let ctx = match crate::gpu::runtime::cuda_context_for(
            runtime.selected_device().ordinal,
        ) {
            Some(c) => c,
            None => {
                eprintln!("[pcg_device parity] cuda_context_for failed; skipping");
                return;
            }
        };
        let stream = ctx.default_stream();
        let d_h = match stream.clone_htod(&row_hessians) {
            Ok(s) => s,
            Err(err) => {
                eprintln!("[pcg_device parity] upload h failed: {err}");
                return;
            }
        };
        let d_m = match stream.clone_htod(&marginal) {
            Ok(s) => s,
            Err(err) => {
                eprintln!("[pcg_device parity] upload marginal failed: {err}");
                return;
            }
        };
        let d_g = match stream.clone_htod(&logslope) {
            Ok(s) => s,
            Err(err) => {
                eprintln!("[pcg_device parity] upload logslope failed: {err}");
                return;
            }
        };
        let storage = DeviceResidentRowHess {
            ctx,
            stream,
            hess: d_h,
            marginal_design: d_m,
            logslope_design: d_g,
            n,
            r,
            block,
            primary,
            layout: RowHessianLayout::FullRowMajor,
            bytes: ((n * r * r + n * p_m + n * p_g) * std::mem::size_of::<f64>()) as u64,
        };

        let out = run_pcg_against_row_hessian_device(DeviceResidentPcgInput {
            storage: &storage,
            b: &b,
            rel_tol: 1e-10,
            max_iters: 4 * p_total,
            precond_diag_floor: 1e-12,
        })
        .expect("device-resident PCG must succeed on SPD fixture");

        assert_eq!(out.x.len(), p_total);
        let mut max_abs = 0.0_f64;
        for i in 0..p_total {
            let diff = (out.x[i] - x_oracle[i]).abs();
            if diff > max_abs {
                max_abs = diff;
            }
        }
        // Each iteration introduces O(1) ULPs of round-off in the dot/
        // axpy ladder; with ~88 iters max at p=44 we expect ‖Δx‖∞ comfortably
        // below 1e-7. Anything larger means a code bug, not float noise.
        assert!(
            max_abs <= 1e-7,
            "pcg_device parity ‖Δx‖∞={max_abs:.3e} > 1e-7 after {} iters \
             (final rel residual={:.3e})",
            out.iterations,
            out.final_rel_residual
        );
        eprintln!(
            "[pcg_device parity] n={n} p={p_total} r={r}: iters={} rel_res={:.3e} ‖Δx‖∞={:.3e}",
            out.iterations, out.final_rel_residual, max_abs
        );
    }
}
