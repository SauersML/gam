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
        PirlsGpuInput, PirlsGpuSharedData, PirlsGpuStep, PirlsStepStreamInput,
        SigmaPirlsGpuWorkspace,
    };
    use crate::gpu::driver::{from_col_major, to_col_major};
    use crate::gpu::solver::{
        cholesky_logdet_from_col_major, context_and_stream, pinned_htod, potrf_in_place,
        potrs_in_place,
    };
    use cudarc::cublas::sys::{
        cublasDdgmm, cublasDgeam, cublasOperation_t, cublasSideMode_t, cublasStatus_t,
    };
    use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
    use cudarc::cusolver::DnHandle;
    use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
    use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

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

        // Direction: the convention is d = −H⁻¹ g.
        let direction_raw = ws
            .stream
            .clone_dtoh(&ws.rhs_dev)
            .map_err(|e| format!("download direction: {e}"))?;
        let mut direction = Array1::from_vec(direction_raw);
        direction.mapv_inplace(|v| -v);

        let h_factor_col = ws
            .stream
            .clone_dtoh(&ws.h_dev)
            .map_err(|e| format!("download Cholesky factor: {e}"))?;
        let logdet = cholesky_logdet_from_col_major(&h_factor_col, p);

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
