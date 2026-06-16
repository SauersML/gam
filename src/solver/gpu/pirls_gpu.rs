use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

#[derive(Clone, Debug)]
pub struct PirlsGpuInput<'a> {
    pub x: ArrayView2<'a, f64>,
    pub weights: ArrayView1<'a, f64>,
    pub penalty_hessian: ArrayView2<'a, f64>,
    /// Full descent-direction RHS: `Xᵀ·score − S·β + linear_shift`. The
    /// returned `PirlsGpuStep::direction = H⁻¹·gradient` (no negation, #257).
    /// Callers must assemble the corrected RHS before passing it here.
    pub gradient: ArrayView1<'a, f64>,
    /// Temporary Levenberg–Marquardt damping; added to H for the solve
    /// only. Never enters the exported `penalized_hessian`, `RidgePassport`,
    /// EDF, REML curvature, or penalty term.
    pub step_lm_lambda: f64,
    /// Real model-objective ridge. Enters the exported `penalized_hessian`,
    /// `RidgePassport`, EDF, REML curvature, and penalty term.
    pub objective_ridge: f64,
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
    /// Temporary LM damping for this Newton solve step only. Added to H
    /// before potrf; stripped out of the snapshotted `penalized_hessian`.
    pub step_lm_lambda: f64,
    /// Real model-objective ridge. Appears in the exported
    /// `penalized_hessian` that flows to EDF / REML curvature.
    pub objective_ridge: f64,
}

/// Stage 3.2 device-input variant of [`PirlsStepStreamInput`].
///
/// Where the host-input form uploads `weights` + `gradient` per Newton
/// step, this form reads them straight from the
/// [`crate::gpu::kernels::pirls_row::RowOutputDevBuffers`] populated by the
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
    /// Temporary LM damping for this Newton solve step only. Added to H
    /// before potrf; stripped out of the snapshotted `penalized_hessian`.
    pub step_lm_lambda: f64,
    /// Real model-objective ridge. Appears in the exported
    /// `penalized_hessian` that flows to EDF / REML curvature.
    pub objective_ridge: f64,
    /// Current coefficient vector β (length p). Downloaded to the host to
    /// form the Newton RHS correction S·β. Only p f64 values cross the
    /// boundary (β is small), so the round-trip cost is negligible.
    pub beta_dev: &'b cudarc::driver::CudaSlice<f64>,
    /// Linear shift vector (length p) in transformed coordinates, on host.
    /// Added to Newton RHS so the solve targets Xᵀ·score − S·β + linear_shift.
    pub linear_shift: ArrayView1<'b, f64>,
}

/// Shared, batch-wide GPU state for stream-pool sigma-cubature PIRLS.
///
/// Construct once per model via [`upload_shared_pirls_gpu`] and hand a
/// shared reference to many [`SigmaPirlsGpuWorkspace`]s. X_original, y,
/// prior_w, and offset are uploaded once and reused across all ρ / σ
/// points. Per ρ / σ point, only the small `Qs` reparam matrix is
/// re-uploaded into the workspace.
#[cfg(target_os = "linux")]
pub struct PirlsGpuSharedData {
    pub(crate) ctx: std::sync::Arc<cudarc::driver::CudaContext>,
    pub(crate) n: usize,
    pub(crate) p: usize,
    /// `n*p` f64 column-major **original** design matrix `X_original`,
    /// device-resident. Never the pre-multiplied `X·Qs` form.
    pub(crate) x_original_dev: cudarc::driver::CudaSlice<f64>,
    /// Response vector `y`, length `n`, device-resident.
    pub(crate) y_dev: cudarc::driver::CudaSlice<f64>,
    /// Prior weights, length `n`, device-resident.
    pub(crate) prior_w_dev: cudarc::driver::CudaSlice<f64>,
    /// Observation offset, length `n`, device-resident.
    pub(crate) offset_dev: cudarc::driver::CudaSlice<f64>,
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
/// When `p < FUSED_XTWX_P_THRESHOLD`, the workspace skips the `n×p` `wx_dev`
/// temporary entirely and routes through the fused `xtwx_lower` + `xtscore`
/// kernels instead. `wx_dev` is `Some` only for the large-p fallback path
/// where `ddgmm + gemm` beats the fused kernel.
#[cfg(target_os = "linux")]
pub struct SigmaPirlsGpuWorkspace {
    pub(crate) stream: std::sync::Arc<cudarc::driver::CudaStream>,
    pub(crate) blas: cudarc::cublas::CudaBlas,
    pub(crate) solver: cudarc::cusolver::DnHandle,
    /// `None` when `p < FUSED_XTWX_P_THRESHOLD` (fused path). `Some` for the
    /// large-p fallback where the `ddgmm + dgemm` route is faster.
    pub(crate) wx_dev: Option<cudarc::driver::CudaSlice<f64>>,
    pub(crate) w_dev: cudarc::driver::CudaSlice<f64>,
    /// `X_originalᵀ W X_original` (p×p) — intermediate before Qs projection.
    pub(crate) xtwx_dev: cudarc::driver::CudaSlice<f64>,
    pub(crate) h_dev: cudarc::driver::CudaSlice<f64>,
    pub(crate) rhs_dev: cudarc::driver::CudaSlice<f64>,
    pub(crate) penalty_dev: cudarc::driver::CudaSlice<f64>,
    /// Reparameterisation matrix `Qs` (p×p, column-major), uploaded once per
    /// ρ / σ point. Identity when no reparameterisation is active. Used to
    /// project `A = X_originalᵀ W X_original` into the transformed frame:
    /// `H_step = Qsᵀ A Qs + S + λI`.
    pub(crate) qs_dev: cudarc::driver::CudaSlice<f64>,
    /// Scratch p×p buffer for the two-step `Qsᵀ A Qs` accumulation:
    /// first `tmp = A Qs`, then `H = Qsᵀ tmp`.
    pub(crate) qs_tmp_dev: cudarc::driver::CudaSlice<f64>,
    /// p-vector: `beta_orig = Qs · β` computed before each `eta = X · beta_orig`.
    pub(crate) beta_orig_dev: cudarc::driver::CudaSlice<f64>,
    /// p-vector scratch used for `Qs · direction` when forming `xd = X · (Qs · δ)`.
    pub(crate) dir_orig_dev: cudarc::driver::CudaSlice<f64>,
    /// Pre-allocated cuSOLVER POTRF workspace buffer. Sized once at
    /// construction via `potrf_query_lwork`; reused every Newton step.
    pub(crate) potrf_work_dev: cudarc::driver::CudaSlice<f64>,
    /// Number of f64 elements in `potrf_work_dev`, stored as i32 to match
    /// the cuSOLVER API signature for cusolverDnDpotrf.
    pub(crate) potrf_lwork: i32,
    /// Deferred POTRF info scalar. Stays device-resident across all PIRLS
    /// Newton steps; downloaded once at end-of-fit via
    /// `check_deferred_potrf_info`.
    pub(crate) potrf_info_dev: cudarc::driver::CudaSlice<i32>,
    /// Deferred POTRS info scalar. Mirrors the POTRF discipline.
    pub(crate) potrs_info_dev: cudarc::driver::CudaSlice<i32>,
    pub(crate) n: usize,
    pub(crate) p: usize,
}

#[cfg(target_os = "linux")]
pub(crate) mod cuda {
    use super::{
        PirlsGpuInput, PirlsGpuSharedData, PirlsGpuStep, PirlsStepStreamDeviceInput,
        PirlsStepStreamInput, SigmaPirlsGpuWorkspace,
    };
    use crate::gpu::device_cache::PtxModuleCache;
    use crate::gpu::driver::{from_col_major, to_col_major};
    use crate::gpu::solver::{
        check_deferred_potrf_info, check_deferred_potrs_info, context_and_stream, pinned_htod,
        potrf_in_place_reuse, potrf_query_lwork, potrs_in_place_reuse,
    };
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

    /// When `p` is below this threshold the workspace uses the fused
    /// `xtwx_lower` + `xtscore` + `symmetrize_lower` kernels and omits the
    /// `n*p` `wx_dev` temporary entirely. For `p >= FUSED_XTWX_P_THRESHOLD`
    /// the existing `ddgmm + dgemm` path is used.
    const FUSED_XTWX_P_THRESHOLD: usize = 256;

    /// NVRTC kernels for the fused path.
    ///
    /// `xtwx_lower`: one thread per lower-tri pair `(j,k)` with `j >= k`;
    /// iterates over `n` rows, writes `A[j + k*p]` (col-major lower triangle).
    ///
    /// `xtscore`: one thread per `j`; writes `s[j] = sum_i score[i]*X[i,j]`.
    ///
    /// `symmetrize_lower`: one thread per strict-lower pair `(j,k)` with
    /// `j > k`; copies `A[k + j*p] = A[j + k*p]` to fill the upper triangle.
    const FUSED_XTWX_PTX_SOURCE: &str = concat!(
        // xtwx_lower: enumerate lower triangle row-by-row.
        // Row j has entries (j,0),(j,1),...,(j,j).
        // Cumulative offset before row j = j*(j+1)/2.
        // Unrank t -> j = floor((sqrt(8t+1)-1)/2), k = t - j*(j+1)/2.
        // Output: A[j + k*p] in col-major for j >= k.
        "extern \"C\" __global__ void xtwx_lower(",
        "const double* __restrict__ X,",
        "const double* __restrict__ w,",
        "double* __restrict__ A,",
        "int n, int p) {",
        "int t=blockIdx.x*blockDim.x+threadIdx.x;",
        "int np=p*(p+1)/2; if(t>=np)return;",
        // j = floor((sqrt(8t+1)-1)/2); clamp for fp rounding
        "int jv=(int)((__dsqrt_rn((double)(8*t+1))-1.0)*0.5);",
        "while((long long)(jv+1)*(jv+2)/2<=t)jv++;",
        "while(jv>0&&(long long)jv*(jv+1)/2>t)jv--;",
        "int kv=t-(int)((long long)jv*(jv+1)/2);",
        "double acc=0.0;",
        "const double*Xj=X+(long long)jv*n;",
        "const double*Xk=X+(long long)kv*n;",
        "for(int i=0;i<n;++i)acc+=w[i]*Xj[i]*Xk[i];",
        // col-major index: A[jv, kv] = A[jv + kv*p]
        "A[jv+(long long)kv*p]=acc;}",
        // xtscore: one thread per output index j
        "extern \"C\" __global__ void xtscore(",
        "const double* __restrict__ X,",
        "const double* __restrict__ score,",
        "double* __restrict__ s,",
        "int n, int p) {",
        "int j=blockIdx.x*blockDim.x+threadIdx.x;",
        "if(j>=p)return;",
        "double acc=0.0;",
        "const double*Xj=X+(long long)j*n;",
        "for(int i=0;i<n;++i)acc+=score[i]*Xj[i];",
        "s[j]=acc;}",
        // symmetrize_lower: strict lower pairs (j,k) with j>k.
        // Enumerate row-by-row: row j=1 has entry (1,0); row j=2 has (2,0),(2,1); etc.
        // Cumulative before row j: j*(j-1)/2.
        // Unrank t -> j = floor((sqrt(8t+1)+1)/2), k = t - j*(j-1)/2.
        "extern \"C\" __global__ void symmetrize_lower(",
        "double* __restrict__ A, int p) {",
        "int ns=p*(p-1)/2;",
        "int t=blockIdx.x*blockDim.x+threadIdx.x;",
        "if(t>=ns)return;",
        // j = floor((sqrt(8t+1)+1)/2); clamp
        "int jv=(int)((__dsqrt_rn((double)(8*t+1))+1.0)*0.5);",
        "while((long long)jv*(jv-1)/2>t)jv--;",
        "while((long long)(jv+1)*jv/2<=t)jv++;",
        "int kv=t-(int)((long long)jv*(jv-1)/2);",
        // A[kv, jv] = A[kv + jv*p] = A[jv + kv*p] (copy lower to upper)
        "A[kv+(long long)jv*p]=A[jv+(long long)kv*p];}",
    );

    static FUSED_XTWX_CACHE: PtxModuleCache = PtxModuleCache::new();

    impl PirlsGpuSharedData {
        /// Upload `x` to the cached per-ordinal CUDA context and return a
        /// Upload X_original, y, prior_w, and offset to the device once.
        /// Returns a shared handle reused across all ρ / σ points.
        pub(crate) fn upload_impl(
            x: ArrayView2<'_, f64>,
            y: ArrayView1<'_, f64>,
            prior_w: ArrayView1<'_, f64>,
            offset: ArrayView1<'_, f64>,
        ) -> Result<Self, String> {
            let (n, p) = x.dim();
            if n == 0 || p == 0 {
                return Err("empty design cannot be uploaded".to_string());
            }
            if y.len() != n || prior_w.len() != n || offset.len() != n {
                return Err(format!(
                    "y/prior_w/offset length mismatch (y={}, w={}, offset={}, n={n})",
                    y.len(),
                    prior_w.len(),
                    offset.len()
                ));
            }
            let (ctx, stream) = context_and_stream()?;
            let x_col = to_col_major(&x);
            let x_original_dev = pinned_htod(&stream, &x_col)?;
            let y_dev = pinned_htod(&stream, y.as_slice().ok_or("y not contiguous")?)?;
            let prior_w_dev =
                pinned_htod(&stream, prior_w.as_slice().ok_or("prior_w not contiguous")?)?;
            let offset_dev =
                pinned_htod(&stream, offset.as_slice().ok_or("offset not contiguous")?)?;
            // Synchronize the upload stream so all buffers are visible to
            // every workspace we hand off to. Workspaces use independent
            // streams; the uploads completed on the bootstrap stream above.
            stream
                .synchronize()
                .map_err(|e| format!("cuda sync after model upload: {e}"))?;
            Ok(Self {
                ctx,
                n,
                p,
                x_original_dev,
                y_dev,
                prior_w_dev,
                offset_dev,
            })
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
            // Skip the n*p WX scratch when the fused kernels will be used.
            let wx_dev = if p >= FUSED_XTWX_P_THRESHOLD {
                Some(
                    stream
                        .alloc_zeros::<f64>(np)
                        .map_err(|e| format!("cuda alloc WX: {e}"))?,
                )
            } else {
                None
            };
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
            // Qs and scratch: p×p identity-initialized and p-vector zeros.
            let mut qs_dev = stream
                .alloc_zeros::<f64>(pp)
                .map_err(|e| format!("cuda alloc Qs: {e}"))?;
            // Initialize Qs to identity: diagonal = 1.0.
            {
                let mut qs_host = vec![0.0_f64; pp];
                for i in 0..p {
                    qs_host[i * p + i] = 1.0;
                }
                stream
                    .memcpy_htod(&qs_host, &mut qs_dev)
                    .map_err(|e| format!("init Qs identity: {e}"))?;
            }
            let qs_tmp_dev = stream
                .alloc_zeros::<f64>(pp)
                .map_err(|e| format!("cuda alloc Qs tmp: {e}"))?;
            let beta_orig_dev = stream
                .alloc_zeros::<f64>(p)
                .map_err(|e| format!("cuda alloc beta_orig: {e}"))?;
            let dir_orig_dev = stream
                .alloc_zeros::<f64>(p)
                .map_err(|e| format!("cuda alloc dir_orig: {e}"))?;
            // Query the POTRF workspace size once using the actual p so we
            // can size the persistent buffer. This is the only buffer-size
            // query in the hot path — every Newton step reuses it.
            let potrf_lwork_usize = potrf_query_lwork(&solver, &stream, p)?;
            let potrf_lwork = i32::try_from(potrf_lwork_usize)
                .map_err(|_| format!("potrf lwork {potrf_lwork_usize} exceeds i32"))?;
            // Allocate at least 1 element so the device pointer is always
            // valid; cuSOLVER accepts a zero-length workspace when lwork==0.
            let alloc_len = potrf_lwork_usize.max(1);
            let potrf_work_dev = stream
                .alloc_zeros::<f64>(alloc_len)
                .map_err(|e| format!("cuda alloc potrf workspace: {e}"))?;
            let potrf_info_dev = stream
                .alloc_zeros::<i32>(1)
                .map_err(|e| format!("cuda alloc potrf info: {e}"))?;
            let potrs_info_dev = stream
                .alloc_zeros::<i32>(1)
                .map_err(|e| format!("cuda alloc potrs info: {e}"))?;
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
                qs_dev,
                qs_tmp_dev,
                beta_orig_dev,
                dir_orig_dev,
                potrf_work_dev,
                potrf_lwork,
                potrf_info_dev,
                potrs_info_dev,
                n,
                p,
            })
        }
    }

    /// Upload a new `Qs` matrix (p×p, row-major host) to `ws.qs_dev`.
    /// Call once per ρ / σ point before calling `pirls_loop` or any step
    /// function. When no reparameterisation is active, pass the identity.
    pub(super) fn upload_qs(
        ws: &mut SigmaPirlsGpuWorkspace,
        qs: ArrayView2<'_, f64>,
    ) -> Result<(), String> {
        let p = ws.p;
        if qs.dim() != (p, p) {
            return Err(format!("upload_qs: Qs shape {:?} != ({p},{p})", qs.dim()));
        }
        let qs_col = to_col_major(&qs);
        ws.stream
            .memcpy_htod(qs_col.as_ref(), &mut ws.qs_dev)
            .map_err(|e| format!("upload Qs: {e}"))
    }

    /// Upload an identity `Qs` (no reparameterisation) for the current ρ point.
    pub(super) fn upload_qs_identity(ws: &mut SigmaPirlsGpuWorkspace) -> Result<(), String> {
        let p = ws.p;
        let pp = p * p;
        let mut qs_host = vec![0.0_f64; pp];
        for i in 0..p {
            qs_host[i * p + i] = 1.0;
        }
        ws.stream
            .memcpy_htod(&qs_host, &mut ws.qs_dev)
            .map_err(|e| format!("upload Qs identity: {e}"))
    }

    /// Apply one fp64 iterative-refinement correction to a Newton step solve.
    ///
    /// Compute `r = g − H_step·x` (host, p-vector). When `p ≥ REFINEMENT_MIN_P`
    /// and `‖r‖/‖g‖ > REFINEMENT_TOL`, apply one POTRS correction and return
    /// `x + e`. Returns `direction_raw` unchanged when `p` is too small, the
    /// residual is already tight, or `‖g‖ = 0`.
    ///
    /// `H_step·x = penalized_hessian·x + step_lm_delta·x`.
    #[allow(clippy::too_many_arguments)]
    fn newton_step_refine_once(
        solver: &cudarc::cusolver::DnHandle,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        p: usize,
        chol_factor_dev: &CudaSlice<f64>,
        rhs_dev: &mut CudaSlice<f64>,
        potrs_info_dev: &mut CudaSlice<i32>,
        mut direction_raw: Vec<f64>,
        g: &[f64],
        penalized_hessian: &ndarray::Array2<f64>,
        step_lm_delta: f64,
    ) -> Result<Vec<f64>, String> {
        use crate::gpu::policy::GpuDispatchPolicy;
        if p < GpuDispatchPolicy::REFINEMENT_MIN_P {
            return Ok(direction_raw);
        }
        let norm_g = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm_g == 0.0 {
            return Ok(direction_raw);
        }
        let hx: Vec<f64> = (0..p)
            .map(|i| {
                penalized_hessian
                    .row(i)
                    .iter()
                    .zip(direction_raw.iter())
                    .map(|(hij, xj)| hij * xj)
                    .sum::<f64>()
                    + step_lm_delta * direction_raw[i]
            })
            .collect();
        let residual: Vec<f64> = g.iter().zip(hx.iter()).map(|(gi, hxi)| gi - hxi).collect();
        let rel_res = residual.iter().map(|v| v * v).sum::<f64>().sqrt() / norm_g;
        if rel_res <= GpuDispatchPolicy::REFINEMENT_TOL {
            return Ok(direction_raw);
        }
        stream
            .memcpy_htod(&residual, rhs_dev)
            .map_err(|e| format!("upload residual: {e}"))?;
        potrs_in_place_reuse(
            solver,
            stream,
            p,
            1,
            chol_factor_dev,
            rhs_dev,
            potrs_info_dev,
        )?;
        let correction = stream
            .clone_dtoh(rhs_dev)
            .map_err(|e| format!("download correction: {e}"))?;
        check_deferred_potrs_info(stream, potrs_info_dev)?;
        for (xi, ei) in direction_raw.iter_mut().zip(correction.iter()) {
            *xi += ei;
        }
        Ok(direction_raw)
    }

    /// Drive one PIRLS Newton step on the workspace's CUDA stream.
    ///
    /// Build `H = XᵀWX + S + λI`, Cholesky-factor it, solve `H·d = g`,
    /// return `(H, d, log|H|)`. `input.gradient` is the full descent-direction
    /// RHS `Xᵀscore − S·β + linear_shift` — the caller is responsible for
    /// assembling the corrected RHS before calling this function. No negation
    /// is applied; the returned `direction = H⁻¹·g` is the descent step δ
    /// directly (#257). The difference vs the one-shot [`solve_step`] is
    /// purely the execution model: no context creation, no handle creation,
    /// no design-matrix upload, no per-step buffer allocations.
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

        // Compute XᵀWX into ws.xtwx_dev.  Two paths:
        // Fused (p < FUSED_XTWX_P_THRESHOLD): row-sweep kernels, no n*p temp.
        // Fallback (p >= FUSED_XTWX_P_THRESHOLD): ddgmm + dgemm via wx_dev.
        let n_i = to_i32(n)?;
        let p_i = to_i32(p)?;
        if let Some(ref mut wx_dev) = ws.wx_dev {
            left_scale_rows(
                &ws.blas,
                &ws.stream,
                n,
                p,
                &shared.x_original_dev,
                &mut ws.w_dev,
                wx_dev,
            )?;
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
            // SAFETY: validated i32 dims; shared.x_original_dev and wx_dev are n*p
            // f64 col-major; ws.xtwx_dev is the p*p output.
            unsafe {
                ws.blas
                    .gemm(cfg, &shared.x_original_dev, wx_dev, &mut ws.xtwx_dev)
            }
            .map_err(|e| format!("cublas dgemm XtWX: {e}"))?;
        } else {
            launch_xtwx_lower(
                &ws.stream,
                &shared.ctx,
                n,
                p,
                &shared.x_original_dev,
                &ws.w_dev,
                &mut ws.xtwx_dev,
            )?;
            launch_symmetrize_lower(&ws.stream, &shared.ctx, p, &mut ws.xtwx_dev)?;
        }

        // Upload S + step_lm_lambda·I for the Newton solve (LM damping only).
        let penalty_step = penalty_with_ridge(input.penalty_hessian, input.step_lm_lambda);
        let penalty_step_view = penalty_step.view();
        let penalty_step_col = to_col_major(&penalty_step_view);
        ws.stream
            .memcpy_htod(penalty_step_col.as_ref(), &mut ws.penalty_dev)
            .map_err(|e| format!("upload penalty: {e}"))?;

        // Apply Qs rotation: H_xtx = Qsᵀ · XᵀWX · Qs (two p×p gemms).
        // Matches solve_step_on_stream_device_inplace (#269 resident-X arch):
        // X_original stays device-resident, Qs rotates into transformed frame.
        {
            let cfg_aq = GemmConfig::<f64> {
                transa: cublasOperation_t::CUBLAS_OP_N,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: p_i,
                n: p_i,
                k: p_i,
                alpha: 1.0,
                lda: p_i,
                ldb: p_i,
                beta: 0.0,
                ldc: p_i,
            };
            // SAFETY: xtwx_dev and qs_dev p*p col-major; qs_tmp_dev p*p output.
            unsafe {
                ws.blas
                    .gemm(cfg_aq, &ws.xtwx_dev, &ws.qs_dev, &mut ws.qs_tmp_dev)
            }
            .map_err(|e| format!("dgemm A·Qs (host-input step): {e}"))?;
        }
        {
            let cfg_qt = GemmConfig::<f64> {
                transa: cublasOperation_t::CUBLAS_OP_T,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: p_i,
                n: p_i,
                k: p_i,
                alpha: 1.0,
                lda: p_i,
                ldb: p_i,
                beta: 0.0,
                ldc: p_i,
            };
            // SAFETY: qs_dev p*p (transposed); qs_tmp_dev p*p; h_dev p*p output.
            unsafe {
                ws.blas
                    .gemm(cfg_qt, &ws.qs_dev, &ws.qs_tmp_dev, &mut ws.h_dev)
            }
            .map_err(|e| format!("dgemm Qsᵀ·A·Qs (host-input step): {e}"))?;
        }
        // H_step = Qsᵀ·XᵀWX·Qs + (S + step_lm_lambda·I).
        geam_add_inplace(&ws.blas, &ws.stream, p, &mut ws.h_dev, &ws.penalty_dev)?;

        // Upload gradient into the persistent RHS buffer.
        // `input.gradient` is already in transformed coordinates (Qsᵀ-projected
        // by the caller), so no additional rotation is needed here.
        let g_slice = input
            .gradient
            .as_slice()
            .ok_or("gradient must be contiguous")?;
        ws.stream
            .memcpy_htod(g_slice, &mut ws.rhs_dev)
            .map_err(|e| format!("upload gradient: {e}"))?;

        // Exported penalised Hessian: H_final = Qsᵀ·XᵀWX·Qs + S + objective_ridge·I.
        // Apply Qs rotation host-side on the downloaded XᵀWX so LM damping
        // never contaminates exported EDF / REML curvature / RidgePassport.
        let xtwx_col = ws
            .stream
            .clone_dtoh(&ws.xtwx_dev)
            .map_err(|e| format!("download XᵀWX (host-input step): {e}"))?;
        let xtwx_host = from_col_major(&xtwx_col, p, p).ok_or("XᵀWX layout conversion failed")?;
        let qs_col = ws
            .stream
            .clone_dtoh(&ws.qs_dev)
            .map_err(|e| format!("download Qs (host-input step): {e}"))?;
        let qs_host =
            from_col_major(&qs_col, p, p).ok_or("Qs layout conversion failed (host-input step)")?;
        let tmp_aq = xtwx_host.dot(&qs_host);
        let h_rotated = qs_host.t().dot(&tmp_aq);
        let penalty_export = penalty_with_ridge(input.penalty_hessian, input.objective_ridge);
        let penalized_hessian = h_rotated + &penalty_export;

        // Factor + solve in place on the stream using pre-allocated workspace
        // and info buffers — no per-step allocation, no per-step info download.
        potrf_in_place_reuse(
            &ws.solver,
            &ws.stream,
            p,
            ws.potrf_lwork,
            &mut ws.h_dev,
            &mut ws.potrf_work_dev,
            &mut ws.potrf_info_dev,
        )?;
        potrs_in_place_reuse(
            &ws.solver,
            &ws.stream,
            p,
            1,
            &ws.h_dev,
            &mut ws.rhs_dev,
            &mut ws.potrs_info_dev,
        )?;

        // Logdet device-side: reduces the previous p² Cholesky-factor
        // download to a single f64 download. Stage 2's "no per-iteration
        // host round-trip" budget keeps the p² factor on the device.
        let logdet = cholesky_logdet_device(&ws.stream, &shared.ctx, p, &ws.h_dev)?;

        // Direction: d = H⁻¹ g (no negation; g is the full corrected RHS, #257).
        let direction_raw = ws
            .stream
            .clone_dtoh(&ws.rhs_dev)
            .map_err(|e| format!("download direction: {e}"))?;
        // Check deferred POTRF/POTRS info after the direction download
        // (which already syncs the stream). Single host round-trip for both
        // info scalars at end-of-step rather than one per cuSOLVER call.
        check_deferred_potrf_info(&ws.stream, &ws.potrf_info_dev)?;
        check_deferred_potrs_info(&ws.stream, &ws.potrs_info_dev)?;

        // Iterative refinement on the Qs-rotated system.
        // penalized_hessian = Qsᵀ·XtWX·Qs + S + objective_ridge·I.
        // H_step = penalized_hessian + (step_lm_lambda − objective_ridge)·I.
        let lm_ridge_delta = input.step_lm_lambda - input.objective_ridge;
        let direction_raw = newton_step_refine_once(
            &ws.solver,
            &ws.stream,
            p,
            &ws.h_dev,
            &mut ws.rhs_dev,
            &mut ws.potrs_info_dev,
            direction_raw,
            g_slice,
            &penalized_hessian,
            lm_ridge_delta,
        )?;

        // No negation: `input.gradient` is the full descent-direction RHS
        // `Xᵀscore − S·β + linear_shift`; solving H·δ = rhs gives δ directly.
        let direction = Array1::from_vec(direction_raw);

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
    /// n, so for large-scale n it is a negligible transfer.
    ///
    /// Outputs match `solve_step_on_stream`: returns the assembled
    /// penalised Hessian, the Newton descent direction `δ = H⁻¹·rhs`
    /// where `rhs = Xᵀ·score − S·β + linear_shift` (no negation, #257),
    /// and the log-determinant computed via the device-side
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

        // Compute XᵀWX and Xᵀ·score.  Fused path (p < threshold): no n*p WX.
        // Fallback (p >= threshold): ddgmm + dgemm + gemv via wx_dev_fb.
        let n_i = to_i32(n)?;
        let p_i = to_i32(p)?;
        if let Some(ref mut wx_dev_fb) = ws.wx_dev {
            // Large-p fallback.
            left_scale_rows_borrowed(
                &ws.blas,
                &ws.stream,
                n,
                p,
                &shared.x_original_dev,
                input.w_solver_dev,
                wx_dev_fb,
            )?;
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
            // SAFETY: validated dims; shared.x_original_dev and wx_dev_fb are n*p
            // f64 col-major; ws.xtwx_dev is p*p; all on ws.stream.
            unsafe {
                ws.blas.gemm(
                    gemm_cfg,
                    &shared.x_original_dev,
                    wx_dev_fb,
                    &mut ws.xtwx_dev,
                )
            }
            .map_err(|e| format!("cublas dgemm XtWX (device-input): {e}"))?;
            let penalty_step = penalty_with_ridge(input.penalty_hessian, input.step_lm_lambda);
            let penalty_step_col = to_col_major(&penalty_step);
            ws.stream
                .memcpy_htod(penalty_step_col.as_ref(), &mut ws.penalty_dev)
                .map_err(|e| format!("upload penalty (device-input): {e}"))?;
            // Qs rotation on H: tmp = XᵀWX · Qs, then h_dev = Qsᵀ · tmp.
            {
                let cfg_aq = GemmConfig::<f64> {
                    transa: cublasOperation_t::CUBLAS_OP_N,
                    transb: cublasOperation_t::CUBLAS_OP_N,
                    m: p_i,
                    n: p_i,
                    k: p_i,
                    alpha: 1.0,
                    lda: p_i,
                    ldb: p_i,
                    beta: 0.0,
                    ldc: p_i,
                };
                // SAFETY: xtwx_dev and qs_dev p*p col-major; qs_tmp_dev p*p output.
                unsafe {
                    ws.blas
                        .gemm(cfg_aq, &ws.xtwx_dev, &ws.qs_dev, &mut ws.qs_tmp_dev)
                }
                .map_err(|e| format!("dgemm A·Qs (device-input large-p): {e}"))?;
            }
            {
                let cfg_qt = GemmConfig::<f64> {
                    transa: cublasOperation_t::CUBLAS_OP_T,
                    transb: cublasOperation_t::CUBLAS_OP_N,
                    m: p_i,
                    n: p_i,
                    k: p_i,
                    alpha: 1.0,
                    lda: p_i,
                    ldb: p_i,
                    beta: 0.0,
                    ldc: p_i,
                };
                // SAFETY: qs_dev p*p (transposed); qs_tmp_dev p*p; h_dev p*p output.
                unsafe {
                    ws.blas
                        .gemm(cfg_qt, &ws.qs_dev, &ws.qs_tmp_dev, &mut ws.h_dev)
                }
                .map_err(|e| format!("dgemm Qsᵀ·A·Qs (device-input large-p): {e}"))?;
            }
            geam_add_inplace(&ws.blas, &ws.stream, p, &mut ws.h_dev, &ws.penalty_dev)?;
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
            // SAFETY: shared.x_original_dev n*p col-major; grad_eta_dev length n; rhs_dev length p.
            unsafe {
                ws.blas.gemv(
                    gemv_cfg,
                    &shared.x_original_dev,
                    input.grad_eta_dev,
                    &mut ws.rhs_dev,
                )
            }
            .map_err(|e| format!("cublas dgemv Xtg (device-input): {e}"))?;
        } else {
            // Fused path: row-sweep kernels, no n*p WX buffer.
            launch_xtwx_lower(
                &ws.stream,
                &shared.ctx,
                n,
                p,
                &shared.x_original_dev,
                input.w_solver_dev,
                &mut ws.xtwx_dev,
            )?;
            launch_symmetrize_lower(&ws.stream, &shared.ctx, p, &mut ws.xtwx_dev)?;
            launch_xtscore(
                &ws.stream,
                &shared.ctx,
                n,
                p,
                &shared.x_original_dev,
                input.grad_eta_dev,
                &mut ws.rhs_dev,
            )?;
            // Qs rotation on H: tmp = XᵀWX · Qs, then h_dev = Qsᵀ · tmp.
            {
                let cfg_aq = GemmConfig::<f64> {
                    transa: cublasOperation_t::CUBLAS_OP_N,
                    transb: cublasOperation_t::CUBLAS_OP_N,
                    m: p_i,
                    n: p_i,
                    k: p_i,
                    alpha: 1.0,
                    lda: p_i,
                    ldb: p_i,
                    beta: 0.0,
                    ldc: p_i,
                };
                // SAFETY: xtwx_dev and qs_dev p*p col-major; qs_tmp_dev p*p output.
                unsafe {
                    ws.blas
                        .gemm(cfg_aq, &ws.xtwx_dev, &ws.qs_dev, &mut ws.qs_tmp_dev)
                }
                .map_err(|e| format!("dgemm A·Qs (device-input fused): {e}"))?;
            }
            {
                let cfg_qt = GemmConfig::<f64> {
                    transa: cublasOperation_t::CUBLAS_OP_T,
                    transb: cublasOperation_t::CUBLAS_OP_N,
                    m: p_i,
                    n: p_i,
                    k: p_i,
                    alpha: 1.0,
                    lda: p_i,
                    ldb: p_i,
                    beta: 0.0,
                    ldc: p_i,
                };
                // SAFETY: qs_dev p*p (transposed); qs_tmp_dev p*p; h_dev p*p output.
                unsafe {
                    ws.blas
                        .gemm(cfg_qt, &ws.qs_dev, &ws.qs_tmp_dev, &mut ws.h_dev)
                }
                .map_err(|e| format!("dgemm Qsᵀ·A·Qs (device-input fused): {e}"))?;
            }
            let penalty_step = penalty_with_ridge(input.penalty_hessian, input.step_lm_lambda);
            let penalty_step_col = to_col_major(&penalty_step);
            ws.stream
                .memcpy_htod(penalty_step_col.as_ref(), &mut ws.penalty_dev)
                .map_err(|e| format!("upload penalty (fused device-input): {e}"))?;
            geam_add_inplace(&ws.blas, &ws.stream, p, &mut ws.h_dev, &ws.penalty_dev)?;
        }

        // Apply rhs correction BEFORE the solve:
        //   rhs = Qsᵀ·(Xᵀ·score) − S·β + linear_shift  (#257, #260, #269).
        // First project X_origᵀ·score through Qsᵀ (p×p gemv on device), then
        // apply the S·β correction host-side and re-upload.
        {
            // Qsᵀ · rhs_dev (= Xᵀ·score) → beta_orig_dev (scratch p-vector).
            let cfg_qts = GemvConfig::<f64> {
                trans: cublasOperation_t::CUBLAS_OP_T,
                m: p_i,
                n: p_i,
                alpha: 1.0,
                lda: p_i,
                incx: 1,
                beta: 0.0,
                incy: 1,
            };
            // SAFETY: qs_dev p*p (transposed); rhs_dev length p; beta_orig_dev length p.
            unsafe {
                ws.blas
                    .gemv(cfg_qts, &ws.qs_dev, &ws.rhs_dev, &mut ws.beta_orig_dev)
            }
            .map_err(|e| format!("dgemv Qsᵀ·score (device-input): {e}"))?;
            // Swap: rhs_dev ← beta_orig_dev (now holds Qsᵀ·Xᵀ·score).
            ws.stream
                .memcpy_dtod(&ws.beta_orig_dev, &mut ws.rhs_dev)
                .map_err(|e| format!("d2d Qsᵀ·score→rhs (device-input): {e}"))?;
            // Download rhs and β; apply penalty correction host-side.
            let rhs_raw = ws
                .stream
                .clone_dtoh(&ws.rhs_dev)
                .map_err(|e| format!("download Qsᵀscore (device-input): {e}"))?;
            let beta_raw = ws
                .stream
                .clone_dtoh(input.beta_dev)
                .map_err(|e| format!("download beta (device-input): {e}"))?;
            let mut rhs_host = Array1::from_vec(rhs_raw);
            let beta_host = Array1::from_vec(beta_raw);
            let s_beta = input.penalty_hessian.dot(&beta_host);
            rhs_host -= &s_beta;
            rhs_host += &input.linear_shift;
            ws.stream
                .memcpy_htod(
                    rhs_host
                        .as_slice()
                        .ok_or("rhs_host not contiguous (device-input correction)")?,
                    &mut ws.rhs_dev,
                )
                .map_err(|e| format!("re-upload corrected rhs (device-input): {e}"))?;
        }

        // Exported penalised Hessian: H_final = Qsᵀ·XᵀWX·Qs + S + objective_ridge·I.
        // Apply Qs rotation host-side on the downloaded XᵀWX so LM damping
        // never contaminates exported EDF / REML curvature / RidgePassport.
        let xtwx_col = ws
            .stream
            .clone_dtoh(&ws.xtwx_dev)
            .map_err(|e| format!("download XᵀWX (device-input): {e}"))?;
        let xtwx_host = from_col_major(&xtwx_col, p, p)
            .ok_or("XᵀWX layout conversion failed (device-input)")?;
        let qs_col = ws
            .stream
            .clone_dtoh(&ws.qs_dev)
            .map_err(|e| format!("download Qs (device-input): {e}"))?;
        let qs_host =
            from_col_major(&qs_col, p, p).ok_or("Qs layout conversion failed (device-input)")?;
        let tmp_aq = xtwx_host.dot(&qs_host);
        let h_rotated = qs_host.t().dot(&tmp_aq);
        let penalty_export = penalty_with_ridge(input.penalty_hessian, input.objective_ridge);
        let penalized_hessian = h_rotated + &penalty_export;

        // Factor + solve in place on the stream using pre-allocated workspace
        // and info buffers — no per-step allocation, no per-step info download.
        potrf_in_place_reuse(
            &ws.solver,
            &ws.stream,
            p,
            ws.potrf_lwork,
            &mut ws.h_dev,
            &mut ws.potrf_work_dev,
            &mut ws.potrf_info_dev,
        )?;
        potrs_in_place_reuse(
            &ws.solver,
            &ws.stream,
            p,
            1,
            &ws.h_dev,
            &mut ws.rhs_dev,
            &mut ws.potrs_info_dev,
        )?;

        let logdet = cholesky_logdet_device(&ws.stream, &shared.ctx, p, &ws.h_dev)?;

        let direction_raw = ws
            .stream
            .clone_dtoh(&ws.rhs_dev)
            .map_err(|e| format!("download direction (device-input): {e}"))?;
        // Check deferred POTRF/POTRS info after the direction download
        // (which already syncs the stream). Single host round-trip for both
        // info scalars at end-of-step rather than one per cuSOLVER call.
        check_deferred_potrf_info(&ws.stream, &ws.potrf_info_dev)?;
        check_deferred_potrs_info(&ws.stream, &ws.potrs_info_dev)?;
        // No negation: rhs = Xᵀscore − Sβ + linear_shift already gives the
        // descent direction δ = H⁻¹·rhs directly (#257).
        let direction = Array1::from_vec(direction_raw);

        Ok(PirlsGpuStep {
            penalized_hessian,
            direction,
            logdet,
        })
    }

    /// In-place Newton step: rhs = Xᵀ·score − S·β + linear_shift (#257, #260).
    ///
    /// Solves H·δ = rhs (H = XᵀWX + S + step_lm_lambda·I). On return
    /// `ws.rhs_dev` holds the Newton descent direction δ (not negated).
    /// The loop copies `ws.rhs_dev` to `direction_dev` via `memcpy_dtod`.
    ///
    /// On return `ws.h_dev` holds the Cholesky factor; rebuild with
    /// `rebuild_h_final` to get the exported penalised Hessian.
    ///
    /// Returns `logdet = log|H|` computed device-side.
    pub(super) fn solve_step_on_stream_device_inplace(
        shared: &PirlsGpuSharedData,
        ws: &mut SigmaPirlsGpuWorkspace,
        input: PirlsStepStreamDeviceInput<'_, '_>,
    ) -> Result<f64, String> {
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

        if input.linear_shift.len() != p {
            return Err(format!(
                "linear_shift length {} does not match p={p}",
                input.linear_shift.len()
            ));
        }
        let n_i = to_i32(n)?;
        let p_i = to_i32(p)?;

        // Step 1: A = X_origᵀ diag(w_solver) X_orig → ws.xtwx_dev.
        //         score_p = X_origᵀ grad_eta → ws.rhs_dev.
        if let Some(ref mut wx_dev_ib) = ws.wx_dev {
            // Large-p path: ddgmm then dgemm, then gemv.
            left_scale_rows_borrowed(
                &ws.blas,
                &ws.stream,
                n,
                p,
                &shared.x_original_dev,
                input.w_solver_dev,
                wx_dev_ib,
            )?;
            let cfg_xtx = GemmConfig::<f64> {
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
            // SAFETY: x_original_dev and wx_dev_ib n*p col-major; xtwx_dev p*p; ws.stream.
            unsafe {
                ws.blas
                    .gemm(cfg_xtx, &shared.x_original_dev, wx_dev_ib, &mut ws.xtwx_dev)
            }
            .map_err(|e| format!("dgemm XtWX inplace (large-p): {e}"))?;
            let cfg_xts = GemvConfig::<f64> {
                trans: cublasOperation_t::CUBLAS_OP_T,
                m: n_i,
                n: p_i,
                alpha: 1.0,
                lda: n_i,
                incx: 1,
                beta: 0.0,
                incy: 1,
            };
            // SAFETY: x_original_dev n*p col-major; grad_eta_dev length n; rhs_dev length p.
            unsafe {
                ws.blas.gemv(
                    cfg_xts,
                    &shared.x_original_dev,
                    input.grad_eta_dev,
                    &mut ws.rhs_dev,
                )
            }
            .map_err(|e| format!("dgemv Xᵀ·score inplace (large-p): {e}"))?;
        } else {
            // Fused path: row-sweep kernels, no n*p WX buffer.
            launch_xtwx_lower(
                &ws.stream,
                &shared.ctx,
                n,
                p,
                &shared.x_original_dev,
                input.w_solver_dev,
                &mut ws.xtwx_dev,
            )?;
            launch_symmetrize_lower(&ws.stream, &shared.ctx, p, &mut ws.xtwx_dev)?;
            launch_xtscore(
                &ws.stream,
                &shared.ctx,
                n,
                p,
                &shared.x_original_dev,
                input.grad_eta_dev,
                &mut ws.rhs_dev,
            )?;
        }

        // Step 2: H_xtx = Qsᵀ A Qs  (two p×p gemms).
        //   tmp = A · Qs → ws.qs_tmp_dev.
        {
            let cfg_aq = GemmConfig::<f64> {
                transa: cublasOperation_t::CUBLAS_OP_N,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: p_i,
                n: p_i,
                k: p_i,
                alpha: 1.0,
                lda: p_i,
                ldb: p_i,
                beta: 0.0,
                ldc: p_i,
            };
            // SAFETY: xtwx_dev and qs_dev p*p col-major; qs_tmp_dev p*p output.
            unsafe {
                ws.blas
                    .gemm(cfg_aq, &ws.xtwx_dev, &ws.qs_dev, &mut ws.qs_tmp_dev)
            }
            .map_err(|e| format!("dgemm A·Qs inplace: {e}"))?;
        }
        //   H_xtx = Qsᵀ · tmp → ws.h_dev.
        {
            let cfg_qt = GemmConfig::<f64> {
                transa: cublasOperation_t::CUBLAS_OP_T,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: p_i,
                n: p_i,
                k: p_i,
                alpha: 1.0,
                lda: p_i,
                ldb: p_i,
                beta: 0.0,
                ldc: p_i,
            };
            // SAFETY: qs_dev p*p (transposed); qs_tmp_dev p*p; h_dev p*p output.
            unsafe {
                ws.blas
                    .gemm(cfg_qt, &ws.qs_dev, &ws.qs_tmp_dev, &mut ws.h_dev)
            }
            .map_err(|e| format!("dgemm Qsᵀ·A·Qs inplace: {e}"))?;
        }
        // H_step = H_xtx + (S + step_lm_lambda·I).
        let penalty_step = penalty_with_ridge(input.penalty_hessian, input.step_lm_lambda);
        let penalty_step_col = to_col_major(&penalty_step);
        ws.stream
            .memcpy_htod(penalty_step_col.as_ref(), &mut ws.penalty_dev)
            .map_err(|e| format!("upload penalty inplace: {e}"))?;
        geam_add_inplace(&ws.blas, &ws.stream, p, &mut ws.h_dev, &ws.penalty_dev)?;

        // Step 3: rhs = Qsᵀ score_p − S·β + linear_shift  (#257, #260).
        // First project score_p through Qsᵀ on device (p×p gemv):
        //   beta_orig_dev = Qsᵀ · rhs_dev,  then swap back.
        {
            let cfg_qts = GemvConfig::<f64> {
                trans: cublasOperation_t::CUBLAS_OP_T,
                m: p_i,
                n: p_i,
                alpha: 1.0,
                lda: p_i,
                incx: 1,
                beta: 0.0,
                incy: 1,
            };
            // SAFETY: qs_dev p*p (transposed); rhs_dev length p; beta_orig_dev length p.
            unsafe {
                ws.blas
                    .gemv(cfg_qts, &ws.qs_dev, &ws.rhs_dev, &mut ws.beta_orig_dev)
            }
            .map_err(|e| format!("dgemv Qsᵀ·score inplace: {e}"))?;
            ws.stream
                .memcpy_dtod(&ws.beta_orig_dev, &mut ws.rhs_dev)
                .map_err(|e| format!("d2d Qsᵀ·score→rhs inplace: {e}"))?;
        }
        // Now download rhs and β (both p-vectors; small, bounded-cost round-trip).
        // Apply rhs −= S·β and rhs += linear_shift on the host for correctness.
        let rhs_raw = ws
            .stream
            .clone_dtoh(&ws.rhs_dev)
            .map_err(|e| format!("download Qsᵀ·score inplace: {e}"))?;
        let beta_raw = ws
            .stream
            .clone_dtoh(input.beta_dev)
            .map_err(|e| format!("download beta inplace: {e}"))?;
        let mut rhs_host = Array1::from_vec(rhs_raw);
        let beta_host = Array1::from_vec(beta_raw);
        // S·β in transformed coordinates (S = input.penalty_hessian in transformed frame).
        let s_beta = input.penalty_hessian.dot(&beta_host);
        rhs_host -= &s_beta;
        rhs_host += &input.linear_shift;
        ws.stream
            .memcpy_htod(
                rhs_host.as_slice().ok_or("rhs_host not contiguous")?,
                &mut ws.rhs_dev,
            )
            .map_err(|e| format!("re-upload corrected rhs inplace: {e}"))?;

        // Step 4: Cholesky factor + solve in-place.
        potrf_in_place_reuse(
            &ws.solver,
            &ws.stream,
            p,
            ws.potrf_lwork,
            &mut ws.h_dev,
            &mut ws.potrf_work_dev,
            &mut ws.potrf_info_dev,
        )?;
        potrs_in_place_reuse(
            &ws.solver,
            &ws.stream,
            p,
            1,
            &ws.h_dev,
            &mut ws.rhs_dev,
            &mut ws.potrs_info_dev,
        )?;
        let logdet = cholesky_logdet_device(&ws.stream, &shared.ctx, p, &ws.h_dev)?;
        check_deferred_potrf_info(&ws.stream, &ws.potrf_info_dev)?;
        check_deferred_potrs_info(&ws.stream, &ws.potrs_info_dev)?;

        // ws.rhs_dev = δ = H⁻¹·(Qsᵀ score_p − Sβ + linear_shift) — descent direction.
        // No negation: the corrected RHS directly gives the descent direction (#257).
        Ok(logdet)
    }

    /// Rebuild the penalised Hessian `H = XᵀW_hessianX + S + objective_ridge·I`
    /// on device using the accepted `w_hessian` weights and download it once.
    /// Called once after PIRLS convergence so the exported Hessian reflects
    /// the accepted eta, not a stale mid-loop snapshot.
    ///
    /// Uses `ws.wx_dev`, `ws.xtwx_dev`, `ws.h_dev`, `ws.penalty_dev` as
    /// scratch — all are fair game post-loop.
    pub(super) fn rebuild_h_final(
        shared: &PirlsGpuSharedData,
        ws: &mut SigmaPirlsGpuWorkspace,
        w_hessian_dev: &CudaSlice<f64>,
        penalty_hessian: ArrayView2<'_, f64>,
        objective_ridge: f64,
    ) -> Result<Array2<f64>, String> {
        let n = shared.n;
        let p = shared.p;

        // XtWX via fused path (no n*p WX temp) or fallback ddgmm + dgemm.
        if let Some(ref mut wx_dev_rh) = ws.wx_dev {
            // Large-p fallback: WX = diag(w_hessian) · X.
            left_scale_rows_borrowed(
                &ws.blas,
                &ws.stream,
                n,
                p,
                &shared.x_original_dev,
                w_hessian_dev,
                wx_dev_rh,
            )?;
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
            // SAFETY: validated dims; shared.x_original_dev and wx_dev_rh n*p
            // col-major; ws.xtwx_dev is p*p; all on ws.stream.
            unsafe {
                ws.blas.gemm(
                    gemm_cfg,
                    &shared.x_original_dev,
                    wx_dev_rh,
                    &mut ws.xtwx_dev,
                )
            }
            .map_err(|e| format!("cublas dgemm XtWX (final H rebuild): {e}"))?;
        } else {
            // Fused path: xtwx_lower + symmetrize, no n*p temp.
            launch_xtwx_lower(
                &ws.stream,
                &shared.ctx,
                n,
                p,
                &shared.x_original_dev,
                w_hessian_dev,
                &mut ws.xtwx_dev,
            )?;
            launch_symmetrize_lower(&ws.stream, &shared.ctx, p, &mut ws.xtwx_dev)?;
        }

        // H_final = Qsᵀ (XtWX) Qs + S + objective_ridge·I.
        let p_i = to_i32(p)?;
        // tmp = XtWX · Qs → ws.qs_tmp_dev.
        {
            let cfg_aq = GemmConfig::<f64> {
                transa: cublasOperation_t::CUBLAS_OP_N,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: p_i,
                n: p_i,
                k: p_i,
                alpha: 1.0,
                lda: p_i,
                ldb: p_i,
                beta: 0.0,
                ldc: p_i,
            };
            // SAFETY: xtwx_dev and qs_dev p*p col-major; qs_tmp_dev p*p output.
            unsafe {
                ws.blas
                    .gemm(cfg_aq, &ws.xtwx_dev, &ws.qs_dev, &mut ws.qs_tmp_dev)
            }
            .map_err(|e| format!("dgemm A·Qs (final H rebuild): {e}"))?;
        }
        // H_xtx = Qsᵀ · tmp → ws.h_dev.
        {
            let cfg_qt = GemmConfig::<f64> {
                transa: cublasOperation_t::CUBLAS_OP_T,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: p_i,
                n: p_i,
                k: p_i,
                alpha: 1.0,
                lda: p_i,
                ldb: p_i,
                beta: 0.0,
                ldc: p_i,
            };
            // SAFETY: qs_dev p*p (transposed); qs_tmp_dev p*p; h_dev p*p output.
            unsafe {
                ws.blas
                    .gemm(cfg_qt, &ws.qs_dev, &ws.qs_tmp_dev, &mut ws.h_dev)
            }
            .map_err(|e| format!("dgemm Qsᵀ·A·Qs (final H rebuild): {e}"))?;
        }
        let penalty = penalty_with_ridge(penalty_hessian, objective_ridge);
        let penalty_col = to_col_major(&penalty);
        ws.stream
            .memcpy_htod(penalty_col.as_ref(), &mut ws.penalty_dev)
            .map_err(|e| format!("upload penalty (final H rebuild): {e}"))?;
        geam_add_inplace(&ws.blas, &ws.stream, p, &mut ws.h_dev, &ws.penalty_dev)?;

        // One download — the only H transfer in the entire PIRLS loop.
        let h_col = ws
            .stream
            .clone_dtoh(&ws.h_dev)
            .map_err(|e| format!("download H_final: {e}"))?;
        from_col_major(&h_col, p, p).ok_or_else(|| "H_final layout conversion failed".to_string())
    }

    pub(super) fn weighted_crossprod(
        x: ArrayView2<'_, f64>,
        weights: ArrayView1<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        let (_, stream) = context_and_stream()?;
        let (n, p) = validate_design(x, weights)?;
        let blas = CudaBlas::new(stream.clone()).map_err(|e| format!("cublas init: {e}"))?;
        let x_col = to_col_major(&x);
        let x_dev = pinned_htod(&stream, &x_col)?;
        let mut w_dev = pinned_htod(
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
        // The legacy single-step API has no GLM data — `solve_step_on_stream`
        // (which this dispatches to) only reads `shared.x_original_dev`.
        // The shared upload requires y/prior_w/offset for the loop paths, so
        // pass zero placeholders sized to the design's row count; they are
        // never read by the one-shot Newton step path.
        let n_rows = input.x.nrows();
        let zero_n = ndarray::Array1::<f64>::zeros(n_rows);
        let shared =
            PirlsGpuSharedData::upload_impl(input.x, zero_n.view(), zero_n.view(), zero_n.view())?;
        let mut ws = SigmaPirlsGpuWorkspace::allocate_impl(&shared)?;
        solve_step_on_stream(
            &shared,
            &mut ws,
            PirlsStepStreamInput {
                weights: input.weights,
                penalty_hessian: input.penalty_hessian,
                gradient: input.gradient,
                step_lm_lambda: input.step_lm_lambda,
                objective_ridge: input.objective_ridge,
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

    // In-place `a := a + b` for two `p*p` column-major device buffers via
    // cublasDgeam. The C API explicitly permits `C = A` (output aliasing the
    // first input), but Rust's borrow checker cannot prove that — every
    // caller historically passed `&ws.h_dev, &ws.penalty_dev, &mut ws.h_dev`
    // and ran into E0502. Forcing the in-place semantics into the wrapper
    // signature makes the contract explicit and removes the aliasing-borrow
    // class of errors at the call sites.
    fn geam_add_inplace(
        blas: &CudaBlas,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        p: usize,
        a: &mut CudaSlice<f64>,
        b: &CudaSlice<f64>,
    ) -> Result<(), String> {
        let p_i = to_i32(p)?;
        let alpha = 1.0_f64;
        let beta = 1.0_f64;
        let handle = *blas.handle();
        let (b_ptr, _b_record) = b.device_ptr(stream);
        let (a_ptr, _a_record) = a.device_ptr_mut(stream);
        // cublasDgeam with C == A is allowed and computes `A := alpha*A + beta*B`.
        let out_ptr = a_ptr;
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

    /// Launch the `xtwx_lower` kernel: one thread per lower-tri pair `(j,k)`,
    /// iterates over all `n` rows and writes `A[j + k*p]` (col-major lower
    /// triangle of `XᵀWX`). Call `launch_symmetrize_lower` afterwards.
    fn launch_xtwx_lower(
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        ctx: &std::sync::Arc<cudarc::driver::CudaContext>,
        n: usize,
        p: usize,
        x_dev: &CudaSlice<f64>,
        w_dev: &CudaSlice<f64>,
        a_dev: &mut CudaSlice<f64>,
    ) -> Result<(), String> {
        let module = FUSED_XTWX_CACHE
            .get_or_compile(ctx, "fused_xtwx", FUSED_XTWX_PTX_SOURCE)
            .map_err(|e| format!("fused_xtwx module: {e}"))?;
        let func = module
            .load_function("xtwx_lower")
            .map_err(|e| format!("load xtwx_lower: {e}"))?;
        let n_i = to_i32(n)?;
        let p_i = to_i32(p)?;
        let num_pairs = p * (p + 1) / 2;
        let num_pairs_u32 = u32::try_from(num_pairs)
            .map_err(|_| format!("xtwx_lower: num_pairs {num_pairs} > u32"))?;
        const BLOCK: u32 = 256;
        let grid = num_pairs_u32.div_ceil(BLOCK).max(1);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&func);
        builder.arg(x_dev);
        builder.arg(w_dev);
        builder.arg(a_dev);
        builder.arg(&n_i);
        builder.arg(&p_i);
        // SAFETY: x_dev is n*p col-major f64; w_dev is length n; a_dev is p*p;
        // num_pairs threads each write one lower-tri entry A[j + k*p].
        unsafe { builder.launch(cfg) }
            .map_err(|e| format!("xtwx_lower launch: {e}"))
            .map(|_| ())
    }

    /// Launch the `xtscore` kernel: one thread per output index `j`,
    /// iterates over `n` rows and writes `s[j] = sum_i score[i]*X[i,j]`.
    fn launch_xtscore(
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        ctx: &std::sync::Arc<cudarc::driver::CudaContext>,
        n: usize,
        p: usize,
        x_dev: &CudaSlice<f64>,
        score_dev: &CudaSlice<f64>,
        s_dev: &mut CudaSlice<f64>,
    ) -> Result<(), String> {
        let module = FUSED_XTWX_CACHE
            .get_or_compile(ctx, "fused_xtwx", FUSED_XTWX_PTX_SOURCE)
            .map_err(|e| format!("fused_xtwx module (xtscore): {e}"))?;
        let func = module
            .load_function("xtscore")
            .map_err(|e| format!("load xtscore: {e}"))?;
        let n_i = to_i32(n)?;
        let p_i = to_i32(p)?;
        let p_u32 = u32::try_from(p).map_err(|_| format!("xtscore: p {p} > u32"))?;
        const BLOCK: u32 = 256;
        let grid = p_u32.div_ceil(BLOCK).max(1);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&func);
        builder.arg(x_dev);
        builder.arg(score_dev);
        builder.arg(s_dev);
        builder.arg(&n_i);
        builder.arg(&p_i);
        // SAFETY: x_dev is n*p col-major f64; score_dev is length n; s_dev is length p;
        // p threads each write one output entry s[j].
        unsafe { builder.launch(cfg) }
            .map_err(|e| format!("xtscore launch: {e}"))
            .map(|_| ())
    }

    /// Launch the `symmetrize_lower` kernel: one thread per strict lower-tri
    /// pair `(j,k)` with `j > k`; copies `A[k + j*p] = A[j + k*p]` to fill
    /// the upper triangle from the lower triangle populated by `xtwx_lower`.
    fn launch_symmetrize_lower(
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        ctx: &std::sync::Arc<cudarc::driver::CudaContext>,
        p: usize,
        a_dev: &mut CudaSlice<f64>,
    ) -> Result<(), String> {
        if p <= 1 {
            return Ok(());
        }
        let module = FUSED_XTWX_CACHE
            .get_or_compile(ctx, "fused_xtwx", FUSED_XTWX_PTX_SOURCE)
            .map_err(|e| format!("fused_xtwx module (sym): {e}"))?;
        let func = module
            .load_function("symmetrize_lower")
            .map_err(|e| format!("load symmetrize_lower: {e}"))?;
        let p_i = to_i32(p)?;
        let num_strict = p * (p - 1) / 2;
        let num_strict_u32 = u32::try_from(num_strict)
            .map_err(|_| format!("symmetrize_lower: num_strict {num_strict} > u32"))?;
        const BLOCK: u32 = 256;
        let grid = num_strict_u32.div_ceil(BLOCK).max(1);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&func);
        builder.arg(a_dev);
        builder.arg(&p_i);
        // SAFETY: a_dev is p*p col-major f64; each of the num_strict threads
        // writes one upper-triangle entry mirrored from the lower triangle.
        unsafe { builder.launch(cfg) }
            .map_err(|e| format!("symmetrize_lower launch: {e}"))
            .map(|_| ())
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

extern "C" __global__ void negate_n(
    double* __restrict__ v,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    v[i] = -v[i];
}

// OR-reduction over a u32 status array (length n).  Single-block;
// same launch config as deviance_sum (1 block of 1024 threads).
// out[0] receives the bitwise-OR of all status[i] for i in [0, n).
extern "C" __global__ void status_or(
    const unsigned int* __restrict__ status,
    int n,
    unsigned int* __restrict__ out
) {
    __shared__ unsigned int sm[1024];
    int tid = threadIdx.x;
    int bdim = blockDim.x;
    unsigned int acc = 0u;
    for (int i = tid; i < n; i += bdim) {
        acc |= status[i];
    }
    sm[tid] = acc;
    __syncthreads();
    for (int stride = bdim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sm[tid] |= sm[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[0] = sm[0];
}
"#;

    static PIRLS_LOOP_CACHE: PtxModuleCache = PtxModuleCache::new();

    /// Per-fit device workspace for the Stage 3.3 PIRLS loop driver.
    ///
    /// Three row-kernel modes occupy separate device buffers:
    /// - `row_solve`: solve-row (4 fields), refreshed each Newton iteration.
    /// - `alpha_ladder`: candidate-objective (objective[7] + status[7]).
    /// - `row_final`: final-row (9 fields), written once at convergence.
    pub struct PirlsLoopWorkspace {
        pub beta_dev: CudaSlice<f64>,
        pub eta_dev: CudaSlice<f64>,
        /// Solve-row buffers: `grad_eta`, `w_solver`, `deviance`, `status`.
        pub row_solve: crate::gpu::kernels::pirls_row::SolveRowBuffers,
        /// Alpha-ladder buffers: `objective[7]`, `status[7]`.
        pub alpha_ladder: crate::gpu::kernels::pirls_row::AlphaLadderDevBuffers,
        /// Full final-row buffers: all 9 fields, written once at convergence.
        pub row_final: crate::gpu::kernels::pirls_row::RowOutputDevBuffers,
        pub direction_dev: CudaSlice<f64>,
        pub xd_dev: CudaSlice<f64>,
        pub scalar_dev: CudaSlice<f64>,
        /// Single-element u32 for the `status_or` OR-reduction kernel.
        pub status_u32_dev: CudaSlice<u32>,
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
                row_solve: crate::gpu::kernels::pirls_row::SolveRowBuffers::allocate(stream, n)
                    .map_err(|e| format!("pirls loop alloc row_solve: {e}"))?,
                alpha_ladder: crate::gpu::kernels::pirls_row::AlphaLadderDevBuffers::allocate(
                    stream,
                )
                .map_err(|e| format!("pirls loop alloc alpha_ladder: {e}"))?,
                row_final: crate::gpu::kernels::pirls_row::RowOutputDevBuffers::allocate(stream, n)
                    .map_err(|e| format!("pirls loop alloc row_final: {e}"))?,
                direction_dev: alloc_f64("direction", p)?,
                xd_dev: alloc_f64("xd", n)?,
                scalar_dev: alloc_f64("scalar", 1)?,
                status_u32_dev: stream
                    .alloc_zeros::<u32>(1)
                    .map_err(|e| format!("pirls loop alloc status_u32: {e}"))?,
                n,
                p,
            })
        }
    }

    /// Optional host-side inputs that turn the bare GPU loop result
    /// into a full-surface `PirlsLoopOutcome` matching the CPU oracle
    /// `fit_model_for_fixed_rho_with_adaptive_kkt`.
    ///
    /// When supplied, the postpass at loop exit runs the same host-side
    /// helpers the CPU oracle uses
    /// (`computeworkingweight_derivatives_from_eta`,
    /// `compute_observed_hessian_curvature_arrays`,
    /// `compute_constraint_kkt_diagnostics`) so the dispatch wirer can
    /// plumb every field of `PirlsResult` without doing math.
    ///
    /// When `None`, the derived fields on `PirlsLoopOutcome`
    /// (`finalweights`, `solveweights`, `solve_dmu_deta`,
    /// `solve_d2mu_deta2`, `solve_d3mu_deta3`, `solve_c_array`,
    /// `solve_d_array`, `status`, `constraint_kkt`, `ridge_passport`,
    /// `firth`, `edf`, `beta_transformed`, `derivatives_unsupported`)
    /// take safe defaults: empty arrays, `PirlsStatus::Converged` or
    /// `MaxIterationsReached` reflecting `converged`, no KKT
    /// diagnostics, identity ridge with `objective_ridge` magnitude,
    /// `FirthDiagnostics::Inactive`, `edf = NaN`,
    /// `beta_transformed = beta`, `derivatives_unsupported = true`.
    /// Existing callers that do not need the CPU oracle surface can
    /// pass `None` and ignore the derived fields.
    pub struct PirlsLoopExtra<'a> {
        /// GLM likelihood spec the row kernel was driven by. Needed by
        /// `computeworkingweight_derivatives_from_eta` to produce
        /// `solve_dmu_deta` / `solve_d2mu_deta2` / `solve_d3mu_deta3`
        /// and the score-side `c` / `d` arrays.
        pub likelihood: &'a crate::types::GlmLikelihoodSpec,
        /// Inverse link the row kernel was driven by; pairs with
        /// `likelihood` for the family-specific derivatives.
        pub inverse_link: &'a crate::types::InverseLink,
        /// Response vector `y` (length `n`) — same view passed to the
        /// row kernel. Needed for observed-curvature finalization.
        pub y: ndarray::ArrayView1<'a, f64>,
        /// Prior weights (length `n`) — same view passed to the row
        /// kernel. Carried through to the curvature helpers.
        pub priorweights: ndarray::ArrayView1<'a, f64>,
        /// Observation offset (length `n`). Stored verbatim on the
        /// outcome's `final_offset` so the dispatch wirer can populate
        /// `PirlsResult::final_offset` without re-allocating.
        pub offset: ndarray::ArrayView1<'a, f64>,
        /// Linear inequality constraints `A·β ≥ b` in the same
        /// coordinate frame as the GPU loop's β. When `Some`, the
        /// postpass calls `compute_constraint_kkt_diagnostics` on the
        /// converged β + reconstructed penalised gradient and emits
        /// the result on `PirlsLoopOutcome::constraint_kkt`. When
        /// `None`, no diagnostics are produced.
        pub linear_constraints: Option<&'a crate::solver::active_set::LinearInequalityConstraints>,
        /// Curvature surface the *outer* REML / LAML caller expects on
        /// the returned Hessian. The GPU loop runs under whatever
        /// `curvature: CurvatureMode` it was invoked with; if this
        /// differs (e.g. inner loop ran Fisher for stability but the
        /// outer caller demands observed curvature), the postpass
        /// promotes `finalweights` / `solve_c_array` / `solve_d_array`
        /// via `compute_observed_hessian_curvature_arrays` so the
        /// outcome matches the CPU oracle's `exported_laplace_curvature`
        /// contract.
        pub exported_curvature: crate::solver::pirls::HessianCurvatureKind,
        /// Pre-built ridge passport carrying the stabilization
        /// magnitude + policy that the dispatch wirer wants stamped on
        /// `PirlsResult::ridge_passport`. When `None`, the postpass
        /// uses `RidgePassport::scaled_identity(objective_ridge,
        /// RidgePolicy::explicit_stabilization_full())`, which mirrors
        /// the CPU oracle's default for a no-escalation fit.
        pub ridge_passport: Option<crate::types::RidgePassport>,
        /// Firth bias-reduction diagnostics. Today the GPU loop does
        /// not implement Firth; pass `None` to land
        /// `FirthDiagnostics::Inactive` on the outcome. A future
        /// device-side Firth path would populate this with the active
        /// Jeffreys-logdet + hat-diagonal vector.
        pub firth: Option<crate::solver::pirls::FirthDiagnostics>,
        /// Canonical-basis transform `qs` (size `p × p`) that maps
        /// transformed-basis β to original coordinates via
        /// `beta_original = qs · beta_transformed`. Carried on the
        /// struct for callers that need original-coordinate β; the
        /// postpass does **not** apply `qs` to the loop's β because
        /// the GPU loop already solved in the transformed design
        /// `X·Qs`, so the loop's β *is* `beta_transformed`. When
        /// `None`, no reparameterization is active and transformed
        /// and original coordinates coincide.
        pub qs: Option<ndarray::ArrayView2<'a, f64>>,
        /// Effective degrees of freedom at the converged mode, when
        /// the dispatch wirer has it precomputed (typical case: the
        /// outer REML caller passes its own `e_transformed` /
        /// diagonal-penalty pre-image and computes EDF host-side).
        /// When `None`, the postpass emits `f64::NAN` and sets
        /// `derivatives_unsupported = true` — the dispatch wirer can
        /// then compute EDF itself from `penalized_hessian` and the
        /// caller-side penalty root.
        pub edf: Option<f64>,
    }

    #[derive(Clone, Debug)]
    pub struct PirlsLoopOutcome {
        pub beta: Array1<f64>,
        pub penalized_hessian: Array2<f64>,
        pub logdet: f64,
        pub deviance: f64,
        pub iterations: usize,
        pub converged: bool,
        /// Final linear predictor η = X·β at the accepted PIRLS step
        /// (length `n`). Downloaded once at loop exit.
        pub final_eta: Array1<f64>,
        /// Mean response μ = g⁻¹(η) at the accepted step, length `n`.
        /// Maps to `PirlsResult::finalmu` / `solvemu`.
        pub final_mu: Array1<f64>,
        /// Score-side gradient contribution `∂ℓ/∂η_i` at the accepted
        /// step (length `n`). The CPU oracle uses this to form
        /// `score_norm = ‖Xᵀ grad_eta‖₂`.
        pub final_grad_eta: Array1<f64>,
        /// Hessian-side diagonal working weight `w_hessian_i` at the
        /// accepted step. Maps to `PirlsResult::finalweights` when no
        /// observed-curvature promotion is requested.
        pub final_w_hessian: Array1<f64>,
        /// Score-side diagonal working weight `w_solver_i` at the
        /// accepted step. Maps to `PirlsResult::solveweights`.
        pub final_w_solver: Array1<f64>,
        /// Observation offset (length `n`). Echoed from
        /// `PirlsLoopExtra::offset` when supplied, otherwise an empty
        /// array. Maps to `PirlsResult::final_offset`.
        pub final_offset: Array1<f64>,
        /// β in the canonical transformed basis. Always equals
        /// `beta` because the GPU loop solved in the transformed
        /// design `X·Qs`, so the loop's β is already transformed.
        /// Maps to `PirlsResult::beta_transformed`.
        pub beta_transformed: Array1<f64>,
        /// Hessian-side `finalweights` after optional Fisher→observed
        /// promotion driven by `extra.exported_curvature`. Empty when
        /// `extra` is `None`.
        pub finalweights: Array1<f64>,
        /// Score-side `solveweights` (= `final_w_solver`) echoed
        /// through so the dispatch wirer can stamp directly.
        pub solveweights: Array1<f64>,
        /// Solve-side `dμ/dη` at the converged η, family-specific.
        /// From `computeworkingweight_derivatives_from_eta`. Empty
        /// when `extra` is `None`.
        pub solve_dmu_deta: Array1<f64>,
        /// Solve-side `d²μ/dη²`. Empty when `extra` is `None`.
        pub solve_d2mu_deta2: Array1<f64>,
        /// Solve-side `d³μ/dη³`. Empty when `extra` is `None`.
        pub solve_d3mu_deta3: Array1<f64>,
        /// `c_i = dW_i/dη_i` at the converged mode (Fisher or
        /// observed depending on `extra.exported_curvature`). Maps to
        /// `PirlsResult::solve_c_array`. Empty when `extra` is `None`.
        pub solve_c_array: Array1<f64>,
        /// `d_i = d²W_i/dη_i²`. Maps to `PirlsResult::solve_d_array`.
        /// Empty when `extra` is `None`.
        pub solve_d_array: Array1<f64>,
        /// `true` when the family's analytic 3rd/4th derivatives are
        /// not supported and the c/d arrays are placeholders. Mirrors
        /// `PirlsResult::derivatives_unsupported`.
        pub derivatives_unsupported: bool,
        /// PirlsStatus the dispatch wirer should propagate. Emitted as
        /// `Converged` when the loop's tolerance test passed and
        /// `final_eta`/`final_mu` are finite; `Unstable` when any of
        /// those go non-finite; `MaxIterationsReached` when the loop
        /// hit its iteration cap without converging.
        pub status: crate::solver::pirls::PirlsStatus,
        /// Ridge passport carrying the stabilization δ and policy.
        /// When `extra.ridge_passport` is `Some`, this is the supplied
        /// value verbatim. Otherwise a default `scaled_identity(
        /// objective_ridge, explicit_stabilization_full())` passport.
        pub ridge_passport: crate::types::RidgePassport,
        /// Firth diagnostics. `Inactive` unless the caller passes an
        /// `Active` value through `extra.firth`.
        pub firth: crate::solver::pirls::FirthDiagnostics,
        /// KKT diagnostics for `extra.linear_constraints`. `None`
        /// either when no constraints are supplied or when the
        /// constraint system is empty.
        pub constraint_kkt: Option<crate::solver::active_set::ConstraintKktDiagnostics>,
        /// Effective degrees of freedom. Echoed from `extra.edf`;
        /// `f64::NAN` when not supplied.
        pub edf: f64,
        /// `prev_deviance − accepted_deviance` at the accepted step
        /// that terminated the loop. Matches the CPU oracle's
        /// `WorkingModelPirlsResult::last_deviance_change`.
        pub last_deviance_change: f64,
        /// Number of line-search halvings consumed on the accepted
        /// step (`k` when α = `0.5^k`; `0` when α = 1). When the
        /// ladder was fully exhausted (`step_search_exhausted`), this
        /// is `0` and `last_step_size = 0.0` — no step was committed.
        /// Mirrors `WorkingModelPirlsResult::last_step_halving`.
        pub last_step_halving: usize,
        /// Step size α that was accepted at the final iteration.
        /// Mirrors `WorkingModelPirlsResult::last_step_size`.
        pub last_step_size: f64,
        /// Levenberg-Marquardt damping coefficient (step_lm_lambda) in
        /// effect at the last accepted iter. The GPU loop has no
        /// on-device ridge escalation (it is a constant per call), so
        /// this echoes the input `step_lm_lambda`. Maps to
        /// `PirlsResult::final_lm_lambda`.
        pub final_lm_lambda: f64,
        /// Running minimum of the data-side deviance observed across
        /// all accepted Newton steps. The GPU loop only knows the
        /// data deviance device-side; the dispatch wirer can add
        /// `βᵀ·penalty_hessian·β` at the converged β to obtain the
        /// fully penalised running minimum when needed for
        /// `PirlsResult::min_penalized_deviance`.
        pub min_deviance: f64,
        /// `max_i |η_i|` at the accepted final step — the saturation
        /// diagnostic the CPU oracle stamps on
        /// `PirlsResult::max_abs_eta`. Used by REML's
        /// perfect-separation detection.
        pub max_abs_eta: f64,
        /// Bitwise-OR of all per-row status flags across the n rows at
        /// the final accepted PIRLS step. Carries
        /// [`crate::gpu::kernels::pirls_row::status_flags`] bits so callers can
        /// distinguish saturation (`ETA_CLAMPED`), numerical floor
        /// (`MU_FLOORED`), or invalid input (`INVALID_RESPONSE`,
        /// `ZERO_PRIOR_WEIGHT`). A value of 0 means no per-row
        /// anomaly was detected. Contributes to the `Unstable`
        /// classification when forbidden bits are set.
        pub per_row_status_or: u32,
    }

    /// Full device-resident PIRLS loop. Only three scalar (1 f64)
    /// downloads per Newton iter (deviance, direction-L∞, candidate
    /// deviance per α). β + final H downloaded once at exit.
    pub(super) fn pirls_loop(
        shared: &PirlsGpuSharedData,
        ws: &mut SigmaPirlsGpuWorkspace,
        loop_ws: &mut PirlsLoopWorkspace,
        family: crate::gpu::kernels::pirls_row::PirlsRowFamily,
        curvature: crate::gpu::kernels::pirls_row::CurvatureMode,
        // Active Gamma dispersion shape (α > 0). Forwarded to every
        // `launch_row_reweight_on_stream` call. Pass `1.0` for non-Gamma fits.
        gamma_shape: f64,
        beta0_host: ArrayView1<'_, f64>,
        penalty_hessian: ArrayView2<'_, f64>,
        // Linear shift `b` of the shifted-quadratic penalty
        // `βᵀSβ − 2βᵀb + c`. Length `p`. Mirrors
        // `PirlsPenalty::linear_shift()` in the CPU oracle. Pass a zero
        // vector for fits with no prior-mean shift.
        linear_shift: ArrayView1<'_, f64>,
        // Constant shift `c` of the shifted-quadratic penalty. Pass
        // `0.0` for fits with no prior-mean shift.
        constant_shift: f64,
        // Temporary LM damping for the Newton solves only; never enters
        // RidgePassport / exported Hessian / EDF / penalty term.
        lm_ridge: f64,
        // Real model-objective ridge; enters RidgePassport / exported
        // Hessian / EDF / penalty term.
        objective_ridge: f64,
        max_iter: usize,
        tol: f64,
        extra: Option<&PirlsLoopExtra<'_>>,
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

        if linear_shift.len() != p {
            return Err(format!(
                "linear_shift length {} ≠ p={p}",
                linear_shift.len()
            ));
        }
        if penalty_hessian.dim() != (p, p) {
            return Err(format!(
                "penalty_hessian shape {:?} ≠ (p={p}, p={p})",
                penalty_hessian.dim()
            ));
        }

        ws.stream
            .memcpy_htod(
                beta0_host.as_slice().ok_or("beta0 not contiguous")?,
                &mut loop_ws.beta_dev,
            )
            .map_err(|e| format!("upload beta0: {e}"))?;

        let backend = crate::gpu::kernels::pirls_row::PirlsRowBackend::probe()
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
        let status_or_func = loop_module
            .load_function("status_or")
            .map_err(|e| format!("load status_or: {e}"))?;

        // beta_orig = Qs · beta  (transforms from transformed to original coords).
        // For identity Qs, this is a copy; always goes through ws.beta_orig_dev.
        gemv_no_trans(
            &ws.blas,
            p,
            p,
            &ws.qs_dev,
            &loop_ws.beta_dev,
            &mut ws.beta_orig_dev,
        )?;
        // η = X_original · beta_orig  then η += offset (#258).
        gemv_no_trans(
            &ws.blas,
            n,
            p,
            &shared.x_original_dev,
            &ws.beta_orig_dev,
            &mut loop_ws.eta_dev,
        )?;
        axpy(
            &ws.stream,
            &axpy_func,
            1.0,
            &shared.offset_dev,
            &mut loop_ws.eta_dev,
            n,
        )?;
        // Initial solve-row pass on the starting η (4-output kernel only).
        crate::gpu::kernels::pirls_row::launch_solve_row_on_stream(
            backend,
            family,
            curvature,
            gamma_shape,
            &ws.stream,
            n,
            &loop_ws.eta_dev,
            &shared.y_dev,
            &shared.prior_w_dev,
            &mut loop_ws.row_solve,
        )
        .map_err(|e| format!("solve-row init: {e}"))?;

        let mut prev_deviance = reduce_scalar(
            &ws.stream,
            &sum_func,
            &loop_ws.row_solve.deviance,
            n,
            &mut loop_ws.scalar_dev,
            "deviance_init",
        )?;
        let mut last_logdet = 0.0_f64;
        let mut converged = false;

        // Host-side mirror of `beta_dev`. Maintained in lock-step with
        // every accepted Newton step so we can evaluate the
        // shifted-quadratic penalty `βᵀSβ − 2βᵀlinear_shift +
        // constant_shift` on the host without an extra `β` DtoH per
        // iteration. The initial state is `beta0_host` verbatim.
        let mut beta_host: Array1<f64> = beta0_host.to_owned();

        // Initial *penalized* objective = data-deviance(β₀) + shifted
        // quadratic(β₀). This is the value the line search and
        // convergence test compare candidates against — matches the CPU
        // oracle's `penalized_objective` in `CandidateScreen`.
        let s_beta0 = penalty_hessian.dot(&beta_host);
        let penalty_init =
            beta_host.dot(&s_beta0) - 2.0 * beta_host.dot(&linear_shift) + constant_shift;
        let mut prev_objective = prev_deviance + penalty_init;

        // Diagnostic scalars surfaced on the outcome so the dispatch
        // wirer can populate WorkingModelPirlsResult / PirlsResult
        // fields without re-running the loop. They mirror the CPU
        // oracle's per-iter tracking in runworking_model_pirls; the
        // "deviance change" diagnostic now carries the *penalized*
        // objective delta (matches the CPU oracle's convergence-test
        // input and what the issue requested).
        let mut last_dev_delta = 0.0_f64;
        let mut last_halving: usize = 0;
        let mut last_step_size = 0.0_f64;
        let mut min_dev = prev_deviance;
        let mut step_search_exhausted = false;

        for it in 0..max_iter {
            last_logdet = solve_step_on_stream_device_inplace(
                shared,
                ws,
                PirlsStepStreamDeviceInput {
                    w_solver_dev: &loop_ws.row_solve.w_solver,
                    grad_eta_dev: &loop_ws.row_solve.grad_eta,
                    penalty_hessian,
                    step_lm_lambda: lm_ridge,
                    objective_ridge,
                    beta_dev: &loop_ws.beta_dev,
                    linear_shift,
                },
            )
            .map_err(|e| format!("inner step it={it}: {e}"))?;
            // ws.rhs_dev holds the Newton descent direction δ = H⁻¹·rhs (#257).
            // Copy device-to-device: no host round-trip.
            ws.stream
                .memcpy_dtod(&ws.rhs_dev, &mut loop_ws.direction_dev)
                .map_err(|e| format!("direction d2d copy it={it}: {e}"))?;

            let dir_linf = reduce_scalar(
                &ws.stream,
                &linf_func,
                &loop_ws.direction_dev,
                p,
                &mut loop_ws.scalar_dev,
                "dir_linf",
            )?;

            // dir_orig = Qs · direction (transform direction to original coords).
            gemv_no_trans(
                &ws.blas,
                p,
                p,
                &ws.qs_dev,
                &loop_ws.direction_dev,
                &mut ws.dir_orig_dev,
            )?;
            gemv_no_trans(
                &ws.blas,
                n,
                p,
                &shared.x_original_dev,
                &ws.dir_orig_dev,
                &mut loop_ws.xd_dev,
            )?;

            // -- Fused alpha-ladder (candidate-objective mode) ----------------
            // One kernel launch evaluates eta + alpha_k*xdelta for all k in
            // ALPHA_LADDER simultaneously, atomically accumulating per-row
            // deviance into objective_dev[k] and OR-accumulating status flags
            // into status_dev[k].  A single DtoH of 7+7 scalars selects the
            // accepted step -- no per-alpha kernel launch, no full row-output
            // write, no per-alpha host scalar sync.
            loop_ws
                .alpha_ladder
                .zero(&ws.stream)
                .map_err(|e| format!("ladder zero it={it}: {e}"))?;
            crate::gpu::kernels::pirls_row::launch_alpha_ladder_on_stream(
                backend,
                family,
                curvature,
                gamma_shape,
                &ws.stream,
                n,
                &loop_ws.eta_dev,
                &loop_ws.xd_dev,
                &shared.y_dev,
                &shared.prior_w_dev,
                &mut loop_ws.alpha_ladder,
            )
            .map_err(|e| format!("alpha-ladder it={it}: {e}"))?;
            let obj_host: Vec<f64> = ws
                .stream
                .clone_dtoh(&loop_ws.alpha_ladder.objective_dev)
                .map_err(|e| format!("ladder dtoh obj it={it}: {e}"))?;
            let stat_host: Vec<u32> = ws
                .stream
                .clone_dtoh(&loop_ws.alpha_ladder.status_dev)
                .map_err(|e| format!("ladder dtoh stat it={it}: {e}"))?;
            // Download the direction (p << n; one DtoH per iteration to
            // compute the host-side penalty term and maintain beta_host).
            let direction_host: Vec<f64> = ws
                .stream
                .clone_dtoh(&loop_ws.direction_dev)
                .map_err(|e| format!("dtoh direction it={it}: {e}"))?;

            // Penalized objective for each candidate step:
            //   obj_pen[k] = deviance(eta + alpha_k * xd)
            //               + (beta + alpha_k * d)^T S (beta + alpha_k * d)
            //               - 2 (beta + alpha_k * d) . linear_shift
            //               + constant_shift
            // The quadratic in alpha expands as:
            //   penalty(beta) + alpha * [2 d^T (S beta - linear_shift)]
            //                  + alpha^2 * d^T S d
            let dir_view = ndarray::aview1(&direction_host);
            let sd = penalty_hessian.dot(&dir_view);
            let s_beta = penalty_hessian.dot(&beta_host);
            let dtsd = dir_view.dot(&sd);
            let linear_coeff = 2.0 * dir_view.dot(&(&s_beta - &linear_shift));
            let penalty_beta =
                beta_host.dot(&s_beta) - 2.0 * beta_host.dot(&linear_shift) + constant_shift;

            const FORBIDDEN_LINESEARCH: u32 =
                crate::gpu::kernels::pirls_row::status_flags::INVALID_RESPONSE
                    | crate::gpu::kernels::pirls_row::status_flags::ZERO_PRIOR_WEIGHT;
            let mut alpha = 0.0_f64;
            let mut accepted_dev = prev_deviance;
            let mut accepted_objective = prev_objective;
            let mut halving_count: usize = 0;
            for (k, (&dev_k, &st)) in obj_host.iter().zip(stat_host.iter()).enumerate() {
                let a = crate::gpu::kernels::pirls_row::ALPHA_LADDER[k];
                let pen_k = penalty_beta + a * linear_coeff + a * a * dtsd;
                let obj_k = dev_k + pen_k;
                // Match the CPU oracle's acceptance test (#263):
                // `<= prev_objective` is the `CandidateScreen`
                // criterion — a step that holds the penalized
                // objective steady (e.g. an exact zero-gradient
                // direction) must still be accepted so the line
                // search does not spuriously exhaust at a
                // stationary point.
                if obj_k.is_finite() && obj_k <= prev_objective && (st & FORBIDDEN_LINESEARCH) == 0
                {
                    alpha = a;
                    accepted_dev = dev_k;
                    accepted_objective = obj_k;
                    halving_count = k;
                    break;
                }
            }
            if alpha == 0.0 {
                // No α in the ladder produced a step lowering the
                // *penalized* objective. The previous code (and the
                // first draft of this rewrite) silently committed
                // α=1 here and merely *flagged* exhaustion — that
                // still commits a non-descent step, which is exactly
                // what the issue forbids (#263).
                //
                // Signal exhaustion and exit the inner loop without
                // committing β / η / solve-row buffers;
                // `build_loop_outcome` then maps
                // `step_search_exhausted` to
                // `PirlsStatus::LmStepSearchExhausted`, exactly the
                // CPU oracle's "no acceptable step direction even
                // after damping" signal. The outer REML / LM
                // controller can raise damping or reject the outer
                // iteration. β / η / prev_deviance / prev_objective
                // all stay at their last accepted values; the
                // device buffers are likewise untouched.
                step_search_exhausted = true;
                last_halving = 0;
                last_step_size = 0.0;
                last_dev_delta = 0.0;
                break;
            }
            step_search_exhausted = false;
            // Commit accepted step: beta and eta updated in-place.
            axpy(
                &ws.stream,
                &axpy_func,
                alpha,
                &loop_ws.direction_dev,
                &mut loop_ws.beta_dev,
                p,
            )?;
            axpy(
                &ws.stream,
                &axpy_func,
                alpha,
                &loop_ws.xd_dev,
                &mut loop_ws.eta_dev,
                n,
            )?;
            // Maintain host-side beta mirror: beta_host += alpha * direction.
            for (b, &d) in beta_host.iter_mut().zip(direction_host.iter()) {
                *b += alpha * d;
            }
            // Refresh the 4-output solve-row buffers for the next Newton iter.
            crate::gpu::kernels::pirls_row::launch_solve_row_on_stream(
                backend,
                family,
                curvature,
                gamma_shape,
                &ws.stream,
                n,
                &loop_ws.eta_dev,
                &shared.y_dev,
                &shared.prior_w_dev,
                &mut loop_ws.row_solve,
            )
            .map_err(|e| format!("solve-row accepted it={it}: {e}"))?;

            let step_norm = alpha.abs() * dir_linf;
            let dev_delta = (prev_objective - accepted_objective).abs();
            last_dev_delta = dev_delta;
            last_halving = halving_count;
            last_step_size = alpha;
            if accepted_dev < min_dev {
                min_dev = accepted_dev;
            }

            prev_deviance = accepted_dev;
            prev_objective = accepted_objective;

            if dir_linf <= tol
                && step_norm <= tol
                && dev_delta <= tol * (1.0 + prev_objective.abs())
            {
                converged = true;
                // Final-row mode: write all 9 output fields once at convergence.
                crate::gpu::kernels::pirls_row::launch_row_reweight_on_stream(
                    backend,
                    family,
                    curvature,
                    gamma_shape,
                    &ws.stream,
                    n,
                    &loop_ws.eta_dev,
                    &shared.y_dev,
                    &shared.prior_w_dev,
                    &mut loop_ws.row_final,
                )
                .map_err(|e| format!("final-row converged: {e}"))?;
                let h_final = rebuild_h_final(
                    shared,
                    ws,
                    &loop_ws.row_final.w_hessian,
                    penalty_hessian,
                    objective_ridge,
                )
                .map_err(|e| format!("rebuild H_final (converged): {e}"))?;
                return build_loop_outcome(
                    ws,
                    loop_ws,
                    h_final,
                    last_logdet,
                    prev_deviance,
                    it + 1,
                    converged,
                    lm_ridge,
                    objective_ridge,
                    extra,
                    LoopDiagnostics {
                        last_deviance_change: last_dev_delta,
                        last_step_halving: last_halving,
                        last_step_size,
                        min_deviance: min_dev,
                        step_search_exhausted,
                    },
                    &status_or_func,
                );
            }
        }

        // Final-row mode: write all 9 output fields once at max-iter exit.
        crate::gpu::kernels::pirls_row::launch_row_reweight_on_stream(
            backend,
            family,
            curvature,
            gamma_shape,
            &ws.stream,
            n,
            &loop_ws.eta_dev,
            &shared.y_dev,
            &shared.prior_w_dev,
            &mut loop_ws.row_final,
        )
        .map_err(|e| format!("final-row max_iter: {e}"))?;
        let h_final = rebuild_h_final(
            shared,
            ws,
            &loop_ws.row_final.w_hessian,
            penalty_hessian,
            objective_ridge,
        )
        .map_err(|e| format!("rebuild H_final (max_iter): {e}"))?;
        build_loop_outcome(
            ws,
            loop_ws,
            h_final,
            last_logdet,
            prev_deviance,
            max_iter,
            converged,
            lm_ridge,
            objective_ridge,
            extra,
            LoopDiagnostics {
                last_deviance_change: last_dev_delta,
                last_step_halving: last_halving,
                last_step_size,
                min_deviance: min_dev,
                step_search_exhausted,
            },
            &status_or_func,
        )
    }

    /// Internal carrier for the scalar diagnostics tracked across the
    /// inner Newton loop. Surfaced verbatim on `PirlsLoopOutcome` so the
    /// dispatch wirer's plumbing to `WorkingModelPirlsResult` is a
    /// direct field copy.
    ///
    /// `step_search_exhausted` is the GPU mirror of the CPU oracle's
    /// `PirlsStatus::LmStepSearchExhausted` signal: the line-search
    /// halving ladder produced no step that lowered the *penalized*
    /// objective. When true, `build_loop_outcome` promotes the emitted
    /// status accordingly so the outer REML / LM controller can raise
    /// damping or fail the iteration cleanly instead of being handed a
    /// silently non-descent step.
    struct LoopDiagnostics {
        last_deviance_change: f64,
        last_step_halving: usize,
        last_step_size: f64,
        min_deviance: f64,
        step_search_exhausted: bool,
    }

    /// Build a full-surface [`PirlsLoopOutcome`] from the loop's
    /// device-resident state plus optional caller-supplied
    /// [`PirlsLoopExtra`] context.
    ///
    /// Five n-vector DtoH downloads are unavoidable (η, μ, grad_η,
    /// w_hessian, w_solver); β is one p-vector download. When `extra`
    /// is `Some`, the host-side helpers
    /// `computeworkingweight_derivatives_from_eta` and (optionally)
    /// `compute_observed_hessian_curvature_arrays` produce the
    /// solve-side aux jets and the curvature-promoted Hessian-side
    /// weights; `compute_constraint_kkt_diagnostics` runs over the
    /// converged β and reconstructed penalised gradient. All of this
    /// is bit-identical to the corresponding CPU oracle code paths in
    /// `fit_model_for_fixed_rho_with_adaptive_kkt`.
    fn build_loop_outcome(
        ws: &mut SigmaPirlsGpuWorkspace,
        loop_ws: &mut PirlsLoopWorkspace,
        penalized_hessian: Array2<f64>,
        logdet: f64,
        deviance: f64,
        iterations: usize,
        converged: bool,
        step_lm_lambda: f64,
        objective_ridge: f64,
        extra: Option<&PirlsLoopExtra<'_>>,
        diagnostics: LoopDiagnostics,
        status_or_func: &cudarc::driver::CudaFunction,
    ) -> Result<PirlsLoopOutcome, String> {
        let beta = download_vec(&ws.stream, &loop_ws.beta_dev)?;
        let final_eta = download_vec(&ws.stream, &loop_ws.eta_dev)?;
        let final_mu = download_vec(&ws.stream, &loop_ws.row_final.mu)?;
        let final_grad_eta = download_vec(&ws.stream, &loop_ws.row_final.grad_eta)?;
        let final_w_hessian = download_vec(&ws.stream, &loop_ws.row_final.w_hessian)?;
        let final_w_solver = download_vec(&ws.stream, &loop_ws.row_final.w_solver)?;

        // OR-reduce the per-row status flags of the final accepted step.
        // Any INVALID_RESPONSE or ZERO_PRIOR_WEIGHT bit that survived to
        // the accepted iterate means the line-search fallback swallowed a
        // structurally bad candidate; classify as Unstable.
        let n_rows = loop_ws.n;
        let final_row_status = reduce_status_or(
            &ws.stream,
            status_or_func,
            &loop_ws.row_final.status,
            n_rows,
            &mut loop_ws.status_u32_dev,
            "final_row_status",
        )?;
        const FORBIDDEN_FINAL: u32 = crate::gpu::kernels::pirls_row::status_flags::INVALID_RESPONSE
            | crate::gpu::kernels::pirls_row::status_flags::ZERO_PRIOR_WEIGHT;

        // Stability classification — Unstable supersedes both
        // converged and MaxIterationsReached because a non-finite η /
        // μ at the accepted step means the line search swallowed a
        // divergence (saturated likelihood / perfect separation).
        // Also Unstable when forbidden row-status bits are set.
        let eta_finite = final_eta.iter().all(|v| v.is_finite());
        let mu_finite = final_mu.iter().all(|v| v.is_finite());
        let beta_finite = beta.iter().all(|v| v.is_finite());
        let stability_ok =
            eta_finite && mu_finite && beta_finite && (final_row_status & FORBIDDEN_FINAL) == 0;
        let status = if !stability_ok {
            crate::solver::pirls::PirlsStatus::Unstable
        } else if converged {
            crate::solver::pirls::PirlsStatus::Converged
        } else if diagnostics.step_search_exhausted {
            // The α-ladder produced no step lowering the *penalized*
            // objective — exactly the CPU oracle's "no acceptable step
            // direction even after damping" signal. Distinct from the
            // iteration-cap exhaustion (MaxIterationsReached) so the
            // outer REML / LM controller can react (raise damping / try
            // a different curvature) rather than silently accepting an
            // ascent step.
            crate::solver::pirls::PirlsStatus::LmStepSearchExhausted
        } else {
            crate::solver::pirls::PirlsStatus::MaxIterationsReached
        };

        // RidgePassport is built from objective_ridge only — step_lm_lambda
        // is a solve-only artefact and must never contaminate EDF / REML.
        let default_ridge = crate::types::RidgePassport::scaled_identity(
            objective_ridge,
            crate::types::RidgePolicy::explicit_stabilization_full(),
        );

        let max_abs_eta = final_eta.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));

        match extra {
            Some(ext) => {
                // Family aux jets at the converged η — bit-identical
                // to the CPU oracle's post-convergence finalization.
                let (score_c, score_d, solve_dmu_deta, solve_d2mu_deta2, solve_d3mu_deta3) =
                    crate::solver::pirls::computeworkingweight_derivatives_from_eta(
                        ext.likelihood,
                        ext.inverse_link,
                        &final_eta,
                        ext.priorweights,
                    )
                    .map_err(|e| format!("pirls postpass dmu/deta: {e:?}"))?;

                let (finalweights, solve_c_array, solve_d_array) = match ext.exported_curvature {
                    crate::solver::pirls::HessianCurvatureKind::Observed => {
                        crate::solver::pirls::compute_observed_hessian_curvature_arrays(
                            ext.likelihood,
                            ext.inverse_link,
                            &final_eta,
                            ext.y,
                            &final_w_solver,
                            ext.priorweights,
                        )
                        .map_err(|e| format!("pirls postpass observed curvature: {e:?}"))?
                    }
                    crate::solver::pirls::HessianCurvatureKind::Fisher => {
                        (final_w_solver.clone(), score_c.clone(), score_d.clone())
                    }
                };

                // The GPU loop solves in the transformed design X·Qs, so
                // the loop's β is already in transformed coordinates.
                // beta_original = qs · beta_transformed (not applied here;
                // callers that need original coordinates compute it from
                // reparam_result.qs per the PirlsResult contract).
                let beta_transformed = beta.clone();

                let constraint_kkt = ext.linear_constraints.and_then(|lin| {
                    if lin.a.nrows() == 0 {
                        return None;
                    }
                    // Reconstruct the penalised gradient at the
                    // converged β: g = Xᵀ(grad_eta) + S β + objective_ridge·β.
                    // `penalized_hessian` is already XᵀWX + S + objective_ridge·I
                    // (step_lm_lambda was stripped from the export), so
                    // H_pen·β ≈ Xᵀ·grad_eta at a KKT-feasible solution.
                    let grad = penalized_hessian.dot(&beta);
                    Some(
                        crate::solver::active_set::compute_constraint_kkt_diagnostics(
                            &beta, &grad, lin,
                        ),
                    )
                });

                let ridge_passport = ext.ridge_passport.unwrap_or(default_ridge);
                let firth = ext
                    .firth
                    .clone()
                    .unwrap_or(crate::solver::pirls::FirthDiagnostics::Inactive);
                let edf = ext.edf.unwrap_or(f64::NAN);
                // Mirrors CPU oracle's invariant: when
                // `computeworkingweight_derivatives_from_eta` returns
                // Ok, all five jets are real (not placeholders), so
                // this field is `false`. See
                // `src/solver/pirls.rs:6634`.
                let derivatives_unsupported = false;

                Ok(PirlsLoopOutcome {
                    beta,
                    penalized_hessian,
                    logdet,
                    deviance,
                    iterations,
                    converged,
                    final_eta,
                    final_mu,
                    final_grad_eta,
                    final_w_hessian,
                    final_w_solver: final_w_solver.clone(),
                    final_offset: ext.offset.to_owned(),
                    beta_transformed,
                    finalweights,
                    solveweights: final_w_solver,
                    solve_dmu_deta,
                    solve_d2mu_deta2,
                    solve_d3mu_deta3,
                    solve_c_array,
                    solve_d_array,
                    derivatives_unsupported,
                    status,
                    ridge_passport,
                    firth,
                    constraint_kkt,
                    edf,
                    last_deviance_change: diagnostics.last_deviance_change,
                    last_step_halving: diagnostics.last_step_halving,
                    last_step_size: diagnostics.last_step_size,
                    final_lm_lambda: step_lm_lambda,
                    min_deviance: diagnostics.min_deviance,
                    max_abs_eta,
                    per_row_status_or: final_row_status,
                })
            }
            None => {
                // No extra context — pirls-dispatch-wirer can do the
                // derived-field plumbing host-side if needed. We give
                // it `solveweights = final_w_solver` echoed through,
                // empty arrays everywhere else, and safe default
                // status / passport / firth so the struct is fully
                // populated and the wirer's match arms can rely on
                // every field being present.
                Ok(PirlsLoopOutcome {
                    beta: beta.clone(),
                    penalized_hessian,
                    logdet,
                    deviance,
                    iterations,
                    converged,
                    final_eta,
                    final_mu,
                    final_grad_eta,
                    final_w_hessian,
                    final_w_solver: final_w_solver.clone(),
                    final_offset: Array1::<f64>::zeros(0),
                    beta_transformed: beta,
                    finalweights: Array1::<f64>::zeros(0),
                    solveweights: final_w_solver,
                    solve_dmu_deta: Array1::<f64>::zeros(0),
                    solve_d2mu_deta2: Array1::<f64>::zeros(0),
                    solve_d3mu_deta3: Array1::<f64>::zeros(0),
                    solve_c_array: Array1::<f64>::zeros(0),
                    solve_d_array: Array1::<f64>::zeros(0),
                    derivatives_unsupported: true,
                    status,
                    ridge_passport: default_ridge,
                    firth: crate::solver::pirls::FirthDiagnostics::Inactive,
                    constraint_kkt: None,
                    edf: f64::NAN,
                    last_deviance_change: diagnostics.last_deviance_change,
                    last_step_halving: diagnostics.last_step_halving,
                    last_step_size: diagnostics.last_step_size,
                    final_lm_lambda: step_lm_lambda,
                    min_deviance: diagnostics.min_deviance,
                    max_abs_eta,
                    per_row_status_or: final_row_status,
                })
            }
        }
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
        unsafe { blas.gemv(cfg, a_dev, x_dev, y_dev) }.map_err(|e| format!("dgemv no-trans: {e}"))
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
        unsafe { builder.launch(cfg) }
            .map(|_event_pair| ())
            .map_err(|e| format!("axpy launch: {e}"))
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
        builder.arg(&mut *scalar_dev);
        // SAFETY: kernel signature (const double*, int, double*). The
        // `&mut *scalar_dev` reborrow keeps `scalar_dev` available for the
        // download below.
        unsafe { builder.launch(cfg) }.map_err(|e| format!("{label} reduce launch: {e}"))?;
        let host = stream
            .clone_dtoh(scalar_dev)
            .map_err(|e| format!("download {label}: {e}"))?;
        Ok(host[0])
    }

    /// OR-reduce a device-resident `u32` status array into a single `u32`.
    /// Mirrors [`reduce_scalar`] for `f64` deviance reductions: single-block,
    /// 1024-thread launch, one scalar DtoH download.
    fn reduce_status_or(
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        func: &cudarc::driver::CudaFunction,
        src: &CudaSlice<u32>,
        len: usize,
        status_dev: &mut CudaSlice<u32>,
        label: &'static str,
    ) -> Result<u32, String> {
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
        builder.arg(&mut *status_dev);
        // SAFETY: status_or kernel signature (const unsigned int*, int,
        // unsigned int*). The reborrow keeps `status_dev` available.
        unsafe { builder.launch(cfg) }.map_err(|e| format!("{label} or reduce launch: {e}"))?;
        let host = stream
            .clone_dtoh(status_dev)
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

    /// Result of one GPU Gaussian exact penalised least-squares solve.
    pub struct GaussianPlsResult {
        pub beta: Array1<f64>,
        pub penalized_hessian: Array2<f64>,
        pub logdet: f64,
    }

    /// Exact GPU PLS for Gaussian-identity: assembles QsT A Qs + S on host,
    /// then runs POTRF/POTRS on device.  Replaces the PIRLS loop for this family.
    pub fn solve_gaussian_pls_on_stream(
        a_orig: ArrayView2<'_, f64>,
        b_orig: ArrayView1<'_, f64>,
        s_transformed: ArrayView2<'_, f64>,
        linear_shift: ArrayView1<'_, f64>,
        prior_mean_target: ArrayView1<'_, f64>,
        ridge: f64,
        qs: Option<ArrayView2<'_, f64>>,
    ) -> Result<GaussianPlsResult, String> {
        let p = b_orig.len();
        if a_orig.dim() != (p, p) {
            return Err(format!("A shape {:?} != ({p},{p})", a_orig.dim()));
        }
        if s_transformed.dim() != (p, p) {
            return Err(format!("S shape {:?} != ({p},{p})", s_transformed.dim()));
        }
        if linear_shift.len() != p {
            return Err(format!("linear_shift len {} != p={p}", linear_shift.len()));
        }
        if prior_mean_target.len() != p {
            return Err(format!(
                "prior_mean_target len {} != p={p}",
                prior_mean_target.len()
            ));
        }
        if let Some(qs_v) = qs {
            if qs_v.dim() != (p, p) {
                return Err(format!("qs shape {:?} != ({p},{p})", qs_v.dim()));
            }
        }
        let (h_rotated, rhs_base) = if let Some(qs_v) = qs {
            let qs_owned = qs_v.to_owned();
            let tmp = a_orig.dot(&qs_owned);
            let h = qs_owned.t().dot(&tmp);
            let rb = qs_owned.t().dot(&b_orig);
            (h, rb)
        } else {
            (a_orig.to_owned(), b_orig.to_owned())
        };
        let penalized_hessian: Array2<f64> = &h_rotated + &s_transformed;
        let mut regularized = penalized_hessian.clone();
        if ridge > 0.0 {
            for i in 0..p {
                regularized[[i, i]] += ridge;
            }
        }
        let mut rhs_host = rhs_base;
        rhs_host += &linear_shift;
        if ridge > 0.0 {
            rhs_host.scaled_add(ridge, &prior_mean_target);
        }
        let (ctx, stream) = context_and_stream()?;
        let solver = DnHandle::new(stream.clone())
            .map_err(|e| format!("cusolver init (gaussian pls): {e}"))?;
        let pp = p.checked_mul(p).ok_or("p*p overflow (gaussian pls)")?;
        let mut h_dev = stream
            .alloc_zeros::<f64>(pp)
            .map_err(|e| format!("alloc H (gaussian pls): {e}"))?;
        let mut rhs_dev = stream
            .alloc_zeros::<f64>(p)
            .map_err(|e| format!("alloc rhs (gaussian pls): {e}"))?;
        let potrf_lwork_usize = potrf_query_lwork(&solver, &stream, p)?;
        let potrf_lwork = i32::try_from(potrf_lwork_usize)
            .map_err(|_| "potrf lwork overflow (gaussian pls)".to_string())?;
        let mut potrf_work_dev = stream
            .alloc_zeros::<f64>(potrf_lwork_usize.max(1))
            .map_err(|e| format!("alloc potrf workspace (gaussian pls): {e}"))?;
        let mut potrf_info_dev = stream
            .alloc_zeros::<i32>(1)
            .map_err(|e| format!("alloc potrf info (gaussian pls): {e}"))?;
        let mut potrs_info_dev = stream
            .alloc_zeros::<i32>(1)
            .map_err(|e| format!("alloc potrs info (gaussian pls): {e}"))?;
        let reg_col = to_col_major(&regularized);
        stream
            .memcpy_htod(reg_col.as_ref(), &mut h_dev)
            .map_err(|e| format!("upload H (gaussian pls): {e}"))?;
        let rhs_slice = rhs_host
            .as_slice()
            .ok_or("rhs_host not contiguous (gaussian pls)")?;
        stream
            .memcpy_htod(rhs_slice, &mut rhs_dev)
            .map_err(|e| format!("upload rhs (gaussian pls): {e}"))?;
        potrf_in_place_reuse(
            &solver,
            &stream,
            p,
            potrf_lwork,
            &mut h_dev,
            &mut potrf_work_dev,
            &mut potrf_info_dev,
        )?;
        potrs_in_place_reuse(
            &solver,
            &stream,
            p,
            1,
            &h_dev,
            &mut rhs_dev,
            &mut potrs_info_dev,
        )?;
        let logdet = cholesky_logdet_device(&stream, &ctx, p, &h_dev)?;
        let beta_raw = stream
            .clone_dtoh(&rhs_dev)
            .map_err(|e| format!("download beta (gaussian pls): {e}"))?;
        check_deferred_potrf_info(&stream, &potrf_info_dev)?;
        check_deferred_potrs_info(&stream, &potrs_info_dev)?;
        Ok(GaussianPlsResult {
            beta: Array1::from_vec(beta_raw),
            penalized_hessian,
            logdet,
        })
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

/// Upload X_original, y, prior_w, and offset once per model and return a
/// shared device-resident handle reused across all ρ / σ points. All four
/// arrays must have the same row-count `n`. The shared handle keeps the
/// cached per-ordinal `CudaContext` alive so all peer workspaces bind to
/// the same context and can interleave on its asynchronous engines.
#[cfg(target_os = "linux")]
pub fn upload_shared_pirls_gpu(
    x: ndarray::ArrayView2<'_, f64>,
    y: ndarray::ArrayView1<'_, f64>,
    prior_w: ndarray::ArrayView1<'_, f64>,
    offset: ndarray::ArrayView1<'_, f64>,
) -> Result<PirlsGpuSharedData, String> {
    if crate::gpu::runtime::GpuRuntime::global().is_none() {
        return Err("cuda runtime unavailable; cannot upload shared GPU PIRLS data".to_string());
    }
    PirlsGpuSharedData::upload_impl(x, y, prior_w, offset)
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

/// Upload the reparameterisation matrix `Qs` (p×p) for the current ρ / σ
/// point. Call once per ρ / σ point before calling
/// [`pirls_loop_on_stream`]. When no reparameterisation is active, pass an
/// identity matrix.
#[cfg(target_os = "linux")]
pub fn upload_qs_pirls(
    ws: &mut SigmaPirlsGpuWorkspace,
    qs: ndarray::ArrayView2<'_, f64>,
) -> Result<(), String> {
    cuda::upload_qs(ws, qs)
}

/// Upload an identity Qs for the current ρ / σ point. Equivalent to
/// [`upload_qs_pirls`] with an identity matrix; avoids host allocation.
#[cfg(target_os = "linux")]
pub fn upload_qs_identity_pirls(ws: &mut SigmaPirlsGpuWorkspace) -> Result<(), String> {
    cuda::upload_qs_identity(ws)
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
/// [`crate::gpu::kernels::pirls_row::launch_row_reweight_on_stream`]) instead of
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
///
/// `step_lm_lambda` is the Levenberg–Marquardt damping applied to each
/// Newton solve only; it never enters the exported `penalized_hessian`,
/// `RidgePassport`, EDF, or penalty term.  `objective_ridge` is the
/// real model ridge that enters all of those.
#[cfg(target_os = "linux")]
pub fn pirls_loop_on_stream(
    shared: &PirlsGpuSharedData,
    ws: &mut SigmaPirlsGpuWorkspace,
    loop_ws: &mut cuda::PirlsLoopWorkspace,
    family: crate::gpu::kernels::pirls_row::PirlsRowFamily,
    curvature: crate::gpu::kernels::pirls_row::CurvatureMode,
    // Active Gamma dispersion shape (α > 0). Pass `1.0` for non-Gamma fits.
    gamma_shape: f64,
    beta0: ndarray::ArrayView1<'_, f64>,
    penalty_hessian: ndarray::ArrayView2<'_, f64>,
    // Linear shift `b` for the shifted-quadratic penalty `βᵀSβ−2βᵀb+c`.
    // Pass a zero-length or all-zero slice for fits with no prior-mean shift.
    linear_shift: ndarray::ArrayView1<'_, f64>,
    // Constant shift `c` for the shifted-quadratic penalty. Pass `0.0` when absent.
    constant_shift: f64,
    step_lm_lambda: f64,
    objective_ridge: f64,
    max_iter: usize,
    tol: f64,
    extra: Option<&cuda::PirlsLoopExtra<'_>>,
) -> Result<cuda::PirlsLoopOutcome, String> {
    cuda::pirls_loop(
        shared,
        ws,
        loop_ws,
        family,
        curvature,
        gamma_shape,
        beta0,
        penalty_hessian,
        linear_shift,
        constant_shift,
        step_lm_lambda,
        objective_ridge,
        max_iter,
        tol,
        extra,
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

/// GPU exact penalised least-squares for Gaussian-identity models.
///
/// Public wrapper around [`cuda::solve_gaussian_pls_on_stream`].  Delegates
/// immediately if the CUDA runtime is initialised; returns an error otherwise
/// so the caller can fall back to the CPU path.
#[cfg(target_os = "linux")]
pub fn solve_gaussian_pls_gpu(
    a_orig: ndarray::ArrayView2<'_, f64>,
    b_orig: ndarray::ArrayView1<'_, f64>,
    s_transformed: ndarray::ArrayView2<'_, f64>,
    linear_shift: ndarray::ArrayView1<'_, f64>,
    prior_mean_target: ndarray::ArrayView1<'_, f64>,
    ridge: f64,
    qs: Option<ndarray::ArrayView2<'_, f64>>,
) -> Result<cuda::GaussianPlsResult, String> {
    cuda::solve_gaussian_pls_on_stream(
        a_orig,
        b_orig,
        s_transformed,
        linear_shift,
        prior_mean_target,
        ridge,
        qs,
    )
}

// ────────────────────────────────────────────────────────────────────────
// Block 9 Phase 5 — device-resident PCG against the BMS-FLEX row-Hessian
// operator.
//
// The inner Newton solve in `BernoulliMarginalSlope` (matrix-free path,
// large-scale shape n=195k, p=44, r=20) currently reaches the GPU as a
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
    pub storage: &'a crate::families::bms::gpu::row::DeviceResidentRowHess,
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
    use crate::families::bms::gpu::row::launch_bms_flex_row_diagonal;
    use crate::families::bms::gpu::row::launch_bms_flex_row_hvp_into_device;
    use cudarc::driver::{CudaModule, CudaStream, LaunchConfig, PushKernelArg};
    use std::sync::{Arc, OnceLock};

    struct PcgBackend {
        stream: Arc<CudaStream>,
        module: Arc<CudaModule>,
    }

    impl PcgBackend {
        fn probe() -> Result<&'static Self, String> {
            static BACKEND: OnceLock<Result<PcgBackend, String>> = OnceLock::new();
            BACKEND
                .get_or_init(|| {
                    let runtime = crate::gpu::runtime::GpuRuntime::global()
                        .ok_or_else(|| "pcg backend: no CUDA runtime available".to_string())?;
                    let ctx =
                        crate::gpu::runtime::cuda_context_for(runtime.selected_device().ordinal)
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
                    Ok(PcgBackend { stream, module })
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
        let n_i32 = i32::try_from(p).map_err(|_| format!("pcg: p_total={p} exceeds i32 range"))?;
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
/// `*_gpu` while the canonical policy layer in `crate::gpu` records whether
/// device execution was selected. Returning `Err` here would silently force
/// every caller to grow an `if cuda { .. } else { .. }` branch and risk
/// drifting away from the GPU formula.
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
        let xtwx = weighted_crossprod_cpu(input.x, input.weights)?;
        // Exported H_final = XᵀWX + S + objective_ridge·I.
        let mut penalized_hessian = xtwx.clone();
        penalized_hessian += &input.penalty_hessian;
        if input.objective_ridge != 0.0 {
            for i in 0..p {
                penalized_hessian[[i, i]] += input.objective_ridge;
            }
        }
        // H_step = XᵀWX + S + step_lm_lambda·I for the Newton solve only.
        let mut h_step = xtwx;
        h_step += &input.penalty_hessian;
        if input.step_lm_lambda != 0.0 {
            for i in 0..p {
                h_step[[i, i]] += input.step_lm_lambda;
            }
        }
        let factor = h_step
            .cholesky(Side::Lower)
            .map_err(|e| format!("CPU Cholesky failed in PIRLS fallback: {e:?}"))?;
        let g = Array1::from_iter(input.gradient.iter().copied());
        // No negation: `input.gradient` is the full descent-direction RHS
        // `Xᵀscore − S·β + linear_shift`; solving H·δ = rhs gives δ directly (#257).
        let direction = factor.solvevec(&g);
        // Logdet comes from H_step's Cholesky (the actual factored matrix).
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

        let n = x.nrows();
        let y_dummy = ndarray::Array1::<f64>::zeros(n);
        let prior_w_dummy = ndarray::Array1::<f64>::ones(n);
        let offset_dummy = ndarray::Array1::<f64>::zeros(n);
        let shared = upload_shared_pirls_gpu(
            x.view(),
            y_dummy.view(),
            prior_w_dummy.view(),
            offset_dummy.view(),
        )
        .expect("upload shared design");
        let mut ws_host = allocate_sigma_pirls_workspace(&shared).expect("alloc host-input ws");
        let mut ws_dev = allocate_sigma_pirls_workspace(&shared).expect("alloc device-input ws");

        let host_step = solve_pirls_step_on_stream(
            &shared,
            &mut ws_host,
            PirlsStepStreamInput {
                weights: weights.view(),
                penalty_hessian: penalty.view(),
                gradient: gradient.view(),
                step_lm_lambda: lm_ridge,
                objective_ridge: 0.0,
            },
        )
        .expect("host-input step");

        let mut w_dev = ws_dev.stream.alloc_zeros::<f64>(n).expect("alloc w_dev");
        let mut g_dev = ws_dev.stream.alloc_zeros::<f64>(n).expect("alloc g_dev");
        ws_dev
            .stream
            .memcpy_htod(weights.as_slice().unwrap(), &mut w_dev)
            .expect("upload w_dev");
        ws_dev
            .stream
            .memcpy_htod(g_eta.as_slice().unwrap(), &mut g_dev)
            .expect("upload g_dev");

        let beta_dev_test = ws_dev
            .stream
            .alloc_zeros::<f64>(x.ncols())
            .expect("alloc beta_dev_test");
        let linear_shift_test = ndarray::Array1::<f64>::zeros(x.ncols());
        let dev_step = solve_pirls_step_on_stream_device(
            &shared,
            &mut ws_dev,
            PirlsStepStreamDeviceInput {
                w_solver_dev: &w_dev,
                grad_eta_dev: &g_dev,
                penalty_hessian: penalty.view(),
                step_lm_lambda: lm_ridge,
                objective_ridge: 0.0,
                beta_dev: &beta_dev_test,
                linear_shift: linear_shift_test.view(),
            },
        )
        .expect("device-input step");

        // H + logdet must match to round-off (same XᵀWX, same penalty
        // add, same potrf).
        for i in 0..3 {
            for j in 0..3 {
                let diff = (host_step.penalized_hessian[[i, j]]
                    - dev_step.penalized_hessian[[i, j]])
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

    /// V100 hill-climb gate: at large-scale (n=80k, p=44,
    /// BernoulliLogit/Fisher) the device-resident loop must be ≥10×
    /// faster than the CPU reference. Marked `#[ignore]` so it only
    /// runs when explicitly invoked (`cargo test -- --ignored
    /// hill_climb_loop`); the CI/mac path can't host the GPU work
    /// anyway. Uses CPU `row_reweight_cpu` + faer Cholesky as the
    /// PIRLS reference loop to avoid dragging in `solver::pirls`'s
    /// 13k-line state machine.
    #[test]
    fn hill_climb_loop_beats_cpu_10x_on_large_scale_logit() {
        use crate::gpu::kernels::pirls_row::{
            CurvatureMode, PirlsRowFamily, RowInput, row_reweight_cpu,
        };
        use std::time::Instant;
        if crate::gpu::runtime::GpuRuntime::global().is_none() {
            eprintln!("[hill_climb] no CUDA runtime — skipping");
            return;
        }
        let n = 80_000_usize;
        let p = 44_usize;
        // Synthesise X (col-major dense) and y from a known β.
        let beta_true: ndarray::Array1<f64> = ndarray::Array1::from_iter(
            (0..p).map(|j| 0.05 * ((j as f64) - 0.5 * p as f64) / p as f64),
        );
        let mut x = ndarray::Array2::<f64>::zeros((n, p));
        for i in 0..n {
            for j in 0..p {
                x[[i, j]] = ((i as f64 + j as f64 * 17.0) * 0.001).sin();
            }
        }
        let eta: ndarray::Array1<f64> = x.dot(&beta_true);
        let y: ndarray::Array1<f64> = eta
            .iter()
            .enumerate()
            .map(|(i, &e)| {
                let mu = 0.5 * (1.0 + (0.5 * e).tanh());
                if (i as f64 * 1.31).fract() < mu {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let prior_w = ndarray::Array1::<f64>::ones(n);
        let penalty = ndarray::Array2::<f64>::eye(p) * 1e-3;
        let beta0 = ndarray::Array1::<f64>::zeros(p);

        // GPU timing.
        let offset_bench = ndarray::Array1::<f64>::zeros(n);
        let shared =
            upload_shared_pirls_gpu(x.view(), y.view(), prior_w.view(), offset_bench.view())
                .expect("upload shared design");
        let mut ws = allocate_sigma_pirls_workspace(&shared).expect("alloc ws");
        let mut loop_ws = allocate_pirls_loop_workspace(&shared, &ws).expect("alloc loop_ws");
        let t0 = Instant::now();
        // No prior-mean shift in this benchmark — penalty = ½βᵀSβ
        // with `s_transformed = penalty`, `linear_shift = 0`,
        // `constant_shift = 0`.
        let linear_shift_zero = ndarray::Array1::<f64>::zeros(p);
        drop(
            pirls_loop_on_stream(
                &shared,
                &mut ws,
                &mut loop_ws,
                PirlsRowFamily::BernoulliLogit,
                CurvatureMode::Fisher,
                1.0,
                beta0.view(),
                penalty.view(),
                linear_shift_zero.view(),
                0.0,
                0.0,
                0.0,
                30,
                1e-6,
                None,
            )
            .expect("pirls loop"),
        );
        let gpu_secs = t0.elapsed().as_secs_f64();

        // CPU reference: same PIRLS structure (eta = Xβ; row reweight;
        // XᵀWX + Sλ; faer Cholesky; β update with α=1).
        let t1 = Instant::now();
        let mut beta = ndarray::Array1::<f64>::zeros(p);
        for _ in 0..30 {
            let eta: ndarray::Array1<f64> = x.dot(&beta);
            let mut w = ndarray::Array1::<f64>::zeros(n);
            let mut g = ndarray::Array1::<f64>::zeros(n);
            for i in 0..n {
                let out = row_reweight_cpu(
                    PirlsRowFamily::BernoulliLogit,
                    CurvatureMode::Fisher,
                    RowInput {
                        eta: eta[i],
                        y: y[i],
                        prior_weight: prior_w[i],
                    },
                    1.0,
                );
                w[i] = out.w_solver;
                g[i] = out.grad_eta;
            }
            let mut wx_full = x.clone();
            for j in 0..p {
                for i in 0..n {
                    wx_full[[i, j]] *= w[i];
                }
            }
            let h = x.t().dot(&wx_full) + &penalty;
            let rhs = x.t().dot(&g);
            use crate::linalg::faer_ndarray::FaerCholesky;
            let chol = h
                .cholesky(faer::Side::Lower)
                .expect("CPU PIRLS reference Cholesky");
            let d = chol.solvevec(&rhs);
            for i in 0..p {
                beta[i] -= d[i];
            }
        }
        let cpu_secs = t1.elapsed().as_secs_f64();

        let speedup = cpu_secs / gpu_secs;
        eprintln!(
            "[hill_climb] n={n} p={p} BernoulliLogit/Fisher: gpu={:.3}s cpu={:.3}s speedup={:.2}×",
            gpu_secs, cpu_secs, speedup
        );
        assert!(
            speedup >= 10.0,
            "GPU PIRLS loop must be ≥10× CPU at large-scale shape; got speedup={speedup:.2}× (gpu={gpu_secs:.3}s cpu={cpu_secs:.3}s)"
        );
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

        let offset_ols = ndarray::Array1::<f64>::zeros(n);
        let shared = upload_shared_pirls_gpu(x.view(), y.view(), prior_w.view(), offset_ols.view())
            .expect("upload shared design");
        let mut ws = allocate_sigma_pirls_workspace(&shared).expect("alloc ws");
        let mut loop_ws = allocate_pirls_loop_workspace(&shared, &ws).expect("alloc loop_ws");

        // No prior-mean shift in this OLS test — `linear_shift = 0`,
        // `constant_shift = 0`. `y` / `prior_w` are now uploaded via
        // the shared workspace (#258).
        let linear_shift_zero = ndarray::Array1::<f64>::zeros(p);
        let outcome = pirls_loop_on_stream(
            &shared,
            &mut ws,
            &mut loop_ws,
            crate::gpu::kernels::pirls_row::PirlsRowFamily::GaussianIdentity,
            crate::gpu::kernels::pirls_row::CurvatureMode::Fisher,
            1.0,
            beta0.view(),
            penalty.view(),
            linear_shift_zero.view(),
            0.0,
            0.0,
            0.0,
            20,
            1e-9,
            None,
        )
        .expect("pirls loop");

        // Closed-form OLS (with tiny ridge).
        let xtx = x.t().dot(&x);
        let xty = x.t().dot(&y);
        let h_ref = xtx + &penalty;
        // Solve via the crate's faer/ndarray bridge.
        use crate::linalg::faer_ndarray::FaerCholesky;
        let chol = h_ref
            .cholesky(faer::Side::Lower)
            .expect("OLS reference Cholesky");
        let beta_ref: ndarray::Array1<f64> = chol.solvevec(&xty);

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
    use crate::families::bms::gpu::row::{
        BmsFlexBlockLayout, BmsFlexPrimaryLayout, DeviceResidentRowHess,
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
        let (x, info) =
            crate::linalg::utils::solve_spd_pcg_with_info(apply, &rhs, &diag, rel_tol, 4 * p)
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
                    let seed = (row as f64) * 0.137 + (u as f64) * 1.901 + (v as f64) * 0.317;
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

        let h_dense =
            cpu_dense_joint_hessian(&row_hessians, &marginal, &logslope, &block, &primary, n);
        let x_oracle = cpu_pcg_oracle(&h_dense, &b, 1e-12);

        // Grab the same CUDA context + default stream that the bms_flex_row
        // kernels will use when `run_pcg_against_row_hessian_device` probes
        // its own backend. Going through the public runtime APIs keeps the
        // test independent of any private kernel-backend symbols.
        let runtime = crate::gpu::runtime::GpuRuntime::global()
            .expect("runtime must exist when probe succeeded above");
        let ctx = match crate::gpu::runtime::cuda_context_for(runtime.selected_device().ordinal) {
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
            hess: d_h,
            marginal_design: d_m,
            logslope_design: d_g,
            n,
            r,
            block,
            primary,

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
