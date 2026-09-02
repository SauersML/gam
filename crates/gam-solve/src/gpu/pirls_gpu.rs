use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Family-aware scalar contract for the built-in GPU PIRLS row kernels.
///
/// The CUDA kernel ABI still has one `double` slot shared by every built-in
/// family, but only Gamma is allowed to populate it. Non-Gamma callers carry
/// an explicit discriminant instead of manufacturing a unit Gamma shape; the
/// final ABI conversion writes a NaN poison value so any future accidental
/// non-Gamma read fails loudly rather than silently becoming unit scale.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PirlsLoopLikelihoodScale(PirlsLoopLikelihoodScaleKind);

#[derive(Clone, Copy, Debug, PartialEq)]
enum PirlsLoopLikelihoodScaleKind {
    NonGamma,
    GammaShape(f64),
}

impl PirlsLoopLikelihoodScale {
    #[inline]
    pub const fn non_gamma() -> Self {
        Self(PirlsLoopLikelihoodScaleKind::NonGamma)
    }

    pub fn gamma_shape(shape: f64) -> Result<Self, String> {
        if shape.is_finite() && shape > 0.0 {
            Ok(Self(PirlsLoopLikelihoodScaleKind::GammaShape(shape)))
        } else {
            Err(format!(
                "GPU PIRLS Gamma shape must be finite and strictly positive, got {shape:?}"
            ))
        }
    }

    #[cfg(target_os = "linux")]
    fn kernel_argument(
        self,
        family: crate::gpu_kernels::pirls_row::PirlsRowFamily,
    ) -> Result<f64, String> {
        use crate::gpu_kernels::pirls_row::PirlsRowFamily;
        match (family, self.0) {
            (PirlsRowFamily::GammaLog, PirlsLoopLikelihoodScaleKind::GammaShape(shape)) => {
                Ok(shape)
            }
            (PirlsRowFamily::GammaLog, PirlsLoopLikelihoodScaleKind::NonGamma) => {
                Err("GPU Gamma row kernel requires an explicit resolved Gamma shape".to_string())
            }
            (_, PirlsLoopLikelihoodScaleKind::NonGamma) => Ok(f64::NAN),
            (_, PirlsLoopLikelihoodScaleKind::GammaShape(shape)) => Err(format!(
                "GPU non-Gamma row kernel {family:?} received Gamma shape {shape:?}"
            )),
        }
    }
}

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
/// [`crate::gpu_kernels::pirls_row::RowOutputDevBuffers`] populated by the
/// device-side row-reweight kernel. The fixed penalty and linear shift are
/// uploaded asynchronously; the Newton RHS correction itself stays on device.
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
    /// Current coefficient vector β (length p), consumed in place by the
    /// device-side Newton RHS correction.
    pub beta_dev: &'b cudarc::driver::CudaSlice<f64>,
    /// Linear shift vector (length p) in transformed coordinates. It is
    /// uploaded to coefficient scratch without synchronizing the stream, then
    /// added by the same kernel that applies `−Sβ`.
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
    use cudarc::cublas::sys::{
        cublasDdgmm, cublasDgeam, cublasOperation_t, cublasSideMode_t, cublasStatus_t,
    };
    use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, Gemv, GemvConfig};
    use cudarc::cusolver::DnHandle;
    use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg};
    use gam_gpu::device_cache::PtxModuleCache;
    use gam_gpu::driver::{from_col_major, to_col_major};
    use gam_gpu::solver::{
        check_deferred_potrf_info, check_deferred_potrs_info, context_and_stream, pinned_htod,
        potrf_in_place_reuse, potrf_query_lwork, potrs_in_place_reuse,
    };
    use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

    /// Device/runtime failures stay distinct from exact statistical row
    /// refusals.  The latter cross the GPU dispatch boundary as their original
    /// typed [`gam_problem::EstimationError`] instead of being stringified or
    /// retried on a different numerical path.
    #[derive(Debug)]
    pub enum PirlsGpuLoopError {
        Geometry(gam_problem::EstimationError),
        Runtime(String),
    }

    impl std::fmt::Display for PirlsGpuLoopError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::Geometry(error) => write!(f, "{error}"),
                Self::Runtime(message) => f.write_str(message),
            }
        }
    }

    impl std::error::Error for PirlsGpuLoopError {}

    impl From<String> for PirlsGpuLoopError {
        fn from(message: String) -> Self {
            Self::Runtime(message)
        }
    }

    impl From<&str> for PirlsGpuLoopError {
        fn from(message: &str) -> Self {
            Self::Runtime(message.to_owned())
        }
    }

    impl From<gam_problem::EstimationError> for PirlsGpuLoopError {
        fn from(error: gam_problem::EstimationError) -> Self {
            Self::Geometry(error)
        }
    }

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
        use gam_gpu::policy::GpuDispatchPolicy;
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
        if input.beta_dev.len() != p {
            return Err(format!(
                "beta_dev length {} does not match p={p}",
                input.beta_dev.len()
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
        // Keep the correction in coefficient space on the device. The prior
        // implementation downloaded both rhs and beta, performed Sβ on the
        // CPU, and uploaded rhs again on every iteration. Those small transfers
        // still drain the entire CUDA stream and dominated the n=80k, p=44
        // device-resident loop (#2430).
        ws.stream
            .memcpy_htod(
                input
                    .linear_shift
                    .as_slice()
                    .ok_or("linear_shift must be contiguous")?,
                &mut ws.dir_orig_dev,
            )
            .map_err(|e| format!("upload linear shift inplace: {e}"))?;
        let loop_module = PIRLS_LOOP_CACHE
            .get_or_compile(&shared.ctx, "pirls_loop", PIRLS_LOOP_PTX_SOURCE)
            .map_err(|e| format!("load rhs-correction module: {e}"))?;
        let correction_func = loop_module
            .load_function("correct_newton_rhs")
            .map_err(|e| format!("load correct_newton_rhs: {e}"))?;
        let cfg = LaunchConfig {
            grid_dim: ((p as u32).div_ceil(256).max(1), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = ws.stream.launch_builder(&correction_func);
        builder.arg(&mut ws.rhs_dev);
        builder.arg(&ws.penalty_dev);
        builder.arg(input.beta_dev);
        builder.arg(&ws.dir_orig_dev);
        builder.arg(&input.step_lm_lambda);
        builder.arg(&p_i);
        builder.arg(&mut ws.beta_orig_dev);
        // SAFETY: correct_newton_rhs receives p-sized rhs/beta/shift vectors
        // and a column-major p×p penalty matrix; the launch covers p threads
        // and preserves `(S + lm I)β` in coefficient scratch for the selector.
        unsafe { builder.launch(cfg) }
            .map_err(|e| format!("correct Newton rhs on device: {e}"))?;

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
        unsafe { builder.launch(cfg) }.map_err(|e| format!("xtwx_lower launch: {e}"))?;
        Ok(())
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
        unsafe { builder.launch(cfg) }.map_err(|e| format!("xtscore launch: {e}"))?;
        Ok(())
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
        unsafe { builder.launch(cfg) }.map_err(|e| format!("symmetrize_lower launch: {e}"))?;
        Ok(())
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
// __device__ annotation required by newer NVRTC JIT semantics (see
// gpu_kernels/pirls_row.rs common_device_prolog — the #2313 hardware sweep).
extern "C" {
    __device__ double fabs(double);
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

// Correct the projected score in place:
//   rhs = Qs^T score - S beta + linear_shift.
// `penalty_step` stores S + lm*I for the factorization, so subtracting its
// product and adding lm*beta recovers the model penalty S exactly.
extern "C" __global__ void correct_newton_rhs(
    double* __restrict__ rhs,
    const double* __restrict__ penalty_step,
    const double* __restrict__ beta,
    const double* __restrict__ linear_shift,
    double lm,
    int p,
    double* __restrict__ penalty_beta
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p) return;
    double s_beta = 0.0;
    for (int j = 0; j < p; ++j) {
        s_beta += penalty_step[i + j * p] * beta[j];
    }
    penalty_beta[i] = s_beta;
    rhs[i] += -s_beta + lm * beta[i] + linear_shift[i];
}

extern "C" __global__ void apply_penalty(
    const double* __restrict__ penalty_step,
    const double* __restrict__ vector,
    int p,
    double* __restrict__ output
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p) return;
    double value = 0.0;
    for (int j = 0; j < p; ++j) {
        value += penalty_step[i + j * p] * vector[j];
    }
    output[i] = value;
}

// Select the first acceptable member of the seven-point line-search ladder
// without exporting beta, direction, objectives, or refusal summaries.
//
// output layout:
//   [0] alpha, [1] accepted data deviance, [2] accepted penalized objective,
//   [3] halving index, [4] ||direction||_inf,
//   [5] first refusal row, [6] first refusal code, [7] all-refused flag.
extern "C" __global__ void select_alpha(
    const double* __restrict__ data_deviance,
    const unsigned int* __restrict__ refusal_summary,
    const double* __restrict__ beta,
    const double* __restrict__ direction,
    const double* __restrict__ penalty_beta_step,
    const double* __restrict__ penalty_direction_step,
    const double* __restrict__ linear_shift,
    const double* __restrict__ direction_linf,
    double previous_deviance,
    double previous_objective,
    double constant_shift,
    double lm,
    int p,
    double* __restrict__ output
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const double alphas[7] = {
        1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625
    };

    double penalty_beta = constant_shift;
    double linear_coeff_half = 0.0;
    double direction_penalty = 0.0;
    for (int i = 0; i < p; ++i) {
        double s_beta = penalty_beta_step[i] - lm * beta[i];
        double s_direction = penalty_direction_step[i] - lm * direction[i];
        penalty_beta += beta[i] * s_beta - 2.0 * beta[i] * linear_shift[i];
        linear_coeff_half += direction[i] * (s_beta - linear_shift[i]);
        direction_penalty += direction[i] * s_direction;
    }

    output[0] = 0.0;
    output[1] = previous_deviance;
    output[2] = previous_objective;
    output[3] = 0.0;
    output[4] = direction_linf[0];
    output[5] = 4294967295.0;
    output[6] = 0.0;
    output[7] = 1.0;

    for (int k = 0; k < 7; ++k) {
        unsigned int row = refusal_summary[k];
        unsigned int code = refusal_summary[7 + k];
        if (k == 0 && row != 0xffffffffu) {
            output[5] = (double)row;
            output[6] = (double)code;
        }
        if (row == 0xffffffffu) {
            output[7] = 0.0;
            double alpha = alphas[k];
            double objective = data_deviance[k]
                + penalty_beta
                + 2.0 * alpha * linear_coeff_half
                + alpha * alpha * direction_penalty;
            if (isfinite(objective) && objective <= previous_objective) {
                output[0] = alpha;
                output[1] = data_deviance[k];
                output[2] = objective;
                output[3] = (double)k;
                return;
            }
        }
    }
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

// Deterministically select the smallest failing row. out[0] is UINT_MAX on
// success, otherwise the row index; out[1] carries that row's refusal code.
extern "C" __global__ void status_first(
    const unsigned int* __restrict__ status,
    int n,
    unsigned int* __restrict__ out
) {
    __shared__ unsigned int sm_row[1024];
    __shared__ unsigned int sm_code[1024];
    int tid = threadIdx.x;
    int bdim = blockDim.x;
    unsigned int best_row = 0xffffffffu;
    unsigned int best_code = 0u;
    for (int i = tid; i < n; i += bdim) {
        unsigned int code = status[i];
        if (code != 0u && (unsigned int)i < best_row) {
            best_row = (unsigned int)i;
            best_code = code;
        }
    }
    sm_row[tid] = best_row;
    sm_code[tid] = best_code;
    __syncthreads();
    for (int stride = bdim / 2; stride > 0; stride >>= 1) {
        if (tid < stride && sm_row[tid + stride] < sm_row[tid]) {
            sm_row[tid] = sm_row[tid + stride];
            sm_code[tid] = sm_code[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[0] = sm_row[0];
        out[1] = sm_code[0];
    }
}

// Same deterministic reduction for the alpha-major [7*n] ladder status
// matrix. One block handles each alpha; outputs are row[0..7), code[7..14).
extern "C" __global__ void status_first_ladder(
    const unsigned int* __restrict__ status,
    int n,
    unsigned int* __restrict__ out
) {
    __shared__ unsigned int sm_row[1024];
    __shared__ unsigned int sm_code[1024];
    int k = blockIdx.x;
    int tid = threadIdx.x;
    int bdim = blockDim.x;
    unsigned int best_row = 0xffffffffu;
    unsigned int best_code = 0u;
    const unsigned int* candidate = status + ((long long)k * n);
    for (int i = tid; i < n; i += bdim) {
        unsigned int code = candidate[i];
        if (code != 0u && (unsigned int)i < best_row) {
            best_row = (unsigned int)i;
            best_code = code;
        }
    }
    sm_row[tid] = best_row;
    sm_code[tid] = best_code;
    __syncthreads();
    for (int stride = bdim / 2; stride > 0; stride >>= 1) {
        if (tid < stride && sm_row[tid + stride] < sm_row[tid]) {
            sm_row[tid] = sm_row[tid + stride];
            sm_code[tid] = sm_code[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[k] = sm_row[0];
        out[7 + k] = sm_code[0];
    }
}
"#;

    static PIRLS_LOOP_CACHE: PtxModuleCache = PtxModuleCache::new();

    /// Per-fit device workspace for the Stage 3.3 PIRLS loop driver.
    ///
    /// Three row-kernel modes occupy separate device buffers:
    /// - `row_solve`: solve-row (4 fields), refreshed each Newton iteration.
    /// - `alpha_ladder`: candidate-objective (objective[7] + status[7*n]).
    /// - `row_final`: five numerical fields + status, written once at convergence.
    pub struct PirlsLoopWorkspace {
        pub beta_dev: CudaSlice<f64>,
        /// Fixed shifted-quadratic linear term, uploaded once per loop.
        pub linear_shift_dev: CudaSlice<f64>,
        pub eta_dev: CudaSlice<f64>,
        /// Solve-row buffers: `grad_eta`, `w_solver`, `deviance`, `status`.
        pub row_solve: crate::gpu_kernels::pirls_row::SolveRowBuffers,
        /// Alpha-ladder buffers: `objective[7]`, alpha-major `status[7*n]`.
        pub alpha_ladder: crate::gpu_kernels::pirls_row::AlphaLadderDevBuffers,
        /// Full production final-row buffers, written once at convergence.
        pub row_final: crate::gpu_kernels::pirls_row::RowOutputDevBuffers,
        pub direction_dev: CudaSlice<f64>,
        /// Parallel `(S + lm I)δ` contraction consumed by `select_alpha`.
        pub penalty_direction_dev: CudaSlice<f64>,
        pub xd_dev: CudaSlice<f64>,
        pub scalar_dev: CudaSlice<f64>,
        /// Compact alpha-selection record, written and selected entirely on
        /// device, then downloaded in one synchronization per Newton step.
        /// Layout is documented by `select_alpha`.
        pub alpha_selection_dev: CudaSlice<f64>,
        /// Fourteen u32 scratch slots: row/code pairs for one row surface or
        /// all seven alpha-ladder candidates.
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
                linear_shift_dev: alloc_f64("linear shift", p)?,
                eta_dev: alloc_f64("eta", n)?,
                row_solve: crate::gpu_kernels::pirls_row::SolveRowBuffers::allocate(stream, n)
                    .map_err(|e| format!("pirls loop alloc row_solve: {e}"))?,
                alpha_ladder: crate::gpu_kernels::pirls_row::AlphaLadderDevBuffers::allocate(
                    stream, n,
                )
                .map_err(|e| format!("pirls loop alloc alpha_ladder: {e}"))?,
                row_final: crate::gpu_kernels::pirls_row::RowOutputDevBuffers::allocate(stream, n)
                    .map_err(|e| format!("pirls loop alloc row_final: {e}"))?,
                direction_dev: alloc_f64("direction", p)?,
                penalty_direction_dev: alloc_f64("penalty direction", p)?,
                xd_dev: alloc_f64("xd", n)?,
                scalar_dev: alloc_f64("scalar", 1)?,
                alpha_selection_dev: alloc_f64("alpha selection", 8)?,
                status_u32_dev: stream
                    .alloc_zeros::<u32>(14)
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
        pub likelihood: &'a gam_problem::GlmLikelihoodSpec,
        /// Inverse link the row kernel was driven by; pairs with
        /// `likelihood` for the family-specific derivatives.
        pub inverse_link: &'a gam_problem::InverseLink,
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
        pub linear_constraints: Option<&'a gam_problem::LinearInequalityConstraints>,
        /// Curvature surface the *outer* REML / LAML caller expects on
        /// the returned Hessian. The GPU loop runs under whatever
        /// `curvature: CurvatureMode` it was invoked with; if this
        /// differs (e.g. inner loop ran Fisher for stability but the
        /// outer caller demands observed curvature), the postpass
        /// promotes `finalweights` / `solve_c_array` / `solve_d_array`
        /// via `compute_observed_hessian_curvature_arrays` so the
        /// outcome matches the CPU oracle's `exported_laplace_curvature`
        /// contract.
        pub exported_curvature: crate::pirls::HessianCurvatureKind,
        /// Pre-built ridge passport carrying the stabilization
        /// magnitude + policy that the dispatch wirer wants stamped on
        /// `PirlsResult::ridge_passport`. When `None`, the postpass
        /// uses `RidgePassport::scaled_identity(objective_ridge,
        /// RidgePolicy::explicit_stabilization_full())`, which mirrors
        /// the CPU oracle's default for a no-escalation fit.
        pub ridge_passport: Option<gam_problem::RidgePassport>,
        /// Firth bias-reduction diagnostics. Today the GPU loop does
        /// not implement Firth; pass `None` to land
        /// `FirthDiagnostics::Inactive` on the outcome. A future
        /// device-side Firth path would populate this with the active
        /// Jeffreys-logdet + hat-diagonal vector.
        pub firth: Option<crate::pirls::FirthDiagnostics>,
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
        pub status: crate::pirls::PirlsStatus,
        /// Ridge passport carrying the stabilization δ and policy.
        /// When `extra.ridge_passport` is `Some`, this is the supplied
        /// value verbatim. Otherwise a default `scaled_identity(
        /// objective_ridge, explicit_stabilization_full())` passport.
        pub ridge_passport: gam_problem::RidgePassport,
        /// Firth diagnostics. `Inactive` unless the caller passes an
        /// `Active` value through `extra.firth`.
        pub firth: crate::pirls::FirthDiagnostics,
        /// KKT diagnostics for `extra.linear_constraints`. `None`
        /// either when no constraints are supplied or when the
        /// constraint system is empty.
        pub constraint_kkt: Option<crate::active_set::ConstraintKktDiagnostics>,
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
    }

    /// Full device-resident PIRLS loop. Candidate deviances, refusal summaries,
    /// direction, beta, and shifted-penalty algebra stay on device; one compact
    /// alpha decision record synchronizes the host per Newton iteration. Beta
    /// and the final Hessian are downloaded once at exit.
    pub(super) fn pirls_loop(
        shared: &PirlsGpuSharedData,
        ws: &mut SigmaPirlsGpuWorkspace,
        loop_ws: &mut PirlsLoopWorkspace,
        family: crate::gpu_kernels::pirls_row::PirlsRowFamily,
        curvature: crate::gpu_kernels::pirls_row::CurvatureMode,
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
    ) -> Result<PirlsLoopOutcome, PirlsGpuLoopError> {
        let n = shared.n;
        let p = shared.p;
        if loop_ws.n != n || loop_ws.p != p {
            return Err(format!(
                "loop workspace ({}, {}) ≠ shared ({n}, {p})",
                loop_ws.n, loop_ws.p
            )
            .into());
        }
        if beta0_host.len() != p {
            return Err(format!("beta0 length {} ≠ p={p}", beta0_host.len()).into());
        }

        if linear_shift.len() != p {
            return Err(format!("linear_shift length {} ≠ p={p}", linear_shift.len()).into());
        }
        if penalty_hessian.dim() != (p, p) {
            return Err(format!(
                "penalty_hessian shape {:?} ≠ (p={p}, p={p})",
                penalty_hessian.dim()
            )
            .into());
        }

        ws.stream
            .memcpy_htod(
                beta0_host.as_slice().ok_or("beta0 not contiguous")?,
                &mut loop_ws.beta_dev,
            )
            .map_err(|e| format!("upload beta0: {e}"))?;
        ws.stream
            .memcpy_htod(
                linear_shift
                    .as_slice()
                    .ok_or("linear_shift not contiguous")?,
                &mut loop_ws.linear_shift_dev,
            )
            .map_err(|e| format!("upload linear_shift: {e}"))?;

        let backend = crate::gpu_kernels::pirls_row::PirlsRowBackend::probe()
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
        let status_first_func = loop_module
            .load_function("status_first")
            .map_err(|e| format!("load status_first: {e}"))?;
        let status_first_ladder_func = loop_module
            .load_function("status_first_ladder")
            .map_err(|e| format!("load status_first_ladder: {e}"))?;
        let select_alpha_func = loop_module
            .load_function("select_alpha")
            .map_err(|e| format!("load select_alpha: {e}"))?;
        let apply_penalty_func = loop_module
            .load_function("apply_penalty")
            .map_err(|e| format!("load apply_penalty: {e}"))?;

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
        crate::gpu_kernels::pirls_row::launch_solve_row_on_stream(
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
        certify_device_rows(
            &ws.stream,
            &status_first_func,
            &loop_ws.row_solve.status,
            &mut loop_ws.status_u32_dev,
            family,
            curvature,
            gamma_shape,
            &loop_ws.eta_dev,
            &shared.y_dev,
            &shared.prior_w_dev,
            n,
            "solve-row init",
        )?;

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

        // Initial *penalized* objective = data-deviance(β₀) + shifted
        // quadratic(β₀). This is the value the line search and
        // convergence test compare candidates against — matches the CPU
        // oracle's `penalized_objective` in `CandidateScreen`.
        let s_beta0 = penalty_hessian.dot(&beta0_host);
        let penalty_init =
            beta0_host.dot(&s_beta0) - 2.0 * beta0_host.dot(&linear_shift) + constant_shift;
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

            launch_scalar_reduction(
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
            // deviance into objective_dev[k] and writing exact per-row refusal
            // codes. A deterministic device reduction returns seven row/code
            // pairs to device memory; `select_alpha` combines those with the
            // exact shifted-quadratic penalty and direction norm. Only its
            // compact decision record crosses to the host.
            loop_ws
                .alpha_ladder
                .zero(&ws.stream)
                .map_err(|e| format!("ladder zero it={it}: {e}"))?;
            crate::gpu_kernels::pirls_row::launch_alpha_ladder_on_stream(
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
            launch_ladder_status_first_reduction(
                &ws.stream,
                &status_first_ladder_func,
                &loop_ws.alpha_ladder.status_dev,
                n,
                &mut loop_ws.status_u32_dev,
            )?;
            let p_i = to_i32(p)?;
            let contraction_cfg = LaunchConfig {
                grid_dim: ((p as u32).div_ceil(256).max(1), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut contraction_builder = ws.stream.launch_builder(&apply_penalty_func);
            contraction_builder.arg(&ws.penalty_dev);
            contraction_builder.arg(&loop_ws.direction_dev);
            contraction_builder.arg(&p_i);
            contraction_builder.arg(&mut loop_ws.penalty_direction_dev);
            // SAFETY: apply_penalty covers p output rows and reads a
            // column-major p×p matrix plus one p-vector.
            unsafe { contraction_builder.launch(contraction_cfg) }
                .map_err(|e| format!("apply penalty to direction it={it}: {e}"))?;
            let selection_cfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = ws.stream.launch_builder(&select_alpha_func);
            builder.arg(&loop_ws.alpha_ladder.objective_dev);
            builder.arg(&loop_ws.status_u32_dev);
            builder.arg(&loop_ws.beta_dev);
            builder.arg(&loop_ws.direction_dev);
            builder.arg(&ws.beta_orig_dev);
            builder.arg(&loop_ws.penalty_direction_dev);
            builder.arg(&loop_ws.linear_shift_dev);
            builder.arg(&loop_ws.scalar_dev);
            builder.arg(&prev_deviance);
            builder.arg(&prev_objective);
            builder.arg(&constant_shift);
            builder.arg(&lm_ridge);
            builder.arg(&p_i);
            builder.arg(&mut loop_ws.alpha_selection_dev);
            // SAFETY: select_alpha is a single-thread coefficient-space
            // reduction over p-sized vectors and the p×p penalty, with seven
            // ladder objectives and fourteen refusal-summary inputs.
            unsafe { builder.launch(selection_cfg) }
                .map_err(|e| format!("select alpha on device it={it}: {e}"))?;
            let selection = ws
                .stream
                .clone_dtoh(&loop_ws.alpha_selection_dev)
                .map_err(|e| format!("download alpha selection it={it}: {e}"))?;
            let alpha = selection[0];
            let accepted_dev = selection[1];
            let accepted_objective = selection[2];
            let halving_count = selection[3] as usize;
            let dir_linf = selection[4];
            let all_candidates_refused = selection[7] != 0.0;
            if alpha == 0.0 {
                if all_candidates_refused {
                    let row = selection[5] as usize;
                    let code = selection[6] as u32;
                    let eta_host = ws
                        .stream
                        .clone_dtoh(&loop_ws.eta_dev)
                        .map_err(|error| format!("ladder refusal eta download: {error}"))?;
                    let xd_host = ws
                        .stream
                        .clone_dtoh(&loop_ws.xd_dev)
                        .map_err(|error| format!("ladder refusal direction download: {error}"))?;
                    let y_host = ws
                        .stream
                        .clone_dtoh(&shared.y_dev)
                        .map_err(|error| format!("ladder refusal response download: {error}"))?;
                    let prior_host =
                        ws.stream.clone_dtoh(&shared.prior_w_dev).map_err(|error| {
                            format!("ladder refusal prior-weight download: {error}")
                        })?;
                    let trial_eta = eta_host[row]
                        + crate::gpu_kernels::pirls_row::ALPHA_LADDER[0] * xd_host[row];
                    return Err(replay_row_refusal(
                        family,
                        curvature,
                        gamma_shape,
                        row,
                        code,
                        trial_eta,
                        y_host[row],
                        prior_host[row],
                    ));
                }
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
            // Refresh the 4-output solve-row buffers for the next Newton iter.
            crate::gpu_kernels::pirls_row::launch_solve_row_on_stream(
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
            certify_device_rows(
                &ws.stream,
                &status_first_func,
                &loop_ws.row_solve.status,
                &mut loop_ws.status_u32_dev,
                family,
                curvature,
                gamma_shape,
                &loop_ws.eta_dev,
                &shared.y_dev,
                &shared.prior_w_dev,
                n,
                "solve-row accepted",
            )?;

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
                // Final-row mode: write the full production row surface once.
                crate::gpu_kernels::pirls_row::launch_row_reweight_on_stream(
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
                certify_device_rows(
                    &ws.stream,
                    &status_first_func,
                    &loop_ws.row_final.status,
                    &mut loop_ws.status_u32_dev,
                    family,
                    curvature,
                    gamma_shape,
                    &loop_ws.eta_dev,
                    &shared.y_dev,
                    &shared.prior_w_dev,
                    n,
                    "final-row converged",
                )?;
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
                );
            }
        }

        // Final-row mode: write the full production row surface once at exit.
        crate::gpu_kernels::pirls_row::launch_row_reweight_on_stream(
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
        certify_device_rows(
            &ws.stream,
            &status_first_func,
            &loop_ws.row_final.status,
            &mut loop_ws.status_u32_dev,
            family,
            curvature,
            gamma_shape,
            &loop_ws.eta_dev,
            &shared.y_dev,
            &shared.prior_w_dev,
            n,
            "final-row max_iter",
        )?;
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
    ) -> Result<PirlsLoopOutcome, PirlsGpuLoopError> {
        let beta = download_vec(&ws.stream, &loop_ws.beta_dev)?;
        let final_eta = download_vec(&ws.stream, &loop_ws.eta_dev)?;
        let final_mu = download_vec(&ws.stream, &loop_ws.row_final.mu)?;
        let final_grad_eta = download_vec(&ws.stream, &loop_ws.row_final.grad_eta)?;
        let final_w_hessian = download_vec(&ws.stream, &loop_ws.row_final.w_hessian)?;
        let final_w_solver = download_vec(&ws.stream, &loop_ws.row_final.w_solver)?;

        // Stability classification — Unstable supersedes both
        // converged and MaxIterationsReached because a non-finite η /
        // μ at the accepted step means the line search swallowed a
        // divergence (saturated likelihood / perfect separation).
        let eta_finite = final_eta.iter().all(|v| v.is_finite());
        let mu_finite = final_mu.iter().all(|v| v.is_finite());
        let beta_finite = beta.iter().all(|v| v.is_finite());
        let stability_ok = eta_finite && mu_finite && beta_finite;
        let status = if !stability_ok {
            crate::pirls::PirlsStatus::Unstable
        } else if converged {
            crate::pirls::PirlsStatus::Converged
        } else if diagnostics.step_search_exhausted {
            // The α-ladder produced no step lowering the *penalized*
            // objective — exactly the CPU oracle's "no acceptable step
            // direction even after damping" signal. Distinct from the
            // iteration-cap exhaustion (MaxIterationsReached) so the
            // outer REML / LM controller can react (raise damping / try
            // a different curvature) rather than silently accepting an
            // ascent step.
            crate::pirls::PirlsStatus::LmStepSearchExhausted
        } else {
            crate::pirls::PirlsStatus::MaxIterationsReached
        };

        // RidgePassport is built from objective_ridge only — step_lm_lambda
        // is a solve-only artefact and must never contaminate EDF / REML.
        let default_ridge = gam_problem::RidgePassport::scaled_identity(
            objective_ridge,
            gam_linalg::RidgePolicy::exact_full_objective(),
        )
        .map_err(gam_problem::EstimationError::from)?;

        let max_abs_eta = final_eta.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));

        match extra {
            Some(ext) => {
                // Family aux jets at the converged η — bit-identical
                // to the CPU oracle's post-convergence finalization.
                let (score_c, score_d, solve_dmu_deta, solve_d2mu_deta2, solve_d3mu_deta3) =
                    crate::pirls::computeworkingweight_derivatives_from_eta(
                        ext.likelihood,
                        ext.inverse_link,
                        &final_eta,
                        ext.priorweights,
                    )
                    .map_err(PirlsGpuLoopError::Geometry)?;

                let (finalweights, solve_c_array, solve_d_array) = match ext.exported_curvature {
                    crate::pirls::HessianCurvatureKind::Observed => {
                        crate::pirls::compute_observed_hessian_curvature_arrays(
                            ext.likelihood,
                            ext.inverse_link,
                            &final_eta,
                            ext.y,
                            &final_w_solver,
                            ext.priorweights,
                        )
                        .map_err(PirlsGpuLoopError::Geometry)?
                    }
                    crate::pirls::HessianCurvatureKind::Fisher => {
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
                    Some(crate::active_set::compute_constraint_kkt_diagnostics(
                        &beta, &grad, lin,
                    ))
                });

                let ridge_passport = ext.ridge_passport.unwrap_or(default_ridge);
                let firth = ext
                    .firth
                    .clone()
                    .unwrap_or(crate::pirls::FirthDiagnostics::Inactive);
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
                    firth: crate::pirls::FirthDiagnostics::Inactive,
                    constraint_kkt: None,
                    edf: f64::NAN,
                    last_deviance_change: diagnostics.last_deviance_change,
                    last_step_halving: diagnostics.last_step_halving,
                    last_step_size: diagnostics.last_step_size,
                    final_lm_lambda: step_lm_lambda,
                    min_deviance: diagnostics.min_deviance,
                    max_abs_eta,
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
        unsafe { builder.launch(cfg) }.map_err(|e| format!("axpy launch: {e}"))?;
        Ok(())
    }

    fn launch_scalar_reduction(
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        func: &cudarc::driver::CudaFunction,
        src: &CudaSlice<f64>,
        len: usize,
        scalar_dev: &mut CudaSlice<f64>,
        label: &'static str,
    ) -> Result<(), String> {
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
        // caller after the asynchronous launch.
        unsafe { builder.launch(cfg) }.map_err(|e| format!("{label} reduce launch: {e}"))?;
        Ok(())
    }

    fn reduce_scalar(
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        func: &cudarc::driver::CudaFunction,
        src: &CudaSlice<f64>,
        len: usize,
        scalar_dev: &mut CudaSlice<f64>,
        label: &'static str,
    ) -> Result<f64, String> {
        launch_scalar_reduction(stream, func, src, len, scalar_dev, label)?;
        let host = stream
            .clone_dtoh(scalar_dev)
            .map_err(|e| format!("download {label}: {e}"))?;
        Ok(host[0])
    }

    /// Deterministically select the smallest non-zero row status with one
    /// scalar-sized transfer.  Outputs `(row, refusal_code)` or `None`.
    fn reduce_status_first(
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        func: &cudarc::driver::CudaFunction,
        src: &CudaSlice<u32>,
        len: usize,
        status_dev: &mut CudaSlice<u32>,
        label: &'static str,
    ) -> Result<Option<(usize, u32)>, String> {
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
        // SAFETY: status_first kernel signature (const unsigned int*, int,
        // unsigned int*). The output has at least two u32 slots.
        unsafe { builder.launch(cfg) }.map_err(|e| format!("{label} first reduce launch: {e}"))?;
        let host = stream
            .clone_dtoh(status_dev)
            .map_err(|e| format!("download {label}: {e}"))?;
        if host[0] == u32::MAX {
            Ok(None)
        } else {
            Ok(Some((host[0] as usize, host[1])))
        }
    }

    /// Reduce the alpha-major `[7*n]` status matrix in one seven-block launch,
    /// leaving all summaries device-resident for the alpha selector.
    fn launch_ladder_status_first_reduction(
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        func: &cudarc::driver::CudaFunction,
        src: &CudaSlice<u32>,
        n: usize,
        status_dev: &mut CudaSlice<u32>,
    ) -> Result<(), String> {
        const THREADS: u32 = 1024;
        let n_i = to_i32(n)?;
        let cfg = LaunchConfig {
            grid_dim: (crate::gpu_kernels::pirls_row::ALPHA_LADDER_LEN as u32, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(func);
        builder.arg(src);
        builder.arg(&n_i);
        builder.arg(&mut *status_dev);
        // SAFETY: status_first_ladder signature is (const u32*, int, u32*);
        // status_dev owns 14 slots (seven rows followed by seven codes).
        unsafe { builder.launch(cfg) }
            .map_err(|e| format!("alpha-ladder status reduction launch: {e}"))?;
        Ok(())
    }

    fn replay_row_refusal(
        family: crate::gpu_kernels::pirls_row::PirlsRowFamily,
        curvature: crate::gpu_kernels::pirls_row::CurvatureMode,
        gamma_shape: f64,
        row: usize,
        code: u32,
        eta: f64,
        y: f64,
        prior_weight: f64,
    ) -> PirlsGpuLoopError {
        let input = crate::gpu_kernels::pirls_row::RowInput {
            eta,
            y,
            prior_weight,
        };
        match crate::gpu_kernels::pirls_row::row_reweight_cpu_at(
            row,
            family,
            curvature,
            input,
            gamma_shape,
        ) {
            Err(error) => PirlsGpuLoopError::Geometry(error),
            Ok(_) => PirlsGpuLoopError::Geometry(
                gam_problem::EstimationError::PirlsRowGeometryUnrepresentable {
                    row,
                    quantity: crate::gpu_kernels::pirls_row::status_codes::quantity(code),
                    eta,
                    value: f64::from(code),
                },
            ),
        }
    }

    fn certify_device_rows(
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        status_first_func: &cudarc::driver::CudaFunction,
        status: &CudaSlice<u32>,
        status_scratch: &mut CudaSlice<u32>,
        family: crate::gpu_kernels::pirls_row::PirlsRowFamily,
        curvature: crate::gpu_kernels::pirls_row::CurvatureMode,
        gamma_shape: f64,
        eta: &CudaSlice<f64>,
        y: &CudaSlice<f64>,
        prior_weight: &CudaSlice<f64>,
        n: usize,
        label: &'static str,
    ) -> Result<(), PirlsGpuLoopError> {
        let Some((_row, _code)) =
            reduce_status_first(stream, status_first_func, status, n, status_scratch, label)?
        else {
            return Ok(());
        };
        let eta_host = stream
            .clone_dtoh(eta)
            .map_err(|error| format!("{label} refusal eta download: {error}"))?;
        let y_host = stream
            .clone_dtoh(y)
            .map_err(|error| format!("{label} refusal response download: {error}"))?;
        let prior_host = stream
            .clone_dtoh(prior_weight)
            .map_err(|error| format!("{label} refusal prior-weight download: {error}"))?;
        let status_host = stream
            .clone_dtoh(status)
            .map_err(|error| format!("{label} refusal status download: {error}"))?;
        crate::gpu_kernels::pirls_row::replay_first_refusal(
            family,
            curvature,
            gamma_shape,
            &eta_host,
            &y_host,
            &prior_host,
            &status_host,
        )
        .map_err(PirlsGpuLoopError::Geometry)
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
        if gam_gpu::device_runtime::GpuRuntime::resolve(gam_gpu::global_policy())
            .map_err(|error| error.to_string())?
            .is_none()
        {
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
        if gam_gpu::device_runtime::GpuRuntime::resolve(gam_gpu::global_policy())
            .map_err(|error| error.to_string())?
            .is_none()
        {
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
    gam_gpu::device_runtime::GpuRuntime::require()
        .map_err(|error| format!("cannot upload shared GPU PIRLS data: {error}"))?;
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
/// `pirls_loop_on_stream`. When no reparameterisation is active, pass an
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

/// Stage 3.3 device-resident PIRLS loop driver. See
/// [`cuda::pirls_loop`] for the full per-iter contract. One compact
/// device-selected alpha record crosses the host boundary per Newton iteration;
/// β and the final penalised Hessian are downloaded once at loop exit.
///
/// `step_lm_lambda` is the Levenberg–Marquardt damping applied to each
/// Newton solve only; it never enters the exported `penalized_hessian`,
/// `RidgePassport`, EDF, or penalty term.  `objective_ridge` is the
/// real model ridge that enters all of those.
#[cfg(target_os = "linux")]
pub(crate) fn pirls_loop_on_stream(
    shared: &PirlsGpuSharedData,
    ws: &mut SigmaPirlsGpuWorkspace,
    loop_ws: &mut cuda::PirlsLoopWorkspace,
    family: crate::gpu_kernels::pirls_row::PirlsRowFamily,
    curvature: crate::gpu_kernels::pirls_row::CurvatureMode,
    likelihood_scale: PirlsLoopLikelihoodScale,
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
) -> Result<cuda::PirlsLoopOutcome, cuda::PirlsGpuLoopError> {
    let gamma_shape = likelihood_scale
        .kernel_argument(family)
        .map_err(cuda::PirlsGpuLoopError::Runtime)?;
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
/// Public wrapper around `cuda::solve_gaussian_pls_on_stream`.  Delegates
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

/// CPU fallback for the PIRLS-step GPU primitives.  When this build has no
/// CUDA runtime probed, the GPU entry points must still return numerically
/// correct results so that callers can route a single code path through
/// `*_gpu` while the canonical policy layer in `crate::gpu` records whether
/// device execution was selected. Returning `Err` here would silently force
/// every caller to grow an `if cuda { .. } else { .. }` branch and risk
/// drifting away from the GPU formula.
mod cpu_fallback {
    use super::{PirlsGpuInput, PirlsGpuStep};
    use crate::estimate::reml::assembly::xt_diag_x_dense_into;
    use faer::Side;
    use gam_linalg::faer_ndarray::FaerCholesky;
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
    gam_gpu::solver::cholesky_solve_gpu(hessian, rhs)
}

/// Solution-only mixed-precision solve (logdet discarded). Skips the redundant
/// fp64 POTRF so the PIRLS Newton direction solve gets the full fp32-factor
/// speedup; the solution is fp64-accurate via iterative refinement.
pub fn cholesky_solve_only_gpu(
    hessian: ArrayView2<'_, f64>,
    rhs: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, String> {
    gam_gpu::solver::cholesky_solve_only_gpu(hessian, rhs)
}

#[cfg(all(test, target_os = "linux"))]
mod pirls_loop_likelihood_scale_tests {
    use super::PirlsLoopLikelihoodScale;
    use crate::gpu_kernels::pirls_row::PirlsRowFamily;

    #[test]
    fn gpu_row_scale_discriminant_rejects_family_mismatch() {
        assert!(
            PirlsLoopLikelihoodScale::non_gamma()
                .kernel_argument(PirlsRowFamily::GammaLog)
                .is_err()
        );
        let gamma = PirlsLoopLikelihoodScale::gamma_shape(2.0).expect("positive Gamma shape");
        assert!(gamma.kernel_argument(PirlsRowFamily::PoissonLog).is_err());
        assert_eq!(
            gamma
                .kernel_argument(PirlsRowFamily::GammaLog)
                .expect("matching Gamma contract"),
            2.0
        );
    }

    #[test]
    fn non_gamma_kernel_scalar_is_poisoned_not_unit_scaled() {
        let abi_value = PirlsLoopLikelihoodScale::non_gamma()
            .kernel_argument(PirlsRowFamily::PoissonLog)
            .expect("matching non-Gamma contract");
        assert!(abi_value.is_nan());
    }
}

/// CPU-fallback contract for the weighted-crossprod GPU dispatcher.
///
/// `weighted_crossprod_gpu` moved here from `gam-gpu` during the #1521 crate
/// carve. On a host with no usable CUDA runtime it must transparently fall back
/// to the dense CPU path, return `Ok`, and produce the exact XᵀWX. This guards
/// the panic-free / Ok-via-CPU-fallback contract previously (loosely) checked in
/// gam-gpu's `cpu_only_host_never_panics_on_gpu_entry_points`, which could no
/// longer reach the function after the carve.
#[cfg(test)]
mod weighted_crossprod_cpu_fallback_tests {
    use super::weighted_crossprod_gpu;
    use ndarray::{Array1, Array2};

    #[test]
    fn weighted_crossprod_gpu_cpu_fallback_matches_dense_xtwx() {
        // Small, below any GPU dispatch threshold → exercises the CPU fallback
        // on a CPU-only host (and stays Ok on a GPU host via the same contract).
        let x = Array2::<f64>::from_shape_fn((4, 3), |(i, j)| (i + j) as f64 + 1.0);
        let w = Array1::<f64>::from_vec(vec![0.5, 1.0, 1.5, 2.0]);

        let got = weighted_crossprod_gpu(x.view(), w.view())
            .expect("weighted_crossprod_gpu must return Ok via CPU fallback on a CPU-only host");

        // Reference XᵀWX = Σ_k w_k x_k x_kᵀ, formed directly.
        let (n, p) = x.dim();
        let mut expected = Array2::<f64>::zeros((p, p));
        for k in 0..n {
            for i in 0..p {
                for j in 0..p {
                    expected[[i, j]] += w[k] * x[[k, i]] * x[[k, j]];
                }
            }
        }

        assert_eq!(got.dim(), (p, p));
        for i in 0..p {
            for j in 0..p {
                let diff = (got[[i, j]] - expected[[i, j]]).abs();
                assert!(
                    diff <= 1e-10,
                    "XtWX[{i},{j}] mismatch: got vs expected diff={diff}"
                );
            }
        }
    }
}
