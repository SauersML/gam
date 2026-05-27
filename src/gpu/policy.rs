#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MixedPrecisionPolicy {
    Off,
    Screening,
    Never,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuDispatchPolicy {
    pub xtwx_n_min: usize,
    pub xtwx_flops_min: usize,
    pub xtwx_use_fused_below_p: usize,
    pub gemm_min_flops: usize,
    pub potrf_min_p: usize,
    pub small_dense_batched_potrf_max_p: usize,
    pub small_dense_batched_potrf_min_batch: usize,
    pub syevd_min_p: usize,
    pub sparse_min_nnz: usize,
    pub fused_kernel_min_n: usize,
    pub keep_design_resident_min_bytes: usize,
    pub prefer_gpu_factorization_min_p: usize,
    pub row_kernel_min_n: usize,
    pub mixed_precision: MixedPrecisionPolicy,
}

impl Default for GpuDispatchPolicy {
    /// Auto-dispatch thresholds tuned for biobank-scale workloads:
    ///
    /// * `gemm_min_flops = 1e8` — generic dense GEMM / GEMV is only worth a
    ///   device hop when the kernel is at least 10⁸ flops (e.g. a 320×320×320
    ///   product). Below that, the launch + PCIe round-trip dominates.
    /// * `xtwx_n_min = 50_000`, `xtwx_use_fused_below_p = 256` —
    ///   `Xᵀ·diag(w)·X` requires both `n > 50k` rows AND `p > 256` columns
    ///   before the device wins; the row threshold ensures we stream-amortize
    ///   the weight broadcast and the column threshold rules out tiny GLM-style
    ///   designs that are bandwidth-bound on CPU already.
    /// * `fused_kernel_min_n = 100_000` — the 2×2 joint-Hessian kernel only
    ///   runs on device when `n > 100k`; below that the CPU streaming pass
    ///   keeps the entire working set resident in L3.
    /// * Cholesky / SyEVD live on device whenever the design is large enough
    ///   that the factorization itself dominates (`p ≥ 512` and `p ≥ 256`).
    fn default() -> Self {
        Self {
            xtwx_n_min: 50_000,
            xtwx_flops_min: 100_000_000,
            xtwx_use_fused_below_p: 256,
            gemm_min_flops: 100_000_000,
            potrf_min_p: 512,
            small_dense_batched_potrf_max_p: 32,
            small_dense_batched_potrf_min_batch: 8,
            syevd_min_p: 256,
            sparse_min_nnz: 1_000_000,
            fused_kernel_min_n: 100_000,
            keep_design_resident_min_bytes: 32 * 1024 * 1024,
            prefer_gpu_factorization_min_p: 512,
            row_kernel_min_n: 50_000,
            mixed_precision: MixedPrecisionPolicy::Off,
        }
    }
}

impl GpuDispatchPolicy {
    pub const fn dense_gemv_target_is_gpu(&self, n: usize, p: usize, resident: bool) -> bool {
        resident || n.saturating_mul(p).saturating_mul(2) >= self.gemm_min_flops
    }

    pub const fn xtwx_target_is_gpu(&self, n: usize, p: usize, materialized: bool) -> bool {
        materialized
            && n >= self.xtwx_n_min
            && n.saturating_mul(p).saturating_mul(p).saturating_mul(2) >= self.xtwx_flops_min
    }

    pub const fn potrf_target_is_gpu(&self, p: usize, h_resident: bool) -> bool {
        h_resident && p >= self.potrf_min_p
    }
}

/// Operation discriminator used by the dispatch decision API. Mirrors
/// `super::GpuOperation` at the policy layer.
#[derive(Clone, Copy, Debug)]
pub enum Operation {
    Gemm,
    Gemv,
    XtDiagX,
    XtDiagY,
}

/// Which `(response, link)` family the Stage 3.3 device-resident PIRLS loop
/// can evaluate without going through the Level-B raw-body NVRTC path.
///
/// Mirrors `PirlsRowFamily::ALL` at the policy layer so the predicate stays
/// linkable from the CPU PIRLS entry without dragging a Linux-only enum into
/// every host compilation unit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PirlsLoopFamilyKind {
    BernoulliLogit,
    BernoulliProbit,
    BernoulliCLogLog,
    PoissonLog,
    GaussianIdentity,
    GammaLog,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PirlsLoopCurvatureKind {
    Fisher,
    Observed,
}

/// Inputs to [`should_run_reml_outer_on_device`]. The admission predicate
/// for routing the *outer* REML BFGS-over-ρ loop onto a fully device-resident
/// driver (rather than the host orchestrator that hops out per step).
///
/// Fields are intentionally lifted from data the CPU REML entry has on hand
/// before it touches the seed generator or the inner P-IRLS loop, so the
/// admission check is allocation-free and can short-circuit before any
/// device call.
#[derive(Clone, Copy, Debug)]
pub struct RemlOuterAdmission {
    /// Active design rows (post-transform).
    pub n: usize,
    /// Active design columns / penalised-Hessian dimension.
    pub p: usize,
    /// Number of smoothing parameters ρ the outer BFGS optimises over.
    pub num_rho: usize,
    /// Inner family / link pair the device-resident PIRLS loop can evaluate.
    /// `None` means the family does not map onto the six JIT-cached row
    /// kernels — the outer loop must stay on the host orchestrator because
    /// the inner step would already hop out anyway.
    pub family: Option<PirlsLoopFamilyKind>,
    /// Curvature surface the inner loop will use; tied to `family` via
    /// `pirls_loop_curvature_for`.
    pub curvature: PirlsLoopCurvatureKind,
    /// True when the CUDA runtime is initialised on this host.
    pub gpu_available: bool,
}

/// Inputs to [`should_use_gpu_pirls_loop`]. Each field comes from data the
/// CPU PIRLS entry has on hand before it touches the eigendecomposition
/// engine, so the admission check itself is allocation-free and can short-
/// circuit before any heavy work happens.
#[derive(Clone, Copy, Debug)]
pub struct PirlsLoopAdmission {
    /// Number of rows in the active (post-transform) design matrix.
    pub n: usize,
    /// Number of columns in the active design (i.e. `p` of `Xᵀ X`).
    pub p: usize,
    /// `Some(_)` when the inner family maps onto one of the six JIT-cached
    /// `PirlsRowFamily` variants; `None` for custom families that still
    /// require Stage 6 Level B and have not yet been admitted here.
    pub family: Option<PirlsLoopFamilyKind>,
    /// Curvature surface the inner loop will use; the GPU loop has Fisher +
    /// Observed kernels, anything else (e.g. expected-projection surrogates)
    /// is not admitted.
    pub curvature: PirlsLoopCurvatureKind,
    /// True when the CUDA runtime is initialised on this host (i.e.
    /// `GpuRuntime::global().is_some()`).
    pub gpu_available: bool,
}

impl GpuDispatchPolicy {
    /// Conservative admission predicate for routing
    /// `fit_model_for_fixed_rho_with_adaptive_kkt` through the Stage 3.3
    /// device-resident PIRLS loop instead of the CPU LM loop.
    ///
    /// The thresholds (`n ≥ 50_000`, `p ≥ 32`) are deliberately well above
    /// the matrix-size where a single PIRLS iter's `XᵀWX + Cholesky` would
    /// be PCIe-bandwidth-bound. Smaller fits stay on the CPU LM loop where
    /// the full `PirlsResult` surface (firth, EDF, per-row weights, …) is
    /// already populated as a free side-effect of the iteration.
    pub const fn should_use_gpu_pirls_loop(&self, adm: PirlsLoopAdmission) -> bool {
        if !adm.gpu_available {
            return false;
        }
        if adm.n < self.row_kernel_min_n {
            return false;
        }
        if adm.p < 32 {
            return false;
        }
        match adm.family {
            Some(_) => true,
            None => false,
        }
    }

    /// Admission predicate for routing the outer REML BFGS-over-ρ loop onto
    /// a device-resident driver that keeps the BFGS state (ρ, gradient,
    /// Hessian approx) on-device and only downloads the per-step scalar
    /// metrics (objective value, gradient norm, convergence flag).
    ///
    /// The thresholds piggyback on the existing inner-PIRLS admission floor
    /// (`n ≥ row_kernel_min_n`, `p ≥ 32`) because the device-resident outer
    /// loop calls `pirls_loop_on_stream` per step and must not pay the host
    /// hop for small fits the inner loop would have rejected anyway. The
    /// `num_rho ≥ 2` floor rules out the trivial single-smoother case where
    /// host orchestration is already negligible and the device BFGS state
    /// (one length-`num_rho` gradient + a `num_rho × num_rho` Hessian
    /// approx) collapses to a couple of scalars not worth keeping on device.
    pub const fn should_run_reml_outer_on_device(&self, adm: RemlOuterAdmission) -> bool {
        if !adm.gpu_available {
            return false;
        }
        if adm.n < self.row_kernel_min_n {
            return false;
        }
        if adm.p < 32 {
            return false;
        }
        if adm.num_rho < 2 {
            return false;
        }
        match adm.family {
            Some(_) => true,
            None => false,
        }
    }
}
