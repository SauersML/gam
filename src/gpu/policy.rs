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
