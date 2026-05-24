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
    fn default() -> Self {
        Self {
            xtwx_n_min: 8_192,
            xtwx_flops_min: 64 * 1024 * 1024,
            xtwx_use_fused_below_p: 256,
            gemm_min_flops: 32 * 1024 * 1024,
            potrf_min_p: 512,
            syevd_min_p: 256,
            sparse_min_nnz: 1_000_000,
            fused_kernel_min_n: 8_192,
            keep_design_resident_min_bytes: 32 * 1024 * 1024,
            prefer_gpu_factorization_min_p: 512,
            row_kernel_min_n: 8_192,
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
