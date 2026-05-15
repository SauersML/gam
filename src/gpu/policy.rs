use serde::{Deserialize, Serialize};
use std::env;

/// User-visible GPU environment policy. Defaults match `GAM_GPU=auto` with no
/// validation, no forced graphs, and FP64 accepted-state numerics.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GpuEnv {
    pub mode: String,
    pub device: Option<usize>,
    pub mem_fraction: f64,
    pub validate: String,
    pub graphs: String,
    pub mixed_precision: String,
    pub profile: bool,
    pub calibrate: String,
    pub min_n_override: Option<usize>,
    pub f32_operator_preconditioner: bool,
    pub family_kernels: bool,
}

impl GpuEnv {
    #[must_use]
    pub fn from_env() -> Self {
        Self {
            mode: env::var("GAM_GPU").unwrap_or_else(|_| "auto".to_string()),
            device: env::var("GAM_GPU_DEVICE").ok().and_then(|v| v.parse().ok()),
            mem_fraction: env::var("GAM_GPU_MEM_FRACTION")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.85),
            validate: env::var("GAM_GPU_VALIDATE").unwrap_or_else(|_| "0".to_string()),
            graphs: env::var("GAM_GPU_GRAPHS").unwrap_or_else(|_| "auto".to_string()),
            mixed_precision: env::var("GAM_GPU_MIXED_PRECISION")
                .unwrap_or_else(|_| "off".to_string()),
            profile: env::var("GAM_GPU_PROFILE")
                .map(|v| v == "1")
                .unwrap_or(false),
            calibrate: env::var("GAM_GPU_CALIBRATE").unwrap_or_else(|_| "auto".to_string()),
            min_n_override: env::var("GAM_GPU_MIN_N").ok().and_then(|v| v.parse().ok()),
            f32_operator_preconditioner: env::var("GAM_GPU_F32_OPERATOR_PRECONDITIONER")
                .map(|v| v == "1")
                .unwrap_or(false),
            family_kernels: env::var("GAM_GPU_FAMILY_KERNELS")
                .map(|v| v == "on" || v == "1")
                .unwrap_or(false),
        }
    }

    #[must_use]
    pub fn disabled(&self) -> bool {
        self.mode.eq_ignore_ascii_case("off")
    }

    #[must_use]
    pub fn forced(&self) -> bool {
        self.mode.eq_ignore_ascii_case("force")
    }
}

/// Calibrated thresholds for every phase. Values are conservative CPU-fallback
/// defaults and are overwritten by calibration when a real device is available.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GpuDispatchPolicy {
    pub xtwx_n_min: usize,
    pub xtwx_flops_min: usize,
    pub xtwx_use_fused_below_p: usize,
    pub gemm_min_flops: usize,
    pub potrf_min_p: usize,
    pub syevd_min_p: usize,
    pub sparse_min_nnz: usize,
    pub fused_kernel_min_n: usize,
    pub row_kernel_min_n: usize,
    pub keep_design_resident_min_bytes: usize,
    pub prefer_gpu_factorization_min_p: usize,
    pub device_chunk_bytes: usize,
}

impl Default for GpuDispatchPolicy {
    fn default() -> Self {
        Self {
            xtwx_n_min: 65_536,
            xtwx_flops_min: 256 * 1024 * 1024,
            xtwx_use_fused_below_p: 256,
            gemm_min_flops: 128 * 1024 * 1024,
            potrf_min_p: 512,
            syevd_min_p: 512,
            sparse_min_nnz: 2_000_000,
            fused_kernel_min_n: 8_192,
            row_kernel_min_n: 8_192,
            keep_design_resident_min_bytes: 64 * 1024 * 1024,
            prefer_gpu_factorization_min_p: 512,
            device_chunk_bytes: 128 * 1024 * 1024,
        }
    }
}

/// Operation families covered by Phases 1--10.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum GpuOpKind {
    DenseGemv,
    DenseAtv,
    DenseGemm,
    DenseXtDiagX,
    DenseXtDiagY,
    JointHessian2x2,
    FusedLinkKernel,
    PirlsResidentIteration,
    LmCandidateScreen,
    DenseCholesky,
    DenseSymmetricEigen,
    SparseXtWx,
    SparseSpmm,
    MatrixFreePcg,
    SpatialKernelApply,
    RemlDenseSpectralTrace,
    HutchPlusPlusTrace,
    ProjectedFactorTrace,
    CustomFamilyRowKernel,
    CudaGraphReplay,
    MultiGpuRowShard,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum GpuBackendDecision {
    Cpu { reason: String },
    Gpu { reason: String },
}

impl GpuDispatchPolicy {
    #[must_use]
    pub fn decide(
        &self,
        op: GpuOpKind,
        n: usize,
        p: usize,
        k: usize,
        nnz: usize,
        device_resident: bool,
        runtime_available: bool,
    ) -> GpuBackendDecision {
        if !runtime_available {
            return GpuBackendDecision::Cpu {
                reason: "CUDA runtime unavailable".to_string(),
            };
        }
        let flops = n.saturating_mul(p.max(1)).saturating_mul(k.max(1));
        let use_gpu = match op {
            GpuOpKind::DenseGemv | GpuOpKind::DenseAtv | GpuOpKind::DenseGemm => {
                device_resident || flops >= self.gemm_min_flops
            }
            GpuOpKind::DenseXtDiagX | GpuOpKind::DenseXtDiagY | GpuOpKind::JointHessian2x2 => {
                device_resident
                    || (n >= self.xtwx_n_min
                        && n.saturating_mul(p).saturating_mul(p) >= self.xtwx_flops_min)
            }
            GpuOpKind::DenseCholesky => device_resident && p >= self.potrf_min_p,
            GpuOpKind::DenseSymmetricEigen => device_resident && p >= self.syevd_min_p,
            GpuOpKind::SparseXtWx | GpuOpKind::SparseSpmm => nnz >= self.sparse_min_nnz,
            GpuOpKind::MatrixFreePcg => device_resident && p >= 2048,
            GpuOpKind::FusedLinkKernel
            | GpuOpKind::PirlsResidentIteration
            | GpuOpKind::LmCandidateScreen => n >= self.fused_kernel_min_n,
            GpuOpKind::SpatialKernelApply => {
                device_resident || n.saturating_mul(p) >= self.keep_design_resident_min_bytes / 8
            }
            GpuOpKind::RemlDenseSpectralTrace => device_resident,
            GpuOpKind::HutchPlusPlusTrace => p >= 128 && k >= 4,
            GpuOpKind::ProjectedFactorTrace => device_resident || flops >= self.gemm_min_flops,
            GpuOpKind::CustomFamilyRowKernel => n >= self.row_kernel_min_n,
            GpuOpKind::CudaGraphReplay => device_resident,
            GpuOpKind::MultiGpuRowShard => false,
        };
        if use_gpu {
            GpuBackendDecision::Gpu {
                reason: format!("policy selected GPU for {op:?}"),
            }
        } else {
            GpuBackendDecision::Cpu {
                reason: format!("policy kept {op:?} on CPU"),
            }
        }
    }
}
