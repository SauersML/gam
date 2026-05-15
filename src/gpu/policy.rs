use crate::gpu::device::{GpuCapability, GpuDeviceInfo};
use std::env;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AccelPolicy {
    Auto,
    CpuOnly,
    GpuOnly,
}

impl AccelPolicy {
    #[must_use]
    pub fn from_env() -> Self {
        match env::var("GAM_GPU").ok().as_deref() {
            Some("off" | "0" | "false" | "cpu") => Self::CpuOnly,
            Some("force" | "on" | "1" | "true" | "gpu") => Self::GpuOnly,
            _ => Self::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuDispatchDecision {
    Cpu,
    Gpu,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuOperation {
    Gemm {
        m: usize,
        n: usize,
        k: usize,
    },
    Gemv {
        m: usize,
        k: usize,
    },
    XtDiagX {
        rows: usize,
        cols: usize,
        resident: bool,
    },
    XtDiagY {
        rows: usize,
        x_cols: usize,
        y_cols: usize,
        resident: bool,
    },
    JointHessian2x2 {
        rows: usize,
        a_cols: usize,
        b_cols: usize,
        resident: bool,
    },
    Cholesky {
        cols: usize,
        resident: bool,
        rhs: usize,
    },
    SparseXtDiagX {
        rows: usize,
        cols: usize,
        nnz: usize,
        resident: bool,
    },
    RowKernel {
        rows: usize,
        axes: usize,
        candidates: usize,
        resident: bool,
    },
}

impl GpuOperation {
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Gemm { .. } => "gemm",
            Self::Gemv { .. } => "gemv",
            Self::XtDiagX { .. } => "xt_diag_x",
            Self::XtDiagY { .. } => "xt_diag_y",
            Self::JointHessian2x2 { .. } => "joint_hessian_2x2",
            Self::Cholesky { .. } => "cholesky",
            Self::SparseXtDiagX { .. } => "sparse_xt_diag_x",
            Self::RowKernel { .. } => "row_kernel",
        }
    }

    #[must_use]
    pub fn estimated_flops(&self) -> f64 {
        match *self {
            Self::Gemm { m, n, k } => 2.0 * (m as f64) * (n as f64) * (k as f64),
            Self::Gemv { m, k } => 2.0 * (m as f64) * (k as f64),
            Self::XtDiagX { rows, cols, .. } => 2.0 * (rows as f64) * (cols as f64).powi(2),
            Self::XtDiagY {
                rows,
                x_cols,
                y_cols,
                ..
            } => 2.0 * (rows as f64) * (x_cols as f64) * (y_cols as f64),
            Self::JointHessian2x2 {
                rows,
                a_cols,
                b_cols,
                ..
            } => {
                let c = a_cols + b_cols;
                2.0 * (rows as f64) * (c as f64).powi(2)
            }
            Self::Cholesky { cols, rhs, .. } => {
                ((cols as f64).powi(3) / 3.0) + 2.0 * (cols as f64).powi(2) * (rhs as f64)
            }
            Self::SparseXtDiagX { nnz, cols, .. } => 2.0 * (nnz as f64) * (cols as f64),
            Self::RowKernel {
                rows,
                axes,
                candidates,
                ..
            } => (rows as f64) * axes.max(1) as f64 * candidates.max(1) as f64 * 128.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GpuDispatchPolicy {
    pub accel_policy: AccelPolicy,
    pub gemm_min_flops: f64,
    pub xtwx_min_flops: f64,
    pub potrf_min_p: usize,
    pub sparse_min_nnz: usize,
    pub row_kernel_min_n: usize,
    pub keep_design_resident: bool,
    pub prefer_gpu_for_f64_factorization: bool,
}

impl GpuDispatchPolicy {
    #[must_use]
    pub fn from_env_and_device(device: Option<&GpuDeviceInfo>) -> Self {
        let tier = device
            .map(GpuDeviceInfo::capability_tier)
            .unwrap_or(GpuCapability::Unknown);
        let mut policy = match tier {
            GpuCapability::HopperOrNewer => Self {
                accel_policy: AccelPolicy::from_env(),
                gemm_min_flops: 8.0e7,
                xtwx_min_flops: 1.5e8,
                potrf_min_p: 768,
                sparse_min_nnz: 1_000_000,
                row_kernel_min_n: 50_000,
                keep_design_resident: true,
                prefer_gpu_for_f64_factorization: true,
            },
            GpuCapability::Datacenter => Self {
                accel_policy: AccelPolicy::from_env(),
                gemm_min_flops: 1.5e8,
                xtwx_min_flops: 3.0e8,
                potrf_min_p: 1024,
                sparse_min_nnz: 2_000_000,
                row_kernel_min_n: 100_000,
                keep_design_resident: true,
                prefer_gpu_for_f64_factorization: false,
            },
            _ => Self {
                accel_policy: AccelPolicy::from_env(),
                gemm_min_flops: 3.0e8,
                xtwx_min_flops: 6.0e8,
                potrf_min_p: 1536,
                sparse_min_nnz: 4_000_000,
                row_kernel_min_n: 250_000,
                keep_design_resident: false,
                prefer_gpu_for_f64_factorization: false,
            },
        };
        if let Ok(value) = env::var("GAM_GPU_MIN_N") {
            if let Ok(parsed) = value.parse::<usize>() {
                policy.row_kernel_min_n = parsed;
            }
        }
        policy
    }

    #[must_use]
    pub fn decide(&self, op: GpuOperation, cuda_available: bool) -> GpuDispatchDecision {
        if self.accel_policy == AccelPolicy::CpuOnly || !cuda_available {
            return GpuDispatchDecision::Cpu;
        }
        if self.accel_policy == AccelPolicy::GpuOnly {
            return GpuDispatchDecision::Gpu;
        }
        let flops = op.estimated_flops();
        let gpu = match op {
            GpuOperation::Gemm { .. } | GpuOperation::Gemv { .. } => flops >= self.gemm_min_flops,
            GpuOperation::XtDiagX { resident, .. }
            | GpuOperation::XtDiagY { resident, .. }
            | GpuOperation::JointHessian2x2 { resident, .. } => {
                resident || flops >= self.xtwx_min_flops
            }
            GpuOperation::Cholesky { cols, resident, .. } => {
                cols >= self.potrf_min_p && (resident || self.prefer_gpu_for_f64_factorization)
            }
            GpuOperation::SparseXtDiagX { nnz, resident, .. } => {
                resident || nnz >= self.sparse_min_nnz
            }
            GpuOperation::RowKernel { rows, resident, .. } => {
                resident || rows >= self.row_kernel_min_n
            }
        };
        if gpu {
            GpuDispatchDecision::Gpu
        } else {
            GpuDispatchDecision::Cpu
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_policy_keeps_small_gemv_on_cpu() {
        let policy = GpuDispatchPolicy {
            accel_policy: AccelPolicy::Auto,
            gemm_min_flops: 1_000_000.0,
            xtwx_min_flops: 1_000_000.0,
            potrf_min_p: 512,
            sparse_min_nnz: 10_000,
            row_kernel_min_n: 10_000,
            keep_design_resident: false,
            prefer_gpu_for_f64_factorization: false,
        };
        assert_eq!(
            policy.decide(GpuOperation::Gemv { m: 32, k: 16 }, true),
            GpuDispatchDecision::Cpu
        );
    }

    #[test]
    fn resident_xtwx_can_use_gpu_even_below_flop_threshold() {
        let policy = GpuDispatchPolicy {
            accel_policy: AccelPolicy::Auto,
            gemm_min_flops: 1.0e12,
            xtwx_min_flops: 1.0e12,
            potrf_min_p: 4096,
            sparse_min_nnz: usize::MAX,
            row_kernel_min_n: usize::MAX,
            keep_design_resident: true,
            prefer_gpu_for_f64_factorization: false,
        };
        assert_eq!(
            policy.decide(
                GpuOperation::XtDiagX {
                    rows: 1024,
                    cols: 16,
                    resident: true,
                },
                true,
            ),
            GpuDispatchDecision::Gpu
        );
    }
}
