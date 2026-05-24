use std::env;

use super::device::{GpuCapability, GpuDeviceInfo};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AccelPolicy {
    Auto,
    CpuOnly,
    GpuOnly,
}

impl AccelPolicy {
    #[must_use]
    pub fn from_env() -> Self {
        match env::var("GAM_GPU")
            .unwrap_or_else(|_| "auto".to_string())
            .to_ascii_lowercase()
            .as_str()
        {
            "off" | "0" | "false" | "cpu" | "cpu-only" => Self::CpuOnly,
            "force" | "on" | "1" | "true" | "gpu" | "gpu-only" => Self::GpuOnly,
            _ => Self::Auto,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GpuDispatchPolicy {
    pub accel_policy: AccelPolicy,
    pub gemm_min_flops: f64,
    pub gemv_min_flops: f64,
    pub xtwx_min_flops: f64,
    pub xtwy_min_flops: f64,
    pub joint_hessian_min_flops: f64,
    pub potrf_min_p: usize,
    pub sparse_min_nnz: usize,
    pub row_kernel_min_n: usize,
    pub keep_design_resident: bool,
    pub prefer_gpu_for_f64_factorization: bool,
    pub mem_fraction: f64,
    pub validate: bool,
    pub profile: bool,
}

impl Default for GpuDispatchPolicy {
    fn default() -> Self {
        Self {
            accel_policy: AccelPolicy::Auto,
            gemm_min_flops: 2.0e9,
            gemv_min_flops: 2.5e8,
            xtwx_min_flops: 1.0e9,
            xtwy_min_flops: 5.0e8,
            joint_hessian_min_flops: 1.0e9,
            potrf_min_p: 1_536,
            sparse_min_nnz: 2_000_000,
            row_kernel_min_n: 250_000,
            keep_design_resident: true,
            prefer_gpu_for_f64_factorization: false,
            mem_fraction: 0.85,
            validate: false,
            profile: false,
        }
    }
}

impl GpuDispatchPolicy {
    #[must_use]
    pub fn from_env_and_device(device: Option<&GpuDeviceInfo>) -> Self {
        let mut policy = Self {
            accel_policy: AccelPolicy::from_env(),
            ..Self::default()
        };
        if let Some(device) = device {
            policy.apply_device_defaults(device);
        }
        policy.mem_fraction =
            env_f64("GAM_GPU_MEM_FRACTION", policy.mem_fraction).clamp(0.05, 0.98);
        policy.gemv_min_flops = env_f64("GAM_GPU_GEMV_MIN_FLOPS", policy.gemv_min_flops);
        policy.gemm_min_flops = env_f64("GAM_GPU_GEMM_MIN_FLOPS", policy.gemm_min_flops);
        policy.xtwx_min_flops = env_f64("GAM_GPU_XTWX_MIN_FLOPS", policy.xtwx_min_flops);
        policy.row_kernel_min_n = env_usize("GAM_GPU_MIN_N", policy.row_kernel_min_n);
        policy.validate = env_bool("GAM_GPU_VALIDATE");
        policy.profile = env_bool("GAM_GPU_PROFILE");
        policy
    }

    pub fn apply_device_defaults(&mut self, device: &GpuDeviceInfo) {
        match device.capability {
            GpuCapability::CudaHopperOrNewer => {
                self.gemm_min_flops = 5.0e8;
                self.gemv_min_flops = 8.0e7;
                self.xtwx_min_flops = 3.5e8;
                self.xtwy_min_flops = 2.0e8;
                self.joint_hessian_min_flops = 3.5e8;
                self.potrf_min_p = 512;
                self.sparse_min_nnz = 500_000;
                self.row_kernel_min_n = 50_000;
                self.prefer_gpu_for_f64_factorization = true;
            }
            GpuCapability::CudaFp64Strong => {
                self.gemm_min_flops = 8.0e8;
                self.gemv_min_flops = 1.2e8;
                self.xtwx_min_flops = 5.0e8;
                self.xtwy_min_flops = 2.5e8;
                self.joint_hessian_min_flops = 5.0e8;
                self.potrf_min_p = 768;
                self.sparse_min_nnz = 800_000;
                self.row_kernel_min_n = 75_000;
                self.prefer_gpu_for_f64_factorization = true;
            }
            GpuCapability::Cuda => {}
            GpuCapability::Unknown => {
                self.accel_policy = AccelPolicy::CpuOnly;
            }
        }
    }

    #[must_use]
    pub fn decide(&self, op: Operation, device_available: bool) -> OperationDecision {
        if self.accel_policy == AccelPolicy::CpuOnly || !device_available {
            return OperationDecision::Cpu;
        }
        if self.accel_policy == AccelPolicy::GpuOnly {
            return OperationDecision::Gpu;
        }
        let flops = op.estimated_flops();
        let resident_bonus = if op.inputs_resident() { 0.25 } else { 1.0 };
        let threshold = match op {
            Operation::Gemm { .. } => self.gemm_min_flops,
            Operation::Gemv { .. } => self.gemv_min_flops,
            Operation::XtDiagX { .. } => self.xtwx_min_flops,
            Operation::XtDiagY { .. } => self.xtwy_min_flops,
            Operation::JointHessian2x2 { .. } => self.joint_hessian_min_flops,
            Operation::Potrf { p, resident } => {
                if resident && p >= self.potrf_min_p {
                    return OperationDecision::Gpu;
                }
                return OperationDecision::Cpu;
            }
            Operation::SparseXtDiagX { nnz, pattern_reuse } => {
                if nnz >= self.sparse_min_nnz && pattern_reuse {
                    return OperationDecision::Gpu;
                }
                return OperationDecision::Cpu;
            }
            Operation::RowKernel { n, batched } => {
                if n >= self.row_kernel_min_n || (batched && n >= self.row_kernel_min_n / 4) {
                    return OperationDecision::Gpu;
                }
                return OperationDecision::Cpu;
            }
        };
        if flops >= threshold * resident_bonus {
            OperationDecision::Gpu
        } else {
            OperationDecision::Cpu
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Operation {
    Gemm {
        m: usize,
        n: usize,
        k: usize,
        resident: bool,
    },
    Gemv {
        m: usize,
        k: usize,
        transposed: bool,
        resident: bool,
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
    Potrf {
        p: usize,
        resident: bool,
    },
    SparseXtDiagX {
        nnz: usize,
        pattern_reuse: bool,
    },
    RowKernel {
        n: usize,
        batched: bool,
    },
}

impl Operation {
    #[must_use]
    pub fn estimated_flops(self) -> f64 {
        match self {
            Self::Gemm { m, n, k, .. } => 2.0 * m as f64 * n as f64 * k as f64,
            Self::Gemv { m, k, .. } => 2.0 * m as f64 * k as f64,
            Self::XtDiagX { rows, cols, .. } => 2.0 * rows as f64 * cols as f64 * cols as f64,
            Self::XtDiagY {
                rows,
                x_cols,
                y_cols,
                ..
            } => 2.0 * rows as f64 * x_cols as f64 * y_cols as f64,
            Self::JointHessian2x2 {
                rows,
                a_cols,
                b_cols,
                ..
            } => 2.0 * rows as f64 * (a_cols * a_cols + a_cols * b_cols + b_cols * b_cols) as f64,
            Self::Potrf { p, .. } => (p as f64).powi(3) / 3.0,
            Self::SparseXtDiagX { nnz, .. } => nnz as f64,
            Self::RowKernel { n, batched } => {
                if batched {
                    200.0 * n as f64
                } else {
                    50.0 * n as f64
                }
            }
        }
    }

    #[must_use]
    pub const fn inputs_resident(self) -> bool {
        match self {
            Self::Gemm { resident, .. }
            | Self::Gemv { resident, .. }
            | Self::XtDiagX { resident, .. }
            | Self::XtDiagY { resident, .. }
            | Self::JointHessian2x2 { resident, .. }
            | Self::Potrf { resident, .. } => resident,
            Self::SparseXtDiagX { .. } | Self::RowKernel { .. } => false,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OperationDecision {
    Cpu,
    Gpu,
}

fn env_f64(name: &str, default: f64) -> f64 {
    env::var(name)
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(default)
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_bool(name: &str) -> bool {
    matches!(
        env::var(name)
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_policy_keeps_tiny_gemv_on_cpu() {
        let policy = GpuDispatchPolicy::default();
        let op = Operation::Gemv {
            m: 100,
            k: 10,
            transposed: false,
            resident: false,
        };
        assert_eq!(policy.decide(op, true), OperationDecision::Cpu);
    }

    #[test]
    fn auto_policy_sends_large_resident_xtwx_to_gpu() {
        let policy = GpuDispatchPolicy::default();
        let op = Operation::XtDiagX {
            rows: 1_000_000,
            cols: 128,
            resident: true,
        };
        assert_eq!(policy.decide(op, true), OperationDecision::Gpu);
    }
}
