use super::device::{GpuCapability, GpuDeviceInfo};

/// User-selected acceleration policy, controlled by `GAM_GPU`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AccelPolicy {
    Auto,
    CpuOnly,
    GpuOnly,
}

/// Shape descriptor for operation-level CPU/GPU dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Operation {
    Gemm {
        m: usize,
        n: usize,
        k: usize,
        resident: bool,
    },
    Gemv {
        rows: usize,
        cols: usize,
        transpose: bool,
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
        cols_a: usize,
        cols_b: usize,
        resident: bool,
    },
    Potrf {
        dim: usize,
        resident: bool,
    },
    SparseXtDiagX {
        rows: usize,
        cols: usize,
        nnz: usize,
        resident: bool,
    },
    RowKernel {
        rows: usize,
        lanes: usize,
        resident: bool,
    },
}

/// Per-kernel thresholds. These are conservative defaults that are refined by
/// future calibration data without changing public fitting APIs.
#[derive(Clone, Debug)]
pub struct GpuDispatchPolicy {
    pub accel_policy: AccelPolicy,
    pub gemm_min_flops: u128,
    pub gemv_min_flops: u128,
    pub xtwx_min_flops: u128,
    pub potrf_min_p: usize,
    pub sparse_min_nnz: usize,
    pub row_kernel_min_n: usize,
    pub keep_design_resident: bool,
    pub prefer_gpu_for_f64_factorization: bool,
}

impl GpuDispatchPolicy {
    #[inline]
    pub fn for_device(accel_policy: AccelPolicy, device: Option<&GpuDeviceInfo>) -> Self {
        let mut policy = Self {
            accel_policy,
            gemm_min_flops: 256_000_000,
            gemv_min_flops: 32_000_000,
            xtwx_min_flops: 512_000_000,
            potrf_min_p: 2_048,
            sparse_min_nnz: 2_000_000,
            row_kernel_min_n: 250_000,
            keep_design_resident: device.is_some(),
            prefer_gpu_for_f64_factorization: false,
        };
        if let Some(info) = device {
            match info.capability {
                GpuCapability::HighEndDatacenter => {
                    policy.gemm_min_flops = 32_000_000;
                    policy.gemv_min_flops = 8_000_000;
                    policy.xtwx_min_flops = 64_000_000;
                    policy.potrf_min_p = 768;
                    policy.sparse_min_nnz = 250_000;
                    policy.row_kernel_min_n = 25_000;
                    policy.prefer_gpu_for_f64_factorization = true;
                }
                GpuCapability::Datacenter => {
                    policy.gemm_min_flops = 64_000_000;
                    policy.gemv_min_flops = 16_000_000;
                    policy.xtwx_min_flops = 128_000_000;
                    policy.potrf_min_p = 1_024;
                    policy.sparse_min_nnz = 500_000;
                    policy.row_kernel_min_n = 75_000;
                }
                GpuCapability::Mainstream => {
                    policy.gemm_min_flops = 128_000_000;
                    policy.xtwx_min_flops = 256_000_000;
                    policy.potrf_min_p = 1_536;
                    policy.sparse_min_nnz = 1_000_000;
                    policy.row_kernel_min_n = 125_000;
                }
                GpuCapability::Legacy => {}
            }
        }
        policy.apply_env_overrides()
    }

    #[inline]
    pub fn should_use_gpu(&self, operation: Operation, cuda_available: bool) -> bool {
        if matches!(self.accel_policy, AccelPolicy::CpuOnly) || !cuda_available {
            return false;
        }
        let force = matches!(self.accel_policy, AccelPolicy::GpuOnly);
        let resident_bonus = match operation {
            Operation::Gemm { resident, .. }
            | Operation::Gemv { resident, .. }
            | Operation::XtDiagX { resident, .. }
            | Operation::XtDiagY { resident, .. }
            | Operation::JointHessian2x2 { resident, .. }
            | Operation::Potrf { resident, .. }
            | Operation::SparseXtDiagX { resident, .. }
            | Operation::RowKernel { resident, .. } => resident,
        };
        if force && resident_bonus {
            return true;
        }
        match operation {
            Operation::Gemm { m, n, k, .. } => {
                2_u128 * m as u128 * n as u128 * k as u128 >= self.gemm_min_flops
            }
            Operation::Gemv { rows, cols, .. } => {
                2_u128 * rows as u128 * cols as u128 >= self.gemv_min_flops
            }
            Operation::XtDiagX { rows, cols, .. } => {
                2_u128 * rows as u128 * cols as u128 * cols as u128 >= self.xtwx_min_flops
            }
            Operation::XtDiagY {
                rows,
                x_cols,
                y_cols,
                ..
            } => 2_u128 * rows as u128 * x_cols as u128 * y_cols as u128 >= self.xtwx_min_flops,
            Operation::JointHessian2x2 {
                rows,
                cols_a,
                cols_b,
                ..
            } => {
                let p = cols_a.saturating_add(cols_b) as u128;
                2_u128 * rows as u128 * p * p >= self.xtwx_min_flops
            }
            Operation::Potrf { dim, .. } => dim >= self.potrf_min_p,
            Operation::SparseXtDiagX { nnz, .. } => nnz >= self.sparse_min_nnz,
            Operation::RowKernel { rows, lanes, .. } => {
                rows.saturating_mul(lanes) >= self.row_kernel_min_n
            }
        }
    }

    fn apply_env_overrides(mut self) -> Self {
        if let Ok(value) = std::env::var("GAM_GPU_MIN_N") {
            if let Ok(parsed) = value.parse::<usize>() {
                self.row_kernel_min_n = parsed;
            }
        }
        self
    }
}

#[inline]
pub fn accel_policy_from_env() -> AccelPolicy {
    match std::env::var("GAM_GPU")
        .unwrap_or_else(|_| "auto".to_string())
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "off" | "0" | "false" | "cpu" | "cpuonly" | "cpu-only" => AccelPolicy::CpuOnly,
        "force" | "on" | "1" | "true" | "gpu" | "gpuonly" | "gpu-only" => AccelPolicy::GpuOnly,
        _ => AccelPolicy::Auto,
    }
}
