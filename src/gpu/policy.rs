use crate::gpu::device::{GpuCapability, GpuDeviceInfo};

/// User-visible acceleration mode, parsed from `GAM_GPU`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AccelPolicy {
    Auto,
    CpuOnly,
    GpuOnly,
}

impl AccelPolicy {
    #[must_use]
    pub fn from_env_value(value: Option<&str>) -> Self {
        match value.unwrap_or("auto").trim().to_ascii_lowercase().as_str() {
            "off" | "0" | "false" | "cpu" | "cpu-only" => Self::CpuOnly,
            "force" | "1" | "true" | "gpu" | "gpu-only" => Self::GpuOnly,
            _ => Self::Auto,
        }
    }
}

/// Operation classes used by calibrated CPU/GPU dispatch.
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
        a_cols: usize,
        b_cols: usize,
        resident: bool,
    },
    DenseSpdSolve {
        dim: usize,
        rhs: usize,
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

impl Operation {
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Operation::Gemm { .. } => "gemm",
            Operation::Gemv {
                transpose: true, ..
            } => "gemv_t",
            Operation::Gemv { .. } => "gemv",
            Operation::XtDiagX { .. } => "xt_diag_x",
            Operation::XtDiagY { .. } => "xt_diag_y",
            Operation::JointHessian2x2 { .. } => "joint_hessian_2x2",
            Operation::DenseSpdSolve { .. } => "dense_spd_solve",
            Operation::SparseXtDiagX { .. } => "sparse_xt_diag_x",
            Operation::RowKernel { .. } => "row_kernel",
        }
    }

    #[must_use]
    pub fn estimated_flops(&self) -> f64 {
        match *self {
            Operation::Gemm { m, n, k, .. } => 2.0 * m as f64 * n as f64 * k as f64,
            Operation::Gemv { rows, cols, .. } => 2.0 * rows as f64 * cols as f64,
            Operation::XtDiagX { rows, cols, .. } => {
                rows as f64 * cols as f64 * (cols as f64 + 1.0)
            }
            Operation::XtDiagY {
                rows,
                x_cols,
                y_cols,
                ..
            } => 2.0 * rows as f64 * x_cols as f64 * y_cols as f64,
            Operation::JointHessian2x2 {
                rows,
                a_cols,
                b_cols,
                ..
            } => {
                let aa = a_cols * a_cols;
                let ab = a_cols * b_cols;
                let bb = b_cols * b_cols;
                2.0 * rows as f64 * (aa + ab + bb) as f64
            }
            Operation::DenseSpdSolve { dim, rhs, .. } => {
                (dim as f64).powi(3) / 3.0 + 2.0 * (dim as f64).powi(2) * rhs as f64
            }
            Operation::SparseXtDiagX { nnz, cols, .. } => 2.0 * nnz as f64 * cols.min(64) as f64,
            Operation::RowKernel { rows, lanes, .. } => rows as f64 * lanes.max(1) as f64 * 64.0,
        }
    }

    #[must_use]
    pub fn resident(&self) -> bool {
        match *self {
            Operation::Gemm { resident, .. }
            | Operation::Gemv { resident, .. }
            | Operation::XtDiagX { resident, .. }
            | Operation::XtDiagY { resident, .. }
            | Operation::JointHessian2x2 { resident, .. }
            | Operation::DenseSpdSolve { resident, .. }
            | Operation::SparseXtDiagX { resident, .. }
            | Operation::RowKernel { resident, .. } => resident,
        }
    }
}

/// Hardware-aware thresholds.  The defaults deliberately favor CPU unless the
/// operation is large or data is already resident, preventing PCIe-copy-only
/// regressions on small fits.
#[derive(Clone, Debug)]
pub struct GpuDispatchPolicy {
    pub accel: AccelPolicy,
    pub gemm_min_flops: f64,
    pub gemv_min_flops: f64,
    pub xtwx_min_flops: f64,
    pub potrf_min_p: usize,
    pub sparse_min_nnz: usize,
    pub row_kernel_min_n: usize,
    pub keep_design_resident: bool,
    pub prefer_gpu_for_f64_factorization: bool,
    pub memory_fraction: f64,
}

impl GpuDispatchPolicy {
    #[must_use]
    pub fn from_env_and_device(device: Option<&GpuDeviceInfo>) -> Self {
        let accel = AccelPolicy::from_env_value(std::env::var("GAM_GPU").ok().as_deref());
        let min_n = std::env::var("GAM_GPU_MIN_N")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(4096);
        let memory_fraction = std::env::var("GAM_GPU_MEM_FRACTION")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or(0.85)
            .clamp(0.05, 0.98);

        let mut policy = Self {
            accel,
            gemm_min_flops: 256_000_000.0,
            gemv_min_flops: 32_000_000.0,
            xtwx_min_flops: 384_000_000.0,
            potrf_min_p: 1536,
            sparse_min_nnz: 2_000_000,
            row_kernel_min_n: min_n,
            keep_design_resident: true,
            prefer_gpu_for_f64_factorization: false,
            memory_fraction,
        };

        if let Some(device) = device {
            match device.capability() {
                GpuCapability::HopperOrNewer => {
                    policy.gemm_min_flops = 64_000_000.0;
                    policy.gemv_min_flops = 12_000_000.0;
                    policy.xtwx_min_flops = 96_000_000.0;
                    policy.potrf_min_p = 768;
                    policy.sparse_min_nnz = 500_000;
                    policy.prefer_gpu_for_f64_factorization = true;
                }
                GpuCapability::Ampere => {
                    policy.gemm_min_flops = 128_000_000.0;
                    policy.gemv_min_flops = 20_000_000.0;
                    policy.xtwx_min_flops = 192_000_000.0;
                    policy.potrf_min_p = 1024;
                    policy.sparse_min_nnz = 1_000_000;
                }
                GpuCapability::TuringOrOlder | GpuCapability::Unknown => {}
            }
        }
        policy
    }

    #[must_use]
    pub fn should_try_gpu(&self, op: Operation, device_available: bool) -> bool {
        if self.accel == AccelPolicy::CpuOnly || !device_available {
            return false;
        }
        if self.accel == AccelPolicy::GpuOnly {
            return true;
        }
        let resident_bonus = if op.resident() { 0.25 } else { 1.0 };
        match op {
            Operation::Gemm { .. } => op.estimated_flops() >= self.gemm_min_flops * resident_bonus,
            Operation::Gemv { .. } => op.estimated_flops() >= self.gemv_min_flops * resident_bonus,
            Operation::XtDiagX { .. } => {
                op.estimated_flops() >= self.xtwx_min_flops * resident_bonus
            }
            Operation::XtDiagY { .. } => {
                op.estimated_flops() >= self.gemm_min_flops * resident_bonus
            }
            Operation::JointHessian2x2 { .. } => {
                op.estimated_flops() >= self.xtwx_min_flops * resident_bonus
            }
            Operation::DenseSpdSolve { dim, resident, .. } => {
                dim >= self.potrf_min_p || (resident && self.prefer_gpu_for_f64_factorization)
            }
            Operation::SparseXtDiagX { nnz, .. } => nnz >= self.sparse_min_nnz,
            Operation::RowKernel { rows, .. } => rows >= self.row_kernel_min_n,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::device::GpuDeviceInfo;

    fn device(cc: Option<(u32, u32)>) -> GpuDeviceInfo {
        GpuDeviceInfo {
            index: 0,
            name: "test".to_string(),
            uuid: None,
            compute_capability: cc,
            total_memory_bytes: Some(80 * 1024 * 1024 * 1024),
            free_memory_bytes: Some(70 * 1024 * 1024 * 1024),
        }
    }

    #[test]
    fn hopper_policy_lowers_dense_thresholds() {
        let unknown = GpuDispatchPolicy::from_env_and_device(None);
        let hopper_device = device(Some((9, 0)));
        let hopper = GpuDispatchPolicy::from_env_and_device(Some(&hopper_device));
        assert!(hopper.gemm_min_flops < unknown.gemm_min_flops);
        assert!(hopper.xtwx_min_flops < unknown.xtwx_min_flops);
        assert!(hopper.prefer_gpu_for_f64_factorization);
    }

    #[test]
    fn auto_policy_keeps_tiny_host_operations_on_cpu() {
        let policy = GpuDispatchPolicy::from_env_and_device(None);
        assert!(!policy.should_try_gpu(
            Operation::Gemv {
                rows: 32,
                cols: 8,
                transpose: false,
                resident: false,
            },
            true,
        ));
    }
}
