//! Workload-size thresholds for GPU dispatch decisions.
//!
//! The policy is intentionally derived from the selected device and never
//! influenced by environment variables or command-line flags: routing is a
//! function of `(workload-shape, device-capability)` only. CPU-only builds
//! see the same defaults; they are simply never consulted.

use super::device::{GpuCapability, GpuDeviceInfo};

/// Per-operation minimum workload sizes the cuBLAS backend needs to clear
/// before GPU dispatch is preferred over the in-process CPU path.
///
/// These thresholds reflect the cost of host↔device transfers plus kernel
/// launch overhead; below them the CPU path is faster end-to-end. The
/// numbers are calibrated against PCIe Gen4 transfer rates and tensor-core
/// SMs — Ampere/Hopper lowers them, Turing raises them.
#[derive(Clone, Debug)]
pub struct DispatchPolicy {
    /// Minimum row count `n` to consider dense `Xᵀ diag(W) X` on the device.
    pub xtwx_min_rows: usize,
    /// Minimum estimated FLOPs to dispatch a dense GEMM/GEMV to the device.
    pub gemm_min_flops: u64,
    /// Minimum estimated FLOPs to dispatch dense GEMV to the device.
    pub gemv_min_flops: u64,
}

impl DispatchPolicy {
    /// Defaults are tuned for the Mainstream (Volta/Turing) bucket.
    fn baseline() -> Self {
        Self {
            xtwx_min_rows: 8_192,
            gemm_min_flops: 300_000_000,
            gemv_min_flops: 80_000_000,
        }
    }

    /// Build a dispatch policy from the selected device (`None` = CPU-only).
    ///
    /// The CPU-only case still returns sensible defaults: the runtime simply
    /// never consults them because [`super::runtime::GpuRuntime::is_available`]
    /// reports false.
    pub fn for_device(device: Option<&GpuDeviceInfo>) -> Self {
        let mut p = Self::baseline();
        let Some(device) = device else { return p };
        match device.capability {
            GpuCapability::Legacy => {
                p.xtwx_min_rows = 65_536;
                p.gemm_min_flops = 512 * 1024 * 1024;
                p.gemv_min_flops = 256 * 1024 * 1024;
            }
            GpuCapability::Mainstream => {
                // baseline values
            }
            GpuCapability::Datacenter => {
                p.xtwx_min_rows = 2_048;
                p.gemm_min_flops = 16 * 1024 * 1024;
                p.gemv_min_flops = 8 * 1024 * 1024;
            }
            GpuCapability::HighEndDatacenter => {
                p.xtwx_min_rows = 1_024;
                p.gemm_min_flops = 4 * 1024 * 1024;
                p.gemv_min_flops = 2 * 1024 * 1024;
            }
        }
        p
    }

    /// Should a dense `Xᵀ diag(w) Y` route to the device?
    pub fn route_xt_diag_y(&self, rows: usize, lhs_cols: usize, rhs_cols: usize) -> bool {
        let flops = (rows as u64)
            .saturating_mul(lhs_cols as u64)
            .saturating_mul(rhs_cols.max(1) as u64)
            .saturating_mul(2);
        rows >= self.xtwx_min_rows && flops >= self.gemm_min_flops
    }

    /// Should a dense GEMM route to the device?
    pub fn route_gemm(&self, m: usize, n: usize, k: usize) -> bool {
        let flops = (m as u64)
            .saturating_mul(n as u64)
            .saturating_mul(k.max(1) as u64)
            .saturating_mul(2);
        flops >= self.gemm_min_flops
    }

    /// Should a dense GEMV route to the device?
    pub fn route_gemv(&self, rows: usize, cols: usize) -> bool {
        rows as u64 * cols as u64 * 2 >= self.gemv_min_flops
    }
}

impl Default for DispatchPolicy {
    fn default() -> Self {
        Self::baseline()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device(cap: GpuCapability) -> GpuDeviceInfo {
        GpuDeviceInfo {
            ordinal: 0,
            name: "test".to_string(),
            compute_capability_major: 8,
            compute_capability_minor: 0,
            total_memory_bytes: 16 * 1024 * 1024 * 1024,
            capability: cap,
        }
    }

    #[test]
    fn newer_arch_lowers_thresholds() {
        let legacy = DispatchPolicy::for_device(Some(&device(GpuCapability::Legacy)));
        let hopper = DispatchPolicy::for_device(Some(&device(GpuCapability::HighEndDatacenter)));
        assert!(hopper.xtwx_min_rows < legacy.xtwx_min_rows);
        assert!(hopper.gemm_min_flops < legacy.gemm_min_flops);
    }

    #[test]
    fn route_xt_diag_y_uses_shape_only() {
        let p = DispatchPolicy::for_device(Some(&device(GpuCapability::Datacenter)));
        assert!(!p.route_xt_diag_y(128, 16, 16));
        assert!(p.route_xt_diag_y(1_000_000, 512, 512));
    }

    #[test]
    fn route_gemm_and_gemv_use_separate_thresholds() {
        let p = DispatchPolicy::for_device(Some(&device(GpuCapability::Datacenter)));
        assert!(!p.route_gemm(128, 128, 128));
        assert!(p.route_gemm(2_048, 2_048, 8));
        assert!(!p.route_gemv(1_024, 1_024));
        assert!(p.route_gemv(8_192, 8_192));
    }
}
