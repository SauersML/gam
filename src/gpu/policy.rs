//! Workload-size thresholds for GPU dispatch decisions.
//!
//! The policy is intentionally derived from the selected device and never
//! influenced by environment variables or command-line flags: routing is a
//! function of `(workload-shape, device-capability)` only. CPU-only builds
//! see the same defaults; they are simply never consulted.

use super::device::{GpuCapability, GpuDeviceInfo};

/// Per-operation minimum workload sizes a device backend needs to clear
/// before GPU dispatch is preferred over the in-process CPU/`faer` path.
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
    /// Minimum trailing dimension `p` for Cholesky on the device.
    pub potrf_min_p: usize,
    /// Minimum `p` for symmetric eigendecomposition on the device.
    pub syevd_min_p: usize,
    /// Minimum non-zeros for sparse SpMV on the device.
    pub sparse_min_nnz: usize,
    /// Minimum row count for fused IRLS-link evaluation on the device.
    pub fused_row_kernel_min_rows: usize,
    /// Minimum bytes for keeping the design matrix resident across iters.
    pub resident_design_min_bytes: usize,
    /// Minimum trace probes for Hutch++ trace estimators on the device.
    pub trace_min_probes: usize,
}

impl DispatchPolicy {
    /// Defaults are tuned for the Mainstream (Volta/Turing) bucket.
    fn baseline() -> Self {
        Self {
            xtwx_min_rows: 8_192,
            gemm_min_flops: 64 * 1024 * 1024,
            potrf_min_p: 512,
            syevd_min_p: 256,
            sparse_min_nnz: 1_000_000,
            fused_row_kernel_min_rows: 8_192,
            resident_design_min_bytes: 32 * 1024 * 1024,
            trace_min_probes: 8,
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
                p.potrf_min_p = 2_048;
                p.fused_row_kernel_min_rows = 65_536;
            }
            GpuCapability::Mainstream => {
                // baseline values
            }
            GpuCapability::Datacenter => {
                p.xtwx_min_rows = 2_048;
                p.gemm_min_flops = 16 * 1024 * 1024;
                p.potrf_min_p = 256;
                p.syevd_min_p = 128;
                p.fused_row_kernel_min_rows = 2_048;
                p.resident_design_min_bytes = 8 * 1024 * 1024;
                p.trace_min_probes = 4;
            }
            GpuCapability::HighEndDatacenter => {
                p.xtwx_min_rows = 1_024;
                p.gemm_min_flops = 4 * 1024 * 1024;
                p.potrf_min_p = 128;
                p.syevd_min_p = 64;
                p.fused_row_kernel_min_rows = 1_024;
                p.resident_design_min_bytes = 4 * 1024 * 1024;
                p.trace_min_probes = 2;
            }
        }
        p
    }

    /// Should a dense `XᵀWX` of shape `(n, p)` route to the device?
    pub fn route_xtwx(&self, n: usize, p: usize, design_resident: bool) -> bool {
        if !design_resident {
            return false;
        }
        let flops = (n as u64).saturating_mul(p as u64).saturating_mul(p as u64);
        n >= self.xtwx_min_rows && flops.saturating_mul(2) >= self.gemm_min_flops
    }

    /// Should a dense GEMV/GEMM of shape `(n, p)` route to the device?
    pub fn route_gemm(&self, n: usize, p: usize, q: usize, lhs_resident: bool) -> bool {
        let flops = (n as u64)
            .saturating_mul(p as u64)
            .saturating_mul(q.max(1) as u64)
            .saturating_mul(2);
        lhs_resident && flops >= self.gemm_min_flops
    }

    /// Should a dense Cholesky on a `p×p` matrix route to the device?
    pub fn route_potrf(&self, p: usize, resident: bool) -> bool {
        resident && p >= self.potrf_min_p
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
    fn route_xtwx_requires_resident_design() {
        let p = DispatchPolicy::for_device(Some(&device(GpuCapability::Datacenter)));
        assert!(!p.route_xtwx(1_000_000, 512, /*design_resident=*/ false));
        assert!(p.route_xtwx(1_000_000, 512, /*design_resident=*/ true));
    }
}
