//! Host-visible description of a CUDA device.

use std::fmt;

/// Snapshot of a CUDA device chosen for dispatch.
///
/// Fields are populated from the driver API at probe time and never refreshed.
/// Backends that care about live free memory should query their own driver
/// handle rather than reading this struct.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuDeviceInfo {
    pub ordinal: usize,
    pub name: String,
    pub compute_capability_major: i32,
    pub compute_capability_minor: i32,
    /// Number of streaming multiprocessors (SMs / SMXs / GPCs depending on
    /// the architecture marketing term). Drives the peak-throughput estimate
    /// in [`super::policy::DispatchPolicy::for_device`].
    pub sm_count: i32,
    pub total_memory_bytes: usize,
}

impl GpuDeviceInfo {
    /// Effective peak FP64 throughput, GFLOPS, at a nominal 1.5 GHz boost.
    ///
    /// FP64 cores per SM by compute-capability major and device family:
    ///
    /// | major | FP64 cores / SM |
    /// |-------|-----------------|
    /// |  6.x  |  32 (P100) or 4 (consumer Pascal) |
    /// |  7.x  |  32 (V100) or 2 (T4 / Turing consumer) |
    /// |  8.x  |  32 (A100) or 2 (L4 / A10 / Ampere consumer) |
    /// |  9.x+ |  64 (H100/B100) |
    ///
    /// CUDA's driver API does not expose FP64 ratio directly, so device-name
    /// matching is used for the datacenter parts with high FP64 throughput.
    /// Everything else is treated as low-ratio FP64 hardware, which keeps
    /// T4/L4/RTX-class devices from accepting work that only A100/H100-class
    /// cards can amortize efficiently.
    pub fn peak_fp64_gflops(&self) -> f64 {
        let cores_per_sm = if self.compute_capability_major >= 9 {
            64.0
        } else if self.has_datacenter_fp64_ratio() {
            32.0
        } else {
            2.0
        };
        let boost_ghz = 1.5_f64;
        let muladd_flops = 2.0_f64;
        self.sm_count.max(1) as f64 * cores_per_sm * boost_ghz * muladd_flops
    }

    /// Heuristic score used to pick the "best" device when multiple are
    /// present. Larger is better; the score is purely ordinal.
    pub fn score(&self) -> f64 {
        let memory_gib = self.total_memory_bytes as f64 / 1_073_741_824.0;
        self.peak_fp64_gflops() + memory_gib * 4.0
    }

    /// Conservative per-dispatch memory budget. CUDA allocations are made in
    /// one shot per operation, so leave room for the driver context, library
    /// workspaces, and other process allocations instead of planning against
    /// the advertised full device memory.
    pub fn dispatch_memory_budget_bytes(&self) -> usize {
        let half = self.total_memory_bytes / 2;
        let seventy_percent = self.total_memory_bytes.saturating_mul(7) / 10;
        let reserve = 512 * 1024 * 1024;
        if self.total_memory_bytes > reserve.saturating_mul(2) {
            seventy_percent.min(self.total_memory_bytes.saturating_sub(reserve))
        } else {
            half
        }
    }

    fn has_datacenter_fp64_ratio(&self) -> bool {
        let name = self.name.to_ascii_uppercase();
        ["A100", "A800", "H100", "H200", "B100", "B200", "V100", "P100"]
            .iter()
            .any(|needle| name.contains(needle))
    }
}

impl fmt::Display for GpuDeviceInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "device {} '{}' sm_{}{} {} SMs {:.1} GiB",
            self.ordinal,
            self.name,
            self.compute_capability_major,
            self.compute_capability_minor,
            self.sm_count,
            self.total_memory_bytes as f64 / 1_073_741_824.0,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make(major: i32, sms: i32, mem_gib: usize) -> GpuDeviceInfo {
        GpuDeviceInfo {
            ordinal: 0,
            name: "test".to_string(),
            compute_capability_major: major,
            compute_capability_minor: 0,
            sm_count: sms,
            total_memory_bytes: mem_gib * 1024 * 1024 * 1024,
        }
    }

    #[test]
    fn peak_fp64_scales_with_sm_count() {
        let small = make(8, 40, 16);
        let large = make(8, 132, 80);
        assert!(large.peak_fp64_gflops() > small.peak_fp64_gflops() * 3.0);
    }

    #[test]
    fn hopper_per_sm_throughput_exceeds_ampere() {
        let a100_like = make(8, 108, 80);
        let h100_like = make(9, 132, 80);
        // h100 doubles FP64 cores/SM on top of more SMs; throughput should be
        // ≥ ~2.4× even before the boost-clock advantage.
        assert!(h100_like.peak_fp64_gflops() / a100_like.peak_fp64_gflops() > 2.0);
    }

    #[test]
    fn score_prefers_higher_compute_and_memory() {
        let mut consumer = make(7, 40, 8);
        consumer.name = "T4".to_string();
        let mut datacenter = make(9, 132, 80);
        datacenter.name = "H100".to_string();
        assert!(datacenter.score() > consumer.score());
    }

    #[test]
    fn low_ratio_devices_get_conservative_fp64_estimate() {
        let mut t4 = make(7, 40, 16);
        t4.name = "Tesla T4".to_string();
        let mut v100 = make(7, 80, 32);
        v100.name = "Tesla V100".to_string();
        assert!(v100.peak_fp64_gflops() > t4.peak_fp64_gflops() * 10.0);
    }
}
