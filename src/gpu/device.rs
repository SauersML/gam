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
    /// Maximum resident threads per multiprocessor reported by the CUDA
    /// driver. Used as a generic throughput proxy without product-name
    /// categorization.
    pub max_threads_per_multiprocessor: i32,
    /// Device core clock in kHz, as reported by the CUDA driver.
    pub clock_rate_khz: i32,
    /// Ratio of single-precision to double-precision throughput reported by
    /// `CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO`.
    pub single_to_double_precision_perf_ratio: i32,
    pub total_memory_bytes: usize,
}

impl GpuDeviceInfo {
    /// Generic effective FP64 throughput proxy, GFLOPS.
    ///
    /// This uses only driver-reported characteristics:
    /// `SMs × max resident threads/SM × clock × 2 FLOP/FMA ÷ SP:DP ratio`.
    /// It is deliberately a dispatch score rather than a marketing peak
    /// claim; the important property is that mixed fleets are weighted by
    /// actual CUDA attributes, not product names.
    pub fn peak_fp64_gflops(&self) -> f64 {
        let sm_count = self.sm_count.max(1) as f64;
        let threads_per_sm = self.max_threads_per_multiprocessor.max(1) as f64;
        let clock_ghz = (self.clock_rate_khz.max(1) as f64) / 1_000_000.0;
        let muladd_flops = 2.0_f64;
        let fp64_ratio = self.single_to_double_precision_perf_ratio.max(1) as f64;
        sm_count * threads_per_sm * clock_ghz * muladd_flops / fp64_ratio
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
        let reserve: usize = 512 * 1024 * 1024;
        if self.total_memory_bytes > reserve.saturating_mul(2) {
            seventy_percent.min(self.total_memory_bytes.saturating_sub(reserve))
        } else {
            half
        }
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
            max_threads_per_multiprocessor: 2048,
            clock_rate_khz: 1_500_000,
            single_to_double_precision_perf_ratio: 32,
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
        let mut smaller = make(8, 108, 80);
        let mut larger = make(9, 132, 80);
        smaller.single_to_double_precision_perf_ratio = 2;
        larger.single_to_double_precision_perf_ratio = 1;
        assert!(larger.peak_fp64_gflops() / smaller.peak_fp64_gflops() > 2.0);
    }

    #[test]
    fn score_prefers_higher_compute_and_memory() {
        let small = make(7, 40, 8);
        let large = make(9, 132, 80);
        assert!(large.score() > small.score());
    }

    #[test]
    fn driver_reported_fp64_ratio_controls_throughput_estimate() {
        let mut low_ratio = make(8, 80, 16);
        low_ratio.single_to_double_precision_perf_ratio = 2;
        let mut high_ratio = make(8, 80, 16);
        high_ratio.single_to_double_precision_perf_ratio = 64;
        assert!(low_ratio.peak_fp64_gflops() > high_ratio.peak_fp64_gflops() * 10.0);
    }
}
