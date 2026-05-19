//! Host-visible description of a CUDA device.

use std::fmt;

use super::calibration::DeviceCalibration;

/// Snapshot of a CUDA device chosen for dispatch.
///
/// Fields are populated from the driver API at probe time and never refreshed.
/// Backends that care about live free memory should query their own driver
/// handle rather than reading this struct.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuDeviceInfo {
    pub ordinal: usize,
    pub name: String,
    pub compute_capability_major: i32,
    pub compute_capability_minor: i32,
    /// Number of streaming multiprocessors (kept for the diagnostic banner —
    /// not consumed by the dispatch policy, which uses measured throughput).
    pub sm_count: i32,
    pub total_memory_bytes: usize,
    /// Measured throughput at probe time: sustained FP64 GFLOPS plus host↔
    /// device bandwidth. Every dispatch threshold reads from this struct, so
    /// the policy reflects what the silicon actually delivers under this
    /// driver/clock state rather than a tabulated architectural prior.
    pub calibration: DeviceCalibration,
}

impl Eq for GpuDeviceInfo {}

impl GpuDeviceInfo {
    /// Measured sustained FP64 throughput, GFLOPS.
    #[inline]
    pub fn peak_fp64_gflops(&self) -> f64 {
        self.calibration.fp64_gflops
    }

    /// Measured one-way host↔device bandwidth, GB/s. Uses the minimum of
    /// measured H2D and D2H since the dispatch round trip pays both.
    #[inline]
    pub fn pcie_gb_per_s(&self) -> f64 {
        self.calibration.h2d_gb_s.min(self.calibration.d2h_gb_s)
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

    fn calibration(fp64: f64) -> DeviceCalibration {
        DeviceCalibration {
            fp64_gflops: fp64,
            h2d_gb_s: 24.0,
            d2h_gb_s: 23.0,
        }
    }

    fn make(major: i32, minor: i32, sms: i32, mem_gib: usize, fp64: f64) -> GpuDeviceInfo {
        GpuDeviceInfo {
            ordinal: 0,
            name: "test".to_string(),
            compute_capability_major: major,
            compute_capability_minor: minor,
            sm_count: sms,
            total_memory_bytes: mem_gib * 1024 * 1024 * 1024,
            calibration: calibration(fp64),
        }
    }

    #[test]
    fn measured_throughput_drives_peak() {
        let slow = make(7, 5, 40, 16, 250.0);
        let fast = make(9, 0, 132, 80, 9000.0);
        assert!(fast.peak_fp64_gflops() > slow.peak_fp64_gflops() * 30.0);
    }

    #[test]
    fn score_prefers_higher_throughput_and_memory() {
        let small = make(7, 5, 40, 8, 200.0);
        let large = make(9, 0, 132, 80, 8000.0);
        assert!(large.score() > small.score());
    }

    #[test]
    fn pcie_uses_min_of_directions() {
        let dev = GpuDeviceInfo {
            ordinal: 0,
            name: "asym".to_string(),
            compute_capability_major: 8,
            compute_capability_minor: 0,
            sm_count: 80,
            total_memory_bytes: 40 * 1024 * 1024 * 1024,
            calibration: DeviceCalibration {
                fp64_gflops: 6000.0,
                h2d_gb_s: 23.0,
                d2h_gb_s: 25.5,
            },
        };
        assert!((dev.pcie_gb_per_s() - 23.0).abs() < 1e-9);
    }
}
