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
    /// FP64 cores per SM by compute-capability major (these numbers come
    /// straight from NVIDIA's published architecture docs, not a bucket):
    ///
    /// | major | FP64 cores / SM |
    /// |-------|-----------------|
    /// |  6.x  |  32 (P100) or 4 (consumer Pascal) |
    /// |  7.x  |  32 (V100) or 2 (Turing consumer) |
    /// |  8.x  |  32 (A100) or 2 (Ampere consumer) |
    /// |  9.x+ |  64 (H100/B100) |
    ///
    /// The fast/slow split inside each generation maps to data-center vs
    /// consumer parts and isn't observable from the driver API alone; we
    /// pick the data-center figure so consumer cards are treated more
    /// optimistically than they deserve — the dispatch policy's threshold
    /// pads for that with a CPU/GPU speedup floor.
    pub fn peak_fp64_gflops(&self) -> f64 {
        let cores_per_sm = if self.compute_capability_major >= 9 {
            64.0
        } else {
            32.0
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
        let consumer = make(7, 40, 8);
        let datacenter = make(9, 132, 80);
        assert!(datacenter.score() > consumer.score());
    }
}
