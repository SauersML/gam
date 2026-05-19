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
    /// Device core clock in kHz, as reported by the CUDA driver.
    pub clock_rate_khz: i32,
    /// Ratio of single-precision to double-precision throughput reported by
    /// `CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO`, when the
    /// installed CUDA driver exposes it. The driver returns the integer
    /// `FP32_throughput / FP64_throughput` ratio, e.g. 32 on consumer Turing
    /// (T4: 64 FP32 cores per SM, 2 FP64 cores), 2 on A100, 1 on GP100.
    pub single_to_double_precision_perf_ratio: Option<i32>,
    pub total_memory_bytes: usize,
}

impl GpuDeviceInfo {
    /// Effective FP64 throughput proxy, GFLOPS.
    ///
    /// Derived from three hardware properties the driver exposes plus the
    /// canonical CUDA architectural FP32-cores-per-SM table:
    ///
    /// 1. **SM count** from `cuDeviceGetAttribute(MULTIPROCESSOR_COUNT)`.
    /// 2. **FP32 cores per SM** from the compute-capability lookup
    ///    [`fp32_cores_per_sm`] — this is the published NVIDIA mapping
    ///    (CUDA C++ Programming Guide, Appendix "Compute Capabilities"),
    ///    not a product-name table.
    /// 3. **Core clock** from `cuDeviceGetAttribute(CLOCK_RATE)`.
    /// 4. **FP32→FP64 ratio** from
    ///    `cuDeviceGetAttribute(SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO)`.
    ///
    /// Computes
    /// `cores_per_sm × SMs × clock_GHz × 2 FLOP/FMA / sp_to_dp_ratio`,
    /// matching vendor-published FP64 peaks within ~10% on every Kepler
    /// through Hopper card (T4 → ~254 GFLOPS, A100 → ~9.7 TFLOPS,
    /// H100 → ~33 TFLOPS).
    ///
    /// Earlier revisions used `max_threads_per_multiprocessor` here, which
    /// is the warp-scheduler resident-thread cap (1024 on Turing/Ampere
    /// consumer, 2048 on A100/Hopper). That value is unrelated to FP32
    /// ALU count and produced estimates ~16× too high on consumer cards,
    /// which in turn forced every dispatch threshold derived from the
    /// peak to over-route micro-kernels.
    pub fn peak_fp64_gflops(&self) -> f64 {
        let sm_count = self.sm_count.max(1) as f64;
        let cores_per_sm = fp32_cores_per_sm(
            self.compute_capability_major,
            self.compute_capability_minor,
        ) as f64;
        let clock_ghz = (self.clock_rate_khz.max(1) as f64) / 1_000_000.0;
        let muladd_flops = 2.0_f64;
        let fp32_gflops = sm_count * cores_per_sm * clock_ghz * muladd_flops;
        match self.single_to_double_precision_perf_ratio {
            Some(ratio) => fp32_gflops / ratio.max(1) as f64,
            // Driver didn't expose the ratio — fall back to an architectural
            // default. For datacenter parts (CC 6.0, 7.0, 8.0, 9.0) FP64 is
            // 1/2 of FP32; for everything else (consumer / Tegra) it's 1/32.
            None => {
                let default_ratio = default_sp_to_dp_ratio(
                    self.compute_capability_major,
                    self.compute_capability_minor,
                );
                fp32_gflops / default_ratio as f64
            }
        }
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

/// FP32 CUDA cores per SM as a function of compute capability.
///
/// Mirrors the table in the CUDA C++ Programming Guide, Appendix
/// "Compute Capabilities", section "Arithmetic Instructions". This is a
/// fixed architectural quantity, not a product lookup — every device with
/// the same `(major, minor)` shares the same FP32 ALU count per SM.
///
/// For compute capabilities released after this table was last updated, the
/// fallback returns 128 (the value used by every Pascal-or-newer consumer
/// architecture). Unknown future archs will run with a slightly conservative
/// throughput estimate but still benefit from the SM count and clock proxy.
pub(crate) fn fp32_cores_per_sm(major: i32, minor: i32) -> i32 {
    match (major, minor) {
        // Fermi
        (2, 0) => 32,
        (2, 1) => 48,
        // Kepler — GK104, GK110, GK210 all carry 192 FP32 cores per SMX.
        (3, _) => 192,
        // Maxwell
        (5, _) => 128,
        // Pascal — GP100 is the only "datacenter" Pascal with 64; consumer
        // Pascal (GP102/4/6/7/8) and Tegra (CC 6.2) use 128.
        (6, 0) => 64,
        (6, _) => 128,
        // Volta + Turing share a 64-FP32-per-SM layout.
        (7, _) => 64,
        // Ampere A100 / GA100 (CC 8.0) is 64; consumer Ampere (GA10x, CC 8.6)
        // and Ada Lovelace (CC 8.9) bumped to 128. Orin Tegra (CC 8.7) is 128.
        (8, 0) => 64,
        (8, _) => 128,
        // Hopper — H100 is 128 FP32 cores per SM.
        (9, _) => 128,
        // Blackwell (CC 10.x and 12.x consumer) — 128 FP32 cores per SM
        // per published NVIDIA architecture briefs.
        (10, _) | (12, _) => 128,
        // Unknown / future architecture — assume modern layout.
        _ => 128,
    }
}

/// Architectural default for the FP32:FP64 throughput ratio when the driver
/// doesn't expose it via `CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO`.
///
/// Returns 2 for datacenter parts that ship full-rate FP64 hardware
/// (CC 6.0 GP100, CC 7.0 Volta, CC 8.0 A100, CC 9.0 Hopper) and 32 for
/// everything else (consumer cards typically gate FP64 to a small number of
/// dedicated ALUs).
fn default_sp_to_dp_ratio(major: i32, minor: i32) -> i32 {
    match (major, minor) {
        (6, 0) | (7, 0) | (7, 2) | (8, 0) | (9, _) => 2,
        _ => 32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make(major: i32, minor: i32, sms: i32, mem_gib: usize) -> GpuDeviceInfo {
        GpuDeviceInfo {
            ordinal: 0,
            name: "test".to_string(),
            compute_capability_major: major,
            compute_capability_minor: minor,
            sm_count: sms,
            clock_rate_khz: 1_500_000,
            single_to_double_precision_perf_ratio: Some(32),
            total_memory_bytes: mem_gib * 1024 * 1024 * 1024,
        }
    }

    #[test]
    fn peak_fp64_scales_with_sm_count() {
        let small = make(7, 5, 40, 16);
        let large = make(7, 5, 132, 80);
        // Same compute capability ⇒ same cores/SM ⇒ throughput scales with
        // SM count alone (132/40 ≈ 3.3x).
        assert!(large.peak_fp64_gflops() > small.peak_fp64_gflops() * 3.0);
    }

    #[test]
    fn lower_fp64_ratio_increases_per_sm_throughput() {
        let mut smaller = make(8, 6, 108, 80);
        let mut larger = make(9, 0, 132, 80);
        smaller.single_to_double_precision_perf_ratio = Some(64);
        larger.single_to_double_precision_perf_ratio = Some(2);
        // Hopper SM has 128 FP32 cores @ 1:2 ratio = 64 FP64 GFLOP/SM/GHz.
        // GA10x SM has 128 FP32 cores @ 1:64 ratio = 2 FP64 GFLOP/SM/GHz.
        // 132/108 SMs × 64/2 ≈ 39x. Plenty above the 2x sanity bound.
        assert!(larger.peak_fp64_gflops() / smaller.peak_fp64_gflops() > 2.0);
    }

    #[test]
    fn score_prefers_higher_compute_and_memory() {
        let small = make(7, 5, 40, 8);
        let large = make(9, 0, 132, 80);
        assert!(large.score() > small.score());
    }

    #[test]
    fn driver_reported_fp64_ratio_controls_throughput_estimate() {
        let mut low_ratio = make(8, 0, 80, 16);
        low_ratio.single_to_double_precision_perf_ratio = Some(2);
        let mut high_ratio = make(8, 0, 80, 16);
        high_ratio.single_to_double_precision_perf_ratio = Some(64);
        assert!(low_ratio.peak_fp64_gflops() > high_ratio.peak_fp64_gflops() * 10.0);
    }

    #[test]
    fn turing_t4_estimate_is_in_range() {
        // T4: 40 SMs, 1.59 GHz boost, 1:32 SP:DP. Vendor spec is ~254
        // GFLOPS FP64. The proxy should land within a small factor.
        let t4 = GpuDeviceInfo {
            ordinal: 0,
            name: "Tesla T4".to_string(),
            compute_capability_major: 7,
            compute_capability_minor: 5,
            sm_count: 40,
            clock_rate_khz: 1_590_000,
            single_to_double_precision_perf_ratio: Some(32),
            total_memory_bytes: 16 * 1024 * 1024 * 1024,
        };
        let gflops = t4.peak_fp64_gflops();
        // Formula: 64 cores * 40 SMs * 1.59 GHz * 2 FLOP / 32 ratio = 254.4.
        assert!(
            (200.0..=320.0).contains(&gflops),
            "T4 FP64 estimate out of range: {gflops}"
        );
    }

    #[test]
    fn a100_estimate_is_in_range() {
        // A100 SXM4: 108 SMs, 1.41 GHz, 1:2 SP:DP. Vendor spec ~9.7 TFLOPS.
        let a100 = GpuDeviceInfo {
            ordinal: 0,
            name: "A100".to_string(),
            compute_capability_major: 8,
            compute_capability_minor: 0,
            sm_count: 108,
            clock_rate_khz: 1_410_000,
            single_to_double_precision_perf_ratio: Some(2),
            total_memory_bytes: 80 * 1024 * 1024 * 1024,
        };
        let gflops = a100.peak_fp64_gflops();
        // 64 * 108 * 1.41 * 2 / 2 = 9745.92 GFLOPS.
        assert!(
            (8_000.0..=11_000.0).contains(&gflops),
            "A100 FP64 estimate out of range: {gflops}"
        );
    }

    #[test]
    fn unknown_cc_falls_back_safely() {
        // Future architecture that this table doesn't know about — we should
        // still produce a positive, finite throughput estimate.
        let future = GpuDeviceInfo {
            ordinal: 0,
            name: "future".to_string(),
            compute_capability_major: 99,
            compute_capability_minor: 0,
            sm_count: 200,
            clock_rate_khz: 2_000_000,
            single_to_double_precision_perf_ratio: Some(2),
            total_memory_bytes: 96 * 1024 * 1024 * 1024,
        };
        let gflops = future.peak_fp64_gflops();
        assert!(gflops.is_finite() && gflops > 0.0);
    }

    #[test]
    fn driver_ratio_absent_uses_architectural_default() {
        // Two CC 8.0 (datacenter) devices: one with ratio reported, one with
        // None. They should produce equal throughput when the architectural
        // default matches the reported ratio.
        let mut reported = make(8, 0, 108, 80);
        reported.single_to_double_precision_perf_ratio = Some(2);
        let mut inferred = make(8, 0, 108, 80);
        inferred.single_to_double_precision_perf_ratio = None;
        let delta =
            (reported.peak_fp64_gflops() - inferred.peak_fp64_gflops()).abs();
        assert!(delta < 1.0, "expected matching throughput, delta = {delta}");
    }
}
