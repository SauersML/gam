#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuCapability {
    pub compute_major: i32,
    pub compute_minor: i32,
    pub has_tensor_cores: bool,
    pub has_fp64_tensor_cores: bool,
    pub has_async_copy: bool,
    pub has_cluster_launch: bool,
    pub has_tma: bool,
    pub min_warp_size: i32,
}

impl GpuCapability {
    pub const fn from_compute_capability(major: i32, minor: i32) -> Self {
        Self {
            compute_major: major,
            compute_minor: minor,
            has_tensor_cores: major >= 7,
            has_fp64_tensor_cores: major >= 8,
            has_async_copy: major >= 8,
            has_cluster_launch: major >= 9,
            has_tma: major >= 9,
            min_warp_size: 32,
        }
    }

    /// NVRTC `--gpu-architecture` virtual-arch string for this device's compute
    /// capability (e.g. `compute_80` for an A100 `8.0`).
    ///
    /// Critical for NVRTC correctness, not just performance: with no
    /// `--gpu-architecture`, NVRTC defaults to a virtual arch below `sm_60`,
    /// where the `atomicAdd(double*, double)` overload (added in compute
    /// capability 6.0) does not exist. A kernel source using `double` atomics
    /// (the SAE arrow/Schur PCG kernels do) then fails to compile, the module
    /// load Errs, and the whole device path silently falls back to the CPU.
    /// Keying the arch to the real device capability admits those kernels.
    ///
    /// Returns a `&'static str` because `cudarc`'s `CompileOptions::arch` is
    /// `Option<&'static str>`. Unknown/future capabilities round DOWN to the
    /// nearest known major to stay valid for the installed NVRTC, never up
    /// (an arch newer than the toolkit knows would itself fail to compile).
    #[must_use]
    pub const fn nvrtc_arch(&self) -> &'static str {
        match (self.compute_major, self.compute_minor) {
            // Newer-than-known majors round DOWN to compute_90 to stay valid for
            // the installed NVRTC (an arch the toolkit doesn't know would itself
            // fail to compile); refine when the toolkit/cudarc gain the arch.
            (major, _) if major >= 9 => "compute_90",
            (8, 9) => "compute_89",
            (8, 6) => "compute_86",
            (8, _) => "compute_80",
            (7, 5) => "compute_75",
            (7, _) => "compute_70",
            // 6.x and anything below: pin the lowest arch that still has the
            // `double` atomicAdd so a 6.x-built toolkit accepts the source. gam
            // requires CC >= 6.0 in practice for the double-atomic kernels.
            _ => "compute_60",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuDeviceInfo {
    pub ordinal: usize,
    pub name: String,
    pub capability: GpuCapability,
    pub sm_count: i32,
    pub max_threads_per_sm: i32,
    pub max_shared_mem_per_block: usize,
    pub l2_cache_bytes: usize,
    pub total_mem_bytes: usize,
    pub free_mem_bytes: usize,
    pub ecc_enabled: bool,
    pub integrated: bool,
    pub mig_mode: bool,
}

impl GpuDeviceInfo {
    /// Fraction of a device's *total* VRAM that any single dispatch is allowed
    /// to budget against. The per-device budget is `min(free, total ·
    /// MEMORY_BUDGET_TOTAL_FRACTION)`: free memory is the hard ceiling, but we
    /// cap at half of *total* so that even on a freshly idle device we leave
    /// headroom for the driver context, cuBLAS/cuSOLVER workspaces, and a
    /// second concurrent tile from the multi-GPU pool. Denominator `2` ⇒ half.
    const MEMORY_BUDGET_TOTAL_DIVISOR: usize = 2;

    /// Per-device byte budget a dispatch may size its buffers against:
    /// `min(free_mem, total_mem / MEMORY_BUDGET_TOTAL_DIVISOR)`. Single source
    /// of truth for both the primary-device budget (`device_runtime::probe`) and the
    /// per-ordinal pool budget (`GpuRuntime::memory_budget_for`).
    #[must_use]
    pub const fn memory_budget_bytes(&self) -> usize {
        let half_total = self.total_mem_bytes / Self::MEMORY_BUDGET_TOTAL_DIVISOR;
        if self.free_mem_bytes < half_total {
            self.free_mem_bytes
        } else {
            half_total
        }
    }

    pub fn score(&self) -> f64 {
        let fp64_bonus = if self.capability.has_fp64_tensor_cores {
            100.0
        } else {
            0.0
        };
        let async_bonus = if self.capability.has_async_copy {
            50.0
        } else {
            0.0
        };
        f64::from(self.sm_count)
            + (self.free_mem_bytes as f64 / 1_073_741_824.0) * 4.0
            + fp64_bonus
            + async_bonus
    }
}

#[cfg(test)]
mod nvrtc_arch_tests {
    use super::GpuCapability;

    fn arch_for(major: i32, minor: i32) -> &'static str {
        GpuCapability::from_compute_capability(major, minor).nvrtc_arch()
    }

    /// Parse the numeric `NN` out of an NVRTC `compute_NN` virtual-arch string.
    fn arch_number(arch: &str) -> i32 {
        arch.strip_prefix("compute_")
            .and_then(|n| n.parse::<i32>().ok())
            .unwrap_or_else(|| panic!("nvrtc_arch must be a `compute_NN` string, got {arch}"))
    }

    /// #1551 — the exact device-capability → NVRTC virtual-arch mapping the SAE
    /// double-atomic PCG kernels rely on. A wrong arch here re-introduces the
    /// "GPU 0%" silent CPU fallback (NVRTC rejecting `atomicAdd(double*,double)`).
    #[test]
    fn nvrtc_arch_maps_known_capabilities() {
        assert_eq!(arch_for(9, 0), "compute_90"); // Hopper H100
        assert_eq!(arch_for(8, 9), "compute_89"); // Ada L4 / L40
        assert_eq!(arch_for(8, 6), "compute_86"); // Ampere A10G / 30xx
        assert_eq!(arch_for(8, 0), "compute_80"); // Ampere A100
        assert_eq!(arch_for(7, 5), "compute_75"); // Turing T4 (the lead's box)
        assert_eq!(arch_for(7, 0), "compute_70"); // Volta V100
        assert_eq!(arch_for(6, 0), "compute_60"); // Pascal P100
        // Minor variants round to the right major bucket.
        assert_eq!(arch_for(8, 7), "compute_80"); // Orin -> 8.x bucket
        assert_eq!(arch_for(7, 2), "compute_70"); // Xavier -> 7.x bucket
    }

    /// #1551 CRITICAL INVARIANT: every capability gam supports (CC >= 6.0) must
    /// map to a virtual arch >= `compute_60`, the lowest arch that provides the
    /// `atomicAdd(double*, double)` overload (added in CC 6.0). Below it the SAE
    /// matvec kernels fail to NVRTC-compile and the device path silently declines.
    #[test]
    fn nvrtc_arch_never_below_double_atomic_floor() {
        for &(major, minor) in &[
            (6, 0),
            (6, 1),
            (7, 0),
            (7, 5),
            (8, 0),
            (8, 6),
            (8, 9),
            (9, 0),
        ] {
            let n = arch_number(arch_for(major, minor));
            assert!(
                n >= 60,
                "CC {major}.{minor} mapped to compute_{n}, below the double-atomicAdd \
                 floor compute_60 — the SAE device PCG would silently fall back to CPU"
            );
        }
    }

    /// Newer-than-known majors must round DOWN to a toolkit-valid arch
    /// (`compute_90`), never up to an arch the installed NVRTC cannot target
    /// (which would itself fail to compile and decline the device path).
    #[test]
    fn nvrtc_arch_future_capabilities_round_down_to_known() {
        assert_eq!(arch_for(10, 0), "compute_90");
        assert_eq!(arch_for(12, 3), "compute_90");
        // And it stays a valid, double-atomic-capable arch.
        assert!(arch_number(arch_for(10, 0)) >= 60);
    }

    /// Sub-6.0 / unknown-low capabilities pin to `compute_60` — the lowest arch
    /// that still carries the double atomicAdd, so a CC6-era toolkit accepts the
    /// kernel source rather than declining.
    #[test]
    fn nvrtc_arch_below_floor_pins_to_compute_60() {
        assert_eq!(arch_for(5, 0), "compute_60");
        assert_eq!(arch_for(3, 5), "compute_60");
    }
}
