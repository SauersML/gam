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
