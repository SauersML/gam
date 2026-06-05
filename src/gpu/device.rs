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
    pub fn fingerprint(&self) -> crate::cache::Fingerprint {
        let mut fp = crate::cache::Fingerprinter::new();
        fp.absorb_str(b"gpu-policy-calibration-schema", "v1");
        fp.absorb_str(b"name", &self.name);
        fp.absorb_u64(b"ordinal", self.ordinal as u64);
        fp.absorb_u64(
            b"compute-major",
            u64::try_from(self.capability.compute_major.max(0)).unwrap_or(0),
        );
        fp.absorb_u64(
            b"compute-minor",
            u64::try_from(self.capability.compute_minor.max(0)).unwrap_or(0),
        );
        fp.absorb_u64(
            b"sm-count",
            u64::try_from(self.sm_count.max(0)).unwrap_or(0),
        );
        fp.absorb_u64(
            b"threads-per-sm",
            u64::try_from(self.max_threads_per_sm.max(0)).unwrap_or(0),
        );
        fp.absorb_u64(
            b"shared-mem-per-block",
            self.max_shared_mem_per_block as u64,
        );
        fp.absorb_u64(b"l2-cache-bytes", self.l2_cache_bytes as u64);
        fp.absorb_u64(b"total-mem-bytes", self.total_mem_bytes as u64);
        fp.absorb_u64(b"ecc-enabled", u64::from(self.ecc_enabled));
        fp.absorb_u64(b"integrated", u64::from(self.integrated));
        fp.absorb_u64(b"mig-mode", u64::from(self.mig_mode));
        fp.finalize()
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
