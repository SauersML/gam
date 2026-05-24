use serde::{Deserialize, Serialize};

/// Capability flags derived from CUDA device attributes.  Compute capability is
/// only a hint; runtime probing fills the concrete memory and SM attributes.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct GpuCapability {
    pub compute_major: i32,
    pub compute_minor: i32,
    pub has_tensor_cores: bool,
    pub has_fp64_tensor_cores: bool,
    pub has_async_copy: bool,
    pub cluster_launch: bool,
    pub tma: bool,
    pub min_warp_size: i32,
}

impl GpuCapability {
    #[must_use]
    pub fn from_compute_capability(major: i32, minor: i32) -> Self {
        Self {
            compute_major: major,
            compute_minor: minor,
            has_tensor_cores: major >= 7,
            has_fp64_tensor_cores: major >= 8,
            has_async_copy: major >= 8,
            cluster_launch: major >= 9,
            tma: major >= 9,
            min_warp_size: 32,
        }
    }
}

/// Device information consumed by dispatch policy and profile logs.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GpuDeviceInfo {
    pub ordinal: usize,
    pub name: String,
    pub capability: GpuCapability,
    pub sm_count: i32,
    pub max_threads_per_sm: i32,
    pub shared_mem_per_block: usize,
    pub l2_cache_bytes: usize,
    pub total_mem_bytes: usize,
    pub free_mem_bytes: usize,
    pub ecc_enabled: bool,
    pub integrated: bool,
    pub mig_mode: bool,
}

impl GpuDeviceInfo {
    #[must_use]
    pub fn score(&self) -> f64 {
        let free_gib = self.free_mem_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
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
        f64::from(self.sm_count.max(0)) + free_gib * 4.0 + fp64_bonus + async_bonus
    }
}
