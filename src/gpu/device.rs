use std::fmt;

/// Coarse hardware tiers used by dispatch policy defaults.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuCapability {
    Unknown,
    Legacy,
    Datacenter,
    HopperOrNewer,
}

/// Runtime-discovered CUDA device metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuDeviceInfo {
    pub ordinal: usize,
    pub name: String,
    pub compute_capability_major: i32,
    pub compute_capability_minor: i32,
    pub total_memory_bytes: Option<usize>,
    pub free_memory_bytes: Option<usize>,
}

impl GpuDeviceInfo {
    #[must_use]
    pub fn compute_capability(&self) -> (i32, i32) {
        (self.compute_capability_major, self.compute_capability_minor)
    }

    #[must_use]
    pub fn capability_tier(&self) -> GpuCapability {
        match self.compute_capability() {
            (major, _) if major >= 9 => GpuCapability::HopperOrNewer,
            (major, _) if major >= 7 => GpuCapability::Datacenter,
            (major, _) if major > 0 => GpuCapability::Legacy,
            _ => GpuCapability::Unknown,
        }
    }

    #[must_use]
    pub fn score(&self) -> u128 {
        let memory = self.total_memory_bytes.unwrap_or(0) as u128;
        let cc = (self.compute_capability_major.max(0) as u128) * 100
            + self.compute_capability_minor.max(0) as u128;
        (cc << 64) | memory
    }
}

impl fmt::Display for GpuDeviceInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CUDA device {}: {} (sm_{}{})",
            self.ordinal, self.name, self.compute_capability_major, self.compute_capability_minor
        )?;
        if let Some(total) = self.total_memory_bytes {
            write!(f, ", total={} MiB", total / (1024 * 1024))?;
        }
        if let Some(free) = self.free_memory_bytes {
            write!(f, ", free={} MiB", free / (1024 * 1024))?;
        }
        Ok(())
    }
}
