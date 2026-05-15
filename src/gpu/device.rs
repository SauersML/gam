use std::fmt;

/// Coarse CUDA capability bucket used by policy thresholds.  Runtime probing
/// still records the raw compute capability when it is available.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuCapability {
    Unknown,
    TuringOrOlder,
    Ampere,
    HopperOrNewer,
}

/// Device properties needed by the operation-level dispatch policy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuDeviceInfo {
    pub index: usize,
    pub name: String,
    pub uuid: Option<String>,
    pub compute_capability: Option<(u32, u32)>,
    pub total_memory_bytes: Option<u64>,
    pub free_memory_bytes: Option<u64>,
}

impl GpuDeviceInfo {
    #[must_use]
    pub fn capability(&self) -> GpuCapability {
        match self.compute_capability {
            Some((major, _)) if major >= 9 => GpuCapability::HopperOrNewer,
            Some((8, _)) => GpuCapability::Ampere,
            Some((major, _)) if major <= 7 => GpuCapability::TuringOrOlder,
            _ => GpuCapability::Unknown,
        }
    }

    #[must_use]
    pub fn memory_score_bytes(&self) -> u64 {
        self.free_memory_bytes
            .or(self.total_memory_bytes)
            .unwrap_or(0)
    }
}

impl fmt::Display for GpuDeviceInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{} {}", self.index, self.name)?;
        if let Some((major, minor)) = self.compute_capability {
            write!(f, " cc={major}.{minor}")?;
        }
        if let Some(total) = self.total_memory_bytes {
            write!(f, " total={}MiB", total / (1024 * 1024))?;
        }
        if let Some(free) = self.free_memory_bytes {
            write!(f, " free={}MiB", free / (1024 * 1024))?;
        }
        Ok(())
    }
}
