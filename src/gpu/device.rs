use std::fmt;

/// Coarse capability class used by dispatch policy before calibration data is
/// available. The exact model is still reported through [`GpuDeviceInfo`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuCapability {
    Legacy,
    Mainstream,
    Datacenter,
    HighEndDatacenter,
}

/// Device selected for acceleration, or the reason acceleration is not active.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GpuSelection {
    CpuOnly { reason: String },
    Cuda { device: GpuDeviceInfo },
}

/// Runtime CUDA device facts used for logging and dispatch decisions.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuDeviceInfo {
    pub ordinal: usize,
    pub name: String,
    pub compute_capability_major: i32,
    pub compute_capability_minor: i32,
    pub total_memory_bytes: usize,
    pub free_memory_bytes: Option<usize>,
    pub capability: GpuCapability,
}

impl GpuDeviceInfo {
    #[inline]
    pub fn compute_capability(&self) -> (i32, i32) {
        (self.compute_capability_major, self.compute_capability_minor)
    }

    #[inline]
    pub fn total_memory_gib(&self) -> f64 {
        self.total_memory_bytes as f64 / 1024.0 / 1024.0 / 1024.0
    }

    #[inline]
    pub fn score(&self) -> u128 {
        let cc_score = (self.compute_capability_major.max(0) as u128) * 10_000
            + (self.compute_capability_minor.max(0) as u128) * 1_000;
        let mem_score = self.total_memory_bytes as u128 / (1024 * 1024);
        cc_score + mem_score
    }
}

impl fmt::Display for GpuDeviceInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CUDA device {}: {} (cc {}.{}, {:.1} GiB total",
            self.ordinal,
            self.name,
            self.compute_capability_major,
            self.compute_capability_minor,
            self.total_memory_gib()
        )?;
        if let Some(free) = self.free_memory_bytes {
            write!(
                f,
                ", {:.1} GiB free",
                free as f64 / 1024.0 / 1024.0 / 1024.0
            )?;
        }
        write!(f, ", {:?})", self.capability)
    }
}

#[inline]
pub fn classify_capability(major: i32, minor: i32, total_memory_bytes: usize) -> GpuCapability {
    let cc10 = major.saturating_mul(10).saturating_add(minor);
    let mem_gib = total_memory_bytes / 1024 / 1024 / 1024;
    if cc10 >= 90 || mem_gib >= 80 {
        GpuCapability::HighEndDatacenter
    } else if cc10 >= 70 && mem_gib >= 16 {
        GpuCapability::Datacenter
    } else if cc10 >= 60 {
        GpuCapability::Mainstream
    } else {
        GpuCapability::Legacy
    }
}
