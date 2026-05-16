//! Host-visible description of a CUDA device.

use std::fmt;

/// Coarse capability bucket derived from the compute capability major number.
///
/// The bucket only drives early heuristics in [`super::policy::DispatchPolicy`];
/// the exact `(major, minor)` is preserved on [`GpuDeviceInfo`] for logging.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuCapability {
    /// Pre-Volta: no tensor cores. Use cautiously; mostly host fallback wins.
    Legacy,
    /// Volta / Turing (sm_70 / sm_75): FP32 tensor cores, async copies absent.
    Mainstream,
    /// Ampere (sm_80 / sm_86): FP64 tensor cores, async copies, large SHM.
    Datacenter,
    /// Hopper / Blackwell (sm_90+): TMA, cluster launch, FP8 tensor cores.
    HighEndDatacenter,
}

impl GpuCapability {
    /// Map a compute-capability tuple to the broad capability bucket.
    pub fn from_compute_capability(major: i32, _minor: i32) -> Self {
        match major {
            x if x >= 9 => Self::HighEndDatacenter,
            x if x >= 8 => Self::Datacenter,
            x if x >= 7 => Self::Mainstream,
            _ => Self::Legacy,
        }
    }

    /// True when the device exposes FP64 tensor cores (Ampere onward).
    #[inline]
    pub fn has_fp64_tensor_cores(self) -> bool {
        matches!(self, Self::Datacenter | Self::HighEndDatacenter)
    }

    /// True when the device exposes hardware async copies (Ampere onward).
    #[inline]
    pub fn has_async_copy(self) -> bool {
        matches!(self, Self::Datacenter | Self::HighEndDatacenter)
    }
}

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
    pub total_memory_bytes: usize,
    pub capability: GpuCapability,
}

impl GpuDeviceInfo {
    /// Heuristic score used to pick the "best" device when multiple are
    /// present. Larger is better; the score is purely ordinal.
    pub fn score(&self) -> f64 {
        let mut score = self.total_memory_bytes as f64 / 1_073_741_824.0;
        if self.capability.has_fp64_tensor_cores() {
            score += 100.0;
        }
        if self.capability.has_async_copy() {
            score += 50.0;
        }
        score += match self.capability {
            GpuCapability::Legacy => 0.0,
            GpuCapability::Mainstream => 10.0,
            GpuCapability::Datacenter => 30.0,
            GpuCapability::HighEndDatacenter => 60.0,
        };
        score
    }
}

impl fmt::Display for GpuDeviceInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "device {} '{}' sm_{}{} {:.1} GiB",
            self.ordinal,
            self.name,
            self.compute_capability_major,
            self.compute_capability_minor,
            self.total_memory_bytes as f64 / 1_073_741_824.0,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_buckets_match_compute_capability() {
        assert_eq!(
            GpuCapability::from_compute_capability(6, 0),
            GpuCapability::Legacy
        );
        assert_eq!(
            GpuCapability::from_compute_capability(7, 5),
            GpuCapability::Mainstream
        );
        assert_eq!(
            GpuCapability::from_compute_capability(8, 0),
            GpuCapability::Datacenter
        );
        assert_eq!(
            GpuCapability::from_compute_capability(9, 0),
            GpuCapability::HighEndDatacenter
        );
    }

    #[test]
    fn device_score_prefers_newer_arch() {
        let hopper = GpuDeviceInfo {
            ordinal: 0,
            name: "H100".to_string(),
            compute_capability_major: 9,
            compute_capability_minor: 0,
            total_memory_bytes: 80 * 1024 * 1024 * 1024,
            capability: GpuCapability::HighEndDatacenter,
        };
        let turing = GpuDeviceInfo {
            ordinal: 1,
            name: "T4".to_string(),
            compute_capability_major: 7,
            compute_capability_minor: 5,
            total_memory_bytes: 16 * 1024 * 1024 * 1024,
            capability: GpuCapability::Mainstream,
        };
        assert!(hopper.score() > turing.score());
    }
}
