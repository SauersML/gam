use std::fmt;

/// Coarse hardware class used by dispatch policy before per-device calibration
/// data is available.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuCapability {
    Unknown,
    Cuda,
    CudaFp64Strong,
    CudaHopperOrNewer,
}

/// Runtime-discovered GPU device metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuDeviceInfo {
    pub ordinal: usize,
    pub name: String,
    pub uuid: Option<String>,
    pub compute_capability_major: Option<i32>,
    pub compute_capability_minor: Option<i32>,
    pub total_memory_bytes: Option<u64>,
    pub free_memory_bytes: Option<u64>,
    pub capability: GpuCapability,
}

impl GpuDeviceInfo {
    #[must_use]
    pub fn compute_capability(&self) -> Option<(i32, i32)> {
        Some((
            self.compute_capability_major?,
            self.compute_capability_minor?,
        ))
    }

    #[must_use]
    pub fn score(&self) -> u64 {
        let cc_score = self.compute_capability().map_or(0_u64, |(maj, min)| {
            (maj.max(0) as u64) * 100 + min.max(0) as u64
        });
        let mem_score = self.total_memory_bytes.unwrap_or(0) / (1024 * 1024 * 1024);
        let class_bonus = match self.capability {
            GpuCapability::Unknown => 0,
            GpuCapability::Cuda => 1_000,
            GpuCapability::CudaFp64Strong => 2_000,
            GpuCapability::CudaHopperOrNewer => 4_000,
        };
        class_bonus + cc_score + mem_score
    }

    #[must_use]
    pub fn from_nvidia_smi_csv(ordinal: usize, line: &str) -> Option<Self> {
        let mut parts = line.split(',').map(str::trim);
        let name = parts.next()?.to_string();
        let uuid = parts
            .next()
            .and_then(|s| (!s.is_empty()).then(|| s.to_string()));
        let cc_raw = parts.next().unwrap_or_default();
        let (major, minor) = parse_compute_capability(cc_raw);
        let total_memory_bytes = parts.next().and_then(parse_mib).map(mib_to_bytes);
        let free_memory_bytes = parts.next().and_then(parse_mib).map(mib_to_bytes);
        let capability = classify_capability(major, minor, &name);
        Some(Self {
            ordinal,
            name,
            uuid,
            compute_capability_major: major,
            compute_capability_minor: minor,
            total_memory_bytes,
            free_memory_bytes,
            capability,
        })
    }
}

impl fmt::Display for GpuDeviceInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let cc = self.compute_capability().map_or_else(
            || "unknown".to_string(),
            |(maj, min)| format!("{maj}.{min}"),
        );
        let total_gib = self.total_memory_bytes.map_or_else(
            || "unknown".to_string(),
            |b| format!("{:.1} GiB", b as f64 / 1024.0_f64.powi(3)),
        );
        write!(
            f,
            "#{ordinal} {name} (cc={cc}, memory={total_gib}, class={class:?})",
            ordinal = self.ordinal,
            name = self.name,
            class = self.capability
        )
    }
}

#[must_use]
pub fn classify_capability(major: Option<i32>, minor: Option<i32>, name: &str) -> GpuCapability {
    if matches!(major, Some(m) if m >= 9) {
        return GpuCapability::CudaHopperOrNewer;
    }
    let upper = name.to_ascii_uppercase();
    if upper.contains("H100") || upper.contains("H200") || upper.contains("B200") {
        return GpuCapability::CudaHopperOrNewer;
    }
    if upper.contains("A100") || upper.contains("V100") || upper.contains("L40") {
        return GpuCapability::CudaFp64Strong;
    }
    if major.is_some()
        || upper.contains("NVIDIA")
        || upper.contains("TESLA")
        || upper.contains("QUADRO")
    {
        return GpuCapability::Cuda;
    }
    let _ = minor;
    GpuCapability::Unknown
}

fn parse_compute_capability(raw: &str) -> (Option<i32>, Option<i32>) {
    let cleaned = raw.trim();
    let Some((major, minor)) = cleaned.split_once('.') else {
        return (cleaned.parse::<i32>().ok(), None);
    };
    (major.parse::<i32>().ok(), minor.parse::<i32>().ok())
}

fn parse_mib(raw: &str) -> Option<u64> {
    raw.split_whitespace().next()?.parse::<u64>().ok()
}

const fn mib_to_bytes(mib: u64) -> u64 {
    mib * 1024 * 1024
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nvidia_smi_csv() {
        let info =
            GpuDeviceInfo::from_nvidia_smi_csv(1, "NVIDIA H200, GPU-deadbeef, 9.0, 143771, 140000")
                .expect("valid csv");
        assert_eq!(info.ordinal, 1);
        assert_eq!(info.compute_capability(), Some((9, 0)));
        assert_eq!(info.capability, GpuCapability::CudaHopperOrNewer);
        assert_eq!(info.total_memory_bytes, Some(143_771 * 1024 * 1024));
    }
}
