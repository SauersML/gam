//! Execution-path telemetry for hot solver/GPU paths.

use serde::{Deserialize, Serialize};

/// Truthful execution-path classifier for a fit's hot inner solve (issue #1017).
///
/// This replaces the lying `used_device: bool` on GPU-owned result structs: a
/// bare boolean could not distinguish "ran on the device with the constant
/// Hessian factors kept resident across iterations" from "re-uploaded and
/// re-factored every iterate" from "silently fell back to the CPU", so a
/// device that quietly declined still reported `used_device = true` at some
/// call sites. Each variant names exactly one of the four real backends the
/// resident solver can take, so telemetry and tests assert the concrete path
/// instead of a yes/no that hid the original silent-fallback bug.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum ExecutionPath {
    /// Host-only arithmetic; no device work was performed.
    #[default]
    Cpu,
    /// GPU path that re-uploads `D`/`B`/`g` and re-factors every iterate (the
    /// pre-residency baseline).
    GpuReupload,
    /// GPU path that keeps the constant Hessian factors resident across
    /// iterations and uploads only the per-iterate gradient, for a single
    /// linearization (one frozen gate/basis frame).
    GpuResidentLinearization,
    /// Full device-resident inner Newton loop (the Phase-3 residency fix).
    GpuResidentFull,
}

impl ExecutionPath {
    /// Stable lowercase identifier for logs/telemetry dictionaries.
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::GpuReupload => "gpu-reupload",
            Self::GpuResidentLinearization => "gpu-resident-linearization",
            Self::GpuResidentFull => "gpu-resident-full",
        }
    }

    /// True when any part of the path executed on the device.
    #[inline]
    pub const fn used_device(self) -> bool {
        !matches!(self, Self::Cpu)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn as_str_cpu() {
        assert_eq!(ExecutionPath::Cpu.as_str(), "cpu");
    }

    #[test]
    fn as_str_gpu_reupload() {
        assert_eq!(ExecutionPath::GpuReupload.as_str(), "gpu-reupload");
    }

    #[test]
    fn as_str_gpu_resident_linearization() {
        assert_eq!(
            ExecutionPath::GpuResidentLinearization.as_str(),
            "gpu-resident-linearization"
        );
    }

    #[test]
    fn as_str_gpu_resident_full() {
        assert_eq!(ExecutionPath::GpuResidentFull.as_str(), "gpu-resident-full");
    }

    #[test]
    fn used_device_false_for_cpu() {
        assert!(!ExecutionPath::Cpu.used_device());
    }

    #[test]
    fn used_device_true_for_all_gpu_variants() {
        assert!(ExecutionPath::GpuReupload.used_device());
        assert!(ExecutionPath::GpuResidentLinearization.used_device());
        assert!(ExecutionPath::GpuResidentFull.used_device());
    }

    #[test]
    fn default_is_cpu() {
        assert_eq!(ExecutionPath::default(), ExecutionPath::Cpu);
        assert!(!ExecutionPath::default().used_device());
    }
}
