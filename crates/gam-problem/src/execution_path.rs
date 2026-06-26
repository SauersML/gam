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
