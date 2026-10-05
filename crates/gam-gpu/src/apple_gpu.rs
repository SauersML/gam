//! The Apple-GPU (Metal) runtime: probe, device selection and policy.
//!
//! Metal has no float64, and every existing [`GpuKernel`] computes a float64
//! quantity, so the Metal backend is a second runtime beside the CUDA one
//! rather than a second implementation of it. [`GpuRuntime`] stays the CUDA
//! runtime and still reports absence on macOS; the float64 kernels stay on
//! the CPU there, and `gpu=required` for one of them is refused naming the
//! missing float64 arithmetic. The kernels in [`MetalKernel`] are the ones
//! whose result is either a screen (an ordering or a discard decision that
//! is sound against a derived band) or carries a derived band
//! ([`crate::precision_bounds`]); only they ever select Metal.
//!
//! The probe mirrors the CUDA probe: enumerate every device, describe each,
//! rank by score, select the best, and cache the lossless outcome for the
//! process. Typed absence (not macOS, no Apple-family GPU, a device that
//! fails the arithmetic self-test) is distinct from a probe fault (a kernel
//! library that does not compile, a queue that cannot be created), and the
//! [`GpuPolicy`] semantics are those of the CUDA runtime: `off` never probes,
//! `auto` maps typed absence to the CPU and keeps faults as errors,
//! `required` turns absence into [`GpuError::RequiredDeviceUnavailable`].
//!
//! [`GpuRuntime`]: crate::GpuRuntime

use crate::gpu_error::GpuError;
use crate::{GpuDecision, GpuEligibility, GpuKernel, GpuPolicy, decide_under, global_policy};
use std::sync::OnceLock;

/// A Metal device as the probe described it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalDeviceInfo {
    /// Position in `MTLCopyAllDevices()` order.
    pub ordinal: usize,
    pub name: String,
    pub registry_id: u64,
    /// The highest `MTLGPUFamilyAppleN` the device supports, `None` for a
    /// non-Apple GPU (an Intel or AMD GPU in an Intel Mac).
    pub apple_family: Option<u32>,
    pub low_power: bool,
    pub removable: bool,
    pub unified_memory: bool,
    pub recommended_max_working_set_bytes: u64,
    pub max_buffer_bytes: usize,
    pub max_threadgroup_memory_bytes: usize,
}

impl MetalDeviceInfo {
    /// The oldest Apple GPU family with the simdgroup matrix unit
    /// (`MTLGPUFamilyApple7`, the M1 generation) the f32 GEMM runs on.
    pub const MIN_APPLE_FAMILY: u32 = 7;

    /// Whether the device can run every Metal kernel gam compiles.
    #[must_use]
    pub fn supports_gam_kernels(&self) -> bool {
        self.apple_family
            .is_some_and(|family| family >= Self::MIN_APPLE_FAMILY)
            && self.max_threadgroup_memory_bytes >= 32 * 1024
    }

    /// Ranking among usable devices, the analogue of
    /// [`crate::GpuDeviceInfo::score`]: a newer Apple family, then a
    /// non-low-power, non-removable device, then the larger working set.
    #[must_use]
    pub fn score(&self) -> f64 {
        let family = f64::from(self.apple_family.unwrap_or(0));
        let power = if self.low_power { 0.0 } else { 50.0 };
        let fixed = if self.removable { 0.0 } else { 10.0 };
        family * 100.0
            + power
            + fixed
            + (self.recommended_max_working_set_bytes as f64 / 1_073_741_824.0)
    }

    /// The byte budget one dispatch may size its buffers against: half the
    /// recommended working set, the Metal analogue of the CUDA budget
    /// (`min(free, total/2)`). On unified memory the working set is shared
    /// with the host, so half leaves the host its own room.
    #[must_use]
    pub fn memory_budget_bytes(&self) -> usize {
        usize::try_from(self.recommended_max_working_set_bytes / 2)
            .unwrap_or(usize::MAX)
            .min(self.max_buffer_bytes)
    }
}

/// A genuine reason no usable Metal device exists on this host.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MetalAbsence {
    /// Metal exists only on Apple platforms; this build is not for macOS.
    UnsupportedPlatform,
    /// Metal reports no device, or none of the Apple family gam's kernels
    /// need.
    NoDevice { reason: String },
    /// A device exists but violated the binary32 arithmetic model every
    /// Metal band assumes in the probe-time self-test.
    ArithmeticModelViolated { reason: String },
}

impl std::fmt::Display for MetalAbsence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedPlatform => f.write_str("Metal is only available on macOS"),
            Self::NoDevice { reason } | Self::ArithmeticModelViolated { reason } => {
                f.write_str(reason)
            }
        }
    }
}

/// The probed Metal runtime: the selected device and every device seen.
#[derive(Debug)]
pub struct MetalRuntime {
    pub device: MetalDeviceInfo,
    pub devices: Vec<MetalDeviceInfo>,
    #[cfg(target_os = "macos")]
    pub(crate) context: crate::metal::MetalContext,
}

/// Lossless outcome of the Metal probe.
#[derive(Debug)]
pub enum MetalAvailability {
    Available(MetalRuntime),
    Absent(MetalAbsence),
}

/// Borrowed view of the cached probe outcome.
#[derive(Clone, Copy, Debug)]
pub enum MetalAvailabilityRef<'a> {
    Available(&'a MetalRuntime),
    Absent(&'a MetalAbsence),
}

impl MetalRuntime {
    /// Probe Metal once. Never cached; [`Self::availability`] caches.
    pub fn probe() -> Result<MetalAvailability, GpuError> {
        #[cfg(target_os = "macos")]
        {
            crate::metal::probe()
        }
        #[cfg(not(target_os = "macos"))]
        {
            Ok(MetalAvailability::Absent(MetalAbsence::UnsupportedPlatform))
        }
    }

    /// The cached probe outcome, faults kept as faults.
    pub fn availability() -> Result<MetalAvailabilityRef<'static>, GpuError> {
        static RUNTIME: OnceLock<Result<MetalAvailability, GpuError>> = OnceLock::new();
        let cached = RUNTIME.get_or_init(|| {
            let outcome = Self::probe();
            match &outcome {
                Ok(MetalAvailability::Available(runtime)) => log::debug!(
                    "[Metal] selected {} (Apple family {:?}, working set {} MiB) of {} device(s)",
                    runtime.device.name,
                    runtime.device.apple_family,
                    runtime.device.recommended_max_working_set_bytes >> 20,
                    runtime.devices.len()
                ),
                Ok(MetalAvailability::Absent(absence)) => {
                    log::debug!("[Metal] acceleration unavailable: {absence}");
                }
                Err(error) => log::debug!("[Metal] probe fault: {error}"),
            }
            outcome
        });
        match cached {
            Ok(MetalAvailability::Available(runtime)) => Ok(MetalAvailabilityRef::Available(runtime)),
            Ok(MetalAvailability::Absent(absence)) => Ok(MetalAvailabilityRef::Absent(absence)),
            Err(error) => Err(error.clone()),
        }
    }

    /// Resolve Metal under `policy`: `off` is `None` without probing, `auto`
    /// maps typed absence to `None`, `required` refuses it. Faults stay errors.
    pub fn resolve(policy: GpuPolicy) -> Result<Option<&'static Self>, GpuError> {
        if policy == GpuPolicy::Off {
            return Ok(None);
        }
        Self::resolve_availability(policy, Self::availability())
    }

    pub(crate) fn resolve_availability<'a>(
        policy: GpuPolicy,
        availability: Result<MetalAvailabilityRef<'a>, GpuError>,
    ) -> Result<Option<&'a Self>, GpuError> {
        match availability? {
            MetalAvailabilityRef::Available(runtime) => Ok(Some(runtime)),
            MetalAvailabilityRef::Absent(_) if policy != GpuPolicy::Required => Ok(None),
            MetalAvailabilityRef::Absent(absence) => Err(GpuError::RequiredDeviceUnavailable {
                reason: format!("gpu=required needs a Metal device: {absence}"),
            }),
        }
    }

    #[must_use]
    pub fn selected_device(&self) -> &MetalDeviceInfo {
        &self.device
    }
}

/// A kernel whose device result is a sound screen or carries a derived band,
/// and so may run in Metal's f32 / df64 arithmetic.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MetalKernel {
    /// Batched f32 matrix products with a derived per-entry band.
    GemmF32,
    /// Batched df64 matrix products with a derived per-entry band.
    GemmDf64,
}

impl MetalKernel {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::GemmF32 => "metal-gemm-f32",
            Self::GemmDf64 => "metal-gemm-df64",
        }
    }

    /// The roadmap kernel this Metal kernel implements, for the shared
    /// decision record.
    #[must_use]
    pub const fn roadmap_kernel(self) -> GpuKernel {
        match self {
            Self::GemmF32 | Self::GemmDf64 => GpuKernel::BandedGemm,
        }
    }
}

/// Whether the Metal backend is compiled into this build.
#[must_use]
pub const fn metal_compiled() -> bool {
    cfg!(target_os = "macos")
}

/// The Metal counterpart of [`crate::decide`]: the same stage order
/// (capability and build, then size, then the device) and the same policy
/// semantics, consulting the Metal runtime. The CPU is never probed for, and
/// the device is probed only when the answer still depends on it.
pub fn decide_metal(
    kernel: MetalKernel,
    eligibility: GpuEligibility,
) -> Result<MetalDecision, GpuError> {
    decide_metal_under_policy(global_policy(), kernel, eligibility)
}

/// [`decide_metal`] under an explicit policy.
pub fn decide_metal_under_policy(
    policy: GpuPolicy,
    kernel: MetalKernel,
    eligibility: GpuEligibility,
) -> Result<MetalDecision, GpuError> {
    let eligibility = if metal_compiled() {
        eligibility
    } else {
        match eligibility {
            GpuEligibility::CapabilityMissing { missing } => {
                GpuEligibility::CapabilityMissing { missing }
            }
            GpuEligibility::BackendNotCompiled
            | GpuEligibility::WorkloadBelowThreshold
            | GpuEligibility::DeviceMeasuredSlower
            | GpuEligibility::Unmeasured
            | GpuEligibility::Eligible => GpuEligibility::BackendNotCompiled,
        }
    };
    let needs_device = match (policy, eligibility) {
        (_, GpuEligibility::CapabilityMissing { .. } | GpuEligibility::BackendNotCompiled)
        | (GpuPolicy::Off, _)
        | (
            GpuPolicy::Auto,
            GpuEligibility::WorkloadBelowThreshold | GpuEligibility::DeviceMeasuredSlower,
        ) => false,
        (GpuPolicy::Auto | GpuPolicy::Required, _) => true,
    };
    let runtime = if needs_device {
        MetalRuntime::resolve(policy)?
    } else {
        None
    };
    let decision = decide_under(
        policy,
        runtime.is_some(),
        kernel.roadmap_kernel(),
        eligibility,
    );
    Ok(MetalDecision {
        kernel,
        decision,
        runtime,
    })
}

/// A Metal backend-selection decision.
#[derive(Clone, Debug)]
pub struct MetalDecision {
    pub kernel: MetalKernel,
    pub decision: GpuDecision,
    /// The resolved runtime when the decision selected the device.
    pub runtime: Option<&'static MetalRuntime>,
}

impl MetalDecision {
    /// The runtime to dispatch on, `None` for the CPU route.
    #[must_use]
    pub fn device(&self) -> Option<&'static MetalRuntime> {
        if self.decision.use_gpu {
            self.runtime
        } else {
            None
        }
    }

    /// Under `required`, the refusal of a decision that did not select Metal,
    /// naming the Metal kernel and why.
    pub fn require_supported(&self) -> Result<(), GpuError> {
        if self.decision.policy == GpuPolicy::Required && !self.decision.use_gpu {
            let why = match self.decision.missing_capability {
                Some(missing) => format!("the request needs {missing}, which the Metal kernel does not provide"),
                None => self.decision.reason.to_string(),
            };
            return Err(GpuError::RequiredDeviceUnavailable {
                reason: format!(
                    "gpu=required requested Metal kernel '{}' and it cannot run: {why}",
                    self.kernel.as_str()
                ),
            });
        }
        Ok(())
    }
}

/// The refusal `gpu=required` gets on macOS for a float64 kernel: the Apple
/// GPU cannot run float64 arithmetic, so no float64 kernel ever reaches it.
#[must_use]
pub fn float64_kernel_refusal(kernel: GpuKernel) -> String {
    format!(
        "gpu=required requested float64 kernel '{}', and the Apple GPU (Metal) has no \
         float64 arithmetic; this kernel runs only on CUDA or the CPU. Use gpu=\"auto\" or \
         gpu=\"off\"",
        kernel.as_str()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(family: Option<u32>, low_power: bool, working_set_gib: u64) -> MetalDeviceInfo {
        MetalDeviceInfo {
            ordinal: 0,
            name: "fixture".to_string(),
            registry_id: 1,
            apple_family: family,
            low_power,
            removable: false,
            unified_memory: true,
            recommended_max_working_set_bytes: working_set_gib << 30,
            max_buffer_bytes: 1 << 34,
            max_threadgroup_memory_bytes: 32 * 1024,
        }
    }

    #[test]
    fn scoring_prefers_newer_family_then_power_then_memory() {
        assert!(info(Some(9), false, 16).score() > info(Some(8), false, 64).score());
        assert!(info(Some(9), false, 16).score() > info(Some(9), true, 16).score());
        assert!(info(Some(9), false, 32).score() > info(Some(9), false, 16).score());
        assert!(!info(None, false, 16).supports_gam_kernels());
        assert!(!info(Some(6), false, 16).supports_gam_kernels());
        assert!(info(Some(7), false, 16).supports_gam_kernels());
        assert_eq!(info(Some(9), false, 16).memory_budget_bytes(), 8 << 30);
    }

    /// Typed absence is the CPU under `auto` and a named refusal under
    /// `required`; a probe fault is an error under both.
    #[test]
    fn metal_resolution_follows_the_cuda_policy_contract() {
        let absence = MetalAbsence::NoDevice {
            reason: "synthetic: no Apple GPU".to_string(),
        };
        let auto = MetalRuntime::resolve_availability(
            GpuPolicy::Auto,
            Ok(MetalAvailabilityRef::Absent(&absence)),
        )
        .expect("absence is not a fault under auto");
        assert!(auto.is_none());
        let required = MetalRuntime::resolve_availability(
            GpuPolicy::Required,
            Ok(MetalAvailabilityRef::Absent(&absence)),
        )
        .expect_err("required refuses absence");
        assert!(
            matches!(&required, GpuError::RequiredDeviceUnavailable { reason }
                if reason.contains("synthetic: no Apple GPU") && reason.contains("Metal")),
            "{required}"
        );
        for policy in [GpuPolicy::Auto, GpuPolicy::Required] {
            let fault = MetalRuntime::resolve_availability(
                policy,
                Err(GpuError::DriverCallFailed {
                    reason: "synthetic library compile failure".to_string(),
                }),
            )
            .expect_err("faults are never absence");
            assert!(matches!(fault, GpuError::DriverCallFailed { .. }));
        }
        assert!(MetalRuntime::resolve(GpuPolicy::Off).expect("off never probes").is_none());
    }

    /// A model the Metal kernel does not compute is never selected and never
    /// probes; `required` names the missing capability.
    #[test]
    fn capability_refusal_names_what_metal_lacks() {
        let missing = "float64 arithmetic";
        for policy in [GpuPolicy::Auto, GpuPolicy::Required, GpuPolicy::Off] {
            let decision = decide_metal_under_policy(
                policy,
                MetalKernel::GemmF32,
                GpuEligibility::CapabilityMissing { missing },
            )
            .expect("a capability refusal never probes");
            assert!(decision.device().is_none());
            assert!(decision.runtime.is_none());
            let refusal = decision.require_supported();
            if policy == GpuPolicy::Required {
                let text = refusal.expect_err("required is refused").to_string();
                assert!(text.contains(missing) && text.contains("metal-gemm-f32"), "{text}");
            } else {
                assert!(refusal.is_ok());
            }
        }
        let text = float64_kernel_refusal(GpuKernel::DenseXtWX);
        assert!(text.contains("dense-xtwx") && text.contains("float64"), "{text}");
    }

    /// Off never selects the device, whatever the workload.
    #[test]
    fn off_selects_the_cpu_without_probing() {
        let decision =
            decide_metal_under_policy(GpuPolicy::Off, MetalKernel::GemmF32, GpuEligibility::Eligible)
                .expect("off never probes");
        assert!(decision.device().is_none());
        assert_eq!(decision.decision.reason, "cpu-gpu-policy-off");
    }

    /// Below the size threshold `auto` stays on the CPU without probing, and
    /// `required` bypasses the threshold. Off macOS the backend is not compiled, which decides
    /// first (`non_macos_builds_refuse_metal_explicitly`).
    #[cfg(target_os = "macos")]
    #[test]
    fn below_threshold_is_cpu_under_auto() {
        let decision = decide_metal_under_policy(
            GpuPolicy::Auto,
            MetalKernel::GemmF32,
            GpuEligibility::WorkloadBelowThreshold,
        )
        .expect("auto below threshold never probes");
        assert!(decision.device().is_none());
        assert_eq!(decision.decision.reason, "cpu-workload-below-gpu-threshold");
    }

    /// Off this platform, every Metal kernel is `BackendNotCompiled`: `auto`
    /// selects the CPU and `required` is refused, without a probe.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn non_macos_builds_refuse_metal_explicitly() {
        let auto =
            decide_metal_under_policy(GpuPolicy::Auto, MetalKernel::GemmF32, GpuEligibility::Eligible)
                .expect("not compiled never probes");
        assert!(auto.device().is_none());
        assert_eq!(auto.decision.reason, "cpu-gpu-backend-not-compiled");
        let required = decide_metal_under_policy(
            GpuPolicy::Required,
            MetalKernel::GemmF32,
            GpuEligibility::Eligible,
        )
        .expect("not compiled never probes");
        assert!(required.require_supported().is_err());
        assert!(matches!(
            MetalRuntime::availability(),
            Ok(MetalAvailabilityRef::Absent(MetalAbsence::UnsupportedPlatform))
        ));
    }
}
