//! GPU acceleration support.
//!
//! Infrastructure modules live at this level and are intentionally callable
//! from CPU-only builds: all public entry points are available without CUDA,
//! and the runtime reports an unavailable backend instead of changing
//! numerical results. CUDA-specific code is compiled only for Linux builds that
//! enable the `cuda` feature, so cudarc is never loaded by default CPU-only
//! builds.

pub mod backend_probe;
pub mod blas;
#[cfg(target_os = "linux")]
pub mod calibration;
pub mod cpu_traits;
pub mod device;
pub mod device_cache;
pub mod driver;
#[macro_use]
pub mod gpu_error;
pub mod device_runtime;
pub mod linalg_dispatch;
pub mod memory;
pub mod numerics_device;
pub mod numerics_host;
pub mod policy;
pub mod pool;
pub mod profile;
pub mod solver;

// Domain-specific GPU kernels are isolated from the infrastructure modules.
pub mod kernels;

pub use cpu_traits::{ExecutionTarget, MatrixLocation};
pub use device::GpuDeviceInfo;
pub use device_runtime::GpuRuntime;
pub use gpu_error::GpuError;
pub use memory::{DeviceBuffer, DeviceCsrMatrix, DeviceMatrix, DeviceVector};
pub use policy::{GpuDispatchPolicy, GpuMixedPrecisionPolicy};
pub use pool::{balanced_partition, scatter_batched};
pub use profile::{GpuExecutionTelemetry, KernelStat, KernelStatsSnapshot};

// ---------------------------------------------------------------------------
// User-facing policy and instrumentation hooks (formerly src/gpu.rs).
//
// The first production-safe step for acceleration is an explicit policy
// layer: `Auto` may opportunistically use supported device-resident kernels,
// `Off` guarantees the CPU path, and `Force` turns an unsupported GPU route
// into a hard error instead of a silent CPU fallback. The numerical kernels
// are wired to call these helpers before selecting a backend; until a vendor
// backend is compiled in this module intentionally reports "unsupported" so
// `force` fails loudly while `auto` remains a correct CPU fallback.
// ---------------------------------------------------------------------------

use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CudaBackendStatus {
    CudaUnavailable,
    CudaReady,
}

#[inline]
pub(crate) fn cuda_backend_status() -> CudaBackendStatus {
    if device_runtime::GpuRuntime::global().is_some() {
        CudaBackendStatus::CudaReady
    } else {
        CudaBackendStatus::CudaUnavailable
    }
}

/// User-facing GPU backend policy.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GpuPolicy {
    /// Let the solver use GPU kernels only for supported, large-enough paths.
    #[default]
    Auto,
    /// Always use CPU kernels.
    Off,
    /// Require GPU kernels and error if the requested path is unsupported.
    Force,
}

impl GpuPolicy {
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "off" => Some(Self::Off),
            "force" => Some(Self::Force),
            _ => None,
        }
    }

    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Off => "off",
            Self::Force => "force",
        }
    }

    /// Whether unsupported GPU dispatch should be surfaced as a hard error.
    #[inline]
    pub const fn is_force(self) -> bool {
        matches!(self, Self::Force)
    }
}

impl fmt::Display for GpuPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Fail-closed GPU residency mode (issue #1017).
///
/// Distinct from [`GpuPolicy`], which governs opportunistic per-kernel dispatch.
/// `GpuMode` is the process-wide *residency contract* the resident solver
/// consults through [`crate::gpu::device_runtime::GpuRuntime::global_or_fail`]:
///
/// * [`GpuMode::Auto`] — use the device when the probe admits it, fall back to
///   CPU otherwise (the current, working behavior; preserved bit-for-bit).
/// * [`GpuMode::Required`] — the device MUST be available; if the runtime is
///   absent the resident path returns a structured error instead of silently
///   running on the CPU. This is the fail-closed guard the reviewers asked for.
/// * [`GpuMode::Off`] — never use the device.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GpuMode {
    /// Use the device when available; fall back to CPU otherwise.
    #[default]
    Auto,
    /// Require the device; error (do not fall back) when it is unavailable.
    Required,
    /// Never use the device.
    Off,
}

impl GpuMode {
    /// Stable lowercase identifier.
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Required => "required",
            Self::Off => "off",
        }
    }
}

impl fmt::Display for GpuMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

static GPU_MODE: OnceLock<GpuMode> = OnceLock::new();

/// Configure the process-wide GPU residency mode. First-writer-wins so
/// concurrent fits cannot race the contract; a redundant late call is ignored.
pub fn set_gpu_mode(mode: GpuMode) {
    GPU_MODE.set(mode).ok();
}

/// Read the process-wide GPU residency mode. Defaults to [`GpuMode::Auto`]
/// without claiming the slot, mirroring [`global_policy`] so an incidental
/// read never locks the mode against a later explicit [`set_gpu_mode`].
#[inline]
pub fn gpu_mode() -> GpuMode {
    match GPU_MODE.get() {
        Some(m) => *m,
        None => GpuMode::Auto,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuKernel {
    DenseMatvec,
    DenseTransposeMatvec,
    DenseXtWX,
    CandidateScreen,
    DenseSolve,
    MatrixFreePcg,
    SparseAssembly,
    SpatialKernelOperator,
    MarginalSlopeRows,
    RemlTrace,
    FinalInference,
}

impl GpuKernel {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DenseMatvec => "dense-matvec",
            Self::DenseTransposeMatvec => "dense-transpose-matvec",
            Self::DenseXtWX => "dense-xtwx",
            Self::CandidateScreen => "candidate-screen",
            Self::DenseSolve => "dense-solve",
            Self::MatrixFreePcg => "matrix-free-pcg",
            Self::SparseAssembly => "sparse-assembly",
            Self::SpatialKernelOperator => "spatial-kernel-operator",
            Self::MarginalSlopeRows => "marginal-slope-rows",
            Self::RemlTrace => "reml-trace",
            Self::FinalInference => "final-inference",
        }
    }
}

/// A backend-selection decision for a single hot kernel.
#[derive(Clone, Debug)]
pub struct GpuDecision {
    pub policy: GpuPolicy,
    pub kernel: GpuKernel,
    pub use_gpu: bool,
    pub reason: &'static str,
}

static POLICY: OnceLock<GpuPolicy> = OnceLock::new();

#[inline]
pub fn global_policy() -> GpuPolicy {
    // Reading the policy must NOT claim the OnceLock slot: returning the
    // default `Auto` via `get_or_init` would race against an explicit
    // `configure_global_policy(...)` made later in the same process and
    // silently lock the policy to `Auto`.  Keep the slot uninitialized
    // until explicitly configured so first-writer-wins applies only to
    // genuine writes, not to incidental reads from probe/dispatch code.
    match POLICY.get() {
        Some(p) => *p,
        None => GpuPolicy::Auto,
    }
}

/// Configure the process-wide policy before solver kernels are selected.
/// If a previous explicit configuration already set the policy, the first
/// value wins so concurrent fits cannot race policy changes.  Reads of
/// `global_policy()` never claim the slot, so the very first explicit
/// configuration always sticks even if dispatch code observed the
/// default `Auto` beforehand.
pub fn configure_global_policy(policy: GpuPolicy) {
    // First-writer-wins semantics; ignore a redundant late call.
    POLICY.set(policy).ok();
}

/// True when direct solver GPU entry points should be attempted.
///
/// `Auto` attempts CUDA only after the runtime probe finds a usable device.
/// `Off` pins the process to CPU. `Force` attempts the GPU path so missing
/// runtime/backend support becomes an explicit error at the callee instead of
/// an implicit CPU route.
#[inline]
pub fn cuda_selected() -> bool {
    match global_policy() {
        GpuPolicy::Auto => device_runtime::GpuRuntime::is_available(),
        GpuPolicy::Off => false,
        GpuPolicy::Force => true,
    }
}

/// Joint eligibility state for a GPU kernel at the call site.
///
/// Callers construct exactly one variant, which encodes both the compile-time
/// backend presence and the runtime workload threshold check.  Replacing the
/// former `(supported: bool, large_enough: bool)` pair removes the possibility
/// of silently swapping the two flags at a call site: each meaningful state
/// has exactly one constructor and the `match` inside [`decide`] is total.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuEligibility {
    /// Vendor backend is not compiled into this build for this kernel.
    BackendNotCompiled,
    /// Backend is compiled in, but the workload (n, m, ...) is below the
    /// runtime threshold for this kernel.
    WorkloadBelowThreshold,
    /// Backend is compiled in and the workload is large enough; the only
    /// remaining gates are policy and runtime probe.
    Eligible,
}

impl GpuEligibility {
    /// Combine the compile-time backend flag with the workload predicate into
    /// the canonical joint state.  Use this only when you genuinely have two
    /// independent booleans; otherwise prefer constructing a variant directly.
    #[inline]
    pub const fn from_flags(supported: bool, large_enough: bool) -> Self {
        if !supported {
            Self::BackendNotCompiled
        } else if !large_enough {
            Self::WorkloadBelowThreshold
        } else {
            Self::Eligible
        }
    }
}

/// Decide whether a GPU kernel may run. This is deliberately conservative:
/// with no compiled vendor backend, `auto` returns CPU fallback and `force`
/// returns an error at the call site through [`GpuDecision::require_supported`].
pub fn decide(kernel: GpuKernel, eligibility: GpuEligibility) -> GpuDecision {
    let policy = global_policy();
    // Auto must consult the actual probed runtime, not only the
    // compile-time eligibility.  Without this, `decide()` would claim
    // GPU when the kernel is "compiled in" even though `GpuRuntime::global()`
    // observed no device — silently producing CPU work via failed dispatch
    // and hiding the cpu_reason from callers wanting to log fallback cause.
    let runtime_available = device_runtime::GpuRuntime::is_available();
    let (use_gpu, reason) = match (policy, eligibility) {
        (GpuPolicy::Off, _) => (false, "cpu-gpu-policy-off"),
        (GpuPolicy::Auto, GpuEligibility::BackendNotCompiled) => {
            (false, "cpu-gpu-backend-not-compiled")
        }
        (GpuPolicy::Auto, _) if !runtime_available => (false, "cpu-gpu-runtime-unavailable"),
        (GpuPolicy::Auto, GpuEligibility::WorkloadBelowThreshold) => {
            (false, "cpu-workload-below-gpu-threshold")
        }
        (GpuPolicy::Auto, GpuEligibility::Eligible) => (true, "gpu-auto-supported"),
        (GpuPolicy::Force, GpuEligibility::BackendNotCompiled) => {
            (false, "cpu-gpu-force-unsupported")
        }
        (GpuPolicy::Force, _) if !runtime_available => (false, "cpu-gpu-force-runtime-unavailable"),
        // Under `force`, the workload-threshold gate is intentionally bypassed:
        // the user explicitly asked for GPU regardless of size.
        (GpuPolicy::Force, GpuEligibility::WorkloadBelowThreshold)
        | (GpuPolicy::Force, GpuEligibility::Eligible) => (true, "gpu-force-supported"),
    };
    GpuDecision {
        policy,
        kernel,
        use_gpu,
        reason,
    }
}

impl GpuDecision {
    pub fn require_supported(&self) -> Result<(), String> {
        if self.policy == GpuPolicy::Force && !self.use_gpu {
            return Err(format!(
                "gpu=force requested kernel '{}' but no supported device backend is available ({})",
                self.kernel.as_str(),
                self.reason
            ));
        }
        Ok(())
    }

    pub fn log(self) {
        log::debug!(
            "[GPU backend] kernel={} policy={} selected={} reason={}",
            self.kernel.as_str(),
            self.policy.as_str(),
            self.use_gpu,
            self.reason
        );
    }
}

/// Emit the roadmap-visible kernels at startup/debug time without affecting
/// numerical execution. This keeps backend coverage auditable as real device
/// kernels are added incrementally.
pub fn log_backend_inventory_once() {
    static LOGGED: OnceLock<()> = OnceLock::new();
    LOGGED.get_or_init(|| {
        let compiled_backends = if cfg!(target_os = "linux") {
            "cuda-dynamic"
        } else {
            "none"
        };
        log::debug!(
            "[GPU backend] policy={} compiled_backends={} kernels=dense-matvec,dense-transpose-matvec,dense-xtwx,candidate-screen,dense-solve,matrix-free-pcg,sparse-assembly,spatial-kernel-operator,marginal-slope-rows,reml-trace,final-inference",
            global_policy().as_str(),
            compiled_backends
        );
    });
}

#[inline]
pub fn try_fast_ab(
    a: ndarray::ArrayView2<'_, f64>,
    b: ndarray::ArrayView2<'_, f64>,
) -> Option<ndarray::Array2<f64>> {
    linalg_dispatch::try_fast_ab(a, b)
}
#[inline]
pub fn try_fast_atb_on_ordinal(
    ordinal: usize,
    a: ndarray::ArrayView2<'_, f64>,
    b: ndarray::ArrayView2<'_, f64>,
) -> Option<ndarray::Array2<f64>> {
    linalg_dispatch::try_fast_atb_on_ordinal(ordinal, a, b)
}
#[inline]
pub fn try_fast_av(
    a: ndarray::ArrayView2<'_, f64>,
    v: ndarray::ArrayView1<'_, f64>,
) -> Option<ndarray::Array1<f64>> {
    linalg_dispatch::try_fast_av(a, v)
}
#[inline]
pub fn try_fast_atv(
    a: ndarray::ArrayView2<'_, f64>,
    v: ndarray::ArrayView1<'_, f64>,
) -> Option<ndarray::Array1<f64>> {
    linalg_dispatch::try_fast_atv(a, v)
}
#[inline]
pub fn try_fast_ab_broadcast_b_batched(
    a: ndarray::ArrayView3<'_, f64>,
    b: ndarray::ArrayView2<'_, f64>,
) -> Option<ndarray::Array3<f64>> {
    linalg_dispatch::try_fast_ab_broadcast_b_batched(a, b)
}
#[inline]
pub fn try_fast_abt_strided_batched(
    a: ndarray::ArrayView3<'_, f64>,
    b: ndarray::ArrayView3<'_, f64>,
) -> Option<ndarray::Array3<f64>> {
    linalg_dispatch::try_fast_abt_strided_batched(a, b)
}
#[inline]
pub fn try_cholesky_lower_inplace(a: &mut ndarray::Array2<f64>) -> Option<()> {
    linalg_dispatch::try_cholesky_lower_inplace(a)
}
#[inline]
pub fn try_cholesky_batched_lower_inplace(matrices: &mut [ndarray::Array2<f64>]) -> Option<()> {
    linalg_dispatch::try_cholesky_batched_lower_inplace(matrices)
}
#[inline]
pub fn try_solve_lower_triangular_matrix(
    lower: ndarray::ArrayView2<'_, f64>,
    rhs: ndarray::ArrayView2<'_, f64>,
) -> Option<ndarray::Array2<f64>> {
    linalg_dispatch::try_solve_lower_triangular_matrix(lower, rhs)
}
#[inline]
pub fn try_solve_upper_triangular_matrix(
    upper: ndarray::ArrayView2<'_, f64>,
    rhs: ndarray::ArrayView2<'_, f64>,
) -> Option<ndarray::Array2<f64>> {
    linalg_dispatch::try_solve_upper_triangular_matrix(upper, rhs)
}
#[cfg(test)]
mod policy_tests {
    use super::*;

    #[test]
    fn parses_canonical_user_gpu_policy_values() {
        assert_eq!(GpuPolicy::parse("auto"), Some(GpuPolicy::Auto));
        assert_eq!(GpuPolicy::parse("off"), Some(GpuPolicy::Off));
        assert_eq!(GpuPolicy::parse("force"), Some(GpuPolicy::Force));
        assert_eq!(GpuPolicy::parse("cpu"), None);
        assert_eq!(GpuPolicy::parse(""), None);
        assert_eq!(GpuPolicy::parse("wat"), None);
    }

    #[test]
    fn execution_path_defaults_to_cpu() {
        use crate::model_types::ExecutionPath;
        // The truthful execution-path classifier must default to the CPU path,
        // so a result struct that is never told otherwise cannot claim the
        // device (the original `used_device: bool` defaulted the same way, but
        // now the "no device" state is a named, non-lying variant).
        assert_eq!(ExecutionPath::default(), ExecutionPath::Cpu);
        assert!(!ExecutionPath::Cpu.used_device());
        assert!(ExecutionPath::GpuResidentFull.used_device());
    }

    #[test]
    fn gpu_mode_required_fails_closed_when_device_absent() {
        use crate::gpu::device_runtime::GpuRuntime;
        // Off always refuses, regardless of hardware.
        assert!(matches!(
            GpuRuntime::global_or_fail(GpuMode::Off),
            Err(GpuError::DriverLibraryUnavailable { .. })
        ));

        if GpuRuntime::is_available() {
            // On a GPU host both Auto and Required must succeed.
            assert!(GpuRuntime::global_or_fail(GpuMode::Required).is_ok());
            assert!(GpuRuntime::global_or_fail(GpuMode::Auto).is_ok());
        } else {
            // Fail-closed: Required surfaces a STRUCTURED error rather than a
            // silent CPU fallback. Auto also reports unavailable (callers there
            // swallow it and fall back), but the variant is what lets Required
            // propagate it as fatal.
            let required = GpuRuntime::global_or_fail(GpuMode::Required);
            assert!(
                matches!(required, Err(GpuError::DriverLibraryUnavailable { .. })),
                "GpuMode::Required must fail closed when the device is absent, got {required:?}"
            );
            assert!(GpuRuntime::global_or_fail(GpuMode::Auto).is_err());
        }
    }

    #[test]
    fn pirls_loop_admission_requires_runtime_size_and_known_family() {
        use crate::gpu::policy::{PirlsLoopAdmission, PirlsLoopCurvatureKind, PirlsLoopFamilyKind};
        let pol = GpuDispatchPolicy::default();
        let base = PirlsLoopAdmission {
            n: 80_000,
            p: 44,
            family: Some(PirlsLoopFamilyKind::BernoulliLogit),
            curvature: PirlsLoopCurvatureKind::Fisher,
            gpu_available: true,
        };
        assert!(pol.should_use_gpu_pirls_loop(base));
        // No runtime → never dispatch.
        assert!(!pol.should_use_gpu_pirls_loop(PirlsLoopAdmission {
            gpu_available: false,
            ..base
        }));
        // Below dense-work floor.
        assert!(!pol.should_use_gpu_pirls_loop(PirlsLoopAdmission { n: 1_000, ..base }));
        // Small n with large p is admitted because 2*n*p^2 clears the work floor.
        assert!(pol.should_use_gpu_pirls_loop(PirlsLoopAdmission {
            n: 2_000,
            p: 2_048,
            ..base
        }));
        // Below column floor.
        assert!(!pol.should_use_gpu_pirls_loop(PirlsLoopAdmission { p: 8, ..base }));
        // Custom family (not in 6 JIT-cached set) declines.
        assert!(!pol.should_use_gpu_pirls_loop(PirlsLoopAdmission {
            family: None,
            ..base
        }));
    }

    #[test]
    fn force_policy_reports_unsupported_kernel() {
        let decision = GpuDecision {
            policy: GpuPolicy::Force,
            kernel: GpuKernel::DenseXtWX,
            use_gpu: false,
            reason: "gpu-force-unsupported",
        };
        let err = decision.require_supported().unwrap_err();
        assert!(err.contains("dense-xtwx"));
        assert!(err.contains("gpu=force"));
    }
}
